
import re
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from flask import Flask, request, render_template

#tokenizer
def simple_tokenize(text: str):
    text = re.sub(r"([.,!?;:()\"'])", r" \1 ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().split(" ") if text.strip() else []

def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k is None or k <= 0 or k >= logits.numel():
        return logits
    values, idx = torch.topk(logits, k)
    filtered = torch.full_like(logits, float("-inf"))
    filtered[idx] = logits[idx]
    return filtered

#model
class LanguageModelLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

#load artifacts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = Path(__file__).resolve().parent.parent / "artifacts" / "lm_lstm_best.pt"
if not ckpt_path.exists():
    raise FileNotFoundError(
        f"Checkpoint not found at {ckpt_path}. Run from the project root or check artifacts path."
    )
ckpt = torch.load(str(ckpt_path), map_location=device)

stoi = ckpt["stoi"]
itos = ckpt["itos"]
pad_idx = ckpt["pad_idx"]
unk_idx = ckpt["unk_idx"]
vocab_size = ckpt["vocab_size"]

embed_dim  = ckpt["embed_dim"]
hidden_dim = ckpt["hidden_dim"]
num_layers = ckpt["num_layers"]
dropout    = ckpt["dropout"]

model = LanguageModelLSTM(vocab_size, embed_dim, hidden_dim, num_layers, dropout, pad_idx).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

@torch.no_grad()
def generate_text(prompt: str, max_new_tokens: int = 60, temperature: float = 1.0, top_k: int = 50) -> str:
    prompt_tokens = simple_tokenize(prompt)
    prompt_ids = [stoi.get(t, unk_idx) for t in prompt_tokens]
    if len(prompt_ids) == 0:
        prompt_ids = [unk_idx]

    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    logits, hidden = model(x, None)

    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        next_logits = logits[0, -1]
        next_logits = next_logits / max(temperature, 1e-6)
        next_logits = top_k_filter(next_logits, top_k)

        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_id)

        x_next = torch.tensor([[next_id]], dtype=torch.long, device=device)
        logits, hidden = model(x_next, hidden)

    return " ".join(itos[i] if i < len(itos) else "<unk>" for i in generated)

#flask
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    output = None

    # defaults
    prompt = "Harry Potter is"
    max_new = 80
    temperature = 1.0
    top_k = 50

    if request.method == "POST":
        prompt = request.form.get("prompt", prompt)
        max_new = int(request.form.get("max_new", max_new))
        temperature = float(request.form.get("temperature", temperature))
        top_k = int(request.form.get("top_k", top_k))

        output = generate_text(
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k
        )

    return render_template(
        "index.html",
        output=output,
        prompt=prompt,
        max_new=max_new,
        temperature=temperature,
        top_k=top_k
    )

if __name__ == "__main__":
    app.run(debug=True)
