# LSTM Text Generator (Harry Potter)

This project implements a word-level LSTM language model trained on narrative text from the Harry Potter book series. The objective is to demonstrate how recurrent neural networks learn statistical language patterns and generate text given an input prompt.

The project was developed as part of an NLP assignment and follows a complete pipeline from dataset preparation to model training and web-based text generation.

---

## Project Structure
```
.
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
├── data/
│   └── hp_kaggle_corpus.txt
├── artifacts/ 
│   └── lm_lstm_best.pt
├── data_raw/ 
│   └── *.txt
├── notebook.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The dataset consists of plain-text versions of the Harry Potter books obtained from Kaggle.  
https://www.kaggle.com/datasets/shubhammaindola/harry-potter-books?resource=download

Processing steps:
1. Load multiple book .txt files
2. Apply minimal text cleaning
3. Merge all books into a single corpus
4. Tokenize the text at the word level

---

## Model Architecture

The language model uses a standard word-level LSTM architecture:

- Embedding layer
- LSTM layer(s)
- Linear output layer over the vocabulary

The model is trained to predict the next word in a sequence using cross-entropy loss.

---

## Training and Evaluation

Model performance is evaluated using cross-entropy loss and perplexity.

Validation loss decreased steadily during early training and reached its minimum at epoch 22. After this point, training loss continued to decrease while validation loss began to increase, indicating overfitting. Therefore, the final model was selected using early stopping based on the minimum validation loss rather than the final training epoch.

The selected checkpoint is saved as:

artifacts/lm_lstm_best.pt

---

## Text Generation Demo

A minimal Flask web application is provided to demonstrate text generation.

Features:
- Prompt-based text generation
- Temperature and top-k sampling
- Clean HTML/CSS interface

### Run the application

cd app
python app.py

Then open the following URL in a browser:

http://127.0.0.1:5000/

---

## Notes

- This project uses a word-level LSTM without attention mechanisms.
- Generated text may show grammatical structure but limited long-range coherence, which is expected for this model type.
- Raw datasets and trained model weights should not be publicly redistributed.

---

## Requirements

Install dependencies using:

pip install -r requirements.txt

Python version: 3.11
