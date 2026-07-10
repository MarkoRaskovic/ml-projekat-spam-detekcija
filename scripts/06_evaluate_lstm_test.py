from pathlib import Path
import re

import joblib
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.csv"
MODEL_PATH = ROOT / "models" / "lstm_model.pt"
VOCAB_PATH = ROOT / "models" / "lstm_vocab.joblib"
RESULTS_PATH = ROOT / "reports" / "06_lstm_test_results.txt"

BATCH_SIZE = 64


device = "mps" if torch.backends.mps.is_available() else "cpu"

test_df = pd.read_csv(TEST_PATH)
model_info = joblib.load(VOCAB_PATH)

vocab = model_info["vocab"]
MAX_LEN = model_info["max_len"]
EMBEDDING_DIM = model_info["embedding_dim"]
LSTM_DIM = model_info["lstm_dim"]


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", str(text).lower())


def encode_message(message):
    tokens = tokenize(message)
    numbers = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    numbers = numbers[:MAX_LEN]
    length = max(1, len(numbers))
    numbers += [vocab["<PAD>"]] * (MAX_LEN - len(numbers))
    return numbers, length


def make_dataset(df):
    encoded_messages = [encode_message(message) for message in df["message"]]
    x = [message for message, _ in encoded_messages]
    lengths = [length for _, length in encoded_messages]
    y = [1 if label == "spam" else 0 for label in df["label"]]

    x = torch.tensor(x, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)

    return TensorDataset(x, lengths, y)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_DIM, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(LSTM_DIM, 1)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.linear(x)
        return x.squeeze(1)


test_dataset = make_dataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

model = LSTMClassifier(len(vocab)).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for x_batch, lengths, y_batch in test_loader:
        x_batch = x_batch.to(device)
        lengths = lengths.to(device)

        logits = model(x_batch, lengths)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= 0.5).int().cpu().tolist()

        all_predictions.extend(predictions)
        all_labels.extend(y_batch.int().tolist())


label_names = ["ham", "spam"]
real_labels = [label_names[label] for label in all_labels]
predicted_labels = [label_names[label] for label in all_predictions]

test_report = classification_report(real_labels, predicted_labels)
test_confusion_matrix = confusion_matrix(real_labels, predicted_labels, labels=["ham", "spam"])

print("TEST SKUP")
print(test_report)

print("MATRICA KONFUZIJE NA TEST SKUPU")
print(pd.DataFrame(test_confusion_matrix, index=["stvarno_ham", "stvarno_spam"], columns=["pred_ham", "pred_spam"]))

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    file.write("TEST SKUP\n")
    file.write(test_report)
    file.write("\n\nMATRICA KONFUZIJE NA TEST SKUPU\n")
    file.write(str(pd.DataFrame(test_confusion_matrix, index=["stvarno_ham", "stvarno_spam"], columns=["pred_ham", "pred_spam"])))

print(f"\nRezultati LSTM testa su sacuvani u: {RESULTS_PATH}")
