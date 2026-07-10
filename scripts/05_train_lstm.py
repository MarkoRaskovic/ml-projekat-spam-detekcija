from collections import Counter
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

TRAIN_PATH = ROOT / "data" / "processed" / "splits" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "processed" / "splits" / "validation.csv"

MODEL_PATH = ROOT / "models" / "lstm_model.pt"
VOCAB_PATH = ROOT / "models" / "lstm_vocab.joblib"
RESULTS_PATH = ROOT / "reports" / "05_lstm_results.txt"


MAX_WORDS = 5000
MAX_LEN = 50
BATCH_SIZE = 64
EPOCHS = 8
EMBEDDING_DIM = 64
LSTM_DIM = 64


torch.manual_seed(42)

# Na MacBook M2 racunaru PyTorch moze koristiti Apple GPU preko MPS-a.
device = "mps" if torch.backends.mps.is_available() else "cpu"


train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", str(text).lower())


# Recnik pravimo samo na trening skupu da ne bismo gledali validacione podatke.
word_counter = Counter()

for message in train_df["message"]:
    word_counter.update(tokenize(message))

vocab = {"<PAD>": 0, "<UNK>": 1}

for word, _ in word_counter.most_common(MAX_WORDS - 2):
    vocab[word] = len(vocab)


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


train_dataset = make_dataset(train_df)
validation_dataset = make_dataset(validation_df)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_DIM, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(LSTM_DIM, 1)

    def forward(self, x, lengths):
        # Padding reci ne treba da utice na LSTM, pa koristimo stvarne duzine poruka.
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.linear(x)
        return x.squeeze(1)


model = LSTMClassifier(len(vocab)).to(device)

spam_count = (train_df["label"] == "spam").sum()
ham_count = (train_df["label"] == "ham").sum()
pos_weight = torch.tensor([ham_count / spam_count], dtype=torch.float32).to(device)

# Spam je manjinska klasa, pa joj dajemo vecu tezinu u funkciji greske.
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for x_batch, lengths, y_batch in train_loader:
        x_batch = x_batch.to(device)
        lengths = lengths.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch, lengths)
        loss = loss_function(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {average_loss:.8f}")


model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for x_batch, lengths, y_batch in validation_loader:
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

validation_report = classification_report(real_labels, predicted_labels)
validation_confusion_matrix = confusion_matrix(real_labels, predicted_labels, labels=["ham", "spam"])

print("\nVALIDACIONI SKUP")
print(validation_report)

print("MATRICA KONFUZIJE NA VALIDACIONOM SKUPU")
print(pd.DataFrame(validation_confusion_matrix, index=["stvarno_ham", "stvarno_spam"], columns=["pred_ham", "pred_spam"]))


MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.save(model.state_dict(), MODEL_PATH)
joblib.dump(
    {
        "vocab": vocab,
        "max_len": MAX_LEN,
        "embedding_dim": EMBEDDING_DIM,
        "lstm_dim": LSTM_DIM,
    },
    VOCAB_PATH,
)

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    file.write("HIPERPARAMETRI\n")
    file.write(f"max_words: {MAX_WORDS}\n")
    file.write(f"max_len: {MAX_LEN}\n")
    file.write(f"batch_size: {BATCH_SIZE}\n")
    file.write(f"epochs: {EPOCHS}\n")
    file.write(f"embedding_dim: {EMBEDDING_DIM}\n")
    file.write(f"lstm_dim: {LSTM_DIM}\n")
    file.write(f"device: {device}\n\n")
    file.write("VALIDACIONI SKUP\n")
    file.write(validation_report)
    file.write("\n\nMATRICA KONFUZIJE NA VALIDACIONOM SKUPU\n")
    file.write(str(pd.DataFrame(validation_confusion_matrix, index=["stvarno_ham", "stvarno_spam"], columns=["pred_ham", "pred_spam"])))

print(f"\nModel je sacuvan u: {MODEL_PATH}")
print(f"Recnik je sacuvan u: {VOCAB_PATH}")
print(f"Rezultati su sacuvani u: {RESULTS_PATH}")
