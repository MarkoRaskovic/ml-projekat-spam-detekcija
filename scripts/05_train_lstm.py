from collections import Counter
from pathlib import Path
import re

import joblib
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "splits" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "processed" / "splits" / "validation.csv"

MODEL_PATH = ROOT / "models" / "lstm_model.pt"
VOCAB_PATH = ROOT / "models" / "lstm_vocab.joblib"
RESULTS_PATH = ROOT / "reports" / "05_lstm_results.txt"


MAX_LEN = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DROPOUT = 0.2


candidate_parameters = [
    {"max_words": 1000, "epochs": 8, "embedding_dim": 32, "lstm_dim": 32},
    {"max_words": 3000, "epochs": 8, "embedding_dim": 64, "lstm_dim": 64},
    {"max_words": 5000, "epochs": 8, "embedding_dim": 64, "lstm_dim": 64},
    {"max_words": 5000, "epochs": 12, "embedding_dim": 64, "lstm_dim": 64},
    {"max_words": 5000, "epochs": 16, "embedding_dim": 64, "lstm_dim": 64},
    {"max_words": 5000, "epochs": 20, "embedding_dim": 64, "lstm_dim": 64},
    {"max_words": 5000, "epochs": 12, "embedding_dim": 128, "lstm_dim": 128},
    {"max_words": 5000, "epochs": 16, "embedding_dim": 128, "lstm_dim": 128},
    {"max_words": 5000, "epochs": 20, "embedding_dim": 128, "lstm_dim": 128},
]


torch.manual_seed(42)

# Na MacBook M2 racunaru PyTorch moze koristiti Apple GPU preko MPS-a.
device = "mps" if torch.backends.mps.is_available() else "cpu"


train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)


def tokenize(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", str(text).lower())


word_counter = Counter()

for message in train_df["message"]:
    word_counter.update(tokenize(message))


def make_vocab(max_words):
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for word, _ in word_counter.most_common(max_words - 2):
        vocab[word] = len(vocab)

    return vocab


def encode_message(message, vocab):
    tokens = tokenize(message)
    numbers = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    numbers = numbers[:MAX_LEN]
    length = max(1, len(numbers))
    numbers += [vocab["<PAD>"]] * (MAX_LEN - len(numbers))
    return numbers, length


def make_dataset(df, vocab):
    encoded_messages = [encode_message(message, vocab) for message in df["message"]]
    x = [message for message, _ in encoded_messages]
    lengths = [length for _, length in encoded_messages]
    y = [1 if label == "spam" else 0 for label in df["label"]]

    x = torch.tensor(x, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float32)

    return TensorDataset(x, lengths, y)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.linear = nn.Linear(lstm_dim, 1)

    def forward(self, x, lengths):
        # Padding reci ne treba da utice na LSTM, pa koristimo stvarne duzine poruka.
        x = self.embedding(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden[-1])
        x = self.linear(x)
        return x.squeeze(1)


def train_model(params):
    torch.manual_seed(42)

    vocab = make_vocab(params["max_words"])
    train_dataset = make_dataset(train_df, vocab)
    validation_dataset = make_dataset(validation_df, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    model = LSTMClassifier(len(vocab), params["embedding_dim"], params["lstm_dim"]).to(device)

    spam_count = (train_df["label"] == "spam").sum()
    ham_count = (train_df["label"] == "ham").sum()
    pos_weight = torch.tensor([ham_count / spam_count], dtype=torch.float32).to(device)

    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(params["epochs"]):
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
        print(f"  epoch {epoch + 1}/{params['epochs']} - loss: {average_loss:.8f}")

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

    spam_precision = precision_score(real_labels, predicted_labels, pos_label="spam", zero_division=0)
    spam_recall = recall_score(real_labels, predicted_labels, pos_label="spam", zero_division=0)
    spam_f1 = f1_score(real_labels, predicted_labels, pos_label="spam", zero_division=0)

    return model, vocab, real_labels, predicted_labels, spam_precision, spam_recall, spam_f1


all_results = []
best_model = None
best_vocab = None
best_params = None
best_real_labels = None
best_predicted_labels = None
best_spam_f1 = -1

for index, params in enumerate(candidate_parameters, start=1):
    print(f"\nKandidat {index}/{len(candidate_parameters)}: {params}")
    model, vocab, real_labels, predicted_labels, spam_precision, spam_recall, spam_f1 = train_model(params)

    all_results.append(
        [
            params["max_words"],
            params["epochs"],
            params["embedding_dim"],
            params["lstm_dim"],
            spam_precision,
            spam_recall,
            spam_f1,
        ]
    )

    if spam_f1 > best_spam_f1:
        best_spam_f1 = spam_f1
        best_model = model
        best_vocab = vocab
        best_params = params
        best_real_labels = real_labels
        best_predicted_labels = predicted_labels


results_table = pd.DataFrame(
    all_results,
    columns=["max_words", "epochs", "embedding_dim", "lstm_dim", "spam_precision", "spam_recall", "spam_f1"],
).sort_values("spam_f1", ascending=False)

validation_report = classification_report(best_real_labels, best_predicted_labels)
validation_confusion_matrix = confusion_matrix(best_real_labels, best_predicted_labels, labels=["ham", "spam"])

print("\nIZBOR HIPERPARAMETARA")
print(results_table.to_string(index=False))
print(f"\nNajbolji parametri: {best_params}")

print("\nVALIDACIONI SKUP")
print(validation_report)

print("MATRICA KONFUZIJE NA VALIDACIONOM SKUPU")
print(pd.DataFrame(validation_confusion_matrix, index=["stvarno_ham", "stvarno_spam"], columns=["pred_ham", "pred_spam"]))


MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.save(best_model.state_dict(), MODEL_PATH)
joblib.dump(
    {
        "vocab": best_vocab,
        "max_len": MAX_LEN,
        "embedding_dim": best_params["embedding_dim"],
        "lstm_dim": best_params["lstm_dim"],
    },
    VOCAB_PATH,
)

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    file.write("IZBOR HIPERPARAMETARA\n")
    file.write(results_table.to_string(index=False))
    file.write(f"\n\nNajbolji parametri: {best_params}\n\n")
    file.write("HIPERPARAMETRI NAJBOLJEG MODELA\n")
    file.write(f"max_words: {best_params['max_words']}\n")
    file.write(f"max_len: {MAX_LEN}\n")
    file.write(f"batch_size: {BATCH_SIZE}\n")
    file.write(f"epochs: {best_params['epochs']}\n")
    file.write(f"embedding_dim: {best_params['embedding_dim']}\n")
    file.write(f"lstm_dim: {best_params['lstm_dim']}\n")
    file.write(f"learning_rate: {LEARNING_RATE}\n")
    file.write(f"dropout: {DROPOUT}\n")
    file.write(f"device: {device}\n\n")
    file.write("VALIDACIONI SKUP\n")
    file.write(validation_report)
    file.write("\n\nMATRICA KONFUZIJE NA VALIDACIONOM SKUPU\n")
    file.write(str(pd.DataFrame(validation_confusion_matrix, index=["stvarno_ham", "stvarno_spam"], columns=["pred_ham", "pred_spam"])))

print(f"\nModel je sacuvan u: {MODEL_PATH}")
print(f"Recnik je sacuvan u: {VOCAB_PATH}")
print(f"Rezultati su sacuvani u: {RESULTS_PATH}")
