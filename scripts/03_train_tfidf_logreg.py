from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

TRAIN_PATH = ROOT / "data" / "processed" / "splits" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "processed" / "splits" / "validation.csv"

MODEL_PATH = ROOT / "models" / "tfidf_logreg_pipeline.joblib"
RESULTS_PATH = ROOT / "reports" / "03_tfidf_logreg_results.txt"


train_df = pd.read_csv(TRAIN_PATH)
validation_df = pd.read_csv(VALIDATION_PATH)

x_train = train_df["message"]
y_train = train_df["label"]

x_validation = validation_df["message"]
y_validation = validation_df["label"]


# Biramo nekoliko brzih kombinacija hiperparametara.
# Najbolji model biramo prema F1 meri za spam klasu na validacionom skupu.
candidate_parameters = [
    {"ngram_range": (1, 1), "min_df": 2, "max_df": 0.95, "C": 1.0},
    {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "C": 1.0},
    {"ngram_range": (1, 2), "min_df": 5, "max_df": 0.90, "C": 5.0},
    {"ngram_range": (1, 2), "min_df": 2, "max_df": 0.95, "C": 5.0},
]


def napravi_model(params):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=params["ngram_range"],
                    min_df=params["min_df"],
                    max_df=params["max_df"],
                    sublinear_tf=True,
                ),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    C=params["C"],
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


all_results = []
best_model = None
best_params = None
best_spam_f1 = -1

for params in candidate_parameters:
    candidate_model = napravi_model(params)
    candidate_model.fit(x_train, y_train)
    predictions = candidate_model.predict(x_validation)

    spam_precision = precision_score(y_validation, predictions, pos_label="spam", zero_division=0)
    spam_recall = recall_score(y_validation, predictions, pos_label="spam", zero_division=0)
    spam_f1 = f1_score(y_validation, predictions, pos_label="spam", zero_division=0)

    all_results.append([params["ngram_range"], params["min_df"], params["max_df"], params["C"], spam_precision, spam_recall, spam_f1])

    if spam_f1 > best_spam_f1:
        best_spam_f1 = spam_f1
        best_params = params
        best_model = candidate_model


validation_predictions = best_model.predict(x_validation)

validation_report = classification_report(y_validation, validation_predictions)
validation_confusion_matrix = confusion_matrix(y_validation, validation_predictions, labels=["ham", "spam"])
results_table = pd.DataFrame(
    all_results,
    columns=["ngram_range", "min_df", "max_df", "C", "spam_precision", "spam_recall", "spam_f1"],
).sort_values("spam_f1", ascending=False)

print("IZBOR HIPERPARAMETARA")
print(results_table)
print(f"\nNajbolji parametri: {best_params}")
print("VALIDACIONI SKUP")
print(validation_report)

print("MATRICA KONFUZIJE NA VALIDACIONOM SKUPU")
print(pd.DataFrame(validation_confusion_matrix, index=["stvarno_ham", "stvarno_spam"], columns=["pred_ham", "pred_spam"]))

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, MODEL_PATH)

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    file.write("IZBOR HIPERPARAMETARA\n")
    file.write(str(results_table))
    file.write(f"\n\nNajbolji parametri: {best_params}\n\n")
    file.write("VALIDACIONI SKUP\n")
    file.write(validation_report)
    file.write("\n\nMATRICA KONFUZIJE NA VALIDACIONOM SKUPU\n")
    file.write(str(pd.DataFrame(validation_confusion_matrix, index=["stvarno_ham", "stvarno_spam"], columns=["pred_ham", "pred_spam"])))

print(f"\nModel je sacuvan u: {MODEL_PATH}")
print(f"Rezultati su sacuvani u: {RESULTS_PATH}")
