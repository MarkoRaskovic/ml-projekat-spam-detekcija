from pathlib import Path

from itertools import product
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
# Najbolji model biramo prema F1 meri za spam i ham klasu na validacionom skupu.
candidate_parameters = [(i,j*0.2) for i in range(2,21) for j in range(1,51)]

# Pravimo model koji izvrsava logisticku regresiju.
def napravi_model(min_df_param,C_param):
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1,2),
                    min_df=min_df_param,
                    max_df=0.90,
                    sublinear_tf=True,
                ),
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    C=C_param,
                    max_iter=1000,
                    class_weight="balanced",    # Koristimo class_weight="balanced" jer podaci nisu balansirani.
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
best_ham_f1 = -1

# Pokrecemo model za sve razlicite parametre i cuvamo najbolje.

for params in candidate_parameters:
    (min_df_param,C_param)=params
    candidate_model = napravi_model(min_df_param,C_param)
    candidate_model.fit(x_train, y_train)
    predictions = candidate_model.predict(x_validation)
    spam_f1 = f1_score(y_validation, predictions, pos_label="spam", zero_division=0)
    ham_f1 = f1_score(y_validation, predictions, pos_label="ham", zero_division=0)

    all_results.append([min_df_param,C_param, spam_f1,ham_f1])

    if spam_f1+ham_f1 > best_spam_f1+ham_f1:
        best_spam_f1 = spam_f1
        best_ham_f1=ham_f1
        best_params = params
        best_model = candidate_model

#Ispisujemo rezultate najboljeg parametra

validation_predictions = best_model.predict(x_validation)

validation_report = classification_report(y_validation, validation_predictions)
validation_confusion_matrix = confusion_matrix(y_validation, validation_predictions, labels=["ham", "spam"])
results_table = pd.DataFrame(
    all_results,
    columns=["min_df", "C", "spam_f1", "ham_f1"],
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

# Zapisujemo rezultate u fajl

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
