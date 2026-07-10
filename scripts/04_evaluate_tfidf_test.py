from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Potrebno je na kraju proveriti model na test skupu podataka.

ROOT = Path(__file__).resolve().parents[1]

TEST_PATH = ROOT / "data" / "processed" / "splits" / "test.csv"
MODEL_PATH = ROOT / "models" / "tfidf_logreg_pipeline.joblib"
RESULTS_PATH = ROOT / "reports" / "04_tfidf_test_results.txt"


test_df = pd.read_csv(TEST_PATH)
model = joblib.load(MODEL_PATH)

x_test = test_df["message"]
y_test = test_df["label"]

test_predictions = model.predict(x_test)

test_report = classification_report(y_test, test_predictions)
test_confusion_matrix = confusion_matrix(y_test, test_predictions, labels=["ham", "spam"])

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

print(f"\nRezultati testa su sacuvani u: {RESULTS_PATH}")
