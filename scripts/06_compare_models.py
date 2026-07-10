from pathlib import Path
import os
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TFIDF_REPORT_PATH = ROOT / "reports" / "03_tfidf_logreg_results.txt"
LSTM_REPORT_PATH = ROOT / "reports" / "05_lstm_results.txt"
CHART_PATH = ROOT / "reports" / "model_comparison.png"
RESULTS_PATH = ROOT / "reports" / "06_model_comparison_results.txt"


def spam_metrics(report_path):
    text = report_path.read_text(encoding="utf-8")
    match = re.search(r"^\s*spam\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", text, re.MULTILINE)

    if match is None:
        raise ValueError(f"Ne mogu da procitam spam metrike iz fajla: {report_path}")

    precision, recall, f1 = match.groups()
    return float(precision), float(recall), float(f1)


tfidf_precision, tfidf_recall, tfidf_f1 = spam_metrics(TFIDF_REPORT_PATH)
lstm_precision, lstm_recall, lstm_f1 = spam_metrics(LSTM_REPORT_PATH)

results = pd.DataFrame(
    [
        ["TF-IDF + Logistic Regression", tfidf_precision, tfidf_recall, tfidf_f1],
        ["LSTM", lstm_precision, lstm_recall, lstm_f1],
    ],
    columns=["model", "spam_precision", "spam_recall", "spam_f1"],
)

print("POREDJENJE MODELA NA VALIDACIONOM SKUPU")
print(results)

ax = results.set_index("model")[["spam_precision", "spam_recall", "spam_f1"]].plot(kind="bar", figsize=(9, 5))
ax.set_title("Poredjenje modela na validacionom skupu")
ax.set_ylabel("Vrednost metrike za spam klasu")
ax.set_ylim(0, 1)
ax.set_xlabel("")
ax.legend(["precision", "recall", "F1"], loc="lower right")
plt.xticks(rotation=0)
plt.tight_layout()

CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(CHART_PATH, dpi=160)

best_model = results.sort_values("spam_f1", ascending=False).iloc[0]
conclusion = (
    "Zakljucak: najbolji model prema F1 meri za spam klasu na validacionom skupu je "
    f"{best_model['model']}."
)

with open(RESULTS_PATH, "w", encoding="utf-8") as file:
    file.write("POREDJENJE MODELA NA VALIDACIONOM SKUPU\n")
    file.write(str(results))
    file.write("\n\n")
    file.write(conclusion)

print(f"\nGrafikon je sacuvan u: {CHART_PATH}")
print(f"Rezultat poredjenja je sacuvan u: {RESULTS_PATH}")
