from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "SMSSpamCollection"
CLEAN_PATH = ROOT / "data" / "processed" / "sms_spam_clean.csv"
MODELING_PATH = ROOT / "data" / "processed" / "sms_spam_modeling.csv"
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
RESULTS_PATH = ROOT / "reports" / "01_prepare_data_results.txt"


rows = []

# Originalni UCI fajl ima format: labela, tab, tekst poruke.
for line in RAW_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
    label, message = line.split("\t", 1)
    rows.append([label, message])

df = pd.DataFrame(rows, columns=["label", "message"])

# Dodajemo jednostavne osobine koje pomazu u osnovnoj analizi podataka.
df["label_id"] = (df["label"] == "spam").astype(int)
df["char_count"] = df["message"].str.len()
df["word_count"] = df["message"].str.split().str.len()
df["digit_count"] = df["message"].str.count(r"\d")
df["has_currency"] = df["message"].str.contains(r"[$\u00a3\u20ac]", regex=True)
df["has_long_number"] = df["message"].str.contains(r"\d{5,}", regex=True)
df["normalized_message"] = df["message"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

def opis_skupa(name, split_df):
    counts = split_df["label"].value_counts()
    percentages = split_df["label"].value_counts(normalize=True) * 100

    return (
        f"{name}\n"
        f"broj poruka: {len(split_df)}\n"
        f"ham: {counts['ham']} ({percentages['ham']:.2f}%)\n"
        f"spam: {counts['spam']} ({percentages['spam']:.2f}%)"
    )


class_counts = df["label"].value_counts()
class_percentages = df["label"].value_counts(normalize=True) * 100
length_stats = df.groupby("label")[["char_count", "word_count", "digit_count"]].mean().round(2)
duplicate_count = df.duplicated(["label", "message"]).sum()

df_modeling = df.drop_duplicates(["label", "message"]).reset_index(drop=True)

# Stratifikovana podela cuva isti odnos ham/spam poruka u svakom skupu.
train_df, temp_df = train_test_split(
    df_modeling,
    test_size=0.30,
    stratify=df_modeling["label"],
    random_state=42,
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42,
)

CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(CLEAN_PATH, index=False)
df_modeling.to_csv(MODELING_PATH, index=False)
train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
validation_df.to_csv(SPLIT_DIR / "validation.csv", index=False)
test_df.to_csv(SPLIT_DIR / "test.csv", index=False)

train_messages = set(train_df["message"])
validation_messages = set(validation_df["message"])
test_messages = set(test_df["message"])

report = []
report.append("PRIPREMA PODATAKA")
report.append("")
report.append("OSNOVNA ANALIZA")
report.append(f"Ukupan broj poruka: {len(df)}")
report.append(f"Broj ham poruka: {class_counts['ham']} ({class_percentages['ham']:.2f}%)")
report.append(f"Broj spam poruka: {class_counts['spam']} ({class_percentages['spam']:.2f}%)")
report.append(f"Broj dupliranih poruka: {duplicate_count}")
report.append("")
report.append("PROSECNE VREDNOSTI PO KLASI")
report.append(str(length_stats))
report.append("")
report.append("PODELA PODATAKA")
report.append(f"Broj poruka posle uklanjanja duplikata: {len(df_modeling)}")
report.append(f"Uklonjeno duplikata: {len(df) - len(df_modeling)}")
report.append("")
report.append(opis_skupa("TRENING SKUP", train_df))
report.append("")
report.append(opis_skupa("VALIDACIONI SKUP", validation_df))
report.append("")
report.append(opis_skupa("TEST SKUP", test_df))
report.append("")
report.append("PROVERA PREKLAPANJA")
report.append(f"train / validation: {len(train_messages & validation_messages)}")
report.append(f"train / test: {len(train_messages & test_messages)}")
report.append(f"validation / test: {len(validation_messages & test_messages)}")
report.append("")
report.append("ZAKLJUCAK")
report.append("Skup je nebalansiran: ham poruke su mnogo brojnije od spam poruka.")
report.append("Spam poruke su u proseku duze i cesce sadrze brojeve.")
report.append("Test skup je napravljen, ali se ne koristi tokom treniranja modela.")

results = "\n".join(report)

print(results)
RESULTS_PATH.write_text(results, encoding="utf-8")

print(f"\nObradjeni podaci su sacuvani u: {CLEAN_PATH}")
print(f"Podaci za modelovanje su sacuvani u: {MODELING_PATH}")
print(f"Train/validation/test skupovi su sacuvani u: {SPLIT_DIR}")
print(f"Rezultati pripreme su sacuvani u: {RESULTS_PATH}")
