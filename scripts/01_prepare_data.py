from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "SMSSpamCollection"
CLEAN_PATH = ROOT / "data" / "processed" / "sms_spam_clean.csv"
MODELING_PATH = ROOT / "data" / "processed" / "sms_spam_modeling.csv"
SPLIT_DIR = ROOT / "data" / "processed" / "splits"


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
# Cuvamo podatke da bi mogli da im pristupima iz drugih skripti
CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(CLEAN_PATH, index=False)
df_modeling.to_csv(MODELING_PATH, index=False)
train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
validation_df.to_csv(SPLIT_DIR / "validation.csv", index=False)
test_df.to_csv(SPLIT_DIR / "test.csv", index=False)
