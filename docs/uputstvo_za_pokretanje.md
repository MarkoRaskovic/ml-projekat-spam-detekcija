# Uputstvo za pokretanje

Ovaj fajl navodi pakete i osnovne korake za podesavanje okruzenja, kako bi projekat mogao da se testira.

## Paketi

Paketi su navedeni u fajlu `requirements.txt`:

```text
pandas
scikit-learn
joblib
torch
matplotlib
ipykernel
```

## Podesavanje okruzenja

Iz glavnog foldera projekta:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Redosled pokretanja skripti

```bash
.venv/bin/python scripts/01_prepare_data.py
.venv/bin/python scripts/03_train_tfidf_logreg.py
.venv/bin/python scripts/05_train_lstm.py
```

Finalna provera na test skupu pokrece se posebno:

```bash
.venv/bin/python scripts/04_evaluate_tfidf_test.py
.venv/bin/python scripts/06_evaluate_lstm_test.py
```

## Jupyter sveske

Sveske u folderu `notebooks/` numerisane su redosledom kojim se projekat pregleda:

1. `01_priprema_podataka.ipynb`
2. `02_treniranje_tfidf_logreg.ipynb`
3. `03_finalna_evaluacija_tfidf.ipynb`
4. `04_treniranje_lstm.ipynb`
5. `05_finalna_evaluacija_lstm.ipynb`
6. `06_poredjenje_modela.ipynb`
