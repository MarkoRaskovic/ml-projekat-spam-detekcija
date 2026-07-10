# Propozicije - checklista projekta

Ovaj fajl mapira zahteve iz propozicija na konkretne delove projekta.

| Zahtev | Gde je pokriven |
|---|---|
| README sa opisom projekta | `README.md` |
| Opis skupa podataka | `README.md`, `notebooks/06_demo_projekta.ipynb` |
| Koriscena literatura | `README.md` |
| Imena clanova tima | `README.md` |
| Python skripte sa jasnim redosledom | `scripts/01_prepare_data.py`, `scripts/03_train_tfidf_logreg.py`, `scripts/05_train_lstm.py`, `scripts/06_compare_models.py` |
| Finalna Jupyter demo sveska | `notebooks/06_demo_projekta.ipynb` |
| Osnovna analiza skupa | `README.md`, `notebooks/06_demo_projekta.ipynb` |
| Ne/balansiranost podataka | `README.md`, `notebooks/06_demo_projekta.ipynb` |
| Train/validation/test podela | `data/processed/splits/`, `notebooks/06_demo_projekta.ipynb` |
| Validacija i izbor hiperparametara | `scripts/03_train_tfidf_logreg.py`, `reports/03_tfidf_logreg_results.txt` |
| Matrice konfuzije i metrike | `reports/03_tfidf_logreg_results.txt`, `reports/04_tfidf_test_results.txt`, `reports/05_lstm_results.txt` |
| Graficko poredjenje modela | `reports/model_comparison.png` |
| Sacuvani modeli | `models/` |
| Listing paketa | `requirements.txt` |
| Smernice za podesavanje okruzenja | `README.md`, `.vscode/tasks.json` |

Napomena: test skup postoji, ali se ne koristi tokom izbora modela i hiperparametara. Koristi se samo u skripti `scripts/04_evaluate_tfidf_test.py` za finalnu procenu.
