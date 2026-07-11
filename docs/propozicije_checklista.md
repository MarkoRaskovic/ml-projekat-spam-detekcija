# Propozicije - checklista projekta

Ovaj fajl mapira zahteve iz propozicija na konkretne delove projekta.

| Zahtev | Gde je pokriven |
|---|---|
| README sa opisom projekta | `README.md` |
| Opis skupa podataka | `README.md`, `notebooks/01_priprema_podataka.ipynb` |
| Koriscena literatura | `README.md` |
| Imena clanova tima | `README.md` |
| Python skripte sa jasnim redosledom | `scripts/01_prepare_data.py`, `scripts/03_train_tfidf_logreg.py`, `scripts/05_train_lstm.py` |
| Numerisane Jupyter sveske | `notebooks/01_priprema_podataka.ipynb` - `notebooks/06_poredjenje_modela.ipynb` |
| Osnovna analiza skupa | `README.md`, `notebooks/01_priprema_podataka.ipynb` |
| Ne/balansiranost podataka | `README.md`, `notebooks/01_priprema_podataka.ipynb` |
| Train/validation/test podela | `data/processed/splits/`, `notebooks/01_priprema_podataka.ipynb` |
| Validacija i izbor hiperparametara | `notebooks/02_treniranje_tfidf_logreg.ipynb`, `notebooks/04_treniranje_lstm.ipynb`, `reports/03_tfidf_logreg_results.txt`, `reports/05_lstm_results.txt` |
| Matrice konfuzije i metrike | `notebooks/03_finalna_evaluacija_tfidf.ipynb`, `notebooks/05_finalna_evaluacija_lstm.ipynb`, `reports/03_tfidf_logreg_results.txt`, `reports/04_tfidf_test_results.txt`, `reports/05_lstm_results.txt`, `reports/06_lstm_test_results.txt` |
| Poredjenje modela | `README.md`, `notebooks/06_poredjenje_modela.ipynb` |
| Sacuvani modeli | `models/` |
| Listing paketa | `requirements.txt`, `docs/uputstvo_za_pokretanje.md` |
| Smernice za podesavanje okruzenja | `docs/uputstvo_za_pokretanje.md`, `.vscode/tasks.json` |

Napomena: test skup postoji, ali se ne koristi tokom izbora modela i hiperparametara. Koristi se samo u skriptama `scripts/04_evaluate_tfidf_test.py` i `scripts/06_evaluate_lstm_test.py` za finalnu procenu.
