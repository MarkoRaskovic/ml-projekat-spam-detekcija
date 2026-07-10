# Detekcija spam SMS poruka pomocu klasicnih NLP metoda

Projekat iz predmeta Masinsko ucenje. Cilj projekta je klasifikacija SMS poruka na dve klase:

- `ham` - regularna poruka
- `spam` - nezeljena/reklamna/prevarna poruka

Glavni model je klasicni NLP pristup `TF-IDF + Logistic Regression`. Zbog zahteva projekta dodat je i jednostavan `LSTM` rekurentni neuronski model.

## Clan tima

- Marko Raskovic

## Skup podataka

Koristi se `UCI SMS Spam Collection` skup podataka. Skup sadrzi SMS poruke obelezene kao `ham` ili `spam`.

Osnovne karakteristike originalnog skupa:

- ukupan broj poruka: 5574
- ham poruke: 4827 (86.60%)
- spam poruke: 747 (13.40%)
- broj dupliranih poruka: 403

Posle uklanjanja duplikata ostaje 5171 poruka. Skup je nebalansiran, jer je spam poruka znatno manje od ham poruka.

## Podela podataka

Podaci se dele stratifikovano, da bi odnos `ham` i `spam` poruka ostao isti u svakom delu:

- trening skup: 70%
- validacioni skup: 15%
- test skup: 15%

Validacioni skup se koristi za izbor modela i hiperparametara. Test skup se cuva za finalnu procenu i ne koristi se pri svakom treniranju.

## Modeli

### TF-IDF + Logistic Regression

Tekstualna poruka se prvo pretvara u numericke atribute pomocu TF-IDF vektorizacije. Zatim se nad tim atributima trenira logisticka regresija.

U skripti se proba nekoliko kombinacija hiperparametara:

- `ngram_range`
- `min_df`
- `max_df`
- `C` za logisticku regresiju

Najbolja kombinacija se bira prema `F1` meri za spam klasu na validacionom skupu.

### LSTM

LSTM model se koristi kao jednostavan rekurentni model. Poruke se tokenizuju, pretvaraju u redne brojeve iz recnika, skracuju ili dopunjuju do iste duzine, a zatim se trenira PyTorch LSTM klasifikator.

Ovaj model je dodat da projekat ispuni zahtev da postoji rekurentni model, ali je glavni odbranjivi model `TF-IDF + Logistic Regression`, jer je brzi, jednostavniji i daje bolje ili uporedive rezultate na ovom malom skupu.

## Rezultati

Rezultati treniranja i evaluacije modela cuvaju se u folderu `reports/`.

Najvazniji fajlovi:

- `reports/03_tfidf_logreg_results.txt` - izbor hiperparametara i validacija TF-IDF modela
- `reports/04_tfidf_test_results.txt` - finalna provera TF-IDF modela na test skupu
- `reports/05_lstm_results.txt` - validacija LSTM modela

Za spam detekciju su najvaznije metrike `precision`, `recall` i `F1-score`, jer je skup nebalansiran.

Kratak pregled trenutnih rezultata:

| Model | Skup | Spam precision | Spam recall | Spam F1 |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | validacija | 0.94 | 0.94 | 0.94 |
| LSTM | validacija | 0.91 | 0.89 | 0.90 |
| TF-IDF + Logistic Regression | test | 0.94 | 0.92 | 0.93 |

## Struktura projekta

```text
data/
  raw/                 originalni UCI podaci
  processed/           obradjeni podaci
  processed/splits/    train, validation i test skup
docs/                  pomocni materijali za predaju i odbranu
models/                sacuvani modeli i recnici
notebooks/             finalna Jupyter demo sveska
reports/               tekstualni rezultati i grafikon poredjenja
scripts/               Python skripte
.vscode/               VS Code podesavanja za lako pokretanje
```

## Pokretanje iz terminala

Prvo napraviti okruzenje i instalirati pakete:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Zatim pokrenuti skripte ovim redosledom:

```bash
.venv/bin/python scripts/01_prepare_data.py
.venv/bin/python scripts/03_train_tfidf_logreg.py
.venv/bin/python scripts/05_train_lstm.py
```

Finalni test TF-IDF modela pokrece se posebno:

```bash
.venv/bin/python scripts/04_evaluate_tfidf_test.py
```

## Pokretanje iz VS Code-a

U VS Code-u otvoriti glavni folder projekta:

```text
/Users/markoraskovic/Desktop/ML projekat
```

Ne treba otvarati podfolder `scripts`, jer se tada ne vide taskovi iz `.vscode/tasks.json`.

Najjednostavnije je otvoriti fajl `ML projekat.code-workspace` iz glavnog foldera projekta.

Zatim izabrati:

`Terminal -> Run Task...`

Redosled taskova:

1. `1. Napravi Python okruzenje`
2. `2. Instaliraj pakete`
3. `3. Pripremi podatke`
4. `4. Pokreni TF-IDF model`
5. `6. Pokreni LSTM model`

Task `5. Finalno testiraj TF-IDF model` koristi se samo za finalnu proveru na test skupu.

## Sacuvani modeli

Modeli su sacuvani u folderu `models/`:

- `tfidf_logreg_pipeline.joblib`
- `lstm_model.pt`
- `lstm_vocab.joblib`

Fajlovi su mali, pa mogu da budu deo GitHub repozitorijuma.

## Literatura

- UCI Machine Learning Repository: SMS Spam Collection
- scikit-learn dokumentacija: `TfidfVectorizer`, `LogisticRegression`, metrike klasifikacije
- PyTorch dokumentacija: `Embedding`, `LSTM`, `BCEWithLogitsLoss`
- Materijali sa vezbi iz Masinskog ucenja: `matf-ml/materijali-sa-vezbi-2026`
