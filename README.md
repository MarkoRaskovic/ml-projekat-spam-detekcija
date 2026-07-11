# Detekcija spam mejlova pomocu klasicnih NLP metoda

Projekat iz predmeta Masinsko ucenje. Cilj projekta je klasifikacija tekstualnih mejlova na dve klase:

- `ham` - regularan mejl
- `spam` - nezeljen/reklamni/prevarni mejl

U projektu su implementirana i evaluirana dva pristupa:

- `TF-IDF + Logistic Regression`
- `LSTM` rekurentna neuronska mreza

## Clan tima

- Marko Raskovic

## Skup podataka

Koristi se `Ling-Spam` skup podataka. Skup sadrzi tekstove mejlova obelezene kao `ham` ili `spam`. U projektu se koristi `bare` verzija korpusa, bez lematizacije i bez unapred uklonjenih stop reci.

Osnovne karakteristike originalnog skupa:

- ukupan broj mejlova: 2893
- ham mejlovi: 2412 (83.37%)
- spam mejlovi: 481 (16.63%)
- broj dupliranih mejlova: 17

Posle uklanjanja duplikata ostaje 2876 mejlova. Skup je nebalansiran, jer je spam mejlova znatno manje od ham mejlova.

## Podela podataka

Podaci se dele stratifikovano, da bi odnos `ham` i `spam` mejlova ostao isti u svakom delu:

- trening skup: 70%
- validacioni skup: 15%
- test skup: 15%

Validacioni skup se koristi za izbor modela i hiperparametara. Test skup se cuva za finalnu procenu i ne koristi se pri svakom treniranju.

## Modeli

### TF-IDF + Logistic Regression

Tekst mejla se prvo pretvara u numericke atribute pomocu TF-IDF vektorizacije. Zatim se nad tim atributima trenira logisticka regresija.

Kod ovog pristupa menjaju se hiperparametri TF-IDF vektorizacije i logisticke regresije:

- `min_df`
- `C` za logisticku regresiju

Model se bira na osnovu rezultata na validacionom skupu. Pretraga hiperparametara je namerno ogranicena na mali broj kandidata da bi se projekat brzo pokretao na laptopu.

### LSTM

LSTM model tekst posmatra kao niz reci. Tekstovi se tokenizuju, pretvaraju u redne brojeve iz recnika, skracuju ili dopunjuju do iste duzine, a zatim se trenira PyTorch LSTM klasifikator.

Kod LSTM pristupa menjaju se:

- velicina recnika
- broj epoha
- dimenzija embedding sloja
- dimenzija LSTM sloja

I ovaj model se bira na osnovu rezultata na validacionom skupu.

## Rezultati

Rezultati treniranja i evaluacije modela cuvaju se u folderu `reports/`.

Najvazniji fajlovi:

- `reports/03_tfidf_logreg_results.txt` - izbor hiperparametara i validacija TF-IDF modela
- `reports/04_tfidf_test_results.txt` - finalna provera TF-IDF modela na test skupu
- `reports/05_lstm_results.txt` - validacija LSTM modela
- `reports/06_lstm_test_results.txt` - finalna provera LSTM modela na test skupu

Za spam detekciju su najvaznije metrike `precision`, `recall` i `F1-score`, jer je skup nebalansiran.

Kratak pregled trenutnih rezultata:

| Model | Skup | Spam precision | Spam recall | Spam F1 |
|---|---|---:|---:|---:|
| TF-IDF + Logistic Regression | validacija | 0.99 | 0.99 | 0.99 |
| LSTM | validacija | 0.98 | 0.91 | 0.95 |
| TF-IDF + Logistic Regression | test | 0.99 | 0.96 | 0.97 |
| LSTM | test | 0.94 | 0.90 | 0.92 |

## Sacuvani modeli

Modeli su sacuvani u folderu `models/`:

- `tfidf_logreg_pipeline.joblib`
- `lstm_model.pt`
- `lstm_vocab.joblib`

Fajlovi su mali, pa mogu da budu deo GitHub repozitorijuma.

## Literatura

- Ling-Spam corpus, Ion Androutsopoulos
- I. Androutsopoulos, J. Koutsias, K.V. Chandrinos, G. Paliouras, C.D. Spyropoulos: `An Evaluation of Naive Bayesian Anti-Spam Filtering`, ECML 2000
- scikit-learn dokumentacija: `TfidfVectorizer`, `LogisticRegression`, metrike klasifikacije
- PyTorch dokumentacija: `Embedding`, `LSTM`, `BCEWithLogitsLoss`
- Materijali sa vezbi iz Masinskog ucenja: `matf-ml/materijali-sa-vezbi-2026`
