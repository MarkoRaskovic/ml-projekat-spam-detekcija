# Detekcija spam SMS poruka pomocu klasicnih NLP metoda

Projekat iz predmeta Masinsko ucenje. Cilj projekta je klasifikacija SMS poruka na dve klase:

- `ham` - regularna poruka
- `spam` - nezeljena/reklamna/prevarna poruka

U projektu su implementirana i evaluirana dva pristupa:

- `TF-IDF + Logistic Regression`
- `LSTM` rekurentna neuronska mreza

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

Kod ovog pristupa menjaju se hiperparametri TF-IDF vektorizacije i logisticke regresije:

- `ngram_range`
- `min_df`
- `max_df`
- `C` za logisticku regresiju

Model se bira na osnovu rezultata na validacionom skupu.

### LSTM

LSTM model poruku posmatra kao niz reci. Poruke se tokenizuju, pretvaraju u redne brojeve iz recnika, skracuju ili dopunjuju do iste duzine, a zatim se trenira PyTorch LSTM klasifikator.

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
| TF-IDF + Logistic Regression | validacija | 0.94 | 0.94 | 0.94 |
| LSTM | validacija | 0.90 | 0.91 | 0.90 |
| TF-IDF + Logistic Regression | test | 0.94 | 0.92 | 0.93 |
| LSTM | test | 0.90 | 0.88 | 0.89 |

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
