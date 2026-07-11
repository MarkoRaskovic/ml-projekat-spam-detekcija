# Plan odbrane

Predvidjeno trajanje prezentacije: do 10 minuta.

## 1. Problem

Cilj je automatska detekcija spam mejlova. To je binarna klasifikacija teksta: svaki mejl je ili `ham` ili `spam`.

## 2. Skup podataka

Koristi se Ling-Spam skup podataka. Skup je nebalansiran, jer je spam klasa manjinska. Zbog toga accuracy nije dovoljna metrika i treba posmatrati precision, recall i F1 za spam klasu.

## 3. Podela podataka

Podaci su podeljeni stratifikovano na trening, validacioni i test skup. Validacioni skup se koristi za izbor hiperparametara, a test skup se cuva za finalnu proveru.

## 4. Klasicni NLP model

Prvi model je `TF-IDF + Logistic Regression`.

Treba objasniti:

- TF-IDF pretvara tekst u brojeve
- logisticka regresija klasifikuje mejl na osnovu tih brojeva
- hiperparametri se biraju prema F1 meri na validaciji
- zbog duzih mejlova pretraga hiperparametara je suzena da bi treniranje bilo brzo

## 5. LSTM model

LSTM je dodat kao rekurentni model. Mejlovi se tokenizuju, pretvaraju u niz brojeva i salju u LSTM. Model pamti redosled reci u tekstu.

## 6. Evaluacija

Fokus odbrane:

- matrica konfuzije
- precision za spam
- recall za spam
- F1 za spam
- zasto se test skup ne koristi pri izboru modela

## 7. Zakljucak

Na tekstualnom spam skupu klasicni TF-IDF model je veoma jak izbor: brzo se trenira, jednostavno se objasnjava i daje dobre rezultate. LSTM ispunjava zahtev za rekurentni model, ali je slozeniji i na ovakvom skupu ne mora da bude bolji od klasicnog pristupa.
