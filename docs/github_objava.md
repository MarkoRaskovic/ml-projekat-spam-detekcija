# Objavljivanje projekta na GitHub

Projekat je lokalno pripremljen za objavu. GitHub CLI je preuzet lokalno u `.tools/` folder, koji je ignorisan kroz `.gitignore` i ne ulazi u repozitorijum.

Kada budes zeleo objavu, iz glavnog foldera projekta pokreni:

```bash
export PATH="$PWD/.tools/bin:$PATH"
export GH_CONFIG_DIR="$PWD/.gh-config"
gh auth login
```

Zatim, za pravljenje GitHub repozitorijuma i prvi push:

```bash
gh repo create ml-projekat-spam-detekcija --private --source=. --remote=origin --push
```

Ako zelis da repozitorijum bude javni, umesto `--private` koristi:

```bash
gh repo create ml-projekat-spam-detekcija --public --source=. --remote=origin --push
```

Pre objave proveri:

```bash
git status
git log --oneline -5
```

Ako postoje necommitovane izmene koje treba da udju na GitHub, prvo ih treba commitovati.
