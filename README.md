# Neural transfer project

## Setup (come funziona)

Il Trasferimento di Stile Neurale (NTS) è una tecnica che genera un’immagine “ibrida” combinando il contenuto di un'immagine (es. foto) con lo stile di un'altra (es. dipinto). Si usa una CNN pre-addestrata (VGG19): il contenuto è rappresentato dalle attivazioni dei layer profondi (catturano struttura/oggetti), mentre lo stile è modellato con le matrici di Gram delle feature su più layer (catturano texture e correlazioni tra filtri). L’immagine generata viene ottimizzata per minimizzare la loss totale

$$
\mathcal{L} = \alpha\, \mathcal{L}_{\text{content}} + \beta\, \mathcal{L}_{\text{style}}
$$

così da bilanciare fedeltà al contenuto e coerenza stilistica. Si è usato VGG19 considerando solo i layer convoluzionali. Per studiare la fusione stile–contenuto:

contenuto = foto di Timothée Chalamet;  
stile = “Notte stellata” (Van Gogh), “L’Urlo” (Munch), “Convergence” (Pollock), “Giudizio universale” (Michelangelo).

# Risultati ottenuti

Dalle ricostruzioni emerge che i primi layer conservano dettagli locali (bordi/texture fini), mentre i layer profondi astraggono la struttura globale/semantica del contenuto. Per lo stile, i layer superficiali rendono pattern minuti, quelli profondi consolidano coerenza stilistica globale. Nelle immagini finali stile+contenuto, il bilanciamento α/β e i pesi dei layer di contenuto determinano quanta texture “invade” la scena: enfatizzare conv1 produce trame fini; conv5 dà tratti stilistici ampi. Tracciando le perdite, lo stile domina la loss nonostante β≪α, perché partendo dalla foto il termine di contenuto è inizialmente piccolo. 


# Prerequisiti:

- Python 3.9+

- pip aggiornato

- (Opzionale) GPU NVIDIA con driver aggiornati

  ---

# Guida d'utilizzo

> Esegui i comandi dalla cartella del progetto (es. `C:\NTS`).  
> Le immagini di input vanno in `data/`, l’output viene salvato in `results/`.

## 1) Preparazione ambiente

### Windows (PowerShell)
```powershell
D:
cd NTS
python -m venv .venv
.\.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux (Bash)
```
cd ~/NTS
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
# 2) Inserisci i dati

Metti le immagini in data/, ad esempio:

```
data/Themothee.png
data/Notte stellata.jpg
```

>Nomi con spazi sono ok se usi le virgolette nei comandi.
>Le immagini RGBA vengono convertite in RGB dal loader.

#3) Esecuzione
Avvio con i default

Se i file sopra esistono:
```
python .\src\core.py
```
Avvio con parametri personalizzati
### Windows (PowerShell) 
```
python .\src\core.py --content_path "data\YourContent.jpg" --style_path "data\Notte stellata.jpg" --output_path "results\out.png" --imsize 384 --steps 400 --lr 0.02 --alpha 1.0 --beta 80000 --init content
```

### macOS / Linux (bash)
```
python ./src/core.py --content_path "data/YourContent.jpg" --style_path "data/Notte stellata.jpg" --output_path "results/out.png" --imsize 384 --steps 400 --lr 0.02 --alpha 1.0 --beta 80000 --init content
```

Parametri principali

-content_path / --style_path / --output_path

-imsize → 256 / 384 / 512 (più grande = più lento, più dettagli)

-steps → iterazioni di ottimizzazione (200–1000 per test rapidi)

-lr → learning rate (0.01–0.05 tipico)

-alpha → peso contenuto

-beta → peso stile (di solito alto: 50_000–100_000)

-init → content (stabile) o noise (più creativo)

Durante il run vedrai log della loss; l’immagine viene salvata in --output_path.
