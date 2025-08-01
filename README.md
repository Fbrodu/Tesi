# Tesi - Classificazione LM386 Originali e Contraffatti

Questo progetto ha l’obiettivo di analizzare dispositivi LM386 tramite l’estrazione di caratteristiche elettriche, con lo scopo di individuare eventuali differenze tra dispositivi originali contraffatti.  
Le misure includono parametri sia DC sia AC.

## Dataset

Il dataset è disponibile in tre versioni:
- `LM386_Features_4D.csv` – dataset completo (Francesca + Simone)
- `LM386_Features_4D_Francesca.csv` – solo dati misurati da Francesca
- `LM386_Features_4D_Simone.csv` – solo dati misurati da Simone

Ogni riga contiene le seguenti colonne:
- `group` – gruppo o famiglia del dispositivo
- `ID` – identificativo del campione
- `original` – True se originale, False se contraffatto
- `quiescent_current` – corrente di riposo (a 6V)
- `voltage_gain` – guadagno misurato a 1kHz
- `cutoff_frequency` – frequenza di taglio (-3 dB)
- `current_slope` – pendenza della curva tensione-corrente
- `T150_aging` 

---

## Workflow del Progetto

### 1. Organizzazione delle Misure
I file delle misure sono contenuti in:
- `Estrazione Features/Current` – misure in DC (corrente)
- `Estrazione Features/Frequenza` – misure in AC (risposta in frequenza)  
All’interno sono presenti sottocartelle per ciascun gruppo di dispositivi.

### 2. Estrazione delle Features
- `Estrazione_features_current.py` – estrae corrente di riposo e pendenza
- `Estrazione_features_freq.py` – estrae guadagno e frequenza di taglio
- `LM386_Merged_Data.py` – unisce le feature estratte in file CSV

I file `LM386_Current_Slope_Francesca.csv` e `LM386_Gain_Freq_Francesca.csv` contengono dati parziali e sono stati usati come passaggi intermedi.

### 3. Scelta dello Scaler

La cartella `Scelta Scaler` contiene uno script e alcuni grafici utili per confrontare diversi metodi di normalizzazione delle features, al fine di identificare il più adatto.

- `LM386_scaler.py` – applica diversi metodi di scaling e genera pairplot per visualizzare la distribuzione dei dati trasformati.

I metodi confrontati sono:
- `MinMaxScaler` → `pairplot_scaler_MinMax.png`
- `StandardScaler` → `pairplot_scaler_Standard.png`
- `RobustScaler` → `pairplot_scaler_Robust.png`
- `Normalizer` → `pairplot_scaler_Normalizer.png`

**RobustScaler** è stato scelto come metodo di scaling principale per le fasi successive del progetto.

### 4. Visualizzazione e Classificazione
- `LM386_2features.py` – genera scatter plot a 2 a 2 tra le features
- `LM386.py` – valuta le 4 features contemporaneamente (da aggiornare)

Sono stati utilizzati i seguenti classificatori con **GridSearchCV** per l’ottimizzazione degli iperparametri:
- **SVM** con kernel lineare
- **SVM** con kernel RBF 
- **k-NN** (K-Nearest Neighbors)

La valutazione dei modelli è stata effettuata con **cross-validation** ripetuta su 5 fold, e le metriche analizzate includono accuratezza, precisione, recall, F1-score e matrice di confusione.  


## File di Output
I grafici e le metriche vengono salvati in:
  - `Output files/Dati Francesca`
  - `Output files/Dati Simone`
  - `Output files/Dati Tot`

Per ciascun set di dati (Francesca, Simone, Totale), sono presenti le seguenti directory:
- `Metrics/` – metriche e valutazioni
- `Plot/` – grafici e confronti visivi tra le features

---


