import os
import pandas as pd
import numpy as np
import re

# === PERCORSI AI FILE ===
folders = {
    'RSAM': '/workspaces/Tesi/Estrazione Features/Frequenza/RSAM',
    'UTC_CD': '/workspaces/Tesi/Estrazione Features/Frequenza/UTC_CD'
}

# === FUNZIONE CHE ANALIZZA UNA CARTELLA DI UN DISPOSITIVO ===
def analyze_file(folder_path, group):
    gains = []
    cutoffs = []

    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path, sep='\t')

            df.columns = [col.strip() for col in df.columns]  # rimuove spazi

            freq = df['Frequency (Hz)'].values
            gain = df['Channel 2 Magnitude (dB)'].values

            # === Calcolo del guadagno a 1 kHz ===
            row_1khz = df[df['Frequency (Hz)'] == 1000]
            Ao = row_1khz['Channel 2 Magnitude (dB)'].values[0]
            gains.append(Ao)

            # === Calcolo della frequenza di taglio ===
            # Definizione del valore target del guadagno per la frequenza di taglio
            target_gain = Ao - 3
            # Calcola la differenza tra ciascun valore di guadagno e il valore target (Ao - 3)
            # Serve per identificare dove il guadagno "scende sotto" il valore target
            diffs = gain - target_gain
            # Individua gli indici in cui cambia il segno della differenza: da positivo a negativo o viceversa
            # Trova i due punti che stanno sopra e sotto il valore target (Ao - 3)
            sign_change = np.where(np.diff(np.sign(diffs)))[0]

            # Cerca se esiste almeno un punto di attraversamento
            if len(sign_change) > 0:

                # Primo intervallo in cui avviene l'attraversamento
                i = sign_change[0]

                # Ricava le due frequenze (x0, x1) e i due guadagni (y0, y1) 
                # Questi due punti delimitano l'intervallo in cui il guadagno passa da sopra a sotto Ao - 3
                x0, x1 = freq[i], freq[i+1]
                y0, y1 = gain[i], gain[i+1]

                # Calcola la pendenza della retta che unisce i due punti
                slope = (y1 - y0) / (x1 - x0)

                # Retta per trovare la frequenza esatta in cui il guadagno è pari a Ao - 3
                cutoff_freq = x0 + (target_gain - y0) / slope
                cutoffs.append(cutoff_freq)

    if gains and cutoffs:
        avg_gain = np.mean(gains)
        avg_cutoff = np.mean(cutoffs)

        # Estrazione ID e flag original
        ID = os.path.basename(folder_path)
        original = True if group == 'RSAM' else False

        return {
            'group': group,
            'ID': ID,
            'original': original,
            'voltage_gain': avg_gain,
            'cutoff_frequency': avg_cutoff
        }
    else:
        return None

# === ANALISI DEI FILE ===
results = []

for group, folder_path in folders.items():
    for device_id in sorted(os.listdir(folder_path)):
        full_path = os.path.join(folder_path, device_id)
        if os.path.isdir(full_path):
            try:
                result = analyze_file(full_path, group)
                if result is not None:
                    results.append(result)
                else:
                    print(f"[!] Dati incompleti per {device_id}")
            except Exception as e:
                print(f"[!] Errore nel file {device_id} ({group}): {e}")

# === SALVATAGGIO ORDINATO PER GRUPPO E NUMERO ===
def extract_number(id_str):
    match = re.search(r'(\d+)$', id_str)
    return int(match.group(1)) if match else -1

# Ordina prima per 'group' (RSAM < UTC_CD), poi per numero estratto da 'ID'
output_df = pd.DataFrame(results)
output_df['ID_num'] = output_df['ID'].apply(extract_number)
output_df = output_df.sort_values(by=['group', 'ID_num'])
output_df = output_df.drop(columns='ID_num')

output_path = '/workspaces/Tesi/Estrazione Features/LM386_Gain_Freq_Francesca.csv'
output_df.to_csv(output_path, index=False)
print(f"File salvato: LM386_Gain_Freq_Francesca.csv in {output_path}")
