import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# === FUNZIONE CHE ANALIZZA I FILE ===
def analyze_file(filepath, group):
    excel_file = pd.ExcelFile(filepath)
    valid_sheets = []

    for sheet_name in excel_file.sheet_names:
        try:
            df = excel_file.parse(sheet_name, nrows=1)
            if list(df.columns) == ['Is', 'Vs']:
                valid_sheets.append(sheet_name)
        except Exception:
            continue

    quiescent_list = []
    slope_list = []

    for sheet in valid_sheets:
        df = excel_file.parse(sheet).dropna()
        df = df.sort_values(by='Vs')

        # Corrente di riposo 6 V
        row_6v = df[df['Vs'] == 6.0]
        quiescent = row_6v['Is'].values[0]
        quiescent_list.append(quiescent)

        # Pendenza tra 6 e 12 V con Regressione lineare ai minimi quadrati
        mask = (df['Vs'] >= 6) & (df['Vs'] <= 12)
        df_fit = df[mask]

        X = df_fit['Vs'].values.reshape(-1, 1)
        y = df_fit['Is'].values
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0] # pendenza
        slope_list.append(slope)

    # Calcolo medie
    avg_quiescent_current = np.mean(quiescent_list)
    avg_slope = np.mean(slope_list)

    filename = os.path.basename(filepath).replace(".xlsx", "")
    ID = filename
    original = True if group == 'RSAM' else False

    return {
        'group': group,
        'ID': ID,
        'original': original,
        'quiescent_current': avg_quiescent_current,
        'current_slope': avg_slope
    }

# === PERCORSI AI FILE ===
folders = {
    'RSAM': '/workspaces/Tesi/Estrazione Features/Current/RSAM',
    'UTC_CD': '/workspaces/Tesi/Estrazione Features/Current/UTC_CD'
}

# === ANALISI DEI FILE ===
results = []

for group, folder_path in folders.items():
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx"):
            filepath = os.path.join(folder_path, filename)
            try:
                result = analyze_file(filepath, group)
                results.append(result)
            except Exception as e:
                print(f"Errore nel file {filename} ({group}): {e}")


# === SALVATAGGIO ORDINATO PER GRUPPO E NUMERO ===
import re

def extract_number(id_str):
    match = re.search(r'(\d+)$', id_str)
    return int(match.group(1)) if match else -1

# Ordina prima per 'group' (RSAM < UTC_CD), poi per numero estratto da 'ID'
output_df = pd.DataFrame(results)
output_df['ID_num'] = output_df['ID'].apply(extract_number)
output_df = output_df.sort_values(by=['group', 'ID_num'])
output_df = output_df.drop(columns='ID_num')

output_path = '/workspaces/Tesi/Estrazione Features/LM386_Current_Slope_Francesca.csv'
output_df.to_csv(output_path, index=False)
print(f"File salvato: LM386_Current_Slope_Francesca.csv in {output_path}")
