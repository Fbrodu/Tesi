import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from io import StringIO

# === PERCORSO ALLA CARTELLA ===
folder_path = "/workspaces/Tesi/Estrazione Features/Current/MAMX"

# === LISTA FILE ===
file_list = [f for f in os.listdir(folder_path) if f.startswith("MAMX") and f.endswith(".txt")]

results = []

# === CICLO SU TUTTI I FILE ===
for filename in sorted(file_list):
    filepath = os.path.join(folder_path, filename)

    with open(filepath, 'r') as f:
        raw_data = f.read()

    # Divisione in blocchi basati su 'Vs\tIs'
    blocks = raw_data.strip().split("Vs\tIs")
    blocks = [b.strip() for b in blocks if b.strip()]

    quiescent_list = []
    slope_list = []

    for i, block in enumerate(blocks):
        try:
            df = pd.read_csv(StringIO("Vs\tIs\n" + block), sep="\t").dropna()
            df = df.sort_values(by="Vs")

            # Corrente a 6V
            row_6v = df[df["Vs"] == 6.0]
            if row_6v.empty:
                print(f"  {filename} - blocco {i+1}: manca Vs = 6.0, salto.")
                continue
            quiescent = row_6v["Is"].values[0] * 1000  # mA
            quiescent_list.append(quiescent)

            # Regressione lineare tra 6 e 12 V
            mask = (df['Vs'] >= 6) & (df['Vs'] <= 12)
            df_fit = df[mask]

            X = df_fit['Vs'].values.reshape(-1, 1)
            y = df_fit['Is'].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0] # pendenza
            slope_list.append(slope)

        except Exception as e:
            print(f" Errore nel file {filename}, blocco {i+1}: {e}")

    if quiescent_list and slope_list:
        results.append({
            "group": "MAMX",
            "ID": filename.replace(".txt", ""),
            "original": True,
            "quiescent_current": np.mean(quiescent_list),
            "current_slope": np.mean(slope_list)
        })
    else:
        print(f"  Nessun dato valido in {filename}")

# === SALVATAGGIO CSV ===
output_df = pd.DataFrame(results)
output_df = output_df.sort_values(by="ID")
output_path = "/workspaces/Tesi/Estrazione Features/LM386_Current_Slope_MAMX.csv"
output_df.to_csv(output_path, index=False)

print(f" File salvato: {output_path}")
