"""
    Script che unisce i due file .csv
"""

import pandas as pd

# === PERCORSI FILE===
gain_freq_path = "/workspaces/Tesi/Estrazione Features/LM386_Gain_Freq_Francesca.csv"
current_slope_path = "/workspaces/Tesi/Estrazione Features/LM386_Current_Slope_Francesca.csv"
francesca_path = "/workspaces/Tesi/LM386_Features_4D_Francesca.csv"
simone_path = "/workspaces/Tesi/LM386_Features_4D_Simone.csv"


# === CARICAMENTO ===
#gain_freq_df = pd.read_csv(gain_freq_path)
#current_slope_df = pd.read_csv(current_slope_path)
francesca_df = pd.read_csv(francesca_path)
simone_df = pd.read_csv(simone_path)

# === UNIONE DEI DATAFRAME SUI CAMPI COMUNI ===
#merged_df = pd.merge(gain_freq_df, current_slope_df, on=["group", "ID", "original"], how="inner")
merged_df = pd.concat([francesca_df, simone_df], ignore_index=True)

# === RIORIDINO DELLE COLONNE  ===
# Da usare solo per unire Current e Freq
#merged_df = merged_df[[
#    "group", "ID", "original",
#    "quiescent_current", "voltage_gain", "cutoff_frequency", "current_slope"
#]]


# === SALVATAGGIO DEL FILE ===
merged_df.to_csv("/workspaces/Tesi/LM386_Features_4D.csv", index=False)

print("File salvato come 'LM386_Features_4D.csv'")
