"""
    Script che unisce i due file .csv
"""

import pandas as pd

# === PERCORSI FILE===
gain_freq_path = "/workspaces/Tesi/Estrazione Features/LM386_Gain_Freq_Francesca.csv"
current_slope_path = "/workspaces/Tesi/Estrazione Features/LM386_Current_Slope_Francesca.csv"
francesca_path = "/workspaces/Tesi/LM386_Features_4D_Francesca.csv"
simone_path = "/workspaces/Tesi/LM386_Features_4D_Simone.csv"
MAMX_f = "/workspaces/Tesi/Estrazione Features/LM386_Gain_Freq_MAMX.csv"
MAMX_c = "/workspaces/Tesi/Estrazione Features/LM386_Current_Slope_MAMX.csv"
MAMX = "/workspaces/Tesi/Estrazione Features/LM386_Features_4D_MAMX.csv"
FG_path = "/workspaces/Tesi/Estrazione Features/LM386_Features_4D_FG.csv"
FS_path = "/workspaces/Tesi/LM386_Features_4D.csv"


# === CARICAMENTO ===
#gain_freq_df = pd.read_csv(gain_freq_path)
#current_slope_df = pd.read_csv(current_slope_path)
#francesca_df = pd.read_csv(francesca_path)
#simone_df = pd.read_csv(simone_path)
#mamx_c_df = pd.read_csv(MAMX_c)
#mamx_f_df = pd.read_csv(MAMX_f)
FG_df = pd.read_csv(FG_path)
#MAMX_df = pd.read_csv(MAMX)
FS_df = pd.read_csv(FS_path)

# === UNIONE DEI DATAFRAME SUI CAMPI COMUNI ===
#merged_df = pd.merge(mamx_f_df, mamx_c_df, on=["group", "ID", "original"], how="inner")
merged_df = pd.concat([FG_df, FS_df], ignore_index=True)

# === RIORIDINO DELLE COLONNE  ===
# Da usare solo per unire Current e Freq
#merged_df = merged_df[[
#    "group", "ID", "original",
#    "quiescent_current", "voltage_gain", "cutoff_frequency", "current_slope"
#]]


# === SALVATAGGIO DEL FILE ===
merged_df.to_csv("/workspaces/Tesi/LM386_Features_4D_all.csv", index=False)

print("File salvato come 'LM386_Features_4D_all.csv'")
