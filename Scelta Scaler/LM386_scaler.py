import os
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, Normalizer
from sklearn.model_selection import train_test_split

# === Caricamento del dataset e definizione della variabile features ===
dataset = pd.read_csv("LM386_Features_4D.csv")
features = ["quiescent_current", "voltage_gain", "cutoff_frequency", "current_slope"]

# === Split 80/20 diviso per ogni gruppo ===
train_list, test_list = [], []
for group_id, group_data in dataset.groupby("group"):
    train_group, test_group = train_test_split(
        group_data, test_size=0.2, shuffle=True, random_state=42,
        stratify=group_data["original"])
    train_list.append(train_group)
    test_list.append(test_group)

train = pd.concat(train_list)
test = pd.concat(test_list)

X_tr = train[features]
y_tr = train["original"]
X_ts = test[features]
y_ts = test["original"]


# === Directory per salvare i plot ===
output_dir = "/workspaces/Tesi/Scelta Scaler"
os.makedirs(output_dir, exist_ok=True)

# === Lista degli scaler da confrontare ===
scalers = {
    "Robust": RobustScaler(),
    "Standard": StandardScaler(),
    "MinMax": MinMaxScaler(),
    "Normalizer": Normalizer()
}

# === Loop sugli scaler ===
for scaler_name, scaler in scalers.items():
    # Applico lo scaling
    X_scaled = scaler.fit_transform(X_tr)
    
    # Creo un DataFrame per il pairplot
    df_scaled = pd.DataFrame(X_scaled, columns=features)
    df_scaled["original"] = y_tr.values

    # Generazione del pairplot
    g = sns.pairplot(df_scaled, hue="original", diag_kind="hist", vars=features)
    g.figure.suptitle(f"Scaler: {scaler_name}", y=1.02, fontsize=14)  
    
    # Salvataggio
    filename = f"pairplot_scaler_{scaler_name}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

print("Pairplot per tutti gli scaler salvati correttamente.")