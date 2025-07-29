"""
        Pairplot
"""
import os
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# === Creazione delle cartelle di output per salvare grafici  ===
plot_dir = "/workspaces/Tesi/Output files/Dati Francesca/Plot"
os.makedirs(plot_dir, exist_ok=True)

# === Caricamento del dataset e definizione della variabile features ===
dataset_F = pd.read_csv("LM386_Features_4D_Francesca.csv")
dataset_S = pd.read_csv("LM386_Features_4D_Simone.csv")
dataset_FG = pd.read_csv("LM386_Features_4D_FG.csv")
features = ["quiescent_current", "voltage_gain", "cutoff_frequency", "current_slope"]

# === Split 80/20 diviso per ogni gruppo ===
train_list, test_list = [], []

def split(dataset):
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

# === Scaling con RobustScaler ===
scaler = RobustScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_ts_scaled = scaler.transform(X_ts)

# === Visualizzazione pairplot delle feature nel training set ===
df_scaled = pd.DataFrame(X_tr_scaled, columns=features)
df_scaled["original"] = y_tr.values
sns.pairplot(df_scaled, hue="original", diag_kind="hist", vars=features)
plt.savefig(os.path.join(plot_dir, "pairplot_scaled_Francesca.png"), dpi=300, bbox_inches='tight')
plt.close()
