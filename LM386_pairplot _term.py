import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import os

# === Caricamento dati ===
dataset = pd.read_csv("LM386_Features_5D_solo_term.csv")
features = ["quiescent_current", "voltage_gain", "cutoff_frequency", "current_slope", "T_low_gain", "T_high_gain"]

# === Split 80/20 (senza divisione per gruppo) ===
train, test = train_test_split(
    dataset,
    test_size=0.2,
    shuffle=True,
    random_state=42,
    stratify=dataset["original"]
)

X_tr = train[features]
y_tr = train["original"]
X_ts = test[features]
y_ts = test["original"]

# === Scaling con RobustScaler ===
scaler = RobustScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_ts_scaled = scaler.transform(X_ts)

# === DataFrame per il pairplot ===
df_plot = pd.DataFrame(X_tr_scaled, columns=features)
df_plot["original"] = y_tr.reset_index(drop=True)

# === Pairplot ===
sns.pairplot(
    df_plot,
    hue="original",
    diag_kind="hist",
    vars=features,
    #plot_kws={"edgecolor": "black"}
)

# === Plot directory ===
plot_dir = "/workspaces/Tesi/Output files/Pairplot"
os.makedirs(plot_dir, exist_ok=True)

# === Salvataggio ===
plt.savefig(os.path.join(plot_dir, "pairplot_temp.png"), dpi=300, bbox_inches="tight")
plt.close()

