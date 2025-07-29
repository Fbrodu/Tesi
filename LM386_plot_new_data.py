import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import os

from sklearn.decomposition import PCA

# === Caricamento dati ===
dataset_F = pd.read_csv("LM386_Features_4D_Francesca.csv")
dataset_S = pd.read_csv("LM386_Features_4D_Simone.csv")
dataset_FG = pd.read_csv("LM386_Features_4D_FG.csv")
features = ["quiescent_current", "voltage_gain", "cutoff_frequency", "current_slope"]

# === Split 80/20 ===
def split_dataset(dataset):
    train_list, test_list = [], []
    for group_id, group_data in dataset.groupby("group"):
        train_group, test_group = train_test_split(
            group_data,
            test_size=0.2,
            shuffle=True,  
            random_state=42,
            stratify=group_data["original"]
        )
        train_list.append(train_group)
        test_list.append(test_group)
    return pd.concat(train_list), pd.concat(test_list)

# === Split 80/20 per ciascun dataset ===
train_S, test_S = split_dataset(dataset_S)
train_F, test_F = split_dataset(dataset_F)
train_FG, test_FG = split_dataset(dataset_FG)

scaler = RobustScaler()


# === Dizionario dei dataset e inizializzazione ===
datasets = {
    "Simone": (train_S, test_S),
    "Francesca": (train_F, test_F),
    "Francesca+Greta": (train_FG, test_FG)
}

# === Contenitori dei DataFrame scalati ===
df_tr_list = []
df_ts_list = []

# === Scaling con fit su Simone ===
scaler = RobustScaler()
pca = PCA(n_components=2, random_state=42)

for i, (label, (train, test)) in enumerate(datasets.items()):
    X_tr = train[features]
    y_tr = train["original"]
    X_ts = test[features]
    y_ts = test["original"]

    if i == 0:
        # Primo dataset → fit + transform
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_tr_pca = pca.fit_transform(X_tr_scaled)
    else:
        # Altri dataset → solo transform
        X_tr_scaled = scaler.transform(X_tr)
        X_tr_pca = pca.transform(X_tr_scaled)
    X_ts_scaled = scaler.transform(X_ts)
    X_ts_pca = pca.transform(X_ts_scaled)

    # DataFrame train
    df_tr = pd.DataFrame(X_tr_pca, columns=["PCA1", "PCA2"])
    df_tr["original"] = y_tr.values
    df_tr["source"] = label
    df_tr["set"] = "train"
    df_tr_list.append(df_tr)

    # DataFrame test
    df_ts = pd.DataFrame(X_ts_pca, columns=["PCA1", "PCA2"])
    df_ts["original"] = y_ts.values
    df_ts["source"] = label
    df_ts["set"] = "test"
    df_ts_list.append(df_ts)


# === Concatenazione dataframe ===
df_all_train = pd.concat(df_tr_list)
df_all_train["legend"] = df_all_train["source"] + " - " + df_all_train["original"].map({True: "True", False: "False"})

# === Plot directory ===
plot_dir = "/workspaces/Tesi/Output files/Pairplot"
os.makedirs(plot_dir, exist_ok=True)

palette = {
    "Simone - True": "#ff7f0e",             # arancione pieno
    "Simone - False": "#1f77b4",            # blu pieno
    "Francesca - True": "#ff7f0e",          # arancione medio
    "Francesca - False": "#1f77b4",         # blu medio
    "Francesca+Greta - True": "#ff7f0e",    # arancione chiaro
}

markers = {
    "Francesca - True": "s",         # quadrato
    "Francesca - False": "s",        # quadrato
    "Simone - True": "o",            # cerchio
    "Simone - False": "o",           # cerchio
    "Francesca+Greta - True": "^",   # triangolo
}


# === Pairplot 
sns.pairplot(
    df_all_train,
    hue="legend",
    palette= palette,
    diag_kind="hist",
    vars= ["PCA1", "PCA2"],
    markers= markers,
    #plot_kws={"edgecolor": "black"}

)

# === Salvataggio ===
plt.savefig(os.path.join(plot_dir, "pairplot_colored_pca_true.png"), dpi=300, bbox_inches="tight")
plt.close()
