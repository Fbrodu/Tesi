"""
Confronto tra classificatori per la rilevazione di dispositivi elettronici contraffatti.

Obiettivo:
---------
Classificare dispositivi LM386 come "originali" o "contraffatti" in base a misure elettriche.
Il confronto viene effettuato utilizzando i seguenti classificatori:
- SVM con kernel lineare
- SVM con kernel RBF 
- K-Nearest Neighbors (k-NN)

Features utilizzate:
- quiescent_current (corrente a riposo a 6V)
- voltage_gain (guadagno per V = 6 V e f = 1 kHz)
- cutoff_frequency (frequenza di taglio a -3 dB)
- current_slope (pendenza curva tensione/corrente tra 6 V e 12 V, calcolata applicando 
                    una regressione lineare ai minimi quadrati)

Procedimento:
----------
- Caricamento dataset: `LM386_Features_4D.csv`
- Suddivisione 80/20 per ogni gruppo di dispositivi
- Scaling: RobustScaler
- Cross-validation:
    - Iperparametri: 3-fold CV (interna)
    - Valutazione: 5-fold CV (esterna)
- Confronto per tutte le coppie di feature (combinazioni 2 a 2)
- Plot delle decision regions e salvataggio
- Calcolo e salvataggio delle metriche (accuracy, precision, recall, f1, best hyp per ogni classificatore)

Output:
-------
- Grafici: nella cartella `/Output files/Plot`
- Metriche: in `/Output files/Metrics/metrics_summary_validation_2f.csv` e `.txt`
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict, Counter
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# === Creazione delle cartelle di output per salvare grafici e metriche ===
plot_dir = "/workspaces/Tesi/Output files/Plot"
metrics_dir = "/workspaces/Tesi/Output files/Metrics"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# === Funzione per visualizzare il dataset su una coppia di feature ===
def plot_dataset(x, y, feat0=0, feat1=1):
    colors = ['b.', 'r.', 'g.', 'k.', 'c.', 'm.']
    labels = ['Fake', 'True']
    class_labels = np.unique(y).astype(int)
    for k in class_labels:
        plt.plot(x[y == k, feat0], x[y == k, feat1], colors[k % 6], label=labels[k])
    plt.legend()

# === Funzione per plottare le decision region dei classificatori ===
def plot_decision_regions(x, y, classifier, resolution=0.01):
    colors = ('blue', 'red', 'lightgreen', 'black', 'cyan', 'magenta')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    x1_margin = (x1_max - x1_min) * 0.1
    x2_margin = (x2_max - x2_min) * 0.1
    x1_min -= x1_margin
    x1_max += x1_margin
    x2_min -= x2_margin
    x2_max += x2_margin

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

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

# === Scaling con RobustScaler ===
scaler = RobustScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_ts_scaled = scaler.transform(X_ts)

# === Visualizzazione pairplot delle feature nel training set ===
df_scaled = pd.DataFrame(X_tr_scaled, columns=features)
df_scaled["original"] = y_tr.values
sns.pairplot(df_scaled, hue="original", diag_kind="hist", vars=features)
plt.savefig(os.path.join(plot_dir, "pairplot_scaled.png"), dpi=300, bbox_inches='tight')
plt.close()

# === Definizione cross-validation ===
hyp_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # per tuning iperparametri
KF_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)   # per valutazione

# === Tuning iperparametri ===
def get_clf_list():
    return [
        GridSearchCV(estimator=svm.SVC(kernel="linear"), param_grid={
            'C': [0.1, 1, 10, 100]}, cv=hyp_cv),

        GridSearchCV(estimator=svm.SVC(kernel="rbf"), param_grid={
            'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10]}, cv=hyp_cv),

        GridSearchCV(estimator=KNeighborsClassifier(), param_grid={
            'n_neighbors': [3, 5, 7, 9, 11]}, cv=hyp_cv),
    ]

clf_names = ['SVM - linear', 'SVM - RBF', 'kNN']
feature_pairs = list(combinations(features, 2))  # Tutte le coppie di feature
results_list = []  # Lista finale dei risultati da salvare

# === Loop su tutte le coppie di feature ===
for feat0, feat1 in feature_pairs:
    print(f"\n--- Evaluating features pair: ({feat0}, {feat1}) ---")

    # Estrazione e scaling per la coppia corrente di feature
    X_tr_pair = train[[feat0, feat1]]
    X_tr_scaled_pair = scaler.fit_transform(X_tr_pair)

    acc_pair = np.zeros((len(clf_names), KF_cv.get_n_splits()))
    labels_y_true_pair = [[] for _ in clf_names]
    labels_y_pred_pair = [[] for _ in clf_names]
    best_params_all_pair = [[] for _ in clf_names]
    cv_results_all_pair = [[] for _ in clf_names]

    # === Cross-validation ===
    for i, (tr_idx, ts_idx) in enumerate(KF_cv.split(X_tr_scaled_pair, y_tr)):
        X_train_fold = X_tr_scaled_pair[tr_idx]
        y_train_fold = y_tr.values[tr_idx]
        X_val_fold = X_tr_scaled_pair[ts_idx]
        y_val_fold = y_tr.values[ts_idx]

        clf_list = get_clf_list()

        for k, clf in enumerate(clf_list):
            clf.fit(X_train_fold, y_train_fold)
            y_pred = clf.predict(X_val_fold)

            acc_pair[k, i] = (y_pred == y_val_fold).mean()
            labels_y_true_pair[k].extend(y_val_fold)
            labels_y_pred_pair[k].extend(y_pred)

            best_params_all_pair[k].append(clf.best_params_)
            cv_results_all_pair[k].append(clf.cv_results_)

    # === Analisi dei risultati per ciascun classificatore ===
    for k, name in enumerate(clf_names):
        print(f"\n{name}")

        # Calcolo dei mean_test_score da tutti i fold
        scores = defaultdict(list)
        for fold_result in cv_results_all_pair[k]:
            for i_param, p in enumerate(fold_result['params']):
                key = tuple(sorted(p.items()))
                scores[key].append(fold_result['mean_test_score'][i_param])

        # Calcolo media e deviazione standard per ciascun classificatore
        mean_std_summary = [
            (dict(key), np.mean(vals), np.std(vals)) for key, vals in scores.items()
        ]
        mean_std_summary.sort(key=lambda x: x[1], reverse=True)

        # === Calcolo dei punteggi mean_test_score ottenuti da GridSearchCV su ciascuno dei 5 fold ===
        # Per ogni combinazione di iperparametri, salvo i mean_test_score ottenuti nei vari fold.
        # Li aggrego per calcolare la media e la deviazione standard su tutti i fold esterni.
        print("Aggregated mean_test_score over outer folds:")
        for param_dict, mean, std in mean_std_summary:
            print(f"    {param_dict} --> Mean: {mean:.3f}, Std: {std:.3f}")


        # === Selezione degli iperparametri con la migliore accuracy media ===
        best_global_params = mean_std_summary[0][0]
        print(f"Best params by mean_test_score aggregation: {best_global_params}")

        # === Visualizzazione degli iperparametri migliori all'interno di ciascun fold ===
        # Per ognuno dei 5 fold, GridSearchCV identifica gli iperparametri migliori,
        # in base al mean_test_score ottenuto nella cross-validation a 3 fold di GridSearchCV.
        # Stampa gli iperparametri selezionati per ciascun fold.
        print("Best hyperparameters across all outer folds:")
        for i_fold, params_fold in enumerate(best_params_all_pair[k]):
            print(f"    Fold {i_fold+1}: {params_fold}")

        # === Identificazione della configurazione di iperparametri più frequente tra i 5 fold ===
        # Calcola quale set di iperparametri è stato selezionato più frequentemente 
        # come "migliore" da GridSearchCV nei singoli fold.

        # Converto ciascun set di iperparametri in una tupla ordinata (è necessario per usare Counter)
        counter = Counter(tuple(sorted(p.items())) for p in best_params_all_pair[k])
        # Seleziono la configurazione più comune tra i 5 fold
        most_common = counter.most_common(1)[0]
        # Riconverto la tupla per stamparlo in forma leggibile
        best_params = dict(most_common[0])
        print(f"Most frequent best hyperparams: {best_params} ({most_common[1]} folds)")


        # === Stampa dell'accuracy media e della sua deviazione standard ===
        # Viene calcolata come media delle accuracy ottenute nei 5 fold
        print(f"Cross-validation accuracy = {acc_pair[k].mean():.2%} +/- {acc_pair[k].std():.2%}")

        # === Stampa della confusion matrix ===
        print("Confusion matrix Cross-validation Data:")
        print(confusion_matrix(labels_y_true_pair[k], labels_y_pred_pair[k]))

        # === Plot delle decision regions ===
        clf = clf_list[k].best_estimator_
        clf.fit(X_tr_scaled_pair, y_tr.values)
        plot_decision_regions(X_tr_scaled_pair, y_tr.values, clf)
        plot_dataset(X_tr_scaled_pair, y_tr.values)
        plt.title(f"{name} - Decision Regions with features ({feat0}, {feat1})")
        filename = f"decision_region_{name.replace(' ', '_')}_{feat0}_{feat1}.png"
        plt.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

        # === Calcolo delle metriche ===
        precision, recall, f1, support = precision_recall_fscore_support(
            labels_y_true_pair[k], labels_y_pred_pair[k], labels=[0, 1], zero_division=0)

        # === Calcolo delle metriche macro e weighted ===
        # Le metriche macro danno pari peso a ciascuna classe, indipendentemente dalla sua frequenza
        # Le metriche weighted invece sono calcolate in base al numero di campioni (support)
        precision_macro = precision.mean()
        recall_macro = recall.mean()
        f1_macro = f1.mean()
        total_support = support.sum()
        precision_weighted = np.sum(precision * support) / total_support
        recall_weighted = np.sum(recall * support) / total_support
        f1_weighted = np.sum(f1 * support) / total_support

        results_list.append({
            "Features": f"{feat0}, {feat1}",
            "Classifier": name,
            "Best hyperparams": best_params,
            "Accuracy Mean": acc_pair[k].mean(),
            "Accuracy Std": acc_pair[k].std(),
            "Precision Fake": precision[0],
            "Recall Fake": recall[0],
            "F1 Fake": f1[0],
            "Support Fake": support[0],
            "Precision True": precision[1],
            "Recall True": recall[1],
            "F1 True": f1[1],
            "Support True": support[1],
            "Precision Macro": precision_macro,
            "Recall Macro": recall_macro,
            "F1 Macro": f1_macro,
            "Precision Weighted": precision_weighted,
            "Recall Weighted": recall_weighted,
            "F1 Weighted": f1_weighted
        })

# === Salvataggio metriche ===
df_results = pd.DataFrame(results_list)
df_results.to_csv(os.path.join(metrics_dir, "metrics_summary_validation_2f.csv"), index=False)
with open(os.path.join(metrics_dir, "metrics_summary_validation_2f.txt"), "w", encoding="utf-8") as f:
    f.write(df_results.to_string(index=False))
