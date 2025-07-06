import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


def plot_dataset(x, y, feat0=0, feat1=1):
    colors = ['b.', 'r.', 'g.', 'k.', 'c.', 'm.']
    labels = ['Fake', 'Real']
    class_labels = np.unique(y).astype(int)
    for k in class_labels:
        plt.plot(x[y == k, feat0], x[y == k, feat1], colors[k % 6], label=labels[k])
    plt.legend()


def plot_decision_regions(x, y, classifier, resolution=0.1):
    # setup marker generator and color map
    colors = ('blue', 'red', 'lightgreen', 'black', 'cyan', 'magenta')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 0.02, x[:, 0].max() + 0.02
    x2_min, x2_max = x[:, 1].min() - 0.02, x[:, 1].max() + 0.02
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


# Leggi il file CSV
dataset = pd.read_csv("LM386_Features_4D.csv")

# Creo label binarie 0 e 1
#dataset["original"] = dataset["original"].astype(int)

# Features
features = ["quiescent_current", "voltage_gain", "cutoff_frequency", "current_slope"]

# Split  80/20 per ogni gruppo di dispositivi
train_list = []
test_list = []

for group_id, group_data in dataset.groupby("group"):
    train_group, test_group = train_test_split(
        group_data, test_size=0.2, shuffle=True, random_state=42, stratify=group_data["original"])
    train_list.append(train_group)
    test_list.append(test_group)

train = pd.concat(train_list)
test = pd.concat(test_list)

# Separo X e y
X_tr = train[features]
y_tr = train["original"]
X_ts = test[features]
y_ts = test["original"]

# Normalizzo
scaler = RobustScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_ts_scaled = scaler.transform(X_ts)

# Plot su dati normalizzati
df_scaled = pd.DataFrame(X_tr_scaled, columns=features)
df_scaled["original"] = y_tr.values

sns.pairplot(df_scaled, hue="original", diag_kind="hist", vars=features)
plt.savefig("pairplot_scaled.png", dpi=300, bbox_inches='tight')
plt.close()

# Cross-val interna (per GridSearchCV)
hyp_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Cross-val esterna (valutazione finale)
KF_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definizione classificatori con grid search
def get_clf_list():
    return [
        GridSearchCV(estimator=svm.SVC(kernel="linear"), param_grid={
            'C': [0.1, 1, 10, 100]}, cv=hyp_cv),
        GridSearchCV(estimator=svm.SVC(kernel="rbf"), param_grid={
            'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10]}, cv=hyp_cv),
        GridSearchCV(estimator=KNeighborsClassifier(), param_grid={
            'n_neighbors': [3, 5, 7]}, cv=hyp_cv),
    ]

clf_names = ['SVM - linear', 'SVM - RBF', 'kNN']

n_classifiers = 3

# Inizializza array per salvare l'accuratezza
acc = np.zeros((len(clf_names), KF_cv.get_n_splits()))

# True e Predicted labels
labels_y_true = [[] for _ in range(n_classifiers)]
labels_y_pred = [[] for _ in range(n_classifiers)]

# Salva migliori hyp e i result per ogni classificatore per ogni fold
best_params_all = [[] for _ in range(n_classifiers)] 
cv_results_all = [ [] for _ in range(n_classifiers) ]

# Report Accuracy, Precision, Recall, F1-score
results_list = []

# Classifier loop
for i, (tr_idx, ts_idx) in enumerate(KF_cv.split(X_tr_scaled, y_tr)):
    X_train_fold = X_tr_scaled[tr_idx]
    y_train_fold = y_tr.values[tr_idx]
    X_val_fold = X_tr_scaled[ts_idx]
    y_val_fold = y_tr.values[ts_idx]

    clf_list = get_clf_list()

    for k, clf in enumerate(clf_list):
        clf.fit(X_train_fold, y_train_fold)

        y_pred = clf.predict(X_val_fold)

        acc[k, i] = (y_pred == y_val_fold).mean()
        labels_y_true[k].extend(y_val_fold)
        labels_y_pred[k].extend(y_pred)

        best_params_all[k].append(clf.best_params_)
        cv_results_all[k].append(clf.cv_results_)

    
# Stampa hyp e accuracy
for k, (name, clf) in enumerate(zip(clf_names, clf_list)):
    print(f"\n{name}")

    # Unione dei risultati da tutti i fold in un DataFrame
    scores = defaultdict(list)
    params = None

    # mean_test_score = salva i risultati di ogni GridSearchCV, per ogni combinazione degli hyp
    for fold_result in cv_results_all[k]:
        if params is None:
            params = fold_result['params']
        for i, p in enumerate(fold_result['params']):
            key = tuple(sorted(p.items()))
            scores[key].append(fold_result['mean_test_score'][i])

    # Calcola media e std per ciascuna combinazione di mean_test_score 
    # media e dev standard su tutti i fold del GridSearchCV
    mean_std_summary = []
    for key, vals in scores.items():
        mean = np.mean(vals)
        std = np.std(vals)
        mean_std_summary.append((dict(key), mean, std))

    # Ordina per mean decrescente
    mean_std_summary.sort(key=lambda x: x[1], reverse=True)
    
    print("Aggregated mean_test_score over outer folds:")
    for param_dict, mean, std in mean_std_summary:
        print(f"    {param_dict} --> Mean: {mean:.3f}, Std: {std:.3f}") 

    best_global_params = mean_std_summary[0][0]
    print(f"Best params by mean_test_score aggregation: {best_global_params}")

    # Best hyp per ogni fold
    print("Best hyperparameters across all outer folds:")
    for i, params in enumerate(best_params_all[k]):
        print(f"    Fold {i+1}: {params}")

    # Best hyp più frequente tra i fold
    from collections import Counter
    counter = Counter(tuple(sorted(p.items())) for p in best_params_all[k])
    most_common = counter.most_common(1)[0]
    best_params = dict(most_common[0])
    print(f"Most frequent best hyperparams: {best_params} ({most_common[1]} folds)")

    print(f"Cross-validation accuracy = {acc[k].mean():.2%} +/- {acc[k].std():.2%}")

    print("Confusion matrix Cross-validation Data:")
    print(confusion_matrix(labels_y_true[k], labels_y_pred[k]))

    # Plot delle regioni decisionali (solo sulle prime due feature)
    clf.fit(X_tr_scaled[:, :2], y_tr)
    plot_decision_regions(X_tr_scaled[:, :2], y_tr, clf)
    plot_dataset(X_tr_scaled[:, :2], y_tr, feat0=0, feat1=1)
    plt.title(f"{name} - Decision Regions with Train Data")
    plt.savefig(f"decision_region_with_train_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Calcolo precision, recall, and F1-score per ogni classe 
    precision, recall, f1, support = precision_recall_fscore_support(labels_y_true[k], labels_y_pred[k], labels=[0, 1],
                                                                     zero_division=0)

    # Calcolo macro-average precision, recall, and F1-score
    # Macro average = media non ponderata per ogni classe
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_macro = f1.mean()

    # Calcolo weighted-average precision, recall, and F1-score
    # Weighted average = media ponderata (tiene conto del support)
    total_support = support.sum()
    precision_weighted = np.sum(precision * support) / total_support
    recall_weighted = np.sum(recall * support) / total_support
    f1_weighted = np.sum(f1 * support) / total_support

    
    results_list.append({
        "Classifier": name,
        "Best hyperparams": dict(best_params),
        "Accuracy Mean": acc[k].mean(),
        "Accuracy Std": acc[k].std(),
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

# Crea DataFrame finale
df_results = pd.DataFrame(results_list)

# Salva il DataFrame con le metriche calcolate in un file CSV
df_results.to_csv("metrics_summary.csv", index=False)

# Salva le stesse metriche in un file txt
with open("metrics_summary.txt", "w", encoding="utf-8") as f:
    f.write(df_results.to_string(index=False))







