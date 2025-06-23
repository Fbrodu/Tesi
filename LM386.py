import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

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
# Rimuovi colonne inutili o non ancora disponibili
X = dataset.drop(columns=["T150_aging"])
# Creo label binarie 0 e 1
y = dataset["original"].astype(int)
# Features
features = ["quiescent_current", "voltage_gain", "cutoff_frequency", "current_slope"]

# Creo una matrice di grafici a dispersione (scatter plots) e istogrammi
sns.pairplot(
    dataset,
    hue="original",
    vars=features,
    diag_kind="hist"  # mostra istogrammi invece di KDE sulla diagonale
)
plt.show()

# Normalizzo
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(dataset[features])

# Creo un nuovo DataFrame con i dati normalizzati
df_scaled = pd.DataFrame(scaled_values, columns=features)

# Aggiungi la colonna "original"
df_scaled["original"] = dataset["original"].values

# Plot su dati normalizzati
sns.pairplot(df_scaled, hue="original", diag_kind="hist", vars=features)
plt.show()

# Cross-validation
splitter = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

clf_list = [
    GridSearchCV(estimator=svm.SVC(kernel="linear"), param_grid={
        'C': [0.1, 1, 10, 100]}, cv=3),
    GridSearchCV(estimator=svm.SVC(kernel="rbf"), param_grid={
        'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10]}, cv=3),
    GridSearchCV(estimator=KNeighborsClassifier(), param_grid={
        'n_neighbors': [3, 5, 7]}, cv=3),

]

clf_names = ['SVM - linear', 'SVM - RBF', 'kNN']

# Inizializza array con zeri per store sccuracy
acc = np.zeros((len(clf_list), splitter.get_n_splits()))

# Inizializza lista per conservare true e predicted labels
labels_y_true = [[] for _ in clf_list]
labels_y_pred = [[] for _ in clf_list]

# Store migliori iperparametri per ogni classificatore
best_params_first_fold = [None] * len(clf_list)

for i, (train_idx, test_idx) in enumerate(splitter.split(dataset)):
    train = dataset.iloc[train_idx] # Indici righe train
    test = dataset.iloc[test_idx] # Indici righe test

    X_tr = train[["quiescent_current", "voltage_gain"]]
    y_tr = train["original"]
    X_ts = test[["quiescent_current", "voltage_gain"]]
    y_ts = test["original"]

    scaler = MinMaxScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_ts_scaled = scaler.transform(X_ts)

    # Classifier loop
    for k, clf in enumerate(clf_list):
        clf.fit(X_tr_scaled, y_tr)

        y_pred = clf.predict(X_ts_scaled)

        # Store accuratezza per ogni classificatore
        acc[k, i] = (y_pred == y_ts).mean()

        # Store tutte le true and predicted labels per ogni classificatore
        labels_y_true[k].extend(y_ts)
        labels_y_pred[k].extend(y_pred)

        # Store migliori parametri per ogni classificatore
        best_params = tuple(sorted(clf.best_params_.items()))
        if i == 0:
            best_params_first_fold[k] = clf.cv_results_

# Stampa iperparametri
for k, name in enumerate(clf_names):
    print(f"\n {name} ")

    cv_results = best_params_first_fold[k]  # cv_results_ from the first outer fold for classifier k
    best_index = np.argmax(cv_results['mean_test_score'])  # Index of the best hyperparameter setting
    best_params = cv_results['params'][best_index]  # Corresponding best hyperparameters
    print(f"Best hyperparams: {dict(best_params)}")
    print("    - Grid scores on development set:")
    means = cv_results['mean_test_score']
    stds = cv_results['std_test_score']
    for mean, std, params in zip(means, stds, cv_results['params']):
        print(f"        {mean:.3f} (+/-{std * 2:.03f}) for {params}")

    print(f"Mean test accuracy = {acc[k].mean():.2%} +/- {acc[k].std():.2%}")

    print("Classification report:")
    print(classification_report(labels_y_true[k], labels_y_pred[k], target_names=["Fake", "Real"]))
    print("Confusion matrix:")
    print(confusion_matrix(labels_y_true[k], labels_y_pred[k]))



