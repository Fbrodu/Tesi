import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

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

# Creo una matrice di grafici a dispersione (scatter plots) e istogrammi
sns.pairplot(
    dataset,
    hue="original",
    vars=features,
    diag_kind="hist"  # mostra istogrammi invece di KDE sulla diagonale
)
plt.savefig("pairplot_raw.png", dpi=300, bbox_inches='tight')
plt.close()

# Split stratificato 80/20 per ogni gruppo di dispositivi
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

# Definizione classificatori con grid search
clf_list = [
    GridSearchCV(estimator=svm.SVC(kernel="linear"), param_grid={
        'C': [0.1, 1, 10, 100]}, cv=3),
    GridSearchCV(estimator=svm.SVC(kernel="rbf"), param_grid={
        'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10]}, cv=3),
    GridSearchCV(estimator=KNeighborsClassifier(), param_grid={
        'n_neighbors': [3, 5, 7]}, cv=3),
]

clf_names = ['SVM - linear', 'SVM - RBF', 'kNN']
# Inizializza array per salvare l'accuratezza
acc = np.zeros(len(clf_list))
# Inizializza lista per conservare true e predicted labels
labels_y_true = [[] for _ in clf_list]
labels_y_pred = [[] for _ in clf_list]

# Store migliori iperparametri per ogni classificatore
best_params_all = [None] * len(clf_list)

# Classifier loop
for k, clf in enumerate(clf_list):
    clf.fit(X_tr_scaled, y_tr)

    y_pred = clf.predict(X_ts_scaled)

    labels_y_true[k].extend(y_ts)
    labels_y_pred[k].extend(y_pred)

    best_params_all[k] = clf.cv_results_
    

# Stampa iperparametri
for k, (name, clf) in enumerate(zip(clf_names, clf_list)):
    print(f"\n{name}")

    cv_results = best_params_all[k]
    best_index = np.argmax(cv_results['mean_test_score'])
    best_params = cv_results['params'][best_index]
    print(f"Best hyperparams: {dict(best_params)}")
    print("    - Grid scores on development set:")
    means = cv_results['mean_test_score']
    stds = cv_results['std_test_score']
    for mean, std, params in zip(means, stds, cv_results['params']):
        print(f"        {mean:.3f} (+/-{std * 2:.03f}) for {params}")

    acc[k] = np.mean(np.array(labels_y_true[k]) == np.array(labels_y_pred[k]))
    print(f"Test accuracy = {acc[k]:.2%}")

    print("Classification report:")
    print(classification_report(labels_y_true[k], labels_y_pred[k], target_names=["Fake", "Real"]))
    print("Confusion matrix:")
    print(confusion_matrix(labels_y_true[k], labels_y_pred[k]))

    # Plot delle regioni decisionali (solo sulle prime due feature)
    clf.fit(X_tr_scaled[:, :2], y_tr)
    plot_decision_regions(X_tr_scaled[:, :2], y_tr, clf)
    plot_dataset(X_ts_scaled[:, :2], y_ts, feat0=0, feat1=1)
    plt.title(f"{name} - Decision Regions with Test Data")
    plt.savefig(f"decision_region_with_test_{name.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

