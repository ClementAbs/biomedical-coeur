"""TP Machine Learning - Cardiomyopathie dilatée sur échographies (corrigé)"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     cross_val_predict, StratifiedKFold, GridSearchCV)
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, precision_recall_curve)
from echo_gen import make_dataset

# ---- Partie 1 : exploration
X, y = make_dataset(n=600, prevalence=0.3, seed=0)   # 1 = pathologique, 0 = normal
print("Images :", X.shape, "| valeurs entre",
      round(float(X.min()), 2), "et", round(float(X.max()), 2))
print("Répartition :", {0: int((y == 0).sum()), 1: int((y == 1).sum())},
      "(0 = normal, 1 = pathologique)")
fig, ax = plt.subplots(2, 5, figsize=(10, 4))
for i in range(4):
    ax[0, i].imshow(X[y == 0][i], cmap="gray"); ax[0, i].set_title("normal")
    ax[1, i].imshow(X[y == 1][i], cmap="gray"); ax[1, i].set_title("pathologique")
ax[0, 4].imshow(X[y == 0].mean(0), cmap="gray"); ax[0, 4].set_title("moyenne normal")
ax[1, 4].imshow(X[y == 1].mean(0), cmap="gray"); ax[1, 4].set_title("moyenne patho.")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig("fig_exploration.png", dpi=110); plt.close()

# ---- Partie 2 : découpage stratifié
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train/Test :", X_tr.shape, X_te.shape,
      "| % patho train =", round(100*y_tr.mean(), 1))

# ---- Partie 3 : extraction de caractéristiques et comparaison de modèles
def extraire_hog(images):
    return np.stack([hog(im, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
                     for im in images])

def pipe(modele, scaler=True):
    etapes = [("hog", FunctionTransformer(extraire_hog))]
    if scaler: etapes.append(("sc", StandardScaler()))
    return Pipeline(etapes + [("m", modele)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Logistique": pipe(LogisticRegression(max_iter=2000)),
    "SVM": pipe(CalibratedClassifierCV(SVC(), ensemble=False)),
    "Random Forest": pipe(RandomForestClassifier(200, random_state=42), scaler=False),
}
for name, m in models.items():
    s = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="f1")
    print(f"{name:14s} F1 CV = {s.mean():.3f} (+/- {s.std():.3f})")

# ---- Partie 4 : optimisation du SVM
grille = {"m__estimator__C": [0.1, 1, 10, 100],
          "m__estimator__gamma": ["scale", 0.0005, 0.005]}
grid = GridSearchCV(models["SVM"], grille, cv=cv, scoring="f1")
grid.fit(X_tr, y_tr)
print("Meilleurs réglages :", grid.best_params_, "| F1 CV =", round(grid.best_score_, 3))

# ---- Partie 5 : évaluation finale sur le test
best = grid.best_estimator_
pred = best.predict(X_te)
proba = best.predict_proba(X_te)[:, 1]
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print(f"Matrice : TN={tn} FP={fp} FN={fn} TP={tp}")
print(f"Sensibilité={recall_score(y_te, pred):.3f} Spécificité={tn/(tn+fp):.3f}",
      f"Précision={precision_score(y_te, pred):.3f} F1={f1_score(y_te, pred):.3f}",
      f"AUC={roc_auc_score(y_te, proba):.3f}")

# ---- Partie 6 : seuil pour une sensibilité cible de 95 %
# Le seuil est choisi sur le TRAIN (probabilités issues de la validation
# croisée), jamais sur le test.
p_cv = cross_val_predict(best, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
prec, rap, seuils = precision_recall_curve(y_tr, p_cv)
seuil = max(s for s, r in zip(seuils, rap[:-1]) if r >= 0.95)
pm = (proba >= seuil).astype(int)
tn2, fp2, fn2, tp2 = confusion_matrix(y_te, pm).ravel()
print(f"Seuil choisi = {seuil:.2f} (au lieu de 0.50)")
print(f"Test : sensibilité={recall_score(y_te, pm):.3f} spécificité={tn2/(tn2+fp2):.3f}",
      f"précision={precision_score(y_te, pm):.3f} | FN={fn2} FP={fp2}")

# ---- Partie 7 : où regarde le modèle ? (carte d'occlusion)
# on prend la patho. la moins "évidente" (proba la plus basse au-dessus
# de 0.5) : la carte est alors plus parlante
cand = np.where((y_te == 1) & (proba > 0.5))[0]
idx = int(cand[np.argmin(proba[cand])])
img = X_te[idx]
base = best.predict_proba(img[None])[0, 1]
carte = np.zeros((16, 16))
for i in range(16):
    for j in range(16):
        im2 = img.copy()
        im2[i*4:i*4+12, j*4:j*4+12] = img.mean()     # on masque un carré de 12x12 pixels
        carte[i, j] = base - best.predict_proba(im2[None])[0, 1]
print(f"Image test n°{idx} : proba patho = {base:.2f} ;",
      f"chute max en masquant une zone = {carte.max():.2f}")
fig, ax = plt.subplots(1, 2, figsize=(7, 3.4))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("image"); ax[0].axis("off")
h = ax[1].imshow(carte, cmap="hot")
ax[1].set_title("importance des zones"); ax[1].axis("off")
plt.colorbar(h, ax=ax[1], fraction=0.046)
plt.tight_layout(); plt.savefig("fig_occlusion.png", dpi=110); plt.close()
