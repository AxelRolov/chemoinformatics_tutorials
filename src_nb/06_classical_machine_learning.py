# %% [markdown]
"""
# 06 · Classical machine learning: QSAR/QSPR models

**Chemoinformatics (UFAZ, L2 S3) — Practical session 6**
Instructor: Alexey Orlov (Université de Strasbourg / UFAZ)

**Learning goals.** After this session you will be able to
- formulate a property-prediction problem as supervised **regression** or **classification** (QSPR / QSAR);
- featurise molecules (descriptors, fingerprints) and train linear models, random forests, gradient boosting, kNN and SVMs with scikit-learn;
- evaluate models properly: train/test split, **cross-validation**, the right **metrics**, and why a **scaffold split** is more honest than a random one;
- interpret a model (feature importances → substructures) and estimate where it can be trusted (**applicability domain**, uncertainty);
- save a model and use it to predict new molecules.

---
> **Credits.** Adapted, with modifications for the UFAZ course, from
> - **TeachOpenCADD** talktorial **T007 · Ligand-based screening: machine learning** (Volkamer lab; [GitHub](https://github.com/volkamerlab/teachopencadd), CC BY 4.0);
> - *Practical Cheminformatics Tutorials* by **Pat Walters** — `ml_models/` (`regression_model`, `classification_model`, `cross_validation`, `comparing_*_models`, `QSAR_in_8_lines`) ([GitHub](https://github.com/PatWalters/practical_cheminformatics_tutorials), MIT);
> - *AI for Chemistry* (EPFL CH-457), **Schwaller group** — `02 - Supervised Learning` ([GitHub](https://github.com/schwallergroup/ai4chem_course), MIT);
> - **MolSSI** cheminformatics workshop — `06_sklearn_fitting`, `07_ESOL_fitting` (MIT);
> - A. D. White, *Deep Learning for Molecules and Materials*, chapter *Machine learning* ([dmol.pub](https://dmol.pub), CC BY-NC 3.0).
> Data: ESOL (Delaney 2004, via MoleculeNet) and the curated EGFR set from session 05 (ChEMBL, CC BY-SA 3.0).
"""

# %%
# @title ⚙️ Setup — run this cell first
import sys, os, subprocess, time, warnings
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "scikit-learn", "xgboost", "seaborn"], check=False)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, Descriptors, rdFingerprintGenerator
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
RDLogger.DisableLog("rdApp.*")
import sklearn; print("scikit-learn", sklearn.__version__)

REPO_RAW = "https://raw.githubusercontent.com/AxelRolov/chemoinformatics_tutorials/main"
def data_path(filename):
    local = os.path.join("..", "data", filename)
    return local if os.path.exists(local) else f"{REPO_RAW}/data/{filename}"

SEED = 42
np.random.seed(SEED)

# %% [markdown]
"""
## 1. The QSAR idea

**Q**uantitative **S**tructure–**A**ctivity (or **P**roperty) **R**elationships assume that a property $y$ is a function of
the structure: $y = f(\text{molecule}) + \varepsilon$. We do not know $f$, so we *learn* it from examples.
The recipe never changes:

1. **Data**: molecules with measured $y$ (session 04–05: curated, deduplicated, standardised);
2. **Features** $X$: numbers describing each molecule (session 02: descriptors, fingerprints);
3. **Model**: a family of functions $f_\theta$ (linear, tree ensemble, kernel, neural network) and a **loss** to minimise;
4. **Validation**: estimate how well $f$ predicts molecules it has *never seen*;
5. **Use**: predict, rank, prioritise — and know when *not* to trust the prediction.

Two flavours: **regression** ($y$ continuous: logS, pIC50) and **classification** ($y$ categorical: active/inactive, toxic/safe).
"""

# %% [markdown]
"""
## 2. Data and features

We use two datasets you already know:
- **ESOL**: 1128 molecules with measured aqueous solubility (logS) — a *regression* problem;
- **EGFR**: 5511 curated ChEMBL compounds with pIC50; active if pIC50 ≥ 6.3 — a *classification* problem (and a regression one).
"""

# %%
esol = pd.read_csv(data_path("esol_delaney.csv")).rename(columns={"measured log solubility in mols per litre": "logS",
                                                                   "Compound ID": "name"})
esol["mol"] = esol["smiles"].apply(Chem.MolFromSmiles)
egfr = pd.read_csv(data_path("EGFR_curated.csv"))
egfr["mol"] = egfr["smiles"].apply(Chem.MolFromSmiles)
print(esol.shape, egfr.shape)

# %%
# Featuriser 1: Morgan fingerprints (ECFP4, 2048 bits)
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def fingerprints(mols):
    return np.array([fpgen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)

# Featuriser 2: all RDKit 2D descriptors, cleaned (no NaN/inf, no constant columns)
DESC_NAMES = [n for n, _ in Descriptors.descList]
def descriptors(mols):
    X = pd.DataFrame([Descriptors.CalcMolDescriptors(m) for m in mols])[DESC_NAMES]
    X = X.replace([np.inf, -np.inf], np.nan)
    return X

t0 = time.time()
X_esol_fp = fingerprints(esol["mol"]);   X_esol_desc = descriptors(esol["mol"])
print(f"ESOL features: {X_esol_fp.shape} fingerprints, {X_esol_desc.shape} descriptors  ({time.time()-t0:.0f} s)")
y_esol = esol["logS"].values

# %%
# Descriptor hygiene: drop columns that are all-NaN or constant, impute the rest with the training median (later, inside a Pipeline)
bad = X_esol_desc.columns[(X_esol_desc.isna().all()) | (X_esol_desc.nunique() <= 1)]
print(len(bad), "useless descriptor columns dropped:", list(bad)[:6], "...")
DESC_KEEP = [c for c in X_esol_desc.columns if c not in bad]
X_esol_desc = X_esol_desc[DESC_KEEP]

# %% [markdown]
"""
## 3. Regression: predicting solubility

### 3.1 Train/test split and a baseline
We hold out 20 % of the molecules. **Everything** (feature scaling, model selection) must be done on the training part only.
A model is only useful if it beats a trivial baseline — here, "predict the mean".
"""

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

idx_train, idx_test = train_test_split(np.arange(len(esol)), test_size=0.2, random_state=SEED)
print(len(idx_train), "training and", len(idx_test), "test molecules")

def regression_report(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label:35s} RMSE = {rmse:.3f}   MAE = {mean_absolute_error(y_true, y_pred):.3f}   R² = {r2_score(y_true, y_pred):.3f}")
    return rmse

baseline = np.full(len(idx_test), y_esol[idx_train].mean())
regression_report(y_esol[idx_test], baseline, "baseline (mean)");

# %% [markdown]
"""
### 3.2 Linear regression on 4 descriptors: rediscovering ESOL
Delaney's 2004 model was a linear regression on logP, MW, rotatable bonds and aromatic proportion. Let's refit it.
"""

# %%
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def aromatic_proportion(mol):
    return sum(a.GetIsAromatic() for a in mol.GetAtoms()) / mol.GetNumHeavyAtoms()

X4 = pd.DataFrame({"logP": X_esol_desc["MolLogP"], "MW": X_esol_desc["MolWt"],
                   "RotB": X_esol_desc["NumRotatableBonds"], "AromProp": esol["mol"].apply(aromatic_proportion)})
lin4 = LinearRegression().fit(X4.iloc[idx_train], y_esol[idx_train])
print("coefficients:", dict(zip(X4.columns, lin4.coef_.round(3))), "intercept:", round(lin4.intercept_, 3))
regression_report(y_esol[idx_test], lin4.predict(X4.iloc[idx_test]), "linear, 4 ESOL descriptors");

# %% [markdown]
"""
Compare with Delaney's published equation: logS = 0.16 − 0.63 clogP − 0.0062 MW + 0.066 RB − 0.74 AP. Similar signs and
magnitudes — the chemistry is the same. Now let's throw more features and more flexible models at the problem.
"""

# %%
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import xgboost as xgb

models_reg = {
    "Ridge (descriptors)": (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)), "desc"),
    "Random forest (descriptors)": (make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED)), "desc"),
    "XGBoost (descriptors)": (make_pipeline(SimpleImputer(strategy="median"), xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, random_state=SEED)), "desc"),
    "Random forest (ECFP4)": (RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED), "fp"),
    "kNN Tanimoto (ECFP4)": (KNeighborsRegressor(n_neighbors=5, metric="jaccard", weights="distance"), "fp"),
    "SVR RBF (descriptors)": (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), SVR(C=10, epsilon=0.1)), "desc"),
}
features = {"desc": X_esol_desc.values, "fp": X_esol_fp.astype(bool)}

results, preds = [], {}
for name, (model, feat) in models_reg.items():
    X = features[feat]
    t0 = time.time()
    model.fit(X[idx_train], y_esol[idx_train])
    p = model.predict(X[idx_test]); preds[name] = p
    results.append({"model": name, "RMSE": np.sqrt(mean_squared_error(y_esol[idx_test], p)),
                    "MAE": mean_absolute_error(y_esol[idx_test], p), "R2": r2_score(y_esol[idx_test], p), "time (s)": time.time() - t0})
pd.DataFrame(results).set_index("model").round(3).sort_values("RMSE")

# %%
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
for ax, (name, p) in zip(axes.ravel(), preds.items()):
    ax.scatter(y_esol[idx_test], p, s=10, alpha=0.6)
    ax.plot([-11, 2], [-11, 2], "k--", lw=1)
    ax.set_title(f"{name}\nRMSE = {np.sqrt(mean_squared_error(y_esol[idx_test], p)):.2f}", fontsize=9)
    ax.set_xlabel("measured logS"); ax.set_ylabel("predicted")
plt.tight_layout(); plt.show()

# %% [markdown]
"""
> Descriptor-based tree ensembles win on this small dataset; fingerprints alone struggle with a *global* property like
> solubility (it depends on the whole molecule, not on the presence of a few substructures). This is a general pattern:
> **the right representation matters more than the model**.

### 3.3 Cross-validation
One split gives one number — and that number depends on which molecules happened to land in the test set.
**k-fold cross-validation** repeats the experiment on k different splits and reports mean ± standard deviation.
"""

# %%
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = {}
for name in ["Ridge (descriptors)", "Random forest (descriptors)", "Random forest (ECFP4)", "XGBoost (descriptors)"]:
    model, feat = models_reg[name]
    scores = cross_val_score(model, features[feat], y_esol, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
    cv_results[name] = -scores
    print(f"{name:30s} RMSE = {-scores.mean():.3f} ± {scores.std():.3f}")

plt.figure(figsize=(6, 3))
plt.boxplot(list(cv_results.values()), tick_labels=[k.split(" (")[0] + "\n(" + k.split(" (")[1] for k in cv_results])
plt.ylabel("RMSE (5-fold CV)"); plt.show()

# %% [markdown]
"""
### 3.4 Hyperparameters
Models have knobs (`n_estimators`, `max_depth`, `alpha`, `C` …) that are **not** learned from the data. Tune them with
cross-validation *inside the training set* — never on the test set, or your test estimate becomes optimistic.
"""

# %%
from sklearn.model_selection import GridSearchCV

rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_jobs=-1, random_state=SEED))
grid = {"randomforestregressor__n_estimators": [100, 300], "randomforestregressor__max_features": ["sqrt", 0.3, 1.0],
        "randomforestregressor__min_samples_leaf": [1, 3]}
search = GridSearchCV(rf, grid, cv=KFold(3, shuffle=True, random_state=SEED), scoring="neg_root_mean_squared_error", n_jobs=1)
search.fit(X_esol_desc.values[idx_train], y_esol[idx_train])
print("best parameters:", {k.split("__")[1]: v for k, v in search.best_params_.items()}, f"CV RMSE = {-search.best_score_:.3f}")
regression_report(y_esol[idx_test], search.predict(X_esol_desc.values[idx_test]), "tuned RF (descriptors), test set");

# %% [markdown]
"""
### Exercise 3.1
1. Try Morgan fingerprints with **counts** (`fpgen.GetCountFingerprintAsNumPy`) and radius 3. Does RF improve?
2. Concatenate descriptors **and** fingerprints (`np.hstack`). Better than either alone?
3. Replace `SimpleImputer(strategy="median")` by `strategy="mean"` — does it matter? Why do we need an imputer at all?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
# 1. count fingerprints, radius 3
cgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
X_cnt = np.array([cgen.GetCountFingerprintAsNumPy(m) for m in esol["mol"]], dtype=np.float32)
rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED).fit(X_cnt[idx_train], y_esol[idx_train])
regression_report(y_esol[idx_test], rf.predict(X_cnt[idx_test]), "RF, count FP r=3")

# 2. descriptors + fingerprints
X_both = np.hstack([np.nan_to_num(X_esol_desc.values, nan=0.0), X_esol_fp])
rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED).fit(X_both[idx_train], y_esol[idx_train])
regression_report(y_esol[idx_test], rf.predict(X_both[idx_test]), "RF, descriptors + FP")
# Usually within noise of descriptors alone: the trees simply ignore the mostly-zero fingerprint columns.

# 3. mean vs median imputation changes little here because only a few descriptors have NaNs
#    (e.g. BCUT2D_* fail for molecules without certain atom types). An imputer is needed at all because
#    scikit-learn estimators refuse arrays containing NaN - and dropping those *columns* would throw away
#    information for the 99 % of molecules where they are defined.
```
</details>
"""


# %% [markdown]
"""
## 4. Classification: active or inactive against EGFR?

### 4.1 Models and metrics
Accuracy is misleading when classes are imbalanced (here 58 % actives — a model that always says "active" gets 58 %).
Use several metrics:

| metric | what it measures |
|---|---|
| **ROC-AUC** | probability that a random active is ranked above a random inactive (0.5 = random, 1 = perfect); threshold-free |
| **PR-AUC / average precision** | precision–recall trade-off; more informative when actives are rare |
| **balanced accuracy** | mean of sensitivity (recall of actives) and specificity (recall of inactives) |
| **MCC** | Matthews correlation coefficient, −1…1; robust summary of the confusion matrix |
| **precision / recall / F1** | at a chosen threshold (default 0.5) |
"""

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef,
                             confusion_matrix, roc_curve, precision_recall_curve, classification_report)

X_egfr = fingerprints(egfr["mol"])
y_egfr = egfr["active"].astype(int).values
print(X_egfr.shape, f"actives: {y_egfr.mean():.1%}")

tr, te = train_test_split(np.arange(len(egfr)), test_size=0.2, random_state=SEED, stratify=y_egfr)

models_clf = {
    "Logistic regression": LogisticRegression(max_iter=2000, C=0.1),
    "Random forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    "kNN Tanimoto (k=5)": KNeighborsClassifier(n_neighbors=5, metric="jaccard"),
    "XGBoost": xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=SEED),
    "SVM (RBF)": SVC(C=1.0, probability=True, random_state=SEED),
}

def classification_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {"ROC-AUC": roc_auc_score(y_true, proba), "PR-AUC": average_precision_score(y_true, proba),
            "bal. acc.": balanced_accuracy_score(y_true, pred), "MCC": matthews_corrcoef(y_true, pred)}

clf_results, probas = [], {}
for name, model in models_clf.items():
    Xtr = X_egfr[tr].astype(bool) if "kNN" in name else X_egfr[tr]
    Xte = X_egfr[te].astype(bool) if "kNN" in name else X_egfr[te]
    t0 = time.time()
    model.fit(Xtr, y_egfr[tr])
    proba = model.predict_proba(Xte)[:, 1]; probas[name] = proba
    clf_results.append({"model": name, **classification_metrics(y_egfr[te], proba), "time (s)": time.time() - t0})
pd.DataFrame(clf_results).set_index("model").round(3)

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for name, proba in probas.items():
    fpr, tpr, _ = roc_curve(y_egfr[te], proba); axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_egfr[te], proba):.2f})")
    prec, rec, _ = precision_recall_curve(y_egfr[te], proba); axes[1].plot(rec, prec, label=name)
axes[0].plot([0, 1], [0, 1], "k--", lw=1); axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate"); axes[0].set_title("ROC"); axes[0].legend(fontsize=7)
axes[1].axhline(y_egfr[te].mean(), c="k", ls="--", lw=1); axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("precision–recall")
cm = confusion_matrix(y_egfr[te], (probas["Random forest"] >= 0.5).astype(int))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2], cbar=False,
            xticklabels=["pred. inactive", "pred. active"], yticklabels=["inactive", "active"])
axes[2].set_title("Random forest – confusion matrix")
plt.tight_layout(); plt.show()

# %%
print(classification_report(y_egfr[te], (probas["Random forest"] >= 0.5).astype(int), target_names=["inactive", "active"]))

# %% [markdown]
"""
### 4.2 The split matters: random vs scaffold split

A random split puts close analogues of every test molecule into the training set — the model is tested on *interpolation*.
In real projects you predict **new chemotypes**. A **scaffold split** puts whole scaffolds (session 05) either in train or
in test, which is much closer to that situation. Watch the metrics drop.
"""

# %%
def scaffold_split(scaffolds, test_size=0.2, seed=SEED):
    """Group molecules by Murcko scaffold; assign whole scaffold groups to the test set (small groups first, DeepChem-style variant)."""
    groups = pd.Series(scaffolds).groupby(pd.Series(scaffolds)).indices          # scaffold -> array of row indices
    rng = np.random.RandomState(seed)
    sizes = sorted(groups.items(), key=lambda kv: (len(kv[1]), rng.rand()))         # smallest scaffolds first -> test
    test, train = [], []
    n_test = int(test_size * len(scaffolds))
    for scaf, idx in sizes:
        (test if len(test) + len(idx) <= n_test else train).extend(idx)
    return np.array(train), np.array(test)

tr_s, te_s = scaffold_split(egfr["scaffold"].values)
print(f"scaffold split: {len(tr_s)} train / {len(te_s)} test; "
      f"{len(set(egfr['scaffold'].iloc[tr_s]) & set(egfr['scaffold'].iloc[te_s]))} scaffolds shared (should be 0)")

# Nearest training neighbour similarity for test molecules: how "new" is the test set?
def max_sim_to_train(X_train, X_test):
    fps_tr = [DataStructs.CreateFromBitString("".join(map(str, row))) for row in X_train]
    out = []
    for row in X_test:
        q = DataStructs.CreateFromBitString("".join(map(str, row)))
        out.append(max(DataStructs.BulkTanimotoSimilarity(q, fps_tr)))
    return np.array(out)

sim_random = max_sim_to_train(X_egfr[tr], X_egfr[te])
sim_scaffold = max_sim_to_train(X_egfr[tr_s], X_egfr[te_s])
plt.figure(figsize=(6, 3))
plt.hist(sim_random, bins=30, alpha=0.6, label=f"random split (median {np.median(sim_random):.2f})")
plt.hist(sim_scaffold, bins=30, alpha=0.6, label=f"scaffold split (median {np.median(sim_scaffold):.2f})")
plt.xlabel("max Tanimoto similarity of a test molecule to the training set"); plt.legend(); plt.show()

# %%
rows = []
for split_name, (a, b) in {"random": (tr, te), "scaffold": (tr_s, te_s)}.items():
    for name in ["Logistic regression", "Random forest", "XGBoost"]:
        model = models_clf[name]
        model.fit(X_egfr[a], y_egfr[a])
        proba = model.predict_proba(X_egfr[b])[:, 1]
        rows.append({"split": split_name, "model": name, **classification_metrics(y_egfr[b], proba)})
pd.DataFrame(rows).set_index(["split", "model"]).round(3)

# %% [markdown]
"""
### 4.3 Sanity check: y-randomisation
If we shuffle the labels and still get a good model, something is wrong (leakage, or a metric that rewards guessing).
"""

# %%
y_shuffled = np.random.RandomState(0).permutation(y_egfr)
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED).fit(X_egfr[tr], y_shuffled[tr])
print(f"ROC-AUC with shuffled labels: {roc_auc_score(y_shuffled[te], rf.predict_proba(X_egfr[te])[:, 1]):.3f}  (should be ≈ 0.5)")

# %% [markdown]
"""
### Exercise 4.1
1. Change the activity threshold to pIC50 ≥ 7 (IC50 ≤ 100 nM). How does the class balance change, and the metrics?
2. Turn the EGFR problem into a **regression** of pIC50 with a random forest on fingerprints. Report RMSE for the random and
   scaffold splits. Is RMSE ≈ 0.7 log units good? (Compare with the experimental uncertainty of IC50 values, roughly 0.3–0.5 log units.)
3. Use `sklearn.model_selection.GroupKFold` with scaffold groups to do a 5-fold *scaffold* cross-validation.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
# 1. stricter threshold
y7 = (egfr["pIC50"] >= 7).astype(int).values
print(f"actives at pIC50>=7: {y7.mean():.1%}")
tr7, te7 = train_test_split(np.arange(len(egfr)), test_size=0.2, random_state=SEED, stratify=y7)
m = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(X_egfr[tr7], y7[tr7])
print(classification_metrics(y7[te7], m.predict_proba(X_egfr[te7])[:, 1]))
# Fewer actives -> PR-AUC drops more than ROC-AUC; ROC-AUC is optimistic on imbalanced data.

# 2. regression instead of classification
y_reg = egfr["pIC50"].values
for name, (a, b) in {"random": (tr, te), "scaffold": (tr_s, te_s)}.items():
    r = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED).fit(X_egfr[a], y_reg[a])
    p = r.predict(X_egfr[b])
    print(f"{name:9s} RMSE = {np.sqrt(mean_squared_error(y_reg[b], p)):.3f}  R2 = {r2_score(y_reg[b], p):.3f}")
# ~0.7-0.9 log units. Experimental IC50 reproducibility across assays is ~0.4-0.5 log units
# (Landrum & Riniker 2024), so we are within about a factor of two of the data's own noise floor:
# a good model, but do not expect it to rank two compounds that differ by 0.3 log units.

# 3. scaffold cross-validation
from sklearn.model_selection import GroupKFold, cross_val_score
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED),
                         X_egfr, y_egfr, groups=egfr["scaffold"], cv=gkf, scoring="roc_auc")
print("scaffold CV ROC-AUC:", scores.round(3), "mean", scores.mean().round(3))
```
</details>
"""


# %% [markdown]
"""
## 5. Opening the black box

### 5.1 Which features does the model use?
Tree ensembles provide **feature importances**. For descriptors, they are directly readable; for fingerprint bits we
can look up the substructure behind an important bit in a molecule where it is set.
"""

# %%
rf_desc = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED))
rf_desc.fit(X_esol_desc.values[idx_train], y_esol[idx_train])
imp = pd.Series(rf_desc[-1].feature_importances_, index=DESC_KEEP).sort_values(ascending=False)
plt.figure(figsize=(6, 3.5)); imp.head(12)[::-1].plot.barh(); plt.xlabel("importance"); plt.title("ESOL: random-forest feature importances"); plt.show()

# %%
rf_fp = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(X_egfr[tr], y_egfr[tr])
top_bits = np.argsort(-rf_fp.feature_importances_)[:6]
print("most important ECFP4 bits for EGFR activity:", top_bits)

# Find an active molecule containing each bit and draw the substructure
ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
tiles, legends = [], []
for bit in top_bits:
    for i in np.where((X_egfr[:, bit] == 1) & (y_egfr == 1))[0][:1]:
        m = egfr["mol"].iloc[i]
        fpgen.GetFingerprint(m, additionalOutput=ao)
        tiles.append((m, int(bit), ao.GetBitInfoMap()))
        legends.append(f"bit {bit}: in {X_egfr[y_egfr == 1, bit].mean():.0%} of actives, {X_egfr[y_egfr == 0, bit].mean():.0%} of inactives")
Draw.DrawMorganBits(tiles, molsPerRow=3, subImgSize=(250, 200), legends=legends)

# %% [markdown]
"""
> For richer explanations use **SHAP** values (`pip install shap`; `shap.TreeExplainer`) — they attribute each prediction to
> each feature, and can be mapped back onto atoms (see Pat Walters' and the `MolFaith` benchmark for caveats).

### 5.2 Applicability domain: when should we *not* trust the model?
A QSAR model interpolates. If a new molecule is far from everything in the training set, the prediction is a guess.
The simplest diagnostic: **similarity to the nearest training neighbour** versus **prediction error**.
"""

# %%
rf_reg = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED).fit(X_egfr[tr_s], egfr["pIC50"].values[tr_s])
pred = rf_reg.predict(X_egfr[te_s])
err = np.abs(pred - egfr["pIC50"].values[te_s])
bins = pd.cut(sim_scaffold, [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
summary = pd.DataFrame({"similarity bin": bins, "abs. error": err}).groupby("similarity bin", observed=True)["abs. error"].agg(["count", "mean", "median"]).round(2)
display(summary)
plt.figure(figsize=(5.5, 3.5))
plt.scatter(sim_scaffold, err, s=8, alpha=0.4)
plt.xlabel("max Tanimoto to training set"); plt.ylabel("|error| in pIC50"); plt.title("Applicability domain (scaffold split)")
plt.show()

# %% [markdown]
"""
### 5.3 Uncertainty from an ensemble
The spread of the individual trees' predictions is a cheap uncertainty estimate. Does it correlate with the error?
"""

# %%
tree_preds = np.stack([t.predict(X_egfr[te_s]) for t in rf_reg.estimators_])
std = tree_preds.std(axis=0)
print(f"Spearman correlation between predicted std and |error|: {pd.Series(std).corr(pd.Series(err), method='spearman'):.2f}")
q = pd.qcut(std, 4, labels=["most confident", "", " ", "least confident"])
pd.DataFrame({"confidence quartile": q, "abs. error": err}).groupby("confidence quartile", observed=True)["abs. error"].mean().round(2)

# %% [markdown]
"""
## 6. Deploying a model

Save the fitted model with `joblib`, reload it later, and predict new molecules — including a check that they are inside
the applicability domain.
"""

# %%
import joblib
final_model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED).fit(X_egfr, y_egfr)
joblib.dump({"model": final_model, "fp": "Morgan r=2, 2048 bits", "threshold": 6.3, "train_fps": X_egfr}, "egfr_rf.joblib")

bundle = joblib.load("egfr_rf.joblib")
new_smiles = {"gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
              "erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
              "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
              "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"}
new_mols = [Chem.MolFromSmiles(s) for s in new_smiles.values()]
X_new = fingerprints(new_mols)
p_active = bundle["model"].predict_proba(X_new)[:, 1]
nn_sim = max_sim_to_train(bundle["train_fps"], X_new)
pd.DataFrame({"P(active)": p_active.round(2), "nearest training similarity": nn_sim.round(2),
              "in domain?": np.where(nn_sim > 0.4, "yes", "extrapolating")}, index=new_smiles.keys())

# %% [markdown]
"""
> **Surprise?** Gefitinib and erlotinib — *approved EGFR inhibitors* — come out with P(active) < 0.5, although they are in the
> training set. Look them up in `egfr`: their pIC50 values are 6.29 and 6.11, i.e. **just below** our 6.3 threshold. The cached
> dataset (TeachOpenCADD T001) kept the *first* IC50 record ChEMBL returned for each compound; for gefitinib that is a
> 515 nM value from one particular assay, whereas other assays report IC50 well below 10 nM. Three lessons:
> a model is only as good as its labels; hard thresholds turn small measurement differences into label flips; and when
> you build your own dataset (session 04) aggregate *all* measurements (median) rather than keeping the first.

## 7. QSAR in 8 lines
Everything above, condensed (after Pat Walters' *QSAR in 8 lines*). Understand each line and you understand the session.
"""

# %%
df = pd.read_csv(data_path("esol_delaney.csv")).rename(columns={"measured log solubility in mols per litre": "logS"})
X = np.array([fpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in df["smiles"]])
y = df["logS"].values
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
model = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
print(f"R² = {r2_score(y_te, model.predict(X_te)):.2f}")

# %% [markdown]
"""
## 8. Exercises
1. **hERG** (`hERG_chembl_walters.csv`): build a classifier (active if pIC50 ≥ 5) with random and scaffold splits. Report ROC-AUC and MCC. Is hERG easier or harder than EGFR? Why might that be?
2. **Learning curve**: train the ESOL RF on 10 %, 25 %, 50 %, 100 % of the training set and plot test RMSE versus training size. Would more data help?
3. **Threshold tuning**: for the EGFR RF, plot precision and recall as a function of the probability threshold. Which threshold would you pick if experiments are expensive (few false positives wanted)?
4. **Consensus**: average the probabilities of RF, XGBoost and logistic regression. Is the ensemble better than the best single model?


> Solutions to 2–4 (try first!):
>
> ```python
> # 2. learning curve
> sizes = [0.1, 0.25, 0.5, 1.0]
> for f in sizes:
>     sub = idx_train[: int(f * len(idx_train))]
>     m = make_pipeline(SimpleImputer(strategy="median"),
>                       RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED))
>     m.fit(X_esol_desc.values[sub], y_esol[sub])
>     print(f, round(np.sqrt(mean_squared_error(y_esol[idx_test], m.predict(X_esol_desc.values[idx_test]))), 3))
>
> # 3. threshold tuning
> prec, rec, thr = precision_recall_curve(y_egfr[te], probas["Random forest"])
> plt.plot(thr, prec[:-1], label="precision"); plt.plot(thr, rec[:-1], label="recall")
> plt.xlabel("threshold"); plt.legend(); plt.show()
>
> # 4. consensus
> cons = np.mean([probas[k] for k in ["Random forest", "XGBoost", "Logistic regression"]], axis=0)
> print(classification_metrics(y_egfr[te], cons))
> ```

## Further reading
- Tropsha, *Best practices for QSAR model development, validation, and exploitation*, Mol. Inf. **2010**, 29, 476.
- Sheridan, *Time-split cross-validation as a method for estimating the goodness of prospective prediction*, J. Chem. Inf. Model. **2013**.
- Walters & Barzilay, *Applications of deep learning in molecule generation and molecular property prediction*, Acc. Chem. Res. **2021** — for what comes next.
- scikit-learn user guide: <https://scikit-learn.org/stable/user_guide.html>.

Next session: **07 · Deep learning** — neural networks on fingerprints and graphs.
"""
