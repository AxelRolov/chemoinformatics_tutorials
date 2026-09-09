# %% [markdown]
"""
# 05 · Classical machine learning: QSAR/QSPR models

**Chemoinformatics practicals — Session 5 of 9**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

**Learning goals.** After this session you will be able to
- formulate a property-prediction problem as supervised **regression** or **classification** (QSPR / QSAR);
- featurise molecules (descriptors, fingerprints) and train linear models, random forests, gradient boosting, kNN and SVMs with scikit-learn;
- evaluate models properly: train/test split, **cross-validation**, the right **metrics**, and why a **scaffold split** is more honest than a random one;
- interpret a model (feature importances → substructures) and estimate where it can be trusted (**applicability domain**, uncertainty);
- save a model and use it to predict new molecules.

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - **TeachOpenCADD** talktorial **T007 · Ligand-based screening: machine learning** (Volkamer lab; [GitHub](https://github.com/volkamerlab/teachopencadd), CC BY 4.0);
> - *Practical Cheminformatics Tutorials* by **Pat Walters** — `ml_models/` (`regression_model`, `classification_model`, `cross_validation`, `comparing_*_models`, `QSAR_in_8_lines`) ([GitHub](https://github.com/PatWalters/practical_cheminformatics_tutorials), MIT);
> - *AI for Chemistry* (EPFL CH-457), **Schwaller group** — `02 - Supervised Learning` ([GitHub](https://github.com/schwallergroup/ai4chem_course), MIT);
> - **MolSSI** cheminformatics workshop — `06_sklearn_fitting`, `07_ESOL_fitting` (MIT);
> - A. D. White, *Deep Learning for Molecules and Materials*, chapter *Machine learning* ([dmol.pub](https://dmol.pub), CC BY-NC 3.0).
> Data: ESOL (Delaney 2004, via MoleculeNet) and the curated EGFR set from session 04 (ChEMBL, CC BY-SA 3.0).
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

1. **Data**: molecules with measured $y$ (session 03–04: curated, deduplicated, standardised);
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

# %% [markdown]
"""
**Loading the two datasets.** The `.rename(...)` is not cosmetic: the ESOL file's target column is literally called
*"measured log solubility in mols per litre"*, and you do not want that string appearing thirty times in the rest of
the notebook.

`esol["smiles"].apply(Chem.MolFromSmiles)` adds a column of RDKit molecule objects. Carrying the parsed molecules in
the dataframe means we parse each SMILES once instead of once per featuriser — with 6600 molecules that matters.

The printed shapes should be `(1128, 11)` and `(5511, 7)`. Printing the shape after every load is a habit worth
acquiring: it is how you notice that a file you thought had 5511 rows now has 5510.
"""

# %%
esol = pd.read_csv(data_path("esol_delaney.csv")).rename(columns={"measured log solubility in mols per litre": "logS",
                                                                   "Compound ID": "name"})
esol["mol"] = esol["smiles"].apply(Chem.MolFromSmiles)
egfr = pd.read_csv(data_path("EGFR_curated.csv"))
egfr["mol"] = egfr["smiles"].apply(Chem.MolFromSmiles)
print(esol.shape, egfr.shape)

# %% [markdown]
"""
### The two featurisers

Everything in this session depends on how we turn a molecule into numbers, so we build **two** representations and
compare them throughout. This is the single most important experimental design choice in the notebook.

**Featuriser 1 · fingerprints.** `GetFingerprintAsNumPy` gives one 2048-long array of 0s and 1s per molecule
(session 02), which we stack into an `(n_molecules, 2048)` matrix. It is *sparse*: **98.9 % of this matrix is zero**, because a
molecule sets only as many bits as it has distinct atom environments — a median of 18 for the mostly small molecules
in ESOL, 40–60 for a typical drug. `dtype=np.uint8` keeps it at 2.3 MB instead of the 18.5 MB the same numbers would
take as float64. Each column means "this particular substructure is present".
"""

# %%
# Featuriser 1: Morgan fingerprints (ECFP4, 2048 bits)
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def fingerprints(mols):
    return np.array([fpgen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)

# %% [markdown]
"""
**Featuriser 2 · descriptors.** `Descriptors.descList` is every 2D descriptor RDKit knows — 217 of them here, from
molecular weight through topological indices to fragment counts. `CalcMolDescriptors(m)` computes them all for one
molecule and returns a dict, so a list of dicts becomes a dataframe in one step; the `[DESC_NAMES]` reindex pins the
column order, which matters as soon as you save a model and reload it later.

`replace([np.inf, -np.inf], np.nan)` is there because a handful of descriptors divide by zero or take a log of zero
for some molecules. Turning those into NaN lets us deal with all the missing values in one place, which is the next
cell.

Unlike the fingerprint, each column here means a *physical or topological quantity* and is dense and real-valued.
"""

# %%
# Featuriser 2: all RDKit 2D descriptors, cleaned (no NaN/inf, no constant columns)
DESC_NAMES = [n for n, _ in Descriptors.descList]
def descriptors(mols):
    X = pd.DataFrame([Descriptors.CalcMolDescriptors(m) for m in mols])[DESC_NAMES]
    X = X.replace([np.inf, -np.inf], np.nan)
    return X

# %% [markdown]
"""
**Compute both.** About 8 seconds for 1128 molecules, nearly all of it in the descriptors — fingerprints are cheap,
200 descriptors are not.

Look at the two shapes side by side: `(1128, 2048)` sparse binary against `(1128, 217)` dense real-valued. Almost
every result in this session traces back to that contrast, and by section 3 you will be able to predict which
representation suits which model.
"""

# %%
t0 = time.time()
X_esol_fp = fingerprints(esol["mol"]);   X_esol_desc = descriptors(esol["mol"])
print(f"ESOL features: {X_esol_fp.shape} fingerprints, {X_esol_desc.shape} descriptors  ({time.time()-t0:.0f} s)")
y_esol = esol["logS"].values

# %% [markdown]
"""
**Cleaning the descriptor matrix.** Two kinds of column are useless: those that are NaN for *every* molecule (a
descriptor that fails on this whole dataset) and those with a single distinct value (`nunique() <= 1` — no
information to learn from). Eighteen of the 217 columns go — mostly radical-electron and rare-fragment counts that are
zero for all 1128 molecules — leaving 199.

The columns that are NaN for only *some* molecules stay. They are filled in later, inside a scikit-learn `Pipeline`,
and that is the point: a pipeline computes the median on the training fold only. Imputing here, before the split,
would let information from the test molecules into the training set — a small but real leak, and the same mistake in
a more damaging form is why section 4.2 exists.
"""

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

# %% [markdown]
"""
**The split.** 20 % held out, `random_state=SEED` so that everyone in the room gets the same 902 training and 226
test molecules and can compare numbers. The convention we follow everywhere below: `idx_train` / `idx_test` are
*index arrays*, so the same split can be applied to the fingerprint matrix and to the descriptor matrix without
duplicating anything.
"""

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

idx_train, idx_test = train_test_split(np.arange(len(esol)), test_size=0.2, random_state=SEED)
print(len(idx_train), "training and", len(idx_test), "test molecules")

# %% [markdown]
"""
**A metric helper, and a baseline to beat.** `regression_report` prints the three numbers you should always report
together:

- **RMSE** — in the units of the target (here log solubility units), and dominated by the worst predictions, because
  errors are squared;
- **MAE** — the typical error, much less sensitive to a few outliers. If RMSE is far above MAE, you have a handful of
  badly missed molecules, and you should go and look at them;
- **R²** — the fraction of variance explained, useful for comparing across datasets but meaningless on its own.

Then the baseline: predict the training mean for every molecule. It scores RMSE = 2.174 and R² = −0.000, and that
zero is not a coincidence — R² is *defined* relative to the mean predictor. Every model below has to beat 2.174 to
have earned its existence. It sounds trivial; published QSAR models have failed it.
"""

# %%
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

# %% [markdown]
"""
**Refitting Delaney's 2004 model.** Four descriptors, chosen by a chemist for physical reasons: solubility falls with
lipophilicity (logP), with size (MW), and with aromatic, flat, well-stacked surface (aromatic proportion).
`aromatic_proportion` is not in RDKit's list, so we write it ourselves — three lines, and a reminder that a
descriptor is just a function from a molecule to a number.

Watch the RMSE: **1.113** against the baseline's 2.174, with four numbers per molecule and a model you could evaluate
by hand. That is the honest benchmark that the rest of the session has to beat.
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
Compare our coefficients with Delaney's published equation:

$$\log S = 0.16 - 0.63\,\text{clogP} - 0.0062\,\text{MW} + 0.066\,\text{RB} - 0.74\,\text{AP}$$

Three of the four agree in sign and rough magnitude: logP dominates and is negative (−0.755 here, −0.63 published),
MW is negative and tiny, aromatic proportion is negative. The chemistry is the same, refitted on a different split
with a different logP implementation.

The rotatable-bond term is the interesting disagreement: Delaney found it slightly **positive**, we get slightly
**negative** (−0.009). Neither is wrong. Rotatable bonds correlate strongly with molecular weight, and when two
descriptors carry nearly the same information a linear model can shift weight between them almost freely — the
*predictions* are stable, the individual *coefficients* are not. This is the first appearance of a theme that returns
in section 5: do not read a coefficient, or a feature importance, as a statement about chemistry unless you have
checked that the descriptors are not collinear.

Now let's throw more features and more flexible models at the problem.
"""

# %% [markdown]
"""
**Six models in one dictionary.** The dictionary maps a name to a `(model, which_features)` pair, so the loop below
is representation-agnostic and adding a seventh model is one line. Read the six choices as a designed experiment
rather than a shopping list:

- **Ridge** — linear, but on all 199 surviving descriptors with an L2 penalty (`alpha=10`) to survive the
  collinearity we just discussed;
- **Random forest** and **XGBoost** on descriptors — the two workhorses of tabular chemistry; bagging versus boosting;
- **Random forest on ECFP4** — the *same model* on the other representation. This pair is the whole point;
- **kNN with Jaccard distance** — the purest similarity-based prediction: "this molecule resembles those five, so
  average their solubility". Jaccard on binary vectors *is* one minus the Tanimoto of session 02;
- **SVR with an RBF kernel** — a kernel method, to show that this family is still competitive on small data.

Every descriptor model is wrapped in `make_pipeline(SimpleImputer(...), ...)` — this is where the deferred imputation
happens — and the ones sensitive to feature scale also get a `StandardScaler`. Trees do not need scaling; Ridge and
SVR very much do.
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

# %% [markdown]
"""
**Train and score them all.** Nothing subtle in the loop; it fits, predicts, and records the metrics plus the wall
time. The table is sorted by RMSE, and it says two things.

**Descriptors beat fingerprints, decisively**: XGBoost on descriptors reaches RMSE 0.674, while the *same* random
forest scores 0.747 on descriptors and 1.159 on fingerprints. Solubility is a whole-molecule property — it depends on
overall polarity and surface, not on the presence of particular substructures — and a fingerprint has no column for
"how greasy is this molecule". The kNN result (1.205) makes the same point from the other side: structurally similar
molecules do *not* reliably have similar solubility.

The **time** column is worth a glance too: SVR reaches essentially the best RMSE in 0.2 s, where the random forest
takes 6.7 s. On 1000 molecules that is irrelevant; on 100 000 it decides what you can actually run.
"""

# %%
results, preds = [], {}
for name, (model, feat) in models_reg.items():
    X = features[feat]
    t0 = time.time()
    model.fit(X[idx_train], y_esol[idx_train])
    p = model.predict(X[idx_test]); preds[name] = p
    results.append({"model": name, "RMSE": np.sqrt(mean_squared_error(y_esol[idx_test], p)),
                    "MAE": mean_absolute_error(y_esol[idx_test], p), "R2": r2_score(y_esol[idx_test], p), "time (s)": time.time() - t0})
pd.DataFrame(results).set_index("model").round(3).sort_values("RMSE")

# %% [markdown]
"""
**Look at the predictions, never only at the metric.** Six scatter plots of measured against predicted, with the
diagonal drawn in. A single RMSE cannot distinguish between a model that is uniformly a bit wrong and one that is
excellent over most of the range and catastrophic at the extremes — and those two need completely different fixes.

Look specifically at the bottom left of each panel, below logS ≈ −7. Every model pulls those very insoluble
molecules towards the mean, because there are few of them to learn from. That is *regression to the mean*, it is
visible here in the shape of the cloud, and it is exactly the region a formulation chemist would care about.
"""

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

# %% [markdown]
"""
**Five splits instead of one.** `KFold(n_splits=5, shuffle=True)` cuts the data into five parts and trains five
times, each part serving once as the test set. `cross_val_score` handles the loop; the negative sign is a
scikit-learn convention (it always *maximises* a score, so RMSE is passed as its negation).

The mean tells you how good the model is; the **standard deviation** tells you whether the differences you have been
reading off the previous table are real. Ridge comes out at 0.685 ± 0.013 and the random forest at 0.654 ± 0.058 —
so the forest's apparent advantage is smaller than its own run-to-run scatter, and calling it "better" from a single
split would be over-reading. XGBoost at 0.604 ± 0.040 is ahead by more than the noise, and that conclusion you can
defend.
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

# %% [markdown]
"""
**The box plot is the honest version of the table.** Five numbers per model, drawn rather than averaged. Two models
whose boxes overlap are not distinguishable on this dataset — and the fingerprint forest's box sits so far above the
others that no amount of tuning will close the gap. When you report model comparisons, show this, not a bar chart of
means.
"""

# %%
plt.figure(figsize=(6, 3))
plt.boxplot(list(cv_results.values()), tick_labels=[k.split(" (")[0] + "\n(" + k.split(" (")[1] for k in cv_results])
plt.ylabel("RMSE (5-fold CV)"); plt.show()

# %% [markdown]
"""
### 3.4 Hyperparameters
Models have knobs (`n_estimators`, `max_depth`, `alpha`, `C` …) that are **not** learned from the data. Tune them with
cross-validation *inside the training set* — never on the test set, or your test estimate becomes optimistic.
"""

# %% [markdown]
"""
**Tuning, done correctly.** `GridSearchCV` tries all 2 × 3 × 2 = 12 combinations, evaluating each by 3-fold
cross-validation **inside the training set**, then refits the winner on the whole training set. The test set is
touched exactly once, at the end. Any tuning that consults the test set — even by eye, even once — turns your test
estimate into a training estimate.

The double-underscore names (`randomforestregressor__n_estimators`) are how you reach a step inside a pipeline:
`<step name in lower case>__<parameter>`.

Expect `max_features=0.3, min_samples_leaf=1, n_estimators=300` with CV RMSE 0.650, and test RMSE 0.705 against the
untuned 0.747. A real but modest gain — which is the typical result. Tuning is worth an hour; it is not worth a week,
and it will never rescue the wrong representation.
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

# %% [markdown]
"""
**Switching problem, target and dataset.** Now the EGFR set, and a *binary* label: `active` is pIC50 ≥ 6.3, which is
58.5 % of the 5511 compounds. Note we use fingerprints here, not descriptors, and that is the right choice for the
opposite reason to solubility: binding to a specific kinase pocket *is* about the presence of particular
substructures.

The long import line is the metric list from the table above — worth reading as a list, because knowing which
metrics exist is most of knowing how to evaluate a classifier.
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

# %% [markdown]
"""
**A stratified split, and five classifiers.** `stratify=y_egfr` keeps the 58.5 / 41.5 class balance identical in both
halves. Without it, a random split of an imbalanced set can hand you a test set with a noticeably different
prevalence, which shifts every threshold-dependent metric for no interesting reason.

The five models mirror the regression zoo — linear, bagged trees, boosted trees, nearest neighbours, kernel — so you
can see whether the ranking of model families carries over from one problem to the other. (It does not, quite: watch
where logistic regression lands here compared with Ridge on solubility.)
"""

# %%
tr, te = train_test_split(np.arange(len(egfr)), test_size=0.2, random_state=SEED, stratify=y_egfr)

models_clf = {
    "Logistic regression": LogisticRegression(max_iter=2000, C=0.1),
    "Random forest": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    "kNN Tanimoto (k=5)": KNeighborsClassifier(n_neighbors=5, metric="jaccard"),
    "XGBoost": xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, random_state=SEED),
    "SVM (RBF)": SVC(C=1.0, probability=True, random_state=SEED),
}

# %% [markdown]
"""
**Four metrics, computed from probabilities rather than labels.** Two of them — ROC-AUC and PR-AUC — need the
probability, because they sweep every possible threshold; the other two need a decision, so they threshold at 0.5.

Keeping the probability around instead of only the predicted class is a habit that pays: it lets you change the
threshold later (exercise 3 at the end of the notebook), calibrate it, or rank a screening library, none of which is
possible once you have thrown the numbers away.
"""

# %%
def classification_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {"ROC-AUC": roc_auc_score(y_true, proba), "PR-AUC": average_precision_score(y_true, proba),
            "bal. acc.": balanced_accuracy_score(y_true, pred), "MCC": matthews_corrcoef(y_true, pred)}

# %% [markdown]
"""
**Train them all, and store the probabilities.** The `"kNN" in name` test converts the fingerprint matrix to `bool`
for the nearest-neighbour model only, because scikit-learn's Jaccard metric expects boolean input.

Read the resulting table with an eye on the spread: ROC-AUC runs from 0.917 to 0.938 — every one of these models
works, and the differences between them are small. That is the usual situation, and it is why the *next* subsection,
about how you split the data, matters far more than which of these five you pick. Note also the cost of that near-tie:
the SVM takes ~98 s against XGBoost's ~3 s for a slightly worse score, because kernel methods scale badly with the
number of molecules.
"""

# %%
clf_results, probas = [], {}
for name, model in models_clf.items():
    Xtr = X_egfr[tr].astype(bool) if "kNN" in name else X_egfr[tr]
    Xte = X_egfr[te].astype(bool) if "kNN" in name else X_egfr[te]
    t0 = time.time()
    model.fit(Xtr, y_egfr[tr])
    proba = model.predict_proba(Xte)[:, 1]; probas[name] = proba
    clf_results.append({"model": name, **classification_metrics(y_egfr[te], proba), "time (s)": time.time() - t0})
pd.DataFrame(clf_results).set_index("model").round(3)

# %% [markdown]
"""
**Three ways to look at a classifier.** The **ROC** curve plots true positives against false positives as the
threshold sweeps; the diagonal is random guessing. The **precision–recall** curve answers the question a screening
chemist actually asks — "if I test the top *n* compounds, what fraction will be active?" — and its baseline is the
dashed line at the prevalence, 0.585, not 0.5. On a set that is 58.5 % active, a PR curve looks flattering; on a
realistic screening deck with 0.1 % actives, the same model's PR curve would collapse while its ROC-AUC barely moved.
That is the reason to plot both.

The **confusion matrix** on the right shows the four raw counts at threshold 0.5, which is what you have to look at
before claiming a model is "87 % accurate".
"""

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

# %% [markdown]
"""
**The per-class report, and the asymmetry it exposes.** Overall accuracy is 0.87, which sounds uniform. It is not:
recall is 0.92 for actives but 0.80 for inactives, so the model finds most of the actives while misfiling one
inactive in five as active.

Which of those errors is expensive depends entirely on your project. Prioritising compounds to test? False positives
cost you plate space. Filtering out compounds you will never look at again? A false negative may be the molecule you
needed. Accuracy hides that trade-off; this table shows it, and the threshold is yours to move.
"""

# %%
print(classification_report(y_egfr[te], (probas["Random forest"] >= 0.5).astype(int), target_names=["inactive", "active"]))

# %% [markdown]
"""
### 4.2 The split matters: random vs scaffold split

A random split puts close analogues of every test molecule into the training set — the model is tested on *interpolation*.
In real projects you predict **new chemotypes**. A **scaffold split** puts whole scaffolds (session 04) either in train or
in test, which is much closer to that situation. Watch the metrics drop.
"""

# %% [markdown]
"""
**Implementing a scaffold split.** The molecules were already assigned Murcko scaffolds in session 04, so the job is
to keep every scaffold group entirely on one side of the split.

The one design decision is the order in which groups are filled into the test set, and here they go **smallest
first** (the DeepChem-style variant), with a random tie-break. That means the test set fills up with rare, one-off
scaffolds while the big well-populated series stay in training — which is deliberately the hardest and most honest
arrangement, and the closest to what happens when a project moves to a new chemical series.
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

# %% [markdown]
"""
**Run it, and verify it.** 4407 training and 1102 test molecules — and then the check that actually matters: the
number of scaffolds appearing on both sides, which must be **0**. Print that assertion rather than trusting the
implementation. A scaffold split with a leak is worse than a random split, because it comes with a false claim of
rigour.
"""

# %%
tr_s, te_s = scaffold_split(egfr["scaffold"].values)
print(f"scaffold split: {len(tr_s)} train / {len(te_s)} test; "
      f"{len(set(egfr['scaffold'].iloc[tr_s]) & set(egfr['scaffold'].iloc[te_s]))} scaffolds shared (should be 0)")

# %% [markdown]
"""
**Quantifying how new the test set is.** A helper that answers, for each test molecule, "how similar is the most
similar training molecule?" — the maximum Tanimoto to the training set.

`CreateFromBitString` rebuilds an RDKit bit vector from our numpy rows so we can use the C++
`BulkTanimotoSimilarity`, which compares one fingerprint against a whole list in one call. It is not elegant, but it
turns a 4407 × 1102 comparison into something that finishes while you read this paragraph.
"""

# %%
# Nearest training neighbour similarity for test molecules: how "new" is the test set?
def max_sim_to_train(X_train, X_test):
    fps_tr = [DataStructs.CreateFromBitString("".join(map(str, row))) for row in X_train]
    out = []
    for row in X_test:
        q = DataStructs.CreateFromBitString("".join(map(str, row)))
        out.append(max(DataStructs.BulkTanimotoSimilarity(q, fps_tr)))
    return np.array(out)

# %% [markdown]
"""
**The two distributions, side by side.** The medians come out at **0.80** for the random split and **0.72** for the
scaffold split. Look at the shape rather than the medians, though: the scaffold split grows a whole left shoulder of
test molecules with similarity 0.2–0.5 that the random split simply does not have, and the random split has a spike
at 1.0 — test molecules with a *near-duplicate* in the training set.

That left shoulder is what "predicting a new chemotype" looks like as a number. Keep it in mind when you read the
next table.
"""

# %%
sim_random = max_sim_to_train(X_egfr[tr], X_egfr[te])
sim_scaffold = max_sim_to_train(X_egfr[tr_s], X_egfr[te_s])
plt.figure(figsize=(6, 3))
plt.hist(sim_random, bins=30, alpha=0.6, label=f"random split (median {np.median(sim_random):.2f})")
plt.hist(sim_scaffold, bins=30, alpha=0.6, label=f"scaffold split (median {np.median(sim_scaffold):.2f})")
plt.xlabel("max Tanimoto similarity of a test molecule to the training set"); plt.legend(); plt.show()

# %% [markdown]
"""
**The most important table in the course.** The same three models, the same features, the same code — only the split
changes.

ROC-AUC falls from 0.933 to 0.897 for the random forest, and MCC from 0.732 to 0.676. Logistic regression drops
further (MCC 0.726 → 0.618), XGBoost too (0.756 → 0.651), so the *ranking of the models changes as well*: the forest
was second under a random split and is first under a scaffold split. A model chosen on a random split may not be the
model you wanted.

Two conclusions to take away. First, the honest performance number for "will this work on a new series?" is the
scaffold one, and it is always the lower one. Second, the drop here is moderate — a few points — because EGFR
inhibitors are a well-covered, densely sampled chemical space; on a sparser dataset the same experiment routinely
turns ROC-AUC 0.9 into 0.65. Whenever you read a QSAR paper, the first question is which split it used.
"""

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

# %% [markdown]
"""
**y-randomisation: the experiment that catches the mistakes you did not think of.** Shuffle the labels so that no
relationship between structure and activity can possibly remain, then train the same model. A trustworthy pipeline
must now score ROC-AUC ≈ 0.5.

We get **0.510**, so the pipeline is clean. A number meaningfully above 0.5 here would prove that the model is
finding signal in something other than the chemistry — duplicate molecules spanning the split, an index accidentally
correlated with the label, a metric computed on the wrong array. It costs one cell and it catches whole classes of
bug, including several that produced retracted papers.
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

# %% [markdown]
"""
**Which descriptors does the forest actually use?** A tree ensemble records how much each feature reduced the loss
across all splits; `feature_importances_` is that, normalised. `rf_desc[-1]` reaches the last step of the pipeline —
the forest itself, past the imputer.

The result is startling and worth sitting with: **MolLogP alone takes about 0.75** of the total importance, and
nothing else reaches 0.03. On this dataset, a 300-tree forest on 199 descriptors is largely an elaborate function of
logP — which is precisely what Delaney's four-descriptor equation said in 2004.

Now recall the collinearity from section 3.2, because it applies with full force here. Several descriptors in this
set are near-duplicates of logP (`MolMR`, `SlogP_VSA*`), and when features are correlated the trees split on
whichever one they happen to sample first, so importance concentrates on one representative and its equally
informative twins look worthless. Read this chart as "the model relies on lipophilicity", never as "MolMR is
irrelevant to solubility".
"""

# %%
rf_desc = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED))
rf_desc.fit(X_esol_desc.values[idx_train], y_esol[idx_train])
imp = pd.Series(rf_desc[-1].feature_importances_, index=DESC_KEEP).sort_values(ascending=False)
plt.figure(figsize=(6, 3.5)); imp.head(12)[::-1].plot.barh(); plt.xlabel("importance"); plt.title("ESOL: random-forest feature importances"); plt.show()

# %% [markdown]
"""
**The same question for fingerprints.** Here the important features are *bits*, and a bit number is meaningless on
its own — bit 1452 tells you nothing until you find out which substructure it encodes.
"""

# %%
rf_fp = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED).fit(X_egfr[tr], y_egfr[tr])
top_bits = np.argsort(-rf_fp.feature_importances_)[:6]
print("most important ECFP4 bits for EGFR activity:", top_bits)

# %% [markdown]
"""
**Turning bits back into chemistry.** This is the cell that makes fingerprint models interpretable. For each
important bit we find an active molecule that sets it, ask RDKit — through `AdditionalOutput`/`GetBitInfoMap`, which
records which atom and radius produced each bit — where it came from, and draw that environment with
`DrawMorganBits`.

The legends carry the number that decides whether a bit is interesting: how often it occurs in actives *versus*
inactives. A bit present in 40 % of actives and 5 % of inactives is a real structural signal, a candidate
pharmacophore fragment you can show a medicinal chemist. A bit present in 40 % of both is just common chemistry that
the forest used for a cheap split. Check both percentages before telling anyone you have found a pharmacophore.
"""

# %%
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

# %% [markdown]
"""
**Applicability domain: where the model has no business predicting.** A regression model on pIC50, trained on the
*scaffold* split so that the test molecules are genuinely unfamiliar, and then the error binned by how similar each
test molecule is to the training set.

The pattern is real but noisier than textbooks suggest. The clean signal is at the top: test molecules with a
training neighbour above 0.7 have a median absolute error of **0.43** log units, against 0.6–0.9 for everything
below. In between, the bins are not monotone — the (0.4, 0.5] bin comes out worst of all, on 61 molecules. So
similarity buys you a *warning*, not a calibrated error bar: below ~0.5 the prediction is unreliable and you cannot
say by how much. That is still worth having, and it is more than most deployed models report. The scatter plot
underneath shows why the medians behave as they do — at every similarity there are a few large errors; what changes
is how many.
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

# %% [markdown]
"""
**A second, independent uncertainty estimate.** A random forest is 300 trees that each saw a different bootstrap
sample. Where they agree, the prediction is robust; where they disagree, it is not — so the standard deviation across
`rf_reg.estimators_` is a free uncertainty estimate, needing no extra training.

The Spearman correlation with the true error is **0.27**: positive, useful, far from a guarantee. The quartile table
is the practical version of the same information, and it is cleanly monotone — mean absolute error 0.50 for the
quartile the forest is most confident about, rising to 0.87 for the least. So you cannot trust the spread as an error
bar on one molecule, but you can absolutely use it to rank a list and decide what to test first. That distinction
between per-molecule and per-batch reliability is what "uncertainty quantification" means in practice.
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

# %% [markdown]
"""
**Saving more than the model.** `joblib.dump` writes a *dictionary*, and what is in it besides the fitted forest is
the actual lesson. The fingerprint recipe (`"Morgan r=2, 2048 bits"`), because a model fed a different fingerprint
silently returns nonsense. The activity threshold (6.3), because "active" is a choice, not a fact. And the training
fingerprints, so the applicability domain can be checked at prediction time.

A pickled estimator on its own is a liability six months later. Note also that the final model is fitted on **all**
5511 compounds: validation is finished, its verdict is recorded, and now we want every molecule we have.
"""

# %%
import joblib
final_model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=SEED).fit(X_egfr, y_egfr)
joblib.dump({"model": final_model, "fp": "Morgan r=2, 2048 bits", "threshold": 6.3, "train_fps": X_egfr}, "egfr_rf.joblib")

# %% [markdown]
"""
**Reload, and predict four molecules you know.** This is the whole deployment path in six lines: load the bundle,
featurise the new SMILES *with the recipe recorded in it*, predict, and — in the same table — report the distance to
the training set.

That last column is what turns a prediction into a decision. Aspirin comes back at P(active) = 0.03 with a nearest
neighbour of 0.36, so the table says `extrapolating`: the answer is probably right, but the model was never asked
about anything like aspirin and does not know that it is right. Imatinib sits at 0.19 with similarity 0.71 — inside
the domain, and a genuine prediction that this EGFR model does not expect much from a BCR-ABL drug.

And then gefitinib and erlotinib, at 0.37 and 0.34. Read the blockquote below before you conclude anything about the
model.
"""

# %%
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
> you build your own dataset (session 03) aggregate *all* measurements (median) rather than keeping the first.

## 7. QSAR in 8 lines
Everything above, condensed (after Pat Walters' *QSAR in 8 lines*). Understand each line and you understand the session.
"""

# %% [markdown]
"""
**Eight lines, and everything that is missing from them.** Load, featurise, split, fit, score: R² = 0.67. This is a
real QSAR model and it took eight lines, which is genuinely how fast the tooling has become.

Now list what those eight lines do *not* do, because that list is the syllabus of this session: no baseline to
compare against, no cross-validation, so we have no idea whether 0.67 is stable; a random split, so the number is an
interpolation estimate; no applicability domain, so it will answer confidently about anything; no interpretation; and
fingerprints on a whole-molecule property, which section 3.2 showed to be the wrong representation here — swap in
descriptors and the same eight lines reach R² ≈ 0.88.

Eight lines to a model. The rest of the session is what makes it trustworthy.
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

Next session: **06 · Deep learning** — neural networks on fingerprints and graphs.
"""
