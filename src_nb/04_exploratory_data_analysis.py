# %% [markdown]
"""
# 04 · Exploratory data analysis: standardisation, scaffolds, clustering and chemical space

**Chemoinformatics practicals — Session 4 of 9**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

**Learning goals.** After this session you will be able to
- **standardise** chemical structures (salts, charges, tautomers) and deduplicate a dataset properly;
- aggregate replicate measurements and detect contradictory data;
- apply **drug-likeness filters** and flag problematic substructures (PAINS);
- analyse **scaffolds** and quantify the structural diversity of a compound set;
- **cluster** molecules by similarity (Butina, k-means) and pick diverse representatives;
- visualise **chemical space** with PCA and UMAP, and spot **activity cliffs**.

We work with the EGFR inhibitor dataset built in session 03 (≈5 500 compounds from ChEMBL).

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - **TeachOpenCADD** talktorials **T002 · Molecular filtering: ADME and lead-likeness**, **T003 · Unwanted substructures**, **T005 · Compound clustering** (Volkamer lab; [GitHub](https://github.com/volkamerlab/teachopencadd), CC BY 4.0); the EGFR data are the T001 output (ChEMBL, CC BY-SA 3.0);
> - *Practical Cheminformatics Tutorials* by **Pat Walters** — `taylor_butina_clustering`, `kmeans_clustering`, `find_scaffolds`, `visualizing_chemical_space`, `ChEMBL_data_curation` ([GitHub](https://github.com/PatWalters/practical_cheminformatics_tutorials), MIT) and his blog posts on data curation; the hERG dataset for the exercise is from the same repository;
> - *AI for Chemistry* (EPFL CH-457), **Schwaller group** — `04 - Unsupervised Learning` (`Clustering`, `DimensionalityReduction`; [GitHub](https://github.com/schwallergroup/ai4chem_course), MIT);
> - RDKit `rdMolStandardize` documentation and the **ChEMBL structure pipeline** (Bento *et al.*, J. Cheminform. 2020).
"""

# %%
# @title ⚙️ Setup — run this cell first (≈1 min on Colab)
import sys, os, subprocess, time
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "mols2grid", "umap-learn", "plotly", "seaborn"], check=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
import mols2grid
from tqdm.auto import tqdm
RDLogger.DisableLog("rdApp.*")

REPO_RAW = "https://raw.githubusercontent.com/AxelRolov/chemoinformatics_tutorials/main"
def data_path(filename):
    local = os.path.join("..", "data", filename)
    return local if os.path.exists(local) else f"{REPO_RAW}/data/{filename}"

df = pd.read_csv(data_path("EGFR_compounds_chembl.csv"), index_col=0).rename(columns={"IC50": "IC50_nM"})
print(df.shape); df.head(3)

# %% [markdown]
"""
## 1. Why standardise?

The same compound can appear in a database as a **salt** (`…Cl`), a **charged form** (`C(=O)[O-]`), a different
**tautomer**, or with/without **stereochemistry**. If we don't unify these, we get duplicated molecules with different
labels, leaky train/test splits, and fingerprints that see "different" molecules where a chemist sees one.

A standard pipeline (this is essentially what ChEMBL itself does — Bento *et al.* 2020):

1. **Cleanup**: sanitise, remove explicit H, normalise functional groups (e.g. nitro `N(=O)=O` → `[N+](=O)[O-]`), reionise;
2. **Parent**: keep the largest organic fragment (removes counter-ions and solvents);
3. **Neutralise**: add/remove protons to neutralise charges where chemically sensible;
4. (optional) **Canonical tautomer**;
5. **Canonical SMILES / InChIKey** for deduplication.
"""

# %%
# Look at some problematic entries in our data
print("entries with several fragments (salts):", df["smiles"].str.contains(r"\.").sum())
print("entries with a formal charge        :", df["smiles"].str.contains(r"[+-]\]").sum())
examples = df[df["smiles"].str.contains(r"\.")]["smiles"].head(4).tolist()
examples += ["OC(=O)CC[NH3+]", "O=[N+]([O-])c1ccc(C(=O)[O-])cc1.[Na+]", "Oc1ccccn1", "O=c1cccc[nH]1"]   # extra examples
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in examples], molsPerRow=4, subImgSize=(230, 150), legends=examples)

# %%
# The RDKit standardisation toolbox
cleaner = rdMolStandardize.CleanupParameters()
lfc = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)
uncharger = rdMolStandardize.Uncharger()
tautomer_enum = rdMolStandardize.TautomerEnumerator()
tautomer_enum.SetMaxTautomers(200)

def standardize(smiles, canonical_tautomer=True):
    """Return the standardised canonical SMILES of the parent, neutral molecule (or None)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = rdMolStandardize.Cleanup(mol, cleaner)          # 1. normalise + reionise
    mol = lfc.choose(mol)                                  # 2. largest organic fragment
    mol = uncharger.uncharge(mol)                          # 3. neutralise
    if canonical_tautomer:
        mol = tautomer_enum.Canonicalize(mol)              # 4. canonical tautomer
    return Chem.MolToSmiles(mol)                           # 5. canonical SMILES

for s in examples:
    print(f"{s:45s} -> {standardize(s)}")

# %% [markdown]
"""
Notice the two pyridinol/pyridone tautomers now map to a **single** SMILES, and the sodium salt became the neutral acid.

> ⚠️ Standardisation is a *modelling decision*, not a truth. Whether you neutralise or keep charges, whether you
> collapse tautomers, whether stereochemistry matters, depends on the question (a permeability model cares about charge!).
> Document your choices.
"""

# %%
# Apply to the whole dataset (~1 min: tautomer canonicalisation is the slow step)
t0 = time.time()
df["smiles_std"] = [standardize(s) for s in tqdm(df["smiles"])]
print(f"{time.time() - t0:.0f} s; failed: {df['smiles_std'].isna().sum()}")
changed = (df["smiles_std"] != df["smiles"]).sum()
print(f"{changed} of {len(df)} SMILES changed by standardisation")

# %%
# Deduplicate on the InChIKey of the standardised structure
df["mol"] = df["smiles_std"].apply(Chem.MolFromSmiles)
df["inchikey"] = df["mol"].apply(Chem.MolToInchiKey)
dups = df[df.duplicated("inchikey", keep=False)].sort_values("inchikey")
print(f"{df['inchikey'].nunique()} unique structures among {len(df)} records -> {len(df) - df['inchikey'].nunique()} duplicates")
dups[["molecule_chembl_id", "smiles", "smiles_std", "pIC50"]].head(8)

# %% [markdown]
"""
### Aggregating replicate measurements

When a structure has several measurements we must decide how to combine them. The **median** is robust; the **range**
tells us whether the measurements agree. Records that disagree by more than ~1 log unit are suspicious (different assay
conditions, or an error) and may be dropped.
"""

# %%
agg = (df.groupby("inchikey")
         .agg(chembl_id=("molecule_chembl_id", "first"), smiles=("smiles_std", "first"),
              pIC50=("pIC50", "median"), pIC50_range=("pIC50", lambda x: x.max() - x.min()), n=("pIC50", "size"))
         .reset_index())
print(agg["n"].value_counts().sort_index().to_dict())
noisy = agg[(agg["n"] > 1) & (agg["pIC50_range"] > 1)]
print(f"{len(noisy)} structures with contradictory replicate values (range > 1 log unit)")
noisy.head()

# %%
data = agg[~agg.index.isin(noisy.index)].copy()
data["mol"] = data["smiles"].apply(Chem.MolFromSmiles)
print("clean dataset:", data.shape)

# %% [markdown]
"""
### Censored values
ChEMBL also stores inequalities (`standard_relation` = `>` or `<`, e.g. "IC50 > 10 µM"). We excluded them in session 03 by
filtering `relation="="`. For classification they can be used as inactives; for regression they must be dropped or handled
with special (survival-type) loss functions.

### Exercise 1.1
Try the alternative **ChEMBL structure pipeline** (`pip install chembl_structure_pipeline`; functions `standardize_molblock`
/ `standardize_mol` and `get_parent_mol`) on the `examples` list. Do you get the same results as our RDKit pipeline?
Where do they differ (hint: tautomers)?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
!pip install -q chembl_structure_pipeline
from chembl_structure_pipeline import standardizer

def chembl_standardize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = standardizer.standardize_mol(mol)
    parent, _ = standardizer.get_parent_mol(mol)
    return Chem.MolToSmiles(parent)

for smi in examples:
    print(f"{smi:45s} RDKit: {standardize(smi):35s} ChEMBL: {chembl_standardize(smi)}")

# The ChEMBL pipeline removes salts and normalises functional groups like ours, but deliberately does NOT
# canonicalise tautomers and does NOT neutralise every charge (it keeps chemically meaningful charge states).
# So the two pyridinol/pyridone tautomers stay distinct. Neither is "correct": pick one and document it.
```
</details>
"""


# %% [markdown]
"""
## 2. Drug-likeness filters and unwanted substructures

Before modelling (or buying compounds), we usually ask two questions: *does this look like a drug?* and *is it likely
to be an assay artefact?*

- **Lipinski's rule of five** (session 01) and its relatives (Veber: rotatable bonds ≤ 10, TPSA ≤ 140 Å²; lead-likeness: MW ≤ 350, logP ≤ 3).
- **PAINS** (Pan-Assay INterference compoundS, Baell & Holloway 2010): substructures that light up many assays by
  aggregation, redox cycling or covalent reactivity. RDKit's `FilterCatalog` has them, plus the Brenk and NIH filters.
"""

# %%
def properties(mol):
    return {"MW": Descriptors.MolWt(mol), "logP": Descriptors.MolLogP(mol), "HBD": rdMolDescriptors.CalcNumHBD(mol),
            "HBA": rdMolDescriptors.CalcNumHBA(mol), "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "RotB": rdMolDescriptors.CalcNumRotatableBonds(mol), "AromRings": rdMolDescriptors.CalcNumAromaticRings(mol),
            "FracCSP3": rdMolDescriptors.CalcFractionCSP3(mol), "HeavyAtoms": mol.GetNumHeavyAtoms()}

props = pd.DataFrame([properties(m) for m in data["mol"]], index=data.index)
data = pd.concat([data, props], axis=1)
data["ro5_violations"] = ((data["MW"] > 500).astype(int) + (data["logP"] > 5) + (data["HBD"] > 5) + (data["HBA"] > 10))
data["ro5_violations"].value_counts().sort_index()

# %%
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, col in zip(axes.ravel(), ["MW", "logP", "HBD", "HBA", "TPSA", "RotB", "AromRings", "FracCSP3"]):
    ax.hist(data[col], bins=30, color="steelblue")
    ax.set_title(col)
plt.tight_layout(); plt.show()

# %%
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
params = FilterCatalogParams()
params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
pains = FilterCatalog(params)

def pains_alert(mol):
    entry = pains.GetFirstMatch(mol)
    return entry.GetDescription() if entry else None

data["PAINS"] = data["mol"].apply(pains_alert)
print(f"{data['PAINS'].notna().sum()} PAINS hits ({data['PAINS'].notna().mean():.1%})")
print(data["PAINS"].value_counts().head(8))

# %%
hits = data[data["PAINS"].notna()].head(6)
Draw.MolsToGridImage(hits["mol"].tolist(), molsPerRow=3, subImgSize=(230, 170), legends=hits["PAINS"].tolist())

# %% [markdown]
"""
> PAINS are *alerts*, not verdicts — many approved drugs would be flagged. Use them to prioritise follow-up experiments,
> not to delete data blindly.

## 3. The activity landscape

Let's define **actives** as pIC50 ≥ 6.3 (IC50 ≤ 500 nM) — the threshold used in TeachOpenCADD — and see how activity
relates to simple properties.
"""

# %%
data["active"] = (data["pIC50"] >= 6.3)
print(data["active"].value_counts().to_dict())
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
data["pIC50"].hist(bins=40, ax=axes[0]); axes[0].axvline(6.3, c="r", ls="--"); axes[0].set_xlabel("pIC50")
sns.boxplot(data=data, x="active", y="MW", ax=axes[1])
sns.boxplot(data=data, x="active", y="logP", ax=axes[2])
plt.tight_layout(); plt.show()

# %%
sns.pairplot(data.sample(800, random_state=0), vars=["MW", "logP", "TPSA", "pIC50"], hue="active",
             plot_kws={"s": 10, "alpha": 0.5}, diag_kind="kde", height=2.2)
plt.show()

# %% [markdown]
"""
## 4. Scaffold analysis

The **Bemis–Murcko scaffold** removes all side chains and keeps ring systems plus the linkers between them.
The **generic** scaffold further replaces every atom by carbon and every bond by a single bond (pure topology).
Scaffolds tell us how *diverse* a set is and are the basis of the **scaffold split** used to evaluate models honestly (session 05).
"""

# %%
def scaffold_smiles(mol, generic=False):
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
    return Chem.MolToSmiles(scaf)

data["scaffold"] = data["mol"].apply(scaffold_smiles)
data["scaffold_generic"] = data["mol"].apply(lambda m: scaffold_smiles(m, generic=True))
n_scaf = data["scaffold"].nunique()
singletons = (data["scaffold"].value_counts() == 1).sum()
print(f"{n_scaf} Murcko scaffolds for {len(data)} molecules ({n_scaf/len(data):.2f} per molecule); "
      f"{singletons} scaffolds appear only once; {data['scaffold_generic'].nunique()} generic scaffolds")

# %%
top_scaf = data["scaffold"].value_counts().head(12)
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in top_scaf.index], molsPerRow=4, subImgSize=(200, 150),
                     legends=[f"{c} compounds" for c in top_scaf.values])

# %%
# Are some scaffolds "privileged"? Activity distribution for the 8 most populated scaffolds
top8 = top_scaf.index[:8]
sub = data[data["scaffold"].isin(top8)].copy()
sub["scaffold_rank"] = sub["scaffold"].map({s: i + 1 for i, s in enumerate(top8)})
plt.figure(figsize=(9, 3.5))
sns.boxplot(data=sub, x="scaffold_rank", y="pIC50", color="lightsteelblue")
plt.axhline(6.3, c="r", ls="--"); plt.xlabel("scaffold (rank by frequency)"); plt.title("pIC50 per scaffold")
plt.show()

# %%
# Cumulative view: how many scaffolds do you need to cover half of the compounds?
counts = data["scaffold"].value_counts().values
cum = np.cumsum(counts) / counts.sum()
plt.figure(figsize=(5, 3.2))
plt.plot(np.arange(1, len(cum) + 1) / len(cum), cum)
plt.xlabel("fraction of scaffolds (most populated first)"); plt.ylabel("fraction of compounds covered")
plt.axhline(0.5, c="gray", ls=":")
plt.title(f"{(cum < 0.5).sum() + 1} scaffolds cover 50 % of the data")
plt.show()

# %% [markdown]
"""
### Exercise 4.1
Browse the most populated scaffold with `mols2grid.display(...)`, sorted by pIC50 (the cell below does it). Which
substituent positions vary, and does any position look decisive for activity? This is the beginning of an
**R-group decomposition** / SAR analysis — RDKit's `rdRGroupDecomposition` automates it, and Pat Walters'
`R_group_analysis` notebook shows the full workflow.
"""

# %%
scaf1 = data[data["scaffold"] == top8[0]].sort_values("pIC50", ascending=False)
mols2grid.display(scaf1, smiles_col="smiles", subset=["img", "chembl_id", "pIC50"], size=(170, 130), n_items_per_page=12,
                  transform={"pIC50": lambda x: f"{x:.2f}"})

# %% [markdown]
"""
## 5. Clustering

Clustering groups molecules so that members of a cluster are similar to each other. Uses: picking diverse compounds to
test, organising SAR, building cluster-based train/test splits, removing redundancy.

### Butina clustering (Taylor–Butina, 1999)
The standard method for fingerprints: compute all pairwise Tanimoto **distances** (1 − similarity), and grow clusters
around "centroids" with the most neighbours within a **distance cutoff**. One parameter, deterministic, fast up to ~10⁵ molecules.
"""

# %%
from rdkit.ML.Cluster import Butina

fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [fpgen.GetFingerprint(m) for m in data["mol"]]
X_fp = np.array([fpgen.GetFingerprintAsNumPy(m) for m in data["mol"]], dtype=np.uint8)

def tanimoto_distance_matrix(fps):
    """Lower-triangle list of 1 - Tanimoto, as Butina.ClusterData expects."""
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])
    return dists

t0 = time.time()
dist_matrix = tanimoto_distance_matrix(fps)
print(f"{len(dist_matrix):,} distances in {time.time() - t0:.1f} s")

# %%
def butina(dist_matrix, n, cutoff):
    clusters = Butina.ClusterData(dist_matrix, n, cutoff, isDistData=True)
    return sorted(clusters, key=len, reverse=True)

# Choosing the cutoff: how does the number of clusters change?
for cutoff in [0.2, 0.3, 0.4, 0.5, 0.6]:
    cl = butina(dist_matrix, len(fps), cutoff)
    print(f"cutoff {cutoff}: {len(cl):5d} clusters, largest {len(cl[0]):4d}, singletons {sum(len(c) == 1 for c in cl):5d}")

# %%
cutoff = 0.4                                      # Tanimoto similarity ≥ 0.6 within a cluster
clusters = butina(dist_matrix, len(fps), cutoff)
data["cluster"] = -1
for cid, members in enumerate(clusters):
    data.iloc[list(members), data.columns.get_loc("cluster")] = cid

sizes = [len(c) for c in clusters]
plt.figure(figsize=(6, 3))
plt.hist(sizes, bins=range(1, 60)); plt.yscale("log")
plt.xlabel("cluster size"); plt.ylabel("number of clusters (log)"); plt.title(f"Butina, cutoff {cutoff}: {len(clusters)} clusters")
plt.show()

# %%
# Cluster centroids (the first member of each cluster is the centroid) for the 8 largest clusters
centroids = [c[0] for c in clusters[:8]]
Draw.MolsToGridImage(data["mol"].iloc[centroids].tolist(), molsPerRow=4, subImgSize=(200, 150),
                     legends=[f"cluster {i}: {len(clusters[i])} cpds, median pIC50 {data['pIC50'].iloc[list(clusters[i])].median():.1f}"
                              for i in range(8)])

# %%
# Inside one cluster: how similar are the members really?
members = list(clusters[0])
intra = [DataStructs.TanimotoSimilarity(fps[members[0]], fps[j]) for j in members[1:]]
print(f"cluster 0: similarity to centroid  min {min(intra):.2f}  mean {np.mean(intra):.2f}")
Draw.MolsToGridImage(data["mol"].iloc[members[:8]].tolist(), molsPerRow=4, subImgSize=(200, 150),
                     legends=[f"pIC50 {p:.1f}" for p in data["pIC50"].iloc[members[:8]]])

# %% [markdown]
"""
### k-means on descriptors
Butina needs a similarity matrix; **k-means** works on any numeric vector and scales to millions of points, but you must
choose *k* and it prefers round clusters. With descriptors we must **standardise** the columns first (MW ~ 400, HBD ~ 2).
"""

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

desc_cols = ["MW", "logP", "HBD", "HBA", "TPSA", "RotB", "AromRings", "FracCSP3", "HeavyAtoms"]
X_desc = StandardScaler().fit_transform(data[desc_cols])

scores = {}
for k in [2, 3, 4, 6, 8, 10, 15]:
    km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X_desc)
    scores[k] = silhouette_score(X_desc, km.labels_, sample_size=2000, random_state=0)
print({k: round(v, 3) for k, v in scores.items()})
best_k = max(scores, key=scores.get)
data["kmeans"] = KMeans(n_clusters=best_k, n_init=10, random_state=0).fit_predict(X_desc)
data.groupby("kmeans")[["MW", "logP", "TPSA", "pIC50"]].mean().round(1)

# %% [markdown]
"""
### Picking a diverse subset
If you can only test 20 compounds, you want them *different* from each other. RDKit's **MaxMin picker** greedily picks
the compound farthest from all already picked ones.
"""

# %%
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
picker = MaxMinPicker()
picks = list(picker.LazyBitVectorPick(fps, len(fps), 12, seed=42))
Draw.MolsToGridImage(data["mol"].iloc[picks].tolist(), molsPerRow=4, subImgSize=(200, 150),
                     legends=[f"pIC50 {p:.1f}" for p in data["pIC50"].iloc[picks]])

# %% [markdown]
"""
## 6. Visualising chemical space

Our molecules live in a 2048-dimensional fingerprint space (or a 9-dimensional descriptor space). To *look* at them we
project to 2D:

- **PCA** (linear): preserves global variance; axes are interpretable combinations of the inputs;
- **t-SNE / UMAP** (non-linear): preserve local neighbourhoods — great for seeing clusters, but distances between clusters are not meaningful;
- **TMAP** (tree map): scales to millions of molecules (Probst & Reymond 2020).
"""

# %%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_desc)
print("explained variance:", np.round(pca.explained_variance_ratio_, 2))
loadings = pd.DataFrame(pca.components_.T, index=desc_cols, columns=["PC1", "PC2"]).round(2)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})
sc = axes[0].scatter(pcs[:, 0], pcs[:, 1], c=data["pIC50"], cmap="viridis", s=6, alpha=0.6)
plt.colorbar(sc, ax=axes[0], label="pIC50"); axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].set_title("PCA of 9 descriptors")
loadings.plot.barh(ax=axes[1]); axes[1].set_title("loadings")
plt.tight_layout(); plt.show()

# %%
import umap
t0 = time.time()
reducer = umap.UMAP(n_neighbors=15, min_dist=0.2, metric="jaccard", random_state=0)
emb = reducer.fit_transform(X_fp.astype(bool))
print(f"UMAP on {X_fp.shape} fingerprints: {time.time() - t0:.0f} s")
data["umap_x"], data["umap_y"] = emb[:, 0], emb[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc = axes[0].scatter(emb[:, 0], emb[:, 1], c=data["pIC50"], cmap="viridis", s=6, alpha=0.7)
plt.colorbar(sc, ax=axes[0], label="pIC50"); axes[0].set_title("UMAP of ECFP4 – coloured by activity")
top_scaf_ids = {s: i for i, s in enumerate(top_scaf.index[:8])}
col = data["scaffold"].map(top_scaf_ids).fillna(-1)
axes[1].scatter(emb[col < 0, 0], emb[col < 0, 1], c="lightgray", s=4)
axes[1].scatter(emb[col >= 0, 0], emb[col >= 0, 1], c=col[col >= 0], cmap="tab10", s=8)
axes[1].set_title("UMAP – top-8 scaffolds highlighted")
for ax in axes: ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()

# %%
# Interactive version: hover to see the ChEMBL ID, SMILES and pIC50 (plotly)
import plotly.express as px
fig = px.scatter(data, x="umap_x", y="umap_y", color="pIC50", hover_data=["chembl_id", "smiles", "scaffold"],
                 color_continuous_scale="viridis", width=750, height=550, title="EGFR compounds – UMAP of ECFP4")
fig.update_traces(marker=dict(size=4))
fig.show()

# %% [markdown]
"""
### Where does the dataset sit relative to approved drugs?
Projecting two sets into the *same* map is how we check whether a training set covers the region we care about
(the **applicability domain** question of session 05).
"""

# %%
drugs = pd.read_csv(data_path("chembl_drugs_walters.smi"), sep=" ", names=["SMILES", "ChEMBL_ID"])
drugs["mol"] = drugs["SMILES"].apply(Chem.MolFromSmiles)
X_drugs = np.array([fpgen.GetFingerprintAsNumPy(m) for m in drugs["mol"]], dtype=bool)
emb_drugs = reducer.transform(X_drugs)          # project onto the *fitted* map

plt.figure(figsize=(6, 5))
plt.scatter(emb[:, 0], emb[:, 1], c="lightgray", s=4, label="EGFR compounds")
plt.scatter(emb_drugs[:, 0], emb_drugs[:, 1], c="crimson", s=6, label="approved drugs")
plt.legend(); plt.xticks([]); plt.yticks([]); plt.title("Approved drugs projected onto the EGFR map")
plt.show()

# %% [markdown]
"""
## 7. Activity cliffs

The similarity principle has exceptions: pairs of very similar molecules with very different activities — **activity cliffs**.
They carry the most SAR information (and are the hardest cases for any model).
"""

# %%
sim_cut, act_cut = 0.8, 2.0
cliffs = []
pIC50 = data["pIC50"].values
for i in tqdm(range(len(fps))):
    sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    for j in np.where(sims >= sim_cut)[0]:
        j += i + 1
        if abs(pIC50[i] - pIC50[j]) >= act_cut:
            cliffs.append((i, j, sims[j - i - 1], pIC50[i], pIC50[j]))
cliffs = pd.DataFrame(cliffs, columns=["i", "j", "similarity", "pIC50_i", "pIC50_j"])
cliffs["delta"] = (cliffs["pIC50_i"] - cliffs["pIC50_j"]).abs()
print(len(cliffs), "activity cliffs (Tanimoto ≥ 0.8, ΔpIC50 ≥ 2)")

# %%
c = cliffs.sort_values("delta", ascending=False).iloc[0]
Draw.MolsToGridImage([data["mol"].iloc[int(c.i)], data["mol"].iloc[int(c.j)]], subImgSize=(300, 220),
                     legends=[f"pIC50 = {c.pIC50_i:.2f}", f"pIC50 = {c.pIC50_j:.2f}   (T = {c.similarity:.2f})"])

# %% [markdown]
"""
What differs between the two molecules? A single atom or group — the kind of detail a fingerprint barely sees.

## 8. Save the curated dataset
"""

# %%
cols = ["chembl_id", "smiles", "inchikey", "pIC50", "active", "scaffold", "cluster"] + desc_cols
data[cols].to_csv("EGFR_curated.csv", index=False)
print("saved EGFR_curated.csv:", data[cols].shape)

# %% [markdown]
"""
## 9. Exercises: repeat with the hERG dataset

`hERG_chembl_walters.csv` (4 042 compounds with pIC50 against the hERG potassium channel — the classic cardiotoxicity anti-target).

1. Standardise and deduplicate. How many duplicates / contradictory replicates do you find?
2. Compute the scaffold statistics. Is hERG data more or less diverse than the EGFR data (scaffolds per molecule)?
3. Cluster with Butina (cutoff 0.4) and plot cluster sizes.
4. UMAP coloured by pIC50: is activity concentrated in specific regions (as for EGFR) or spread out? Think about *why*
   (hERG binding is driven by general properties — lipophilicity, a basic amine — more than by a specific scaffold).
5. Bonus: correlate pIC50 with `logP`. Which is the better single predictor of hERG activity: logP or MW?
"""

# %%
herg = pd.read_csv(data_path("hERG_chembl_walters.csv"))
herg.head()

# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
# 1. standardise and deduplicate
herg["smiles_std"] = [standardize(s) for s in tqdm(herg["SMILES"])]
herg["mol"] = herg["smiles_std"].apply(Chem.MolFromSmiles)
herg["inchikey"] = herg["mol"].apply(Chem.MolToInchiKey)
h = (herg.groupby("inchikey")
       .agg(name=("Name", "first"), smiles=("smiles_std", "first"), pIC50=("pIC50", "median"),
            rng=("pIC50", lambda x: x.max() - x.min()), n=("pIC50", "size")).reset_index())
print(len(herg), "records ->", len(h), "unique structures;", (h["n"] > 1).sum(), "with replicates,",
      ((h["n"] > 1) & (h["rng"] > 1)).sum(), "contradictory")
h = h[~((h["n"] > 1) & (h["rng"] > 1))].copy()
h["mol"] = h["smiles"].apply(Chem.MolFromSmiles)

# 2. scaffolds
h["scaffold"] = h["mol"].apply(scaffold_smiles)
print(f"{h['scaffold'].nunique()} scaffolds for {len(h)} molecules = {h['scaffold'].nunique()/len(h):.2f} per molecule")
# ~0.6-0.7 per molecule vs 0.35 for EGFR: the hERG set is far more diverse (it is an anti-target,
# so compounds come from many unrelated projects).

# 3. clustering
h_fps = [fpgen.GetFingerprint(m) for m in h["mol"]]
h_clusters = butina(tanimoto_distance_matrix(h_fps), len(h_fps), 0.4)
print(len(h_clusters), "clusters, largest", len(h_clusters[0]))
plt.hist([len(c) for c in h_clusters], bins=range(1, 40)); plt.yscale("log"); plt.xlabel("cluster size"); plt.show()

# 4. UMAP
Xh = np.array([fpgen.GetFingerprintAsNumPy(m) for m in h["mol"]], dtype=bool)
emb_h = umap.UMAP(n_neighbors=15, min_dist=0.2, metric="jaccard", random_state=0).fit_transform(Xh)
plt.scatter(emb_h[:, 0], emb_h[:, 1], c=h["pIC50"], cmap="viridis", s=6)
plt.colorbar(label="pIC50"); plt.xticks([]); plt.yticks([]); plt.title("hERG chemical space"); plt.show()
# Activity is much more spread out than for EGFR - hERG blockade is a property-driven liability.

# 5. single-descriptor correlations
hp = pd.DataFrame([properties(m) for m in h["mol"]])
for col in ["logP", "MW", "TPSA"]:
    print(col, "Pearson r =", round(hp[col].corr(h["pIC50"]), 2))
# logP correlates clearly (~0.4-0.5), MW less, TPSA negatively: the classic hERG pharmacophore
# (lipophilic + basic amine) is visible in a single descriptor.
```
</details>
"""


# %% [markdown]
"""
## Further reading
- Fourches, Muratov, Tropsha, *Trust, but verify: on the importance of chemical structure curation*, J. Chem. Inf. Model. **2010**, 50, 1189.
- Bento *et al.*, *An open source chemical structure curation pipeline using RDKit*, J. Cheminform. **2020**, 12, 51.
- Bemis & Murcko, *The properties of known drugs. 1. Molecular frameworks*, J. Med. Chem. **1996**, 39, 2887.
- Butina, *Unsupervised data base clustering based on Daylight's fingerprint and Tanimoto similarity*, J. Chem. Inf. Comput. Sci. **1999**, 39, 747.
- Probst & Reymond, *Visualization of very large high-dimensional data sets as minimum spanning trees* (TMAP), J. Cheminform. **2020**.
- Pat Walters' blog, *Practical Cheminformatics*: <https://practicalcheminformatics.blogspot.com>.

Next session: **05 · Classical machine learning** — QSAR models on the curated dataset.
"""
