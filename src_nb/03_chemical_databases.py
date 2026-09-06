# %% [markdown]
"""
# 03 · Chemical databases: PubChem, ChEMBL, the PDB and open datasets

**Chemoinformatics practicals — Session 3 of 9**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

**Learning goals.** After this session you will be able to
- name the major public chemical data resources and what each one is good for;
- query **PubChem** programmatically (name → structure → properties, similarity search);
- build a bioactivity dataset for a protein target from **ChEMBL** with its Python client, and explain IC50 / pIC50;
- load open **machine-learning-ready datasets** from the Hugging Face Hub (OpenADMET) and know where else to look (MoleculeNet, TDC, Polaris);
- retrieve a protein–ligand structure and its ligand from the **Protein Data Bank**;
- recognise the data-quality questions that every downloaded dataset raises (units, duplicates, censored values, licences).

> 🌐 This notebook talks to web services. If a service is temporarily down, the cells fall back to cached copies
> shipped with the course so you can continue.

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - **TeachOpenCADD** talktorials **T001 · Compound data acquisition (ChEMBL)**, **T011 · Querying online API webservices**, **T013 · Data acquisition from PubChem** and **T008 · Protein data acquisition (PDB)** (Volkamer lab; [GitHub](https://github.com/volkamerlab/teachopencadd), CC BY 4.0). The cached EGFR dataset `EGFR_compounds_chembl.csv` is the T001 output (ChEMBL data, CC BY-SA 3.0);
> - *Practical Cheminformatics Tutorials* by **Pat Walters** — `ChEMBL_data_curation`, `working_with_ChEMBL_drug_data` ([GitHub](https://github.com/PatWalters/practical_cheminformatics_tutorials), MIT) and `chembl-downloader` by Charles Tapley Hoyt ([GitHub](https://github.com/cthoyt/chembl-downloader), MIT);
> - **OpenADMET** challenge tutorials and datasets ([GitHub](https://github.com/OpenADMET), Apache-2.0; ExpansionRx data CC BY 4.0);
> - the ChEMBL, PubChem and RCSB PDB API documentation.
"""

# %%
# @title ⚙️ Setup — run this cell first
import sys, os, subprocess, time, json
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "chembl_webresource_client", "pubchempy",
                    "mols2grid", "py3Dmol", "huggingface_hub", "datasets"], check=False)

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
from tqdm.auto import tqdm

REPO_RAW = "https://raw.githubusercontent.com/AxelRolov/chemoinformatics_tutorials/main"
def data_path(filename):
    local = os.path.join("..", "data", filename)
    return local if os.path.exists(local) else f"{REPO_RAW}/data/{filename}"

def online(url, timeout=20):
    """True if a URL answers (used to skip live queries gracefully when a service is down)."""
    try:
        return requests.head(url, timeout=timeout, allow_redirects=True).status_code < 500
    except requests.RequestException:
        return False

# %% [markdown]
"""
## 1. The landscape of chemical data

| resource | content | size (2026) | licence | access |
|---|---|---|---|---|
| [PubChem](https://pubchem.ncbi.nlm.nih.gov) (NIH) | compounds, substances, bioassays, patents, literature links | >120 M compounds | public domain | web, PUG-REST API |
| [ChEMBL](https://www.ebi.ac.uk/chembl) (EMBL-EBI) | *manually curated* bioactivities from medicinal-chemistry literature; drugs & mechanisms | 2.5 M compounds, 20 M activities, 16 k targets | CC BY-SA 3.0 | web, REST API, Python client, SQL dump |
| [ZINC](https://cartblanche22.docking.org) | purchasable compounds for virtual screening | billions (make-on-demand) | free | web, downloads |
| [DrugBank](https://go.drugbank.com) | drugs, targets, pharmacology | ~17 k drugs | CC BY-NC 4.0 (academic) | web, XML |
| [RCSB PDB](https://www.rcsb.org) | 3D structures of proteins, nucleic acids, complexes | >230 k entries | CC0 | web, REST, file downloads |
| [COCONUT](https://coconut.naturalproducts.net) | natural products | ~700 k | CC0 | web, API |
| [BindingDB](https://www.bindingdb.org) | binding affinities (Ki, Kd, IC50) | 3 M data points | CC BY 3.0 | web, downloads |
| [Hugging Face Hub](https://huggingface.co/datasets) | ML-ready datasets (OpenADMET, ChemBench, …) | – | per dataset | `datasets`, `pd.read_csv("hf://…")` |
| [MoleculeNet](https://moleculenet.org) / [TDC](https://tdcommons.ai) / [Polaris](https://polarishub.io) | benchmark datasets & splits for property prediction | – | mostly permissive | Python packages |

**Identifiers** you will keep meeting: CAS number (proprietary!), PubChem **CID**, **ChEMBL ID** (`CHEMBL25` = aspirin),
**InChIKey** (structure hash), **UniProt** accession (protein, `P00533` = human EGFR), **PDB ID** (`3POZ`).

Two golden rules: (1) *read the licence* before redistributing data; (2) *record the version/date* of what you downloaded
— databases change, and your results must stay reproducible.
"""

# %% [markdown]
"""
## 2. PubChem: from a name to a structure

PubChem's **PUG-REST** interface is a URL pattern: `…/compound/<input type>/<identifier>/<what you want>/<format>`.
Let's call it directly first, so the mechanism is not a black box, then use the `pubchempy` wrapper.
"""

# %%
PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PUBCHEM_UP = online(f"{PUBCHEM}/compound/name/aspirin/cids/JSON")
print("PubChem reachable:", PUBCHEM_UP)

def pubchem_properties(name, props=("MolecularFormula", "MolecularWeight", "SMILES", "XLogP", "InChIKey", "IUPACName")):
    """Look up a compound by name and return a dict of properties (PUG-REST)."""
    url = f"{PUBCHEM}/compound/name/{requests.utils.quote(name)}/property/{','.join(props)}/JSON"
    r = requests.get(url, timeout=30)
    if r.status_code == 400 and "SMILES" in props:                    # older servers: legacy property names
        props = tuple(p.replace("SMILES", "IsomericSMILES") for p in props)
        r = requests.get(f"{PUBCHEM}/compound/name/{requests.utils.quote(name)}/property/{','.join(props)}/JSON", timeout=30)
    r.raise_for_status()
    rec = r.json()["PropertyTable"]["Properties"][0]
    rec["SMILES"] = rec.get("SMILES", rec.get("IsomericSMILES"))
    return rec

if PUBCHEM_UP:
    rec = pubchem_properties("gefitinib")
    for k, v in rec.items():
        print(f"{k:16s} {v}")
    display(Chem.MolFromSmiles(rec["SMILES"]))
else:
    print("PubChem is offline right now – skipping the live query.")

# %%
# A small batch: several drugs at once (be polite: PubChem asks for ≤ 5 requests per second)
names = ["aspirin", "caffeine", "ibuprofen", "paracetamol", "metformin", "imatinib", "atorvastatin"]
if PUBCHEM_UP:
    rows = []
    for n in names:
        try:
            rows.append({"name": n, **pubchem_properties(n)})
        except Exception as e:
            print(n, "->", e)
        time.sleep(0.25)
    drugs = pd.DataFrame(rows)
    display(drugs[["name", "CID", "MolecularFormula", "MolecularWeight", "XLogP", "SMILES"]])

# %%
# The pubchempy wrapper does the same with Python objects
import pubchempy as pcp
if PUBCHEM_UP:
    c = pcp.get_compounds("imatinib", "name")[0]
    print(c.cid, c.molecular_formula, c.molecular_weight)
    print("synonyms:", pcp.get_synonyms("imatinib", "name")[0]["Synonym"][:8])

# %%
# Structure-based queries: 2D similarity search (Tanimoto on PubChem fingerprints) for gefitinib analogues
if PUBCHEM_UP:
    smi = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
    url = f"{PUBCHEM}/compound/fastsimilarity_2d/smiles/{requests.utils.quote(smi, safe='')}/cids/JSON?Threshold=90&MaxRecords=12"
    cids = requests.get(url, timeout=60).json()["IdentifierList"]["CID"]
    print(len(cids), "similar compounds:", cids)
    props = requests.get(f"{PUBCHEM}/compound/cid/{','.join(map(str, cids))}/property/SMILES,IUPACName/JSON", timeout=60)
    if props.status_code == 400:
        props = requests.get(f"{PUBCHEM}/compound/cid/{','.join(map(str, cids))}/property/IsomericSMILES,IUPACName/JSON", timeout=60)
    sim_df = pd.DataFrame(props.json()["PropertyTable"]["Properties"])
    sim_df["SMILES"] = sim_df.get("SMILES", sim_df.get("IsomericSMILES"))
    display(Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in sim_df["SMILES"].head(8)], molsPerRow=4,
                                 subImgSize=(200, 160), legends=[str(c) for c in sim_df["CID"].head(8)]))

# %% [markdown]
"""
### Exercise 2.1
Look up **five drugs of your choice** in PubChem and compute their Lipinski properties with RDKit (session 01).
Do PubChem's `XLogP` and RDKit's `MolLogP` agree? (They are different models of the same quantity.)
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
my_drugs = ["warfarin", "diazepam", "omeprazole", "simvastatin", "ciprofloxacin"]
rows = []
for n in my_drugs:
    rec = pubchem_properties(n)
    m = Chem.MolFromSmiles(rec["SMILES"])
    rows.append({"name": n, "XLogP (PubChem)": rec["XLogP"], "MolLogP (RDKit)": round(Descriptors.MolLogP(m), 2),
                 "MW": round(Descriptors.MolWt(m), 1)})
df5 = pd.DataFrame(rows)
print(df5)
print("correlation:", df5["XLogP (PubChem)"].corr(df5["MolLogP (RDKit)"]).round(2))
# They correlate strongly but differ by up to ~1 log unit: XLogP3 is a fitted atom/knowledge-based
# model, MolLogP is Crippen's atom-contribution model. Never mix logP values from different methods
# in one dataset without saying which method you used.
```
</details>
"""


# %% [markdown]
"""
## 3. ChEMBL: bioactivity data for a target

ChEMBL is *the* source of curated structure–activity data. Its data model, simplified:

- a **target** (protein, protein complex, cell line…) identified by a ChEMBL ID and linked to UniProt;
- an **assay** (an experiment described in a paper) with a type: **B**inding, **F**unctional, **A**DMET, **T**oxicity, **P**hysicochemical;
- an **activity**: one molecule tested in one assay → `standard_type` (IC50, Ki, EC50 …), `standard_relation` (=, <, >),
  `standard_value`, `standard_units`, and `pchembl_value` = −log₁₀(value in M) when available.

### IC50 and pIC50
IC50 is the concentration inhibiting 50 % of the activity. Because affinities span many orders of magnitude, we model
$\mathrm{pIC50} = -\log_{10}(\mathrm{IC50}\,[\mathrm{M}]) = 9 - \log_{10}(\mathrm{IC50}\,[\mathrm{nM}])$.
So 1 nM → pIC50 = 9 (very potent), 1 µM → 6, 100 µM → 4.
"""

# %%
CHEMBL_UP = online("https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL25.json")
print("ChEMBL reachable:", CHEMBL_UP)

# The REST API, by hand: one molecule record
if CHEMBL_UP:
    rec = requests.get("https://www.ebi.ac.uk/chembl/api/data/molecule/CHEMBL939.json", timeout=30).json()
    print(rec["pref_name"], "| max phase:", rec["max_phase"], "| first approval:", rec["first_approval"])
    print(rec["molecule_structures"]["canonical_smiles"])
    print("properties:", {k: rec["molecule_properties"][k] for k in ["full_mwt", "alogp", "hbd", "hba", "psa"]})

# %% [markdown]
"""
The official **Python client** (`chembl_webresource_client`) turns the same API into filter/only calls and handles paging and caching.
"""

# %%
uniprot_id = "P00533"        # human EGFR
if CHEMBL_UP:
    from chembl_webresource_client.new_client import new_client
    targets_api = new_client.target
    activities_api = new_client.activity
    molecules_api = new_client.molecule
    targets = pd.DataFrame(targets_api.filter(target_components__accession=uniprot_id)
                           .only("target_chembl_id", "organism", "pref_name", "target_type"))
    display(targets)
    target_id = targets.query("target_type == 'SINGLE PROTEIN'")["target_chembl_id"].iloc[0]
else:
    target_id = "CHEMBL203"
print("EGFR target:", target_id)

# %%
# Fetch IC50 binding-assay activities. EGFR has >10 000 IC50 records; we cap the download for the classroom.
MAX_RECORDS = 3000
if CHEMBL_UP:
    query = activities_api.filter(target_chembl_id=target_id, standard_type="IC50", relation="=", assay_type="B").only(
        "activity_id", "assay_chembl_id", "molecule_chembl_id", "canonical_smiles",
        "standard_type", "standard_relation", "standard_value", "standard_units", "pchembl_value", "document_year")
    records = []
    for i, rec in enumerate(tqdm(query, total=MAX_RECORDS)):
        records.append(rec)
        if i + 1 >= MAX_RECORDS:
            break
    acts = pd.DataFrame(records)
    acts.to_csv("EGFR_IC50_raw.csv", index=False)
    print(acts.shape)
    display(acts.head())

# %% [markdown]
"""
### From raw records to a modelling table

Downloaded data is never ready to use. Typical steps (we go deeper in session 04):
1. keep only rows with a value and the right units (nM);
2. convert to pIC50;
3. one molecule may have several measurements → aggregate (median);
4. drop molecules without a structure.
"""

# %%
if CHEMBL_UP:
    df = acts.dropna(subset=["standard_value", "canonical_smiles"]).copy()
    df["standard_value"] = df["standard_value"].astype(float)
    print("units present:", df["standard_units"].value_counts().to_dict())
    df = df[df["standard_units"] == "nM"]
    df["pIC50"] = 9 - np.log10(df["standard_value"])
    egfr = (df.groupby("molecule_chembl_id")
              .agg(smiles=("canonical_smiles", "first"), pIC50=("pIC50", "median"), n_measurements=("pIC50", "size"))
              .reset_index())
    print(len(egfr), "unique molecules;", (egfr["n_measurements"] > 1).sum(), "with replicate measurements")
else:
    # Cached dataset produced by TeachOpenCADD T001 (5568 EGFR compounds with IC50, ChEMBL 27)
    egfr = pd.read_csv(data_path("EGFR_compounds_chembl.csv"), index_col=0).rename(columns={"IC50": "IC50_nM"})
    print("Using the cached EGFR dataset:", egfr.shape)
egfr.head()

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
egfr["pIC50"].hist(bins=40, ax=axes[0]); axes[0].set_xlabel("pIC50"); axes[0].set_ylabel("molecules")
axes[0].axvline(6.3, c="r", ls="--", label="500 nM"); axes[0].legend()
egfr["mol"] = egfr["smiles"].apply(Chem.MolFromSmiles)
egfr["MW"] = egfr["mol"].apply(Descriptors.MolWt)
axes[1].scatter(egfr["MW"], egfr["pIC50"], s=5, alpha=0.3); axes[1].set_xlabel("MW"); axes[1].set_ylabel("pIC50")
plt.tight_layout(); plt.show()

# %%
# The most potent compounds - do you recognise the quinazoline scaffold of gefitinib/erlotinib?
top = egfr.sort_values("pIC50", ascending=False).head(6)
Draw.MolsToGridImage(top["mol"].tolist(), molsPerRow=3, subImgSize=(230, 170),
                     legends=[f"{i}  pIC50={p:.1f}" for i, p in zip(top["molecule_chembl_id"], top["pIC50"])])

# %%
egfr.drop(columns=["mol"]).to_csv("EGFR_pIC50.csv", index=False)
print("saved EGFR_pIC50.csv - we reuse this file in sessions 04–05")

# %% [markdown]
"""
### Drugs and mechanisms
ChEMBL also knows which molecules are **approved drugs** and *why they work* (mechanism of action, target).
"""

# %%
if CHEMBL_UP:
    mech = pd.DataFrame(new_client.mechanism.filter(target_chembl_id=target_id)
                        .only("molecule_chembl_id", "mechanism_of_action", "action_type", "max_phase"))
    names = pd.DataFrame(molecules_api.filter(molecule_chembl_id__in=list(mech["molecule_chembl_id"]))
                         .only("molecule_chembl_id", "pref_name", "first_approval"))
    display(mech.merge(names, on="molecule_chembl_id").sort_values("first_approval").head(15))

# %% [markdown]
"""
> **Alternatives.** For large or reproducible extractions, download the whole ChEMBL SQLite dump (~5 GB) and query it
> with SQL — `chembl_downloader` (C. T. Hoyt) automates this and is used in Pat Walters' *ChEMBL data curation* notebook.

### Exercise 3.1
Build the same table for another kinase, e.g. **ABL1** (UniProt `P00519`), or for the anti-target **hERG** (`Q12809`, use `MAX_RECORDS=2000`).
How many unique molecules do you get? What is the median pIC50?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
def chembl_dataset(uniprot_id, max_records=2000, standard_type="IC50"):
    t = pd.DataFrame(targets_api.filter(target_components__accession=uniprot_id)
                     .only("target_chembl_id", "target_type", "pref_name"))
    tid = t.query("target_type == 'SINGLE PROTEIN'")["target_chembl_id"].iloc[0]
    q = activities_api.filter(target_chembl_id=tid, standard_type=standard_type, relation="=", assay_type="B").only(
        "molecule_chembl_id", "canonical_smiles", "standard_value", "standard_units")
    recs = []
    for i, r in enumerate(tqdm(q, total=max_records)):
        recs.append(r)
        if i + 1 >= max_records: break
    d = pd.DataFrame(recs).dropna(subset=["standard_value", "canonical_smiles"])
    d = d[d["standard_units"] == "nM"].copy()
    d["pIC50"] = 9 - np.log10(d["standard_value"].astype(float))
    out = d.groupby("molecule_chembl_id").agg(smiles=("canonical_smiles", "first"), pIC50=("pIC50", "median")).reset_index()
    print(f"{uniprot_id}: {len(out)} unique molecules, median pIC50 {out['pIC50'].median():.2f}")
    return out

if CHEMBL_UP:
    abl1 = chembl_dataset("P00519")      # ABL1
    herg = chembl_dataset("Q12809")      # hERG
```
</details>
"""


# %% [markdown]
"""
## 4. Open ML datasets: Hugging Face Hub, MoleculeNet, TDC, Polaris

Increasingly, curated datasets are published directly in machine-learning-ready form. The **Hugging Face Hub** hosts
thousands of them and pandas can read them with an `hf://` path. Example: the **OpenADMET × ExpansionRx challenge**
training set — 5 300 real drug-discovery compounds with nine ADMET endpoints (LogD, kinetic solubility, microsomal
clearance, Caco-2 permeability, protein binding), released under CC BY 4.0 in 2026.
"""

# %%
HF_FILE = "hf://datasets/openadmet/openadmet-expansionrx-challenge-train-data/expansion_data_train.csv"
try:
    admet = pd.read_csv(HF_FILE)
    HF_UP = True
except Exception as e:
    HF_UP = False
    print("Hugging Face not reachable:", str(e)[:120])

if HF_UP:
    print(admet.shape)
    display(admet.head())
    print("\nmeasurements per endpoint:")
    print(admet.drop(columns=["Molecule Name", "SMILES"]).notna().sum())

# %%
if HF_UP:
    endpoints = [c for c in admet.columns if c not in ("Molecule Name", "SMILES")]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for ax, col in zip(axes.ravel(), endpoints):
        vals = admet[col].dropna()
        if vals.min() > 0 and vals.max() / max(vals.min(), 1e-9) > 100:
            ax.hist(np.log10(vals), bins=30); ax.set_xlabel(f"log10 {col}")
        else:
            ax.hist(vals, bins=30); ax.set_xlabel(col)
    plt.tight_layout(); plt.show()

# %% [markdown]
"""
Note how **sparse** real data is: most molecules were measured on only a few endpoints, and clearance values span three
orders of magnitude (hence the log scale). Both facts matter when we model (sessions 05–06).

The `datasets` library gives the same data with streaming, splits and versioning:

```python
from datasets import load_dataset
ds = load_dataset("openadmet/openadmet-expansionrx-challenge-train-data")
```

Other places to find benchmark data:
- **MoleculeNet** (ESOL, FreeSolv, Lipophilicity, BBBP, Tox21, HIV, …) — via `deepchem` or the CSVs on GitHub (we used ESOL in session 00);
- **Therapeutics Data Commons** (`pip install PyTDC`): `from tdc.single_pred import ADME; ADME(name="Caco2_Wang").get_data()`;
- **Polaris** (`polarishub.io`): curated benchmarks with fixed splits and leaderboards.
"""

# %% [markdown]
"""
## 5. The Protein Data Bank: structures of targets and complexes

The RCSB **Data API** returns metadata as JSON; files are downloaded from `files.rcsb.org`. Let's fetch **3POZ**,
the EGFR kinase domain bound to an inhibitor — the structure used in TeachOpenCADD's MD talktorial.
"""

# %%
PDB_UP = online("https://data.rcsb.org/rest/v1/core/entry/3POZ")
print("RCSB PDB reachable:", PDB_UP)
pdb_id = "3POZ"
if PDB_UP:
    entry = requests.get(f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}", timeout=30).json()
    print(entry["struct"]["title"])
    print("method:", entry["exptl"][0]["method"], "| resolution (Å):", entry["rcsb_entry_info"].get("resolution_combined"))
    print("released:", entry["rcsb_accession_info"]["initial_release_date"][:10])
    ligand_ids = entry["rcsb_entry_container_identifiers"].get("non_polymer_entity_ids", [])
    print("non-polymer entities:", ligand_ids)

# %%
if PDB_UP:
    # Which ligands are in the entry, and what are their structures?
    for eid in ligand_ids:
        ent = requests.get(f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{eid}", timeout=30).json()
        comp_id = ent["pdbx_entity_nonpoly"]["comp_id"]
        comp = requests.get(f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}", timeout=30).json()
        smiles = comp["rcsb_chem_comp_descriptor"]["smiles"]
        print(comp_id, "|", comp["chem_comp"]["name"][:60], "|", smiles)
        if comp["chem_comp"]["type"] != "non-polymer" or comp_id in ("HOH",):
            continue
        display(Chem.MolFromSmiles(smiles))

# %%
if PDB_UP:
    pdb_text = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=60).text
    open(f"{pdb_id}.pdb", "w").write(pdb_text)
    import py3Dmol
    view = py3Dmol.view(width=600, height=400)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.addStyle({"hetflag": True, "not": {"resn": "HOH"}}, {"stick": {}})
    view.zoomTo({"hetflag": True, "not": {"resn": "HOH"}}); view.show()

# %% [markdown]
"""
### Exercise 5.1
Use the RCSB **search API** (`https://search.rcsb.org/rcsbsearch/v2/query`) or simply the website to find how many PDB
entries exist for human EGFR (UniProt P00533). Pick one with a *different* inhibitor, download it, and draw the ligand.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
query = {
    "query": {"type": "terminal", "service": "text",
              "parameters": {"attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                             "operator": "exact_match", "value": "P00533"}},
    "return_type": "entry", "request_options": {"paginate": {"start": 0, "rows": 10}},
}
r = requests.post("https://search.rcsb.org/rcsbsearch/v2/query", json=query, timeout=30).json()
print(r["total_count"], "EGFR entries; first ten:", [x["identifier"] for x in r["result_set"]])

# then repeat the ligand-extraction code above with one of those PDB IDs, e.g. "1M17" (erlotinib) or "4WKQ" (gefitinib)
```
</details>
"""


# %% [markdown]
"""
## 6. Summary

- **PubChem** answers "what is this compound?" (identity, properties, vendors, patents) for >100 M compounds.
- **ChEMBL** answers "what does it do?" — curated bioactivities you can turn into QSAR datasets after cleaning.
- The **PDB** answers "what does the target look like, and how does the ligand bind?"
- **Hub-hosted datasets** (Hugging Face, TDC, Polaris) give ML-ready tables with known splits — use them to *benchmark*, not to replace understanding the raw data.

Every dataset we build today raises the questions of **session 04**: are the structures standardised? Are there duplicates
or contradictory measurements? Which scaffolds dominate? Where does the data live in chemical space?

## Further reading
- Zdrazil *et al.*, *The ChEMBL Database in 2023*, Nucleic Acids Res. **2024**; ChEMBL API docs: <https://chembl.gitbook.io/chembl-interface-documentation/web-services>.
- Kim *et al.*, *PubChem 2025 update*, Nucleic Acids Res. **2025**; PUG-REST tutorial: <https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest-tutorial>.
- Burley *et al.*, *RCSB Protein Data Bank*, Nucleic Acids Res. **2025**; RCSB API: <https://data.rcsb.org>.
- Walters, *ChEMBL data curation* notebook, and Landrum & Riniker, *Combining IC50 or Ki values from different sources is a source of significant noise*, J. Chem. Inf. Model. **2024**, 64, 1560.

Next session: **04 · Exploratory data analysis** — standardisation, clustering, scaffolds and chemical space.
"""
