# %% [markdown]
"""
# 02 · Molecular representations: from structures to numbers

**Chemoinformatics practicals — Session 2 of 11**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

**Learning goals.** After this session you will be able to
- name the main families of molecular representations (line notations, graphs, descriptors, fingerprints, 3D) and explain what each one captures and loses;
- convert between SMILES, InChI, InChIKey and SELFIES, and explain why SELFIES are "robust";
- build the **graph** (adjacency matrix + atom features) that graph neural networks consume;
- compute **descriptors** and **fingerprints** (MACCS, Morgan/ECFP, atom pairs) and understand their parameters;
- quantify molecular **similarity** with the Tanimoto coefficient and run a nearest-neighbour search;
- generate a 3D **conformer** and compute shape descriptors.

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - **TeachOpenCADD** talktorial **T033 · Molecular representations** (Volkamer lab; [GitHub](https://github.com/volkamerlab/teachopencadd), CC BY 4.0) — Sydow *et al.*, *J. Chem. Inf. Model.* 2019 and *Nucleic Acids Res.* 2022;
> - *Practical Cheminformatics Tutorials* by **Pat Walters** — fingerprint and similarity material ([GitHub](https://github.com/PatWalters/practical_cheminformatics_tutorials), MIT); the approved-drug list is from his [datafiles](https://github.com/PatWalters/datafiles) (ChEMBL, CC BY-SA 3.0);
> - **MolSSI** cheminformatics workshop — `03_molecular_similarity` ([GitHub](https://github.com/MolSSI-Education/molssi-cheminformatics), MIT);
> - A. D. White, *Deep Learning for Molecules and Materials* ([dmol.pub](https://dmol.pub), CC BY-NC 3.0) — the representation taxonomy;
> - the [RDKit fingerprint documentation](https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity) and Greg Landrum's [RDKit blog](https://greglandrum.github.io/rdkit-blog/).
"""

# %%
# @title ⚙️ Setup — run this cell first
import sys, os, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "selfies", "mols2grid", "py3Dmol", "networkx"], check=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem, rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
IPythonConsole.molSize = (300, 220)

REPO_RAW = "https://raw.githubusercontent.com/AxelRolov/chemoinformatics_tutorials/main"
def data_path(filename):
    local = os.path.join("..", "data", filename)
    return local if os.path.exists(local) else f"{REPO_RAW}/data/{filename}"

drugs = pd.read_csv(data_path("chembl_drugs_walters.smi"), sep=" ", names=["SMILES", "ChEMBL_ID"])
drugs["mol"] = drugs["SMILES"].apply(Chem.MolFromSmiles)
print(len(drugs), "approved drugs loaded")

# %% [markdown]
"""
## 1. Why do we need representations?

A molecule is a three-dimensional, flexible, quantum-mechanical object. A computer (and a machine-learning model)
needs a **finite list of numbers or symbols**. Every representation is a *choice* about what to keep:

| family | example | keeps | loses | typical use |
|---|---|---|---|---|
| **names & formulas** | caffeine, C₈H₁₀N₄O₂ | identity | almost everything | communication |
| **line notations** | SMILES, InChI, SELFIES | full 2D connectivity (+ stereo) | 3D, conformations | storage, databases, language models |
| **molecular graph** | atoms = nodes, bonds = edges | connectivity + atom/bond features | 3D | graph neural networks |
| **descriptors** | MW, logP, TPSA, number of rings… | interpretable global properties | local detail | QSAR, filtering |
| **fingerprints** | MACCS, Morgan/ECFP | presence of substructures as a bit vector | exact topology, 3D | similarity search, ML |
| **3D representations** | conformers, point clouds, surfaces, voxels | geometry, shape, electrostatics | (need conformer sampling) | docking, 3D-QSAR, MD |
| **learned embeddings** | vectors from a neural network | whatever the training task needs | interpretability | modern ML |

Today we work through the first six rows; learned representations come in sessions 06–07.
"""

# %% [markdown]
"""
## 2. Line notations: SMILES, InChI, InChIKey, SELFIES

You already know SMILES. Two other standards are everywhere in databases:

- **InChI** (IUPAC International Chemical Identifier) is a *layered*, canonical string: formula / connectivity / H atoms / charge / stereo / isotopes. Same molecule ⇒ same InChI, whatever software produced it.
- **InChIKey** is a 27-character hash of the InChI — fixed length, easy to index and to search for on the web. The first 14 characters encode the skeleton, the next 8 the stereo/isotope layers.
"""

# %%
caffeine = Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
print("SMILES  :", Chem.MolToSmiles(caffeine))
print("InChI   :", Chem.MolToInchi(caffeine))
print("InChIKey:", Chem.MolToInchiKey(caffeine))
print("Formula :", rdMolDescriptors.CalcMolFormula(caffeine))

# %%
# Stereoisomers share the first block of the InChIKey but differ in the second
for smi in ["C[C@H](N)C(=O)O", "C[C@@H](N)C(=O)O", "CC(N)C(=O)O"]:
    m = Chem.MolFromSmiles(smi)
    print(f"{smi:20s} {Chem.MolToInchiKey(m)}")

# %% [markdown]
"""
### SELFIES: a robust alternative to SMILES

Generative models (session 07) write molecules character by character. With SMILES, a single wrong character
often gives an **invalid** molecule (unclosed ring, wrong valence). **SELFIES** (SELF-referencIng Embedded Strings,
Krenn *et al.* 2020) were designed so that *every* string decodes to a valid molecule. Let's test that claim.
"""

# %%
import selfies as sf
smi = "CC(=O)Oc1ccccc1C(=O)O"          # aspirin
sel = sf.encoder(smi)
print("SMILES :", smi)
print("SELFIES:", sel)
print("tokens :", list(sf.split_selfies(sel)))
print("back   :", sf.decoder(sel))

# %%
import random
random.seed(0)

def mutate_string(tokens, alphabet, n_mut=1):
    """Replace n_mut random tokens by random tokens of the alphabet."""
    tokens = list(tokens)
    for _ in range(n_mut):
        tokens[random.randrange(len(tokens))] = random.choice(alphabet)
    return tokens

smiles_alphabet = list("CNOSFClBr()=#123456[]@+-cnos")
selfies_alphabet = list(sf.get_semantic_robust_alphabet())

def validity_after_mutation(n_trials=300, n_mut=2):
    ok_smiles = ok_selfies = 0
    for _ in range(n_trials):
        base = drugs["SMILES"].sample(1).iloc[0]
        # SMILES: mutate characters
        mutated = "".join(mutate_string(base, smiles_alphabet, n_mut))
        ok_smiles += Chem.MolFromSmiles(mutated) is not None
        # SELFIES: mutate tokens then decode
        try:
            toks = list(sf.split_selfies(sf.encoder(base)))
            mutated_sel = "".join(mutate_string(toks, selfies_alphabet, n_mut))
            ok_selfies += Chem.MolFromSmiles(sf.decoder(mutated_sel)) is not None
        except Exception:
            pass
    return ok_smiles / n_trials, ok_selfies / n_trials

from rdkit import RDLogger; RDLogger.DisableLog("rdApp.*")   # silence parse warnings
v_smi, v_sel = validity_after_mutation()
RDLogger.EnableLog("rdApp.*")
print(f"Validity after 2 random mutations:  SMILES {v_smi:.0%}   SELFIES {v_sel:.0%}")

# %% [markdown]
"""
> **Take-away.** Line notations are compact and exact, but they are *strings*, not numbers: a model must first learn
> the grammar. SELFIES remove the "invalid molecule" problem at the price of readability.
"""

# %% [markdown]
"""
## 3. The molecular graph

Chemists have always drawn molecules as graphs. For a computer, a graph is an **adjacency matrix** $A$
($A_{ij}=1$ if atoms $i$ and $j$ are bonded) plus a **feature matrix** $X$ (one row per atom: element, charge,
aromaticity, …). This is exactly the input of a **graph neural network** (session 06).
"""

# %%
IPythonConsole.drawOptions.addAtomIndices = True
paracetamol = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
paracetamol

# %%
A = Chem.GetAdjacencyMatrix(paracetamol)
print(A.shape)
print(A)

# %%
# Bond orders can be kept in a weighted adjacency matrix
Chem.GetAdjacencyMatrix(paracetamol, useBO=True)

# %%
# Node features: one-hot element + a few numeric properties
ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "other"]

def atom_features(atom):
    onehot = [int(atom.GetSymbol() == e) for e in ELEMENTS[:-1]]
    onehot.append(int(atom.GetSymbol() not in ELEMENTS[:-1]))
    return onehot + [atom.GetDegree(), atom.GetTotalNumHs(), atom.GetFormalCharge(), int(atom.GetIsAromatic())]

X = np.array([atom_features(a) for a in paracetamol.GetAtoms()])
pd.DataFrame(X, columns=ELEMENTS + ["degree", "numH", "charge", "aromatic"])

# %%
# Edge list ("COO" format used by PyTorch Geometric): pairs of atom indices, both directions
edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in paracetamol.GetBonds()]
edge_index = np.array(edges + [(j, i) for i, j in edges]).T
print(edge_index.shape)
print(edge_index)

# %%
import networkx as nx
G = nx.from_numpy_array(A)
labels = {a.GetIdx(): a.GetSymbol() for a in paracetamol.GetAtoms()}
colors = ["lightgray" if s == "C" else "skyblue" if s == "N" else "salmon" for s in labels.values()]
plt.figure(figsize=(4, 3.5))
nx.draw(G, labels=labels, node_color=colors, with_labels=True, node_size=500, font_size=9)
plt.title("paracetamol as a graph"); plt.show()
IPythonConsole.drawOptions.addAtomIndices = False

# %% [markdown]
"""
> **Graph invariance.** Renumber the atoms and you get a different $A$ and $X$ — but the *same* molecule.
> Descriptors and fingerprints (below) are designed to be **invariant** to atom ordering; graph neural networks
> achieve the same by construction (permutation-equivariant message passing).
"""

# %% [markdown]
"""
## 4. Molecular descriptors

A **descriptor** is a number computed from the structure. RDKit ships ~200 2D descriptors in several families:

- *constitutional*: MW, heavy-atom count, fraction of sp³ carbons (`FractionCSP3`), ring counts;
- *physico-chemical estimates*: `MolLogP` (Crippen), `TPSA`, molar refractivity;
- *topological indices*: Balaban J, Kier & Hall χ and κ indices, Bertz complexity;
- *electronic/charge-based*: partial-charge extrema, VSA descriptors (`PEOE_VSA*`, `SMR_VSA*`, `SlogP_VSA*`);
- *fragment counts*: `fr_*` (number of halogens, amides, aromatic rings …).

We compute them for all approved drugs.
"""

# %%
from rdkit.Chem import Descriptors
desc_names = [n for n, _ in Descriptors.descList]
print(len(desc_names), "descriptors, e.g.:", desc_names[:8], "...")

def all_descriptors(mol):
    return Descriptors.CalcMolDescriptors(mol)

desc_df = pd.DataFrame([all_descriptors(m) for m in drugs["mol"]])
desc_df.index = drugs["ChEMBL_ID"]
desc_df.iloc[:5, :8]

# %%
# Some descriptors are nearly constant or heavily correlated: a first look
subset = ["MolWt", "HeavyAtomCount", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors",
          "NumRotatableBonds", "RingCount", "NumAromaticRings", "FractionCSP3", "BertzCT", "qed"]
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(desc_df[subset].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax, annot_kws={"size": 7})
ax.set_title("Pearson correlation between descriptors (approved drugs)")
plt.show()

# %% [markdown]
"""
`qed` (Quantitative Estimate of Drug-likeness, Bickerton 2012) is itself a *composite* of 8 descriptors — a reminder
that descriptors are models too, with their own assumptions.

### Exercise 4.1
Which descriptor in `desc_df` is most strongly correlated (positively or negatively) with `MolLogP`? Use `.corr()` and sort.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
corr = desc_df.corr(numeric_only=True)["MolLogP"].drop("MolLogP").dropna()
print(corr.abs().sort_values(ascending=False).head(10))
```
</details>
"""

# %% [markdown]
"""
## 5. Fingerprints

A **fingerprint** encodes a molecule as a long binary (or count) vector: each position answers "is this feature present?".

- **MACCS keys** (166 bits): a fixed dictionary of substructures (e.g. "has a S–S bond", "≥ 2 aromatic rings"). Interpretable, coarse.
- **Morgan / ECFP** (Extended-Connectivity FingerPrints): for every atom, hash its neighbourhood up to a **radius** $r$
  (ECFP4 ⇔ radius 2). The hashed identifiers are folded into a vector of fixed size (typically 1024 or 2048 bits).
  Two different features can land on the same bit — a **collision**. Radius and size are the two knobs.
- **Feature Morgan / FCFP**: the same, but atoms are described by pharmacophoric role (donor, acceptor, aromatic…) instead of element.
- **Atom pairs / topological torsions**: pairs of atoms with their topological distance; sequences of 4 bonded atoms.
- **RDKit topological fingerprint** (Daylight-like): hashed linear paths of 1–7 bonds.
"""

# %%
aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")

maccs = MACCSkeys.GenMACCSKeys(aspirin)
print("MACCS  :", maccs.GetNumBits(), "bits,", maccs.GetNumOnBits(), "on")

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
morgan = mfpgen.GetFingerprint(aspirin)
print("Morgan :", morgan.GetNumBits(), "bits,", morgan.GetNumOnBits(), "on")
print("on bits:", list(morgan.GetOnBits())[:12], "...")

# %%
# Fingerprints as numpy arrays (what scikit-learn wants)
arr = mfpgen.GetFingerprintAsNumPy(aspirin)
print(arr.shape, arr.dtype, arr.sum())

# Count fingerprints keep how many times a feature occurs
counts = mfpgen.GetCountFingerprintAsNumPy(aspirin)
print("max count:", counts.max())

# %%
# Which substructure does a bit correspond to? RDKit can tell you ("bit info").
ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
fp = mfpgen.GetFingerprint(aspirin, additionalOutput=ao)
bit_info = ao.GetBitInfoMap()
some_bits = list(bit_info.keys())[:6]
Draw.DrawMorganBits([(aspirin, b, bit_info) for b in some_bits], molsPerRow=3,
                    legends=[f"bit {b} (radius {bit_info[b][0][1]})" for b in some_bits])

# %%
# The effect of radius and size: how many bits are set, how many collisions?
rows = []
for radius in [1, 2, 3]:
    for size in [512, 1024, 2048, 4096]:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size)
        n_on = [gen.GetFingerprint(m).GetNumOnBits() for m in drugs["mol"]]
        # number of distinct unfolded features vs. folded bits -> collisions
        n_feat = [len(gen.GetSparseCountFingerprint(m).GetNonzeroElements()) for m in drugs["mol"]]
        rows.append({"radius": radius, "size": size, "mean bits on": np.mean(n_on),
                     "mean features": np.mean(n_feat), "mean collisions": np.mean(np.array(n_feat) - np.array(n_on))})
pd.DataFrame(rows).round(1)

# %% [markdown]
"""
### Exercise 5.1
Compute the MACCS, Morgan (r=2, 2048) and *feature* Morgan fingerprint of paracetamol.
For the feature Morgan generator use `rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())`.
How many bits are on in each?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
fcfp_gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen())
for name, fp in [("MACCS", MACCSkeys.GenMACCSKeys(paracetamol)),
                 ("ECFP4", mfpgen.GetFingerprint(paracetamol)),
                 ("FCFP4", fcfp_gen.GetFingerprint(paracetamol))]:
    print(name, fp.GetNumOnBits())
```
</details>
"""

# %% [markdown]
"""
## 6. Molecular similarity

With fingerprints, similarity becomes set overlap. For bit vectors $a$ and $b$ the **Tanimoto** (Jaccard) coefficient is

$$T(a,b) = \frac{|a \cap b|}{|a \cup b|} = \frac{c}{n_a + n_b - c}$$

where $c$ is the number of bits on in both. $T=1$ for identical fingerprints, $0$ when they share nothing.
The **Dice** coefficient $2c/(n_a+n_b)$ is a common alternative. Rules of thumb (ECFP4): $T>0.7$ "very similar",
$0.4$–$0.7$ "related", $<0.3$ "different" — but always calibrate on your own data (session 04).

The **similarity principle** (Johnson & Maggiora, 1990): similar molecules *tend* to have similar properties.
It underlies virtual screening, clustering, and every nearest-neighbour model. Its exceptions are called **activity cliffs**.
"""

# %%
pairs = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "salicylic acid": "OC(=O)c1ccccc1O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
}
names = list(pairs)
fps = {n: mfpgen.GetFingerprint(Chem.MolFromSmiles(s)) for n, s in pairs.items()}
sim = pd.DataFrame([[DataStructs.TanimotoSimilarity(fps[a], fps[b]) for b in names] for a in names],
                   index=names, columns=names)
sim.round(2)

# %%
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in pairs.values()], legends=names, molsPerRow=5, subImgSize=(180, 150))

# %%
# Different fingerprints give different numbers - similarity is *relative to a representation*
fp_types = {
    "MACCS": lambda m: MACCSkeys.GenMACCSKeys(m),
    "ECFP4 (r=2)": lambda m: mfpgen.GetFingerprint(m),
    "ECFP2 (r=1)": lambda m: rdFingerprintGenerator.GetMorganGenerator(radius=1, fpSize=2048).GetFingerprint(m),
    "AtomPair": lambda m: rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048).GetFingerprint(m),
    "RDKit topological": lambda m: rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048).GetFingerprint(m),
}
a, b = Chem.MolFromSmiles(pairs["aspirin"]), Chem.MolFromSmiles(pairs["salicylic acid"])
for name, f in fp_types.items():
    print(f"{name:18s} Tanimoto(aspirin, salicylic acid) = {DataStructs.TanimotoSimilarity(f(a), f(b)):.2f}")

# %% [markdown]
"""
### Nearest-neighbour search in the drug set

`BulkTanimotoSimilarity` compares one query against a list of fingerprints very fast (millions per second).
"""

# %%
drug_fps = [mfpgen.GetFingerprint(m) for m in drugs["mol"]]

def nearest_neighbours(query_smiles, k=6):
    q = mfpgen.GetFingerprint(Chem.MolFromSmiles(query_smiles))
    sims = np.array(DataStructs.BulkTanimotoSimilarity(q, drug_fps))
    idx = np.argsort(-sims)[:k]
    hits = drugs.iloc[idx].copy()
    hits["similarity"] = sims[idx]
    return hits

hits = nearest_neighbours("CC(C)Cc1ccc(C(C)C(=O)O)cc1")     # ibuprofen
Draw.MolsToGridImage(hits["mol"].tolist(), molsPerRow=3, subImgSize=(220, 170),
                     legends=[f"{i}  T={s:.2f}" for i, s in zip(hits["ChEMBL_ID"], hits["similarity"])])

# %%
# What does a "random" similarity look like? The distribution over all drug pairs tells us what T is *meaningful*.
n = len(drug_fps)
all_sims = []
for i in range(n):
    all_sims.extend(DataStructs.BulkTanimotoSimilarity(drug_fps[i], drug_fps[i + 1:]))
all_sims = np.array(all_sims)
plt.figure(figsize=(6, 3.2))
plt.hist(all_sims, bins=60, log=True)
plt.xlabel("Tanimoto similarity (ECFP4)"); plt.ylabel("number of drug pairs (log)")
plt.title(f"{len(all_sims):,} pairs; 99th percentile = {np.percentile(all_sims, 99):.2f}")
plt.show()

# %% [markdown]
"""
> Most random pairs of drugs have $T<0.3$. A hit with $T=0.6$ is therefore genuinely unusual — that is why the
> thresholds above work. Note that MACCS similarities are systematically higher (fewer, denser bits), so a threshold
> for one fingerprint does not transfer to another.
"""

# %% [markdown]
"""
### Exercise 6.1
1. Find the 5 approved drugs most similar to **caffeine** (ECFP4). Do you recognise them?
2. Repeat with MACCS keys. Are the neighbours the same?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
hits = nearest_neighbours(pairs["caffeine"], k=5)
display(Draw.MolsToGridImage(hits["mol"].tolist(), legends=[f"{i} T={s:.2f}" for i, s in zip(hits["ChEMBL_ID"], hits["similarity"])]))

maccs_fps = [MACCSkeys.GenMACCSKeys(m) for m in drugs["mol"]]
q = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(pairs["caffeine"]))
sims = np.array(DataStructs.BulkTanimotoSimilarity(q, maccs_fps))
idx = np.argsort(-sims)[:5]
display(Draw.MolsToGridImage(drugs["mol"].iloc[idx].tolist(), legends=[f"{i} T={s:.2f}" for i, s in zip(drugs["ChEMBL_ID"].iloc[idx], sims[idx])]))
```
</details>
"""

# %% [markdown]
"""
## 7. Into the third dimension

Everything so far was 2D. Many properties (binding to a protein, crystal packing, spectroscopy) depend on **3D shape**.
RDKit can generate plausible 3D **conformers** with the ETKDG algorithm (distance geometry + experimental torsion
preferences) and refine them with a force field (MMFF94). We look at this properly in session 10; here is a first taste.
"""

# %%
mol3d = Chem.AddHs(Chem.MolFromSmiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1"))    # ibuprofen with explicit H
params = AllChem.ETKDGv3(); params.randomSeed = 42
AllChem.EmbedMolecule(mol3d, params)
AllChem.MMFFOptimizeMolecule(mol3d)
conf = mol3d.GetConformer()
coords = conf.GetPositions()
print("coordinates of the first 5 atoms (Å):\n", np.round(coords[:5], 3))

# %%
import py3Dmol
view = py3Dmol.view(width=400, height=300)
view.addModel(Chem.MolToMolBlock(mol3d), "mol")
view.setStyle({"stick": {}})
view.zoomTo()
view.show()

# %% [markdown]
"""
### 3D shape descriptors: where do drugs live in shape space?

The **normalised principal moments of inertia** (NPR1 = I₁/I₃, NPR2 = I₂/I₃; Sauer & Schwarz 2003) place any molecule
inside a triangle whose corners are *rod* (0,1), *disc* (0.5,0.5) and *sphere* (1,1). Let's compute them for 300 drugs.
"""

# %%
from rdkit.Chem import rdMolDescriptors as rdmd

def npr_from_smiles(smi, seed=0):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3(); p.randomSeed = seed
    if AllChem.EmbedMolecule(m, p) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    return rdmd.CalcNPR1(m), rdmd.CalcNPR2(m)

sample = drugs.sample(300, random_state=0)
npr = [npr_from_smiles(s) for s in sample["SMILES"]]
npr = np.array([x for x in npr if x is not None])

plt.figure(figsize=(5, 4.5))
plt.plot([0, 0.5, 1, 0], [1, 0.5, 1, 1], "k-", lw=1)
plt.scatter(npr[:, 0], npr[:, 1], s=12, alpha=0.6)
plt.text(0, 1.02, "rod"); plt.text(0.5, 0.46, "disc"); plt.text(0.97, 1.02, "sphere")
plt.xlabel("NPR1"); plt.ylabel("NPR2"); plt.title("Shape space of approved drugs")
plt.show()

# %% [markdown]
"""
Most drugs sit near the rod–disc edge: flat, elongated molecules. Truly spherical drugs are rare — a fact that has
driven interest in "escaping flatland" (Lovering *et al.* 2009, `FractionCSP3`).

## 8. Summary and exercises

| representation | Python object | size | invariant to atom order? | reversible? |
|---|---|---|---|---|
| SMILES / SELFIES | `str` | variable | canonical form: yes | yes |
| InChIKey | `str` | 27 chars | yes | no (hash) |
| graph (A, X) | `np.ndarray` | N×N, N×F | no (equivariant) | yes |
| descriptors | `np.ndarray` | ~200 | yes | no |
| fingerprint | bit vector | 166–4096 | yes | no (lossy hash) |
| conformer | N×3 coordinates | 3N | no (rotations!) | yes |

**Exercises**
1. Compute Morgan fingerprints (r=2, 1024 bits) for all drugs as a numpy matrix `X` of shape (1203, 1024). What fraction of the entries are 1? Why is this called a *sparse* representation?
2. For 20 random drugs, compute pairwise Tanimoto with ECFP4 and with MACCS; make a scatter plot of one against the other. Are they linearly related?
3. Pick two approved drugs you know to have the *same* indication (e.g. two β-blockers) and two with different indications. Do their Tanimoto similarities follow the similarity principle?
4. (Harder) The **Murcko scaffold** of a molecule keeps only its ring systems and the linkers between them (`from rdkit.Chem.Scaffolds import MurckoScaffold; MurckoScaffold.GetScaffoldForMol(mol)`). What are the 5 most frequent scaffolds among approved drugs?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solutions (1 and 4)</b></summary>

```python
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
X = np.array([gen.GetFingerprintAsNumPy(m) for m in drugs["mol"]])
print(X.shape, X.mean())            # ~3 % of bits are on

from rdkit.Chem.Scaffolds import MurckoScaffold
scaf = drugs["mol"].apply(lambda m: Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)))
top = scaf.value_counts().head(5)
print(top)
display(Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in top.index], legends=[str(c) for c in top.values]))
```
</details>
"""

# %% [markdown]
"""
## Further reading

- D. Rogers, M. Hahn, *Extended-Connectivity Fingerprints*, J. Chem. Inf. Model. **2010**, 50, 742.
- M. Krenn *et al.*, *SELFIES and the future of molecular string representations*, Patterns **2022**.
- L. David *et al.*, *Molecular representations in AI-driven drug discovery: a review and practical guide*, J. Cheminform. **2020**, 12, 56.
- TeachOpenCADD T033 for a longer discussion incl. protein representations.

Next session: **03 · Chemical databases** — where the data comes from.
"""
