# %% [markdown]
"""
# 01 · RDKit basics: molecules as Python objects

**Chemoinformatics practicals — Session 1 of 9**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

**Learning goals.** After this session you will be able to
- write and read **SMILES** strings and understand what "canonical" means;
- create RDKit molecule objects, draw them, and inspect their atoms and bonds;
- search for **substructures** with SMARTS patterns and highlight them;
- compute simple molecular **properties** (MW, logP, TPSA, H-bond donors/acceptors) and apply Lipinski's rule of five;
- read and write molecule files (SMILES, SDF) and combine RDKit with `pandas`.

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - **MolSSI** cheminformatics workshop by Jessica A. Nash — `01_molecule_representation`, `02_rdkit_intro` ([GitHub](https://github.com/MolSSI-Education/molssi-cheminformatics), MIT);
> - *Practical Cheminformatics Tutorials* by **Pat Walters** — `A_Whirlwind_Introduction_To_The_RDKit`, `SMILES_tutorial`, `SMARTS_tutorial` ([GitHub](https://github.com/PatWalters/practical_cheminformatics_tutorials), MIT). The list of approved drugs (`chembl_drugs.smi`) comes from Pat Walters' [datafiles](https://github.com/PatWalters/datafiles) repository (ChEMBL data, CC BY-SA 3.0);
> - *AI for Chemistry* (EPFL CH-457), **Schwaller group** — `01d_rdkit_basics` ([GitHub](https://github.com/schwallergroup/ai4chem_course), MIT);
> - the [RDKit *Getting Started in Python*](https://www.rdkit.org/docs/GettingStartedInPython.html) documentation (BSD).
"""

# %%
# @title ⚙️ Setup — run this cell first
import sys, os, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "mols2grid"], check=False)

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, AllChem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True          # crisp drawings
IPythonConsole.molSize = (300, 220)
import rdkit; print("RDKit", rdkit.__version__)

REPO_RAW = "https://raw.githubusercontent.com/AxelRolov/chemoinformatics_tutorials/main"
def data_path(filename):
    local = os.path.join("..", "data", filename)
    return local if os.path.exists(local) else f"{REPO_RAW}/data/{filename}"

# %% [markdown]
"""
## 1. What is RDKit?

[RDKit](https://www.rdkit.org) is the open-source toolkit that almost every chemoinformatics project relies on.
It knows how to *parse* chemical structures, *draw* them, compute *descriptors* and *fingerprints*,
do *substructure searches*, generate *3D conformers*, and much more. The central object is the **molecule** (`Mol`).

We create molecules most often from **SMILES** (Simplified Molecular-Input Line-Entry System) strings.
"""

# %%
methane = Chem.MolFromSmiles("C")
methane            # a Mol object; in a notebook it is displayed as a drawing

# %%
ethanol = Chem.MolFromSmiles("CCO")
ethanol

# %%
type(ethanol)

# %% [markdown]
"""
## 2. SMILES in 10 minutes

SMILES writes a molecule as a string by walking along its graph. The main rules:

| rule | example | meaning |
|---|---|---|
| atoms are element symbols; hydrogens are implicit | `C`, `N`, `O`, `Cl`, `Br` | methane, ammonia, water, HCl, HBr |
| adjacent atoms are bonded (single bond) | `CCO` | ethanol |
| `=` double, `#` triple bond | `C=C`, `C#N`, `O=C=O` | ethene, hydrogen cyanide, CO₂ |
| parentheses = branches | `CC(C)C`, `CC(=O)O` | isobutane, acetic acid |
| a digit opens/closes a ring | `C1CCCCC1` | cyclohexane |
| lowercase = aromatic atom | `c1ccccc1`, `c1ccncc1` | benzene, pyridine |
| square brackets for charges, isotopes, explicit H | `[NH4+]`, `[O-]`, `[13CH4]`, `[nH]` | ammonium, oxide, ¹³C-methane, pyrrole N–H |
| `.` separates disconnected parts | `[Na+].[Cl-]` | sodium chloride |
| `@`/`@@` tetrahedral chirality, `/` `\` double-bond geometry | `C[C@H](N)C(=O)O`, `F/C=C/F` | L-alanine, (E)-1,2-difluoroethene |

The ring digit trick: `C1CCCCC1` means "open ring bond 1 at the first atom … close it at the last atom".
"""

# %%
smiles_examples = {
    "acetic acid": "CC(=O)O",
    "isobutane": "CC(C)C",
    "cyclohexane": "C1CCCCC1",
    "benzene": "c1ccccc1",
    "pyridine": "c1ccncc1",
    "toluene": "Cc1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "L-alanine": "C[C@H](N)C(=O)O",
    "sodium acetate": "CC(=O)[O-].[Na+]",
}
mols = [Chem.MolFromSmiles(s) for s in smiles_examples.values()]
Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(220, 180), legends=list(smiles_examples.keys()))

# %% [markdown]
"""
### Exercise 2.1 — write SMILES

Write SMILES for the following molecules and draw them (replace the `None`s). Check that the drawing looks right!

1. propanol (CH₃CH₂CH₂OH)
2. acetone
3. phenol
4. aspirin (acetylsalicylic acid: a benzene ring carrying –COOH and –O–C(=O)CH₃ on adjacent carbons)
5. caffeine (harder — two fused rings; you may look it up, but try first)
"""

# %%
my_smiles = {
    "propanol": None,
    "acetone": None,
    "phenol": None,
    "aspirin": None,
    "caffeine": None,
}
valid = {k: v for k, v in my_smiles.items() if v is not None}
if valid:
    display(Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in valid.values()],
                                 molsPerRow=3, subImgSize=(220, 180), legends=list(valid.keys())))

# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
my_smiles = {
    "propanol": "CCCO",
    "acetone": "CC(=O)C",
    "phenol": "Oc1ccccc1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
}
```
</details>
"""

# %% [markdown]
"""
### Invalid SMILES

If a SMILES string cannot be parsed, `MolFromSmiles` returns `None` and prints a warning. **Always check for `None`**
when you process many molecules — a single bad entry can otherwise crash your whole script.
"""

# %%
bad = Chem.MolFromSmiles("C1CC")      # ring bond 1 never closed
print(bad)
bad2 = Chem.MolFromSmiles("c1cccc1")  # 5-membered "aromatic" carbon ring: cannot be kekulized
print(bad2)

# %% [markdown]
"""
### Canonical SMILES: one molecule, many strings

The same molecule can be written in many ways (start from a different atom, walk in a different direction).
RDKit produces a unique **canonical SMILES** for a given molecule, which is how we recognise duplicates.
"""

# %%
for s in ["OCC", "CCO", "C(O)C", "[CH3][CH2][OH]"]:
    m = Chem.MolFromSmiles(s)
    print(f"{s:16s} -> {Chem.MolToSmiles(m)}")

# %%
# Aromaticity is perceived automatically: Kekulé and aromatic forms give the same canonical SMILES
print(Chem.MolToSmiles(Chem.MolFromSmiles("C1=CC=CC=C1")))
print(Chem.MolToSmiles(Chem.MolFromSmiles("c1ccccc1")))

# %%
# Random (non-canonical) SMILES are used in deep learning as "data augmentation" - a molecule has many valid names
caffeine = Chem.MolFromSmiles("Cn1cnc2c1c(=O)n(C)c(=O)n2C")
for i in range(5):
    print(Chem.MolToSmiles(caffeine, doRandom=True, canonical=False))

# %% [markdown]
"""
## 3. Inside a molecule: atoms and bonds

An RDKit molecule is a **graph**: atoms are nodes, bonds are edges. Let's look inside.
"""

# %%
IPythonConsole.drawOptions.addAtomIndices = True    # show atom indices in drawings
aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
aspirin

# %%
print("heavy atoms:", aspirin.GetNumAtoms())
print("all atoms (incl. H):", aspirin.GetNumAtoms(onlyExplicit=False))
print("bonds:", aspirin.GetNumBonds())

# %%
for atom in aspirin.GetAtoms():
    print(f"{atom.GetIdx():2d} {atom.GetSymbol():2s} degree={atom.GetDegree()} "
          f"Hs={atom.GetTotalNumHs()} aromatic={atom.GetIsAromatic()} "
          f"hybridization={atom.GetHybridization()} charge={atom.GetFormalCharge()}")

# %%
for bond in aspirin.GetBonds():
    print(f"{bond.GetBeginAtomIdx():2d}-{bond.GetEndAtomIdx():2d} {str(bond.GetBondType()):10s} "
          f"aromatic={bond.GetIsAromatic()} in_ring={bond.IsInRing()}")

# %%
# Rings
ri = aspirin.GetRingInfo()
print("number of rings:", ri.NumRings())
print("atoms in each ring:", [list(r) for r in ri.AtomRings()])

# %%
IPythonConsole.drawOptions.addAtomIndices = False

# %% [markdown]
"""
### Hydrogens

By default RDKit keeps hydrogens **implicit** (they are counted, not stored as atoms). This is what you want
for most 2D work; for 3D geometry or when hydrogens matter (e.g. tautomers), make them explicit with `Chem.AddHs`.
"""

# %%
aspirin_H = Chem.AddHs(aspirin)
print(aspirin_H.GetNumAtoms())
Chem.RemoveHs(aspirin_H).GetNumAtoms()

# %% [markdown]
"""
### Exercise 3.1

Write a function `count_element(mol, symbol)` that returns how many atoms of a given element a molecule has,
then compute the molecular formula of caffeine yourself (C, H, N, O) and compare with
`rdMolDescriptors.CalcMolFormula(caffeine)`. Hint: hydrogens are implicit — `atom.GetTotalNumHs()`.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
def count_element(mol, symbol):
    if symbol == "H":
        return sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == symbol)

print({el: count_element(caffeine, el) for el in "CHNO"})
print(rdMolDescriptors.CalcMolFormula(caffeine))    # C8H10N4O2
```
</details>
"""

# %% [markdown]
"""
## 4. Substructure search with SMARTS

**SMARTS** is a pattern language that extends SMILES: it describes *what to look for* rather than one specific molecule.

| SMARTS | matches |
|---|---|
| `C(=O)O` (plain SMILES also works as a pattern) | carboxylic acid / ester carbon with two O's |
| `[OX2H]` | an O with 2 connections and 1 H — a hydroxyl group |
| `[NX3;H2]` | primary amine |
| `[#6]` | any carbon (aromatic or not) |
| `[R]` | any ring atom; `[r6]` an atom in a 6-membered ring |
| `[!#6;!#1]` | a heteroatom (not C, not H) |
| `c1ccccc1` | a benzene ring |
| `S(=O)(=O)N` | sulfonamide |
| `[$(C=O)]` | recursive SMARTS: an atom that is a carbonyl carbon |

Two functions do the work: `HasSubstructMatch` (yes/no) and `GetSubstructMatches` (which atoms).
"""

# %%
ester = Chem.MolFromSmarts("[#6][CX3](=O)O[#6]")     # ester: C-C(=O)-O-C
acid = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")         # carboxylic acid
print("aspirin has ester:", aspirin.HasSubstructMatch(ester))
print("aspirin has acid :", aspirin.HasSubstructMatch(acid))
print("matching atoms   :", aspirin.GetSubstructMatches(acid))

# %%
# Highlighting the match(es)
matches = aspirin.GetSubstructMatches(ester)
aspirin.__sssAtoms = [i for m in matches for i in m]   # RDKit highlights these atoms in the notebook
aspirin

# %%
# Counting functional groups over a list of molecules
patterns = {
    "hydroxyl": "[OX2H]",
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "primary amine": "[NX3;H2;!$(NC=O)]",
    "amide": "C(=O)[NX3]",
    "sulfonamide": "S(=O)(=O)N",
    "halogen": "[F,Cl,Br,I]",
    "benzene ring": "c1ccccc1",
}
patterns = {name: Chem.MolFromSmarts(s) for name, s in patterns.items()}

drugs = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "sulfamethoxazole": "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",
    "fluoxetine": "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1",
}
rows = []
for name, smi in drugs.items():
    m = Chem.MolFromSmiles(smi)
    rows.append({"drug": name, **{p: len(m.GetSubstructMatches(q)) for p, q in patterns.items()}})
pd.DataFrame(rows).set_index("drug")

# %% [markdown]
"""
### Exercise 4.1

1. Write a SMARTS for a **nitro group** and test it on nitrobenzene `O=[N+]([O-])c1ccccc1`.
2. Which of the 5 drugs above contain a **secondary amine** (`[NX3;H1;!$(NC=O)]`)? Highlight it.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
print(Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1").HasSubstructMatch(nitro))

sec_amine = Chem.MolFromSmarts("[NX3;H1;!$(NC=O)]")
for name, smi in drugs.items():
    m = Chem.MolFromSmiles(smi)
    if m.HasSubstructMatch(sec_amine):
        print(name)              # fluoxetine
        display(m)
```
</details>
"""

# %% [markdown]
"""
## 5. Molecular properties and Lipinski's rule of five

RDKit computes hundreds of descriptors (we will study them systematically next session). Today, the handful that
medicinal chemists use every day. Lipinski's **rule of five** (Ro5) says an orally available drug *tends* to have
MW ≤ 500, logP ≤ 5, ≤ 5 H-bond donors and ≤ 10 H-bond acceptors.
"""

# %%
def basic_properties(mol):
    return {
        "MW": Descriptors.MolWt(mol),
        "logP": Descriptors.MolLogP(mol),        # Crippen atom-contribution logP
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),  # topological polar surface area
        "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "arom_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
    }

pd.DataFrame({name: basic_properties(Chem.MolFromSmiles(smi)) for name, smi in drugs.items()}).T.round(2)

# %%
def passes_ro5(mol):
    p = basic_properties(mol)
    violations = (p["MW"] > 500) + (p["logP"] > 5) + (p["HBD"] > 5) + (p["HBA"] > 10)
    return violations <= 1          # Lipinski allowed one violation

for name, smi in drugs.items():
    print(f"{name:18s} Ro5: {passes_ro5(Chem.MolFromSmiles(smi))}")

# %% [markdown]
"""
## 6. Many molecules: RDKit + pandas

Real work involves thousands of molecules in a table. The pattern is always the same: a column of SMILES,
a column of `Mol` objects created with `apply`, then columns of computed properties. `mols2grid` gives an interactive
browsable grid. Let's load ~1200 approved drugs from ChEMBL.

> RDKit also has a `PandasTools` module that renders molecules inside DataFrames; it is convenient but fragile
> across pandas versions, so we use plain pandas here.
"""

# %%
drugs_df = pd.read_csv(data_path("chembl_drugs_walters.smi"), sep=" ", names=["SMILES", "ChEMBL_ID"])
drugs_df["mol"] = drugs_df["SMILES"].apply(Chem.MolFromSmiles)
print(drugs_df.shape, "molecules failed to parse:", drugs_df["mol"].isna().sum())
drugs_df.head()

# %%
# Compute properties for every drug (apply our function to the molecule column)
props = pd.DataFrame(list(drugs_df["mol"].apply(basic_properties)))
drugs_df = pd.concat([drugs_df, props], axis=1)
drugs_df["Ro5"] = drugs_df["mol"].apply(passes_ro5)
drugs_df[["ChEMBL_ID", "MW", "logP", "HBD", "HBA", "TPSA", "Ro5"]].describe().round(2)

# %%
print(f"Fraction of approved drugs passing Ro5: {drugs_df['Ro5'].mean():.1%}")

# %%
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))
for ax, col, limit in zip(axes, ["MW", "logP", "TPSA"], [500, 5, 140]):
    drugs_df[col].hist(bins=40, ax=ax)
    ax.axvline(limit, color="red", ls="--")
    ax.set_title(col)
plt.tight_layout(); plt.show()

# %%
# Substructure search over the whole table: which drugs contain a sulfonamide?
sulfonamide = Chem.MolFromSmarts("S(=O)(=O)N")
hits = drugs_df[drugs_df["mol"].apply(lambda m: m.HasSubstructMatch(sulfonamide))]
print(len(hits), "sulfonamide-containing drugs")
Draw.MolsToGridImage(hits["mol"].head(12).tolist(), molsPerRow=4, subImgSize=(200, 160),
                     legends=hits["ChEMBL_ID"].head(12).tolist())

# %%
# An interactive grid you can scroll, sort and filter (hover to see values)
import mols2grid
mols2grid.display(drugs_df.head(200), smiles_col="SMILES", subset=["img", "ChEMBL_ID", "MW"],
                  tooltip=["logP", "TPSA", "HBD", "HBA"], size=(160, 120), n_items_per_page=12)

# %% [markdown]
"""
## 7. Reading and writing molecule files

Besides SMILES, the most common format is **SDF / MOL** (a "MOL block" with atom coordinates and a property list).
"""

# %%
# MOL block: atom coordinates + bond table. Coordinates are 2D here (computed for drawing).
AllChem.Compute2DCoords(aspirin)
print(Chem.MolToMolBlock(aspirin)[:600], "...")

# %%
# Write an SDF with properties, then read it back
mols_to_write = drugs_df.head(20)
with Chem.SDWriter("first_20_drugs.sdf") as w:
    for _, row in mols_to_write.iterrows():
        m = Chem.Mol(row["mol"])
        m.SetProp("_Name", row["ChEMBL_ID"])
        m.SetProp("MW", f"{row['MW']:.2f}")
        w.write(m)

read_back = [m for m in Chem.SDMolSupplier("first_20_drugs.sdf")]
print(len(read_back), read_back[0].GetProp("_Name"), read_back[0].GetProp("MW"))

# %%
# ...or directly into a DataFrame (one row per molecule, one column per SD property)
sdf_df = pd.DataFrame([{"name": m.GetProp("_Name"), "mol": m, **m.GetPropsAsDict()}
                       for m in Chem.SDMolSupplier("first_20_drugs.sdf")])
sdf_df.head(3)

# %%
# Other useful conversions
print(Chem.MolToInchi(aspirin))
print(Chem.MolToInchiKey(aspirin))      # a fixed-length hash of the InChI - great as a database key
print(Chem.MolToSmiles(aspirin))

# %% [markdown]
"""
## 8. Bonus: chemical reactions as SMARTS

RDKit can also apply reaction templates: `reactants >> products`. Here is an amide coupling.
"""

# %%
rxn = AllChem.ReactionFromSmarts("[C:1](=[O:2])[OH].[N;H2:3]>>[C:1](=[O:2])[N:3]")
acid_m = Chem.MolFromSmiles("CC(C)Cc1ccc(C(C)C(=O)O)cc1")     # ibuprofen
amine_m = Chem.MolFromSmiles("NCc1ccccc1")                    # benzylamine
products = rxn.RunReactants((acid_m, amine_m))
product = products[0][0]
Chem.SanitizeMol(product)
Draw.MolsToGridImage([acid_m, amine_m, product], legends=["acid", "amine", "amide product"], subImgSize=(250, 180))

# %% [markdown]
"""
## 9. Exercises to finish

1. Among the 1200 drugs, how many contain **at least one fluorine** atom? And a **trifluoromethyl** group `C(F)(F)F`?
2. Plot `logP` versus `MW` for all drugs, colouring points that fail Ro5 in red.
3. Find the drug with the **highest TPSA** and draw it. Does it look orally available?
4. (Harder) Write a function `largest_ring_size(mol)` that returns the size of the largest ring, and find drugs with a macrocycle (ring ≥ 12 atoms).
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solutions</b></summary>

```python
F = Chem.MolFromSmarts("F"); CF3 = Chem.MolFromSmarts("C(F)(F)F")
print(drugs_df["mol"].apply(lambda m: m.HasSubstructMatch(F)).sum(),
      drugs_df["mol"].apply(lambda m: m.HasSubstructMatch(CF3)).sum())

colors = np.where(drugs_df["Ro5"], "steelblue", "red")
plt.scatter(drugs_df["MW"], drugs_df["logP"], c=colors, s=8, alpha=0.6)
plt.xlabel("MW"); plt.ylabel("logP"); plt.show()

top = drugs_df.loc[drugs_df["TPSA"].idxmax()]
print(top["ChEMBL_ID"], top["TPSA"]); display(top["mol"])

def largest_ring_size(mol):
    rings = mol.GetRingInfo().AtomRings()
    return max((len(r) for r in rings), default=0)
drugs_df["max_ring"] = drugs_df["mol"].apply(largest_ring_size)
macro = drugs_df[drugs_df["max_ring"] >= 12]
print(len(macro)); display(Draw.MolsToGridImage(macro["mol"].head(6).tolist(), legends=macro["ChEMBL_ID"].head(6).tolist()))
```
</details>
"""

# %% [markdown]
"""
## Further reading

- RDKit documentation: [Getting Started in Python](https://www.rdkit.org/docs/GettingStartedInPython.html) and the [RDKit blog](https://greglandrum.github.io/rdkit-blog/).
- Daylight [SMILES](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html) and [SMARTS](https://www.daylight.com/dayhtml/doc/theory/theory.smarts.html) theory manuals.
- Pat Walters, *Practical Cheminformatics Tutorials* — the `fundamentals` folder goes deeper into SMILES, SMARTS, stereochemistry and tautomers.

Next session: **02 · Molecular representations** — turning molecules into numbers for machine learning.
"""
