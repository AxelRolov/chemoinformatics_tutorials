# %% [markdown]
"""
# 09 · Protein–ligand docking: where does the drug bind, and can a score tell us how well?

**Chemoinformatics practicals — Session 9 of 11**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

> 🌐 This notebook downloads one crystal structure from the RCSB PDB. If the PDB is unreachable, it falls back to a
> copy shipped with the course. No GPU is needed: AutoDock Vina runs on the CPU (the whole notebook computes for
> about 6–8 minutes on the 2 cores of a free Colab runtime).

**Learning goals.** After this session you will be able to
- explain what a **docking** program does — a *search* over ligand poses driven by a *scoring function* — and what it does not do;
- **prepare a receptor** from a PDB entry (waters, buffer molecules, modified residues, missing atoms, protonation) and a **ligand** from a SMILES, and say what the PDBQT format adds to a structure;
- run **AutoDock Vina** from Python, read its output, and validate it by **redocking** the crystal ligand (the 2 Å rule — and why it can fail for the right reasons);
- distinguish a **pose** problem from a **scoring** problem, using exhaustiveness, seeds and per-fragment RMSD;
- run a mini **virtual screen** and show, with data, that a docking score is not a binding affinity;
- describe a protein–ligand pose as a set of **interactions** (hinge hydrogen bond, hydrophobic contacts, halogen contacts) and compare poses that way.

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - **TeachOpenCADD** talktorials **T015 · Protein–ligand docking** and **T016 · Protein–ligand interactions** (Volkamer lab; [GitHub](https://github.com/volkamerlab/teachopencadd), CC BY 4.0) — the redocking workflow, box definition from the co-crystallised ligand, and the interaction analysis of the EGFR–inhibitor complex;
> - the **CCPBioSim** *biosim-analysis-workshop* docking notebook ([GitHub](https://github.com/CCPBioSim/biosim-analysis-workshop), CC BY-SA 4.0) — scoring-function discussion and pitfalls;
> - **AutoDock Vina** ([GitHub](https://github.com/ccsb-scripps/AutoDock-Vina), Apache-2.0; Trott & Olson 2010, Eberhardt *et al.* 2021) and **Meeko** ([GitHub](https://github.com/forlilab/Meeko), LGPL-2.1) documentation;
> - **PDBFixer** ([GitHub](https://github.com/openmm/pdbfixer), MIT) and **ProLIF** ([GitHub](https://github.com/chemosim-lab/ProLIF), Apache-2.0) documentation.
"""

# %%
# @title ⚙️ Setup — run this cell first (≈1–2 min on Colab)
import sys, os, io, time, json, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "vina", "meeko", "pdbfixer", "openmm",
                    "py3Dmol", "prolif", "MDAnalysis", "scipy"], check=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolAlign, Descriptors
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning); warnings.filterwarnings("ignore", category=UserWarning)
import py3Dmol
import pdbfixer
import openmm.app as app
from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate, Polymer, ResidueChemTemplates
from vina import Vina
import vina

REPO_RAW = "https://raw.githubusercontent.com/AxelRolov/chemoinformatics_tutorials/main"
def fetch(filename, subdir="data"):
    """Return a local path to a course file (download from GitHub if needed)."""
    local = os.path.join("..", subdir, filename)
    if os.path.exists(local):
        return local
    import urllib.request
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    urllib.request.urlretrieve(f"{REPO_RAW}/{subdir}/{filename}", filename)
    return filename

def get_pdb(pdb_id):
    """Download a PDB entry from the RCSB; fall back to the copy cached in the course repository."""
    try:
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb", timeout=30)
        r.raise_for_status()
        text, source = r.text, "RCSB PDB"
    except Exception as e:
        text, source = open(fetch(f"pdb/{pdb_id}.pdb")).read(), "course cache (RCSB unreachable)"
    path = f"{pdb_id}.pdb"
    with open(path, "w") as f:
        f.write(text)
    print(f"{pdb_id}: {len(text.splitlines())} lines from {source}")
    return path

print("Vina", vina.__version__, "| RDKit", Chem.rdBase.rdkitVersion, "| CPUs:", os.cpu_count())

# %% [markdown]
"""
## 1. What docking is — and is not

Everything up to session 08 treated a molecule as a graph and a property as a number to predict. Session 03 downloaded a
protein–ligand structure from the PDB and session 02 gave a first taste of 3D conformers; today the two meet. Docking
asks: **given a protein structure and a small molecule, where and how does the molecule sit in the binding site?**
A docking program has two parts, and everything that follows is about keeping them apart in your head:

| part | what it does | typical choice in Vina |
|---|---|---|
| **search** | generates candidate **poses** — positions, orientations and torsion angles of the ligand inside a **box** | Monte-Carlo perturbations + local optimisation, repeated `exhaustiveness` times |
| **scoring function** | assigns each pose a number meant to look like a binding free energy | a weighted sum of steric (two Gaussians + repulsion), hydrophobic and hydrogen-bond terms between atom pairs, minus a penalty per rotatable bond |

The output is a **ranked list of poses with scores in kcal/mol**. Three things a docking run does *not* do:
1. it does not move the protein (the receptor is **rigid** — every side chain stays where the crystallographer left it);
2. it does not know about water (explicit waters are removed; the scoring function only mimics desolvation);
3. it does not compute a binding free energy, whatever the unit says. We will *measure* how far the scores are from affinities in section 7.

Two questions therefore need two validations: *is the pose right?* (redocking: put the crystal ligand back — section 5)
and *does the score rank binders?* (a small screen with known actives and inactives — section 7).

Our system, as in the whole course, is the **EGFR kinase** with **gefitinib**: PDB entry
[4WKQ](https://www.rcsb.org/structure/4WKQ), 1.85 Å resolution — the best-resolved structure of this complex
(TeachOpenCADD T015 uses the older 3.25 Å entry 2ITO of the same complex).
"""

# %% [markdown]
"""
## 2. Get and read the structure

`get_pdb` (defined in the setup cell) downloads the entry as a text file in the legacy **PDB format** you met in session 03
and in Lecture 4: one fixed-width line per atom. Before any preparation, we read what the file says about *itself*:
the resolution, which molecules are in it, and — crucially — which residues the crystallographer could **not** see.
"""

# %%
pdb_path = get_pdb("4WKQ")
lines = open(pdb_path).read().splitlines()

records = pd.Series([l[:6].strip() for l in lines]).value_counts()
print("record types:", dict(records[["ATOM", "HETATM", "SEQRES", "CONECT"]]))
print([l for l in lines if l.startswith("REMARK   2 RESOLUTION")][0].strip())

het = pd.Series([l[17:20].strip() for l in lines if l.startswith("HETATM")]).value_counts()
print("hetero groups (atoms):", dict(het))
for l in lines:
    if l.startswith("HETNAM"):
        print("  ", l[11:].strip())

# %% [markdown]
"""
The `HETATM` records hold everything that is not a standard amino acid: our ligand **IRE** (gefitinib, 31 heavy atoms),
a sodium ion, a **MES** buffer molecule that crystallised along, 121 **waters**, and **CSX** — an *S-oxy cysteine*, a
chemically modified residue (residue 797, the cysteine that covalent EGFR inhibitors target). Each of these needs a
decision before docking. Next, the residues that are *absent*: `REMARK 465` lists what was in the crystallised
construct (`SEQRES`, 330 residues) but has no coordinates.
"""

# %%
AMINO_ACIDS = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO",
               "SER", "THR", "TRP", "TYR", "VAL"}
missing_ids = [int(l[21:26]) for l in lines if l.startswith("REMARK 465") and l[15:18] in AMINO_ACIDS]
print(len(missing_ids), "missing residues:", missing_ids)

# %% [markdown]
"""
Thirty-three residues are missing: both termini, three residues of the glycine-rich **P-loop** (721–723, the "roof"
over the ATP site), five of the loop after strand β3 (747–751) and the whole **activation loop** (985–1006). Missing
residues are disordered in the crystal, not absent from the protein — they are simply moving too much to be seen.
For docking into a rigid receptor we will leave the gaps as they are (the P-loop tip is 7 Å from the ligand — we
will lose a little of the pocket's roof); for MD in session 11 the chain must be continuous and we will model them.

Finally, the ligand's own **B-factors** (column 61–66, in Å²) tell us how well each atom is defined — remember
Lecture 4: a large B-factor means the atom is smeared out.
"""

# %%
lig_lines = [l for l in lines if l.startswith("HETATM") and l[17:20] == "IRE"]
bfac = pd.Series({l[12:16].strip(): float(l[60:66]) for l in lig_lines})
tail = ["CAK", "CAJ", "CAN", "NBE", "CAP", "CAM", "OAU", "CAL", "CAO"]        # the propoxy-morpholine chain
print("B-factors (Å²) per atom:", " ".join(f"{k}:{v:.0f}" for k, v in bfac.items()))
print(f"mean B: core and anilino ring {bfac.drop(tail).mean():.0f} Å²  |  propoxy-morpholine tail {bfac[tail].mean():.0f} Å²")

# %% [markdown]
"""
The quinazoline core and the anilino ring (`N1`, `C2`, `N3`, `C4` … `CAY`–`CAE`, `CL`, `FAB`) have B ≈ 40–48 Å², like the
surrounding protein. The propoxy-morpholine tail (`CAK` … `CAO`, `NBE`) has B ≈ 75–105 Å²: it points into solvent and the
crystal barely sees it. Keep this in mind when we judge poses by RMSD.

Let us look at the complex. Protein as cartoon, gefitinib as sticks, waters hidden.
"""

# %%
view = py3Dmol.view(width=600, height=420)
view.addModel(open(pdb_path).read(), "pdb")
view.setStyle({"cartoon": {"color": "lightgray"}})
view.setStyle({"resn": "IRE"}, {"stick": {"colorscheme": "greenCarbon"}})
view.setStyle({"resn": "MES"}, {"stick": {"colorscheme": "orangeCarbon"}})
view.addStyle({"resi": [745, 790, 793, 797], "not": {"resn": "HOH"}}, {"stick": {"colorscheme": "whiteCarbon", "radius": 0.15}})
view.zoomTo({"resn": "IRE"}); view.show()

# %% [markdown]
"""
Gefitinib sits in the ATP pocket between the two lobes of the kinase. The thin white side chains are the residues you
will keep meeting: **Met793** (the *hinge*, whose backbone N–H donates a hydrogen bond to the quinazoline N1),
**Thr790** (the *gatekeeper*, mutated to Met in resistant tumours), **Lys745** (the catalytic lysine) and **Cys797**.
The orange MES molecule is a crystallisation artefact sitting on the surface — it must go.

## 3. Prepare the receptor

Docking programs want a clean protein: no ligand, no buffer, no water, complete side chains, hydrogens where the pH says.
**PDBFixer** (from the OpenMM project) does each of these steps explicitly. The order matters in one place:
`findMissingResidues` compares the coordinates against `SEQRES`, so it must run *before* we rename CSX to CYS.
"""

# %%
def prepare_receptor(pdb_path, out_path, ph=7.4, build_loops=False, verbose=True):
    """Clean a PDB file for docking/MD: remove heterogens, fix residues and atoms, add hydrogens at the given pH."""
    fixer = pdbfixer.PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()                       # gaps in the chain, from SEQRES
    if not build_loops:
        fixer.missingResidues = {}                    # keep the gaps (docking: rigid receptor anyway)
    else:                                             # model internal gaps only, never extend the termini
        chains = list(fixer.topology.chains())
        for key in list(fixer.missingResidues):
            if key[1] == 0 or key[1] == len(list(chains[key[0]].residues())):
                del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    if verbose:
        print("non-standard residues:", [(r.name, r.id, "->", std) for r, std in fixer.nonstandardResidues])
    fixer.replaceNonstandardResidues()                # CSX -> CYS
    fixer.removeHeterogens(keepWater=False)           # ligand, buffer, ions, waters
    fixer.findMissingAtoms()
    if verbose:
        print("residues with missing atoms:", len(fixer.missingAtoms), "e.g.",
              [f"{r.name}{r.id}: {'/'.join(a.name for a in atoms)}" for r, atoms in list(fixer.missingAtoms.items())[:3]])
    fixer.addMissingAtoms()                           # side chains (and loops, if requested)
    fixer.addMissingHydrogens(ph)                     # protonation states at this pH
    with open(out_path, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    n_res = sum(1 for _ in fixer.topology.residues())
    if verbose:
        print(f"receptor written to {out_path}: {n_res} residues, {fixer.topology.getNumAtoms()} atoms")
    return fixer

t0 = time.time()
fixer = prepare_receptor(pdb_path, "receptor.pdb")
print(f"({time.time() - t0:.1f} s)")

# %% [markdown]
"""
Sixteen residues had truncated side chains (surface lysines and glutamates whose ends were disordered — PDBFixer rebuilds
them in a standard conformation), CSX797 became a cysteine, and 4791 atoms remain: 2331 heavy atoms plus ~2460
hydrogens. Note what `addMissingHydrogens(7.4)` decided *for* us: aspartates and glutamates deprotonated, lysines and
arginines protonated, histidines neutral. For a residue whose protonation is uncertain (a histidine next to the ligand,
say) you would inspect it by hand; there is none in this pocket.

### The PDBQT format

AutoDock programs read a variant of PDB called **PDBQT**: each atom line carries a partial charge (**Q**) and an AutoDock
atom **T**ype (`C` aliphatic carbon, `A` aromatic carbon, `OA` oxygen acceptor, `N`, `NA` nitrogen acceptor, `HD` polar
hydrogen …). Non-polar hydrogens are merged into their carbons. **Meeko**, the preparation library of the AutoDock team,
builds it for a protein from the fixed PDB. (Vina's scoring function does not use the charges — the `Q` column matters
only for the AutoDock4 scoring function; the types are what Vina reads.)
"""

# %%
t0 = time.time()
templates = ResidueChemTemplates.create_from_defaults()
polymer = Polymer.from_pdb_string(open("receptor.pdb").read(), templates, MoleculePreparation())
receptor_pdbqt, flex_pdbqt = PDBQTWriterLegacy.write_string_from_polymer(polymer)
open("receptor.pdbqt", "w").write(receptor_pdbqt)

pdbqt_lines = receptor_pdbqt.splitlines()
types = pd.Series([l[77:79].strip() for l in pdbqt_lines if l.startswith("ATOM")]).value_counts()
print(f"receptor.pdbqt: {sum(1 for l in pdbqt_lines if l.startswith('ATOM'))} atom lines ({time.time() - t0:.1f} s)")
print("AutoDock atom types:", dict(types))
print("\n".join(pdbqt_lines[:3]))

# %% [markdown]
"""
About 2880 atom lines for 4791 atoms: the ~1900 non-polar hydrogens were merged. Only the polar hydrogens (`HD`, the
ones that can donate a hydrogen bond) are kept — 509 of them.

## 4. Prepare the ligand

The ligand needs two things the receptor did not: a **3D conformer** (from the SMILES, with RDKit's ETKDG — the first
taste of session 02; session 10 looks at conformers properly) and a **torsion tree** — the list of rotatable bonds Vina
is allowed to turn during the search. Meeko
writes both into the ligand PDBQT. We also decide the **protonation state**: gefitinib's morpholine nitrogen has
pKa ≈ 7.2, so at pH 7.4 roughly half the molecules are protonated. We dock the neutral form — for Vina it makes
almost no difference (no electrostatics in the scoring function), for MD in session 11 it will matter.
"""

# %%
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"

def ligand_from_smiles(smiles, seed=42):
    """3D-embed a SMILES (ETKDG + MMFF94), hydrogens included."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol

def ligand_to_pdbqt(mol):
    """Meeko: atom types, charges and the torsion tree for a 3D RDKit molecule with hydrogens."""
    setups = MoleculePreparation().prepare(mol)
    pdbqt, ok, err = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        raise RuntimeError(err)
    return pdbqt

gefitinib_3d = ligand_from_smiles(GEFITINIB)
ligand_pdbqt = ligand_to_pdbqt(gefitinib_3d)
print("rotatable bonds (BRANCH records):", sum(1 for l in ligand_pdbqt.splitlines() if l.startswith("BRANCH")))
print("\n".join(ligand_pdbqt.splitlines()[:6]), "\n...")
Chem.MolFromSmiles(GEFITINIB)

# %% [markdown]
"""
Eight rotatable bonds → Vina searches a space of 3 (position) + 3 (orientation) + 8 (torsions) = **14 dimensions**. The
`REMARK SMILES` line lets Meeko turn poses back into RDKit molecules later.

### The crystal ligand, for reference

To validate docking we need the **experimental pose**. We extract the `IRE` atoms from the PDB file, and — because a PDB
file has no bond orders — assign them from the SMILES with `AssignBondOrdersFromTemplate` (the same trick session 03
used). Adding hydrogens with coordinates gives us the crystal ligand in the same form as the docked ones.
"""

# %%
def crystal_ligand(pdb_path, resname, smiles):
    """Extract a ligand from a PDB file and give it correct bond orders from its SMILES."""
    block = "\n".join(l.rstrip("\n") for l in open(pdb_path) if l.startswith("HETATM") and l[17:20] == resname) + "\nEND\n"
    raw = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
    mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smiles), Chem.RemoveHs(raw, sanitize=False))
    Chem.SanitizeMol(mol)
    return Chem.AddHs(mol, addCoords=True)

crystal = crystal_ligand(pdb_path, "IRE", GEFITINIB)
print(Chem.MolToSmiles(Chem.RemoveHs(crystal)) == Chem.MolToSmiles(Chem.MolFromSmiles(GEFITINIB)), "-> same molecule")
Chem.MolToMolFile(crystal, "gefitinib_crystal.sdf")
crystal_pdbqt = ligand_to_pdbqt(crystal)

# %% [markdown]
"""
## 5. The search box, and redocking

Vina searches inside a rectangular **box**. We centre it on the crystal ligand and make it the ligand's extent plus 8 Å,
with a minimum of 20 Å per side — big enough for the ligand to turn around, small enough that the search stays
efficient. (Without a co-crystallised ligand you would centre the box on a pocket predicted by a cavity detector —
TeachOpenCADD T014 — or on residues you know matter, here Met793 and Thr790.)
"""

# %%
xyz = Chem.RemoveHs(crystal).GetConformer().GetPositions()
center = xyz.mean(axis=0)
box = np.maximum(xyz.max(axis=0) - xyz.min(axis=0) + 8.0, 20.0)
print("box centre (Å):", center.round(2), "| box size (Å):", box.round(1))

view = py3Dmol.view(width=600, height=420)
view.addModel(open("receptor.pdb").read(), "pdb")
view.setStyle({"cartoon": {"color": "lightgray"}})
view.addModel(Chem.MolToMolBlock(crystal), "mol")
view.setStyle({"model": 1}, {"stick": {"colorscheme": "greenCarbon"}})
view.addBox({"center": {"x": center[0], "y": center[1], "z": center[2]},
             "dimensions": {"w": box[0], "h": box[1], "d": box[2]}, "color": "magenta", "wireframe": True})
view.zoomTo({"model": 1}); view.show()

# %% [markdown]
"""
### Scoring the experimental pose

Before searching anything, we ask Vina how it *scores* the true answer. `set_ligand_from_string` loads the crystal
ligand in its crystal coordinates; `compute_vina_maps` precomputes the scoring function on a grid inside the box;
`score()` evaluates the pose as it is; `optimize()` lets it relax in Vina's own potential.
"""

# %%
v = Vina(sf_name="vina", seed=42, verbosity=0)
v.set_receptor("receptor.pdbqt")
v.set_ligand_from_string(crystal_pdbqt)
v.compute_vina_maps(center=center.tolist(), box_size=box.tolist())

e = v.score()
print(f"crystal pose: total {e[0]:.2f} = intermolecular {e[1]:.2f} + intramolecular {e[5]:.2f} "
      f"+ torsional penalty {e[6]:.2f} - unbound intramolecular {e[7]:.2f}   (kcal/mol)")
e_opt = v.optimize()
v.write_pose("crystal_optimised.pdbqt", overwrite=True)
optimised = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(open("crystal_optimised.pdbqt").read(), skip_typing=True))[0]
shift = rdMolAlign.CalcRMS(Chem.RemoveHs(optimised), Chem.RemoveHs(crystal))
print(f"after local optimisation: {e_opt[0]:.2f} kcal/mol, heavy atoms moved by {shift:.2f} Å RMSD")

# %% [markdown]
"""
Read the decomposition: the ligand–protein interaction term is about −10.7 kcal/mol, but Vina then adds a penalty of
**+3.4 kcal/mol for the eight rotatable bonds** (an entropy-like correction for the flexibility frozen on binding)
and reports −7.3. The crystal pose is already a local minimum of Vina's function: optimisation moves it by 0.2 Å.

### Redocking

Now the real test. We give Vina the *randomly embedded* gefitinib conformer from the SMILES — no memory of the crystal —
and let it search. `exhaustiveness=8` (the default) means eight independent Monte-Carlo runs; `n_poses=9` keeps the nine
best distinct poses (at least 1 Å RMSD apart).
"""

# %%
def dock(ligand_pdbqt, exhaustiveness=8, n_poses=9, seed=42, sf_name="vina", center=center, box=box):
    """Run Vina and return (energies array, RDKit molecule with one conformer per pose)."""
    v = Vina(sf_name=sf_name, seed=seed, verbosity=0)
    v.set_receptor("receptor.pdbqt")
    v.set_ligand_from_string(ligand_pdbqt)
    v.compute_vina_maps(center=list(center), box_size=list(box))
    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    energies = v.energies(n_poses=n_poses)
    poses = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(v.poses(n_poses=n_poses), skip_typing=True))[0]
    return energies, poses

t0 = time.time()
energies, poses = dock(ligand_pdbqt, exhaustiveness=8)
print(f"{poses.GetNumConformers()} poses in {time.time() - t0:.0f} s")

# %% [markdown]
"""
### Did we get the crystal pose back?

The standard yardstick is the heavy-atom **RMSD to the crystal pose, without re-alignment** (the frame is the protein's):
a pose within **2 Å** counts as reproduced. `CalcRMS` handles symmetry (swapping the two equivalent morpholine arms is
not an error). We add a second column — the RMSD of the **quinazoline–anilino core** only — because we saw in section 2
that the tail is disordered in the crystal itself.
"""

# %%
CORE = Chem.MolFromSmarts("c1ccc2ncnc(Nc3ccccc3)c2c1")     # 4-anilinoquinazoline: gefitinib, erlotinib, ...

def pose_rmsd(poses, reference, core=CORE):
    """Heavy-atom RMSD (symmetry-aware, no alignment) and core-only RMSD of every conformer of `poses` to `reference`."""
    p, r = Chem.RemoveHs(poses), Chem.RemoveHs(reference)
    same_molecule = Chem.MolToSmiles(p) == Chem.MolToSmiles(r)
    matches = p.GetSubstructMatches(core, uniquify=False)          # all symmetry-equivalent ways to map the core
    rm = list(r.GetSubstructMatch(core))
    R = r.GetConformer().GetPositions()[rm]
    full, core_only = [], []
    for cid in range(p.GetNumConformers()):
        full.append(rdMolAlign.CalcRMS(p, r, prbId=cid, refId=0) if same_molecule else np.nan)
        P = p.GetConformer(cid).GetPositions()
        core_only.append(min(np.sqrt(((P[list(m)] - R) ** 2).sum(axis=1).mean()) for m in matches) if matches and rm else np.nan)
    return np.array(full), np.array(core_only)

def results_table(energies, poses, reference):
    full, core = pose_rmsd(poses, reference)
    return pd.DataFrame({"mode": np.arange(1, len(full) + 1), "score (kcal/mol)": energies[:len(full), 0].round(2),
                         "RMSD all heavy atoms (Å)": full.round(2), "RMSD core (Å)": core.round(2)})

redock = results_table(energies, poses, crystal)
redock

# %% [markdown]
"""
Look at the two RMSD columns together. The **top-ranked pose has the core within 0.5 Å of the crystal** — the
quinazoline in the hinge, the anilino ring in the back pocket, exactly right — but its total RMSD is above 3 Å, because
the propoxy-morpholine tail lies along a different groove on the protein surface. The tail with B ≈ 100 Å² is a part of
the molecule the crystal itself could hardly place. Judged by the strict 2 Å rule, redocking "failed"; judged on the
atoms that are actually ordered, it succeeded. Pose 3 (2.1 Å) is the one that happens to put the tail where the
crystallographer modelled it.

This is the general lesson: **always ask *which* atoms are wrong**. A pose can be right where it matters and wrong where
nothing matters — or the reverse.
"""

# %%
view = py3Dmol.view(width=600, height=420)
view.addModel(open("receptor.pdb").read(), "pdb")
view.setStyle({"cartoon": {"color": "lightgray", "opacity": 0.7}})
view.addStyle({"resi": [793, 790, 745, 797]}, {"stick": {"colorscheme": "whiteCarbon", "radius": 0.12}})
view.addModel(Chem.MolToMolBlock(crystal), "mol")
view.setStyle({"model": 1}, {"stick": {"colorscheme": "greenCarbon"}})
for k, cid in enumerate([0, 2]):
    view.addModel(Chem.MolToMolBlock(poses, confId=cid), "mol")
    view.setStyle({"model": 2 + k}, {"stick": {"colorscheme": ["magentaCarbon", "cyanCarbon"][k], "radius": 0.2}})
view.zoomTo({"model": 1}); view.show()
print("green: crystal | magenta: pose 1 | cyan: pose 3")

# %% [markdown]
"""
### Is it a search problem or a scoring problem?

Two separate things can go wrong. If the search never *visits* the right pose, more sampling helps: raise
`exhaustiveness`. If the search finds it but the scoring function *prefers* a wrong one, more sampling cannot help —
the crystal pose we scored above (−7.3) is indeed ranked below several docked poses (−8.2). Docking is also
**stochastic**: a different random seed gives a different list. Let us look at both knobs.
"""

# %%
rows = []
for exh, seed in [(1, 1), (1, 2), (8, 1), (8, 2)]:
    t0 = time.time()
    e, p = dock(ligand_pdbqt, exhaustiveness=exh, seed=seed)
    full, core = pose_rmsd(p, crystal)
    rows.append({"exhaustiveness": exh, "seed": seed, "time (s)": round(time.time() - t0),
                 "best score": e[0, 0].round(2), "RMSD of pose 1 (Å)": full[0].round(2),
                 "core RMSD of pose 1 (Å)": core[0].round(2), "best RMSD in top 9 (Å)": full.min().round(2)})
pd.DataFrame(rows)

# %% [markdown]
"""
Even exhaustiveness 1 finds the right core here: this pocket is an easy case — a deep, well-defined slot with one strong
hydrogen bond to anchor the quinazoline. What changes with more search and with the seed is everything *else*: the
scores in the second decimal, the placement of the tail, and which of the nine poses happens to be the one within 2 Å
(the "best RMSD" column moves between 2.6 and 3.4 Å). The take-home rule: **check the convergence of your own docking**
by changing the seed before you believe a single run — and expect to be shown more than one plausible pose.
(Exercise 1 asks for exhaustiveness 32 — about five times slower.)

## 6. Cross-docking: a second ligand, no crystal structure

Redocking is the easy case: the pocket was shaped around this very ligand. The realistic case is **cross-docking** —
a *different* ligand into this receptor conformation. **Erlotinib** is also a 4-anilinoquinazoline, so its core should
land where gefitinib's does; we can check that with the core RMSD against *gefitinib's* crystal core, without needing
erlotinib's own structure (1M17, if you want to compare later).
"""

# %%
ERLOTINIB = "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"
t0 = time.time()
e_erl, poses_erl = dock(ligand_to_pdbqt(ligand_from_smiles(ERLOTINIB)), exhaustiveness=8)
full_erl, core_erl = pose_rmsd(poses_erl, crystal)
print(f"erlotinib docked in {time.time() - t0:.0f} s")
pd.DataFrame({"mode": np.arange(1, len(core_erl) + 1), "score (kcal/mol)": e_erl[:len(core_erl), 0].round(2),
              "core RMSD to gefitinib crystal core (Å)": core_erl.round(2)})

# %% [markdown]
"""
The full-molecule RMSD makes no sense here (different molecules), but the **core RMSD does**: pose 1 places erlotinib's
quinazoline within an ångström of gefitinib's. The hinge-binding motif is transferable — which is exactly why the
4-anilinoquinazoline scaffold became a drug class.

## 7. A mini virtual screen: is the score an affinity?

Now the second validation. We take, from the curated EGFR set of session 04, six very potent compounds (pIC50 ≥ 9),
six very weak ones (pIC50 ≤ 5), the two drugs, and four molecules that are certainly not EGFR inhibitors (caffeine,
aspirin, ibuprofen, glucose). We dock all eighteen with `exhaustiveness=4` (≈ 3–4 minutes) and keep the best score.
If docking scores were affinities, the potent set should score clearly below the weak set.
"""

# %%
egfr = pd.read_csv(fetch("EGFR_curated.csv"))
egfr["heavy_atoms"] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in egfr.smiles]
mid_size = egfr[egfr.heavy_atoms.between(20, 34)]                       # comparable sizes to the drugs
potent = mid_size[mid_size.pIC50 >= 9].sample(6, random_state=7).assign(set="potent (pIC50 ≥ 9)")
weak = mid_size[mid_size.pIC50 <= 5].sample(6, random_state=7).assign(set="weak (pIC50 ≤ 5)")
drugs = pd.DataFrame({"chembl_id": ["gefitinib", "erlotinib"], "smiles": [GEFITINIB, ERLOTINIB], "pIC50": [9.0, 9.3], "set": "drug"})
decoys = pd.DataFrame({"chembl_id": ["caffeine", "aspirin", "ibuprofen", "glucose"],
                       "smiles": ["Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
                                  "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"], "pIC50": np.nan, "set": "non-binder"})
screen = pd.concat([drugs, potent, weak, decoys], ignore_index=True)[["chembl_id", "smiles", "pIC50", "set"]]
screen["heavy_atoms"] = [Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in screen.smiles]
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in screen.smiles], molsPerRow=6, subImgSize=(200, 160),
                     legends=[f"{i} | pIC50 {p:.1f}" if p == p else i for i, p in zip(screen.chembl_id, screen.pIC50)])

# %%
t0 = time.time()
scores = []
for i, row in screen.iterrows():
    e, _ = dock(ligand_to_pdbqt(ligand_from_smiles(row.smiles)), exhaustiveness=4, n_poses=3)
    scores.append(e[0, 0])
    print(f"{row.chembl_id:14s} {row.set:18s} score {e[0, 0]:6.2f}   ({time.time() - t0:.0f} s)")
screen["score"] = scores
screen.to_csv("mini_screen.csv", index=False)

# %%
from scipy.stats import spearmanr

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
colors = {"drug": "tab:green", "potent (pIC50 ≥ 9)": "tab:blue", "weak (pIC50 ≤ 5)": "tab:orange", "non-binder": "tab:gray"}
for label, grp in screen.groupby("set"):
    axes[0].scatter(grp.pIC50.fillna(3.0), grp.score, label=label, c=colors[label], s=50)
    axes[1].scatter(grp.heavy_atoms, grp.score, label=label, c=colors[label], s=50)
axes[0].set_xlabel("pIC50 (non-binders drawn at 3)"); axes[0].set_ylabel("Vina score (kcal/mol)")
axes[1].set_xlabel("heavy atoms"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.show()

measured = screen.dropna(subset=["pIC50"])
print(screen.groupby("set").score.agg(["mean", "min", "max"]).round(2))
print(f"Spearman ρ(score, pIC50)      = {spearmanr(measured.score, measured.pIC50)[0]:+.2f}  (18 - 4 = {len(measured)} molecules)")
print(f"Spearman ρ(score, heavy atoms) = {spearmanr(screen.score, screen.heavy_atoms)[0]:+.2f}  (all {len(screen)})")

# %% [markdown]
"""
Two observations, and they are the point of this section.

1. **The score separates small from large, not weak from potent.** Caffeine, aspirin, ibuprofen and glucose (12–15 heavy
   atoms) score around −5 to −6 kcal/mol; every drug-sized molecule scores −7 to −9.5. But the potent EGFR inhibitors
   (pIC50 ≥ 9, i.e. sub-nanomolar) and the weak ones (pIC50 ≤ 5, i.e. > 10 µM) — a **10,000-fold difference in
   affinity** — get the *same* scores (means −7.9 and −7.7).
2. The correlation with pIC50 is essentially zero, the correlation with the **number of heavy atoms** is strongly negative:
   more atoms, more pairwise terms, better score. This size bias is well documented for every additive scoring function.

So what *is* a docking score good for? Enrichment: in a screen of a million molecules it pushes plausible binders — the
right size, the right shape, a hydrogen bond in the right place — towards the top, and that is valuable. It cannot
rank the last hundred by affinity. For that you need the methods of Lecture 5 (MM/GBSA, free-energy perturbation) or,
more often, an assay.

> Note that our "weak" compounds are still EGFR ligands from ChEMBL (someone made and measured them). A screen against
> truly random drug-like decoys, as in the DUD-E benchmark, would look more flattering — and would say less.

## 8. Describing a pose by its interactions

An RMSD says *how far*; it does not say *what changed*. Medicinal chemists describe a pose as a list of
**interactions**: which residues form hydrogen bonds, which hydrophobic contacts, is there π-stacking, a salt bridge, a
halogen bond? **ProLIF** computes such an *interaction fingerprint* from geometric rules (distance and angle cut-offs per
interaction type), the same idea as **PLIP** in TeachOpenCADD T016.

We compare the crystal pose with the top three docked poses. The protein is loaded through MDAnalysis, the ligands are
our RDKit molecules with hydrogens.
"""

# %%
import MDAnalysis as mda
import prolif as plf

u = mda.Universe("receptor.pdb")
protein = plf.Molecule.from_mda(u.select_atoms("protein"), NoImplicit=False)

def plf_ligand(mol, conf_id=-1):
    """A ProLIF ligand from an RDKit conformer, with any PDB residue labels stripped so that all ligands are named alike."""
    clean = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol, confId=conf_id), removeHs=False)
    return plf.Molecule.from_rdkit(clean, resname="LIG")

ligands = [plf_ligand(crystal)] + [plf_ligand(poses, cid) for cid in range(3)]

fp = plf.Fingerprint(["HBDonor", "HBAcceptor", "XBDonor", "Hydrophobic", "PiStacking", "Cationic", "Anionic", "CationPi", "PiCation"])
fp.run_from_iterable(ligands, protein, progress=False)
ifp = fp.to_dataframe()
ifp.index = ["crystal", "pose 1", "pose 2", "pose 3"]
table = ifp.T.droplevel("ligand").astype(int)
table

# %% [markdown]
"""
Read the table row by row. `MET793 · HBAcceptor` — the ligand *accepts* a hydrogen bond from Met793's backbone N–H: the
**hinge interaction** that every ATP-competitive kinase inhibitor makes. It is present in the crystal and in every docked
pose. `LEU788 · XBDonor` is the halogen contact of the chloro/fluoro-phenyl ring in the back pocket, seen only in the
crystal; the hydrophobic contact with **Cys797** appears only when the tail turns towards the solvent front. A compact
way to compare two poses is the **Jaccard similarity** of their interaction sets.
"""

# %%
def jaccard(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / (a | b).sum()

for pose in ["pose 1", "pose 2", "pose 3"]:
    print(f"crystal vs {pose}: Jaccard {jaccard(table['crystal'], table[pose]):.2f}")

hinge = u.select_atoms("resid 793 and name N")
for name, m in [("crystal", Chem.RemoveHs(crystal))] + [(f"pose {k+1}", Chem.RemoveHs(Chem.Mol(poses, confId=k))) for k in range(3)]:
    ring_n = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N" and a.GetIsAromatic()]     # the two quinazoline nitrogens
    d = np.linalg.norm(m.GetConformer().GetPositions()[ring_n] - hinge.positions[0], axis=1).min()
    print(f"{name}: shortest quinazoline N ··· Met793 N distance = {d:.2f} Å")

# %% [markdown]
"""
The hinge N···N distance of ~3 Å in all poses is the geometric face of the same fact. When you dock a *new* molecule into
this pocket, "does it make the Met793 hydrogen bond?" is a better first question than "what is its score?".

## 9. Hand-over to sessions 10 and 11

Docking gives a **static** pose in a rigid pocket, scored by a function that ignores water. Session 10 introduces the
tool that removes both simplifications — molecular dynamics, on a small peptide where everything can be watched — and
session 11 turns it on this complex: **is the pose stable when everything is allowed to move?** We save what session 11
needs — the prepared receptor, the crystal ligand and the docked poses — and download them.
"""

# %%
w = Chem.SDWriter("gefitinib_docked_poses.sdf")
for cid in range(poses.GetNumConformers()):
    poses.SetProp("_Name", f"gefitinib_pose_{cid + 1}"); poses.SetProp("vina_score", f"{energies[cid, 0]:.2f}")
    w.write(poses, confId=cid)
w.close()
print("written: receptor.pdb, gefitinib_crystal.sdf, gefitinib_docked_poses.sdf")
if IN_COLAB:
    from google.colab import files
    for f in ["receptor.pdb", "gefitinib_crystal.sdf", "gefitinib_docked_poses.sdf"]:
        files.download(f)

# %% [markdown]
"""
## 10. What we learned, and what to keep in mind

| question | our answer |
|---|---|
| Where does gefitinib bind? | in the ATP pocket, quinazoline N1 accepting a hydrogen bond from the Met793 hinge — redocking reproduces the core to 0.5 Å |
| Is the 2 Å rule enough? | no — look at *which* atoms deviate (here a tail the crystal itself hardly sees) |
| Is one run enough? | no — change the seed, raise the exhaustiveness, check the top pose is stable |
| Is the score an affinity? | no — ρ(score, pIC50) ≈ 0 while ρ(score, size) ≈ −0.8; scores enrich, they do not rank |
| How to compare poses? | by their interactions (hinge H-bond, hydrophobic and halogen contacts), not only by RMSD |

**Pitfalls we walked around** (Lecture 4, section 6): the receptor is rigid — dock into several crystal structures or
use flexible side chains (Vina supports `flex_pdbqt`); waters are gone — some bridge ligand and protein and should be
kept; protonation and tautomers are your decision, not the program's; and a box that is too small hides the answer while
one that is too large wastes the search.

**Tools beyond Vina.** *smina* and *gnina* (Vina forks; gnina rescores with a convolutional network), *Glide* and *GOLD*
(commercial), *DiffDock* (generative), and co-folding models — *AlphaFold 3*, *Boltz-2*, *Chai* — that predict
protein and ligand together, with the same caveat about affinities (PoseBusters checks the physical plausibility of
their poses).

## Exercises

### Exercise 1 — more search
Redock gefitinib with `exhaustiveness=32` (≈ 2–3 min). Does the top pose change? Does the *best* RMSD among the nine
poses improve? Is this a search problem or a scoring problem?

### Exercise 2 — a larger box
Dock gefitinib again with a **30 Å** box (same centre). What happens to the time, the best score and the core RMSD?

### Exercise 3 — another scoring function
Vina ships **Vinardo** (`sf_name="vinardo"`). Redock gefitinib with it (recompute the maps — `dock()` does). Compare
the ranking of the poses and their core RMSDs with the Vina results. Do the two functions agree on the top pose?

### Exercise 4 — score-versus-size, on your own molecules
Take the 20 most potent and 20 least potent compounds of `egfr` **without** the size filter (`heavy_atoms.between`).
Dock them at `exhaustiveness=2` (≈ 5 min) and recompute both Spearman correlations. Does the size filter we used above
make the scoring function look better or worse than it is?

### Exercise 5 — interactions of the wrong poses
Compute the ProLIF fingerprints of *all nine* gefitinib poses and plot the Jaccard similarity to the crystal against
the core RMSD. Do the poses that lose the Met793 hydrogen bond also have a large core RMSD?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solutions (sketches)</b></summary>

```python
# 1
e32, p32 = dock(ligand_pdbqt, exhaustiveness=32)
display(results_table(e32, p32, crystal))
# the top pose (core ~0.5 A) is unchanged; the best full RMSD stays ~2.1-2.4 A: the scoring function, not the
# search, decides where the tail goes -> a scoring problem

# 2
e30, p30 = dock(ligand_pdbqt, exhaustiveness=8, box=[30, 30, 30])
display(results_table(e30, p30, crystal))

# 3
ev, pv = dock(ligand_pdbqt, exhaustiveness=8, sf_name="vinardo")
display(results_table(ev, pv, crystal))

# 4
top = egfr.nlargest(20, "pIC50"); bottom = egfr.nsmallest(20, "pIC50")
sel = pd.concat([top, bottom])
sel["score"] = [dock(ligand_to_pdbqt(ligand_from_smiles(s)), exhaustiveness=2, n_poses=1)[0][0, 0] for s in sel.smiles]
print(spearmanr(sel.score, sel.pIC50), spearmanr(sel.score, sel.heavy_atoms))

# 5
ligs = [plf_ligand(crystal)] + [plf_ligand(poses, c) for c in range(9)]
fp9 = plf.Fingerprint(["HBAcceptor", "HBDonor", "XBDonor", "Hydrophobic"]); fp9.run_from_iterable(ligs, protein, progress=False)
t9 = fp9.to_dataframe().T.droplevel("ligand").astype(int); t9.columns = ["crystal"] + [f"pose {k+1}" for k in range(9)]
full, core = pose_rmsd(poses, crystal)
jac = [jaccard(t9["crystal"], t9[f"pose {k+1}"]) for k in range(9)]
plt.scatter(core, jac); plt.xlabel("core RMSD (A)"); plt.ylabel("Jaccard to crystal"); plt.show()
```
</details>

## Further reading
- TeachOpenCADD **T015** (docking with smina) and **T016** (interactions with PLIP), **T014** (binding-site detection).
- Trott & Olson, *AutoDock Vina*, J. Comput. Chem. **2010**, 31, 455; Eberhardt *et al.*, *AutoDock Vina 1.2.0*, J. Chem. Inf. Model. **2021**, 61, 3891.
- Meeko documentation: <https://meeko.readthedocs.io>; ProLIF documentation: <https://prolif.readthedocs.io>.
- Buttenschoen *et al.*, *PoseBusters*, Chem. Sci. **2024** — what a physically valid pose is, and how often AI docking fails it.
- Warren *et al.*, *A critical assessment of docking programs and scoring functions*, J. Med. Chem. **2006**, 49, 5912 — still the reference on "pose yes, affinity no".
"""
