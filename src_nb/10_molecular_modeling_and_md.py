# %% [markdown]
"""
# 10 · Molecular modeling basics: conformers, force fields and molecular dynamics

**Chemoinformatics practicals — Session 10 of 11**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

> ⚡ **Tip.** This notebook runs on CPU (the simulation takes about two minutes), but is ~20× faster with a GPU: in
> Colab choose *Runtime → Change runtime type → T4 GPU* before you start, and the simulation becomes ten times longer
> for the same wait.

**What we are going to do, in one paragraph.** Until now a molecule was a *graph*: atoms and bonds, no shape. Today we
give it a shape (section 1), learn how a computer scores a shape as "comfortable" or "strained" (section 2), and then
let a small molecule *move* — we put a two-residue peptide in a box of water, apply Newton's laws to every atom, and
record a movie of a few tens of picoseconds of its life (section 3). Then we **watch** the movie (section 4) and learn
how to turn 300 000 numbers of trajectory into three plots that mean something (section 5).

**Learning goals.** After this session you will be able to
- generate 3D **conformers** of a small molecule and explain why one SMILES corresponds to many geometries;
- say what a molecular-mechanics **force field** is (balls and springs, with a formula) and read a torsion energy profile;
- explain, in plain words, the ingredients of a **molecular dynamics** (MD) simulation: time step, thermostat, barostat, periodic box, water;
- build a solvated system, run a short MD simulation with **OpenMM**, and **watch it** as a movie;
- analyse a trajectory with **MDAnalysis**: check that the run is healthy, measure how much the molecule moves (RMSD), and reduce its shape to two angles (the **Ramachandran plot**) — and understand *why* those two angles;
- situate MD among the tools of computer-aided drug design (docking, binding free energies, 3D representations for ML).

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - **TeachOpenCADD** talktorials **T019 · Molecular dynamics simulation** and **T020 · Analyzing molecular dynamics simulations** (Volkamer lab; [GitHub](https://github.com/volkamerlab/teachopencadd), CC BY 4.0) and **T033** (conformers);
> - **IBM3202 – Molecular Modeling and Simulation** Colab tutorials `lab07_MDsims`, `lab08_MDanalysis` by the **pb3lab** (Pontificia Universidad Católica de Chile; [GitHub](https://github.com/pb3lab/ibm3202), MIT);
> - the **OpenMM cookbook & tutorials** ([GitHub](https://github.com/openmm/openmm-cookbook), MIT) — the solvated villin head-piece file `villin.pdb` and the OpenMM test system `alanine-dipeptide-explicit.pdb` ([OpenMM](https://github.com/openmm/openmm), MIT/LGPL);
> - the **CCPBioSim** *biosim-analysis-workshop* ([GitHub](https://github.com/CCPBioSim/biosim-analysis-workshop), CC BY-SA 4.0) for the analysis logic;
> - [MDAnalysis](https://www.mdanalysis.org) user guide (GPL-2.0+ for the code; docs CC BY-SA).
"""

# %%
# @title ⚙️ Setup — run this cell first (≈1–2 min on Colab)
import sys, os, subprocess, time
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "openmm", "MDAnalysis", "py3Dmol", "plotly", "ipywidgets"], check=False)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolTransforms, rdMolAlign
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
import py3Dmol
import openmm as mm
import openmm.app as app
from openmm import unit, Vec3
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.dihedrals import Ramachandran
import plotly.graph_objects as go
from ipywidgets import interact
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning); warnings.filterwarnings("ignore", category=UserWarning)

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

platforms = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
print("OpenMM", mm.__version__, "| platforms:", platforms)
GPU = any(p in platforms for p in ("CUDA", "OpenCL"))
print("GPU available for MD:", GPU)

# %% [markdown]
"""
## 1. From a SMILES to many 3D structures: conformers

A SMILES fixes the *topology*; the *geometry* is not unique. Rotatable bonds give a molecule a whole landscape of
**conformers**, of which several may be populated at room temperature. Generating them is the first step of any 3D method.

RDKit's **ETKDG** (Experimental-Torsion Knowledge Distance Geometry; Riniker & Landrum 2015) embeds random
coordinates that satisfy the distance constraints of the graph, then corrects torsions using preferences learned from crystal
structures. We then relax each conformer with the **MMFF94** force field.
"""

# %% [markdown]
"""
### Step 1 · The molecule

Gefitinib again — the EGFR inhibitor of sessions 04–09. Count its rotatable bonds as you look at the drawing: the
methoxy, the anilino link, the whole propoxy–morpholine tail. Eight of them, and each can sit at several angles. The
2D drawing below is *one* molecule; the 3D question "what does it look like?" has hundreds of answers.
"""

# %%
smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"     # gefitinib, an EGFR kinase inhibitor
mol = Chem.MolFromSmiles(smiles)
mol

# %% [markdown]
"""
### Step 2 · Generate conformers

Four things happen in this cell, in order:

- `AddHs` — a 3D structure needs its hydrogens; the SMILES leaves them implicit.
- `EmbedMultipleConfs` with **ETKDG** guesses 50 sets of 3D coordinates. `randomSeed=42` makes the guess
  reproducible; `pruneRmsThresh=0.5` throws away any new conformer that is within 0.5 Å of one we already have, so
  the 50 requests give fewer, genuinely different shapes.
- `MMFFOptimizeMoleculeConfs` relaxes each guess with the MMFF94 force field (section 2 explains what that means) and
  returns its energy in kcal/mol.
- we subtract the lowest energy, so the best conformer is at 0 and every other number reads "this much worse".
"""

# %%
molH = Chem.AddHs(mol)
params = AllChem.ETKDGv3()
params.randomSeed = 42
params.pruneRmsThresh = 0.5            # discard conformers closer than 0.5 Å to an existing one
conf_ids = list(AllChem.EmbedMultipleConfs(molH, numConfs=50, params=params))
print(len(conf_ids), "distinct conformers embedded")

# Force-field optimisation returns (converged flag, energy in kcal/mol) for each conformer
results = AllChem.MMFFOptimizeMoleculeConfs(molH, maxIters=2000)
energies = np.array([e for _, e in results])
energies -= energies.min()
print("relative MMFF energies (kcal/mol):", np.round(np.sort(energies)[:10], 2), "...")

# %% [markdown]
"""
### Step 3 · The energy ladder

A histogram of those relative energies. The useful yardstick is the thermal energy at room temperature,
$k_BT \approx 0.6$ kcal/mol: conformers within one or two of those of the minimum are populated at 300 K, conformers
10 kcal/mol up essentially never occur. Notice how spread out the ladder is — most of what ETKDG proposed is chemically
possible but thermally irrelevant.
"""

# %%
plt.figure(figsize=(5, 3))
plt.hist(energies, bins=20)
plt.xlabel("MMFF94 energy relative to the minimum (kcal/mol)"); plt.ylabel("conformers")
plt.show()

# %% [markdown]
"""
### Step 4 · How different are they, really?

Energy does not tell you whether two conformers *look* different. For that we superpose each pair as well as possible
(rotate and shift one onto the other) and measure the **RMSD** — the root-mean-square distance between matching heavy
atoms. `GetBestRMS` does the superposition and also tries the symmetric atom orderings, so swapping the two equivalent
arms of the morpholine does not count as a difference. A mean pairwise RMSD near 1.7 Å says the conformers are not
small variations of one shape; the flexible tail lands in genuinely different places.
"""

# %%
# How different are the conformers? Heavy-atom RMSD after optimal superposition
heavy = Chem.RemoveHs(molH)
n = heavy.GetNumConformers()
rmsd_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        rmsd_matrix[i, j] = rmsd_matrix[j, i] = rdMolAlign.GetBestRMS(heavy, heavy, prbId=i, refId=j)
print(f"mean pairwise RMSD: {rmsd_matrix[np.triu_indices(n, 1)].mean():.2f} Å")

# %% [markdown]
"""
### Step 5 · Look at the five lowest

The five lowest-energy conformers, each aligned onto the best one, drawn in five colours. Rotate the picture with the
mouse. The rigid quinazoline–anilino core overlaps almost perfectly; the propoxy–morpholine tail fans out in every
direction. That is the general picture for a drug-like molecule: a rigid core that defines it and flexible parts that
adopt whatever the surroundings ask for — and the surroundings, in the end, are a protein pocket.
"""

# %%
# Overlay the 5 lowest-energy conformers, aligned on the lowest one
order = np.argsort(energies)
for cid in order[1:5]:
    rdMolAlign.AlignMol(heavy, heavy, prbCid=int(cid), refCid=int(order[0]))

view = py3Dmol.view(width=500, height=350)
colors = ["red", "orange", "yellow", "green", "blue"]
for k, cid in enumerate(order[:5]):
    view.addModel(Chem.MolToMolBlock(heavy, confId=int(cid)), "mol")
    view.setStyle({"model": k}, {"stick": {"color": colors[k], "radius": 0.15}})
view.zoomTo(); view.show()

# %% [markdown]
"""
> Where would you find *the* conformation of gefitinib? In the protein: PDB entry [4WKQ](https://www.rcsb.org/structure/4WKQ)
> shows it bound to EGFR. The bioactive conformation is usually **not** the global minimum in vacuum — one of the reasons
> flexible docking and MD exist.

### Exercise 1.1
Generate 50 conformers for **ibuprofen** (`CC(C)Cc1ccc(C(C)C(=O)O)cc1`) and for **caffeine**. Compare the number of
distinct conformers (after pruning) and the energy spread. Explain the difference in terms of rotatable bonds.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
for name, smi in [("ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"), ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C")]:
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3(); p.randomSeed = 0; p.pruneRmsThresh = 0.5
    ids = AllChem.EmbedMultipleConfs(m, 50, p)
    e = np.array([x[1] for x in AllChem.MMFFOptimizeMoleculeConfs(m)])
    print(f"{name:10s} {len(ids):2d} conformers, energy spread {e.max()-e.min():.1f} kcal/mol, "
          f"{Chem.rdMolDescriptors.CalcNumRotatableBonds(Chem.MolFromSmiles(smi))} rotatable bonds")
```
Caffeine is rigid (0 rotatable bonds): a single conformer survives pruning.
</details>
"""

# %% [markdown]
"""
## 2. What is a force field?

Quantum chemistry is too expensive for thousands of atoms and millions of time steps. **Molecular mechanics** replaces
electrons by a simple, parametrised potential energy function of the nuclear coordinates $\mathbf r$:

$$
E(\mathbf r) = \underbrace{\sum_{\text{bonds}} k_b (r-r_0)^2}_{\text{stretch}}
+ \underbrace{\sum_{\text{angles}} k_\theta (\theta-\theta_0)^2}_{\text{bend}}
+ \underbrace{\sum_{\text{torsions}} \sum_n V_n\,[1+\cos(n\phi-\gamma)]}_{\text{torsion}}
+ \underbrace{\sum_{i<j} 4\varepsilon_{ij}\Big[\big(\tfrac{\sigma_{ij}}{r_{ij}}\big)^{12}-\big(\tfrac{\sigma_{ij}}{r_{ij}}\big)^{6}\Big]}_{\text{van der Waals}}
+ \underbrace{\sum_{i<j} \frac{q_i q_j}{4\pi\varepsilon_0 r_{ij}}}_{\text{electrostatics}}
$$

The constants ($k_b, r_0, V_n, \sigma, \varepsilon, q$) are the **parameters**, fitted to experiment and/or quantum
calculations for each **atom type**. Families you will meet: **MMFF94** and **UFF** (general small molecules, in RDKit),
**AMBER**, **CHARMM**, **OPLS** (proteins, nucleic acids, lipids), **GAFF** / **OpenFF** (drug-like ligands for use with AMBER).

Let's *see* the torsion term with the MMFF94 energy profile of butane around its central C–C bond.
"""

# %% [markdown]
"""
### Step 1 · A relaxed torsion scan

Butane has one interesting bond: the central C–C. We turn it in 10° steps from −180° to 180° and ask the force field
how much energy each angle costs. Two details make this a *relaxed* scan rather than a naive one: at each angle the
dihedral is **held** with a stiff constraint (`MMFFAddTorsionConstraint`), and everything else — bond lengths, the
other angles, the hydrogens — is **allowed to relax** (`Minimize`). Without the relaxation, the hydrogens of the two
methyl groups would crash into each other at the eclipsed angles and the barriers would come out far too high.
"""

# %%
butane = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
AllChem.EmbedMolecule(butane, randomSeed=1)
AllChem.MMFFOptimizeMolecule(butane)
carbons = [a.GetIdx() for a in butane.GetAtoms() if a.GetAtomicNum() == 6]

angles = np.arange(-180, 181, 10)
profile = []
for ang in angles:
    m = Chem.Mol(butane)
    rdMolTransforms.SetDihedralDeg(m.GetConformer(), *carbons, float(ang))
    ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m))
    ff.MMFFAddTorsionConstraint(*carbons, False, float(ang) - 0.5, float(ang) + 0.5, 1000.0)   # hold the dihedral
    ff.Minimize(maxIts=500)                                                                  # relax everything else
    profile.append(ff.CalcEnergy())
profile = np.array(profile) - min(profile)

# %% [markdown]
"""
### Step 2 · The profile

Energy against dihedral angle, with the minimum set to zero. You know this curve from organic chemistry — but there it
was a sketch, and here it is *computed*, from the torsion term of the force field plus the van der Waals repulsion of
the hydrogens. The two together reproduce the textbook.
"""

# %%
plt.figure(figsize=(6, 3.2))
plt.plot(angles, profile, "o-")
for x, lab in [(-180, "anti"), (-60, "gauche"), (0, "syn (eclipsed)"), (60, "gauche"), (180, "anti")]:
    plt.annotate(lab, (x, profile[list(angles).index(x)] + 0.3), ha="center", fontsize=8)
plt.xlabel("C–C–C–C dihedral (°)"); plt.ylabel("MMFF94 energy (kcal/mol)"); plt.title("Torsional profile of butane")
plt.show()

# %% [markdown]
"""
The classic textbook curve: *anti* minimum, *gauche* minima ~0.8 kcal/mol higher, eclipsed barriers of 3–5 kcal/mol.
At 300 K, $k_BT \approx 0.6$ kcal/mol, so both minima are populated and barriers are crossed on the picosecond scale —
exactly the motions MD will show us.

### Exercise 2.1
Scan the O=C–O–C dihedral of methyl acetate `COC(C)=O` (find the 4 atom indices with `mol.GetSubstructMatch(Chem.MolFromSmarts("O=C-O-C"))`).
Which conformation is preferred, and roughly how much higher is the other minimum?
"""

# %%
# YOUR CODE HERE


# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
ester = Chem.AddHs(Chem.MolFromSmiles("COC(C)=O"))
AllChem.EmbedMolecule(ester, randomSeed=2); AllChem.MMFFOptimizeMolecule(ester)
idx = ester.GetSubstructMatch(Chem.MolFromSmarts("O=C-O-C"))
print("dihedral atoms:", idx)

angles = np.arange(-180, 181, 15)
prof = []
for ang in angles:
    m = Chem.Mol(ester)
    rdMolTransforms.SetDihedralDeg(m.GetConformer(), *idx, float(ang))
    ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m))
    ff.MMFFAddTorsionConstraint(*idx, False, float(ang) - 0.5, float(ang) + 0.5, 1000.0)
    ff.Minimize(maxIts=500)
    prof.append(ff.CalcEnergy())
prof = np.array(prof) - min(prof)
plt.plot(angles, prof, "o-"); plt.xlabel("O=C-O-C dihedral (deg)"); plt.ylabel("MMFF energy (kcal/mol)"); plt.show()
print("energy at 0 deg (Z/syn):", round(prof[list(angles).index(0)], 2),
      " at 180 deg (E/anti):", round(prof[list(angles).index(180)], 2))
```

The **Z (syn)** ester conformation, with the carbonyl O and the methyl on the same side, is strongly preferred;
the E (anti) form sits several kcal/mol higher (experimentally ~5 kcal/mol for methyl acetate). This is why esters
are drawn Z by default, and why a force field that got this term wrong would misplace every ester side chain.
</details>
"""


# %% [markdown]
"""
## 3. Molecular dynamics: letting the molecule move

### The idea in three sentences

A force field (section 2) gives the energy of any arrangement of atoms — and therefore the **force** on every atom
(force = minus the slope of the energy). Newton tells us what a force does: it accelerates the atom. So if we know all
the positions now, we can compute all the forces, move every atom a tiny bit, and repeat. Do that a few million times
and you have a **movie** of the molecule's life. That is molecular dynamics.

### The ingredients, in plain words

| ingredient | what it is | why we need it | what we use |
|---|---|---|---|
| **time step** | how far forward each "repeat" moves the clock | the fastest motion (a bond to a hydrogen vibrates every ~10 fs) must be caught, or the atoms fly apart | 2 fs, with the X–H bonds frozen |
| **thermostat** | a gentle random kick plus friction on every atom | keeps the temperature at 300 K, like a molecule in a warm bath | Langevin |
| **barostat** | occasionally shrinks or grows the box | keeps the pressure at 1 bar, like a piston on a cylinder | Monte Carlo barostat |
| **water** | explicit water molecules around the solute | a molecule in vacuum behaves nothing like one in solution | TIP3P |
| **periodic box** | the box is copied infinitely in every direction; an atom leaving through one face comes back through the opposite one | fakes an infinite liquid with a few hundred molecules and no surface | a 3 nm cube |
| **long-range electrostatics** | a trick for adding up charge interactions with all the infinite copies | Coulomb forces decay slowly, you cannot just cut them off | PME |

Numbers to keep in mind: a 2 fs step means **500 000 steps for one nanosecond**. A modern GPU manages about a
microsecond per day for a small protein; a laptop CPU, a few nanoseconds. Today's simulation is 20 ps on CPU, 200 ps
on GPU — short, on purpose, so that it finishes while you read.

### The molecule: alanine dipeptide

We simulate the smallest thing that behaves like a protein backbone: **one alanine**, capped at both ends (an acetyl
group, ACE, and an N-methyl amide, NME) so that it has a real peptide bond on each side. Twenty-two atoms. It is called
the "hydrogen atom of protein folding": every question about how a protein backbone moves can be asked here first.
"""

# %% [markdown]
"""
### Step 1 · Get the peptide

The course ships alanine dipeptide in a PDB file that already contains some water. We keep only the **peptide** from it
and add our own water in the next step, so that you see where every atom in the box comes from. `Modeller` is OpenMM's
tool for editing a structure: adding or deleting atoms, water, ions, hydrogens.
"""

# %%
pdb = app.PDBFile(fetch("md/alanine_dipeptide_solvated.pdb"))
modeller = app.Modeller(pdb.topology, pdb.positions)
modeller.deleteWater()
peptide_atoms = modeller.topology.getNumAtoms()
print("alanine dipeptide:", peptide_atoms, "atoms in", [r.name for r in modeller.topology.residues()])

view = py3Dmol.view(width=450, height=300)
with open("ala2_dry.pdb", "w") as f:
    app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
view.addModel(open("ala2_dry.pdb").read(), "pdb")
view.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
view.addResLabels({"resn": ["ACE", "ALA", "NME"]}, {"fontSize": 13, "backgroundOpacity": 0.6})
view.zoomTo(); view.show()

# %% [markdown]
"""
Three residues: the acetyl cap, the alanine, the N-methyl cap. Turn it with the mouse. The two bonds that matter for
the whole session are the ones on either side of the alanine's central carbon (the Cα): everything else in this
molecule is rigid, and those two bonds are where all the interesting motion happens. We will come back to them in
section 5.

### Step 2 · Add the water

`addSolvent` puts the peptide in the centre of a cube and fills the rest with water molecules taken from a
pre-equilibrated liquid — so the density is right from the start. A 3 nm cube holds about 870 waters. The force field
has to be given here because the water model (TIP3P) is part of it.

Why not simulate in vacuum and save all that computing? Because a molecule in vacuum is a different molecule: no
hydrogen-bonding partners, no screening of charges, nothing to bump into. The water is not decoration; it is most of
the physics, and most of the cost — 2600 of our 2620 atoms.
"""

# %%
forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")      # protein force field + water model
modeller.addSolvent(forcefield, model="tip3p", boxSize=Vec3(3.0, 3.0, 3.0) * unit.nanometer)

n_atoms = modeller.topology.getNumAtoms()
n_waters = sum(1 for r in modeller.topology.residues() if r.name == "HOH")
box = [v[i].value_in_unit(unit.nanometer) for i, v in enumerate(modeller.topology.getPeriodicBoxVectors())]
density = n_waters * 18.015 / 6.022e23 / (np.prod(box) * 1e-21)
print(f"{n_atoms} atoms = {peptide_atoms} peptide + {n_waters} waters × 3 | box {box} nm | density {density:.2f} g/cm³")

with open("ala2_system.pdb", "w") as f:                # the starting structure: our topology for everything that follows
    app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

# %% [markdown]
"""
### Step 3 · Look at what we built

Two views of the same box. On the left, everything: the peptide as sticks in the middle, the water as thin lines, and
the periodic box drawn in magenta. It should look *full* — a liquid, not a mist. On the right, only the peptide and the
water molecules within 4 Å of it: its first hydration shell, the molecules it actually touches.
"""

# %%
system_pdb = open("ala2_system.pdb").read()
half = [b * 10 / 2 for b in box]                        # box centre, in Å (PDB units)

view = py3Dmol.view(width=900, height=380, viewergrid=(1, 2))
view.addModel(system_pdb, "pdb", viewer=(0, 0))
view.setStyle({"resn": "HOH"}, {"line": {"colorscheme": "cyanCarbon", "opacity": 0.6}}, viewer=(0, 0))
view.setStyle({"not": {"resn": "HOH"}}, {"stick": {"radius": 0.3}}, viewer=(0, 0))
view.addBox({"center": {"x": half[0], "y": half[1], "z": half[2]},
             "dimensions": {"w": box[0] * 10, "h": box[1] * 10, "d": box[2] * 10}, "color": "magenta", "wireframe": True}, viewer=(0, 0))
view.zoomTo(viewer=(0, 0))

view.addModel(system_pdb, "pdb", viewer=(0, 1))
view.setStyle({}, {}, viewer=(0, 1))                   # hide everything, then show the peptide and its shell
view.setStyle({"not": {"resn": "HOH"}}, {"stick": {}, "sphere": {"scale": 0.25}}, viewer=(0, 1))
view.setStyle({"resn": "HOH", "within": {"distance": 4.0, "sel": {"not": {"resn": "HOH"}}}},
              {"stick": {"radius": 0.12, "colorscheme": "cyanCarbon"}}, viewer=(0, 1))
view.zoomTo({"not": {"resn": "HOH"}}, viewer=(0, 1))
view.show()

# %% [markdown]
"""
### Step 4 · The four OpenMM objects

Every OpenMM simulation is built from the same four objects, always in this order. It is worth learning them by name,
because every MD script you will ever read — including session 11 — is this pattern with different inputs.

1. **`ForceField`** — the rulebook: which atom types exist and what their parameters are (section 2). Already created above.
2. **`System`** — the rulebook *applied to our atoms*: the complete list of every bond, angle, torsion and non-bonded
   pair in the box, with its parameters. `createSystem` builds it from the topology; the arguments say how to treat
   the long-range electrostatics (`PME`), where to cut the short-range forces (1 nm), and which bonds to freeze
   (`HBonds` — every bond to a hydrogen, which is what allows the 2 fs step). Then we *add a force* to it: the barostat,
   which is not a physical interaction but is implemented as one.
3. **`Integrator`** — the rule for advancing the clock. `LangevinMiddleIntegrator(300 K, 1/ps, 2 fs)` reads: keep the
   temperature at 300 K, with a friction of 1 per picosecond (how strongly the "bath" is coupled), stepping 2 fs at a time.
   The thermostat is built into the integrator.
4. **`Simulation`** — glues the three together with a set of starting positions, and picks the fastest platform available
   (CUDA on a GPU, otherwise CPU). This is the object we actually drive.
"""

# %%
system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=app.PME,                 # long-range electrostatics
    nonbondedCutoff=1.0 * unit.nanometer,    # short-range forces cut off at 1 nm
    constraints=app.HBonds,                  # rigid X–H bonds -> 2 fs step is safe
)
system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25))   # constant pressure (NPT)

integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds)
integrator.setRandomNumberSeed(1)                                                # reproducible thermostat noise

simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)

print("running on platform:", simulation.context.getPlatform().getName())
print("force terms in the System:", [f.__class__.__name__ for f in system.getForces()])

# %% [markdown]
"""
Read the list of force terms and match it with the formula of section 2: `HarmonicBondForce` is the stretch term,
`HarmonicAngleForce` the bend, `PeriodicTorsionForce` the torsion, `NonbondedForce` is van der Waals **and**
electrostatics together. `CMMotionRemover` stops the whole box from drifting; `MonteCarloBarostat` is our piston.

### Step 5 · Minimise first

The starting structure has water molecules that were dropped in from a template and may sit too close to the peptide
or to each other. Two atoms 0.5 Å apart carry an enormous repulsive force; start the dynamics like that and they shoot
off at thousands of metres per second, and the simulation "explodes" (you would see `NaN` in the energies).

**Minimisation** slides every atom downhill in energy — no temperature, no time, just "relax" — until the worst
clashes are gone. Watch the potential energy: it should drop by a lot, and the structure barely changes to the eye.
"""

# %%
E0 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
simulation.minimizeEnergy(maxIterations=500)
E1 = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
print(f"potential energy before minimisation: {E0:12,.0f} kJ/mol")
print(f"potential energy after  minimisation: {E1:12,.0f} kJ/mol")

# %% [markdown]
"""
### Step 6 · Run it — and record the movie

A simulation only produces what you ask it to record. **Reporters** are OpenMM's recorders, and we attach three:

- a **`DCDReporter`** that writes the positions of every atom to a file every `report_every` steps — each saved set of
  positions is a **frame**, and the file of frames is the **trajectory**. This is the movie. We save 100 frames;
- a **`StateDataReporter`** to a CSV file with the temperature, energy, volume and speed at the same moments — the
  simulation's dashboard, which section 5 reads first;
- a second `StateDataReporter` to the screen, five times during the run, so you can see it is alive.

Before starting, `setVelocitiesToTemperature` gives every atom a random velocity drawn from the 300 K distribution
(otherwise the atoms would start frozen at 0 K). Then `simulation.step(n_steps)` is the whole of molecular dynamics:
compute forces, move, repeat, `n_steps` times.
"""

# %%
n_steps = 100_000 if GPU else 10_000         # 200 ps on a GPU, 20 ps on CPU (about 2 minutes)
report_every = n_steps // 100                # -> 100 frames whatever the length of the run
dt_frame = report_every * 0.002              # picoseconds between two saved frames

simulation.reporters.clear()
simulation.reporters.append(app.DCDReporter("ala2_traj.dcd", report_every))
simulation.reporters.append(app.StateDataReporter("ala2_log.csv", report_every, step=True, time=True,
                                                  potentialEnergy=True, temperature=True, volume=True, speed=True))
simulation.reporters.append(app.StateDataReporter(sys.stdout, n_steps // 5, step=True, time=True,
                                                  temperature=True, speed=True, remainingTime=True, totalSteps=n_steps))

simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
t0 = time.time()
simulation.step(n_steps)
print(f"\nsimulated {n_steps * 2 / 1000:.0f} ps in {time.time() - t0:.0f} s  ->  100 frames, {dt_frame:.1f} ps apart, in ala2_traj.dcd")

with open("ala2_final.pdb", "w") as f:       # the last frame, as an ordinary PDB file
    app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)

# %% [markdown]
"""
The printed lines are the simulation's heartbeat: the temperature should sit near 300 K from the first report onwards
(the thermostat works fast), and the speed, in nanoseconds per simulated day, tells you what a longer run would cost.
On two CPU cores expect ~20 ns/day; on a T4 GPU, several hundred.

## 4. Watch it

### Step 1 · Load the trajectory

**MDAnalysis** is the library for *reading* simulations. It needs two things: a **topology** (which atoms exist and
how they are connected — our starting PDB) and a **trajectory** (where they were at each frame — the DCD). Together
they make a `Universe`. Atoms are picked with a small selection language, close to plain English:
`"not resname HOH"` is everything that is not water.
"""

# %%
u = mda.Universe("ala2_system.pdb", "ala2_traj.dcd")
peptide = u.select_atoms("not resname HOH")
water = u.select_atoms("resname HOH")
print(u, "|", len(u.trajectory), "frames,", f"{dt_frame:.1f} ps apart")
print(peptide.n_atoms, "peptide atoms |", water.n_residues, "water molecules")

# %% [markdown]
"""
### Step 2 · Hold the camera still

A movie of the raw trajectory is disappointing: the peptide tumbles and drifts through the box, so all you see is a
molecule slowly rotating. What we want to see is how it **changes shape**. So before filming we **align** every frame
onto the first one, using the peptide backbone as the anchor: each frame is rotated and shifted so that the backbone
atoms sit as close as possible to where they were in frame 0. Rotation and drift disappear; only the internal motion
is left. This is what every MD movie you have ever seen does, and it is also the first step of the RMSD calculation
in section 5.

`in_memory=True` means the aligned coordinates replace the ones read from the file, for the rest of the notebook.
"""

# %%
align.AlignTraj(u, u, select="not resname HOH and backbone", in_memory=True).run()
print("all", len(u.trajectory), "frames aligned on the peptide backbone of frame 0")

# %% [markdown]
"""
### Step 3 · The movie of the peptide

We write the peptide's coordinates in every frame to one PDB file (a *multi-model* PDB: 100 structures one after the
other), and hand it to py3Dmol as an animation. **Press play** with the ▶ that appears; drag to rotate; scroll to zoom.

What to look for: the two caps and the alanine's methyl swing around; the backbone wobbles; the N–H and C=O groups
vibrate. What you will probably *not* see in 20 ps is a real change of shape — a flip of the backbone from one
conformation to another. That absence is a lesson in itself, and section 5 puts a number on it.
"""

# %%
def write_movie(atoms, filename, max_frames=100):
    """Write an AtomGroup at up to max_frames evenly spaced frames as a multi-model PDB (for py3Dmol)."""
    frames = np.linspace(0, len(u.trajectory) - 1, min(max_frames, len(u.trajectory))).astype(int)
    with mda.Writer(filename, atoms.n_atoms, multiframe=True) as w:
        for fr in frames:
            u.trajectory[fr]
            w.write(atoms)
    return open(filename).read()

movie_peptide = write_movie(peptide, "movie_peptide.pdb")

view = py3Dmol.view(width=550, height=400)
view.addModelsAsFrames(movie_peptide, "pdb")
view.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
view.animate({"loop": "backAndForth", "interval": 60})
view.zoomTo(); view.show()

# %% [markdown]
"""
### Step 4 · The same movie with the water it touches

Now add the water — but not all 870 molecules, which would hide the peptide. We take the molecules that were within
4 Å of the peptide **in the first frame** and follow *those same molecules* through the run. Because the camera is
fixed on the peptide, you see them do what liquid water does: rattle in place for a moment, then wander off and get
replaced by others (which we are not drawing). A water molecule stays in the first shell of a small solute for only a
few picoseconds. Nothing is broken when they leave — that *is* the liquid.
"""

# %%
u.trajectory[0]
shell = u.select_atoms("byres (resname HOH and around 4 (not resname HOH))")
print(shell.n_residues, "water molecules within 4 Å of the peptide in frame 0")

movie_shell = write_movie(peptide + shell, "movie_shell.pdb")
view = py3Dmol.view(width=550, height=400)
view.addModelsAsFrames(movie_shell, "pdb")
view.setStyle({"resn": "HOH"}, {"stick": {"radius": 0.1, "colorscheme": "cyanCarbon"}})
view.setStyle({"not": {"resn": "HOH"}}, {"stick": {}, "sphere": {"scale": 0.25}})
view.animate({"loop": "backAndForth", "interval": 60})
view.zoomTo({"not": {"resn": "HOH"}}); view.show()

# %% [markdown]
"""
### Step 5 · Any single frame, on demand

The movie shows everything at once; sometimes you want to stop at one moment and look. Move the slider: the viewer
redraws that frame, and the text reports the simulation time. (Section 5 attaches plots to this same slider.)
"""

# %%
def show_frame(frame):
    u.trajectory[frame]
    peptide.write("_frame.pdb")
    v = py3Dmol.view(width=450, height=320)
    v.addModel(open("_frame.pdb").read(), "pdb")
    v.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
    v.zoomTo(); v.show()
    print(f"frame {frame} of {len(u.trajectory) - 1}  =  t = {frame * dt_frame:.1f} ps")

interact(show_frame, frame=(0, len(u.trajectory) - 1, 1));

# %% [markdown]
"""
## 5. Measure it

Watching is necessary but not sufficient: your eye cannot tell 300 K from 320 K, or 1.2 Å of motion from 1.8. Three
kinds of numbers, from the coarsest to the finest.

### 5.1 Is the simulation healthy?

Before analysing anything, read the dashboard. Three curves from the CSV log, and what a healthy run looks like:

- **Temperature**: climbs to 300 K within the first picosecond (the thermostat) and then fluctuates around it. The
  fluctuations are real physics — a small system has a visible temperature noise, about ±5–10 K here — not an error.
- **Potential energy**: drifts down at the start as the box settles, then flattens. A steady *upward* drift, or spikes,
  means something is wrong.
- **Box volume**: the barostat's work. It should hover around 26–27 nm³ (the 3 nm cube was already the right density,
  so it only breathes). If it collapsed or blew up, the starting density was wrong.

The first few picoseconds, while these curves are still moving, are **equilibration**; you do not analyse them. Real
studies equilibrate for nanoseconds and throw that part away.
"""

# %%
log = pd.read_csv("ala2_log.csv")
log.columns = [c.split(" (")[0].strip('#"') for c in log.columns]
fig, axes = plt.subplots(1, 3, figsize=(13, 3))
axes[0].plot(log["Time"], log["Temperature"]); axes[0].axhline(300, c="r", ls="--"); axes[0].set_ylabel("temperature (K)")
axes[1].plot(log["Time"], log["Potential Energy"]); axes[1].set_ylabel("potential energy (kJ/mol)")
axes[2].plot(log["Time"], log["Box Volume"]); axes[2].set_ylabel("box volume (nm³)")
for ax in axes: ax.set_xlabel("time (ps)")
plt.tight_layout(); plt.show()
settled = log[log["Time"] > 2]                  # skip the first 2 ps of equilibration
print(f"after 2 ps: temperature {settled['Temperature'].mean():.0f} ± {settled['Temperature'].std():.0f} K | "
      f"volume {settled['Box Volume'].mean():.1f} ± {settled['Box Volume'].std():.2f} nm³ | speed {log['Speed'].iloc[-1]:.0f} ns/day")

# %% [markdown]
"""
### 5.2 How much does the molecule move? The RMSD

The **root-mean-square deviation** answers one question: *how far, on average, are the atoms from where they were in
the reference frame?* For each frame, the structure is superposed on the reference (as in the movie), the distance
each chosen atom moved is squared, averaged over the atoms, and square-rooted. One number per frame, in ångström.

Which atoms you choose changes the answer, and that is the point of the widget below:

- the **backbone** (N, Cα, C, O — the chain itself) moves least: it is held by the peptide bonds;
- **all heavy atoms** adds the methyl groups, which rotate freely — more motion;
- **everything including hydrogens** adds the fastest, least interesting vibrations — more still;
- **just the Cα** is a single atom — try it, and read what the widget says instead of a curve.

The **reference frame** matters too. Frame 0 is the usual choice ("how far from the start?"), but a frame in the
middle of the run gives a different curve — a structure is never "far" from itself. Pick the atoms and the reference,
and read the mean and maximum printed below the plot.
"""

# %%
SELECTIONS = {
    "backbone (N, Cα, C, O)": "not resname HOH and backbone",
    "all heavy atoms": "not resname HOH and not name H*",
    "everything, hydrogens too": "not resname HOH",
    "just the Cα": "not resname HOH and name CA",
}
time_ps = np.arange(len(u.trajectory)) * dt_frame

def rmsd_curve(selection, reference_frame=0):
    """RMSD (Å) of the chosen atoms to the chosen reference frame, after optimal superposition."""
    return rms.RMSD(u, u, select=SELECTIONS[selection], ref_frame=reference_frame).run().results.rmsd[:, 2]

def show_rmsd(selection, reference_frame=0):
    n_sel = u.select_atoms(SELECTIONS[selection]).n_atoms
    if n_sel < 3:
        print(f"{n_sel} atom selected. The superposition step can always put 1 atom (or 2) exactly onto the reference,\n"
              "so the RMSD is zero by construction and the fit itself is undefined - MDAnalysis returns NaN.\n"
              "You need at least three atoms that are not on a line before 'how much did it move' means anything.")
        return
    y = rmsd_curve(selection, reference_frame)
    plt.figure(figsize=(7, 3))
    plt.plot(time_ps, y)
    plt.axvline(reference_frame * dt_frame, c="r", ls="--", label=f"reference = frame {reference_frame}")
    plt.xlabel("time (ps)"); plt.ylabel("RMSD (Å)"); plt.title(selection); plt.legend(); plt.show()
    print(f"{u.select_atoms(SELECTIONS[selection]).n_atoms} atoms | mean RMSD {y.mean():.2f} Å | max {y.max():.2f} Å")

interact(show_rmsd, selection=list(SELECTIONS), reference_frame=(0, len(u.trajectory) - 1, 1));

# %% [markdown]
"""
For a molecule this small and this rigid, an RMSD of about half an ångström on the heavy atoms (a little over one
with the hydrogens included) means "wobbling around one shape"; you would need a jump to 2–3 Å to say the shape
changed. Keep those numbers in mind for session 11, where the same
measurement on a ligand in a protein pocket decides whether a docking pose was right.

### 5.3 Two angles say it all: the Ramachandran plot

**Why do we need this?** The RMSD says *how much* the molecule moved, not *where it went*. And the raw trajectory —
2620 atoms × 3 coordinates × 100 frames, three-quarters of a million numbers — is unreadable. We need a small number
of quantities that capture *the shape*. Choosing them well is most of the art of analysing a simulation.

For a peptide backbone the choice is made for us by chemistry. Bond lengths do not change; bond angles hardly change;
the peptide bond itself (C–N) is flat and rigid. The **only real freedom** is rotation around the two bonds on either
side of each Cα — the ones we pointed at in section 3:

- **φ (phi)**: rotation about the N–Cα bond;
- **ψ (psi)**: rotation about the Cα–C bond.

Two numbers per residue describe the shape of the whole backbone. For our single alanine, **the entire conformation is
one point in a square from −180° to 180°** in each direction. That square is the **Ramachandran plot** (1963).

Two things make it more than a convenient plot. First, most of the square is *forbidden*: for most (φ, ψ) pairs, atoms
of neighbouring residues crash into each other. The allowed regions are a few islands, and they have names — the
**β** region (top left, extended chain, φ ≈ −120°, ψ ≈ +130°), the **right-handed α-helix** region (centre left,
φ ≈ −60°, ψ ≈ −45°), and a small **left-handed α** island (top right, φ > 0). Every protein structure ever solved has
its residues sitting on those islands; when crystallographers validate a new structure, the Ramachandran plot is the
first thing they check. Second, for us: the plot turns "did the molecule change shape?" into "did the point jump to a
different island?" — a question with a clear answer.

The code asks MDAnalysis for φ and ψ of the alanine in every frame (`Ramachandran`), then draws each frame as a point
coloured by time.
"""

# %%
rama = Ramachandran(u.select_atoms("resname ALA")).run()
phi_psi = rama.results.angles[:, 0, :]          # shape (frames, 2): one (phi, psi) pair per frame
phi, psi = phi_psi[:, 0], phi_psi[:, 1]

fig, ax = plt.subplots(figsize=(5.5, 5))
for (x0, x1, y0, y1, name) in [(-180, -45, 90, 180, "β / PPII"), (-100, -30, -80, 0, "right-handed α"), (30, 100, 0, 90, "left-handed α")]:
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="lightgray", ec="none", alpha=0.5)); ax.text(x0 + 5, y1 - 15, name, fontsize=8, color="dimgray")
sc = ax.scatter(phi, psi, c=time_ps, cmap="viridis", s=14, zorder=3)
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5)
ax.set_xlabel("φ (°)  — rotation about N–Cα"); ax.set_ylabel("ψ (°)  — rotation about Cα–C"); ax.set_title("Ramachandran plot of our alanine (colour = time)")
plt.colorbar(sc, label="time (ps)"); plt.show()

in_beta = ((phi < -45) & (psi > 90)).mean(); in_alpha = ((phi > -100) & (phi < -30) & (psi > -80) & (psi < 0)).mean()
print(f"frames in the β/PPII island: {in_beta:.0%}   in the right-handed α island: {in_alpha:.0%}")

# %% [markdown]
"""
Read your plot. The points form a small cloud on **one** island — most likely β/PPII, the extended shape the peptide
started in. The cloud's size is the wobbling the RMSD measured; the *absence* of points on the α island is the fact the
RMSD could not tell you: in 20 ps at 300 K, alanine dipeptide did **not** cross to the other stable shape, although
the α conformation is perfectly real (it is what every helix in every protein is made of) and only a few kJ/mol away.

That is the **sampling problem** in one picture. The crossing happens on the scale of hundreds of picoseconds to
nanoseconds — ten to a hundred times longer than we ran. A simulation shows you only what had time to happen, and
"the molecule stayed put" may mean "the molecule is stable" or may mean "we did not wait". Distinguishing the two is
what long runs, many repeats and the enhanced-sampling methods of Lecture 5 are for. If you have a GPU, the 200 ps run
sometimes catches a crossing: look for a second cloud.

### 5.4 The two angles along time — and every frame on demand

The same two numbers, now against time (hover to read the exact values; drag on the axis to zoom), and below it the
slider from section 4 again, this time showing **where the frame sits on the Ramachandran plot** next to the 3D
structure. Move the slider to the frames where φ or ψ jumps and look at what the molecule did.
"""

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_ps, y=phi, name="φ", mode="lines+markers", marker=dict(size=4)))
fig.add_trace(go.Scatter(x=time_ps, y=psi, name="ψ", mode="lines+markers", marker=dict(size=4)))
fig.add_trace(go.Scatter(x=time_ps, y=rmsd_curve("all heavy atoms"), name="heavy-atom RMSD (Å, right axis)", mode="lines",
                         line=dict(dash="dot", color="gray"), yaxis="y2"))
fig.update_layout(height=360, width=850, xaxis_title="time (ps)", yaxis_title="angle (°)", hovermode="x unified",
                  yaxis2=dict(title="RMSD (Å)", overlaying="y", side="right", range=[0, 3]),
                  legend=dict(orientation="h", y=1.15), margin=dict(t=30, b=40), xaxis=dict(rangeslider=dict(visible=True)))
fig.show()

# %% [markdown]
"""
And the frame explorer: the slider picks a frame, the plot shows where that frame sits on the Ramachandran map (red
dot), and the viewer shows the molecule at that moment. Go to the extremes of the cloud and compare the two shapes.
"""

# %%
def explore(frame):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(phi, psi, s=8, c="lightgray"); ax.scatter(phi[frame], psi[frame], s=120, c="red", zorder=3)
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5)
    ax.set_xlabel("φ (°)"); ax.set_ylabel("ψ (°)"); ax.set_title(f"frame {frame}  (t = {frame * dt_frame:.1f} ps)"); plt.show()
    print(f"φ = {phi[frame]:7.1f}°   ψ = {psi[frame]:7.1f}°")
    show_frame(frame)

interact(explore, frame=(0, len(u.trajectory) - 1, 1));

# %% [markdown]
"""
### Exercise 5.1
1. In the RMSD widget, why does "just the Cα" give no curve? What is the smallest number of atoms for which an RMSD after superposition can be non-zero, and why?
2. Compute the water density from the average box volume in `log` (count the waters with `water.n_residues`; M = 18.015 g/mol; 1 nm³ = 10⁻²¹ cm³). Is it close to 1.0 g/cm³? Why is it a little below?
3. (If you have a GPU) rerun the simulation at **400 K**: change the temperature in *both* the integrator and the barostat. How does the Ramachandran cloud change, and does the point visit the α island now?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution (1–2)</b></summary>

1. Superposition removes translation and rotation. A single atom can always be moved exactly onto its reference
   position, so its RMSD is zero by construction; two atoms can always be aligned too (rotate the pair onto the
   reference pair) — three non-collinear atoms are the smallest set with a non-zero RMSD, because a triangle can change
   its shape.

```python
V_cm3 = log["Box Volume"].mean() * 1e-21           # nm^3 -> cm^3
mass_g = water.n_residues * 18.015 / 6.022e23
print(f"density ≈ {mass_g / V_cm3:.3f} g/cm3")
```
2. About 0.97 g/cm³. Slightly below 1.0 because the volume we divide by includes the space taken by the peptide,
   and because TIP3P water at 300 K and 1 bar is itself a little less dense than real water.
</details>
"""

# %% [markdown]
"""
## 6. Optional (GPU): a real protein — the villin headpiece

The 35-residue villin headpiece (HP35) is a classic fast-folding mini-protein. The file below (from the OpenMM tutorials)
is already solvated. It is 8867 atoms against the dipeptide's 2269, and that is the point of the section: the same six
lines of OpenMM, a system four times larger.

Everything you learned on the dipeptide transfers directly: the same four objects, the same reporters, the same
MDAnalysis analysis. Two additions appear because this is a protein: the RMSD is now measured on the **Cα atoms only**
(one per residue — the standard choice for proteins, because it follows the fold and ignores the side chains), and a
new quantity, the **RMSF** — the root-mean-square *fluctuation* of each residue around its average position, which
tells you *which parts* of the protein move (loops and termini) and which are rigid (the helices).

On a T4 GPU the 100 ps take about a minute. On two CPU cores this system runs at roughly **4.6 ns/day**, so the same
100 ps take **about half an hour** — which is why the cell only runs when a GPU is present. Set `RUN_PROTEIN = True`
to force it anyway, and reduce `simv.step(50_000)` if you only want to watch the machinery work.
"""

# %%
RUN_PROTEIN = GPU
if RUN_PROTEIN:
    vil = app.PDBFile(fetch("md/villin_headpiece_solvated.pdb"))
    print("atoms:", vil.topology.getNumAtoms())
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    sysv = ff.createSystem(vil.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer, constraints=app.HBonds)
    integ = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds)
    simv = app.Simulation(vil.topology, sysv, integ)
    simv.context.setPositions(vil.positions)
    simv.minimizeEnergy(maxIterations=500)
    simv.reporters.append(app.DCDReporter("villin_traj.dcd", 500))
    simv.reporters.append(app.StateDataReporter(sys.stdout, 10_000, step=True, time=True, temperature=True, speed=True))
    simv.context.setVelocitiesToTemperature(300 * unit.kelvin)
    simv.step(50_000)                       # 100 ps
    with open("villin_final.pdb", "w") as f:
        app.PDBFile.writeFile(simv.topology, simv.context.getState(getPositions=True).getPositions(), f)

    uv = mda.Universe(fetch("md/villin_headpiece_solvated.pdb"), "villin_traj.dcd")
    prot = uv.select_atoms("protein")
    Rv = rms.RMSD(uv, uv, select="protein and name CA", ref_frame=0).run()
    # RMSF per residue after aligning on the C-alpha atoms
    aligner = align.AlignTraj(uv, uv, select="protein and name CA", in_memory=True).run()
    ca = uv.select_atoms("protein and name CA")
    rmsf = rms.RMSF(ca).run()
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    axes[0].plot(Rv.results.rmsd[:, 1], Rv.results.rmsd[:, 2]); axes[0].set_xlabel("time (ps)"); axes[0].set_ylabel("Cα RMSD (Å)")
    axes[1].bar(ca.resids, rmsf.results.rmsf); axes[1].set_xlabel("residue"); axes[1].set_ylabel("Cα RMSF (Å)")
    plt.tight_layout(); plt.show()
else:
    print("Skipped (no GPU detected). Set RUN_PROTEIN = True to run it on CPU (~30 min for 100 ps).")

# %% [markdown]
"""
## 7. Where MD sits in computer-aided drug design

| task | method | what you learn |
|---|---|---|
| Where does a ligand bind, and how? | **docking** (AutoDock Vina, Gnina; TeachOpenCADD T015 — **session 09**) | pose + crude score |
| Is the pose stable? Which interactions persist? | **protein–ligand MD** (T019/T020 → **session 11**; interaction fingerprints with ProLIF) | dynamics of the complex |
| How strongly does it bind? | **free-energy methods** (FEP, TI, MM/GBSA) | ΔG of binding, ± 1 kcal/mol |
| Which pockets exist? Are they druggable? | MD + pocket detection (T014) | cryptic pockets, flexibility |
| Features for machine learning | conformers, 3D pharmacophores, MD-derived descriptors, 3D/equivariant GNNs (T036) | representation beyond 2D |

**Protein–ligand MD in practice** needs three extra steps we skipped today: preparing the protein (missing atoms and
loops, protonation — `pdbfixer`, PROPKA), parametrising the ligand (GAFF or OpenFF), and much longer sampling (≥ 100 ns,
several replicas). Session 09 already prepared the EGFR receptor and docked gefitinib into it; **session 11** takes that
pose and does the rest (following TeachOpenCADD **T019/T020**).

## Exercises to finish
1. Re-run the alanine dipeptide simulation with a **4 fs** time step (keep `constraints=app.HBonds`). What happens to the temperature and energy, and why? (Hint: hydrogen mass repartitioning is what people do to get away with 4 fs.)
2. Replace the Langevin integrator by `mm.VerletIntegrator(2*unit.femtoseconds)` (no thermostat) and plot the temperature. Which ensemble is this?
3. Compute the **radius of gyration** of the peptide along the trajectory (`peptide.radius_of_gyration()` for each frame).
4. Generate 20 conformers of gefitinib and compute their NPR shape descriptors (session 02). How much does the *shape* change between conformers?

## Further reading
- TeachOpenCADD T019/T020 (protein–ligand MD with OpenMM, analysis with MDAnalysis) — adapted in session 11.
- OpenMM user guide: <http://docs.openmm.org/latest/userguide/> and the OpenMM cookbook.
- MDAnalysis user guide: <https://userguide.mdanalysis.org/>.
- Braun *et al.*, *Best practices for foundations in molecular simulations*, Living J. Comp. Mol. Sci. **2019**, 1, 5957.
- Höltje, Sippl, Rognan, Folkers, *Molecular Modeling: Basic Principles and Applications* (course bibliography).

---

**Next: from a peptide in water back to the drug in its protein.** Session 09 asked *where* gefitinib binds to EGFR and
found that a docking score is not an affinity. Session 11 takes the docked (or crystal) pose and does for the whole
protein–ligand complex what we did today for alanine dipeptide: parametrise, solvate, equilibrate, simulate, analyse —
and asks whether the pose survives.

Notice what sessions 09–11 change about sessions 02–08. Everything there described molecules as **graphs**: a
fingerprint, a descriptor vector, an adjacency matrix. That approximation carried us a long way, and for most
property-prediction tasks it still wins. But a ligand does not bind as a graph; it binds as a particular conformation of
a flexible object, in water, to a protein that is itself moving. Molecular modeling is where that physics comes back.
"""
