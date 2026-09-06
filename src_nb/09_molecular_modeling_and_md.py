# %% [markdown]
"""
# 09 · Molecular modeling basics: conformers, force fields and molecular dynamics

**Chemoinformatics practicals — Session 9 of 9**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

> ⚡ **Tip.** This notebook runs on CPU, but is ~20× faster with a GPU: in Colab choose
> *Runtime → Change runtime type → T4 GPU* before you start.

**Learning goals.** After this session you will be able to
- generate and compare 3D **conformers** of a small molecule and explain why one SMILES corresponds to many geometries;
- describe the terms of a molecular-mechanics **force field** and reproduce a torsion energy profile;
- explain the ingredients of a **molecular dynamics** (MD) simulation: integrator, time step, thermostat, periodic box, solvent;
- run a short MD simulation of a solvated peptide with **OpenMM** and analyse it with **MDAnalysis** (RMSD, Ramachandran plot);
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "openmm", "MDAnalysis", "py3Dmol"], check=False)

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
from openmm import unit
import MDAnalysis as mda

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

# %%
smiles = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"     # gefitinib, an EGFR kinase inhibitor
mol = Chem.MolFromSmiles(smiles)
mol

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

# %%
plt.figure(figsize=(5, 3))
plt.hist(energies, bins=20)
plt.xlabel("MMFF94 energy relative to the minimum (kcal/mol)"); plt.ylabel("conformers")
plt.show()

# %%
# How different are the conformers? Heavy-atom RMSD after optimal superposition
heavy = Chem.RemoveHs(molH)
n = heavy.GetNumConformers()
rms = np.zeros((n, n))
for i in range(n):
    for j in range(i + 1, n):
        rms[i, j] = rms[j, i] = rdMolAlign.GetBestRMS(heavy, heavy, prbId=i, refId=j)
print(f"mean pairwise RMSD: {rms[np.triu_indices(n, 1)].mean():.2f} Å")

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
## 3. Molecular dynamics in a nutshell

MD integrates **Newton's equations** for every atom, with forces $\mathbf F_i = -\nabla_i E(\mathbf r)$ from the force field:

1. compute forces from current positions;
2. update velocities and positions over a small time step $\Delta t$ (the *velocity-Verlet* or *leap-frog* integrator);
3. repeat, saving snapshots ("frames") every few hundred steps → a **trajectory**.

Key practical ingredients:

| ingredient | why | typical choice |
|---|---|---|
| **time step** | must resolve the fastest motion (X–H stretch ~10 fs) | 2 fs with H-bond constraints |
| **thermostat** | keeps the average temperature at $T$ (canonical ensemble) | Langevin dynamics, friction 1 ps⁻¹ |
| **barostat** | keeps pressure at 1 bar (NPT) | Monte Carlo barostat |
| **periodic boundary conditions** | fake an infinite system with a small box | cubic/rhombic dodecahedron box |
| **long-range electrostatics** | Coulomb decays slowly | Particle Mesh Ewald (PME), cutoff 1 nm |
| **explicit solvent** | water matters! | TIP3P water model + ions |

A 2 fs step means 500,000 steps per nanosecond; a modern GPU does ~1 µs/day for a small protein. We will simulate
**alanine dipeptide** (the "hydrogen atom of protein folding": one φ/ψ pair) in a box of ~750 water molecules.
"""

# %%
pdb_path = fetch("md/alanine_dipeptide_solvated.pdb")
pdb = app.PDBFile(pdb_path)
print("atoms:", pdb.topology.getNumAtoms(), "| residues:", [r.name for r in pdb.topology.residues()][:3],
      "| waters:", sum(1 for r in pdb.topology.residues() if r.name == "HOH"))
print("box vectors (nm):", [round(v.x, 2) for v in pdb.topology.getPeriodicBoxVectors()])

# %%
view = py3Dmol.view(width=500, height=350)
view.addModel(open(pdb_path).read(), "pdb")
view.setStyle({"resn": "HOH"}, {"line": {"opacity": 0.3}})
view.setStyle({"not": {"resn": "HOH"}}, {"stick": {}})
view.zoomTo({"not": {"resn": "HOH"}}); view.show()

# %% [markdown]
"""
### Building the simulation

The OpenMM workflow is always the same four objects: **ForceField → System → Integrator → Simulation**.
"""

# %%
forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")     # protein FF + water model

system = forcefield.createSystem(
    pdb.topology,
    nonbondedMethod=app.PME,                 # long-range electrostatics
    nonbondedCutoff=1.0 * unit.nanometer,
    constraints=app.HBonds,                  # rigid X–H bonds allow a 2 fs step
)
system.addForce(mm.MonteCarloBarostat(1.0 * unit.bar, 300 * unit.kelvin, 25))   # NPT ensemble

integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtoseconds)
integrator.setRandomNumberSeed(1)

simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)
print("running on platform:", simulation.context.getPlatform().getName())
print("energy terms in the System:", [f.__class__.__name__ for f in system.getForces()])

# %%
# Energy minimisation removes bad contacts from the starting structure
state0 = simulation.context.getState(getEnergy=True)
simulation.minimizeEnergy(maxIterations=500)
state1 = simulation.context.getState(getEnergy=True)
print(f"potential energy before: {state0.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.0f} kJ/mol")
print(f"potential energy after : {state1.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.0f} kJ/mol")

# %%
# Production run. Reporters write the trajectory (DCD) and a log of thermodynamic quantities.
n_steps = 100_000 if GPU else 10_000          # 200 ps on GPU, 20 ps on CPU (≈1–2 min)
report_every = 200                            # save a frame every 0.4 ps

simulation.reporters.clear()
simulation.reporters.append(app.DCDReporter("ala2_traj.dcd", report_every))
simulation.reporters.append(app.StateDataReporter("ala2_log.csv", report_every, step=True, time=True,
                                                  potentialEnergy=True, temperature=True, volume=True, speed=True))
simulation.reporters.append(app.StateDataReporter(sys.stdout, n_steps // 5, step=True, time=True,
                                                  temperature=True, speed=True, remainingTime=True, totalSteps=n_steps))

simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
t0 = time.time()
simulation.step(n_steps)
print(f"\nsimulated {n_steps * 2 / 1000:.0f} ps in {time.time() - t0:.0f} s")

# Save the final structure (topology for the analysis tools)
with open("ala2_final.pdb", "w") as f:
    app.PDBFile.writeFile(simulation.topology, simulation.context.getState(getPositions=True).getPositions(), f)

# %% [markdown]
"""
## 4. Analysing the trajectory

**MDAnalysis** loads a topology (PDB) plus a trajectory (DCD) into a `Universe`. Atoms are selected with a
VMD-like selection language; analysis classes iterate over frames.
"""

# %%
log = pd.read_csv("ala2_log.csv")
log.columns = [c.split(" (")[0].strip('#"') for c in log.columns]
fig, axes = plt.subplots(1, 3, figsize=(13, 3))
axes[0].plot(log["Time"], log["Temperature"]); axes[0].axhline(300, c="r", ls="--"); axes[0].set_ylabel("T (K)")
axes[1].plot(log["Time"], log["Potential Energy"]); axes[1].set_ylabel("E_pot (kJ/mol)")
axes[2].plot(log["Time"], log["Box Volume"]); axes[2].set_ylabel("volume (nm³)")
for ax in axes: ax.set_xlabel("time (ps)")
plt.tight_layout(); plt.show()

# %%
u = mda.Universe(pdb_path, "ala2_traj.dcd")
print(u, "|", len(u.trajectory), "frames")
peptide = u.select_atoms("not resname HOH")
print(peptide.n_atoms, "peptide atoms;", u.select_atoms("resname HOH").n_residues, "waters")

# %%
# RMSD of the peptide heavy atoms relative to the first frame
from MDAnalysis.analysis import rms, align
heavy_sel = "not resname HOH and not name H*"
R = rms.RMSD(u, u, select=heavy_sel, ref_frame=0).run()
rmsd = R.results.rmsd            # columns: frame, time, RMSD
plt.figure(figsize=(6, 3))
plt.plot(np.arange(len(rmsd)) * report_every * 0.002, rmsd[:, 2])
plt.xlabel("time (ps)"); plt.ylabel("heavy-atom RMSD to frame 0 (Å)")
plt.show()

# %% [markdown]
"""
### The Ramachandran plot

For a peptide, the two backbone dihedrals **φ** (C–N–Cα–C) and **ψ** (N–Cα–C–N) summarise the conformation.
Alanine dipeptide visits a few basins: the extended β/PPII region (φ ≈ −70…−150°, ψ ≈ +120…180°), the right-handed
α-helix region (φ ≈ −60°, ψ ≈ −45°), and — rarely — the left-handed αL region (φ > 0). How many basins does your short
simulation explore?
"""

# %%
from MDAnalysis.analysis.dihedrals import Ramachandran
rama = Ramachandran(u.select_atoms("resname ALA")).run()
phi_psi = rama.results.angles[:, 0, :]          # (frames, 2)

fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(phi_psi[:, 0], phi_psi[:, 1], c=np.arange(len(phi_psi)), cmap="viridis", s=12)
ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.axhline(0, c="gray", lw=0.5); ax.axvline(0, c="gray", lw=0.5)
ax.set_xlabel("φ (°)"); ax.set_ylabel("ψ (°)"); ax.set_title("Ramachandran plot (colour = time)")
plt.colorbar(sc, label="frame"); plt.show()

# %%
plt.figure(figsize=(7, 3))
plt.plot(phi_psi[:, 0], label="φ"); plt.plot(phi_psi[:, 1], label="ψ")
plt.xlabel("frame"); plt.ylabel("angle (°)"); plt.legend(); plt.show()

# %%
# Look at a few frames of the peptide (water hidden)
view = py3Dmol.view(width=500, height=350)
for k, frame in enumerate(np.linspace(0, len(u.trajectory) - 1, 5).astype(int)):
    u.trajectory[frame]
    peptide.write(f"frame_{k}.pdb")
    view.addModel(open(f"frame_{k}.pdb").read(), "pdb")
    view.setStyle({"model": k}, {"stick": {"colorscheme": ["redCarbon", "orangeCarbon", "yellowCarbon", "greenCarbon", "blueCarbon"][k]}})
view.zoomTo(); view.show()

# %% [markdown]
"""
### Exercise 4.1
1. Compute the fraction of frames in the α-helical region (−100° < φ < −30° and −80° < ψ < 0°) and in the β/PPII region (φ < −30° and ψ > 90°).
2. What is the average box volume, and hence the water density in g/cm³ (count the waters with `u.select_atoms("resname HOH").n_residues`; M = 18.015 g/mol)? Is it close to 1.0?
3. (If you have a GPU) rerun the simulation at 400 K. How does the Ramachandran plot change?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution (1–2)</b></summary>

```python
phi, psi = phi_psi[:, 0], phi_psi[:, 1]
alpha = ((phi > -100) & (phi < -30) & (psi > -80) & (psi < 0)).mean()
beta = ((phi < -30) & (psi > 90)).mean()
print(f"alpha: {alpha:.0%}   beta/PPII: {beta:.0%}")

n_wat = u.select_atoms("resname HOH").n_residues
V_cm3 = log["Box Volume"].mean() * 1e-21           # nm^3 -> cm^3
mass_g = n_wat * 18.015 / 6.022e23
print(f"density ≈ {mass_g / V_cm3:.3f} g/cm3")
```
</details>
"""

# %% [markdown]
"""
## 5. Optional (GPU): a real protein — the villin headpiece

The 35-residue villin headpiece (HP35) is a classic fast-folding mini-protein. The file below (from the OpenMM tutorials)
is already solvated. On a T4 GPU, 100 ps take about a minute; on CPU it is ~10 minutes, so the cell only runs when a
GPU is present (change `RUN_PROTEIN` to force it).
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
    print("Skipped (no GPU detected). Set RUN_PROTEIN = True to run on CPU (~10 min).")

# %% [markdown]
"""
## 6. Where MD sits in computer-aided drug design

| task | method | what you learn |
|---|---|---|
| Where does a ligand bind, and how? | **docking** (AutoDock Vina, Gnina; TeachOpenCADD T015) | pose + crude score |
| Is the pose stable? Which interactions persist? | **protein–ligand MD** (T019/T020; interaction fingerprints with ProLIF) | dynamics of the complex |
| How strongly does it bind? | **free-energy methods** (FEP, TI, MM/GBSA) | ΔG of binding, ± 1 kcal/mol |
| Which pockets exist? Are they druggable? | MD + pocket detection (T014) | cryptic pockets, flexibility |
| Features for machine learning | conformers, 3D pharmacophores, MD-derived descriptors, 3D/equivariant GNNs (T036) | representation beyond 2D |

**Protein–ligand MD in practice** needs three extra steps we skipped today: preparing the protein (missing atoms,
protonation — `pdbfixer`, PROPKA), parametrising the ligand (GAFF or OpenFF via `openmmforcefields`), and much longer
sampling (≥ 100 ns, several replicas). TeachOpenCADD **T019** walks through exactly that for EGFR–gefitinib analogues.

## Exercises to finish
1. Re-run the alanine dipeptide simulation with a **4 fs** time step (keep `constraints=app.HBonds`). What happens to the temperature and energy, and why? (Hint: hydrogen mass repartitioning is what people do to get away with 4 fs.)
2. Replace the Langevin integrator by `mm.VerletIntegrator(2*unit.femtoseconds)` (no thermostat) and plot the temperature. Which ensemble is this?
3. Compute the **radius of gyration** of the peptide along the trajectory (`peptide.radius_of_gyration()` for each frame).
4. Generate 20 conformers of gefitinib and compute their NPR shape descriptors (session 02). How much does the *shape* change between conformers?

## Further reading
- TeachOpenCADD T019/T020 (protein–ligand MD with OpenMM, analysis with MDAnalysis).
- OpenMM user guide: <http://docs.openmm.org/latest/userguide/> and the OpenMM cookbook.
- MDAnalysis user guide: <https://userguide.mdanalysis.org/>.
- Braun *et al.*, *Best practices for foundations in molecular simulations*, Living J. Comp. Mol. Sci. **2019**, 1, 5957.
- Höltje, Sippl, Rognan, Folkers, *Molecular Modeling: Basic Principles and Applications* (course bibliography).

---

**This was the last session. Congratulations!** Over nine notebooks you went from `print("Hello")` to curating real
bioactivity data, building and validating QSAR models, training graph networks, generating molecules and putting an
LLM to work with chemistry tools — and finally to giving molecules coordinates and letting them move.

Notice what this last session changes about the previous eight. Everything from session 02 onwards described molecules
as **graphs**: a fingerprint, a descriptor vector, an adjacency matrix. That approximation carried us a long way, and
for most property-prediction tasks it still wins. But a ligand does not bind as a graph; it binds as a particular
conformation of a flexible object, in water, to a protein that is itself moving. Molecular modeling is where that
physics comes back, and it is the bridge to what the field is building next: 3D and equivariant neural networks,
structure-based generative models, co-folding, and free-energy methods.

Two habits are worth keeping from all nine sessions: *look at your data before you model it*, and *ask what your
representation is throwing away*.
"""
