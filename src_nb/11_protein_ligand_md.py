# %% [markdown]
"""
# 11 · Molecular dynamics of a protein–ligand complex: does the pose survive?

**Chemoinformatics practicals — Session 11 of 11**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

> ⚡ **Tip.** This notebook runs on CPU, but the simulation of a ~45 000-atom system is ~50× faster on a GPU: in Colab
> choose *Runtime → Change runtime type → T4 GPU* before you start. On CPU the notebook shrinks its own run to a few
> picoseconds and analyses a **100 ps trajectory of the same system computed in advance** — you lose nothing of the
> analysis, and you still get the movies.

**What we are going to do, in one paragraph.** Session 09 left gefitinib sitting in the pocket of EGFR as a single,
frozen picture. Today we let that picture move. That needs five preparations — close the gaps in the protein chain,
extract the ligand, teach the force field what gefitinib is, put everything in water, warm it up gently — and then a
simulation of 100 ps, which we **watch as a movie** and then **measure**: did the ligand stay? which contacts held?
does the protein move where the crystal says it should? Every step has a picture, and the code is the same OpenMM and
MDAnalysis you used on the dipeptide in session 10 — just with a bigger box.

**Learning goals.** After this session you will be able to
- explain why a docked pose is a hypothesis that MD can test, and what MD adds (flexibility, water, time);
- **prepare a protein for MD** (why chain breaks must be closed, protonation) and a **ligand** (bond orders, hydrogens, starting pose);
- explain what **parametrising a ligand** means — atom types, partial charges, a residue template — and do it with GAFF;
- build a solvated protein–ligand system in OpenMM, run **minimisation → restrained equilibration → production**, and say what each stage is for;
- analyse a protein–ligand trajectory with MDAnalysis and ProLIF: RMSD of protein and ligand, RMSF vs crystallographic B-factors, the **hinge hydrogen bond** over time, interaction persistence;
- name what a 100 ps simulation can and cannot tell you about binding.

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - **TeachOpenCADD** talktorials **T019 · Molecular dynamics simulation** and **T020 · Analyzing molecular dynamics simulations** (Volkamer lab; [GitHub](https://github.com/volkamerlab/teachopencadd), CC BY 4.0) — the protein/ligand preparation functions, the OpenMM workflow for a protein–ligand complex, and the RMSD / hydrogen-bond analysis of the EGFR–inhibitor trajectory;
> - the **OpenMM** user guide and cookbook ([GitHub](https://github.com/openmm/openmm-cookbook), MIT) — restraints, reporters, hydrogen mass repartitioning;
> - the **GAFF** force field (Wang *et al.* 2004; parameters from AmberTools, redistributed as `gaff-2.11.xml` by **openmmforcefields**, MIT) and **Open Babel**'s GAFF atom typer ([GitHub](https://github.com/openbabel/openbabel), GPL-2.0);
> - **MDAnalysis** ([website](https://www.mdanalysis.org), GPL-2.0+) and **ProLIF** ([GitHub](https://github.com/chemosim-lab/ProLIF), Apache-2.0) documentation;
> - the **CCPBioSim** *biosim-analysis-workshop* ([GitHub](https://github.com/CCPBioSim/biosim-analysis-workshop), CC BY-SA 4.0) for the analysis logic.
"""

# %%
# @title ⚙️ Setup — run this cell first (≈2 min on Colab)
import sys, os, io, time, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "openmm", "pdbfixer", "MDAnalysis",
                    "py3Dmol", "prolif", "openbabel-wheel", "requests", "ipywidgets"], check=False)

from openbabel import openbabel as ob, pybel        # Open Babel must be imported before other SWIG libraries (e.g. vina)
ob.obErrorLog.SetOutputLevel(0)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning); warnings.filterwarnings("ignore", category=UserWarning)
import py3Dmol
import pdbfixer
import openmm as mm
import openmm.app as app
from openmm import unit
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
import prolif as plf
from ipywidgets import interact

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
    except Exception:
        text, source = open(fetch(f"pdb/{pdb_id}.pdb")).read(), "course cache (RCSB unreachable)"
    path = f"{pdb_id}.pdb"
    with open(path, "w") as f:
        f.write(text)
    print(f"{pdb_id}: {len(text.splitlines())} lines from {source}")
    return path

platforms = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
GPU = any(p in platforms for p in ("CUDA", "OpenCL"))
print("OpenMM", mm.__version__, "| platforms:", platforms, "| GPU:", GPU, "| Open Babel", ob.OBReleaseVersion())

# %% [markdown]
"""
## 1. From a snapshot to a movie

Session 09 ended with a **pose**: gefitinib in the ATP pocket of EGFR, hinge hydrogen bond to Met793, scored by a
function that treats the protein as a rigid sculpture and the water as absent. Both simplifications are wrong, and a
medicinal chemist wants to know whether the pose *holds* when they are removed. That is the question MD answers:

- **Is the pose stable?** If the ligand drifts away from its starting position in a few hundred picoseconds, the pose
  was probably wrong (the reverse — staying put — is necessary, not sufficient).
- **Which interactions persist?** A hydrogen bond present in 95 % of frames means something; one present in 20 % is
  decoration. Docking cannot make that distinction.
- **What does water do?** Explicit water competes for hydrogen bonds and fills the space the ligand leaves free.
- **How does the protein respond?** Side chains relax around the ligand; loops that were disordered in the crystal move.

The workflow is the seven-step one of Lecture 4. Session 10 did it for a peptide in water; the difference today is the
**ligand**: a molecule the protein force field has never heard of, which we must *parametrise* ourselves.

| step | session 10 (alanine dipeptide) | today (EGFR–gefitinib) |
|---|---|---|
| structure | shipped, ready | PDB entry 4WKQ: gaps to close, modified residue, ligand to extract |
| force field | AMBER ff14SB | ff14SB for the protein **+ GAFF 2.11 for gefitinib** |
| system | 866 waters, 2 620 atoms | ~13 200 waters + 80 ions, ~45 000 atoms |
| equilibration | minimisation only | minimisation → restrained NVT → NPT |
| production | 20–200 ps | 2 ps (CPU) / 100 ps (GPU), plus the same 100 ps run computed in advance |
| analysis | RMSD, Ramachandran | RMSD protein & ligand, RMSF vs B-factors, hinge H-bond, interaction persistence |
"""

# %% [markdown]
"""
## 2. Prepare the protein — for MD this time

### Step 1 · Get the crystal structure, and look at what is in it

PDB entry **4WKQ**: the EGFR kinase domain with gefitinib bound, at 1.85 Å resolution — the same file as session 09.
The picture shows everything the file contains: the protein as a grey ribbon, **gefitinib in green** in the cleft
between the two lobes, the crystallisation waters as small red dots, a buffer molecule (MES, orange) stuck to the
surface, and one sodium ion (purple). The three residues drawn as thin white sticks are the ones that will matter today:
**Met793** (the *hinge*, whose backbone N–H bonds to gefitinib), **Thr790** (the *gatekeeper*) and **Lys745** (the
catalytic lysine). Everything that is not protein or gefitinib has to go before we simulate.
"""

# %%
pdb_path = get_pdb("4WKQ")
crystal_text = open(pdb_path).read()

view = py3Dmol.view(width=650, height=430)
view.addModel(crystal_text, "pdb")
view.setStyle({"cartoon": {"color": "lightgray"}})
view.setStyle({"resn": "IRE"}, {"stick": {"colorscheme": "greenCarbon", "radius": 0.25}})
view.setStyle({"resn": "MES"}, {"stick": {"colorscheme": "orangeCarbon"}})
view.setStyle({"resn": "HOH"}, {"sphere": {"radius": 0.3, "color": "red"}})
view.setStyle({"resn": "NA"}, {"sphere": {"radius": 0.8, "color": "purple"}})
view.addStyle({"resi": [745, 790, 793], "not": {"resn": "HOH"}}, {"stick": {"colorscheme": "whiteCarbon", "radius": 0.12}})
view.addResLabels({"resi": [745, 790, 793]}, {"fontSize": 11, "backgroundOpacity": 0.6})
view.zoomTo({"resn": "IRE"}); view.show()

# %% [markdown]
"""
### Step 2 · Clean it — twice, to see why the gaps matter

We reuse the PDBFixer recipe of session 09 with one difference: `build_loops=True`. In docking the receptor was a rigid
set of atoms and gaps in the chain did not matter. In MD the chain is a **bonded** object: OpenMM connects consecutive
residues of a chain with a peptide bond, so a gap of 22 missing residues becomes a single "bond" 25 Å long with a
spring constant of a covalent bond — an energy of millions of kJ/mol that would tear the protein apart on the first
step. Either the gaps are closed, or the fragments are treated as separate chains. We close them: PDBFixer builds the
missing residues in a plausible but arbitrary conformation (they will relax during equilibration). We never *extend*
the termini — those residues were simply not in the construct.
"""

# %%
def prepare_receptor(pdb_path, out_path, ph=7.4, build_loops=False, verbose=True):
    """Clean a PDB file for docking/MD: remove heterogens, fix residues and atoms, add hydrogens at the given pH."""
    fixer = pdbfixer.PDBFixer(filename=pdb_path)
    fixer.findMissingResidues()                       # gaps in the chain, from SEQRES
    built = []
    if not build_loops:
        fixer.missingResidues = {}
    else:                                             # model internal gaps only, never extend the termini
        chains = list(fixer.topology.chains())
        for key in list(fixer.missingResidues):
            if key[1] == 0 or key[1] == len(list(chains[key[0]].residues())):
                del fixer.missingResidues[key]
            else:
                built.append((key, fixer.missingResidues[key]))
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()                # CSX -> CYS
    fixer.removeHeterogens(keepWater=False)           # ligand, buffer, ions, waters
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=1)                     # side chains (and loops, if requested; seed -> same loops every run)
    fixer.addMissingHydrogens(ph)
    with open(out_path, "w") as f:
        app.PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
    if verbose:
        print(f"{out_path}: {sum(1 for _ in fixer.topology.residues())} residues, {fixer.topology.getNumAtoms()} atoms;",
              "loops built:", [f"{len(v)} residues after chain position {k[1]}" for k, v in built] or "none")
    return fixer

t0 = time.time()
prepare_receptor(pdb_path, "protein_gaps.pdb", build_loops=False)
fixer = prepare_receptor(pdb_path, "protein.pdb", build_loops=True)
print(f"({time.time() - t0:.0f} s — loop building is an MD refinement, slow on CPU)")

# %% [markdown]
"""
327 residues instead of 297: the three loops (721–723, 747–751, 985–1006) are now present.

### Step 3 · See the difference

Left: the protein as the crystallographer left it — three places where the ribbon simply stops, because those
residues were too disordered to be seen in the electron density. Right: the same protein after PDBFixer, with the
**rebuilt loops in orange**. The longest one, 22 residues, is a whole segment that had no experimental coordinates at
all; PDBFixer placed it in a plausible but invented conformation. Keep that in mind when the analysis shows those
residues moving a lot — they were never real to begin with.
"""

# %%
LOOPS = {"resi": ["721-723", "747-751", "985-1006"]}
view = py3Dmol.view(width=900, height=400, viewergrid=(1, 2))
view.addModel(open("protein_gaps.pdb").read(), "pdb", viewer=(0, 0))
view.setStyle({"cartoon": {"color": "lightgray"}}, viewer=(0, 0))
view.addLabel("crystal: 297 residues, 3 gaps", {"position": {"x": 0, "y": 0, "z": 0}, "useScreen": True, "screenOffset": {"x": 10, "y": 10}, "fontSize": 13}, viewer=(0, 0))
view.addModel(open("protein.pdb").read(), "pdb", viewer=(0, 1))
view.setStyle({"cartoon": {"color": "lightgray"}}, viewer=(0, 1))
view.addStyle(LOOPS, {"cartoon": {"color": "orange"}}, viewer=(0, 1))
view.addLabel("PDBFixer: 327 residues, loops rebuilt (orange)", {"position": {"x": 0, "y": 0, "z": 0}, "useScreen": True, "screenOffset": {"x": 10, "y": 10}, "fontSize": 13}, viewer=(0, 1))
view.zoomTo(); view.show()

# %% [markdown]
"""
### Step 4 · Put a number on it

Why not simply simulate the gapped protein? Because to OpenMM a chain is a *bonded* object: it connects consecutive
residues with a peptide bond whether or not they are anywhere near each other. So we ask the force field for the
**bond-stretching energy** of both versions, in vacuum, without ligand or water. A bond pulled across a 25 Å gap
with a covalent spring constant stores a truly enormous energy — and would release it on the very first step of
dynamics, tearing the protein apart.
"""

# %%
ff_protein = app.ForceField("amber14-all.xml")
for label in ["protein_gaps.pdb", "protein.pdb"]:
    pdb = app.PDBFile(label)
    system = ff_protein.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)
    for f in system.getForces():
        f.setForceGroup(1 if isinstance(f, mm.HarmonicBondForce) else 0)
    ctx = mm.Context(system, mm.VerletIntegrator(1 * unit.femtosecond))
    ctx.setPositions(pdb.positions)
    e_bond = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print(f"{label:18s} bond-stretching energy = {e_bond:>12,.0f} kJ/mol")
    del ctx

# %% [markdown]
"""
Almost three million kJ/mol in the bond term of the gapped protein — three "bonds" stretched across the gaps — against
seventy thousand for the continuous chain. (The continuous chain's value is itself far above that of a relaxed protein,
because PDBFixer's rebuilt loops and freshly placed hydrogens are not yet minimised; that is the next step.)

### Step 5 · Relax what we invented — before trusting it

PDBFixer builds a missing loop by a quick simulation from a random start, and the result is a *guess*: sometimes a
proline ring comes out strained, sometimes the last built residue lands almost on top of the first crystal residue.
Such a spot carries a gigantic force, and if we simply add water and start the dynamics, that force launches an atom
across the box on the first step and the simulation dies with the message every MD practitioner learns to dread:
**`Particle coordinate is NaN`**. (We know, because it happened to this notebook in about half of its test runs before
this step existed.)

The cure is cheap and general: **minimise the protein alone, in vacuum, with the crystal atoms held in place** by
springs, so that only the invented loops (and the hydrogens) are free to relax. The springs are a `CustomExternalForce`
— a force we write ourselves as a formula — and we will meet the exact same object again in section 6. Watch three
numbers: the largest force on any atom before and after, how far the loops moved, and how little the crystal atoms did.
"""

# %%
BUILT = set(range(721, 724)) | set(range(747, 752)) | set(range(985, 1007))      # the residues PDBFixer invented

def largest_force(topology, positions, system):
    """The atom carrying the largest force, and that force (kJ/mol/nm): a quick check for clashes."""
    ctx = mm.Context(system, mm.VerletIntegrator(1 * unit.femtosecond)); ctx.setPositions(positions)
    F = np.linalg.norm(ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer), axis=1)
    atom = list(topology.atoms())[int(F.argmax())]
    return f"{F.max():,.0f} kJ/mol/nm on {atom.residue.name}{atom.residue.id} {atom.name}"

def relax_built_loops(topology, positions, k=1000.0):
    """Minimise in vacuum with every crystal heavy atom tied to its position by a spring of stiffness k."""
    system = ff_protein.createSystem(topology, nonbondedMethod=app.NoCutoff)
    springs = mm.CustomExternalForce("0.5*k*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
    springs.addGlobalParameter("k", k * unit.kilojoule_per_mole / unit.nanometer**2)
    for name in ("x0", "y0", "z0"):
        springs.addPerParticleParameter(name)
    for atom in topology.atoms():
        if atom.element.symbol != "H" and int(atom.residue.id) not in BUILT:
            springs.addParticle(atom.index, positions[atom.index])
    system.addForce(springs)
    print("before:", largest_force(topology, positions, system))
    ctx = mm.Context(system, mm.VerletIntegrator(1 * unit.femtosecond)); ctx.setPositions(positions)
    mm.LocalEnergyMinimizer.minimize(ctx)                     # to convergence: a few seconds for 5000 atoms
    relaxed = ctx.getState(getPositions=True).getPositions()
    print("after: ", largest_force(topology, relaxed, system))
    return relaxed

t0 = time.time()
built_protein = app.PDBFile("protein.pdb")
relaxed_positions = relax_built_loops(built_protein.topology, built_protein.positions)
with open("protein_relaxed.pdb", "w") as f:
    app.PDBFile.writeFile(built_protein.topology, relaxed_positions, f, keepIds=True)

moved = np.linalg.norm(np.array(relaxed_positions.value_in_unit(unit.angstrom)) - np.array(built_protein.positions.value_in_unit(unit.angstrom)), axis=1)
is_built = np.array([int(a.residue.id) in BUILT for a in built_protein.topology.atoms()])
is_heavy = np.array([a.element.symbol != "H" for a in built_protein.topology.atoms()])
print(f"rebuilt loops moved {moved[is_built & is_heavy].mean():.2f} Å on average (max {moved[is_built & is_heavy].max():.1f}), "
      f"crystal heavy atoms {moved[~is_built & is_heavy].mean():.2f} Å  ({time.time() - t0:.0f} s)")

# %% [markdown]
"""
From an absurd 3 × 10²¹ kJ/mol/nm — two atoms of the built loop practically on top of each other — to under a hundred,
in twenty seconds, while the crystal moved by a quarter of an ångström. Left: the loops as PDBFixer built them; right: after relaxation — the differences are local, a few residues
that were in trouble. This is the structure we take forward. The general lesson outlives this notebook: **whenever a
program invents coordinates for you, relax them and check the forces before you spend GPU hours on them.**
"""

# %%
view = py3Dmol.view(width=900, height=380, viewergrid=(1, 2))
for col, (fname, title) in enumerate([("protein.pdb", "loops as built by PDBFixer"), ("protein_relaxed.pdb", "after vacuum relaxation")]):
    view.addModel(open(fname).read(), "pdb", viewer=(0, col))
    view.setStyle({"cartoon": {"color": "lightgray", "opacity": 0.5}}, viewer=(0, col))
    view.addStyle(LOOPS, {"stick": {"colorscheme": "orangeCarbon", "radius": 0.15}}, viewer=(0, col))
    view.addLabel(title, {"position": {"x": 0, "y": 0, "z": 0}, "useScreen": True, "screenOffset": {"x": 10, "y": 10}, "fontSize": 13}, viewer=(0, col))
    view.zoomTo({"resi": ["985-1006"]}, viewer=(0, col))
view.show()

# %% [markdown]
"""
## 3. Prepare the ligand

Two possible starting poses, both from session 09: the **crystal pose** (extracted from 4WKQ with correct bond orders,
as in session 09) or the **top docked pose** (the file `gefitinib_docked_poses.sdf` you downloaded; a copy from the
course run is shipped for convenience). Set `LIGAND_START` below. We use the crystal pose for the main run and leave
the docked one for Exercise 1 — the comparison is the whole point.

The morpholine nitrogen (pKa ≈ 7.2) is kept neutral, as in docking; with explicit electrostatics this choice now
matters, and a careful study would run both protonation states.
"""

# %%
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
LIGAND_START = "crystal"          # "crystal" or "docked"

def crystal_ligand(pdb_path, resname, smiles):
    """Extract a ligand from a PDB file and give it correct bond orders from its SMILES."""
    block = "\n".join(l.rstrip("\n") for l in open(pdb_path) if l.startswith("HETATM") and l[17:20] == resname) + "\nEND\n"
    raw = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
    mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(smiles), Chem.RemoveHs(raw, sanitize=False))
    Chem.SanitizeMol(mol)
    return Chem.AddHs(mol, addCoords=True)

if LIGAND_START == "crystal":
    ligand = crystal_ligand(pdb_path, "IRE", GEFITINIB)
else:
    sdf = "gefitinib_docked_poses.sdf" if os.path.exists("gefitinib_docked_poses.sdf") else fetch("md/gefitinib_docked_poses.sdf")
    ligand = Chem.SDMolSupplier(sdf, removeHs=False)[0]
    ligand = Chem.AddHs(Chem.RemoveHs(ligand), addCoords=True)      # standard hydrogens (Meeko merges some)
print(f"ligand from the {LIGAND_START} pose: {ligand.GetNumAtoms()} atoms ({ligand.GetNumHeavyAtoms()} heavy),",
      f"formal charge {Chem.GetFormalCharge(ligand)}")
Chem.MolToMolFile(ligand, "ligand_start.sdf")
Chem.MolFromSmiles(GEFITINIB)

# %% [markdown]
"""
The same molecule twice: the flat 2D drawing above, and below the **crystal pose in 3D**, with hydrogens added and
the atoms coloured by element (grey C, blue N, red O, green Cl, light green F). Rotate it. The quinazoline and the
anilino ring are almost coplanar; the propoxy–morpholine tail curls away. This exact geometry is where the simulation
will start.
"""

# %%
view = py3Dmol.view(width=500, height=340)
view.addModel(Chem.MolToMolBlock(ligand), "mol")
view.setStyle({"stick": {"radius": 0.2}, "sphere": {"scale": 0.25}})
view.zoomTo(); view.show()

# %% [markdown]
"""
## 4. Parametrise the ligand

A protein force field is a *dictionary*: for each of the 20 amino acids, a **template** listing the atoms, their **atom
types**, their **partial charges** and their bonds; and for each atom type (or pair, triple, quadruple of types) the
parameters of the energy terms you met in session 10 — bond lengths and stiffnesses, angles, torsions, Lennard-Jones
radii. Gefitinib is not in the dictionary. Parametrising a ligand means writing its entry:

1. **atom types** — assign each atom to one of the force field's general types. The **General AMBER Force Field
   (GAFF)** has about a hundred types (`ca` aromatic carbon, `c3` sp³ carbon, `os` ether oxygen, `nb` aromatic
   nitrogen, `nh` amine nitrogen attached to an aromatic ring, `cl`, `f`, `ha`/`h1`/`hn` hydrogens …), defined by rules
   on the chemical environment. Open Babel implements the GAFF typer; we borrow it.
2. **partial charges** — GAFF expects **AM1-BCC** charges: a semi-empirical quantum calculation (AM1) plus bond-charge
   corrections, computed by `antechamber` (AmberTools). That program is not pip-installable, so we use the **MMFF94**
   charge model from RDKit instead — the same bond-charge-increment idea, fitted to reproduce electrostatic potentials,
   but not what GAFF was validated with. This is the one honest compromise in this notebook; production work uses
   AM1-BCC (or RESP), and the OpenFF toolkit automates the whole step. Exercise 3 measures how much it matters.
3. **the template** — an XML block naming the atoms, types, charges and bonds, which OpenMM matches to the ligand by its
   bond graph; all bonded and Lennard-Jones parameters then come from the GAFF file itself (`gaff-2.11.xml`,
   redistributed by the openmmforcefields project).

This is exactly what TeachOpenCADD T019 does with `openmmforcefields`' `GAFFTemplateGenerator` in one line — we do it in
three short functions so that you see what the line does.
"""

# %%
def gaff_atom_types(mol):
    """GAFF atom types for an RDKit molecule (with hydrogens), from Open Babel's GAFF implementation."""
    obmol = pybel.readstring("mol", Chem.MolToMolBlock(mol)).OBMol
    ff = ob.OBForceField.FindForceField("GAFF")
    assert ff.Setup(obmol), "Open Babel could not type this molecule with GAFF"
    ff.GetAtomTypes(obmol)
    return [ob.toPairData(obmol.GetAtom(i).GetData("FFAtomType")).GetValue() for i in range(1, obmol.NumAtoms() + 1)]

def mmff_charges(mol):
    """MMFF94 partial charges (RDKit), corrected so that they sum exactly to the formal charge."""
    props = AllChem.MMFFGetMoleculeProperties(mol)
    q = np.array([props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())])
    return q - (q.sum() - Chem.GetFormalCharge(mol)) / len(q)

types = gaff_atom_types(ligand)
charges = mmff_charges(ligand)
summary = pd.DataFrame({"element": [a.GetSymbol() for a in ligand.GetAtoms()], "GAFF type": types, "charge": charges.round(3)})
print("atom types used:", dict(summary["GAFF type"].value_counts()))
print("sum of charges:", charges.sum().round(6), "| most negative:", summary.loc[summary.charge.idxmin()].to_dict(),
      "| most positive:", summary.loc[summary.charge.idxmax()].to_dict())
summary.head(12)

# %% [markdown]
"""
Look at the types: the quinazoline carbons and nitrogens are `ca`/`nb`, the anilino N–H is `nh`, the ether oxygens
`os`, the morpholine nitrogen `n3`, the hydrogens are split by environment (`ha` on aromatic carbon, `h1` on carbons
next to N/O, `hc` on plain aliphatic carbon, `hn` on nitrogen). The charges make chemical sense: the nitrogens and
oxygens are negative, the carbons bonded to them positive.

### The two ingredients, on the molecule

Left: every heavy atom labelled with its **GAFF type** — this is what the force field "sees" instead of elements.
Right: the molecule coloured by **partial charge**, red for negative, blue for positive, white for neutral. Charges are
the only parameters that are specific to *this* molecule; all the rest (bond lengths, angles, torsions, radii) comes
from the type. The three most negative atoms should be the morpholine nitrogen and the ether oxygens; the aromatic
carbons carrying substituents should be blue.
"""

# %%
def mol_with_bfactor(mol, values):
    """A PDB block of the molecule with `values` written in the B-factor column (a trick to colour atoms by any number)."""
    lines = []
    for line in Chem.MolToPDBBlock(mol).splitlines():
        if line.startswith(("ATOM", "HETATM")):
            idx = int(line[6:11]) - 1
            line = line[:60] + f"{values[idx]:6.2f}" + line[66:]
        lines.append(line)
    return "\n".join(lines)

view = py3Dmol.view(width=900, height=380, viewergrid=(1, 2))
view.addModel(Chem.MolToMolBlock(ligand), "mol", viewer=(0, 0))
view.setStyle({"stick": {"radius": 0.15, "color": "lightgray"}}, viewer=(0, 0))
for atom, gaff_type in zip(ligand.GetAtoms(), types):
    if atom.GetAtomicNum() > 1:
        pos = ligand.GetConformer().GetAtomPosition(atom.GetIdx())
        view.addLabel(gaff_type, {"position": {"x": pos.x, "y": pos.y, "z": pos.z}, "fontSize": 11, "backgroundColor": "white",
                                  "backgroundOpacity": 0.7, "fontColor": "black", "borderThickness": 0}, viewer=(0, 0))
view.zoomTo(viewer=(0, 0))
view.addModel(mol_with_bfactor(ligand, charges), "pdb", viewer=(0, 1))
view.setStyle({"stick": {"colorscheme": {"prop": "b", "gradient": "rwb", "min": 0.5, "max": -0.5}, "radius": 0.2},
               "sphere": {"colorscheme": {"prop": "b", "gradient": "rwb", "min": 0.5, "max": -0.5}, "scale": 0.3}}, viewer=(0, 1))
view.zoomTo(viewer=(0, 1))
view.show()
print("left: GAFF atom types | right: partial charge, red = negative (-0.5), blue = positive (+0.5)")

# %% [markdown]
"""
Now the template. Atom names must be unique within the residue (we number by element), and every bond is listed.
"""

# %%
def ligand_template_xml(mol, types, charges, resname="LIG"):
    """OpenMM residue template for a small molecule: atoms (name, GAFF type, charge) and bonds."""
    counts, names = {}, []
    for a in mol.GetAtoms():
        el = a.GetSymbol(); counts[el] = counts.get(el, 0) + 1; names.append(f"{el}{counts[el]}")
    lines = ["<ForceField>", "  <Residues>", f'    <Residue name="{resname}">']
    lines += [f'      <Atom name="{n}" type="{t}" charge="{q:.5f}"/>' for n, t, q in zip(names, types, charges)]
    lines += [f'      <Bond atomName1="{names[b.GetBeginAtomIdx()]}" atomName2="{names[b.GetEndAtomIdx()]}"/>' for b in mol.GetBonds()]
    lines += ["    </Residue>", "  </Residues>", "</ForceField>"]
    return "\n".join(lines), names

template_xml, lig_names = ligand_template_xml(ligand, types, charges)
open("ligand_template.xml", "w").write(template_xml)
print("\n".join(template_xml.splitlines()[:8]), "\n      ...\n", "\n".join(template_xml.splitlines()[-4:]))

# %% [markdown]
"""
Where do the numbers for a `ca`–`ca` bond come from? From the GAFF parameter file. Let us look inside it — this *is* the
force field, in the sense of session 10: one line per parameter.
"""

# %%
gaff_path = fetch("md/gaff-2.11.xml")
gaff = open(gaff_path).read()
import re
for pattern, what in [(r'<Bond class1="ca" class2="ca"[^>]*/>', "aromatic C–C bond: length (nm), k (kJ/mol/nm²)"),
                      (r'<Angle class1="ca" class2="ca" class3="ca"[^>]*/>', "aromatic C–C–C angle: angle (rad), k"),
                      (r'<Proper class1="" class2="ca" class3="ca" class4=""[^>]*/>', "torsion X–ca–ca–X: periodicity, phase, k"),
                      (r'<Proper class1="" class2="c3" class3="c3" class4=""[^>]*/>', "torsion X–c3–c3–X (the butane term of session 10)"),
                      (r'<Atom class="ca"[^>]*/>', "Lennard-Jones of ca: sigma (nm), epsilon (kJ/mol)")]:
    print(f"{what}\n   {re.search(pattern, gaff).group(0)}")
print(f"\nin total: {gaff.count('<Bond ')} bond, {gaff.count('<Angle ')} angle, {gaff.count('<Proper ')} torsion, "
      f"{gaff.count('<Improper ')} improper and {gaff.count('<Atom class')} Lennard-Jones entries")

# %% [markdown]
"""
### Quality control: the ligand alone

Before touching the protein, we check that the parameters describe a sane molecule: build the ligand alone in vacuum,
minimise, and see how far it moves from the crystal geometry. A wrong atom type or a missing torsion shows up here as
a distorted ring or a large shift.
"""

# %%
def ligand_topology(mol, names, resname="LIG"):
    """An OpenMM Topology (one residue) plus positions for an RDKit molecule."""
    top = app.Topology(); chain = top.addChain("L"); res = top.addResidue(resname, chain)
    atoms = [top.addAtom(n, app.Element.getBySymbol(a.GetSymbol()), res) for n, a in zip(names, mol.GetAtoms())]
    for b in mol.GetBonds():
        top.addBond(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()])
    return top, mol.GetConformer().GetPositions() * unit.angstrom

lig_top, lig_pos = ligand_topology(ligand, lig_names)
forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml", gaff_path, "ligand_template.xml")

lig_system = forcefield.createSystem(lig_top, nonbondedMethod=app.NoCutoff)
lig_sim = app.Simulation(lig_top, lig_system, mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond))
lig_sim.context.setPositions(lig_pos)
e0 = lig_sim.context.getState(getEnergy=True).getPotentialEnergy()
lig_sim.minimizeEnergy()
e1 = lig_sim.context.getState(getEnergy=True).getPotentialEnergy()
new_pos = lig_sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.angstrom)
heavy = [a.GetIdx() for a in ligand.GetAtoms() if a.GetAtomicNum() > 1]
shift = np.sqrt(((new_pos[heavy] - ligand.GetConformer().GetPositions()[heavy]) ** 2).sum(1).mean())
print("force terms:", [f.__class__.__name__ for f in lig_system.getForces()])
print(f"ligand in vacuum: {e0.value_in_unit(unit.kilocalorie_per_mole):.1f} -> {e1.value_in_unit(unit.kilocalorie_per_mole):.1f} kcal/mol "
      f"after minimisation; heavy atoms moved {shift:.2f} Å RMSD from the crystal geometry")

# %% [markdown]
"""
Half an ångström: the crystal geometry is close to a minimum of the GAFF energy, as it should be. (Try a wrong type on
purpose in Exercise 4 and watch this number.)

## 5. Build the solvated complex

Protein and ligand are joined into one `Modeller`, then solvated. Two choices differ from session 10:
- **`boxShape="dodecahedron"`**: a rhombic dodecahedron holds the same padding with 71 % of the volume of a cube, so
  ~30 % fewer waters to simulate (Lecture 4);
- **0.15 M NaCl**, roughly physiological, added as Na⁺/Cl⁻ after neutralising the system's net charge.
"""

# %%
t0 = time.time()
protein = app.PDBFile("protein_relaxed.pdb")                    # the loops we relaxed in section 2
modeller = app.Modeller(protein.topology, protein.positions)
modeller.add(lig_top, lig_pos)
n_complex = modeller.topology.getNumAtoms()
modeller.addSolvent(forcefield, padding=1.0 * unit.nanometer, ionicStrength=0.15 * unit.molar, model="tip3p", boxShape="dodecahedron")
counts = pd.Series([r.name for r in modeller.topology.residues()]).value_counts()
print(f"complex: {n_complex} atoms | solvated: {modeller.topology.getNumAtoms()} atoms | "
      f"{counts['HOH']} waters, {counts.get('NA', 0)} Na+, {counts.get('CL', 0)} Cl-  ({time.time() - t0:.0f} s)")
with open("system_start.pdb", "w") as f:
    app.PDBFile.writeFile(modeller.topology, modeller.positions, f)

# %% [markdown]
"""
### What the box looks like

The whole system, about 45 000 atoms: the protein as a grey ribbon, gefitinib in green, the ~13 200 water molecules
as a pale blue haze, and the 80 ions as spheres. The haze is not a cube — a rhombic dodecahedron looks like a box with the
corners cut off — and it is *full*: this is liquid water at the right density, not a mist. Every one of those water
molecules will be moved at every one of the 50 000 steps of the production run; that is where the computing time goes.
"""

# %%
view = py3Dmol.view(width=600, height=420)
view.addModel(open("system_start.pdb").read(), "pdb")
view.setStyle({"resn": "HOH"}, {"line": {"opacity": 0.15, "color": "lightblue"}})
view.setStyle({"resn": ["NA", "CL"]}, {"sphere": {"radius": 0.7}})
view.setStyle({"not": {"resn": ["HOH", "NA", "CL", "LIG"]}}, {"cartoon": {"color": "lightgray"}})
view.setStyle({"resn": "LIG"}, {"stick": {"colorscheme": "greenCarbon"}})
view.zoomTo(); view.show()

# %% [markdown]
"""
## 6. Minimise, equilibrate, produce

The **System** is built as in session 10 (PME, 1 nm cutoff, rigid X–H bonds, 2 fs). Then three stages that session 10
skipped and that any published protocol contains (Braun *et al.* 2019):

1. **minimisation** — remove the clashes of the rebuilt loops, the added hydrogens and the placed waters;
2. **restrained NVT** — heat to 300 K while holding the protein and ligand heavy atoms with harmonic springs, so that
   only the water rearranges (a `CustomExternalForce` with a global strength `k_res` we can switch off);
3. **NPT** — release the restraints, switch on the barostat, let the density settle;
4. **production** — the part we analyse.

On a T4 GPU the whole thing takes ~5 minutes for 100 ps of production; on CPU we run a token 2 ps (so that every cell
executes) and analyse the 100 ps trajectory computed in advance with this very code.

> **About the time step.** Lecture 4 mentioned **hydrogen mass repartitioning** (`hydrogenMass=1.5*amu` in
> `createSystem`, then a 4 fs step): mass is moved from heavy atoms onto their hydrogens, the fastest vibrations slow
> down, and the step can be doubled. We tried it for the precomputed trajectory: the run was fine for 100 ps and then
> died with *"Particle coordinate is NaN"*. That is what an unstable integration looks like — no warning, then an
> explosion — and why 2 fs with constrained X–H bonds remains the safe default. The trick works, but it wants a
> well-equilibrated system and, ideally, the heavier 3 amu repartitioning of the original method (Hopkins *et al.*
> 2015). Exercise 6 lets you try.
"""

# %%
system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * unit.nanometer,
                                 constraints=app.HBonds)

# harmonic positional restraints on solute heavy atoms (not on the residues PDBFixer modelled: BUILT, from section 2)
restraint = mm.CustomExternalForce("0.5*k_res*periodicdistance(x, y, z, x0, y0, z0)^2")
restraint.addGlobalParameter("k_res", 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
for p in ("x0", "y0", "z0"):
    restraint.addPerParticleParameter(p)
n_restrained = 0
for atom in modeller.topology.atoms():
    if atom.residue.name in ("HOH", "NA", "CL") or atom.element.symbol == "H":
        continue
    if atom.residue.name != "LIG" and int(atom.residue.id) in BUILT:
        continue
    restraint.addParticle(atom.index, modeller.positions[atom.index]); n_restrained += 1
system.addForce(restraint)
barostat = mm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin, 25)
system.addForce(barostat)

integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds)
integrator.setRandomNumberSeed(1)
simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
print(f"{n_restrained} restrained heavy atoms | forces: {[f.__class__.__name__ for f in system.getForces()]}")
print("platform:", simulation.context.getPlatform().getName())

# %% [markdown]
"""
### What the springs hold

The restraints in a picture: every atom drawn as a **grey sphere** is tied to its starting position by a spring during
the first equilibration stage — the whole protein and the ligand. The **orange** loops are not: they were invented by
PDBFixer, and holding them in place would only preserve a guess. Water and ions (not drawn) are free from the start.
The idea is that the *solvent* should adapt to the solute, not the other way round: water molecules placed by a
program have to find their hydrogen-bond partners, and while they do so we do not want the protein to be pushed
around by a still-unphysical environment.
"""

# %%
solute_heavy = {"not": {"or": [{"resn": ["HOH", "NA", "CL"]}, {"elem": "H"}]}}        # protein + ligand, heavy atoms only
view = py3Dmol.view(width=600, height=420)
view.addModel(open("system_start.pdb").read(), "pdb")
view.setStyle({}, {})                                                                    # hide everything first
view.setStyle(solute_heavy, {"sphere": {"radius": 0.6, "color": "gray", "opacity": 0.55}})
view.setStyle({"resi": ["721-723", "747-751", "985-1006"], "not": {"elem": "H"}}, {"sphere": {"radius": 0.6, "color": "orange", "opacity": 0.9}})
view.setStyle({"resn": "LIG", "not": {"elem": "H"}}, {"sphere": {"radius": 0.7, "color": "green"}})
view.zoomTo({"not": {"resn": ["HOH", "NA", "CL"]}}); view.show()
print(f"grey: {n_restrained} restrained heavy atoms (protein + ligand, ligand in green) | orange: the rebuilt loops, left free")

# %% [markdown]
"""
### Stage 1 · Minimise

The same idea as in session 10, on a system seventeen times larger: slide every atom downhill until the worst
contacts are gone — now mostly between the water molecules a program dropped in and the protein surface. Because the
protein was already relaxed in section 2, the starting energy is *negative* from the outset; minimisation still lowers
it by another ~300 000 kJ/mol as the water settles. (Without the section-2 step this number would start in the
millions or billions — and the run would often not survive the next cell.)

The cell also writes `complex_minimised.pdb`: the protein and ligand **without the water**, at the minimised
positions. That file is the topology for the analysis — the trajectory we save will contain only those atoms, because
17 700 water molecules at every frame is a lot of disk for something we will not look at.
"""

# %%
N_MIN, N_EQ, N_PROD = (500, 2500, 50_000) if GPU else (100, 100, 1000)     # GPU: 5 ps + 5 ps + 100 ps ; CPU: token run
print(f"this run: {N_MIN} minimisation steps, 2 × {N_EQ * 0.002:.1f} ps equilibration, {N_PROD * 0.002:.0f} ps production")

t0 = time.time()
e0 = simulation.context.getState(getEnergy=True).getPotentialEnergy()
simulation.minimizeEnergy(maxIterations=N_MIN)
e1 = simulation.context.getState(getEnergy=True).getPotentialEnergy()
print(f"minimised: {e0.value_in_unit(unit.kilojoule_per_mole):,.0f} -> {e1.value_in_unit(unit.kilojoule_per_mole):,.0f} kJ/mol ({time.time() - t0:.0f} s)")

# protein + ligand only: the topology we will analyse, and the atom subset written to the trajectory
solute_idx = [a.index for a in modeller.topology.atoms() if a.residue.name not in ("HOH", "NA", "CL")]
solute = app.Modeller(modeller.topology, simulation.context.getState(getPositions=True).getPositions())
solute.delete([r for r in solute.topology.residues() if r.name in ("HOH", "NA", "CL")])
with open("complex_minimised.pdb", "w") as f:
    app.PDBFile.writeFile(solute.topology, solute.positions, f, keepIds=True)      # keep the crystal residue numbers

# %% [markdown]
"""
### Stages 2–4 · Warm up with the springs on, release, produce

Three runs of the *same* simulation object, with two switches flipped between them:

- **Restrained NVT.** The barostat is switched off (`setFrequency(0)`: never attempt a volume move) and the springs are on
  (`k_res = 1000` kJ/mol/nm²). Velocities are drawn for 300 K and the water is allowed to settle around a protein that
  cannot move. `reinitialize(preserveState=True)` is the incantation needed after changing a force's setting — OpenMM
  has to rebuild its internal machinery, and we ask it to keep the current positions and velocities while it does.
- **NPT.** Springs off (`k_res = 0`), barostat on (`setFrequency(25)`: try a volume move every 25 steps). The box finds
  its density with a free protein.
- **Production.** A `DCDReporter` is added — only now, so that the equilibration frames never enter the analysis —
  saving every 1000 steps (2 ps). Two details: `atomSubset=solute_idx` writes protein and ligand only, and
  `enforcePeriodicBox=False` stops OpenMM from wrapping molecules back into the box, which would split the protein
  across the periodic boundary in the picture.

The printed speed, in ns/day, is the number to compare between CPU and GPU: one to two orders of magnitude.
"""

# %%
simulation.reporters.append(app.StateDataReporter("md_log.csv", 50, step=True, time=True, potentialEnergy=True,
                                                  temperature=True, volume=True, density=True, speed=True))
# 1. NVT, restrained: barostat off, springs on
barostat.setFrequency(0); simulation.context.reinitialize(preserveState=True)
simulation.context.setParameter("k_res", 1000.0)
simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 1)
t0 = time.time(); simulation.step(N_EQ); print(f"restrained NVT: {N_EQ * 0.002:.1f} ps in {time.time() - t0:.0f} s")
# 2. NPT, released: barostat on, springs off
barostat.setFrequency(25); simulation.context.reinitialize(preserveState=True)
simulation.context.setParameter("k_res", 0.0)
t0 = time.time(); simulation.step(N_EQ); print(f"NPT: {N_EQ * 0.002:.1f} ps in {time.time() - t0:.0f} s")
# 3. production, saving protein + ligand every 2 ps (1000 steps)
simulation.reporters.append(app.DCDReporter("traj_complex.dcd", 1000, enforcePeriodicBox=False, atomSubset=solute_idx))
t0 = time.time(); simulation.step(N_PROD); dt = time.time() - t0
print(f"production: {N_PROD * 0.002:.0f} ps in {dt:.0f} s  ->  {N_PROD * 0.002 / 1000 / (dt / 86400):.1f} ns/day")

# %% [markdown]
"""
The thermodynamic log shows the stages: temperature climbing to 300 K in the first picosecond, the density settling
near 1.0 g/mL once the barostat is on, the potential energy drifting down as the rebuilt loops relax.
"""

# %%
log = pd.read_csv("md_log.csv")
log.columns = [c.split(" (")[0].strip('#"') for c in log.columns]
fig, axes = plt.subplots(1, 3, figsize=(13, 3))
axes[0].plot(log["Time"], log["Temperature"]); axes[0].axhline(300, c="r", ls="--"); axes[0].set_ylabel("T (K)")
axes[1].plot(log["Time"], log["Density"]); axes[1].set_ylabel("density (g/mL)")
axes[2].plot(log["Time"], log["Potential Energy"] / 1000); axes[2].set_ylabel("E_pot (MJ/mol)")
for ax in axes: ax.set_xlabel("time (ps)")
plt.tight_layout(); plt.show()

# %% [markdown]
"""
## 7. Analysis

### Which trajectory?

If you ran on a GPU you have 100 ps of your own (50 frames). On CPU you have 2 ps — enough to check that the code
works, not enough to say anything. So we also load the **100 ps trajectory computed in advance** with the same protocol
(crystal start, same force field, loops built by a lucky seed before the relaxation step of section 2 existed; 5 ps
restrained NVT, 5 ps NPT and a further 80 ps of NPT before production; 50 frames every 2 ps; two CPU cores, one
night), so that everybody analyses the same thing. `TRAJ` decides which one the cells below use — your own run if it
has at least 25 frames, the precomputed one otherwise.
"""

# %%
own_frames = mda.Universe("complex_minimised.pdb", "traj_complex.dcd").trajectory.n_frames
TRAJ = "own" if own_frames >= 25 else "precomputed"
if TRAJ == "own":
    top_file, traj_file = "complex_minimised.pdb", "traj_complex.dcd"
else:
    top_file, traj_file = fetch("md/egfr_gefitinib_complex.pdb"), fetch("md/egfr_gefitinib_100ps.xtc")
u = mda.Universe(top_file, traj_file)
ref = mda.Universe(top_file)                       # the minimised starting structure = the crystal pose
u.select_atoms("protein").guess_bonds()            # the PDB carries CONECT records for the ligand only; ProLIF needs protein bonds too
print(f"using the {TRAJ} trajectory: {u.trajectory.n_frames} frames, {u.atoms.n_atoms} atoms, "
      f"{u.select_atoms('protein').n_residues} protein residues, ligand atoms: {u.select_atoms('resname LIG').n_atoms}")
dt_frame = 2.0    # ps between frames

# %% [markdown]
"""
### Hold the camera on the protein

As in session 10: before filming or measuring anything, every frame is superposed on the reference structure — here
on the protein's **Cα atoms**, so that the protein stays put in the picture and only its internal motion, and the
ligand's motion *relative to it*, remain. `in_memory=True` makes the alignment permanent for the rest of the notebook.
"""

# %%
align.AlignTraj(u, ref, select="protein and name CA", in_memory=True).run()
lig_ag = u.select_atoms("resname LIG")
u.trajectory[0]
pocket_ag = u.select_atoms("byres (protein and around 6 group lig)", lig=lig_ag)       # residues within 6 Å at the start
print(f"aligned {u.trajectory.n_frames} frames on the protein Cα | pocket: {pocket_ag.n_residues} residues within 6 Å of the ligand")

def write_movie(atoms, filename, frames):
    """Write an AtomGroup at the given frames as a multi-model PDB, for py3Dmol animation."""
    with mda.Writer(filename, atoms.n_atoms, multiframe=True) as w:
        for fr in frames:
            u.trajectory[fr]
            w.write(atoms)
    return open(filename).read()

# %% [markdown]
"""
### Movie 1 · The ligand in its pocket

Gefitinib (green) and the 35 residues within 6 Å of it, all 50 frames, 2 ps apart — 100 ps of the complex's life.
**Press ▶.** The pocket residues are drawn as thin sticks so that the ligand stays visible; Met793, the hinge, is in
white so you can watch the hydrogen bond.

What to look for: the quinazoline core barely moves — it is wedged between the two lobes; the anilino ring rocks in the
back pocket; the morpholine tail, which sticks out into the solvent, swings much more. Side chains around the ligand
flicker constantly. Nothing leaves. That "nothing leaves" is the first result of the session, and the RMSD plot below
turns it into a number.
"""

# %%
movie_pocket = write_movie(lig_ag + pocket_ag, "movie_pocket.pdb", range(u.trajectory.n_frames))
view = py3Dmol.view(width=650, height=450)
view.addModelsAsFrames(movie_pocket, "pdb")
view.setStyle({"stick": {"radius": 0.1, "colorscheme": "grayCarbon"}})
view.setStyle({"resi": 793}, {"stick": {"radius": 0.18, "colorscheme": "whiteCarbon"}})
view.setStyle({"resn": "LIG"}, {"stick": {"radius": 0.25, "colorscheme": "greenCarbon"}})
view.animate({"loop": "backAndForth", "interval": 120})
view.zoomTo({"resn": "LIG"}); view.show()

# %% [markdown]
"""
### Movie 2 · The whole protein breathing

The same 100 ps seen from far away: the protein backbone as a ribbon (25 of the 50 frames, to keep the file small),
the ligand in green, the **rebuilt loops in orange**. The body of the protein trembles; the orange loops flap. That
is not a bug — it is exactly what you would expect from residues that had no experimental position and were placed
by a program — and the RMSF analysis below quantifies it.
"""

# %%
backbone_ag = u.select_atoms("protein and (name N or name CA or name C or name O)") + lig_ag
movie_protein = write_movie(backbone_ag, "movie_protein.pdb", np.linspace(0, u.trajectory.n_frames - 1, 25).astype(int))
view = py3Dmol.view(width=650, height=450)
view.addModelsAsFrames(movie_protein, "pdb")
view.setStyle({"cartoon": {"color": "lightgray"}})
view.addStyle({"resi": ["721-723", "747-751", "985-1006"]}, {"cartoon": {"color": "orange"}})
view.setStyle({"resn": "LIG"}, {"stick": {"radius": 0.3, "colorscheme": "greenCarbon"}})
view.animate({"loop": "backAndForth", "interval": 150})
view.zoomTo(); view.show()

# %% [markdown]
"""
### RMSD — protein and ligand, on the protein's frame

We superpose every frame on the protein **backbone** of the starting structure and measure three things: the backbone
RMSD (how much the protein moved), the RMSD of the *ligand heavy atoms* in that same superposition (did the ligand stay
where it started — the docking criterion, now over time), and the RMSD of the three rebuilt loops (which we expect to
move most).
"""

# %%
R = rms.RMSD(u, ref, select="protein and backbone",
             groupselections=["resname LIG and not name H*", "protein and backbone and (resid 721:723 or resid 747:751 or resid 985:1006)"],
             ref_frame=0).run()
rmsd = pd.DataFrame(R.results.rmsd[:, 2:], columns=["protein backbone", "ligand (heavy atoms)", "rebuilt loops"])
rmsd["time (ps)"] = np.arange(len(rmsd)) * dt_frame
ax = rmsd.plot(x="time (ps)", figsize=(8, 3.5)); ax.set_ylabel("RMSD to start (Å)"); ax.axhline(2, c="gray", ls=":")
plt.show()
print(rmsd.drop(columns="time (ps)").describe().loc[["mean", "max"]].round(2))

# %% [markdown]
"""
### Choose your own atoms

The three curves above are three choices of atoms. The widget lets you make others: the **pocket residues** (are the
residues touching the ligand as stable as the ligand itself?), a **single residue** of your choice (type its number —
try 793 for the hinge, 1000 for the middle of the rebuilt loop, 790 for the gatekeeper), or the **ligand's core only**
(quinazoline + anilino ring, without the floppy tail — does the core sit still while the tail moves?). The reference
frame is adjustable too. Every curve is computed after superposition on the protein backbone, so a residue's RMSD
means "how far did it move relative to the protein", not "how far did the protein carry it".
"""

# %%
core_names = [lig_names[a.GetIdx()] for a in ligand.GetAtoms() if a.GetAtomicNum() > 1 and a.IsInRing() and a.GetIsAromatic()]
RMSD_CHOICES = {
    "ligand, all heavy atoms": "resname LIG and not name H*",
    "ligand, aromatic core only": "resname LIG and name " + " ".join(core_names),
    "protein backbone": "protein and backbone",
    "pocket residues (backbone)": "protein and backbone and byres around 6 group lig",
    "rebuilt loops (backbone)": "protein and backbone and (resid 721:723 or resid 747:751 or resid 985:1006)",
    "one residue (heavy atoms) - use the box below": None,
}
def show_rmsd(what, residue=793, reference_frame=0):
    sel = RMSD_CHOICES[what] or f"protein and resid {residue} and not name H*"
    group = u.select_atoms(sel, lig=lig_ag)
    if group.n_atoms < 3:
        print(f"selection '{sel}' has {group.n_atoms} atoms - nothing to measure"); return
    R = rms.RMSD(u, u, select="protein and backbone", groupselections=[sel], ref_frame=reference_frame).run()   # reference = a frame of the trajectory
    y = R.results.rmsd[:, 3]
    plt.figure(figsize=(8, 3.2))
    plt.plot(rmsd["time (ps)"], y, label=f"{what}" + (f" {residue}" if RMSD_CHOICES[what] is None else ""))
    plt.axvline(reference_frame * dt_frame, c="r", ls="--", label=f"reference = frame {reference_frame}"); plt.axhline(2, c="gray", ls=":")
    plt.xlabel("time (ps)"); plt.ylabel("RMSD (Å), protein-backbone frame"); plt.legend(); plt.show()
    print(f"{group.n_atoms} atoms | mean {y.mean():.2f} Å | max {y.max():.2f} Å")

interact(show_rmsd, what=list(RMSD_CHOICES), residue=(700, 1020, 1), reference_frame=(0, u.trajectory.n_frames - 1, 1));

# %% [markdown]
"""
The ligand stays within 1–2 Å of its crystal pose for the whole trajectory (mean 1.4 Å, never above 2 Å) — the same
size as the protein's own backbone motion (1.8 Å) — while the rebuilt loops, which had no experimental coordinates,
have drifted 4–5 Å. A ligand that left the pocket would show up as a steadily rising line. Remember the caveat from
Lecture 4: 100 ps is a very short time; a pose can be stable on this scale and still be wrong. What we have is *no
evidence against* the pose — and, as the next two analyses show, one detail worth a closer look.

### RMSF — where the protein moves, compared with what the crystal says

The per-residue **root-mean-square fluctuation** measures how much each Cα moves around its average position. The
crystallographer measured the same thing, in a different form: the **B-factor**, with $B = \\tfrac{8\\pi^2}{3}\\langle u^2\\rangle$.
So we can compare our 100 ps of simulation with the experiment residue by residue.
"""

# %%
ca = u.select_atoms("protein and name CA")                     # the trajectory is already aligned on these atoms
rmsf = rms.RMSF(ca).run().results.rmsf

bfac = {}
for l in open(pdb_path):
    if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[16] in " A":
        bfac[int(l[22:26])] = float(l[60:66])
b_rmsf = np.array([np.sqrt(3 * bfac[r] / (8 * np.pi**2)) if r in bfac else np.nan for r in ca.resids])

fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(ca.resids, rmsf, label=f"MD RMSF ({TRAJ} trajectory)")
ax.plot(ca.resids, b_rmsf, label="from crystal B-factors", alpha=0.7)
for a, b in [(721, 723), (747, 751), (985, 1006)]:
    ax.axvspan(a, b, color="orange", alpha=0.2)
ax.set_xlabel("residue"); ax.set_ylabel("Cα fluctuation (Å)"); ax.legend(); plt.show()
ok = ~np.isnan(b_rmsf)
from scipy.stats import spearmanr
print(f"Spearman correlation MD RMSF vs crystal B-factor (residues present in the crystal): {spearmanr(rmsf[ok], b_rmsf[ok])[0]:.2f}")

# %% [markdown]
"""
The rebuilt loops (orange bands) fluctuate most — as they should, they were disordered in the crystal. Elsewhere the two
curves follow each other: the simulation finds flexible what the crystal found flexible (rank correlation ≈ 0.6). The
agreement is far from perfect, and it should not be: B-factors also contain crystal-packing effects and refinement
conventions, and 100 ps samples only the fastest motions.

Colour the structure by RMSF to see where the motion is.
"""

# %%
# write the RMSF into the B-factor column (61-66) of the PDB text, then colour by that column
rmsf_by_resid = dict(zip(ca.resids.tolist(), rmsf.tolist()))
solute_pdb = "".join(l[:60] + f"{rmsf_by_resid.get(int(l[22:26]), 0.0):6.2f}" + l[66:] if l.startswith("ATOM") else l
                     for l in open(top_file))
view = py3Dmol.view(width=600, height=420)
view.addModel(solute_pdb, "pdb")
view.setStyle({"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb", "min": float(max(rmsf)), "max": float(min(rmsf))}}})
view.setStyle({"resn": "LIG"}, {"stick": {"colorscheme": "greenCarbon"}})
view.zoomTo(); view.show()
print("blue: rigid — red: mobile (colour = Cα RMSF of the residue)")

# %% [markdown]
"""
### The hinge hydrogen bond — first a picture

Session 09 identified the defining interaction of every quinazoline kinase inhibitor: the backbone **N–H of Met793**,
in the hinge between the two lobes, donating a hydrogen bond to a **nitrogen of the quinazoline ring**. Before
measuring it over time, look at it in the first frame: the ligand in green, Met793 in white, and the N–H···N contact
drawn as a dashed line with its length. In the crystal this distance is 2.98 Å.
"""

# %%
aromatic_N = [lig_names[a.GetIdx()] for a in ligand.GetAtoms() if a.GetSymbol() == "N" and a.GetIsAromatic()]
acceptors = u.select_atoms("resname LIG and name " + " ".join(aromatic_N))
donor_N, donor_H = u.select_atoms("resid 793 and name N"), u.select_atoms("resid 793 and name H")

def hinge_geometry(frame):
    """Distance N(Met793)···N(ligand, closest aromatic N) and the N-H···N angle at one frame."""
    u.trajectory[frame]
    d = np.linalg.norm(acceptors.positions - donor_N.positions[0], axis=1); k = int(d.argmin())
    v1 = donor_N.positions[0] - donor_H.positions[0]; v2 = acceptors.positions[k] - donor_H.positions[0]
    ang = np.degrees(np.arccos(np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)))
    return float(d[k]), float(ang), acceptors[k]

def hinge_view(frame, width=600, height=400):
    d, ang, acc = hinge_geometry(frame)
    (lig_ag + pocket_ag).write("_pocket_frame.pdb")
    v = py3Dmol.view(width=width, height=height)
    v.addModel(open("_pocket_frame.pdb").read(), "pdb")
    v.setStyle({"stick": {"radius": 0.1, "colorscheme": "grayCarbon"}})
    v.setStyle({"resi": 793}, {"stick": {"radius": 0.2, "colorscheme": "whiteCarbon"}})
    v.setStyle({"resn": "LIG"}, {"stick": {"radius": 0.22, "colorscheme": "greenCarbon"}})
    p1, p2 = donor_H.positions[0], acc.position
    v.addCylinder({"start": {"x": float(p1[0]), "y": float(p1[1]), "z": float(p1[2])}, "end": {"x": float(p2[0]), "y": float(p2[1]), "z": float(p2[2])},
                   "radius": 0.07, "dashed": True, "color": "magenta"})
    mid = (donor_N.positions[0] + acc.position) / 2
    v.addLabel(f"{d:.2f} Å", {"position": {"x": float(mid[0]), "y": float(mid[1]), "z": float(mid[2])}, "fontSize": 13, "backgroundColor": "magenta"})
    v.addResLabels({"resi": 793}, {"fontSize": 11, "backgroundOpacity": 0.6})
    v.zoomTo({"resn": "LIG"}); v.show()
    return d, ang

d0, a0 = hinge_view(0)
print(f"frame 0: N(Met793)···N(quinazoline) = {d0:.2f} Å, N–H···N angle = {a0:.0f}°   (crystal: 2.98 Å)")

# %% [markdown]
"""
### The hinge hydrogen bond over time

Session 09 identified the defining interaction: Met793's backbone N–H donating a hydrogen bond to a quinazoline
nitrogen. A hydrogen bond in MD is defined by geometry — donor–acceptor distance below ~3.5 Å and a D–H···A angle
above ~130° (stricter or looser thresholds exist; we report two). We follow both through the trajectory, exactly as
TeachOpenCADD T020 does for its EGFR ligand.
"""

# %%
print("quinazoline nitrogens:", aromatic_N, "| donor: Met793 N-H")
geom = np.array([hinge_geometry(fr)[:2] for fr in range(u.trajectory.n_frames)])
dist, angle = geom[:, 0], geom[:, 1]
t = np.arange(len(dist)) * dt_frame

fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
axes[0].plot(t, dist); axes[0].axhline(3.5, c="r", ls="--"); axes[0].set_ylabel("N(Met793) ··· N(ligand) (Å)")
axes[1].plot(t, angle); axes[1].axhline(130, c="r", ls="--"); axes[1].set_ylabel("N–H···N angle (°)")
for ax in axes: ax.set_xlabel("time (ps)")
plt.tight_layout(); plt.show()
for d_max, a_min in [(3.5, 130), (3.2, 150)]:
    present = (dist < d_max) & (angle > a_min)
    print(f"hydrogen bond present (d < {d_max} Å, angle > {a_min}°): {present.mean():.0%} of frames")
print(f"distance {dist.mean():.2f} ± {dist.std():.2f} Å, angle {angle.mean():.0f} ± {angle.std():.0f}°")

# %% [markdown]
"""
Here is the detail. The quinazoline nitrogen stays pointed at Met793 — the angle is a respectable 150° and the
distance never drifts away — but the distance settles at **3.5 ± 0.2 Å**, half an ångström longer than the 2.98 Å of
the crystal. A loose criterion counts the bond in ~40 % of frames; a strict one never. The ligand has slid about half
an ångström out of the hinge, which is also where its 1.4 Å RMSD comes from. TeachOpenCADD T020 saw the same for its
inhibitor: the hinge bond sitting "at the upper end of hydrogen-bond lengths". Different ligand, same residue, same
observation — and three candidate explanations you can test: the ligand's **charges** (MMFF94 rather than AM1-BCC may
under-polarise the ring nitrogen — Exercise 3), the **starting model** (the P-loop above the pocket was rebuilt from
nothing), and **time** (100 ps starting from a minimised crystal is not an equilibrated ensemble). This is what a
simulation is for: it turns "the pose looks fine" into a specific question.
"""

# %% [markdown]
"""
### Any frame, on demand

Movie 1 showed everything at once. This slider stops at one frame: the pocket at that moment with the hinge distance
drawn, and, underneath, the ligand RMSD curve with the frame marked. Look at the frames where the RMSD peaks and at
the frames where the hinge distance is longest — are they the same frames?
"""

# %%
lig_rmsd = rmsd["ligand (heavy atoms)"].values
def explore(frame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    axes[0].plot(rmsd["time (ps)"], lig_rmsd, c="green"); axes[0].scatter(frame * dt_frame, lig_rmsd[frame], c="red", s=80, zorder=3)
    axes[0].set_xlabel("time (ps)"); axes[0].set_ylabel("ligand RMSD (Å)")
    axes[1].plot(rmsd["time (ps)"], dist, c="magenta"); axes[1].scatter(frame * dt_frame, dist[frame], c="red", s=80, zorder=3)
    axes[1].axhline(3.5, c="gray", ls=":"); axes[1].set_xlabel("time (ps)"); axes[1].set_ylabel("hinge N···N (Å)")
    plt.tight_layout(); plt.show()
    d, ang = hinge_view(frame, width=550, height=360)
    print(f"frame {frame}  t = {frame * dt_frame:.0f} ps | ligand RMSD {lig_rmsd[frame]:.2f} Å | hinge N···N {d:.2f} Å, angle {ang:.0f}°")

interact(explore, frame=(0, u.trajectory.n_frames - 1, 1));

# %% [markdown]
"""
### Interaction persistence with ProLIF

The hinge bond is one interaction. ProLIF (session 09) computes the whole fingerprint **for every frame**; the fraction
of frames in which each interaction occurs is its **persistence**. This is the dynamic version of the interaction table of
session 09, and the honest way to say which contacts matter.
"""

# %%
lig_ag = u.select_atoms("resname LIG")
pocket_ag = u.select_atoms("protein and byres around 8 group lig", lig=lig_ag)
fp = plf.Fingerprint(["HBDonor", "HBAcceptor", "XBDonor", "Hydrophobic", "PiStacking", "Cationic", "Anionic"])
fp.run(u.trajectory[::2], lig_ag, pocket_ag, progress=False, n_jobs=1)
ifp = fp.to_dataframe()
persistence = ifp.mean().droplevel("ligand").sort_values(ascending=False)
persistence = persistence[persistence > 0.1]
ax = persistence.plot.barh(figsize=(7, 0.35 * len(persistence) + 1)); ax.invert_yaxis()
ax.set_xlabel("fraction of frames"); ax.set_title("interactions present in > 10 % of frames"); plt.show()
print(persistence.round(2).to_string())

# %% [markdown]
"""
The anchors of this pose are in the **back pocket**: the hydrophobic contacts of the halogenated anilino ring with
Lys745 and Met766 (present in ≥ 90 % of frames) and its halogen contact with Leu788 (≈ 85 %). The hinge hydrogen bond
to Met793, the interaction every textbook names first, is only present in ~45 % of frames by ProLIF's geometric
criterion — the same stretched contact we measured above. Contacts of the solvent-exposed tail (Val726, Leu844) come
and go. Compare with the crystal-pose fingerprint of session 09: every interaction that was there is here, now with a
frequency attached — and the frequencies do not rank them the way intuition would.
"""

# %% [markdown]
"""
### One picture to keep

Finally, a summary picture: the ligand at five moments of the trajectory superposed (green = start … purple = end),
with the pocket residues of the first frame as thin white sticks. This is the image that says "the pose survived".
"""

# %%
view = py3Dmol.view(width=600, height=420)
view.addModel(solute_pdb, "pdb")
view.setStyle({"cartoon": {"color": "lightgray", "opacity": 0.6}})
view.addStyle({"resi": [745, 788, 790, 793, 797, 844]}, {"stick": {"colorscheme": "whiteCarbon", "radius": 0.12}})
colors = ["greenCarbon", "yellowCarbon", "orangeCarbon", "redCarbon", "purpleCarbon"]
for k, frame in enumerate(np.linspace(0, u.trajectory.n_frames - 1, 5).astype(int)):
    u.trajectory[frame]
    lig_ag.write(f"lig_frame_{k}.pdb")
    view.addModel(open(f"lig_frame_{k}.pdb").read(), "pdb")
    view.setStyle({"model": k + 1}, {"stick": {"colorscheme": colors[k], "radius": 0.2}})
view.zoomTo({"resn": "LIG"}); view.show()
print("ligand at 5 times along the trajectory, green = start ... purple = end")

# %% [markdown]
"""
## 8. What this simulation can and cannot tell you

**Can:** whether a docked pose is *immediately* unstable; which interactions are anchors and which are decoration; how
the protein and the water accommodate the ligand; whether crystal contacts survive (the back-pocket ones did; the
hinge bond loosened — a result that points straight at the ligand charges). All of this at a cost of minutes on a GPU,
which is why a short MD run after docking has become routine.

**Cannot:** rank ligands by affinity (100 ps says nothing about ΔG — for that see Lecture 5: MM/GBSA, alchemical free
energies); find *another* pose (100 ps cannot leave the starting basin); sample slow protein motions — the DFG loop flip
or the αC-helix "in/out" transition of kinases takes microseconds. Real studies run several **replicas** of 100 ns to
microseconds, start from more than one pose, and check that replicas agree.

**What we simplified, and how it is done properly:** MMFF94 instead of AM1-BCC charges (use the OpenFF toolkit or
antechamber); a single protonation state (run both, or compute pKa shifts); loops rebuilt by PDBFixer (loop modelling
or a structure without gaps); one run, one seed.

## Exercises

### Exercise 1 — start from the docked pose
Set `LIGAND_START = "docked"` and rerun (or run the two set-ups side by side). Plot the ligand RMSD to the **crystal**
pose for both trajectories on one axis (align on the protein backbone of `protein.pdb`, compare the ligand heavy atoms
to `gefitinib_crystal.sdf` from session 09). Does the docked pose converge towards the crystal one, stay where it
started, or drift away?

### Exercise 2 — a protein-only run (T019 quiz)
Simulate the protein without gefitinib for the same time. Compare the RMSF of the pocket residues (Leu718, Val726,
Ala743, Lys745, Thr790, Met793, Cys797, Leu844) with and without the ligand. Does the ligand rigidify the pocket?

### Exercise 3 — do the charges matter?
Replace `mmff_charges` by Gasteiger charges (`AllChem.ComputeGasteigerCharges(mol)` then `atom.GetDoubleProp("_GasteigerCharge")`)
and rerun the ligand-in-vacuum check and, if you have a GPU, the simulation. How different are the two charge sets
(correlation, largest difference), and does the hinge hydrogen bond persistence change?

### Exercise 4 — break the force field on purpose
Change the type of the quinazoline nitrogens from `nb` to `n3` in `types` before writing the template. What does the
ligand-in-vacuum check report? Why is this a useful test to run every time you parametrise a molecule?

### Exercise 5 — the TeachOpenCADD trajectory
TeachOpenCADD T020 ships a **1 ns** trajectory of EGFR with a different inhibitor (PDB 3POZ, ligand `03P`, 50 frames):
```python
base = "https://raw.githubusercontent.com/volkamerlab/teachopencadd/master/teachopencadd/talktorials/T020_md_analysis/data/"
for f in ["topology.pdb", "trajectory.xtc"]:
    open(f, "wb").write(requests.get(base + f, timeout=120).content)
u2 = mda.Universe("topology.pdb", "trajectory.xtc")
```
Run the RMSD, RMSF and hinge-hydrogen-bond analyses on it (`resname 03P`; the hinge is again Met793). Which system
keeps its hinge bond more consistently?

### Exercise 6 — the 4 fs step (GPU)
Rebuild the system with `hydrogenMass=1.5 * unit.amu` (and then with `3 * unit.amu`), keep 2 fs for the two
equilibration stages, then `integrator.setStepSize(4 * unit.femtoseconds)` for production. Watch the temperature and the
potential energy in `md_log.csv`. How far does each run get before the integration fails — if it fails at all in your
100 ps? What does that tell you about testing a protocol on a short run?
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solutions (sketches)</b></summary>

```python
# 1 — after a second run with LIGAND_START = "docked", saved as traj_docked.dcd / complex_docked.pdb:
#     build a reference universe from protein.pdb + gefitinib_crystal.sdf is fiddly because atom orders differ;
#     the simplest route is to keep the *same* topology (both runs have identical atom order) and compare each frame's
#     ligand heavy atoms with the crystal-start minimised structure complex_minimised.pdb:
u_d = mda.Universe("complex_docked.pdb", "traj_docked.dcd")
R_d = rms.RMSD(u_d, ref, select="protein and backbone", groupselections=["resname LIG and not name H*"]).run()
plt.plot(rmsd["time (ps)"], rmsd["ligand (heavy atoms)"], label="crystal start")
plt.plot(np.arange(len(R_d.results.rmsd)) * dt_frame, R_d.results.rmsd[:, 3], label="docked start"); plt.legend(); plt.show()

# 2 — rebuild the Modeller from protein.pdb only (no modeller.add(...)), same protocol, then rms.RMSF on the pocket CA atoms
pocket = [718, 726, 743, 745, 790, 793, 797, 844]
print(pd.Series(rmsf, index=ca.resids).loc[pocket])

# 3
AllChem.ComputeGasteigerCharges(ligand)
q_g = np.array([a.GetDoubleProp("_GasteigerCharge") for a in ligand.GetAtoms()])
print(np.corrcoef(q_g, charges)[0, 1], np.abs(q_g - charges).max())

# 4
bad = ["n3" if t == "nb" else t for t in types]
xml_bad, _ = ligand_template_xml(ligand, bad, charges); open("bad_template.xml", "w").write(xml_bad)
ff_bad = app.ForceField("amber14-all.xml", "amber14/tip3p.xml", gaff_path, "bad_template.xml")
# ... repeat the vacuum minimisation: the rings pucker (n3 is sp3) and the shift grows well beyond 0.5 A

# 5 — same functions with u2, "resname 03P", "resid 793"
```
</details>

## Further reading
- TeachOpenCADD **T019** and **T020**; the OpenMM user guide (<http://docs.openmm.org>) and cookbook.
- Wang *et al.*, *Development and testing of a general amber force field*, J. Comput. Chem. **2004**, 25, 1157 (GAFF).
- Jakalian *et al.*, *Fast, efficient generation of high-quality atomic charges. AM1-BCC*, J. Comput. Chem. **2002**, 23, 1623.
- Braun *et al.*, *Best practices for foundations in molecular simulations*, LiveCoMS **2019**, 1, 5957.
- Hopkins *et al.*, *Long-time-step molecular dynamics through hydrogen mass repartitioning*, J. Chem. Theory Comput. **2015**, 11, 1864.
- Bouysset & Fiorucci, *ProLIF: a library to encode molecular interactions as fingerprints*, J. Cheminform. **2021**, 13, 72.

---

**This was the last session. Congratulations!** Over eleven notebooks you went from `print("Hello")` to curating real
bioactivity data, building and validating QSAR models, training graph networks, generating molecules, putting an LLM to
work with chemistry tools — and finally to giving molecules coordinates, docking them into a protein and letting the
whole complex move.

Notice what the last three sessions changed about the previous eight. Everything from session 02 onwards described
molecules as **graphs**: a fingerprint, a descriptor vector, an adjacency matrix. That approximation carried us a long
way, and for most property-prediction tasks it still wins. But a ligand does not bind as a graph; it binds as a
particular conformation of a flexible object, in water, to a protein that is itself moving. Molecular modeling is where
that physics comes back, and it is the bridge to what the field is building next: 3D and equivariant neural networks,
structure-based generative models, co-folding, and free-energy methods.

Two habits are worth keeping from all eleven sessions: *look at your data before you model it*, and *ask what your
representation is throwing away*.
"""
