# Chemoinformatics tutorials

An eleven-session practical course in chemoinformatics, from a first line of Python to AI agents that design and
evaluate molecules, and on to docking and molecular dynamics of a drug in its protein. **Everything runs in Google Colab** — click a badge, run the cells, no installation.

## This is a compilation — the credit belongs to others

These notebooks are assembled from the excellent open resources that
the chemoinformatics community has published, in particular:

- **[TeachOpenCADD](https://github.com/volkamerlab/teachopencadd)** — the Volkamer lab's teaching platform for
  computer-aided drug design (talktorials T001–T035)
- **[Practical Cheminformatics Tutorials](https://github.com/PatWalters/practical_cheminformatics_tutorials)** —
  Pat Walters' collection, and his blog *Practical Cheminformatics*
- **[AI for Chemistry (EPFL CH-457)](https://github.com/schwallergroup/ai4chem_course)** and
  **[Practical Programming in Chemistry (CH-200)](https://github.com/schwallergroup/practical-programming-in-chemistry-exercises)** —
  the Schwaller group at EPFL
- **[MolSSI cheminformatics workshop](https://github.com/MolSSI-Education/molssi-cheminformatics)** — Jessica A. Nash
  and the Molecular Sciences Software Institute
- **[Deep Learning for Molecules and Materials](https://dmol.pub)** — Andrew D. White
- **[IBM3202](https://github.com/pb3lab/ibm3202)** — the pb3lab's Colab tutorials on molecular modeling and simulation
- **[OpenMM](https://github.com/openmm/openmm)** and its cookbook, **[MDAnalysis](https://www.mdanalysis.org)**, and the
  **[CCPBioSim](https://github.com/CCPBioSim/biosim-analysis-workshop)** analysis workshop
- **[smolagents](https://github.com/huggingface/smolagents)** (Hugging Face) and the
  **[LLM agents for chemistry](https://github.com/hesengg/Tutorial_LLM_Agent_Chemistry)** tutorial
- and of course **[RDKit](https://www.rdkit.org)**, without which none of this exists

**Thank you to all of these authors** for making their work reusable. Every notebook names its own sources in its
first cell, and [`CREDITS.md`](CREDITS.md) lists all of them with licences.

What this collection adds: the material was **updated** to current library versions, **ported** so that each notebook
runs start to finish in Colab, **sequenced** into eleven self-contained sessions, and given exercises with hidden
solutions and connective text. That assembly and revision were done **with Claude** (Anthropic) and then reviewed.
Any errors introduced along the way are ours, not the original authors'.

## The course

| # | Session | Open in Colab | What you learn |
|---|---|---|---|
| 00 | **Python crash course for chemists** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/00_python_crash_course.ipynb) | notebooks, variables, collections, loops, functions, numpy, matplotlib, pandas on a real solubility dataset |
| 01 | **RDKit basics** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/01_rdkit_basics.ipynb) | SMILES, canonical forms, atoms and bonds, SMARTS substructure search, properties, Lipinski, SDF files |
| 02 | **Molecular representations** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/02_molecular_representations.ipynb) | InChI, SELFIES, molecular graphs, descriptors, fingerprints, Tanimoto similarity, 3D shape |
| 03 | **Chemical databases** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/03_chemical_databases.ipynb) | PubChem PUG-REST, ChEMBL client, IC50/pIC50, Hugging Face datasets (OpenADMET), RCSB PDB |
| 04 | **Exploratory data analysis** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/04_exploratory_data_analysis.ipynb) | standardisation, deduplication, PAINS, scaffolds, Butina & k-means clustering, PCA/UMAP, activity cliffs |
| 05 | **Classical machine learning** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/05_classical_machine_learning.ipynb) | QSAR/QSPR, RF/XGBoost/SVM/kNN, cross-validation, metrics, scaffold split, interpretation, applicability domain |
| 06 | **Deep learning** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/06_deep_learning.ipynb) | PyTorch from scratch, MLPs, graph neural networks (PyG), ChemBERTa embeddings, Chemprop |
| 07 | **Generative AI** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/07_generative_ai.ipynb) | SMILES LSTM, validity/uniqueness/novelty, fine-tuning, REINVENT-style RL, genetic algorithms, synthesisability |
| 08 | **Agentic AI for chemistry** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/08_agentic_ai_for_chemistry.ipynb) | LLM hallucination, tool calling, chemistry tools, single & multi-agent systems, agent benchmarking |
| 09 | **Protein–ligand docking** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/09_protein_ligand_docking.ipynb) | PDB structure preparation, AutoDock Vina, redocking and the 2 Å rule, exhaustiveness, a mini virtual screen (score ≠ affinity), interaction fingerprints |
| 10 | **Molecular modeling & MD** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/10_molecular_modeling_and_md.ipynb) | conformers, force fields, a real OpenMM simulation, MDAnalysis, Ramachandran plots |
| 11 | **MD of a protein–ligand complex** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/11_protein_ligand_md.ipynb) | loop building, GAFF ligand parametrisation, solvation, minimisation/equilibration/production, RMSD/RMSF vs B-factors, hinge H-bond, interaction persistence |

Each notebook is self-contained: it installs what it needs, downloads its data from this repository, and ends with
exercises (with hidden solutions) and further reading. Sessions 06, 07, 10 and 11 benefit from a GPU runtime
(*Runtime → Change runtime type → T4 GPU*), but all of them also run on CPU (session 11 then analyses a
trajectory computed in advance instead of its own).

## For students

1. Click the **Open in Colab** badge of the session.
2. Run the first cell (⚙️ Setup) and wait for the installation to finish.
3. Work through the notebook with `Shift + Enter`. Change things, break them, fix them.
4. To keep your work: *File → Save a copy in Drive*.

Session 08 needs an LLM API key — see the instructions at the top of that notebook. It is set up for
**DeepSeek** (<https://platform.deepseek.com/api_keys>); switching to another provider, including Google Gemini's
free tier, is a one-line change in that notebook.

## For instructors

Notebooks are generated from [jupytext](https://jupytext.readthedocs.io) percent-format Python sources in `src_nb/`,
which is what you should edit (they diff cleanly in git). Rebuild with:

```bash
pip install jupytext nbformat nbclient
python src_nb/build.py            # rebuild all notebooks (clean, no outputs)
python src_nb/build.py 05         # rebuild one
python src_nb/build.py 05 --execute   # rebuild and run it, reporting any error
```

The build script injects the *Open in Colab* badge, strips outputs and sets the kernel metadata.
If you fork the repository, change `REPO` in `src_nb/build.py` and `REPO_RAW` in each notebook's setup cell
so the badges and data downloads point at your fork.

```
notebooks/     the .ipynb files students open (generated - do not edit by hand)
src_nb/        jupytext sources + build.py
data/          datasets used by the notebooks (see data/README.md for provenance)
models/        pre-trained SMILES LSTM for session 07
utils/         helper scripts (e.g. pre-training the LSTM)
CREDITS.md     full attribution of every adapted source
```

## Data

| file | content | source & licence |
|---|---|---|
| `esol_delaney.csv` | 1128 compounds, measured aqueous solubility | Delaney 2004, via [DeepChem/MoleculeNet](https://github.com/deepchem/deepchem) (MIT) |
| `EGFR_compounds_chembl.csv` | 5568 EGFR inhibitors with IC50 | [TeachOpenCADD](https://github.com/volkamerlab/teachopencadd) T001 output; ChEMBL (CC BY-SA 3.0) |
| `EGFR_curated.csv` | the same set, standardised and deduplicated with the session-04 pipeline (5511 compounds; session 04 itself runs on a 1500-record subset for speed) | derived here; ChEMBL (CC BY-SA 3.0) |
| `hERG_chembl_walters.csv` | 4042 compounds with hERG pIC50 | [PatWalters/practical_cheminformatics_tutorials](https://github.com/PatWalters/practical_cheminformatics_tutorials) (MIT); ChEMBL data |
| `chembl_drugs_walters.smi` | 1203 approved drugs | [PatWalters/datafiles](https://github.com/PatWalters/datafiles); ChEMBL (CC BY-SA 3.0) |
| `zinc_50k.csv` | 50 000 ZINC molecules with logP/QED/SA | subset of the ZINC-250k set from [chemical_vae](https://github.com/aspuru-guzik-group/chemical_vae) (Apache-2.0) |
| `md/alanine_dipeptide_solvated.pdb` | solvated alanine dipeptide | [OpenMM](https://github.com/openmm/openmm) test systems (MIT) |
| `md/villin_headpiece_solvated.pdb` | solvated villin headpiece HP35 | [OpenMM cookbook](https://github.com/openmm/openmm-cookbook) (MIT) |
| `pdb/4WKQ.pdb` | EGFR kinase domain with gefitinib, 1.85 Å (coordinates and header; ANISOU/refinement remarks stripped) — offline fallback for sessions 09 and 11 | [RCSB PDB](https://www.rcsb.org/structure/4WKQ) (CC0) |
| `md/gaff-2.11.xml` | the General AMBER Force Field 2.11 as an OpenMM ffxml | [openmmforcefields](https://github.com/openmm/openmmforcefields) (MIT); parameters from AmberTools |
| `md/egfr_gefitinib_complex.pdb`, `md/egfr_gefitinib_100ps.xtc`, `md/egfr_gefitinib_md_log.csv` | 100 ps MD of EGFR–gefitinib (protein + ligand, 50 frames), produced with the session-11 code | derived here |
| `md/gefitinib_docked_poses.sdf` | the nine Vina poses of session 09 (redocking into 4WKQ) | derived here |

## Credits

These notebooks stand on the shoulders of the open teaching material of the chemoinformatics community.
Each notebook names its sources in its first cell; [`CREDITS.md`](CREDITS.md) lists them all with licences.
The largest debts are to **TeachOpenCADD** (Volkamer lab), **Practical Cheminformatics Tutorials** (Pat Walters),
**AI for Chemistry / CH-457** (Schwaller group, EPFL), the **MolSSI** cheminformatics workshop,
*Deep Learning for Molecules and Materials* (Andrew White), **IBM3202** (pb3lab), the **OpenMM** and **MDAnalysis**
projects, **smolagents** (Hugging Face) and the *LLM agents for chemistry* tutorial by hesengg.

If you reuse this material, please keep the attributions and cite the **original authors**, not this repository.

## Licence

The compilation (sequencing, updated code, exercises, connective text): **CC BY 4.0** — see [`LICENSE`](LICENSE).
Code snippets: **MIT**. Adapted third-party material keeps its original licence, as recorded in `CREDITS.md`;
in particular, content derived from TeachOpenCADD is CC BY 4.0 and content derived from ChEMBL is CC BY-SA 3.0.
