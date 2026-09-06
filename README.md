# Chemoinformatics tutorials — UFAZ

Practical notebooks for the **Chemoinformatics** course (L2 S3) at the
[French-Azerbaijani University (UFAZ)](https://ufaz.az), taught by **Alexey Orlov**
(Laboratoire de Chémoinformatique, Université de Strasbourg).

Nine sessions take students from their first line of Python to AI agents that design and evaluate molecules.
**Everything runs in Google Colab** — click a badge, run the cells, no installation.

## The course

| # | Session | Open in Colab | What you learn |
|---|---|---|---|
| 00 | **Python crash course for chemists** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/00_python_crash_course.ipynb) | notebooks, variables, collections, loops, functions, numpy, matplotlib, pandas on a real solubility dataset |
| 01 | **RDKit basics** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/01_rdkit_basics.ipynb) | SMILES, canonical forms, atoms and bonds, SMARTS substructure search, properties, Lipinski, SDF files |
| 02 | **Molecular representations** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/02_molecular_representations.ipynb) | InChI, SELFIES, molecular graphs, descriptors, fingerprints, Tanimoto similarity, 3D shape |
| 03 | **Molecular modeling & MD** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/03_molecular_modeling_and_md.ipynb) | conformers, force fields, a real OpenMM simulation, MDAnalysis, Ramachandran plots |
| 04 | **Chemical databases** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/04_chemical_databases.ipynb) | PubChem PUG-REST, ChEMBL client, IC50/pIC50, Hugging Face datasets (OpenADMET), RCSB PDB |
| 05 | **Exploratory data analysis** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/05_exploratory_data_analysis.ipynb) | standardisation, deduplication, PAINS, scaffolds, Butina & k-means clustering, PCA/UMAP, activity cliffs |
| 06 | **Classical machine learning** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/06_classical_machine_learning.ipynb) | QSAR/QSPR, RF/XGBoost/SVM/kNN, cross-validation, metrics, scaffold split, interpretation, applicability domain |
| 07 | **Deep learning** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/07_deep_learning.ipynb) | PyTorch from scratch, MLPs, graph neural networks (PyG), ChemBERTa embeddings, Chemprop |
| 08 | **Generative AI** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/08_generative_ai.ipynb) | SMILES LSTM, validity/uniqueness/novelty, fine-tuning, REINVENT-style RL, genetic algorithms, synthesisability |
| 09 | **Agentic AI for chemistry** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AxelRolov/chemoinformatics_tutorials/blob/main/notebooks/09_agentic_ai_for_chemistry.ipynb) | LLM hallucination, tool calling, chemistry tools, single & multi-agent systems, agent benchmarking |

Each notebook is self-contained: it installs what it needs, downloads its data from this repository, and ends with
exercises (with hidden solutions) and further reading. Sessions 03, 07 and 08 benefit from a GPU runtime
(*Runtime → Change runtime type → T4 GPU*), but all of them also run on CPU.

## For students

1. Click the **Open in Colab** badge of the session.
2. Run the first cell (⚙️ Setup) and wait for the installation to finish.
3. Work through the notebook with `Shift + Enter`. Change things, break them, fix them.
4. To keep your work: *File → Save a copy in Drive*.

Session 09 needs a free LLM API key — see the instructions at the top of that notebook
(<https://aistudio.google.com/apikey>, no credit card).

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
models/        pre-trained SMILES LSTM for session 08
utils/         helper scripts (e.g. pre-training the LSTM)
CREDITS.md     full attribution of every adapted source
```

## Data

| file | content | source & licence |
|---|---|---|
| `esol_delaney.csv` | 1128 compounds, measured aqueous solubility | Delaney 2004, via [DeepChem/MoleculeNet](https://github.com/deepchem/deepchem) (MIT) |
| `EGFR_compounds_chembl.csv` | 5568 EGFR inhibitors with IC50 | [TeachOpenCADD](https://github.com/volkamerlab/teachopencadd) T001 output; ChEMBL (CC BY-SA 3.0) |
| `EGFR_curated.csv` | the same set, standardised and deduplicated in session 05 | derived here; ChEMBL (CC BY-SA 3.0) |
| `hERG_chembl_walters.csv` | 4042 compounds with hERG pIC50 | [PatWalters/practical_cheminformatics_tutorials](https://github.com/PatWalters/practical_cheminformatics_tutorials) (MIT); ChEMBL data |
| `chembl_drugs_walters.smi` | 1203 approved drugs | [PatWalters/datafiles](https://github.com/PatWalters/datafiles); ChEMBL (CC BY-SA 3.0) |
| `zinc_50k.csv` | 50 000 ZINC molecules with logP/QED/SA | subset of the ZINC-250k set from [chemical_vae](https://github.com/aspuru-guzik-group/chemical_vae) (Apache-2.0) |
| `md/alanine_dipeptide_solvated.pdb` | solvated alanine dipeptide | [OpenMM](https://github.com/openmm/openmm) test systems (MIT) |
| `md/villin_headpiece_solvated.pdb` | solvated villin headpiece HP35 | [OpenMM cookbook](https://github.com/openmm/openmm-cookbook) (MIT) |

## Credits

These notebooks stand on the shoulders of the open teaching material of the chemoinformatics community.
Each notebook names its sources in its first cell; [`CREDITS.md`](CREDITS.md) lists them all with licences.
The largest debts are to **TeachOpenCADD** (Volkamer lab), **Practical Cheminformatics Tutorials** (Pat Walters),
**AI for Chemistry / CH-457** (Schwaller group, EPFL), the **MolSSI** cheminformatics workshop,
*Deep Learning for Molecules and Materials* (Andrew White), **IBM3202** (pb3lab), the **OpenMM** and **MDAnalysis**
projects, **smolagents** (Hugging Face) and the *LLM agents for chemistry* tutorial by hesengg.

If you reuse this material, please keep the attributions and cite the original authors.

## Licence

Course material (notebooks, text, figures): **CC BY 4.0** — see [`LICENSE`](LICENSE).
Code snippets: **MIT**. Adapted third-party material keeps its original licence, as recorded in `CREDITS.md`;
in particular, content derived from TeachOpenCADD is CC BY 4.0 and content derived from ChEMBL is CC BY-SA 3.0.
