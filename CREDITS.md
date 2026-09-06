# Credits and attribution

These tutorials are **adaptations** of open teaching material generously published by the chemoinformatics community.
Nothing here would exist without it, and the credit for the substance belongs entirely to the authors listed below —
**thank you** for publishing your work in a form others can learn from and build on.

Below is the full list of sources, what was taken from each, and under which licence. Every notebook also names its own
sources in its first cell. If you reuse this material, please keep these attributions and cite the original authors
rather than this repository.

---

## Primary sources

### TeachOpenCADD — Volkamer lab
<https://github.com/volkamerlab/teachopencadd> · **CC BY 4.0** (code and text)

- **T001** Compound data acquisition (ChEMBL) → session 04 (ChEMBL workflow, IC50/pIC50 theory). The cached dataset
  `data/EGFR_compounds_chembl.csv` is the output of this talktorial.
- **T002** Molecular filtering: ADME and lead-likeness → session 05 (property filters, Ro5 statistics).
- **T003** Unwanted substructures → session 05 (PAINS).
- **T005** Compound clustering → session 05 (Butina clustering, distance matrices, cluster analysis).
- **T007** Ligand-based screening: machine learning → session 06 (QSAR setup, metrics, ROC curves).
- **T008/T011/T013** Protein data acquisition (PDB), online API webservices, PubChem → session 04.
- **T019/T020** Molecular dynamics simulation and analysis → session 03 (OpenMM workflow, force-field theory, analysis logic).
- **T022** Ligand-based screening: neural networks → session 07.
- **T033** Molecular representations → session 02 (representation taxonomy, conformers, fingerprint bit visualisation).
- **T034/T035** Recurrent and graph neural networks → sessions 07–08.

References: Sydow *et al.*, *TeachOpenCADD: a teaching platform for computer-aided drug design*, J. Cheminform. **2019**, 11, 29;
Sydow *et al.*, Nucleic Acids Res. **2022**, 50, W753.

### Practical Cheminformatics Tutorials — Patrick Walters
<https://github.com/PatWalters/practical_cheminformatics_tutorials> · **MIT**

- `fundamentals/A_Whirlwind_Introduction_To_The_RDKit`, `SMILES_tutorial`, `SMARTS_tutorial`, `pandas_intro` → sessions 00–01.
- `misc/ChEMBL_data_curation`, `working_with_ChEMBL_drug_data` → sessions 04–05.
- `clustering/taylor_butina_clustering`, `kmeans_clustering` → session 05.
- `sar_analysis/find_scaffolds`, `R_group_analysis` → session 05 (scaffold analysis).
- `misc/visualizing_chemical_space` → session 05 (PCA + t-SNE/UMAP workflow).
- `ml_models/*` (`regression_model`, `classification_model`, `cross_validation`, `comparing_*`, `QSAR_in_8_lines`) → session 06.
- `generative/SMILES_RNN` → session 08.
- `chemprop/run_chemprop` → session 07.
- Datasets: `data/hERG.csv` (session 05 exercises) and `chembl_drugs.smi` from
  <https://github.com/PatWalters/datafiles>.

Also: Pat Walters' blog *Practical Cheminformatics* — <https://practicalcheminformatics.blogspot.com>.

### AI for Chemistry (EPFL CH-457) — Schwaller group (LIAC)
<https://github.com/schwallergroup/ai4chem_course> · **MIT**

- `01 - Basics` (`01a_python_crash_course`, `01b_python_essentials_pandas`, `01c_python_essentials_plotting`, `01d_rdkit_basics`) → sessions 00–01.
- `02 - Supervised Learning` → session 06.
- `03 - Intro to Deep Learning` (`01_intro_to_dl`, `02_graph_nns`, `03_gnn_simple_example`) → session 07.
- `04 - Unsupervised Learning` (`Clustering`, `DimensionalityReduction`) → session 05.
- `05 - Generative Models`, `06 - Generative Models 2 / SMILES-LSTM-Walkthrough` → session 08.

### Practical Programming in Chemistry (EPFL CH-200) — Schwaller group
<https://github.com/schwallergroup/practical-programming-in-chemistry-exercises> · **MIT** — exercise style, session 00.

### MolSSI cheminformatics workshop — Jessica A. Nash
<https://github.com/MolSSI-Education/molssi-cheminformatics> · **MIT**

- `00_python_basics` → session 00 (the ΔG/energy-unit examples and "check your understanding" style).
- `01_molecule_representation`, `02_rdkit_intro` → session 01 (atoms/bonds/editing molecules).
- `03_molecular_similarity` → session 02.
- `06_sklearn_fitting_1`, `07_ESOL_fitting` → session 06.

### Deep Learning for Molecules and Materials — Andrew D. White
<https://dmol.pub> · **CC BY-NC 3.0** · Living J. Comput. Mol. Sci. **2021**, 3, 1499
(also in the course bibliography) — the representation taxonomy (session 02) and the deep-learning/GNN chapters (session 07).

### IBM3202 — Molecular Modeling and Simulation — pb3lab (PUC Chile)
<https://github.com/pb3lab/ibm3202> · **MIT** — `lab07_MDsims`, `lab08_MDanalysis` → session 03
(Colab-first MD teaching approach, trajectory analysis).

### OpenMM and its cookbook
<https://github.com/openmm/openmm> (MIT/LGPL), <https://github.com/openmm/openmm-cookbook> (MIT) — session 03 workflow;
`data/md/alanine_dipeptide_solvated.pdb` and `data/md/villin_headpiece_solvated.pdb` come from these repositories.

### MDAnalysis
<https://www.mdanalysis.org> · code GPL-2.0+, docs CC BY-SA — session 03 analysis (RMSD, RMSF, Ramachandran).
Michaud-Agrawal *et al.*, J. Comput. Chem. **2011**; Gowers *et al.*, SciPy **2016**.

### CCPBioSim biosim-analysis-workshop
<https://github.com/CCPBioSim/biosim-analysis-workshop> · **CC BY-SA 4.0** — session 03 analysis structure.

### Tutorial: LLM agents for chemistry — hesengg
<https://github.com/hesengg/Tutorial_LLM_Agent_Chemistry> — session 09: the multi-agent architecture
(expert agents + coordinator), tool-design conventions and the RDKit tool set.

### smolagents — Hugging Face
<https://github.com/huggingface/smolagents> · **Apache-2.0** — the agent framework used in session 09
(`ToolCallingAgent`, `CodeAgent`, `@tool`, `managed_agents`).

### Introduction to AI for pharma students — Koch group
<https://github.com/kochgroup/intro_pharma_ai> · **CC BY-NC-SA 4.0** — consulted for the pedagogical progression of
session 07 (first neural net → PyTorch → GNN). No code reproduced (note the NonCommercial-ShareAlike terms).

---

## Software, models and datasets

| project | used for | licence |
|---|---|---|
| [RDKit](https://www.rdkit.org) | everything chemical | BSD-3 |
| [scikit-learn](https://scikit-learn.org) | classical ML (session 06) | BSD-3 |
| [XGBoost](https://xgboost.readthedocs.io) | gradient boosting (session 06) | Apache-2.0 |
| [PyTorch](https://pytorch.org) / [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) | deep learning, GNNs (sessions 07–08) | BSD-3 / MIT |
| [UMAP](https://github.com/lmcinnes/umap) | chemical-space maps (session 05) | BSD-3 |
| [mols2grid](https://github.com/cbouy/mols2grid) | interactive molecule grids | Apache-2.0 |
| [SELFIES](https://github.com/aspuru-guzik-group/selfies) | robust string representation (session 02) | Apache-2.0 |
| [chembl_webresource_client](https://github.com/chembl/chembl_webresource_client) | ChEMBL API (session 04) | Apache-2.0 |
| [chembl-downloader](https://github.com/cthoyt/chembl-downloader) (C. T. Hoyt) | mentioned for reproducible ChEMBL extraction | MIT |
| [PubChemPy](https://github.com/mcs07/PubChemPy) | PubChem access (session 04) | MIT |
| [ChEMBL structure pipeline](https://github.com/chembl/ChEMBL_Structure_Pipeline) | standardisation reference (session 05) | MIT |
| [py3Dmol](https://github.com/3dmol/3Dmol.js) | 3D visualisation | BSD-3 |
| [ChemBERTa-77M-MTR](https://huggingface.co/DeepChem/ChemBERTa-77M-MTR) (DeepChem) | pre-trained embeddings (session 07) | MIT |
| [Chemprop](https://github.com/chemprop/chemprop) | D-MPNN, discussed in session 07 | MIT |
| [REINVENT 4](https://github.com/MolecularAI/REINVENT4) (MolecularAI) | RL formulation of session 08 | Apache-2.0 |
| [CReM](https://github.com/DrrDom/crem) (P. Polishchuk) | mutation-based design idea (session 08) | BSD-3 |
| [GuacaMol](https://github.com/BenevolentAI/guacamol) / [MOSES](https://github.com/molecularsets/moses) | generative metrics (session 08) | MIT |
| [OpenADMET](https://github.com/OpenADMET) challenge tutorials & datasets | ADMET data (session 04) | Apache-2.0; ExpansionRx data CC BY 4.0 |
| [chemical_vae](https://github.com/aspuru-guzik-group/chemical_vae) (Gómez-Bombarelli *et al.*) | source of the ZINC-250k set subsampled into `data/zinc_50k.csv` | Apache-2.0 |
| [DeepChem](https://github.com/deepchem/deepchem) / MoleculeNet | ESOL dataset | MIT |
| [ChEMBL](https://www.ebi.ac.uk/chembl) | bioactivity data | CC BY-SA 3.0 |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov) | compound data | public domain |
| [RCSB PDB](https://www.rcsb.org) | structures | CC0 |

Additional inspiration for session 09 tasks: `MauricioCafiero/CheMLAgent`, `hoon-ock/AgentD`,
ChemCrow (Bran *et al.*, Nat. Mach. Intell. 2024), Coscientist (Boiko *et al.*, Nature 2023).

---

## What this compilation adds

Everything of substance above comes from the projects listed. What was added here — updating the code to current
library versions, porting every notebook to run start-to-finish in Colab, sequencing, exercises, and connective
text — was assembled and revised **with Claude** (Anthropic) and then reviewed. Specifically: the sequencing into
nine self-contained sessions; the learning objectives, exercises and hidden solutions; the unified EGFR dataset and its
curation pipeline (session 05); the pre-trained SMILES LSTM and the REINVENT-style RL and genetic-algorithm
implementations of session 08; the chemistry tool set, multi-agent system and agent benchmark of session 09; the
jupytext build system; and the connective text.

None of that would be worth much on its own. If you cite anything from this repository, cite the original projects.
