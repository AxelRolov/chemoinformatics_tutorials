# Datasets

All files here are redistributed from public sources for teaching purposes. Provenance and licences:

| file | rows | content | source | licence |
|---|---|---|---|---|
| `esol_delaney.csv` | 1128 | measured aqueous solubility (logS) + 4 ESOL descriptors | Delaney, *J. Chem. Inf. Comput. Sci.* **2004**, 44, 1000, via [DeepChem](https://github.com/deepchem/deepchem/blob/master/datasets/delaney-processed.csv) | MIT (file), data from the publication |
| `EGFR_compounds_chembl.csv` | 5568 | EGFR (P00533) inhibitors: ChEMBL ID, IC50 (nM), SMILES, pIC50 | output of [TeachOpenCADD T001](https://github.com/volkamerlab/teachopencadd) (ChEMBL 27) | notebook CC BY 4.0; data ChEMBL CC BY-SA 3.0 |
| `EGFR_curated.csv` | 5511 | the same set after standardisation, tautomer canonicalisation, InChIKey deduplication, median aggregation and removal of contradictory replicates; plus `active` (pIC50 ≥ 6.3) and Murcko `scaffold` | produced by `notebooks/04_exploratory_data_analysis.ipynb` | data ChEMBL CC BY-SA 3.0 |
| `hERG_chembl_walters.csv` | 4042 | hERG channel pIC50 | [PatWalters/practical_cheminformatics_tutorials](https://github.com/PatWalters/practical_cheminformatics_tutorials/blob/main/data/hERG.csv) | MIT (file); data ChEMBL |
| `chembl_drugs_walters.smi` | 1203 | approved drugs (SMILES + ChEMBL ID) | [PatWalters/datafiles](https://github.com/PatWalters/datafiles) | data ChEMBL CC BY-SA 3.0 |
| `zinc_50k.csv` | 50000 | drug-like ZINC molecules with logP, QED, SA score | random subset (SMILES ≤ 80 chars, canonicalised, deduplicated) of the ZINC-250k file in [chemical_vae](https://github.com/aspuru-guzik-group/chemical_vae) (Gómez-Bombarelli *et al.*, *ACS Cent. Sci.* 2018) | Apache-2.0 |
| `md/alanine_dipeptide_solvated.pdb` | 2269 atoms | alanine dipeptide + 749 TIP3P waters, periodic box | [OpenMM](https://github.com/openmm/openmm) test systems | MIT |
| `md/villin_headpiece_solvated.pdb` | 8867 atoms | villin headpiece HP35 in explicit water | [OpenMM cookbook](https://github.com/openmm/openmm-cookbook) tutorials | MIT |
| `pdb/4WKQ.pdb` | 2331 protein + 172 hetero atoms | EGFR kinase domain (694–1020) with gefitinib (IRE), NA, MES, 121 waters; 1.85 Å. Header, SEQRES, REMARK 465 (missing residues), CRYST1, ATOM/HETATM, CONECT kept; ANISOU and other REMARK records removed to keep the file small. Offline fallback for sessions 09 and 11 | [RCSB PDB 4WKQ](https://www.rcsb.org/structure/4WKQ), Yosaatmadja, Squire, McKeage & Flanagan (deposited 2014, to be published) | CC0 |
| `md/gaff-2.11.xml` | 935 bond, 5312 angle, 980 torsion entries | General AMBER Force Field 2.11 in OpenMM ffxml form, used to parametrise the ligand in session 11 | [openmmforcefields](https://github.com/openmm/openmmforcefields) `openmmforcefields/ffxml/amber/gaff/ffxml/gaff-2.11.xml` (converted from AmberTools 24.8 `gaff-2.11.dat`) | MIT (conversion); Wang *et al.* 2004 |
| `md/egfr_gefitinib_complex.pdb` | 5312 atoms | minimised EGFR (loops rebuilt by PDBFixer) + gefitinib, the topology of the trajectory below | produced by the session-11 code | CC BY 4.0 |
| `md/egfr_gefitinib_100ps.xtc` | 50 frames | 100 ps production MD of the complex (protein + ligand atoms only, every 2 ps): ff14SB + GAFF 2.11 with MMFF94 ligand charges, TIP3P, 0.15 M NaCl, rhombic dodecahedron (1 nm padding, 52 177 atoms), PME 1 nm, Langevin 300 K / MC barostat 1 bar, X–H constraints, hydrogen mass 1.5 amu. Protocol: 500 steps minimisation, 5 ps restrained NVT, 5 ps NPT, then 80 ps NPT at 4 fs (a first 4 fs production attempt that later became unstable — see session 11) used as further equilibration, then this 100 ps at 2 fs. Started from the crystal pose. | produced with the session-11 code (`md_production.py` + `md_restart.py`) on 2 CPU cores | CC BY 4.0 |
| `md/egfr_gefitinib_md_log.csv` | every 0.5 ps | OpenMM StateDataReporter log of the 100 ps production (temperature, density, energy) | idem | CC BY 4.0 |
| `md/gefitinib_docked_poses.sdf` | 9 poses | AutoDock Vina redocking poses of gefitinib into prepared 4WKQ (session 09, exhaustiveness 8, seed 42) with the `vina_score` property | produced by the session-09 code | CC BY 4.0 |

## Reproducing the derived files

```bash
# EGFR_curated.csv - run the session 04 notebook, or the equivalent short script:
python - <<'PY'
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
# ... see notebooks/04_exploratory_data_analysis.ipynb, sections 1 and 4
PY

# zinc_50k.csv - subsample of the original 250k file (see notebooks/08 credits)
```
