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
