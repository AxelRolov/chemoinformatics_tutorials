# Instructor notes

Practical notes for running these eleven sessions as a course (~2 h per session works well, in person or hybrid).
The material is a compilation of other people's open teaching resources — see [`CREDITS.md`](CREDITS.md) — updated
and revised with Claude; please pass the credit on to the original authors when you use it.

## Mapping to typical course objectives

If your syllabus is written as learning outcomes, this is roughly how the sessions map onto the outcomes an
introductory chemoinformatics course usually states.

| Learning objective | Sessions |
|---|---|
| define the scope, objectives and applications of chemoinformatics | 00 intro, 01, and the closing discussions of 10 and 11 |
| identify and use major chemical data resources and databases | **03** |
| explain and compare methods for representing chemical structures digitally | **01, 02** |
| understand molecular-modeling principles and their relevance to representation, property prediction and design | **09, 10, 11** (+ 3D shape in 02, 3D GNNs mentioned in 06) |
| prepare and curate chemical datasets | **04** (+ the curation part of 03) |
| construct and interpret basic QSPR/QSAR models | **05** |
| describe the main machine-learning approaches used in chemoinformatics | **05, 06** |
| apply validation strategies and performance metrics | **05** §3.3, §4.1–4.3 (scaffold split, y-randomisation) |
| explain deep learning and generative AI in chemical contexts | **06, 07** |
| discuss the potential and limitations of LLMs and agentic AI | **08** |
| critically assess the reliability of AI-based approaches | throughout: 04 (data quality), 05 (applicability domain, label noise), 07 (synthesisability, reward hacking), 08 (hallucination, agent benchmarking) |

## Timing and prerequisites

| # | ~time on Colab (CPU) | GPU useful? | needs a key | depends on |
|---|---|---|---|---|
| 00 | 60–90 min | no | no | — |
| 01 | 90 min | no | no | 00 |
| 02 | 90–120 min | no | no | 01 |
| 03 | 90 min | no | no | 00–01 |
| 04 | 100 min (standardisation ≈ 30 s, UMAP ≈ 20 s) | no | no | 03 |
| 05 | 120 min (all models ≈ 5 min total) | no | no | 04 |
| 06 | 90 min (MLP + GCN ≈ 1 min GPU, 1 min CPU each) | **yes** | no | 05 |
| 07 | 90 min (RL ≈ 3 min, GA ≈ 2 min) | **yes** | no | 02, 06 |
| 08 | 90 min | no | **yes** (DeepSeek key) | 05 |
| 09 | 100 min (Vina: redocking ≈ 30 s, seed table ≈ 1 min, mini screen ≈ 4 min — CPU only) | no | no | 02, 03 |
| 10 | 100 min (box building instant; MD ≈ 1.5 min CPU / 10 s GPU; the movies and widgets need a live Colab kernel) | **yes** | no | 01–02 |
| 11 | 120 min (loop building 1–2 min, vacuum relaxation 20 s; GPU: ≈ 8 min of simulation; CPU: ≈ 7 min token run + precomputed trajectory) | **yes** | no | 09, 10 |

Sessions 04 → 05 → 08 form a chain through the EGFR dataset; `data/EGFR_curated.csv` is committed so a student who
missed session 04 can still do 05 and 08 (session 09 also draws its screening set from it). Sessions 09 → 10 → 11
form the structural thread: 09 docks gefitinib into EGFR and saves `gefitinib_docked_poses.sdf`, 10 introduces
molecular dynamics on a peptide, 11 simulates the docked complex (a copy of the poses is in `data/md/`, so 11 also
works on its own). The order is deliberate: docking first, because it needs only a structure and a scoring
function; MD second, because it needs the force-field and integrator concepts that 10 builds up on a small system.

## Practical set-up

- **Before session 08**, sort out API keys. The notebook is set up for **DeepSeek**
  (<https://platform.deepseek.com/api_keys>, secret name `DEEPSEEK_API_KEY`, model `deepseek/deepseek-v4-flash`).
  DeepSeek is a paid API with no free tier, so decide in advance whether you hand out one departmental key or ask
  students to create their own; either way the notebook's usage is a handful of short requests per run. Doing the
  key set-up live costs about 15 minutes. Changing provider is one line — `MODEL_ID` plus the secret name in
  `get_api_key()` — and LiteLLM also handles Gemini (free tier, no card), OpenAI, Anthropic, Mistral, Azure and Ollama.
  If a smaller model produces malformed tool calls, switch that exercise to the `CodeAgent`, which does not rely on
  the provider's structured tool-calling API.
- **Colab quotas**: free GPU access is not guaranteed. Every notebook falls back to CPU; sessions 06, 07, 10 and 11 detect the
  GPU and shrink the workload automatically (`GPU`/`device` variables in the setup cells). Session 11 on CPU runs a
  token 2 ps simulation so that every cell executes, then analyses `data/md/egfr_gefitinib_100ps.xtc` (produced with
  the same code); with a GPU it analyses its own 100 ps run.
- **Sessions 09 and 11 need the RCSB PDB** for one download (entry 4WKQ). If it is unreachable they use `data/pdb/4WKQ.pdb`.
- **Session 11's ligand parametrisation is deliberately pip-only**: GAFF atom types from Open Babel, MMFF94 charges from
  RDKit, and a hand-written residue template on top of `gaff-2.11.xml`. The standard route (TeachOpenCADD T019:
  `openmmforcefields` + OpenFF toolkit + AmberTools for AM1-BCC charges) needs conda, which Colab does not have without a
  kernel restart. Say in class that MMFF94 charges are the compromise (Exercise 3 quantifies it), and that production
  work uses AM1-BCC/RESP or OpenFF.
- **Offline resilience**: session 03 checks whether PubChem/ChEMBL/PDB/Hugging Face answer, and uses the cached datasets
  in `data/` when they don't — so a firewalled classroom can still run it.
- **Session 04 runs on a subset.** Standardising all 5568 EGFR records costs about two minutes, almost all of it in the
  tautomer canonicalisation (~25 ms per molecule; capping `SetMaxTautomers` lower does not help much). The loading cell
  therefore takes `N_SAMPLE = 1500` records — every salt and every charged entry, plus a random fill — which brings the
  standardisation cell down to ~30 s while keeping 18 InChIKey duplicates and 2 contradictory-replicate structures for
  the deduplication and aggregation demos. Set `N_SAMPLE = None` for the full run. Note that the subset is enriched in
  messy structures (29 % against 8 %), so section 1's proportions are not representative of ChEMBL — say so in class.
  The `data/EGFR_curated.csv` committed here is still the **full** 5511-compound curation, so sessions 05, 07 and 08 are
  unaffected.
- **Version drift**: notebooks pin nothing on purpose, so they follow Colab's stack. If something breaks after a Colab
  update, run `python src_nb/build.py <nn> --execute` locally to see the error, fix `src_nb/*.py`, and rebuild.
- **Session 09 compiles AutoDock Vina on Colab.** PyPI has `vina` wheels only up to Python 3.12 and Colab now runs a
  newer Python, so the setup cell installs the rest of the stack first (pip installs all-or-nothing — with one combined
  `pip install`, Vina's failure took RDKit down with it), then tries `pip install --only-binary=:all: vina` and, when no
  wheel exists, installs SWIG and the Boost headers with `apt-get` and builds Vina from source (≈2 min on the 2 cores of
  a free runtime; ≈3 min for the whole cell). The moment the Vina maintainers publish wheels for Colab's Python the fast
  path takes over by itself. Locally, `pip install vina` needs the same `swig` + `libboost-*-dev` packages on Python ≥ 3.13.

## Suggested assessment

- **Weekly**: the exercises at the end of each notebook (solutions are in collapsible `<details>` blocks — tell students
  to try first). Ask for the completed notebook.
- **Mid-course mini-project** (after 05): pick a target or an endpoint from ChEMBL or TDC that is *not* EGFR, run the whole
  pipeline 03 → 04 → 05, and report: dataset provenance and curation decisions, scaffold-split performance vs. a baseline,
  applicability-domain analysis, and three molecules the model likes with a critical comment. Session 04's hERG exercises
  are a good starting point.
- **Final project** (after 08): either (a) goal-directed design — define a multi-parameter objective, run the RL and the GA
  of session 07, filter for PAINS/SA/Ro5, and defend 5 proposed molecules; or (b) build a chemistry agent with three new
  tools and a 10-question benchmark showing that it beats the bare LLM.
- **Discussion topics** that work well as short oral presentations: the label-noise case in 05 §6 (why gefitinib scores
  below 0.5), why a genetic algorithm beats neural models on GuacaMol, what "novel" should mean for a generated molecule,
  why the redocking in 09 "fails" the 2 Å rule with the core in place (and whether the rule is the right one), and what a
  100 ps trajectory can and cannot say about binding (11).

## Common student difficulties

1. **Indentation and cell order** (session 00). Emphasise that the kernel remembers the order cells were *run*, not their position;
   *Runtime → Restart and run all* is the fix for confusing states.
2. **`None` from `MolFromSmiles`** (01). The habit of checking for `None` before using a molecule prevents most later crashes.
3. **Confusing IC50 with pIC50** (03). Insist on the log scale and the direction (higher pIC50 = more potent).
4. **Believing a good random-split metric** (05). The random vs. scaffold split table is the single most important slide of the course.
5. **Reading generated molecules uncritically** (07). Always ask "could a chemist make this?" — the SA-score box plot answers it.
6. **Trusting the agent's prose** (08). The benchmark section is there to make the distinction between the LLM's words and the tools' numbers concrete.
7. **Reading a docking score as an affinity** (09). The mini screen (ρ ≈ 0 with pIC50, ρ ≈ −0.8 with size) is the cure; have students predict the outcome before running it.
8. **Import order in 11**: Open Babel must be imported before other SWIG-based libraries (Vina); the setup cell does this, but a student who adds `from vina import Vina` at the top will crash Open Babel with `swig::stop_iteration`.

## Editing the material

Edit `src_nb/*.py` (jupytext percent format), never `notebooks/*.ipynb`:

```bash
pip install jupytext nbformat nbclient
python src_nb/build.py 04 --execute   # rebuild + run session 04, report errors
python src_nb/build.py                # rebuild all, outputs stripped
```

`build.py` is **reproducible on purpose**: rebuilding an unchanged source produces byte-identical
notebooks. nbformat 4.5+ writes a random per-cell `id` on every save, so the build overrides them with
deterministic ids derived from each cell's content (`assign_stable_ids`). Without that, the notebooks would
differ on every rebuild and the "in sync with `src_nb`" check in CI could never pass. Keep that property if
you change the build.

`build.py --execute` runs the notebooks **headless** (no front-end). Sessions 10 and 11 use `ipywidgets.interact`
for their interactive plots and viewers, and a matplotlib figure drawn inside an `interact` callback makes nbclient
wait for its whole timeout even though the kernel has finished. So `execute()` injects a first cell into the
*throwaway* executed copy that replaces `ipywidgets.interact` by a function calling the callback once with mid-range
arguments (`HEADLESS_PRELUDE`). The committed notebooks are untouched; in Colab the widgets behave normally. If you add
an `interact` to another notebook, nothing else is needed.

**Session 11 relaxes PDBFixer's loops before solvation** (its section 2, step 5). PDBFixer builds the three missing
loops of 4WKQ by a short stochastic simulation, and in about half of the runs the result contained a clash — a strained
proline, or a loop end placed on top of a crystal atom — that 100 minimisation steps could not remove, so the
restrained-NVT stage died with *"Particle coordinate is NaN"*. The notebook now (a) builds the loops with a fixed seed,
so every student gets the same system (~45 000 atoms, 13 200 waters), and (b) minimises the protein alone in vacuum
with the crystal heavy atoms restrained before adding water. That takes 20 s and has made the CPU protocol reliable in
every test run since. Keep both if you touch the preparation.

Conventions used in the sources:
- markdown cells are triple-quoted strings after `# %% [markdown]`;
- the first markdown cell carries the title, learning goals and the credits block (the build script prepends the Colab badge);
- every notebook starts with a `# @title ⚙️ Setup` cell that pip-installs only when `IN_COLAB`;
- data is loaded through a `data_path()` / `fetch()` helper so notebooks work both inside a clone and standalone on Colab;
- exercises are `# YOUR CODE HERE` cells followed by a `<details><summary><b>Solution</b></summary>` markdown cell.

If you fork the repository, update `REPO` in `src_nb/build.py` **and** `REPO_RAW` in each setup cell.
