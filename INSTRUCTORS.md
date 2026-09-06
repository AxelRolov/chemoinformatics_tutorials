# Instructor notes

Practical notes for running these nine sessions as a course (~2 h per session works well, in person or hybrid).
The material is a compilation of other people's open teaching resources — see [`CREDITS.md`](CREDITS.md) — updated
and revised with Claude; please pass the credit on to the original authors when you use it.

## Mapping to typical course objectives

If your syllabus is written as learning outcomes, this is roughly how the sessions map onto the outcomes an
introductory chemoinformatics course usually states.

| Learning objective | Sessions |
|---|---|
| define the scope, objectives and applications of chemoinformatics | 00 intro, 01, and the closing discussion of 09 |
| identify and use major chemical data resources and databases | **03** |
| explain and compare methods for representing chemical structures digitally | **01, 02** |
| understand molecular-modeling principles and their relevance to representation, property prediction and design | **09** (+ 3D shape in 02, 3D GNNs mentioned in 06) |
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
| 04 | 120 min (standardisation ≈ 100 s, UMAP ≈ 30 s) | no | no | 03 |
| 05 | 120 min (all models ≈ 5 min total) | no | no | 04 |
| 06 | 90 min (MLP + GCN ≈ 1 min GPU, 1 min CPU each) | **yes** | no | 05 |
| 07 | 90 min (RL ≈ 3 min, GA ≈ 2 min) | **yes** | no | 02, 06 |
| 08 | 90 min | no | **yes** (free Gemini key) | 05 |
| 09 | 90 min (MD cell ≈ 2 min CPU / 10 s GPU) | **yes** | no | 01–02 |

Sessions 04 → 05 → 08 form a chain through the EGFR dataset; `data/EGFR_curated.csv` is committed so a student who
missed session 04 can still do 05 and 08.

## Practical set-up

- **Before session 08**, ask students to create a free Gemini API key (<https://aistudio.google.com/apikey>) and add it as
  a Colab secret named `GEMINI_API_KEY`. Doing this live costs 15 minutes. If your institution prefers another provider,
  change one line: `MODEL_ID` in the notebook (LiteLLM handles OpenAI, Anthropic, Mistral, Azure, Ollama…).
- **Colab quotas**: free GPU access is not guaranteed. Every notebook falls back to CPU; sessions 06, 07 and 09 detect the
  GPU and shrink the workload automatically (`GPU`/`device` variables in the setup cells).
- **Offline resilience**: session 03 checks whether PubChem/ChEMBL/PDB/Hugging Face answer, and uses the cached datasets
  in `data/` when they don't — so a firewalled classroom can still run it.
- **Version drift**: notebooks pin nothing on purpose, so they follow Colab's stack. If something breaks after a Colab
  update, run `python src_nb/build.py <nn> --execute` locally to see the error, fix `src_nb/*.py`, and rebuild.

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
  below 0.5), why a genetic algorithm beats neural models on GuacaMol, and what "novel" should mean for a generated molecule.

## Common student difficulties

1. **Indentation and cell order** (session 00). Emphasise that the kernel remembers the order cells were *run*, not their position;
   *Runtime → Restart and run all* is the fix for confusing states.
2. **`None` from `MolFromSmiles`** (01). The habit of checking for `None` before using a molecule prevents most later crashes.
3. **Confusing IC50 with pIC50** (03). Insist on the log scale and the direction (higher pIC50 = more potent).
4. **Believing a good random-split metric** (05). The random vs. scaffold split table is the single most important slide of the course.
5. **Reading generated molecules uncritically** (07). Always ask "could a chemist make this?" — the SA-score box plot answers it.
6. **Trusting the agent's prose** (08). The benchmark section is there to make the distinction between the LLM's words and the tools' numbers concrete.

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

Conventions used in the sources:
- markdown cells are triple-quoted strings after `# %% [markdown]`;
- the first markdown cell carries the title, learning goals and the credits block (the build script prepends the Colab badge);
- every notebook starts with a `# @title ⚙️ Setup` cell that pip-installs only when `IN_COLAB`;
- data is loaded through a `data_path()` / `fetch()` helper so notebooks work both inside a clone and standalone on Colab;
- exercises are `# YOUR CODE HERE` cells followed by a `<details><summary><b>Solution</b></summary>` markdown cell.

If you fork the repository, update `REPO` in `src_nb/build.py` **and** `REPO_RAW` in each setup cell.
