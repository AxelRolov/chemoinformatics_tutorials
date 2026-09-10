# %% [markdown]
"""
# 08 · Agentic AI for chemistry: LLMs that use tools

**Chemoinformatics practicals — Session 8 of 11**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

**Learning goals.** After this session you will be able to
- explain what a large language model (LLM) is, what it *cannot* do alone in chemistry (hallucinated SMILES, arithmetic, current data);
- describe the **agent loop**: think → choose a tool → observe the result → repeat → answer;
- write **chemistry tools** (RDKit descriptors, PubChem lookup, similarity search, a trained QSAR model) and give them to an agent;
- run a **single agent** and a small **multi-agent** system on real questions, and read its trace critically;
- evaluate agents (does the answer match a ground truth computed by RDKit?) and discuss safety, cost and reproducibility.

> 🔑 **You need an API key.** This notebook is set up for **DeepSeek**. Create a key at
> <https://platform.deepseek.com/api_keys>, then in Colab click the 🔑 key icon in the left sidebar →
> *Add new secret* → name it `DEEPSEEK_API_KEY`, paste the key, and enable *Notebook access*.
> Your instructor may give you a key instead — in that case just paste it when the notebook asks.
>
> DeepSeek is a **paid** API (there is no free tier), but it is inexpensive and one pass through this notebook is
> a handful of short requests. The code goes through **LiteLLM**, so switching provider is a one-line change:
> the commented alternatives in section 2 include Google Gemini, whose free tier needs no card.

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - *Tutorial: LLM agents for chemistry* by **He Seng / hesengg** ([GitHub](https://github.com/hesengg/Tutorial_LLM_Agent_Chemistry)) — the multi-agent structure and the RDKit/tool design;
> - **smolagents** by Hugging Face ([GitHub](https://github.com/huggingface/smolagents), Apache-2.0) — the agent framework and its `CodeAgent` idea;
> - **ChemCrow** (Bran *et al.*, *Nat. Mach. Intell.* 2024) and **Coscientist** (Boiko *et al.*, *Nature* 2023) — the chemistry-agent concept;
> - `MauricioCafiero/CheMLAgent` and `hoon-ock/AgentD` (drug-discovery agents) for task inspiration;
> - our own sessions 01–07 for the tools (RDKit descriptors, similarity, the EGFR QSAR model).
"""

# %%
# @title ⚙️ Setup — run this cell first
import sys, os, subprocess, json, time, textwrap, warnings
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "smolagents[litellm]", "scikit-learn", "pubchempy"], check=False)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, Descriptors, QED, rdFingerprintGenerator, AllChem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
RDLogger.DisableLog("rdApp.*")

REPO_RAW = "https://raw.githubusercontent.com/AxelRolov/chemoinformatics_tutorials/main"
def data_path(filename):
    local = os.path.join("..", "data", filename)
    return local if os.path.exists(local) else f"{REPO_RAW}/data/{filename}"

os.makedirs("agent_outputs", exist_ok=True)

# %% [markdown]
"""
## 1. Why chemistry needs agents, not just chatbots

An LLM predicts the next token of text. It has read a lot of chemistry, so it can *sound* right — but on its own it:

- **hallucinates structures**: ask for "the SMILES of compound X" and you may get a plausible-looking but wrong string;
- **cannot count**: molecular formulas, ring counts and molecular weights come out approximately right, which is worse than wrong;
- **has a knowledge cutoff**: no access to the paper published last week, or to your in-house data;
- **cannot run experiments or code** — unless we give it the means.

An **agent** is an LLM inside a loop with **tools**:

```
 user question
      │
      ▼
 ┌─────────────┐   "I need the SMILES of aspirin"
 │     LLM     │ ─────────────────────────────────▶  tool: get_smiles("aspirin")
 │  (reasons)  │ ◀─────────────────────────────────  observation: "CC(=O)Oc1ccccc1C(=O)O"
 └─────────────┘   ... repeat until it can answer
      │
      ▼
   final answer
```

The LLM decides *which* tool to call and with *which arguments*; the tool does the exact work (RDKit, a database query,
a trained model). The LLM becomes an *orchestrator* of reliable components — this is what ChemCrow (2024) demonstrated for
chemistry, and what agents like Coscientist extended to running real experiments.

Two agent styles:
- **tool-calling agents**: the model emits a structured call `{"name": "get_descriptors", "arguments": {...}}`;
- **code agents** (smolagents' speciality): the model writes a *Python snippet* that calls the tools — more expressive
  (loops, pandas, chaining) but must be executed in a sandbox with a restricted import list.
"""

# %% [markdown]
"""
## 2. Connecting to a model

We use **smolagents** with **LiteLLM**, which speaks to any provider through one interface: you name the model as
`provider/model`, LiteLLM finds the matching key in the environment and translates the request. The default below
is **DeepSeek**, whose API is OpenAI-compatible (`https://api.deepseek.com`); the commented lines show alternatives,
and nothing else in the notebook changes when you switch.
"""

# %% [markdown]
"""
**Finding your key, three ways.** `get_api_key` tries Colab Secrets first (the 🔑 icon in the left sidebar). That is
the right place for it: the key lives in your Google account, not in the notebook, so it is not saved into the file
and not shared when you share the notebook. If that fails it falls back to an environment variable, for running
locally, and finally to `getpass`, which hides what you type and keeps the key only in memory for this session.

Whatever happens, `HAVE_KEY` records whether we got one, and every agent cell below is wrapped in `if HAVE_KEY:`.
So the notebook runs top to bottom without a key — you see the tools working and the agent sections quietly skipped.
"""

# %%
# Read the API key: Colab secrets first, then an environment variable, then ask.
def get_api_key(name="DEEPSEEK_API_KEY"):
    try:
        from google.colab import userdata
        key = userdata.get(name)
        if key: return key
    except Exception:
        pass
    key = os.environ.get(name)
    if not key:
        try:
            from getpass import getpass
            key = getpass(f"Paste your {name} (input hidden, nothing is stored): ")
        except Exception:
            key = None
    return key

API_KEY = get_api_key()
os.environ["DEEPSEEK_API_KEY"] = API_KEY or ""
HAVE_KEY = bool(API_KEY)
print("API key found:", HAVE_KEY)

# %% [markdown]
"""
**Choosing the model.** LiteLLM names models as `provider/model` and works out which environment variable holds the
key. `deepseek-v4-flash` is the cheap, fast one and the default for this course; `deepseek-v4-pro` reasons better and
is worth trying if the agent gets stuck in a loop. Uncomment one of the other lines to change provider — but change
the secret name in `get_api_key()` above as well, because each provider reads its own variable. DeepSeek is paid with
no free tier; Gemini's free tier needs no card, if you would rather not put one in.
"""

# %%
MODEL_ID = "deepseek/deepseek-v4-flash"    # fast and cheap - the default for this course
# MODEL_ID = "deepseek/deepseek-v4-pro"     # stronger reasoning; try it if the agent gets stuck
# MODEL_ID = "gemini/gemini-2.5-flash"      # needs GEMINI_API_KEY (free tier, no card)
# MODEL_ID = "openai/gpt-4.1-mini"          # needs OPENAI_API_KEY
# MODEL_ID = "anthropic/claude-haiku-4-5"   # needs ANTHROPIC_API_KEY
# MODEL_ID = "mistral/mistral-small-latest" # needs MISTRAL_API_KEY

# If you switch provider, also change the secret name in get_api_key() above - each provider reads its own
# environment variable (DEEPSEEK_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, ...).

# %% [markdown]
"""
**Four imports, and what each is for.** `LiteLLMModel` wraps the provider behind one interface; `tool` is the
decorator that turns an ordinary Python function into something an LLM can call; `ToolCallingAgent` and `CodeAgent`
are the two agent styles described at the end of section 1.

`temperature=0.2` is deliberate. For creative writing you want a high temperature; for an agent you want the most
likely next action almost every time, because an agent that improvises is an agent that loops. The
`... if HAVE_KEY else None` is what lets the rest of the notebook be read without a key.
"""

# %%
from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel, tool

model = LiteLLMModel(model_id=MODEL_ID, api_key=API_KEY, temperature=0.2) if HAVE_KEY else None

# %% [markdown]
"""
**A smoke test before anything complicated.** One plain chat request: no tools, no agent loop. If this prints a
sentence, then your key, your network access and your model name are all correct. If it raises, fix it *here*, where
there is only one thing that can be wrong — debugging a key problem through an agent's stack trace is miserable.

The nested message format (`content` as a list of typed parts) is the multimodal convention; a plain string also
works for text-only models.
"""

# %%
if HAVE_KEY:
    from smolagents.models import ChatMessage, MessageRole
    reply = model([{"role": "user", "content": [{"type": "text", "text": "In one sentence: what is a SMILES string?"}]}])
    print(reply.content)

# %% [markdown]
"""
### The hallucination problem, demonstrated

Let's ask the raw model for a molecular weight and a SMILES, and check both with RDKit.
"""

# %% [markdown]
"""
**The demonstration that motivates the whole notebook.** Two questions the model will answer with complete
confidence: give me a structure, give me a molecular weight. Then we check both with RDKit instead of by eye.

Read the checks carefully, because there is a subtlety. `Chem.MolFromSmiles` returning a molecule proves only that
the string is *parsable* — not that it is remdesivir. So we also print the molecular formula and compare it with
PubChem's reference (C27H35N6O8P, 602.6 g/mol). A hallucinated SMILES is usually perfectly valid chemistry; it is
just a different compound.

Run this cell two or three times. The failure is not reproducible, which is the worst property an error can have.
"""

# %%
QUESTIONS = ["What is the SMILES string of remdesivir? Answer with the SMILES only, no other text.",
             "What is the exact molecular weight of remdesivir in g/mol? Answer with a number only."]
if HAVE_KEY:
    answers = []
    for q in QUESTIONS:
        r = model([{"role": "user", "content": [{"type": "text", "text": q}]}]).content.strip()
        answers.append(r); print(q, "\n  ->", r[:120], "\n")

    smi = answers[0].split()[0].strip("`")
    mol = Chem.MolFromSmiles(smi)
    print("Is the SMILES parsable by RDKit?", mol is not None)
    if mol:
        print(f"Molecular weight computed by RDKit: {Descriptors.MolWt(mol):.2f}")
        print("Reference (PubChem CID 121304016): 602.6 g/mol, formula C27H35N6O8P")
        print("Formula of the generated SMILES:", Chem.rdMolDescriptors.CalcMolFormula(mol))
        display(mol)

# %% [markdown]
"""
Sometimes the model gets it right, sometimes it is subtly wrong (a stereocentre, a phosphate group) — and it never tells
you which. **Never trust an LLM-generated structure without verification.** That is precisely what tools are for.

## 3. Writing chemistry tools

A tool in smolagents is a plain Python function with the `@tool` decorator, type hints and a docstring — the docstring is
what the LLM reads to decide when and how to use it, so write it as if for a colleague.
"""

# %% [markdown]
"""
**Tool 1 · look up a structure.** The `@tool` decorator reads the function's signature, type hints and docstring and
turns them into the JSON schema the model sees. So the docstring is not documentation — **it is prompt**, and the
`Args:` section is required: without it smolagents refuses to build the tool. Write it as you would write an
instruction to a new colleague, including when *not* to use the tool ("instead of recalling SMILES from memory").

Two details in the body are worth copying into your own tools:

- The loop over `("SMILES", "IsomericSMILES", "CanonicalSMILES")` exists because PubChem's REST API renamed this
  property, and different endpoints answer to different names. Then the result is canonicalised through RDKit, so
  the agent always sees the same spelling for the same molecule.
- On failure the function **returns** a string starting with `ERROR:` instead of raising. An exception would kill
  the agent's loop; an error *message* is something the model can read, and react to, and report honestly.
"""

# %%
@tool
def get_smiles(name: str) -> str:
    """Look up the canonical SMILES of a compound by its name using PubChem. Use this instead of recalling SMILES from memory.

    Args:
        name: common or IUPAC name of the compound, e.g. "aspirin" or "gefitinib".
    """
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    for prop in ("SMILES", "IsomericSMILES", "CanonicalSMILES"):
        try:
            r = requests.get(f"{base}/{requests.utils.quote(name, safe='')}/property/{prop}/JSON", timeout=30)
            if r.status_code == 200:
                rec = r.json()["PropertyTable"]["Properties"][0]
                smiles = rec.get(prop)
                if smiles:
                    m = Chem.MolFromSmiles(smiles)
                    return Chem.MolToSmiles(m) if m else smiles
        except Exception:
            continue
    return f"ERROR: could not find '{name}' in PubChem"


# %% [markdown]
"""
**Tool 2 · compute properties.** One call returns everything the model needs for a Lipinski discussion, which saves
a round trip per property. Three choices to notice:

- It returns `json.dumps(...)`, a string — tool outputs are always text, and JSON is the format models parse most
  reliably.
- Everything is **rounded**. Hand a model twelve decimal places and it will faithfully copy them into an answer that
  claims a precision the method does not have.
- `lipinski_violations` is counted *here*, in Python, rather than left to the model. Counting is exactly what LLMs
  are worst at, so anything countable belongs on this side of the boundary.
"""

# %%
@tool
def get_descriptors(smiles: str) -> str:
    """Compute the standard molecular descriptors of a molecule with RDKit: molecular formula, molecular weight, logP,
    TPSA, hydrogen-bond donors and acceptors, rotatable bonds, rings, QED drug-likeness and Lipinski rule-of-five violations.

    Args:
        smiles: SMILES string of the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"ERROR: '{smiles}' is not a valid SMILES"
    from rdkit.Chem import rdMolDescriptors as rd
    d = {"formula": rd.CalcMolFormula(mol), "MW": round(Descriptors.MolWt(mol), 2),
         "logP": round(Descriptors.MolLogP(mol), 2), "TPSA": round(rd.CalcTPSA(mol), 1),
         "HBD": rd.CalcNumHBD(mol), "HBA": rd.CalcNumHBA(mol), "rotatable_bonds": rd.CalcNumRotatableBonds(mol),
         "rings": rd.CalcNumRings(mol), "aromatic_rings": rd.CalcNumAromaticRings(mol),
         "heavy_atoms": mol.GetNumHeavyAtoms(), "QED": round(QED.qed(mol), 3)}
    d["lipinski_violations"] = int(d["MW"] > 500) + int(d["logP"] > 5) + int(d["HBD"] > 5) + int(d["HBA"] > 10)
    return json.dumps(d)


# %% [markdown]
"""
**Tool 3 · compare two molecules.** The code is three lines from session 02. The interesting part is the last
sentence of the docstring: *"above 0.7 means very similar, below 0.3 means structurally different"*.

The model has no idea what a Tanimoto of 0.45 means. Putting the interpretation in the docstring is what turns a
reported number into an answer a student can use — and it is the cheapest way to inject domain knowledge into an
agent. When an agent misinterprets your tool's output, the fix is almost always in the docstring, not in the code.
"""

# %%
@tool
def tanimoto_similarity(smiles_1: str, smiles_2: str) -> str:
    """Compute the Tanimoto similarity between two molecules using Morgan (ECFP4) fingerprints. Returns a value in [0, 1];
    above 0.7 means very similar, below 0.3 means structurally different.

    Args:
        smiles_1: SMILES of the first molecule.
        smiles_2: SMILES of the second molecule.
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    m1, m2 = Chem.MolFromSmiles(smiles_1), Chem.MolFromSmiles(smiles_2)
    if m1 is None or m2 is None:
        return "ERROR: invalid SMILES"
    return f"{DataStructs.TanimotoSimilarity(gen.GetFingerprint(m1), gen.GetFingerprint(m2)):.3f}"


# %% [markdown]
"""
**Tool 4 · substructure search.** SMARTS from session 01, wrapped so the model can ask structural questions. It
returns both the number of matches and the first ten atom-index tuples, so the answer can say *where* the match is —
capped at ten, because a tool that dumps a thousand tuples floods the context window and costs money.

Note that the docstring carries two example patterns. Examples in a docstring act as few-shot prompts, and they
measurably improve how well a small model uses a tool with a fiddly argument like SMARTS.
"""

# %%
@tool
def substructure_search(smiles: str, smarts: str) -> str:
    """Check whether a molecule contains a substructure given as a SMARTS pattern, and return the number of matches.

    Args:
        smiles: SMILES of the molecule to search in.
        smarts: SMARTS pattern, e.g. "c1ccccc1" for a benzene ring or "[CX3](=O)[OX2H1]" for a carboxylic acid.
    """
    mol, patt = Chem.MolFromSmiles(smiles), Chem.MolFromSmarts(smarts)
    if mol is None or patt is None:
        return "ERROR: invalid SMILES or SMARTS"
    matches = mol.GetSubstructMatches(patt)
    return json.dumps({"matches": len(matches), "atom_indices": [list(m) for m in matches[:10]]})


# %% [markdown]
"""
**Test every tool without the agent first.** This is not optional ceremony. If a tool is broken you must find out
now, because through an agent you cannot tell a broken tool from a confused model, and you will spend an hour
rewriting prompts to fix a bug in `requests`.

Expect aspirin's formula `C9H8O4` and MW 180.16, a Tanimoto of 0.448 against salicylic acid — two molecules a chemist
would call close relatives, scoring under 0.5, which is worth remembering when you read similarity numbers — and one
carboxylic-acid match.

One line may disappoint you: `get_smiles("aspirin")` prints its `ERROR:` string whenever PubChem is unreachable
(it is blocked in some networks, including the sandbox these notebooks were built in). That is a useful accident:
it shows you the exact failure the agent will have to cope with. On Colab it returns the SMILES.
"""

# %%
# Quick check that the tools work on their own
print(get_smiles("aspirin"))
print(get_descriptors("CC(=O)Oc1ccccc1C(=O)O"))
print(tanimoto_similarity("CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1O"))
print(substructure_search("CC(=O)Oc1ccccc1C(=O)O", "[CX3](=O)[OX2H1]"))

# %% [markdown]
"""
### A tool that wraps *our own* model

Sessions 04–05 produced a curated EGFR dataset and a QSAR model. Wrapping it as a tool lets the agent use in-house
knowledge that no LLM has seen — this is where agents become useful in a real lab.
"""

# %% [markdown]
"""
**Training the in-house model.** This is all of session 05 in seven lines: load the curated EGFR set, fingerprint it,
fit a random forest on **all** 5511 compounds. No test split here on purpose — the model was already validated in
session 05; what we want now is the best available predictor. `_train_fps` is kept because the tool has to report how
far a query is from the training data.

The leading underscores are a convention marking these as the tool's internals rather than variables the notebook's
reader is meant to use.
"""

# %%
from sklearn.ensemble import RandomForestClassifier

_egfr = pd.read_csv(data_path("EGFR_curated.csv"))
_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_mols = [Chem.MolFromSmiles(s) for s in _egfr["smiles"]]
_X = np.array([_gen.GetFingerprintAsNumPy(m) for m in _mols])
_y = _egfr["active"].astype(int).values
_qsar = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=0).fit(_X, _y)
_train_fps = [_gen.GetFingerprint(m) for m in _mols]
print("QSAR model trained on", len(_egfr), "EGFR compounds")


# %% [markdown]
"""
**Tool 5 · the in-house QSAR model.** This is the tool no LLM can replace, and the reason agents earn their place in
a real lab: it was trained on *your* data, which the model has never seen and could not have memorised.

Look at what it returns — two numbers and a flag, not one number. The probability comes with the
`nearest_training_similarity` and a boolean `reliable`. That is the applicability domain of session 05 pushed inside
the tool's own output, so that the agent physically cannot receive a probability without also receiving the caveat.
The alternative — hoping the model remembers to ask about the domain — does not work. Design your tools so the
warning travels with the number.
"""

# %%
@tool
def predict_egfr_activity(smiles: str) -> str:
    """Predict whether a molecule is likely to inhibit the EGFR kinase, using an in-house random-forest QSAR model trained on
    5511 curated ChEMBL compounds (active = pIC50 >= 6.3). Returns the predicted probability of being active and the
    similarity to the closest training compound; if that similarity is below 0.4 the prediction is an extrapolation and
    should not be trusted.

    Args:
        smiles: SMILES string of the molecule to evaluate.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"ERROR: '{smiles}' is not a valid SMILES"
    p = float(_qsar.predict_proba(_gen.GetFingerprintAsNumPy(mol).reshape(1, -1))[0, 1])
    nn = max(DataStructs.BulkTanimotoSimilarity(_gen.GetFingerprint(mol), _train_fps))
    return json.dumps({"probability_active": round(p, 3), "nearest_training_similarity": round(nn, 3),
                       "reliable": bool(nn >= 0.4)})


# %% [markdown]
"""
**Tool 6 · show the neighbours.** A probability is an opinion; the five nearest known inhibitors with their measured
pIC50 values are evidence, and a chemist can judge evidence.

`top_k` is clamped with `max(1, min(int(top_k), 20))`. The model will eventually ask for 100, or for "five" as a
string, and defending a tool against its caller is ordinary practice rather than paranoia.
"""

# %%
@tool
def find_similar_actives(smiles: str, top_k: int = 5) -> str:
    """Find the most similar known EGFR inhibitors to a query molecule in the in-house ChEMBL dataset, with their measured pIC50.

    Args:
        smiles: SMILES of the query molecule.
        top_k: how many neighbours to return (1-20).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "ERROR: invalid SMILES"
    sims = np.array(DataStructs.BulkTanimotoSimilarity(_gen.GetFingerprint(mol), _train_fps))
    idx = np.argsort(-sims)[:max(1, min(int(top_k), 20))]
    return json.dumps([{"chembl_id": _egfr["chembl_id"].iloc[i], "smiles": _egfr["smiles"].iloc[i],
                        "pIC50": round(float(_egfr["pIC50"].iloc[i]), 2), "similarity": round(float(sims[i]), 3)}
                       for i in idx])

# %% [markdown]
"""
**Two sanity checks, with a lesson in each.** Gefitinib comes back at probability 0.357 with a nearest-training
similarity of exactly **1.0** — it *is* in the training set, and the model still puts it below 0.5. Aspirin comes
back at 0.031 with similarity 0.36, so `reliable` is false: the low probability is almost certainly right, but the
model had no business being asked, and the tool says so.
"""

# %%
print(predict_egfr_activity("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"))   # gefitinib
print(predict_egfr_activity("CC(=O)Oc1ccccc1C(=O)O")[:120])                       # aspirin

# %% [markdown]
"""
> Gefitinib comes back with a probability *below* 0.5 even though it is an approved EGFR inhibitor. This is not a bug in
> the agent: it is the label noise we diagnosed in session 05 (the cached dataset kept one IC50 record per compound, and
> gefitinib's happens to be 515 nM, just under our 6.3 threshold). A good agent reports what the tool says; a good chemist
> knows what the tool was trained on. Keep this in mind when reading the agent's answers below.
"""

# %% [markdown]
"""
## 4. A first agent

`ToolCallingAgent` asks the model to emit structured tool calls. Watch the trace: each step shows what the model decided
and what it observed. This transparency is the point — you can audit every number in the final answer.
"""

# %% [markdown]
"""
**The toolbox and the house rules.** Six tools, and a system prompt written almost entirely as *prohibitions*. That
is deliberate: "never write a SMILES string from memory" is a rule you can check and the model can follow, whereas
"be accurate" is a wish.

Read the five rules and notice that each one names a failure we have already produced in this notebook: the
hallucinated remdesivir, the estimated molecular weight, the PubChem outage, the gefitinib probability. Instructions
earn their place by fixing an observed failure — do not write them from imagination.
"""

# %%
TOOLS = [get_smiles, get_descriptors, tanimoto_similarity, substructure_search, predict_egfr_activity, find_similar_actives]

CHEM_INSTRUCTIONS = """You are a careful chemoinformatics assistant helping bachelor students.
Rules:
- NEVER write a SMILES string from memory: always obtain it with the get_smiles tool.
- NEVER estimate a molecular property yourself: always use get_descriptors or the other tools.
- If a tool returns an ERROR, say so and try a different approach; do not invent the answer.
- When you use predict_egfr_activity, always report the nearest_training_similarity and warn the user if reliable is false.
- Give short, precise answers, and state which tools produced which numbers."""

# %% [markdown]
"""
**The first agent run.** `max_steps=8` caps the loop; without a cap, a confused agent calls tools until your credit
runs out.

The question needs three things: a structure, its properties, and a rule applied to them. So watch the **trace**, not
the answer: you should see `get_smiles("caffeine")` and then `get_descriptors(...)` on the string that came back.
Reading traces is the real skill of this section. An answer that is right by a wrong route — a property recalled from
memory that happens to be correct — will fail on the next question, and only the trace tells you which you have.
"""

# %%
if HAVE_KEY:
    agent = ToolCallingAgent(tools=TOOLS, model=model, instructions=CHEM_INSTRUCTIONS, max_steps=8)
    result = agent.run("What is the molecular weight and logP of caffeine, and does it satisfy Lipinski's rule of five?")
    print("\nFINAL ANSWER:\n", result)

# %% [markdown]
"""
**A question that needs the same tool twice.** Two structure lookups, then two different tools on the results.

Notice that we hand the agent the SMARTS pattern in the question — asking a model to write SMARTS from memory is
precisely what this notebook tells you not to do.

The answer contains a useful surprise: the Tanimoto is about **0.41**, and both molecules contain the quinazoline.
Two drugs with the same core, the same target and the same clinical use score barely above the 0.3 that the tool's
own docstring calls "structurally different". Fingerprint similarity is a narrow question, not a verdict.
"""

# %%
if HAVE_KEY:
    result = agent.run(
        "Compare gefitinib and erlotinib: how similar are they (Tanimoto), and do they both contain a quinazoline ring "
        "(SMARTS: c1ccc2ncncc2c1)? Report the numbers."
    )
    print("\nFINAL ANSWER:\n", result)

# %% [markdown]
"""
**A question with a trap in it.** Imatinib is a kinase inhibitor, so the model's prose will *want* to say "likely
active" — but its real target is BCR-ABL, and the in-house model knows only EGFR chemistry.

Two things to check in the answer. Did the agent report the reliability flag, as its instructions require? And does
its closing sentence agree with its own numbers? Correct tool output with wrong narration is the single most common
agent failure in practice, and it is invisible unless you read both.
"""

# %%
if HAVE_KEY:
    result = agent.run(
        "I am considering testing the drug imatinib against EGFR. Use the in-house model to predict whether it is active, "
        "tell me how reliable that prediction is, and show me the three most similar compounds in the EGFR dataset with their pIC50."
    )
    print("\nFINAL ANSWER:\n", result)

# %% [markdown]
"""
### The code agent

A `CodeAgent` writes Python that calls the tools. It can loop over a list of molecules, sort results, use pandas — things
that would take many separate tool calls. The price: it *executes code*, so we restrict the allowed imports
(`additional_authorized_imports`) and never run it on untrusted input.

There is a second, practical reason to know this agent type. A `ToolCallingAgent` depends on the provider's
**structured tool-calling** API, and support for it varies in quality between models. A `CodeAgent` needs nothing
but a model that can write Python, so it works with essentially any chat model. If you switch to a smaller or
self-hosted model and the tool-calling agent starts producing malformed calls, try the code agent before blaming
your tools — or move up to `deepseek/deepseek-v4-pro`, which DeepSeek's own tool-calling documentation uses in its
examples.

**What this particular run tests.** Five molecules, three properties each, sorted, plus a rule applied. A
`ToolCallingAgent` would need ten round trips — `get_smiles` then `get_descriptors` for each molecule, one LLM call
apiece — and would then have to hold fifteen numbers in its context while sorting them by hand. The code agent writes
one loop and runs it, and pandas does the sorting exactly. Watch the Python it generates: that snippet is the honest, auditable record of
what it actually did, which is more than you get from most software.
"""

# %%
if HAVE_KEY:
    code_agent = CodeAgent(
        tools=TOOLS, model=model, instructions=CHEM_INSTRUCTIONS, max_steps=8,
        additional_authorized_imports=["json", "math", "statistics", "numpy", "pandas", "rdkit", "rdkit.Chem",
                                        "rdkit.Chem.Descriptors", "rdkit.Chem.rdMolDescriptors"],
    )
    result = code_agent.run(
        "For these five drugs — aspirin, ibuprofen, paracetamol, imatinib, atorvastatin — get the SMILES, compute MW, logP "
        "and QED, and give me a markdown table sorted by decreasing QED. Also say which ones violate Lipinski's rule of five."
    )
    print("\nFINAL ANSWER:\n", result)

# %% [markdown]
"""
## 5. A small multi-agent system

Real workflows split responsibilities: one agent knows the databases, another the modelling, a **manager** plans and
delegates. smolagents implements this by giving a manager agent other agents as "tools" (`managed_agents`).
"""

# %% [markdown]
"""
**Three agents: two specialists and a manager.** Each specialist gets only the tools it needs, plus a `name` and — the
part that matters — a `description`. The description is what the manager reads when it decides whom to delegate to;
it plays exactly the role a docstring plays for a tool. A vague description gives you a manager that guesses.

The manager is a `CodeAgent` with `tools=[]`: its only tools *are* the two experts, and it writes Python that calls
them. Splitting the toolbox this way keeps every individual decision simple, which is why a small multi-agent system
often beats one agent holding twenty tools. The price is real: each delegation is another full LLM conversation, so
this cell is the slowest and most expensive in the notebook.
"""

# %%
if HAVE_KEY:
    lookup_agent = ToolCallingAgent(
        tools=[get_smiles, get_descriptors, substructure_search], model=model, max_steps=6,
        name="lookup_expert",
        description="Finds compounds by name in PubChem, returns their SMILES, descriptors and substructure matches.",
        instructions="You retrieve chemical structures and compute descriptors. Always use the tools, never memory.")

    modelling_agent = ToolCallingAgent(
        tools=[predict_egfr_activity, find_similar_actives, tanimoto_similarity], model=model, max_steps=6,
        name="qsar_expert",
        description="Predicts EGFR activity with the in-house QSAR model and finds similar known inhibitors with their pIC50.",
        instructions="You evaluate molecules against EGFR. Always report the applicability-domain warning when reliable is false.")

    manager = CodeAgent(
        tools=[], model=model, managed_agents=[lookup_agent, modelling_agent], max_steps=10,
        instructions=("You coordinate two experts. Plan the task as a short list of steps, delegate each step to the right "
                      "expert, then summarise. Never guess chemical facts yourself. Finish with a concise report that states "
                      "which numbers came from which expert."),
        additional_authorized_imports=["json", "pandas", "numpy"])

    result = manager.run(
        "Triage three candidate molecules for an EGFR project: osimertinib, lapatinib and metformin. For each: get the SMILES, "
        "compute MW/logP/QED, predict EGFR activity with the in-house model (with its reliability), and find its closest known "
        "inhibitor. Then rank them and recommend which to test first, with one sentence of justification each."
    )
    print("\nFINAL ANSWER:\n", result)

# %% [markdown]
"""
## 6. Evaluating an agent

An agent's answer is only useful if it is *right*. Build a small benchmark whose ground truth we compute with RDKit,
and measure how often the agent gets it right. (This is a miniature of what ChemBench and ChemLLMBench do.)
"""

# %% [markdown]
"""
**The benchmark.** Five questions with numeric answers, chosen so that each probes a different weakness: a molecular
weight (arithmetic over a formula), two counts — aromatic rings and hydrogen-bond donors — (counting, the classic
failure), a Crippen logP (a *specific method* the model cannot reproduce, only approximate), and rotatable bonds (a
definition question, where RDKit's convention is the only sensible ground truth).

`REFERENCE_SMILES` is the offline fallback, so the benchmark still has a ground truth when PubChem is unreachable.
"""

# %%
# Reference SMILES (from PubChem) so the ground truth can be computed even if the network is unavailable
REFERENCE_SMILES = {
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
}

BENCHMARK = [
    {"question": "What is the molecular weight of caffeine? Answer with a number only.", "name": "caffeine", "key": "MW"},
    {"question": "How many aromatic rings does gefitinib have? Answer with a number only.", "name": "gefitinib", "key": "aromatic_rings"},
    {"question": "How many hydrogen-bond donors does atorvastatin have? Answer with a number only.", "name": "atorvastatin", "key": "HBD"},
    {"question": "What is the logP (Crippen) of ibuprofen? Answer with a number only.", "name": "ibuprofen", "key": "logP"},
    {"question": "How many rotatable bonds does imatinib have? Answer with a number only.", "name": "imatinib", "key": "rotatable_bonds"},
]

# %% [markdown]
"""
**Ground truth is computed, never typed in.** `ground_truth` runs our own tools — `get_smiles`, falling back to the
reference SMILES, then `get_descriptors` — so every answer comes from RDKit. The printed dictionary should read
caffeine 194.19, gefitinib 3, atorvastatin 4, ibuprofen 3.07, imatinib 7.

Be precise about what this measures. We are testing whether the agent agrees with **RDKit**, not whether it agrees
with nature: a different logP implementation would give a different "truth" for ibuprofen. Every benchmark measures
agreement with its reference, and choosing that reference honestly is the actual work of building one.
"""

# %%
def ground_truth(entry):
    """Ground truth from RDKit. Uses PubChem for the structure, falling back to the reference SMILES above."""
    smi = get_smiles(entry["name"])
    if smi.startswith("ERROR"):
        smi = REFERENCE_SMILES[entry["name"]]
    return json.loads(get_descriptors(smi))[entry["key"]]

truth = {e["name"]: ground_truth(e) for e in BENCHMARK}
print(truth)

# %% [markdown]
"""
**Grading free text.** Models answer in sentences even when told not to, so `extract_number` pulls the first number
out of the reply with a regex.

This is crude, and knowing *how* crude is part of the lesson: it would take the "5" out of "the rule of 5 is
satisfied". Every LLM benchmark you read in a paper has a function like this one somewhere, and its failures are
rarely reported. When you see a headline accuracy number, ask how the answers were parsed.
"""

# %%
def extract_number(text):
    import re
    m = re.findall(r"-?\d+\.?\d*", str(text).replace(",", ""))
    return float(m[0]) if m else None

# %% [markdown]
"""
**The experiment.** For each question we ask twice: the bare model, then a **fresh** agent with the tools — a new
agent per question, so that nothing leaks through its memory from the previous answer.

`tol = max(0.05 * abs(gt), 0.11)` is the grading tolerance: 5 % of the true value, with a small absolute floor so
that an answer near zero is not held to an impossible standard. Be clear-eyed about how lenient that is. `max` means
the floor can only ever *widen* the window, so for the aromatic-ring question (truth 3) the tolerance is 0.15 and an
answer of 3.1 would be scored correct — this code cannot force the integer answers to match exactly, and an honest
benchmark of counting ability would compare integers with `==`. Tighten it and see whether the bare model's score
changes. `verbosity_level=0` silences the traces so the table stays readable.

Expect the agent close to 100 % and the bare model well below it, with its errors concentrated in the counting
questions — the ones where sounding right and being right come apart.
"""

# %%
if HAVE_KEY:
    rows = []
    for e in BENCHMARK:
        gt = truth[e["name"]]
        tol = max(0.05 * abs(gt), 0.11)                          # 5 % tolerance, with a small absolute floor
        # (a) the bare LLM
        raw = model([{"role": "user", "content": [{"type": "text", "text": e["question"]}]}]).content
        raw_val = extract_number(raw)
        # (b) the agent with tools
        ag = ToolCallingAgent(tools=TOOLS, model=model, instructions=CHEM_INSTRUCTIONS, max_steps=6, verbosity_level=0)
        ag_val = extract_number(ag.run(e["question"]))
        rows.append({"question": e["name"] + " / " + e["key"], "truth": gt,
                     "LLM alone": raw_val, "LLM ok": raw_val is not None and abs(raw_val - gt) <= tol,
                     "agent": ag_val, "agent ok": ag_val is not None and abs(ag_val - gt) <= tol})
    bench = pd.DataFrame(rows)
    display(bench)
    print(f"\naccuracy — LLM alone: {bench['LLM ok'].mean():.0%}   agent with tools: {bench['agent ok'].mean():.0%}")

# %% [markdown]
"""
Run this a few times: the bare model's answers fluctuate (it is sampling text), while the agent's answers are stable
because the numbers come from RDKit. **That difference is the whole point of tool use.**

## 7. Limits, costs and good practice

| issue | what to do |
|---|---|
| **hallucination** | never let the LLM produce facts; only orchestrate tools. Verify structures with RDKit/PubChem |
| **error propagation** | a wrong SMILES early ruins everything after: validate tool outputs, add sanity checks |
| **code execution** | restrict imports, run in a sandbox (smolagents supports E2B/Docker executors), never on untrusted input |
| **cost and latency** | each step is an LLM call; cap `max_steps`, use small models for simple sub-tasks |
| **reproducibility** | agents are stochastic; log the full trace (`agent.memory.steps`), fix temperature, report the model version |
| **safety** | ChemCrow-style agents include a "safety tool" that refuses controlled/dangerous substances; an agent connected to lab robots needs human sign-off |
| **evaluation** | build task-specific benchmarks (like section 6); see ChemBench, ChemLLMBench, LAB-Bench |

### Where this is going
- **Literature agents**: PaperQA / paper-qa for grounded question answering over PDFs; askchem for searching findings.
- **Autonomous experimentation**: Coscientist (Boiko 2023) planned and executed a palladium-catalysed coupling on a robotic platform.
- **Specialised chemistry agents**: retrosynthesis (AiZynthFinder as a tool), spectra interpretation, ADMET triage (`AgentD`, `CheMLAgent`).
- **MCP (Model Context Protocol)** servers expose chemistry tools to any LLM client — e.g. ChemLint, ChEMBL and PubMed servers.

## 8. Exercises

1. **Add a tool** `sa_score(smiles)` returning the synthetic accessibility score (session 07) and ask the agent to rank the
   molecules generated in session 07 by "interesting *and* makeable".
2. **Add a scaffold tool** returning the Murcko scaffold, and ask the agent whether two compounds are scaffold hops of each other.
3. **Break the agent**: ask it something outside its tools ("what is the melting point of gefitinib?"). Does it admit it does not know,
   or invent a number? Improve the instructions so that it refuses cleanly.
4. **Reduce the tool set**: remove `get_smiles` and re-run the benchmark. How does accuracy change? This measures the value of each tool.
5. **Project**: connect a real MCP chemistry server (e.g. the ChEMBL MCP server) with `smolagents.MCPClient` and let the agent query live bioactivity data.
"""

# %%
# YOUR CODE HERE (exercises 1-4)


# %% [markdown]
"""
<details><summary><b>Solutions 1-2: two extra tools</b></summary>

```python
from rdkit.Chem import RDConfig
from rdkit.Chem.Scaffolds import MurckoScaffold
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

@tool
def sa_score(smiles: str) -> str:
    '''Estimate how hard a molecule is to synthesise (Ertl & Schuffenhauer synthetic accessibility score):
    1 means easy to make, 10 means very hard. Values above 6 usually indicate an impractical molecule.

    Args:
        smiles: SMILES string of the molecule.
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "ERROR: invalid SMILES"
    return f"{sascorer.calculateScore(mol):.2f}"

@tool
def murcko_scaffold(smiles: str) -> str:
    '''Return the Bemis-Murcko scaffold of a molecule (its ring systems plus the linkers between them) as SMILES.
    Two molecules with the same scaffold belong to the same chemical series; a "scaffold hop" keeps the activity
    while changing the scaffold.

    Args:
        smiles: SMILES string of the molecule.
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "ERROR: invalid SMILES"
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))

EXTENDED_TOOLS = TOOLS + [sa_score, murcko_scaffold]
agent2 = ToolCallingAgent(tools=EXTENDED_TOOLS, model=model, instructions=CHEM_INSTRUCTIONS, max_steps=10)
print(agent2.run("Are gefitinib and imatinib scaffold hops of each other? Compare their Murcko scaffolds and their "
                 "Tanimoto similarity, and say which one is easier to synthesise."))
```

Note that a tool's docstring is its interface: `@tool` parses the `Args:` section, so every argument must be
documented and type-hinted or smolagents refuses to build the tool.
</details>

<details><summary><b>Solutions 3-4: refusing gracefully, and measuring a tool's value</b></summary>

```python
# 3. Make refusal explicit in the instructions
STRICTER = CHEM_INSTRUCTIONS + (
    "\n- If no tool can answer the question (an experimental melting point, a boiling point, a price, a clinical "
    "outcome), reply exactly: \"I cannot answer this: none of my tools provides that information.\" "
    "Never estimate such a value.")
agent3 = ToolCallingAgent(tools=TOOLS, model=model, instructions=STRICTER, max_steps=6)
print(agent3.run("What is the melting point of gefitinib?"))

# 4. Remove the lookup tool and re-run the benchmark: accuracy collapses, because every other tool
#    depends on getting the right structure first.
reduced = [t for t in TOOLS if t.name != "get_smiles"]
ag = ToolCallingAgent(tools=reduced, model=model, instructions=CHEM_INSTRUCTIONS, max_steps=6, verbosity_level=0)
ok = 0
for e in BENCHMARK:
    v = extract_number(ag.run(e["question"]))
    gt = truth[e["name"]]
    ok += v is not None and abs(v - gt) <= max(0.05 * abs(gt), 0.11)
print(f"accuracy without get_smiles: {ok / len(BENCHMARK):.0%}")
```
</details>
"""
# %% [markdown]
"""
## Further reading
- Bran *et al.*, *Augmenting large language models with chemistry tools* (ChemCrow), Nat. Mach. Intell. **2024**, 6, 525.
- Boiko *et al.*, *Autonomous chemical research with large language models*, Nature **2023**, 624, 570.
- Mirza *et al.*, *A framework for evaluating the chemical knowledge and reasoning abilities of LLMs* (ChemBench), 2024.
- smolagents documentation: <https://huggingface.co/docs/smolagents>.
- M. Ramos, *A review of LLM agents for chemistry*, and the [awesome-chemistry-agents](https://github.com/topics/chemistry-agents) lists.

---

That closes the data-and-AI thread of the course: you have gone from `print("Hello")` to agents that design and
evaluate molecules, working throughout with **2D** representations of structure.

The last three sessions drop the 2D approximation: **09 · Protein–ligand docking** puts a drug into its protein,
**10 · Molecular modeling and molecular dynamics** gives molecules coordinates and watches them move, and **11 · MD of a
protein–ligand complex** tests whether the docked pose holds. It is the natural place to end, because it shows what the
representations of session 02 were leaving out, and where the next generation of models (3D and equivariant networks,
structure-based generation, co-folding) is heading.
"""
