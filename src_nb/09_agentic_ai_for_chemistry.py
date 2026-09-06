# %% [markdown]
"""
# 09 · Agentic AI for chemistry: LLMs that use tools

**Chemoinformatics (UFAZ, L2 S3) — Practical session 9**
Instructor: Alexey Orlov (Université de Strasbourg / UFAZ)

**Learning goals.** After this session you will be able to
- explain what a large language model (LLM) is, what it *cannot* do alone in chemistry (hallucinated SMILES, arithmetic, current data);
- describe the **agent loop**: think → choose a tool → observe the result → repeat → answer;
- write **chemistry tools** (RDKit descriptors, PubChem lookup, similarity search, a trained QSAR model) and give them to an agent;
- run a **single agent** and a small **multi-agent** system on real questions, and read its trace critically;
- evaluate agents (does the answer match a ground truth computed by RDKit?) and discuss safety, cost and reproducibility.

> 🔑 **You need a free API key.** Get one at <https://aistudio.google.com/apikey> (Google account, no credit card).
> In Colab, click the 🔑 key icon in the left sidebar → *Add new secret* → name it `GEMINI_API_KEY`, paste the key,
> and enable *Notebook access*. The code is written with **LiteLLM**, so a single line switches to OpenAI, Anthropic,
> Mistral or a local model.

---
> **Credits.** Adapted, with modifications for the UFAZ course, from
> - *Tutorial: LLM agents for chemistry* by **He Seng / hesengg** ([GitHub](https://github.com/hesengg/Tutorial_LLM_Agent_Chemistry)) — the multi-agent structure and the RDKit/tool design;
> - **smolagents** by Hugging Face ([GitHub](https://github.com/huggingface/smolagents), Apache-2.0) — the agent framework and its `CodeAgent` idea;
> - **ChemCrow** (Bran *et al.*, *Nat. Mach. Intell.* 2024) and **Coscientist** (Boiko *et al.*, *Nature* 2023) — the chemistry-agent concept;
> - `MauricioCafiero/CheMLAgent` and `hoon-ock/AgentD` (drug-discovery agents) for task inspiration;
> - our own sessions 01–08 for the tools (RDKit descriptors, similarity, the EGFR QSAR model).
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

We use **smolagents** with **LiteLLM**, which speaks to any provider through one interface. The default below is
Gemini's free tier; the commented lines show alternatives.
"""

# %%
# Read the API key: Colab secrets first, then an environment variable, then ask.
def get_api_key(name="GEMINI_API_KEY"):
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
os.environ["GEMINI_API_KEY"] = API_KEY or ""
HAVE_KEY = bool(API_KEY)
print("API key found:", HAVE_KEY)

# %%
MODEL_ID = "gemini/gemini-2.5-flash"     # free tier, fast, good at tool use
# MODEL_ID = "openai/gpt-4.1-mini"       # needs OPENAI_API_KEY
# MODEL_ID = "anthropic/claude-haiku-4-5"# needs ANTHROPIC_API_KEY
# MODEL_ID = "mistral/mistral-small-latest"

from smolagents import CodeAgent, ToolCallingAgent, LiteLLMModel, tool

model = LiteLLMModel(model_id=MODEL_ID, api_key=API_KEY, temperature=0.2) if HAVE_KEY else None

if HAVE_KEY:
    from smolagents.models import ChatMessage, MessageRole
    reply = model([{"role": "user", "content": [{"type": "text", "text": "In one sentence: what is a SMILES string?"}]}])
    print(reply.content)

# %% [markdown]
"""
### The hallucination problem, demonstrated

Let's ask the raw model for a molecular weight and a SMILES, and check both with RDKit.
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


# Quick check that the tools work on their own
print(get_smiles("aspirin"))
print(get_descriptors("CC(=O)Oc1ccccc1C(=O)O"))
print(tanimoto_similarity("CC(=O)Oc1ccccc1C(=O)O", "OC(=O)c1ccccc1O"))
print(substructure_search("CC(=O)Oc1ccccc1C(=O)O", "[CX3](=O)[OX2H1]"))

# %% [markdown]
"""
### A tool that wraps *our own* model

Sessions 05–06 produced a curated EGFR dataset and a QSAR model. Wrapping it as a tool lets the agent use in-house
knowledge that no LLM has seen — this is where agents become useful in a real lab.
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

print(predict_egfr_activity("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"))   # gefitinib
print(predict_egfr_activity("CC(=O)Oc1ccccc1C(=O)O")[:120])                       # aspirin

# %% [markdown]
"""
> Gefitinib comes back with a probability *below* 0.5 even though it is an approved EGFR inhibitor. This is not a bug in
> the agent: it is the label noise we diagnosed in session 06 (the cached dataset kept one IC50 record per compound, and
> gefitinib's happens to be 515 nM, just under our 6.3 threshold). A good agent reports what the tool says; a good chemist
> knows what the tool was trained on. Keep this in mind when reading the agent's answers below.
"""

# %%

# %% [markdown]
"""
## 4. A first agent

`ToolCallingAgent` asks the model to emit structured tool calls. Watch the trace: each step shows what the model decided
and what it observed. This transparency is the point — you can audit every number in the final answer.
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

if HAVE_KEY:
    agent = ToolCallingAgent(tools=TOOLS, model=model, instructions=CHEM_INSTRUCTIONS, max_steps=8)
    result = agent.run("What is the molecular weight and logP of caffeine, and does it satisfy Lipinski's rule of five?")
    print("\nFINAL ANSWER:\n", result)

# %%
if HAVE_KEY:
    result = agent.run(
        "Compare gefitinib and erlotinib: how similar are they (Tanimoto), and do they both contain a quinazoline ring "
        "(SMARTS: c1ccc2ncncc2c1)? Report the numbers."
    )
    print("\nFINAL ANSWER:\n", result)

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

def ground_truth(entry):
    """Ground truth from RDKit. Uses PubChem for the structure, falling back to the reference SMILES above."""
    smi = get_smiles(entry["name"])
    if smi.startswith("ERROR"):
        smi = REFERENCE_SMILES[entry["name"]]
    return json.loads(get_descriptors(smi))[entry["key"]]

truth = {e["name"]: ground_truth(e) for e in BENCHMARK}
print(truth)

# %%
def extract_number(text):
    import re
    m = re.findall(r"-?\d+\.?\d*", str(text).replace(",", ""))
    return float(m[0]) if m else None

if HAVE_KEY:
    rows = []
    for e in BENCHMARK:
        gt = truth[e["name"]]
        tol = max(0.05 * abs(gt), 0.11)                          # 5 % tolerance (integers must match exactly)
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

1. **Add a tool** `sa_score(smiles)` returning the synthetic accessibility score (session 08) and ask the agent to rank the
   molecules generated in session 08 by "interesting *and* makeable".
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

**This was the last session. Congratulations!** You have gone from `print("Hello")` to agents that design and evaluate molecules.
Everything you built is in this repository — reuse it for your projects, and keep the habit that runs through all nine notebooks:
*look at your data, question your model, and verify what the machine tells you.*
"""
