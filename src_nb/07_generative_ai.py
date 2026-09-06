# %% [markdown]
"""
# 07 · Generative AI for molecules: designing new compounds

**Chemoinformatics practicals — Session 7 of 9**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

> ⚡ In Colab choose *Runtime → Change runtime type → T4 GPU*. A pre-trained model is provided, so nothing here takes more than a few minutes.

**Learning goals.** After this session you will be able to
- explain what "generative" means for molecules and how a model *samples* new structures;
- use a **SMILES language model** (character-level LSTM) to generate molecules and evaluate them with **validity, uniqueness, novelty** and property distributions;
- **bias** a generative model towards a design objective by **fine-tuning** (transfer learning) and by **reinforcement learning** (a simple REINVENT-style loop with a scoring function);
- perform **goal-directed design** with a genetic algorithm on molecular graphs (CReM-style mutations) as a strong, interpretable baseline;
- discuss what generative models cannot do: synthesisability, novelty ≠ usefulness, and how the field evaluates itself (GuacaMol, MOSES).

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - *AI for Chemistry* (EPFL CH-457), **Schwaller group** — `05 - Generative Models` and `06 - Generative Models 2 / SMILES-LSTM-Walkthrough` ([GitHub](https://github.com/schwallergroup/ai4chem_course), MIT);
> - *Practical Cheminformatics Tutorials* by **Pat Walters** — `generative/SMILES_RNN` ([GitHub](https://github.com/PatWalters/practical_cheminformatics_tutorials), MIT);
> - **TeachOpenCADD** talktorial **T034 · RNN-based molecular property prediction / SMILES generation** (Volkamer lab; [GitHub](https://github.com/volkamerlab/teachopencadd), CC BY 4.0);
> - the **REINVENT** family (MolecularAI; [GitHub](https://github.com/MolecularAI/REINVENT4), Apache-2.0) for the RL formulation, and **CReM** by P. Polishchuk ([GitHub](https://github.com/DrrDom/crem), BSD-3) for the mutation idea;
> - **GuacaMol** (BenevolentAI, MIT) and **MOSES** (molecularsets, MIT) for the evaluation metrics;
> - Training data: 50 000 molecules sampled from the ZINC-250k set used by Gómez-Bombarelli *et al.* (chemical_vae repository, Apache-2.0).
"""

# %%
# @title ⚙️ Setup — run this cell first
import sys, os, subprocess, time, re, math, random, warnings
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "selfies", "seaborn"], check=False)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, Descriptors, QED, rdFingerprintGenerator, AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Scaffolds import MurckoScaffold
IPythonConsole.ipython_useSVG = True
RDLogger.DisableLog("rdApp.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
torch.manual_seed(0); np.random.seed(0); random.seed(0)

REPO_RAW = "https://raw.githubusercontent.com/AxelRolov/chemoinformatics_tutorials/main"
def fetch(filename, subdir="data"):
    local = os.path.join("..", subdir, filename)
    if os.path.exists(local):
        return local
    import urllib.request
    target = os.path.basename(filename)
    if not os.path.exists(target):
        urllib.request.urlretrieve(f"{REPO_RAW}/{subdir}/{filename}", target)
    return target

# %% [markdown]
"""
## 1. What is a generative model?

A **discriminative** model (sessions 05–06) learns $p(y \mid x)$: given a molecule, predict a property.
A **generative** model learns $p(x)$: the distribution of molecules itself, so that we can **sample new ones**.

Main families in chemistry:

| family | representation | idea | examples |
|---|---|---|---|
| **language models** | SMILES / SELFIES strings | predict the next token; sample token by token | RNN/LSTM, GPT-style, REINVENT, MolGPT |
| **variational autoencoders (VAE)** | strings or graphs | encode into a continuous latent space, decode back; optimise *in* the latent space | ChemVAE, JT-VAE |
| **generative adversarial / flow / diffusion** | graphs, 3D point clouds | learn to denoise or transport noise into molecules | MolDiff, DiffSBDD, FlowMol |
| **graph-based / fragment-based** | molecular graph | add atoms/fragments step by step | GraphINVENT, MoLeR, CReM |
| **evolutionary algorithms** | graph | mutate and select — no neural network at all | GB-GA, CReM, genetic algorithms |

Two tasks to distinguish:
- **distribution learning**: generate molecules that look like the training set (valid, drug-like, diverse);
- **goal-directed design**: generate molecules that *optimise* an objective (predicted activity, QED, similarity to a target, multi-parameter score).
"""

# %% [markdown]
"""
## 2. A SMILES language model

A SMILES string is a sequence of tokens. The model reads the tokens seen so far and predicts a probability distribution
over the next token: $p(t_{i} \mid t_{1..i-1})$. Training maximises the likelihood of real SMILES (cross-entropy loss);
generation samples from the predicted distribution, one token at a time, from `<bos>` until `<eos>`.

The tokenizer must respect chemistry: `Cl` is one token, not `C` + `l`; `[nH]` is one token.
"""

# %%
SMILES_REGEX = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

def tokenize(smiles):
    return SMILES_REGEX.findall(smiles)

print(tokenize("CC(=O)Oc1ccccc1C(=O)O"))
print(tokenize("Clc1ccc([nH]2)cc1.[Na+]"))

# %%
class Vocab:
    def __init__(self, tokens):
        self.itos = SPECIAL + sorted(set(tokens) - set(SPECIAL))
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos, self.unk = 0, 1, 2, 3
    def __len__(self): return len(self.itos)
    def encode(self, smiles): return [self.bos] + [self.stoi.get(t, self.unk) for t in tokenize(smiles)] + [self.eos]
    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.eos: break
            if i not in (self.pad, self.bos): out.append(self.itos[i])
        return "".join(out)

zinc = pd.read_csv(fetch("zinc_50k.csv"))
print(zinc.shape); zinc.head(3)

# %%
class SmilesLSTM(nn.Module):
    """Character-level (token-level) language model over SMILES."""
    def __init__(self, vocab_size, emb=128, hidden=512, layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, vocab_size)
    def forward(self, x, state=None):
        h, state = self.lstm(self.embedding(x), state)
        return self.out(h), state

# %% [markdown]
"""
### Loading the pre-trained model
Training from scratch on 50 000 SMILES takes ~30 min on CPU, so the course ships a checkpoint
(`models/smiles_lstm_zinc50k.pt`, produced by `utils/pretrain_smiles_lstm.py` — read it: it is the same code as here).
"""

# %%
ckpt_path = fetch("smiles_lstm_zinc50k.pt", subdir="models")
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
vocab = Vocab([])
vocab.itos = ckpt["vocab"]; vocab.stoi = {t: i for i, t in enumerate(vocab.itos)}
prior = SmilesLSTM(len(vocab), **ckpt["config"]).to(device)
prior.load_state_dict(ckpt["state_dict"]); prior.eval()
print(f"vocabulary: {len(vocab)} tokens | training: {ckpt['training']}")
print(vocab.itos)

# %% [markdown]
"""
### Sampling

At each step we take the model's output logits, divide them by a **temperature** $T$ and sample:
low $T$ → conservative, repetitive molecules; high $T$ → more diverse but more invalid strings.
"""

# %%
@torch.no_grad()
def sample_smiles(model, n=100, max_len=120, temperature=1.0, batch_size=256):
    model.eval()
    out = []
    while len(out) < n:
        b = min(batch_size, n - len(out))
        x = torch.full((b, 1), vocab.bos, dtype=torch.long, device=device)
        state, finished = None, torch.zeros(b, dtype=torch.bool, device=device)
        seqs = [[] for _ in range(b)]
        for _ in range(max_len):
            logits, state = model(x, state)
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            x = torch.multinomial(probs, 1)
            for i, tok in enumerate(x.squeeze(1).tolist()):
                if not finished[i]:
                    if tok == vocab.eos: finished[i] = True
                    else: seqs[i].append(tok)
            if finished.all(): break
        out.extend(vocab.decode(s) for s in seqs)
    return out[:n]

t0 = time.time()
samples = sample_smiles(prior, n=1000, temperature=1.0)
print(f"{len(samples)} strings sampled in {time.time() - t0:.0f} s")
print(samples[:5])

# %% [markdown]
"""
### Evaluation: validity, uniqueness, novelty

The three basic metrics of distribution learning (GuacaMol, MOSES):
- **validity**: fraction of generated strings that RDKit can parse;
- **uniqueness**: fraction of distinct molecules among the valid ones;
- **novelty**: fraction of valid, unique molecules *not* in the training set.

Good numbers alone are not enough: the generated molecules should also have property distributions similar to the
training data (measured e.g. by the Fréchet ChemNet Distance, FCD).
"""

# %%
train_canonical = set(zinc["smiles"])

def evaluate_generation(smiles_list, reference=train_canonical, verbose=True):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid = [(s, m) for s, m in zip(smiles_list, mols) if m is not None and m.GetNumAtoms() > 0]
    canon = [Chem.MolToSmiles(m) for _, m in valid]
    unique = set(canon)
    novel = unique - reference
    stats = {"validity": len(valid) / len(smiles_list),
             "uniqueness": len(unique) / max(len(valid), 1),
             "novelty": len(novel) / max(len(unique), 1)}
    if verbose:
        print(" | ".join(f"{k}: {v:.1%}" for k, v in stats.items()))
    return stats, [m for _, m in valid], canon

stats, valid_mols, canon = evaluate_generation(samples)

# %%
# The effect of temperature
rows = []
for T in [0.6, 0.8, 1.0, 1.2, 1.5]:
    s, _, _ = evaluate_generation(sample_smiles(prior, n=500, temperature=T), verbose=False)
    rows.append({"temperature": T, **s})
pd.DataFrame(rows).round(3)

# %%
Draw.MolsToGridImage(valid_mols[:12], molsPerRow=4, subImgSize=(200, 160),
                     legends=[f"MW {Descriptors.MolWt(m):.0f}, QED {QED.qed(m):.2f}" for m in valid_mols[:12]])

# %%
# Do the generated molecules look like the training set?
def property_frame(mols, label):
    return pd.DataFrame({"MW": [Descriptors.MolWt(m) for m in mols],
                         "logP": [Descriptors.MolLogP(m) for m in mols],
                         "QED": [QED.qed(m) for m in mols],
                         "TPSA": [Descriptors.TPSA(m) for m in mols],
                         "set": label})

train_sample = [Chem.MolFromSmiles(s) for s in zinc["smiles"].sample(1000, random_state=0)]
comp = pd.concat([property_frame(train_sample, "ZINC training set"), property_frame(valid_mols, "generated")])
fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
for ax, col in zip(axes, ["MW", "logP", "QED", "TPSA"]):
    sns.kdeplot(data=comp, x=col, hue="set", common_norm=False, ax=ax, fill=True, alpha=0.3)
plt.tight_layout(); plt.show()

# %% [markdown]
"""
The model has learned the *shape* of the training distribution — that is exactly what "distribution learning" means.

### Exercise 2.1
1. Sample 200 molecules at T = 0.7 and at T = 1.3 and compare the distributions of MW and the number of rings.
2. What fraction of generated molecules contain a scaffold *not* present in the training set? (Use `MurckoScaffold`.)
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
# 1. temperature and property distributions
for T in [0.7, 1.3]:
    _, mols_T, _ = evaluate_generation(sample_smiles(prior, n=200, temperature=T), verbose=False)
    mw = [Descriptors.MolWt(m) for m in mols_T]
    rings = [Chem.rdMolDescriptors.CalcNumRings(m) for m in mols_T]
    print(f"T={T}: n={len(mols_T)}  MW {np.mean(mw):.0f} +/- {np.std(mw):.0f}   rings {np.mean(rings):.1f}")
# Higher T -> more (and weirder) rings and a wider MW spread, at the cost of validity.

# 2. novel scaffolds
train_scaffolds = {Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)))
                   for s in zinc["smiles"].sample(10000, random_state=0)}
gen_scaffolds = [Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) for m in valid_mols]
new = [s for s in gen_scaffolds if s not in train_scaffolds]
print(f"{len(new)/len(gen_scaffolds):.0%} of generated molecules have a scaffold absent from the 10k training sample")
```
</details>
"""


# %% [markdown]
"""
## 3. Goal-directed design I: fine-tuning (transfer learning)

The prior generates "generic ZINC-like" molecules. Suppose we want EGFR-inhibitor-like molecules. The simplest approach:
continue training the prior on a small set of known actives. The model shifts towards that chemical series.
"""

# %%
egfr = pd.read_csv(fetch("EGFR_curated.csv"))
actives = egfr[egfr["pIC50"] >= 7.5]["smiles"].tolist()
print(len(actives), "highly active EGFR compounds for fine-tuning")
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in actives[:6]], molsPerRow=3, subImgSize=(200, 150))

# %%
def pad_batch(seqs, pad=0):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [pad] * (L - len(s)) for s in seqs], dtype=torch.long)

def finetune(model, smiles_list, epochs=8, lr=3e-4, batch_size=64):
    model = model.to(device).train()
    data = [vocab.encode(s) for s in smiles_list if len(tokenize(s)) < 110]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        perm = np.random.permutation(len(data)); tot = 0
        for i in range(0, len(data), batch_size):
            batch = pad_batch([data[j] for j in perm[i:i + batch_size]]).to(device)
            logits, _ = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * batch.size(0)
        print(f"epoch {epoch}: loss {tot / len(data):.3f}")
    return model.eval()

import copy
agent_ft = finetune(copy.deepcopy(prior), actives, epochs=8)

# %%
ft_samples = sample_smiles(agent_ft, n=500, temperature=1.0)
stats_ft, ft_mols, ft_canon = evaluate_generation(ft_samples)
Draw.MolsToGridImage(ft_mols[:12], molsPerRow=4, subImgSize=(200, 160))

# %% [markdown]
"""
> Notice that **validity drops** after fine-tuning (from ~75 % to ~50 %). We trained a 13 M-parameter model for 8 epochs on
> 1685 molecules: it starts to overfit, and EGFR inhibitors are longer and more complex strings than the average ZINC
> molecule, so more of the generated strings fail to close a ring. Fewer epochs, a lower learning rate, or mixing in ZINC
> molecules ("experience replay") would keep validity up. This trade-off — focus versus validity — is characteristic of
> transfer learning on small chemical datasets.
"""

# %%
# Are the fine-tuned molecules more similar to EGFR actives than the prior's molecules?
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
active_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in actives]

def max_similarity_to_actives(mols):
    return np.array([max(DataStructs.BulkTanimotoSimilarity(fpgen.GetFingerprint(m), active_fps)) for m in mols])

sim_prior = max_similarity_to_actives(valid_mols[:300])
sim_ft = max_similarity_to_actives(ft_mols[:300])
plt.figure(figsize=(6, 3))
plt.hist(sim_prior, bins=30, alpha=0.6, label=f"prior (median {np.median(sim_prior):.2f})")
plt.hist(sim_ft, bins=30, alpha=0.6, label=f"fine-tuned (median {np.median(sim_ft):.2f})")
plt.xlabel("max Tanimoto to a known EGFR active"); plt.legend(); plt.show()

# %% [markdown]
"""
## 4. Goal-directed design II: reinforcement learning (REINVENT-style)

Fine-tuning needs examples of what we want. **Reinforcement learning** only needs a **scoring function** $S(\text{molecule}) \in [0,1]$:
we sample molecules, score them, and increase the likelihood of the good ones.

REINVENT's trick is to stay close to the prior so the agent does not collapse onto weird, unsynthesisable strings:
the loss pushes the agent's log-likelihood towards an *augmented likelihood*

$$\log P_{\text{aug}}(x) = \log P_{\text{prior}}(x) + \sigma\, S(x), \qquad
L = \big(\log P_{\text{aug}}(x) - \log P_{\text{agent}}(x)\big)^2 .$$

Our objective: **molecules similar to gefitinib, with good QED and reasonable size** — a multi-parameter score.
"""

# %%
GEFITINIB = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
target_fp = fpgen.GetFingerprint(Chem.MolFromSmiles(GEFITINIB))

def score_molecule(mol):
    """Multi-parameter score in [0, 1]: similarity to gefitinib × drug-likeness × size penalty."""
    if mol is None:
        return 0.0
    sim = DataStructs.TanimotoSimilarity(fpgen.GetFingerprint(mol), target_fp)
    qed = QED.qed(mol)
    mw = Descriptors.MolWt(mol)
    size_ok = 1.0 if 250 <= mw <= 550 else max(0.0, 1 - abs(mw - 400) / 400)
    return float(sim ** 0.5 * qed * size_ok)          # sqrt to keep the similarity signal alive at low values

for name, smi in [("gefitinib itself", GEFITINIB), ("erlotinib", "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC"),
                  ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"), ("a random generated one", canon[0])]:
    print(f"{name:24s} score = {score_molecule(Chem.MolFromSmiles(smi)):.3f}")

# %%
@torch.no_grad()
def sample_with_logprob(model, n, max_len=120, temperature=1.0):
    """Sample n SMILES and return (strings, token sequences) - log-probabilities are recomputed with gradients later."""
    model.eval()
    x = torch.full((n, 1), vocab.bos, dtype=torch.long, device=device)
    state = None
    finished = torch.zeros(n, dtype=torch.bool, device=device)
    seqs = [[] for _ in range(n)]
    for _ in range(max_len):
        logits, state = model(x, state)
        probs = F.softmax(logits[:, -1] / temperature, dim=-1)
        x = torch.multinomial(probs, 1)
        for i, tok in enumerate(x.squeeze(1).tolist()):
            if not finished[i]:
                if tok == vocab.eos: finished[i] = True
                else: seqs[i].append(tok)
        if finished.all(): break
    return [vocab.decode(s) for s in seqs], seqs

def sequence_logprob(model, seqs):
    """Sum of log p(token) over each sequence (with gradients, for the agent)."""
    batch = pad_batch([[vocab.bos] + s + [vocab.eos] for s in seqs]).to(device)
    logits, _ = model(batch[:, :-1])
    logp = F.log_softmax(logits, dim=-1)
    target = batch[:, 1:]
    token_logp = logp.gather(2, target.unsqueeze(-1)).squeeze(-1)
    mask = (target != vocab.pad).float()
    return (token_logp * mask).sum(1)

def reinvent(prior, epochs=120, batch=64, sigma=60, lr=1e-4, verbose_every=20):
    agent = copy.deepcopy(prior).to(device)
    opt = torch.optim.Adam(agent.parameters(), lr=lr)
    history, best = [], []
    for epoch in range(epochs):
        smiles, seqs = sample_with_logprob(agent, batch)
        keep = [(s, q) for s, q in zip(smiles, seqs) if len(q) > 3]
        if not keep: continue
        smiles, seqs = zip(*keep)
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        scores = torch.tensor([score_molecule(m) for m in mols], dtype=torch.float32, device=device)

        agent_lp = sequence_logprob(agent, list(seqs))
        with torch.no_grad():
            prior_lp = sequence_logprob(prior, list(seqs))
        augmented = prior_lp + sigma * scores
        loss = ((augmented - agent_lp) ** 2).mean()
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 1.0); opt.step()

        valid = [(m, float(s)) for m, s in zip(mols, scores) if m is not None]
        history.append({"epoch": epoch, "mean score": float(scores.mean()), "max score": float(scores.max()),
                        "validity": len(valid) / len(mols)})
        best.extend(valid)
        if verbose_every and epoch % verbose_every == 0:
            print(f"epoch {epoch:3d}  mean score {history[-1]['mean score']:.3f}  "
                  f"max {history[-1]['max score']:.3f}  validity {history[-1]['validity']:.0%}")
    return agent, pd.DataFrame(history), best

t0 = time.time()
agent_rl, hist, best_mols = reinvent(prior, epochs=120)
print(f"{time.time() - t0:.0f} s")

# %%
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.plot(hist["epoch"], hist["mean score"], label="mean score")
ax.plot(hist["epoch"], hist["max score"], label="best in batch", alpha=0.6)
ax.set_xlabel("RL epoch"); ax.set_ylabel("score"); ax.legend(); ax.set_title("REINVENT-style optimisation")
plt.show()

# %%
# The best molecules found during the run
seen, top = set(), []
for m, s in sorted(best_mols, key=lambda x: -x[1]):
    smi = Chem.MolToSmiles(m)
    if smi not in seen:
        seen.add(smi); top.append((m, s, smi))
    if len(top) == 8: break

Draw.MolsToGridImage([t[0] for t in top] + [Chem.MolFromSmiles(GEFITINIB)], molsPerRow=3, subImgSize=(220, 170),
                     legends=[f"score {t[1]:.2f}, novel: {t[2] not in train_canonical}" for t in top] + ["GEFITINIB (target)"])

# %% [markdown]
"""
> **Reward hacking.** With a weak score (say, similarity alone) the agent quickly learns to output the target molecule
> itself, or long repetitive strings that trick the fingerprint. This is why real campaigns use multi-parameter scores
> including synthesisability (SA score), predicted ADMET, diversity bonuses, and why they check the output *by hand*.

### Exercise 4.1
1. Change `sigma` to 20 and to 120. What happens to the mean score, the validity and the diversity of the output?
2. Replace the scoring function by "**QED only**". What kind of molecules does the agent converge to?
3. Add a **synthesisability** term: `from rdkit.Chem import RDConfig; import sys; sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score')); import sascorer; sascorer.calculateScore(mol)` (1 = easy, 10 = hard). Score = similarity × QED × (10 − SA)/9.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
# 1. sigma controls how strongly the score outweighs the prior
for sigma in [20, 60, 120]:
    _, h, b = reinvent(prior, epochs=40, sigma=sigma, verbose_every=0)
    uniq = len({Chem.MolToSmiles(m) for m, _ in b})
    print(f"sigma={sigma:3d}: final mean score {h['mean score'].iloc[-5:].mean():.3f}  "
          f"validity {h['validity'].iloc[-5:].mean():.0%}  unique molecules {uniq}")
# Small sigma: the agent barely moves. Large sigma: the score dominates, the agent drifts away from
# the prior, validity and diversity collapse (mode collapse onto a few strings).

# 2. QED only -> small, simple, very "drug-like-by-numbers" molecules; QED is maximised around
#    MW 300, logP 2-3, so the agent converges to a narrow region of chemical space.

# 3. with synthesisability
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

def score_with_sa(mol):
    if mol is None: return 0.0
    sim = DataStructs.TanimotoSimilarity(fpgen.GetFingerprint(mol), target_fp)
    return float(sim ** 0.5 * QED.qed(mol) * (10 - sascorer.calculateScore(mol)) / 9)

_backup = score_molecule
score_molecule = score_with_sa
agent_sa, hist_sa, best_sa = reinvent(prior, epochs=60)
score_molecule = _backup                       # reinvent() reads the global name
print("median SA of the best molecules:",
      round(np.median([sascorer.calculateScore(m) for m, _ in best_sa[-100:]]), 2))
```
</details>
"""


# %% [markdown]
"""
## 5. Goal-directed design III: a genetic algorithm (no neural network)

Evolutionary algorithms remain a *very* strong baseline (Tripp & Hernández-Lobato 2023; GB-GA of Jensen 2019):
start from a population, **mutate** and **cross over** molecules, keep the best, repeat. Here we implement simple
graph mutations in the spirit of CReM (Polishchuk 2020): replace a substituent, add a ring, delete an atom.
"""

# %%
FRAGMENTS = ["C", "CC", "CCC", "C(C)C", "O", "OC", "N", "NC", "N(C)C", "F", "Cl", "Br", "C#N", "C(=O)N", "C(=O)O",
             "S(=O)(=O)N", "c1ccccc1", "c1ccncc1", "C1CCNCC1", "C1CCOCC1", "OCCN1CCOCC1", "OC", "OCC"]

def mutate(mol, rng):
    """Simple mutations: substitute a hydrogen by a fragment, or delete a terminal atom."""
    mol = Chem.Mol(mol)
    op = rng.random()
    if op < 0.7:                                    # attach a fragment to a random atom with a free valence
        candidates = [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() > 0]
        if not candidates: return None
        idx = rng.choice(candidates)
        frag = Chem.MolFromSmiles(rng.choice(FRAGMENTS))
        combo = Chem.RWMol(Chem.CombineMols(mol, frag))
        combo.AddBond(int(idx), mol.GetNumAtoms(), Chem.BondType.SINGLE)
        new = combo.GetMol()
    else:                                           # delete a terminal heavy atom
        terminals = [a.GetIdx() for a in mol.GetAtoms() if a.GetDegree() == 1 and not a.IsInRing()]
        if not terminals: return None
        rw = Chem.RWMol(mol); rw.RemoveAtom(int(rng.choice(terminals)))
        new = rw.GetMol()
    try:
        Chem.SanitizeMol(new)
    except Exception:
        return None
    return new if new.GetNumAtoms() > 5 else None

def genetic_algorithm(seed_smiles, score_fn, generations=60, population=60, elite=15, seed=0):
    """Mutate-and-select loop. Also records the mean properties of the elite population each generation."""
    rng = np.random.default_rng(seed)
    pop = [Chem.MolFromSmiles(s) for s in seed_smiles]
    history = []
    for gen in range(generations):
        children = []
        while len(children) < population:
            child = mutate(pop[rng.integers(len(pop))], rng)
            if child is not None:
                children.append(child)
        scored = sorted([(m, score_fn(m)) for m in pop + children], key=lambda x: -x[1])
        seen, keep = set(), []                                    # keep unique elites
        for m, sc in scored:
            smi = Chem.MolToSmiles(m)
            if smi not in seen:
                seen.add(smi); keep.append((m, sc))
            if len(keep) == elite: break
        pop = [m for m, _ in keep]
        history.append({"generation": gen, "best": keep[0][1], "mean elite": np.mean([sc for _, sc in keep]),
                        "MW": np.mean([Descriptors.MolWt(m) for m in pop]),
                        "QED": np.mean([QED.qed(m) for m in pop])})
    return pop, pd.DataFrame(history)

seeds = zinc["smiles"].sample(15, random_state=1).tolist()
t0 = time.time()
ga_pop, ga_hist = genetic_algorithm(seeds, score_molecule, generations=60, seed=1)
print(f"{time.time() - t0:.0f} s")

plt.figure(figsize=(6, 3))
plt.plot(ga_hist["generation"], ga_hist["best"], label="best")
plt.plot(ga_hist["generation"], ga_hist["mean elite"], label="mean of elite")
plt.xlabel("generation"); plt.ylabel("score"); plt.legend(); plt.title("Genetic algorithm"); plt.show()

Draw.MolsToGridImage(ga_pop[:6], molsPerRow=3, subImgSize=(220, 170),
                     legends=[f"score {score_molecule(m):.2f}" for m in ga_pop[:6]])

# %% [markdown]
"""
Compare the three approaches on the same objective:

| approach | needs | strength | weakness |
|---|---|---|---|
| fine-tuning | examples of the target series | simple, stable | cannot go beyond the given series; needs data |
| RL (REINVENT) | a scoring function | explores, multi-objective | reward hacking; hyperparameters; compute |
| genetic algorithm | a scoring function | no training, strong baseline, interpretable moves | mutations may be unrealistic; local optima |

## 6. Do the generated molecules make sense?

Three checks every generative-chemistry paper should pass (and most do not, fully):

1. **Chemical validity and stability** — beyond RDKit parsing: strange valences, unstable groups, reactive functionalities.
2. **Synthesisability** — the SA score (Ertl & Schuffenhauer 2009) or, better, a retrosynthesis tool (AiZynthFinder, Synplanner).
3. **Genuine novelty vs. memorisation** — is the molecule really new, or a training molecule with a shifted methyl?
"""

# %%
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer

sets = {"ZINC training": train_sample[:300], "prior samples": valid_mols[:300],
        "fine-tuned": ft_mols[:300], "RL best": [m for m, _ in best_mols[-300:]], "GA population": ga_pop}
rows = []
for name, mols in sets.items():
    for m in mols:
        if m is None: continue
        rows.append({"set": name, "SA score": sascorer.calculateScore(m), "QED": QED.qed(m), "MW": Descriptors.MolWt(m)})
sa_df = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, col in zip(axes, ["SA score", "QED", "MW"]):
    sns.boxplot(data=sa_df, x="set", y=col, ax=ax); ax.tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()
sa_df.groupby("set")[["SA score", "QED", "MW"]].median().round(2)

# %% [markdown]
"""
Read this table carefully — and notice that it does **not** show a disaster. All four generators sit close to the ZINC
training set, and the GA even has the *highest* QED. That is not luck: our score contains a **QED term** and a molecular-weight
window, which quietly keep the molecules drug-like. The interesting experiment is to remove that guard-rail and watch what
the optimiser does when the only thing that pays is fingerprint overlap.
"""

# %%
def score_similarity_only(mol):
    """A deliberately naive objective: Tanimoto similarity to gefitinib and nothing else."""
    if mol is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fpgen.GetFingerprint(mol), target_fp)

ga_pop_naive, ga_hist_naive = genetic_algorithm(seeds, score_similarity_only, generations=60, seed=1)

def elite_summary(pop, label):
    return {"objective": label, "MW": np.median([Descriptors.MolWt(m) for m in pop]),
            "SA score": np.median([sascorer.calculateScore(m) for m in pop]),
            "QED": np.median([QED.qed(m) for m in pop])}

pd.DataFrame([elite_summary([Chem.MolFromSmiles(s) for s in seeds], "starting population"),
              elite_summary(ga_pop, "similarity x QED x size"),
              elite_summary(ga_pop_naive, "similarity only")]).set_index("objective").round(2)

# %%
# The *trajectory* is what shows the drift: follow the elite population over the generations
fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
for label, h, pop_hist in [("similarity x QED x size", ga_hist, ga_pop), ("similarity only", ga_hist_naive, ga_pop_naive)]:
    axes[0].plot(h["generation"], h["best"], label=label)
    axes[1].plot(h["generation"], h["MW"], label=label)
    axes[2].plot(h["generation"], h["QED"], label=label)
axes[0].set_ylabel("best score in the population"); axes[1].set_ylabel("mean MW of the elite")
axes[2].set_ylabel("mean QED of the elite")
for ax in axes: ax.set_xlabel("generation"); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

Draw.MolsToGridImage(ga_pop_naive[:6], molsPerRow=3, subImgSize=(240, 190),
                     legends=[f"sim {score_similarity_only(m):.2f} | MW {Descriptors.MolWt(m):.0f} | "
                              f"SA {sascorer.calculateScore(m):.1f} | QED {QED.qed(m):.2f}" for m in ga_pop_naive[:6]])

# %% [markdown]
"""
> With the similarity-only objective the elite population drifts to **higher molecular weight**, a **higher SA score than
> the multi-parameter run** (harder to make) and a **QED cut roughly in half**: the optimiser bolts fragments on to maximise
> fingerprint overlap, and nothing in the score punishes it. With the multi-parameter objective the same algorithm stays
> compact and drug-like. Two lessons:
>
> 1. **A generative model is only as sensible as its objective function.** Every property you care about must appear in the
>    score, or the optimiser will trade it away. This is the best-documented failure mode of generative chemistry
>    (Gao & Coley, *The synthesizability of molecules proposed by generative models*, J. Chem. Inf. Model. 2020).
> 2. **Fingerprint similarity is a leaky proxy.** Adding atoms can raise Tanimoto overlap without making the molecule more
>    like the target in any way a chemist would recognise — the optimiser is exploiting the representation, not the chemistry.

## 7. Benchmarks and what comes next

- **GuacaMol** (BenevolentAI 2019): distribution-learning + 20 goal-directed tasks (rediscovery, similarity, MPO). A well-tuned
  *genetic algorithm* beats most neural models on it — a healthy reality check.
- **MOSES** (2020): standardised dataset, metrics (validity, FCD, scaffold similarity, novelty) and baselines.
- **REINVENT 4** (MolecularAI, Apache-2.0): production-grade RL for de novo design, linker design, scaffold hopping, with
  transformer priors and rich scoring components. `ReinventCommunity` has ready-made notebooks.
- **Structure-based generation**: models that generate *inside a protein pocket* (DiffSBDD, TargetDiff, PocketXMol) — 3D diffusion models.
- **Foundation & multimodal models**: molecular generation conditioned on text ("a soluble EGFR inhibitor without a nitro group"), the
  bridge to session 08.

## Exercises
1. **Scaffold hopping**: change the scoring function to reward molecules similar to gefitinib in *pharmacophore* terms but with a
   *different* Murcko scaffold: `score = sim × (scaffold != gefitinib_scaffold)`. What comes out?
2. **SELFIES**: repeat section 2 using SELFIES tokens (session 02). Validity should be 100 % by construction — check it, and discuss what this buys you.
3. **Latent-space exploration**: train a small VAE on 5 000 ZINC molecules (see the Schwaller group's `Molecular Generative Models` notebook) and interpolate between two molecules in latent space.
4. **Honest evaluation**: take your 100 best RL molecules, filter them by SA score < 4, PAINS-free (session 04) and Ro5-compliant. How many survive?

## Further reading
- Segler *et al.*, *Generating focused molecule libraries for drug discovery with RNNs*, ACS Cent. Sci. **2018**, 4, 120.
- Olivecrona *et al.*, *Molecular de-novo design through deep reinforcement learning* (REINVENT), J. Cheminform. **2017**, 9, 48.
- Brown *et al.*, *GuacaMol: benchmarking models for de novo molecular design*, J. Chem. Inf. Model. **2019**, 59, 1096.
- Gao & Coley, *The synthesizability of molecules proposed by generative models*, J. Chem. Inf. Model. **2020**, 60, 5714.
- Polishchuk, *CReM: chemically reasonable mutations framework*, J. Cheminform. **2020**, 12, 28.

Next session: **08 · Agentic AI** — LLMs that use chemistry tools.
"""
