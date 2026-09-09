# %% [markdown]
"""
# 06 · Deep learning for molecules: neural networks, graph neural networks and chemical language models

**Chemoinformatics practicals — Session 6 of 9**

> **Where this comes from.** These notebooks are a compilation of open teaching material generously published
> by the chemoinformatics community. The original authors are named in the credits below and in
> [`CREDITS.md`](https://github.com/AxelRolov/chemoinformatics_tutorials/blob/main/CREDITS.md) — all credit for the
> substance belongs to them, and we thank them for making their work reusable. This collection updates that material
> to current library versions, ports it to run start-to-finish in Google Colab, and adds exercises and connective
> text; the assembly and revision were done with **Claude** (Anthropic) and then reviewed.

> ⚡ In Colab choose *Runtime → Change runtime type → T4 GPU* for faster training (everything also runs on CPU in a few minutes).

**Learning goals.** After this session you will be able to
- explain what a neural network is (layers, activations, loss, gradient descent, back-propagation) and train one with **PyTorch**;
- build a **multilayer perceptron** on fingerprints and compare it with the random forest of session 05;
- turn molecules into graphs and train a **graph neural network** (message passing) with PyTorch Geometric;
- use a pre-trained **chemical language model** (ChemBERTa) as a feature extractor — your first *foundation model*;
- recognise the pitfalls of deep learning on small chemical datasets (overfitting, need for baselines, splits).

---
> **Credits — thank you to the original authors.** This session adapts, updates and revises material from
> - **TeachOpenCADD** talktorials **T022 · Ligand-based screening: neural networks** and **T035 · GNN-based molecular property prediction** (Volkamer lab; [GitHub](https://github.com/volkamerlab/teachopencadd), CC BY 4.0);
> - *AI for Chemistry* (EPFL CH-457), **Schwaller group** — `03 - Intro to Deep Learning` (`01_intro_to_dl`, `02_graph_nns`, `03_gnn_simple_example`; [GitHub](https://github.com/schwallergroup/ai4chem_course), MIT);
> - A. D. White, *Deep Learning for Molecules and Materials*, chapters *Deep learning*, *Graph neural networks* ([dmol.pub](https://dmol.pub), CC BY-NC 3.0);
> - *Introduction to AI for pharma students* by the **Koch group** (`07 - First Neural Net`, `08 - PyTorch`, `13 - Graph Neural Networks`; [GitHub](https://github.com/kochgroup/intro_pharma_ai), CC BY-NC-SA 4.0 — consulted for structure, not copied);
> - Pat Walters' `chemprop/run_chemprop` notebook ([GitHub](https://github.com/PatWalters/practical_cheminformatics_tutorials), MIT);
> - the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io) tutorials (MIT). Data: ESOL (Delaney 2004) and the curated EGFR set (ChEMBL, CC BY-SA 3.0).
"""

# %%
# @title ⚙️ Setup — run this cell first (≈1–2 min on Colab)
import sys, os, subprocess, time, warnings, math
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "torch_geometric", "scikit-learn", "seaborn", "transformers"], check=False)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdFingerprintGenerator
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
RDLogger.DisableLog("rdApp.*")
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
from sklearn.ensemble import RandomForestRegressor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("PyTorch", torch.__version__, "| device:", device)
torch.manual_seed(0); np.random.seed(0)

REPO_RAW = "https://raw.githubusercontent.com/AxelRolov/chemoinformatics_tutorials/main"
def data_path(filename):
    local = os.path.join("..", "data", filename)
    return local if os.path.exists(local) else f"{REPO_RAW}/data/{filename}"

def rmse(y, p): return float(np.sqrt(mean_squared_error(y, p)))

# %% [markdown]
"""
## 1. Neural networks in one picture

A **neuron** computes a weighted sum of its inputs and applies a non-linear **activation**:
$h = \sigma(\mathbf w \cdot \mathbf x + b)$. A **layer** stacks many neurons; a **network** stacks layers:
$\hat y = W_3\,\sigma(W_2\,\sigma(W_1 \mathbf x + b_1) + b_2) + b_3$. With enough neurons it can approximate any function.

Training = **minimising a loss** $L(\theta)$ (e.g. mean squared error) over the parameters $\theta = \{W_i, b_i\}$ by
**gradient descent**: $\theta \leftarrow \theta - \eta \nabla_\theta L$. The gradient is computed automatically by
**back-propagation** (the chain rule, run backwards through the network) — in PyTorch this is `loss.backward()`.

Let's see the whole mechanism on the smallest possible example: fitting a straight line.
"""

# %% [markdown]
"""
**The whole of deep learning, in fourteen lines.** Read this cell slowly, because every training loop in the rest of
the course is this one with more machinery around it.

- `requires_grad=True` on `w` and `b` is the switch that matters. It tells PyTorch to record every operation these
  two numbers take part in, building a graph it can later differentiate. Without it, `loss.backward()` has nothing
  to work with.
- `y_hat = w * x + b` is the **forward pass**: our current guess.
- `loss = ((y_hat - y) ** 2).mean()` measures how wrong that guess is.
- `loss.backward()` is **back-propagation** — the chain rule run backwards through the recorded graph. It does not
  change anything; it *fills in* `w.grad` and `b.grad` with $\partial L/\partial w$ and $\partial L/\partial b$.
- `w -= lr * w.grad` takes the step downhill. It sits inside `torch.no_grad()` because updating the parameters is
  bookkeeping, not part of the function being differentiated — record it and you would be differentiating your own
  optimiser.
- `w.grad.zero_()` is the one that catches everyone. PyTorch **accumulates** gradients rather than replacing them, so
  a forgotten `zero_()` means step 50 uses the sum of the first fifty gradients. In the rest of the notebook the
  optimiser does this for us, as `opt.zero_grad()`.

Starting from `w = b = 0`, a hundred steps recover `w = 2.06, b = -0.99` against the true 2 and −1. The gap is the
noise we added, not a failure of the optimiser.
"""

# %%
# Toy data: y = 2x - 1 + noise
x = torch.linspace(-1, 1, 50).unsqueeze(1)
y = 2 * x - 1 + 0.2 * torch.randn_like(x)

w = torch.zeros(1, requires_grad=True)        # parameters we will learn
b = torch.zeros(1, requires_grad=True)
lr = 0.1
history = []
for step in range(100):
    y_hat = w * x + b                          # forward pass
    loss = ((y_hat - y) ** 2).mean()           # mean squared error
    loss.backward()                            # back-propagation: computes dloss/dw, dloss/db
    with torch.no_grad():
        w -= lr * w.grad; b -= lr * b.grad     # gradient-descent step
        w.grad.zero_(); b.grad.zero_()
    history.append(loss.item())
print(f"learned w = {w.item():.2f}, b = {b.item():.2f}  (true: 2, -1)")
# %% [markdown]
"""
**Two views of the same run.** On the left, the loss against step number — the shape you will be staring at for the
rest of this session. Note that it falls steeply and then flattens: gradient descent makes fast progress while the
error is large and fine adjustments at the end.

On the right, the fitted line through the data. With one parameter pair and fifty points you can check the answer by
eye, which is exactly why we start here. Once the model has a million parameters, the loss curve is *all* you have.
"""

# %%
plt.figure(figsize=(8, 2.8)); plt.subplot(1, 2, 1); plt.plot(history); plt.xlabel("step"); plt.ylabel("loss")
plt.subplot(1, 2, 2); plt.scatter(x, y, s=8); plt.plot(x, (w * x + b).detach(), "r"); plt.xlabel("x"); plt.ylabel("y"); plt.show()

# %% [markdown]
"""
Everything else in deep learning is this loop with (i) a bigger function, (ii) a smarter optimiser (Adam), (iii) data fed in
**mini-batches**, and (iv) tricks against overfitting (dropout, weight decay, early stopping).

## 2. A multilayer perceptron on fingerprints

Same problem as session 05: predict ESOL solubility from Morgan fingerprints. We'll keep a **validation set** apart from
the test set to decide when to stop training.
"""

# %% [markdown]
"""
**The same data as session 05, deliberately.** ESOL, Morgan fingerprints, a 20 % test set with `random_state=42` —
identical to session 05 — so that every number in this notebook can be compared with the random forest directly.
Reusing a benchmark rather than inventing a new one is how you find out whether a fancier model is actually better.

One change from session 05: `dtype=np.float32`. Neural networks work in single precision (it is what GPUs are fast
at), and passing a `float64` array is a common source of dtype errors later.
"""

# %%
esol = pd.read_csv(data_path("esol_delaney.csv")).rename(columns={"measured log solubility in mols per litre": "logS"})
esol["mol"] = esol["smiles"].apply(Chem.MolFromSmiles)
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X_fp = np.array([fpgen.GetFingerprintAsNumPy(m) for m in esol["mol"]], dtype=np.float32)
y = esol["logS"].values.astype(np.float32)

# %% [markdown]
"""
**Three sets, not two.** Session 05 needed only train and test. A neural network also needs a **validation** set,
because we have to decide *when to stop training*, and that decision is a form of model selection — make it on the
test set and your test estimate is no longer honest.

So the split happens twice: 20 % test, then 15 % of the remainder as validation, leaving 766 / 136 / 226. The
`idx_trainval` array is kept because the random forest we compare against does not need a validation set, and giving
it all 902 molecules keeps the comparison fair.
"""

# %%
idx_trainval, idx_test = train_test_split(np.arange(len(esol)), test_size=0.2, random_state=42)
idx_train, idx_val = train_test_split(idx_trainval, test_size=0.15, random_state=42)
print(len(idx_train), "train /", len(idx_val), "validation /", len(idx_test), "test")

# %% [markdown]
"""
**Standardising the target.** logS runs from about −11 to +2, so the squared error of a badly-predicted insoluble
molecule is enormous, and the first gradient steps are dominated by scale rather than by structure. Subtracting the
mean and dividing by the standard deviation puts the target on a roughly unit scale, which lets one learning rate
work for every molecule.

`y_mean` and `y_std` come from the **training molecules only** — computing them on all the data would leak the test
set's distribution into training. They are kept in variables because every prediction has to be multiplied back out
before it can be compared with a measured solubility, which is what `predict_mlp` does a few cells below.
"""

# %%
# Standardise the target (helps optimisation); we un-standardise predictions later
y_mean, y_std = y[idx_train].mean(), y[idx_train].std()
y_norm = (y - y_mean) / y_std

# %% [markdown]
"""
**Defining a network in PyTorch.** Two conventions to learn here, because every model you write will use them.

First, a model is a class inheriting from `nn.Module`, with `super().__init__()` called first — that is what
registers the parameters so the optimiser can find them. Second, `forward` defines the computation, and you never
call it directly: `model(x)` invokes it through `__call__`, which also runs any registered forward hooks — one of
several reasons never to call `model.forward(x)` yourself. (Train/eval mode is separate: `model.train()` and
`model.eval()` just set a `self.training` flag that layers like `Dropout` read inside their own `forward`.)

The architecture itself is the plain default: `Linear → ReLU → Dropout`, twice, then a linear output. `hidden=(512,
128)` is a funnel, compressing 2048 sparse bits to 512 then 128 features. `Dropout(0.3)` zeroes 30 % of the
activations at random *during training only* — it is the reason we must remember `model.eval()` before predicting.
`squeeze(-1)` turns the `(batch, 1)` output into `(batch,)` so it matches the shape of the targets.

The parameter count printed at the end is the number worth remembering: **1.1 million parameters, trained on 766
molecules**. Fifteen hundred parameters per molecule. Everything about the next two cells follows from that ratio.
"""

# %%
class MLP(nn.Module):
    def __init__(self, n_in, hidden=(512, 128), dropout=0.3):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = MLP(2048)
print(model)
print("trainable parameters:", sum(p.numel() for p in model.parameters()))

# %% [markdown]
"""
**The training loop, with three things the toy example did not have.**

- **Mini-batches.** `torch.randperm` reshuffles the training set each epoch and we step over it in slices of 64.
  Batching is partly about memory and speed, but the noise it injects into each gradient also helps generalisation.
- **Adam** instead of hand-written gradient descent, with `weight_decay=1e-4` — an L2 penalty on the weights, the
  same regularisation idea as Ridge in session 05.
- **Early stopping, done properly.** After each epoch the model is switched to `eval()` mode, both losses are
  computed under `torch.no_grad()`, and whenever the validation loss improves we take a full **copy** of the weights
  (`{k: v.clone() ...}` — without `.clone()` you would store references that keep changing). At the end we
  `load_state_dict(best_state)`, so the model returned is the best one seen, not the last one trained.

That `train()` / `eval()` pair is not optional. In `train()` mode dropout is active and the network is deliberately
handicapped; forget to switch and your validation loss is measured on a crippled model.
"""

# %%
def to_tensor(a): return torch.tensor(a, dtype=torch.float32, device=device)

def train_mlp(model, X, y, idx_train, idx_val, epochs=150, lr=1e-3, batch_size=64, weight_decay=1e-4, verbose=True):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xtr, ytr = to_tensor(X[idx_train]), to_tensor(y[idx_train])
    Xva, yva = to_tensor(X[idx_val]), to_tensor(y[idx_val])
    best_val, best_state, curve = np.inf, None, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(Xtr), batch_size):                    # mini-batches
            bi = perm[i:i + batch_size]
            loss = F.mse_loss(model(Xtr[bi]), ytr[bi])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tr_loss = F.mse_loss(model(Xtr), ytr).item(); va_loss = F.mse_loss(model(Xva), yva).item()
        curve.append((tr_loss, va_loss))
        if va_loss < best_val:                                        # keep the best model on validation data
            best_val, best_state = va_loss, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 25 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}  train MSE {tr_loss:.3f}  val MSE {va_loss:.3f}")
    model.load_state_dict(best_state)
    return model, np.array(curve)

# %% [markdown]
"""
**Run it — and read the printed numbers, not the final RMSE.** 150 epochs, about four minutes on a CPU and seconds on
a GPU.

The training loss falls to **0.018** while the validation loss stalls around **0.24** — more than ten times higher.
That gap *is* overfitting, in numbers: the network has essentially memorised 766 molecules. And notice when it
happened: by epoch 25 the validation loss is already 0.268, so almost nothing after epoch 25 helped. Early stopping
kept the model honest; the remaining 125 epochs were spent memorising.
"""

# %%
t0 = time.time()
mlp, curve = train_mlp(MLP(2048), X_fp, y_norm, idx_train, idx_val)
print(f"{time.time() - t0:.0f} s")

# %% [markdown]
"""
**Predicting means undoing the standardisation.** `model.eval()` turns dropout off, `torch.no_grad()` skips building
the gradient graph (faster and less memory), `.cpu().numpy()` brings the tensor back to numpy — and then
`* y_std + y_mean` converts the standardised output back into log solubility units. Miss that last step and your
RMSE will look suspiciously excellent, because it will be measured in standard deviations.
"""

# %%
def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        return model(to_tensor(X)).cpu().numpy() * y_std + y_mean

# %% [markdown]
"""
**The picture of overfitting.** The two curves separate early and never come back together: training loss keeps
falling, validation loss flattens. This shape is the single most useful diagnostic in deep learning, and it tells you
what to do next — the fix for a large gap is *more regularisation or more data*, never more epochs. (Exercise 2.1
asks you to remove the regularisation and watch the validation curve turn upwards.)
"""

# %%
plt.figure(figsize=(5, 3)); plt.plot(curve[:, 0], label="train"); plt.plot(curve[:, 1], label="validation")
plt.xlabel("epoch"); plt.ylabel("MSE (standardised)"); plt.legend(); plt.title("Learning curves"); plt.show()

# %% [markdown]
"""
**The comparison that justifies the session — or doesn't.** MLP: test RMSE **1.087**, R² 0.750. Random forest on the
same fingerprints: **1.175**, R² 0.708.

So a 1.1-million-parameter network, four minutes of training and three regularisation techniques buy about 0.09 log
units over a random forest that fits in ten seconds and has no hyperparameters worth tuning. On 900 molecules, that
is what deep learning on a fixed, hand-designed representation gets you.

Worth holding both numbers in mind: session 05 reached RMSE 0.674 on this dataset — with descriptors instead of
fingerprints. Neither model here is close to that. The bottleneck is not the model; it is the representation. Which
is precisely what section 3 attacks.
"""

# %%
p_mlp = predict_mlp(mlp, X_fp[idx_test])
print(f"MLP on ECFP4:  test RMSE = {rmse(y[idx_test], p_mlp):.3f}   R² = {r2_score(y[idx_test], p_mlp):.3f}")
rf = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(X_fp[idx_trainval], y[idx_trainval])
p_rf = rf.predict(X_fp[idx_test])
print(f"RF  on ECFP4:  test RMSE = {rmse(y[idx_test], p_rf):.3f}   R² = {r2_score(y[idx_test], p_rf):.3f}")

# %% [markdown]
"""
The validation loss stops improving long before the training loss does: the network **memorises** the training set.
Early stopping, dropout and weight decay are what keep it honest. On 900 molecules, an MLP on fingerprints is roughly on par
with the random forest — deep learning's advantage appears when it can *learn the representation* from richer input.

### Exercise 2.1
1. Set `dropout=0` and `weight_decay=0`. What happens to the gap between the two curves?
2. Try `hidden=(64,)` and `hidden=(1024, 512, 128)`. Bigger is better?
3. Train on **descriptors** instead of fingerprints (standardise them with `StandardScaler` first!). Compare with session 05.
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
# 1. no regularisation -> the validation loss stops improving early and then rises: classic overfitting
mlp_no_reg, curve_no_reg = train_mlp(MLP(2048, dropout=0.0), X_fp, y_norm, idx_train, idx_val,
                                     weight_decay=0.0, verbose=False)
plt.plot(curve_no_reg[:, 0], label="train (no reg.)"); plt.plot(curve_no_reg[:, 1], label="val (no reg.)")
plt.plot(curve[:, 0], "--", label="train (dropout+wd)"); plt.plot(curve[:, 1], "--", label="val (dropout+wd)")
plt.legend(); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.show()

# 2. capacity
for hidden in [(64,), (512, 128), (1024, 512, 128)]:
    m, _ = train_mlp(MLP(2048, hidden=hidden), X_fp, y_norm, idx_train, idx_val, verbose=False)
    print(hidden, "test RMSE", round(rmse(y[idx_test], predict_mlp(m, X_fp[idx_test])), 3))
# Bigger is not better on 900 molecules: capacity beyond ~1e5 parameters just overfits faster.

# 3. descriptors
from sklearn.preprocessing import StandardScaler
from rdkit.Chem import Descriptors as D
Xd = pd.DataFrame([D.CalcMolDescriptors(m) for m in esol["mol"]]).replace([np.inf, -np.inf], np.nan)
Xd = Xd.loc[:, Xd.notna().all() & (Xd.nunique() > 1)]
Xd = StandardScaler().fit_transform(Xd).astype(np.float32)
m, _ = train_mlp(MLP(Xd.shape[1]), Xd, y_norm, idx_train, idx_val, verbose=False)
print("MLP on descriptors: test RMSE", round(rmse(y[idx_test], predict_mlp(m, Xd[idx_test])), 3))
# ~0.7, i.e. as good as the GCN and the boosted trees of session 05 - the representation, not the model, was the bottleneck.
```
</details>
"""


# %% [markdown]
"""
## 3. Graph neural networks: learning the representation

Fingerprints are hand-designed. A **graph neural network** learns its own features directly from the molecular graph
(session 02): each atom starts with a feature vector $h_v^{(0)}$ and repeatedly **aggregates messages from its neighbours**:

$$h_v^{(k+1)} = \phi\Big(h_v^{(k)},\ \sum_{u \in \mathcal N(v)} \psi\big(h_u^{(k)}, e_{uv}\big)\Big)$$

After $K$ rounds, each atom "knows" its environment up to $K$ bonds away — exactly like a Morgan fingerprint of radius $K$,
except that $\phi$ and $\psi$ are learned. A **readout** (sum/mean over atoms) then gives one vector per molecule, fed to an MLP.
This is *message passing* (Gilmer *et al.* 2017); GCN, GIN, GAT, MPNN, D-MPNN (Chemprop) are all variants of $\phi$ and $\psi$.
"""

# %% [markdown]
"""
**Turning a molecule into a graph, in three pieces.** The imports first: `Data` is PyG's container for one graph,
`DataLoader` batches graphs, and `GCNConv` / `GINConv` are message-passing layers while `global_mean_pool` /
`global_add_pool` are readouts. `ELEMENTS` and `HYBRID` are the vocabularies for the one-hot encodings that follow.
"""

# %%
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
HYBRID = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]

# %% [markdown]
"""
**Choosing the atom features.** Every atom becomes a 25-dimensional vector, and this function is where a chemist's
judgement enters a graph neural network — it is the analogue of choosing descriptors in session 05.

Note that almost everything is **one-hot** rather than numeric: `[int(atom.GetDegree() == d) for d in range(5)]`
gives five binary columns instead of one column holding 0–4. That is deliberate. A single numeric column asserts that
degree 4 is "twice as much" as degree 2 and that the relationship is linear; one-hot columns let the network learn a
separate weight for each case, which is what you want for a categorical quantity. The `int(symbol not in ELEMENTS)`
column is the catch-all "other element" bucket — without it, a boron atom would silently become an all-zero vector.

Formal charge is the one exception, left numeric, because −1 / 0 / +1 really is ordered.
"""

# %%
def atom_features(atom):
    f = [int(atom.GetSymbol() == e) for e in ELEMENTS] + [int(atom.GetSymbol() not in ELEMENTS)]
    f += [int(atom.GetDegree() == d) for d in range(5)]
    f += [int(atom.GetTotalNumHs() == h) for h in range(4)]
    f += [int(atom.GetHybridization() == h) for h in HYBRID]
    f += [int(atom.GetIsAromatic()), int(atom.IsInRing()), atom.GetFormalCharge()]
    return f

# %% [markdown]
"""
**Building the `Data` object.** Three fields matter.

`x` is the atom-feature matrix, one row per atom. `edge_index` is a `(2, n_edges)` tensor listing bonds as pairs of
atom indices — and the crucial line is `edges + [(j, i) for i, j in edges]`: PyG expects **directed** edges, so every
bond has to be listed in both directions or messages will only ever flow one way along it. Forgetting that is the
most common beginner bug in PyG, and it does not raise an error, it just trains a worse model. `.t().contiguous()`
puts it in the shape and memory layout PyG wants.

The `if edges` guard handles single atoms, which have no bonds; and `y` goes in as a one-element tensor because the
target belongs to the whole graph, not to any atom.

Printing the first molecule shows `Data(x=[32, 25], edge_index=[2, 68], y=[1])`: 32 atoms, 25 features each, and 68
directed edges — so 34 bonds. Check that the numbers are twice what you expect; if not, you forgot the reverse edges.
"""

# %%
def mol_to_graph(mol, y=None):
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    edge_index = torch.tensor(edges + [(j, i) for i, j in edges], dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data

# %% [markdown]
"""
**Convert one molecule and look at it.** Always inspect a single example before building 1128 of them. `Data` prints
the shape of every field it holds, which is enough to catch a transposed `edge_index` or a missing target.
"""

# %%
g = mol_to_graph(esol["mol"][0], y_norm[0])
print(g, "\nnode feature length:", g.x.shape[1])

# %% [markdown]
"""
**Batching graphs is stranger than batching vectors.** Molecules have different numbers of atoms, so they cannot be
stacked into a rectangular tensor. PyG's solution is to **concatenate** the graphs of a batch into one big
disconnected graph and add a `batch` vector saying which molecule each atom belongs to.

The printout makes it concrete: `DataBatch(x=[416, 25], edge_index=[2, 844], y=[32], batch=[416], ptr=[33])` — 32
molecules averaging 13 atoms, so 416 atom rows, one target per molecule, and a 416-long `batch` vector. That vector
is what `global_mean_pool` uses to average each molecule's atoms separately at readout time. Message passing needs no
knowledge of the batch at all, because there are no edges between molecules.

Only the training loader shuffles; validation and test use larger batches since no gradients are stored.
"""

# %%
graphs = [mol_to_graph(m, t) for m, t in zip(esol["mol"], y_norm)]
train_loader = DataLoader([graphs[i] for i in idx_train], batch_size=32, shuffle=True)
val_loader = DataLoader([graphs[i] for i in idx_val], batch_size=128)
test_loader = DataLoader([graphs[i] for i in idx_test], batch_size=128)
batch = next(iter(train_loader))
print(batch)                     # PyG concatenates the graphs of a batch into one big disconnected graph

# %% [markdown]
"""
**The network: three rounds of message passing, then a readout.** Compare it with the MLP and notice how little
there is.

`self.convs` is a list of `GCNConv` layers — the first maps 25 input features to 128, the rest 128 to 128. Each call
`conv(h, data.edge_index)` is one round in which every atom collects a normalised sum of its neighbours' vectors and
transforms it. After three rounds each atom's vector encodes its environment up to three bonds away, which is exactly
the information a Morgan fingerprint of radius 3 records — except that here *what* is recorded is learned from the
solubility data rather than fixed in advance.

`BatchNorm1d` between rounds keeps activations from drifting as they are repeatedly mixed; deep GNNs are hard to
train without it.

`embed()` ends with `global_mean_pool(h, data.batch)`, the **readout**: average all of a molecule's atom vectors into
one 128-dimensional molecular vector. It is written as a separate method on purpose — that vector is a learned
molecular representation, and section 3's last cell looks at it. `forward()` is then just `head(embed(...))`.
"""

# %%
class GCN(nn.Module):
    def __init__(self, n_in, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([GCNConv(n_in if i == 0 else hidden, hidden) for i in range(n_layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def embed(self, data):                                     # molecule-level learned representation
        h = data.x
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h, data.edge_index)))         # message passing round
            h = F.dropout(h, self.dropout, self.training)
        return global_mean_pool(h, data.batch)                 # readout

    def forward(self, data):
        return self.head(self.embed(data)).squeeze(-1)

# %% [markdown]
"""
**The GNN training loop.** Structurally identical to `train_mlp` — same Adam, same best-on-validation checkpoint —
with two differences.

The batching is now handled by the `DataLoader`, so we iterate `for data in train_loader` and move each batch to the
device with `data.to(device)`; `data.num_graphs` is how many molecules it holds.

And there is a **learning-rate scheduler**: `ReduceLROnPlateau(factor=0.5, patience=10)` halves the learning rate
whenever the validation loss has not improved for ten epochs. This is standard practice and it is why the loss curve
below has a visible knee — big steps to find the right region, small steps to settle into it.
"""

# %%
def train_gnn(model, train_loader, val_loader, epochs=120, lr=1e-3, verbose=True):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    best, best_state, curve = np.inf, None, []
    for epoch in range(epochs):
        model.train(); tot = 0
        for data in train_loader:
            data = data.to(device)
            loss = F.mse_loss(model(data), data.y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * data.num_graphs
        val = evaluate_gnn(model, val_loader)
        sched.step(val); curve.append((tot / len(train_loader.dataset), val))
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
        if verbose and (epoch % 20 == 0 or epoch == epochs - 1):
            print(f"epoch {epoch:3d}  train MSE {curve[-1][0]:.3f}  val MSE {val:.3f}")
    model.load_state_dict(best_state)
    return model, np.array(curve)

# %% [markdown]
"""
**Two evaluation helpers.** Both are decorated with `@torch.no_grad()`, which switches off gradient tracking for the
whole function — the decorator form is tidier than wrapping the body, and forgetting it is a common cause of
mysterious memory growth during evaluation.

`evaluate_gnn` uses `reduction="sum"` and divides by the dataset size at the end, rather than averaging per batch.
That matters whenever the last batch is smaller than the others: averaging averages would silently over-weight it.
`predict_gnn` concatenates the per-batch predictions and undoes the target standardisation, exactly as `predict_mlp`
did.
"""

# %%
@torch.no_grad()
def evaluate_gnn(model, loader):
    model.eval(); tot = 0
    for data in loader:
        data = data.to(device)
        tot += F.mse_loss(model(data), data.y, reduction="sum").item()
    return tot / len(loader.dataset)

@torch.no_grad()
def predict_gnn(model, loader):
    model.eval()
    return np.concatenate([model(d.to(device)).cpu().numpy() for d in loader]) * y_std + y_mean

# %% [markdown]
"""
**Run it, and compare three numbers.** About eight minutes on a CPU, well under one on a GPU.

Test RMSE **0.690**, R² 0.899 — against 1.087 for the MLP and 1.175 for the random forest on the same molecules and
the same split. That is a *large* improvement, and it is not because the GCN is a bigger model: it has fewer
parameters than the MLP. It is because it built its own features from the graph instead of being handed 2048
pre-decided bits.

Look at the training trace too: the final gap is train 0.063 against validation 0.099, where the MLP's was 0.018
against 0.239. The GCN both fits better *and* generalises better, because message passing has the structure of the
problem built into it and cannot memorise a molecule as an arbitrary pattern of bits.

For calibration: session 05's best classical model, XGBoost on 199 curated descriptors, reached 0.674. The GCN
matched it — starting from nothing but atoms and bonds. Two decades of descriptor design, rediscovered from data in
eight minutes. That is the honest case for deep learning in chemistry, and note carefully what it is *not*: it is not
"beats the classical model", it is "reaches the same place without needing the chemistry to be pre-encoded".
"""

# %%
t0 = time.time()
gcn, curve_g = train_gnn(GCN(graphs[0].x.shape[1]), train_loader, val_loader)
print(f"{time.time() - t0:.0f} s")
p_gcn = predict_gnn(gcn, test_loader)
print(f"GCN:  test RMSE = {rmse(y[idx_test], p_gcn):.3f}   R² = {r2_score(y[idx_test], p_gcn):.3f}")

# %% [markdown]
"""
**The three-panel summary.** The GCN's learning curves on the left — much closer together than the MLP's — and then
the two parity plots side by side on the same axes.

Look at the low-solubility tail again. The MLP's cloud fans out below logS = −6 while the GCN's stays close to the
diagonal, so the improvement is not spread evenly: it is concentrated exactly in the region where a fingerprint has
nothing to say, because "very insoluble" is about the whole molecule.
"""

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].plot(curve_g[:, 0], label="train"); axes[0].plot(curve_g[:, 1], label="validation"); axes[0].legend(); axes[0].set_title("GCN learning curves"); axes[0].set_xlabel("epoch")
for ax, (name, p) in zip(axes[1:], [("MLP on ECFP4", p_mlp), ("GCN", p_gcn)]):
    ax.scatter(y[idx_test], p, s=10, alpha=0.6); ax.plot([-11, 2], [-11, 2], "k--", lw=1)
    ax.set_title(f"{name}: RMSE {rmse(y[idx_test], p):.2f}"); ax.set_xlabel("measured logS"); ax.set_ylabel("predicted")
plt.tight_layout(); plt.show()

# %% [markdown]
"""
### What did the GNN learn? The embedding space

The vector produced by `embed()` is a **learned molecular representation**. Let's project it to 2D (PCA) and colour by logS:
molecules with similar solubility should end up close together — the network has organised chemical space around the task.

The cell runs every molecule through `gcn.embed(...)` — no gradients, batches of 256 — to get a 1128 × 128 matrix,
then keeps the two directions of largest variance with `PCA(n_components=2)`. The axes are unlabelled and left
without ticks on purpose: principal components of a learned embedding have no units and no meaning individually, and
only the *arrangement* is interpretable.

What to look for is a smooth colour gradient. Compare it mentally with the PCA of fingerprints or descriptors in
session 04, which was organised by structural family instead: this embedding is organised by **the property we
trained on**, because that is the only thing the loss ever asked about. Train the same architecture on toxicity and
the same molecules would rearrange. A learned representation is task-specific, which is its strength and its
limitation — and it is why the transfer-learning approach in section 4 is interesting.
"""

# %%
from sklearn.decomposition import PCA
all_loader = DataLoader(graphs, batch_size=256)
gcn.eval()
with torch.no_grad():
    emb = np.concatenate([gcn.embed(d.to(device)).cpu().numpy() for d in all_loader])
pcs = PCA(n_components=2).fit_transform(emb)
plt.figure(figsize=(5.5, 4.5))
sc = plt.scatter(pcs[:, 0], pcs[:, 1], c=y, cmap="viridis", s=8)
plt.colorbar(sc, label="measured logS"); plt.title("PCA of the GCN embedding (128-d)"); plt.xticks([]); plt.yticks([]); plt.show()

# %% [markdown]
"""
### Exercise 3.1
1. Replace `GCNConv` by `GINConv` (Graph Isomorphism Network): `GINConv(nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden)))`. GIN is provably more expressive than GCN. Is it better here?
2. Change the readout from `global_mean_pool` to `global_add_pool`. For which kinds of property would a *sum* make more sense than a *mean*?
3. Use `n_layers=1` and `n_layers=6`. Explain the results in terms of the "radius" each atom sees (and *over-smoothing*).
"""

# %%
# YOUR CODE HERE



# %% [markdown]
"""
<details><summary><b>Solution</b></summary>

```python
# 1. GIN
class GIN(nn.Module):
    def __init__(self, n_in, hidden=128, n_layers=3, dropout=0.1):
        super().__init__()
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        for i in range(n_layers):
            d = n_in if i == 0 else hidden
            self.convs.append(GINConv(nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, hidden))))
            self.norms.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def embed(self, data):
        h = data.x
        for conv, norm in zip(self.convs, self.norms):
            h = F.relu(norm(conv(h, data.edge_index)))
            h = F.dropout(h, self.dropout, self.training)
        return global_mean_pool(h, data.batch)
    def forward(self, data):
        return self.head(self.embed(data)).squeeze(-1)

gin, _ = train_gnn(GIN(graphs[0].x.shape[1]), train_loader, val_loader, verbose=False)
print("GIN test RMSE", round(rmse(y[idx_test], predict_gnn(gin, test_loader)), 3))

# 2. sum vs mean readout: a *sum* is right for properties that grow with size (total energy, molar refractivity);
#    a *mean* for size-independent ones. logS is roughly extensive in the hydrophobic surface, so sum often helps.

# 3. depth
for L in [1, 3, 6]:
    m, _ = train_gnn(GCN(graphs[0].x.shape[1], n_layers=L), train_loader, val_loader, verbose=False)
    print(f"{L} layers: test RMSE", round(rmse(y[idx_test], predict_gnn(m, test_loader)), 3))
# 1 layer = each atom sees only its direct neighbours (like ECFP radius 1) - too local.
# 6 layers = every atom's representation mixes the whole molecule and they all become similar
# ("over-smoothing"), so performance stops improving or degrades. 3-5 rounds is the usual sweet spot.
```
</details>
"""


# %% [markdown]
"""
## 4. Chemical language models: pre-trained representations

Fingerprints are designed, GNN features are learned *for one task*. A third option: **transfer learning** — a large
transformer pre-trained on millions of SMILES (masked-token prediction, like BERT) provides an *embedding* for any molecule,
which we then feed to a simple model. This is the "foundation model" idea applied to chemistry.

We use **ChemBERTa-2** (Ahmad *et al.* 2022) from the Hugging Face Hub, in its `77M-MTR` variant. Read that name
carefully, because it is easy to misread: the 77 M counts the **pre-training molecules**, not the parameters. This is
a small model — just a few transformer layers with 384 hidden dimensions, a few million weights; the cell below
prints the exact count.
What it brings is not size but exposure to 77 million SMILES.
"""

# %% [markdown]
"""
**Downloading somebody else's pre-trained transformer.** `AutoTokenizer` and `AutoModel` fetch the tokenizer and the
weights from the Hugging Face Hub by name and cache them locally; `.eval()` puts the transformer in inference
mode, since we are going to use it frozen, as a feature extractor.

The whole thing sits in a `try/except` that sets `HF_OK`, because this is the one cell in the course that *must*
reach the internet — huggingface.co is blocked on some university and corporate networks (it was blocked in the
sandbox these notebooks were built in, so the printed message is the one you will see there). If it fails, the rest
of the section is skipped and nothing else in the notebook breaks. On Colab it works.
"""

# %%
try:
    from transformers import AutoTokenizer, AutoModel
    MODEL_ID = "DeepChem/ChemBERTa-77M-MTR"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    lm = AutoModel.from_pretrained(MODEL_ID).to(device).eval()
    HF_OK = True
    print("loaded", MODEL_ID, "|", sum(p.numel() for p in lm.parameters()) / 1e6, "M parameters")
except Exception as e:
    HF_OK = False
    print("Could not load the model:", str(e)[:150])
    print("\nThis needs internet access to huggingface.co. It works on Google Colab;"
          "\nif you are running elsewhere behind a firewall, skip this section.")

# %% [markdown]
"""
**Using a language model as a featuriser.** This cell does three things, and the middle one is the interesting one.

`embed_smiles` runs SMILES through the transformer in batches. `tok(..., padding=True, truncation=True)` produces
token ids plus an `attention_mask` marking real tokens against padding, and `last_hidden_state` is one 384-dimensional
vector **per token**. To get one vector per molecule we average over the tokens — but only over the real ones, which
is what `(h * mask).sum(1) / mask.sum(1)` computes. Averaging over the padding too would make a molecule's embedding
depend on how long its batch-mates happened to be.

Then `tok.tokenize("CC(=O)Oc1ccccc1C(=O)O")` prints the tokenisation, worth comparing with the hand-written regex
tokenizer of session 07 — this one was learned from data (byte-pair encoding) and merges frequent character
sequences into single tokens.

Finally the embeddings go into a **Ridge regression and a random forest**, and that is the entire point of transfer
learning: the expensive part was pre-training on 77 million molecules, which somebody else already paid for. We add
a linear model on top and get a competitive predictor from a dataset far too small to train a transformer on.
"""

# %%
if HF_OK:
    @torch.no_grad()
    def embed_smiles(smiles_list, batch_size=64):
        out = []
        for i in range(0, len(smiles_list), batch_size):
            enc = tok(list(smiles_list[i:i + batch_size]), padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            h = lm(**enc).last_hidden_state                                  # (batch, tokens, 384)
            mask = enc["attention_mask"].unsqueeze(-1)
            out.append(((h * mask).sum(1) / mask.sum(1)).cpu().numpy())      # mean over tokens
        return np.concatenate(out)

    print("SMILES tokens:", tok.tokenize("CC(=O)Oc1ccccc1C(=O)O"))
    t0 = time.time()
    E = embed_smiles(esol["smiles"].tolist())
    print("embeddings:", E.shape, f"({time.time() - t0:.0f} s)")

    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10)).fit(E[idx_trainval], y[idx_trainval])
    rf_e = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0).fit(E[idx_trainval], y[idx_trainval])
    for name, m in [("Ridge on ChemBERTa embeddings", ridge), ("RF on ChemBERTa embeddings", rf_e)]:
        p = m.predict(E[idx_test]); print(f"{name:32s} test RMSE = {rmse(y[idx_test], p):.3f}   R² = {r2_score(y[idx_test], p):.3f}")

# %% [markdown]
"""
> Fine-tuning the whole transformer on ESOL (instead of freezing it) typically gives another boost, at the cost of GPU time.
> The same recipe — pre-train on unlabelled molecules, adapt to a small labelled task — is behind most 2024–2026 "molecular
> foundation models" (MolFormer, CheMeleon, Uni-Mol, MoLFormer-XL …).

## 5. The state of the art in one command: Chemprop

**Chemprop** (Yang *et al.* 2019; Heid *et al.* 2024) implements the directed message-passing neural network (D-MPNN) that
won many property-prediction benchmarks and is used widely in industry. It has a command-line interface:

```bash
pip install chemprop
chemprop train --data-path esol.csv --task-type regression --smiles-columns smiles --target-columns logS \
               --split-type scaffold_balanced --epochs 50 --output-dir chemprop_esol
chemprop predict --test-path new_molecules.csv --model-paths chemprop_esol --preds-path predictions.csv
```

Pat Walters' `run_chemprop.ipynb` shows the Python API. For a small dataset expect RMSE ≈ 0.6 on ESOL — similar to the
best classical models of session 05, which is the honest message: **on ~1000 molecules, deep learning rarely beats a good
random forest**; it shines with 10⁴–10⁶ molecules, multi-task data, or when 3D/quantum information is included.
"""

# %% [markdown]
"""
## 6. Summary: which model when?

| situation | recommended first try |
|---|---|
| < 1 000 molecules, single endpoint | RF / XGBoost on descriptors + fingerprints (session 05) |
| 1 000 – 100 000 molecules | D-MPNN (Chemprop) or GIN with proper validation; compare with RF |
| many related endpoints (ADMET panel) | multi-task GNN / fine-tuned chemical language model |
| 3D-dependent property (binding, conformational energies) | 3D / equivariant GNNs (SchNet, DimeNet, e3nn; TeachOpenCADD T036) |
| very little data | pre-trained embeddings + linear model; transfer learning |

Always report a **baseline**, use a **scaffold split**, and estimate **uncertainty** (session 05).

## Exercises
1. Train the GCN for **EGFR classification** (binary cross-entropy: `F.binary_cross_entropy_with_logits`, ROC-AUC). Compare with the RF of session 05 on the same scaffold split.
2. Add **edge features** (bond type one-hot) using `torch_geometric.nn.NNConv` or `GINEConv`.
3. Data augmentation: train the MLP on fingerprints of *randomised* SMILES (`Chem.MolToSmiles(m, doRandom=True)`) — does it change anything? Why not? Then think about what randomised SMILES would change for a *sequence* model.
4. (Project) Fine-tune ChemBERTa end-to-end on ESOL with a regression head (`AutoModelForSequenceClassification`, `num_labels=1`).

## Further reading
- A. D. White, *Deep Learning for Molecules and Materials* — <https://dmol.pub> (free book; course bibliography).
- Gilmer *et al.*, *Neural message passing for quantum chemistry*, ICML 2017. Xu *et al.*, *How powerful are graph neural networks?* (GIN), ICLR 2019.
- Heid *et al.*, *Chemprop: a machine learning package for chemical property prediction*, J. Chem. Inf. Model. **2024**, 64, 9.
- Ahmad *et al.*, *ChemBERTa-2: towards chemical foundation models*, arXiv 2022.
- PyTorch Geometric tutorials: <https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html>.

Next session: **07 · Generative AI** — networks that *write* molecules.
"""
