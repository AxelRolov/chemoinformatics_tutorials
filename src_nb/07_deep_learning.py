# %% [markdown]
"""
# 07 · Deep learning for molecules: neural networks, graph neural networks and chemical language models

**Chemoinformatics (UFAZ, L2 S3) — Practical session 7**
Instructor: Alexey Orlov (Université de Strasbourg / UFAZ)

> ⚡ In Colab choose *Runtime → Change runtime type → T4 GPU* for faster training (everything also runs on CPU in a few minutes).

**Learning goals.** After this session you will be able to
- explain what a neural network is (layers, activations, loss, gradient descent, back-propagation) and train one with **PyTorch**;
- build a **multilayer perceptron** on fingerprints and compare it with the random forest of session 06;
- turn molecules into graphs and train a **graph neural network** (message passing) with PyTorch Geometric;
- use a pre-trained **chemical language model** (ChemBERTa) as a feature extractor — your first *foundation model*;
- recognise the pitfalls of deep learning on small chemical datasets (overfitting, need for baselines, splits).

---
> **Credits.** Adapted, with modifications for the UFAZ course, from
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
plt.figure(figsize=(8, 2.8)); plt.subplot(1, 2, 1); plt.plot(history); plt.xlabel("step"); plt.ylabel("loss")
plt.subplot(1, 2, 2); plt.scatter(x, y, s=8); plt.plot(x, (w * x + b).detach(), "r"); plt.xlabel("x"); plt.ylabel("y"); plt.show()

# %% [markdown]
"""
Everything else in deep learning is this loop with (i) a bigger function, (ii) a smarter optimiser (Adam), (iii) data fed in
**mini-batches**, and (iv) tricks against overfitting (dropout, weight decay, early stopping).

## 2. A multilayer perceptron on fingerprints

Same problem as session 06: predict ESOL solubility from Morgan fingerprints. We'll keep a **validation set** apart from
the test set to decide when to stop training.
"""

# %%
esol = pd.read_csv(data_path("esol_delaney.csv")).rename(columns={"measured log solubility in mols per litre": "logS"})
esol["mol"] = esol["smiles"].apply(Chem.MolFromSmiles)
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X_fp = np.array([fpgen.GetFingerprintAsNumPy(m) for m in esol["mol"]], dtype=np.float32)
y = esol["logS"].values.astype(np.float32)

idx_trainval, idx_test = train_test_split(np.arange(len(esol)), test_size=0.2, random_state=42)
idx_train, idx_val = train_test_split(idx_trainval, test_size=0.15, random_state=42)
print(len(idx_train), "train /", len(idx_val), "validation /", len(idx_test), "test")

# Standardise the target (helps optimisation); we un-standardise predictions later
y_mean, y_std = y[idx_train].mean(), y[idx_train].std()
y_norm = (y - y_mean) / y_std

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

t0 = time.time()
mlp, curve = train_mlp(MLP(2048), X_fp, y_norm, idx_train, idx_val)
print(f"{time.time() - t0:.0f} s")

# %%
def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        return model(to_tensor(X)).cpu().numpy() * y_std + y_mean

plt.figure(figsize=(5, 3)); plt.plot(curve[:, 0], label="train"); plt.plot(curve[:, 1], label="validation")
plt.xlabel("epoch"); plt.ylabel("MSE (standardised)"); plt.legend(); plt.title("Learning curves"); plt.show()

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
3. Train on **descriptors** instead of fingerprints (standardise them with `StandardScaler` first!). Compare with session 06.
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
# ~0.7, i.e. as good as the GCN and the boosted trees of session 06 - the representation, not the model, was the bottleneck.
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

# %%
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, global_add_pool

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
HYBRID = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]

def atom_features(atom):
    f = [int(atom.GetSymbol() == e) for e in ELEMENTS] + [int(atom.GetSymbol() not in ELEMENTS)]
    f += [int(atom.GetDegree() == d) for d in range(5)]
    f += [int(atom.GetTotalNumHs() == h) for h in range(4)]
    f += [int(atom.GetHybridization() == h) for h in HYBRID]
    f += [int(atom.GetIsAromatic()), int(atom.IsInRing()), atom.GetFormalCharge()]
    return f

def mol_to_graph(mol, y=None):
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    edge_index = torch.tensor(edges + [(j, i) for i, j in edges], dtype=torch.long).t().contiguous() if edges else torch.zeros((2, 0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float)
    return data

g = mol_to_graph(esol["mol"][0], y_norm[0])
print(g, "\nnode feature length:", g.x.shape[1])

# %%
graphs = [mol_to_graph(m, t) for m, t in zip(esol["mol"], y_norm)]
train_loader = DataLoader([graphs[i] for i in idx_train], batch_size=32, shuffle=True)
val_loader = DataLoader([graphs[i] for i in idx_val], batch_size=128)
test_loader = DataLoader([graphs[i] for i in idx_test], batch_size=128)
batch = next(iter(train_loader))
print(batch)                     # PyG concatenates the graphs of a batch into one big disconnected graph

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

t0 = time.time()
gcn, curve_g = train_gnn(GCN(graphs[0].x.shape[1]), train_loader, val_loader)
print(f"{time.time() - t0:.0f} s")
p_gcn = predict_gnn(gcn, test_loader)
print(f"GCN:  test RMSE = {rmse(y[idx_test], p_gcn):.3f}   R² = {r2_score(y[idx_test], p_gcn):.3f}")

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

We use **ChemBERTa-2** (Ahmad *et al.* 2022; 77 M parameters, pre-trained on 77 M PubChem SMILES) from the Hugging Face Hub.
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
best classical models of session 06, which is the honest message: **on ~1000 molecules, deep learning rarely beats a good
random forest**; it shines with 10⁴–10⁶ molecules, multi-task data, or when 3D/quantum information is included.
"""

# %% [markdown]
"""
## 6. Summary: which model when?

| situation | recommended first try |
|---|---|
| < 1 000 molecules, single endpoint | RF / XGBoost on descriptors + fingerprints (session 06) |
| 1 000 – 100 000 molecules | D-MPNN (Chemprop) or GIN with proper validation; compare with RF |
| many related endpoints (ADMET panel) | multi-task GNN / fine-tuned chemical language model |
| 3D-dependent property (binding, conformational energies) | 3D / equivariant GNNs (SchNet, DimeNet, e3nn; TeachOpenCADD T036) |
| very little data | pre-trained embeddings + linear model; transfer learning |

Always report a **baseline**, use a **scaffold split**, and estimate **uncertainty** (session 06).

## Exercises
1. Train the GCN for **EGFR classification** (binary cross-entropy: `F.binary_cross_entropy_with_logits`, ROC-AUC). Compare with the RF of session 06 on the same scaffold split.
2. Add **edge features** (bond type one-hot) using `torch_geometric.nn.NNConv` or `GINEConv`.
3. Data augmentation: train the MLP on fingerprints of *randomised* SMILES (`Chem.MolToSmiles(m, doRandom=True)`) — does it change anything? Why not? Then think about what randomised SMILES would change for a *sequence* model.
4. (Project) Fine-tune ChemBERTa end-to-end on ESOL with a regression head (`AutoModelForSequenceClassification`, `num_labels=1`).

## Further reading
- A. D. White, *Deep Learning for Molecules and Materials* — <https://dmol.pub> (free book; course bibliography).
- Gilmer *et al.*, *Neural message passing for quantum chemistry*, ICML 2017. Xu *et al.*, *How powerful are graph neural networks?* (GIN), ICLR 2019.
- Heid *et al.*, *Chemprop: a machine learning package for chemical property prediction*, J. Chem. Inf. Model. **2024**, 64, 9.
- Ahmad *et al.*, *ChemBERTa-2: towards chemical foundation models*, arXiv 2022.
- PyTorch Geometric tutorials: <https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html>.

Next session: **08 · Generative AI** — networks that *write* molecules.
"""
