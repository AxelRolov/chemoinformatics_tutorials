"""Pre-train the SMILES LSTM used in notebook 08 on the 50k ZINC subset shipped in data/.

    python utils/pretrain_smiles_lstm.py --epochs 6 --out models/smiles_lstm_zinc50k.pt

The model/tokenizer code is identical to the one in notebooks/08_generative_ai.ipynb so the checkpoint loads there.
On a GPU this takes a few minutes; on CPU roughly 5 min per epoch.
"""
import argparse, re, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

SMILES_REGEX = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")
SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize(smiles):
    return SMILES_REGEX.findall(smiles)


class Vocab:
    def __init__(self, tokens):
        self.itos = SPECIAL + sorted(set(tokens) - set(SPECIAL))
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad, self.bos, self.eos, self.unk = 0, 1, 2, 3

    def __len__(self):
        return len(self.itos)

    def encode(self, smiles):
        return [self.bos] + [self.stoi.get(t, self.unk) for t in tokenize(smiles)] + [self.eos]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.eos:
                break
            if i not in (self.pad, self.bos):
                out.append(self.itos[i])
        return "".join(out)


class SmilesLSTM(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=512, layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x, state=None):
        h, state = self.lstm(self.embedding(x), state)
        return self.out(h), state


def pad_batch(seqs, pad=0):
    L = max(len(s) for s in seqs)
    return torch.tensor([s + [pad] * (L - len(s)) for s in seqs], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/zinc_50k.csv")
    ap.add_argument("--out", default="models/smiles_lstm_zinc50k.pt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    smiles = pd.read_csv(args.data)["smiles"].tolist()
    vocab = Vocab([t for s in smiles for t in tokenize(s)])
    data = [vocab.encode(s) for s in smiles]
    print(f"{len(data)} SMILES, vocab {len(vocab)}, device {device}")

    model = SmilesLSTM(len(vocab)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * math.ceil(len(data) / args.batch))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(len(data))
        tot, n, t0 = 0.0, 0, time.time()
        for i in range(0, len(data), args.batch):
            batch = pad_batch([data[j] for j in perm[i:i + args.batch]]).to(device)
            logits, _ = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item() * batch.size(0); n += batch.size(0)
            if (i // args.batch) % 50 == 0:
                print(f"  epoch {epoch} batch {i // args.batch}: loss {loss.item():.3f}", flush=True)
        print(f"epoch {epoch}: mean loss {tot / n:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        torch.save({"state_dict": model.state_dict(), "vocab": vocab.itos,
                    "config": {"emb": 128, "hidden": 512, "layers": 2, "dropout": 0.2},
                    "training": {"data": "zinc_50k.csv", "epochs_done": epoch + 1, "loss": tot / n}}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
