import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn.models import GCN, GraphSAGE
from torch_geometric.utils import to_undirected

from common import ALL_COLS, LOCAL_COLS, SEEDS, best_threshold, load_data, load_split, per_step_scores, scores

RUNS_PATH = "results/gnn_runs.csv"
PER_STEP_PATH = "results/gnn_per_step.csv"
HIDDEN = 64
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 5e-4
MAX_EPOCHS = 1000
PATIENCE = 100
# Weber et al. weight the loss 0.7 illicit / 0.3 licit.
POS_WEIGHT = 0.7 / 0.3

MODELS = [("gcn", "local"), ("gcn", "all"), ("sage", "local"), ("sage", "all")]


def build_model(name, in_dim):
    cls = GCN if name == "gcn" else GraphSAGE
    return cls(in_dim, HIDDEN, num_layers=2, out_channels=1, dropout=DROPOUT)


def standardise(x, train_idx):
    mean = x[train_idx].mean(0, keepdims=True)
    std = x[train_idx].std(0, keepdims=True) + 1e-6
    return (x - mean) / std


def train_gnn(name, x, edge_index, y, tr, val, seed, device):
    torch.manual_seed(seed)
    model = build_model(name, x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    pos_weight = torch.tensor(POS_WEIGHT, device=device)
    y_tr = y[tr].float()
    y_val = y[val].cpu().numpy()

    best_ap, best_state, best_epoch = -1.0, None, 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(x, edge_index).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits[tr], y_tr, pos_weight=pos_weight)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            p_val = torch.sigmoid(model(x, edge_index).squeeze(-1)[val]).cpu().numpy()
        ap = average_precision_score(y_val, p_val)
        if ap > best_ap:
            best_ap, best_epoch = ap, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x, edge_index).squeeze(-1)).cpu().numpy()
    return p, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.use_deterministic_algorithms(True, warn_only=True)

    x_np, y_np, t, edges = load_data()
    edge_index = to_undirected(torch.from_numpy(edges).long()).to(device)
    y = torch.from_numpy(y_np.astype(np.int64)).to(device)
    feats = {"local": x_np[:, LOCAL_COLS], "all": x_np[:, ALL_COLS]}

    runs, steps = [], []
    for split in ["temporal", "random"]:
        for seed in args.seeds:
            tr, val, te = load_split(split, seed)
            for model, fs in MODELS:
                start = time.time()
                x = torch.from_numpy(standardise(feats[fs], tr)).to(device)
                p, best_epoch = train_gnn(model, x, edge_index, y, torch.from_numpy(tr).to(device),
                                          torch.from_numpy(val).to(device), seed, device)
                thr = best_threshold(y_np[val], p[val])
                res = scores(y_np[te], p[te], thr)
                key = {"model": model, "features": fs, "split": split, "seed": seed}
                runs.append({**key, "threshold": thr, "val_f1": scores(y_np[val], p[val], thr)["f1"],
                             "best_epoch": best_epoch, **res})
                if split == "temporal":
                    steps += [{**key, **r} for r in per_step_scores(y_np[te], p[te], t[te], thr)]
                print(f"{split} seed {seed} {model} {fs}: f1 {res['f1']:.3f} auprc {res['auprc']:.3f} "
                      f"(best epoch {best_epoch}, {time.time() - start:.0f}s)")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(runs).to_csv(RUNS_PATH, index=False)
    pd.DataFrame(steps).to_csv(PER_STEP_PATH, index=False)
    print(f"wrote {RUNS_PATH} and {PER_STEP_PATH}")


if __name__ == "__main__":
    main()
