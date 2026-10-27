import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PYG_ROOT = "data/raw/elliptic_pyg"
KAGGLE_DIR = "data/raw/elliptic"
OUT_DIR = "data/processed"
STEP_STATS_PATH = "results/step_stats.csv"
CSV_NAMES = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv"]

TRAIN_STEPS = (1, 29)
VAL_STEPS = (30, 34)
TEST_STEPS = (35, 49)
SEEDS = [0, 1, 2, 3, 4]


def find_raw_dir() -> str:
    # PyG is only used to fetch the CSVs; its processed tensor drops the time step column.
    try:
        from torch_geometric.datasets import EllipticBitcoinDataset
        return EllipticBitcoinDataset(PYG_ROOT).raw_dir
    except Exception as e:
        print(f"PyG download failed: {e}")
    if all(os.path.exists(os.path.join(KAGGLE_DIR, f)) for f in CSV_NAMES):
        return KAGGLE_DIR
    sys.exit(f"Put {', '.join(CSV_NAMES)} from Kaggle in {KAGGLE_DIR}/ and run again.")


def read_csvs(raw_dir: str):
    feats = pd.read_csv(os.path.join(raw_dir, CSV_NAMES[0]), header=None)
    classes = pd.read_csv(os.path.join(raw_dir, CSV_NAMES[1]))
    edges = pd.read_csv(os.path.join(raw_dir, CSV_NAMES[2]))

    tx_ids = feats[0].to_numpy()
    x = feats.iloc[:, 1:].to_numpy(dtype=np.float32)  # time step + 165 features
    t = feats[1].to_numpy().astype(np.int16)

    label_map = {"1": 1, "2": 0, "unknown": -1}
    classes = classes.set_index("txId")["class"].astype(str).map(label_map)
    y = classes.loc[tx_ids].to_numpy().astype(np.int8)

    index = pd.Series(np.arange(len(tx_ids)), index=tx_ids)
    edge_index = np.stack([index.loc[edges["txId1"]].to_numpy(), index.loc[edges["txId2"]].to_numpy()])
    return x, y, t, edge_index


def in_steps(t, steps):
    return (t >= steps[0]) & (t <= steps[1])


def make_splits(y, t) -> dict:
    labelled = y >= 0
    splits = {
        "temporal_train": np.where(labelled & in_steps(t, TRAIN_STEPS))[0],
        "temporal_val": np.where(labelled & in_steps(t, VAL_STEPS))[0],
        "temporal_test": np.where(labelled & in_steps(t, TEST_STEPS))[0],
    }
    n_val, n_test = len(splits["temporal_val"]), len(splits["temporal_test"])
    idx = np.where(labelled)[0]
    for seed in SEEDS:
        rest, test = train_test_split(idx, test_size=n_test, stratify=y[idx], random_state=seed)
        train, val = train_test_split(rest, test_size=n_val, stratify=y[rest], random_state=seed)
        splits[f"random_{seed}_train"] = np.sort(train)
        splits[f"random_{seed}_val"] = np.sort(val)
        splits[f"random_{seed}_test"] = np.sort(test)
    return splits


def step_stats(y, t) -> pd.DataFrame:
    df = pd.DataFrame({"step": t, "y": y})
    out = df.groupby("step")["y"].agg(
        n="size", n_labelled=lambda s: (s >= 0).sum(), n_illicit=lambda s: (s == 1).sum()
    )
    out["illicit_share_of_labelled"] = out["n_illicit"] / out["n_labelled"]
    return out.reset_index()


def main():
    raw_dir = find_raw_dir()
    print(f"reading CSVs from {raw_dir}")
    x, y, t, edges = read_csvs(raw_dir)
    print(f"nodes {len(y)}, edges {edges.shape[1]}, features {x.shape[1]}")
    print(f"illicit {(y == 1).sum()}, licit {(y == 0).sum()}, unknown {(y == -1).sum()}")

    cross = (t[edges[0]] != t[edges[1]]).sum()
    print(f"edges crossing time steps: {cross}")
    print(f"self loops: {(edges[0] == edges[1]).sum()}")

    splits = make_splits(y, t)
    for k in ["temporal_train", "temporal_val", "temporal_test", "random_0_train", "random_0_val", "random_0_test"]:
        idx = splits[k]
        print(f"{k}: {len(idx)} nodes, {y[idx].mean():.3f} illicit")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "elliptic.npz"), x=x, y=y, t=t, edges=edges)
    np.savez_compressed(os.path.join(OUT_DIR, "splits.npz"), **splits)
    os.makedirs(os.path.dirname(STEP_STATS_PATH), exist_ok=True)
    step_stats(y, t).to_csv(STEP_STATS_PATH, index=False)
    print(f"saved arrays to {OUT_DIR} and step counts to {STEP_STATS_PATH}")


if __name__ == "__main__":
    main()
