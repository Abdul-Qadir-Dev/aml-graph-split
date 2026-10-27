# Shared by 02_baselines.py and 03_gnn.py so both score predictions the same way.
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

DATA_PATH = "data/processed/elliptic.npz"
SPLITS_PATH = "data/processed/splits.npz"
SEEDS = [0, 1, 2, 3, 4]

# Column 0 is the time step, 1-93 are the other local features, 94-165 the aggregated ones.
LOCAL_COLS = slice(0, 94)
ALL_COLS = slice(0, 166)


def load_data():
    d = np.load(DATA_PATH)
    return d["x"], d["y"], d["t"], d["edges"]


def load_split(kind: str, seed: int):
    s = np.load(SPLITS_PATH)
    key = "temporal" if kind == "temporal" else f"random_{seed}"
    return s[f"{key}_train"], s[f"{key}_val"], s[f"{key}_test"]


def best_threshold(y_val, p_val) -> float:
    # Scan every distinct score on validation and keep the one with the highest illicit F1.
    order = np.argsort(-p_val)
    p_sorted, y_sorted = p_val[order], y_val[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    f1 = 2 * tp / (tp + fp + y_val.sum())
    # Only cut where the score changes, so ties go together.
    last_of_tie = np.r_[p_sorted[1:] != p_sorted[:-1], True]
    f1 = np.where(last_of_tie, f1, -1)
    return float(p_sorted[np.argmax(f1)])


def scores(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    return {
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "auprc": average_precision_score(y, p),
    }


def per_step_scores(y, p, t, thr) -> list[dict]:
    rows = []
    for step in np.unique(t):
        m = t == step
        pred = (p[m] >= thr).astype(int)
        rows.append({
            "step": int(step),
            "n": int(m.sum()),
            "n_illicit": int(y[m].sum()),
            "n_pred_illicit": int(pred.sum()),
            "f1": f1_score(y[m], pred, zero_division=0) if y[m].sum() > 0 else np.nan,
            "recall": recall_score(y[m], pred, zero_division=0) if y[m].sum() > 0 else np.nan,
            "precision": precision_score(y[m], pred, zero_division=0) if pred.sum() > 0 else np.nan,
        })
    return rows
