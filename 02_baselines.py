import argparse
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import ALL_COLS, LOCAL_COLS, SEEDS, best_threshold, load_data, load_split, per_step_scores, scores

RUNS_PATH = "results/baselines_runs.csv"
PER_STEP_PATH = "results/baselines_per_step.csv"
N_THREADS = 6


def neighbour_mean(x_local, edges, n):
    # Mean of each node's neighbours' local features, ignoring edge direction.
    # Leaves out the time step column, since neighbours always share it.
    a = sp.coo_matrix((np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(n, n))
    a = ((a + a.T) > 0).astype(np.float32).tocsr()
    deg = np.asarray(a.sum(axis=1)).ravel()
    out = a @ x_local[:, 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = out / deg[:, None]
    out[deg == 0] = np.nan  # LightGBM treats these as missing
    return out.astype(np.float32)


def fit_logreg(x_tr, y_tr, x_val, x_te, seed):
    scaler = StandardScaler().fit(x_tr)
    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(scaler.transform(x_tr), y_tr)
    return model.predict_proba(scaler.transform(x_val))[:, 1], model.predict_proba(scaler.transform(x_te))[:, 1]


def fit_lgbm(x_tr, y_tr, x_val, y_val, x_te, seed):
    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=31,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=seed, n_jobs=N_THREADS, deterministic=True, force_row_wise=True, verbose=-1,
    )
    model.fit(
        x_tr, y_tr, eval_set=[(x_val, y_val)], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)],
    )
    return model.predict_proba(x_val)[:, 1], model.predict_proba(x_te)[:, 1]


def feature_sets(x, edges):
    nbr = neighbour_mean(x[:, LOCAL_COLS], edges, len(x))
    return {
        "local": x[:, LOCAL_COLS],
        "all": x[:, ALL_COLS],
        "local+nbr": np.hstack([x[:, LOCAL_COLS], nbr]),
        "all+nbr": np.hstack([x[:, ALL_COLS], nbr]),
    }


MODELS = [
    ("logreg", "local"),
    ("lgbm", "local"),
    ("lgbm", "all"),
    ("lgbm", "local+nbr"),
    ("lgbm", "all+nbr"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=["temporal", "random"])
    args = parser.parse_args()

    x, y, t, edges = load_data()
    feats = feature_sets(x, edges)
    runs, steps = [], []
    for split in args.splits:
        for seed in SEEDS:
            tr, val, te = load_split(split, seed)
            for model, fs in MODELS:
                xf = feats[fs]
                if model == "logreg":
                    p_val, p_te = fit_logreg(xf[tr], y[tr], xf[val], xf[te], seed)
                else:
                    p_val, p_te = fit_lgbm(xf[tr], y[tr], xf[val], y[val], xf[te], seed)
                thr = best_threshold(y[val], p_val)
                res = scores(y[te], p_te, thr)
                key = {"model": model, "features": fs, "split": split, "seed": seed}
                runs.append({**key, "threshold": thr, "val_f1": scores(y[val], p_val, thr)["f1"], **res})
                if split == "temporal":
                    steps += [{**key, **r} for r in per_step_scores(y[te], p_te, t[te], thr)]
                print(f"{split} seed {seed} {model} {fs}: f1 {res['f1']:.3f} auprc {res['auprc']:.3f}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(runs).to_csv(RUNS_PATH, index=False)
    pd.DataFrame(steps).to_csv(PER_STEP_PATH, index=False)
    print(f"wrote {RUNS_PATH} and {PER_STEP_PATH}")


if __name__ == "__main__":
    main()
