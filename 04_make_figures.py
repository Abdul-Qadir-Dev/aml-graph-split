import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

RUN_FILES = ["results/baselines_runs.csv", "results/gnn_runs.csv"]
STEP_FILES = ["results/baselines_per_step.csv", "results/gnn_per_step.csv"]
STEP_STATS = "results/step_stats.csv"
TABLE_PATH = "results/main_table.csv"
FIG_DIR = "figures"
TEST_START = 35

ORDER = ["logreg local", "lgbm local", "lgbm all", "lgbm local+nbr", "lgbm all+nbr",
         "gcn local", "gcn all", "sage local", "sage all"]
# Series shown in the per-step plot, one per model family.
STEP_MODELS = ["logreg local", "lgbm all", "lgbm all+nbr", "gcn all", "sage all"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(files):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["name"] = df["model"] + " " + df["features"]
    return df


def main_table(runs):
    agg = runs.groupby(["name", "split"])[["f1", "auprc"]].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    wide = agg.unstack("split")
    wide.columns = [f"{split}_{col}" for col, split in wide.columns]
    wide = wide.reindex(ORDER)
    wide["f1_gap"] = wide["random_f1_mean"] - wide["temporal_f1_mean"]
    wide["auprc_gap"] = wide["random_auprc_mean"] - wide["temporal_auprc_mean"]
    cols = ["temporal_f1_mean", "temporal_f1_std", "temporal_auprc_mean", "temporal_auprc_std",
            "random_f1_mean", "random_f1_std", "random_auprc_mean", "random_auprc_std",
            "f1_gap", "auprc_gap"]
    return wide[cols].round(4).rename_axis("model").reset_index()


def style(ax):
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_per_step(steps, stats, path):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.2]})
    for name, color in zip(STEP_MODELS, COLORS):
        g = steps[steps["name"] == name].groupby("step")["f1"]
        mean, std = g.mean(), g.std().fillna(0)
        ax.plot(mean.index, mean.values, color=color, linewidth=2, marker="o", markersize=4, label=name)
        ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
    ax.axvline(43, color=MUTED, linestyle="--", linewidth=1)
    ax.text(43.15, 0.98, "step 43", color=MUTED, fontsize=9, va="top")
    ax.set_ylim(0, 1)
    ax.set_ylabel("illicit F1", color=INK)
    ax.set_title("Illicit F1 per test time step (temporal split, mean of 5 seeds, band = 1 sd)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower left")
    style(ax)

    test = stats[stats["step"] >= TEST_START]
    ax2.bar(test["step"], test["n_illicit"], color=MUTED, width=0.7)
    for s, n in zip(test["step"], test["n_illicit"]):
        ax2.text(s, n + 4, str(n), ha="center", fontsize=8, color=INK)
    ax2.axvline(43, color=MUTED, linestyle="--", linewidth=1)
    ax2.set_ylabel("illicit labels", color=INK)
    ax2.set_xlabel("time step", color=INK)
    ax2.set_xticks(test["step"])
    style(ax2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gap(table, path):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = range(len(table))
    w = 0.38
    ax.bar([i - w / 2 for i in x], table["temporal_f1_mean"], w, yerr=table["temporal_f1_std"],
           color=COLORS[0], label="temporal split", error_kw={"ecolor": INK, "lw": 1})
    ax.bar([i + w / 2 for i in x], table["random_f1_mean"], w, yerr=table["random_f1_std"],
           color=COLORS[1], label="random split", error_kw={"ecolor": INK, "lw": 1})
    ax.set_xticks(list(x))
    ax.set_xticklabels(table["model"], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("illicit F1 on test", color=INK)
    ax.set_title("Random vs temporal split, illicit F1 (mean of 5 seeds, bars = 1 sd)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    runs = load(RUN_FILES)
    steps = load(STEP_FILES)
    stats = pd.read_csv(STEP_STATS)
    table = main_table(runs)
    table.to_csv(TABLE_PATH, index=False)
    print(table.to_string(index=False))

    os.makedirs(FIG_DIR, exist_ok=True)
    plot_per_step(steps, stats, os.path.join(FIG_DIR, "f1_per_step.png"))
    plot_gap(table, os.path.join(FIG_DIR, "random_vs_temporal.png"))
    print(f"wrote {TABLE_PATH} and figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
