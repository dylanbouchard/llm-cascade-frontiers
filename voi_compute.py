"""
voi_compute.py - Appendix A1: Value of UQ Information.

For each (dataset, pair, scorer) cell, computes:
  - AUROC of confidence score predicting cheap model correctness (test set)
  - AUROC of low confidence predicting positive escalation benefit (test set)
  - Normalized cascade gain over a no-signal random-escalation baseline

Outputs per dataset (in results/voi/{dataset}/):
  results.parquet   - [pair, scorer, split, cheap_auroc, benefit_auroc, gain]
  summary.csv       - median AUROC/gain per (pair, scorer) cell

Combined figure:
  figures/figA1_voi_scatter.pdf   - scorer choice ablation by dataset

Usage:
    python3 voi_compute.py [dataset ...]
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "llm_cascade_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from optuna_frontier import PRICE_PER_TOKEN, compute_costs, DATA_DIR
from fig2_compute import (
    non_dominated_models, make_split, slice_data, valid_pairs_from_data,
    CACHE_VERSION,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_DIR  = Path("results/voi")
FIG_DIR  = Path("figures")
N_SPLITS = 50
N_THRESH = 200
SEED     = 42
VOI_CACHE_VERSION = CACHE_VERSION + "-benefit-auc-v1"

DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
DATASET_LABELS = {
    "mmlu":          "MMLU",
    "triviaqa":      "TriviaQA",
    "math_hard":     "MATH",
    "simpleqa":      "SimpleQA",
    "livecodebench": "LiveCodeBench",
}

DATASET_EXCLUSIONS: dict[str, list[str]] = {
    "livecodebench": ["gpt-oss-20b"],
}

BASE_SCORERS = [
    "mean_token_negentropy",
    "min_token_negentropy",
    "probability_margin",
    "min_probability",
    "sequence_probability",
]
ALL_SCORERS = BASE_SCORERS + ["logreg_ensemble"]

SCORER_COLORS = {
    "mean_token_negentropy": "#4878CF",
    "min_token_negentropy":  "#6ACC65",
    "probability_margin":    "#D65F5F",
    "min_probability":       "#B47CC7",
    "sequence_probability":  "#C4AD66",
    "logreg_ensemble":       "#1A1A1A",
}

SCORER_LABELS = {
    "mean_token_negentropy": "Mean neg.",
    "min_token_negentropy":  "Min neg.",
    "probability_margin":    "Margin",
    "min_probability":       "Min prob.",
    "sequence_probability":  "Seq. prob.",
    "logreg_ensemble":       "LogReg",
}

MODEL_COST_ORDER = sorted(PRICE_PER_TOKEN, key=lambda m: sum(PRICE_PER_TOKEN[m]))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_full_multi(dataset: str) -> tuple[dict, dict]:
    """Load all models for dataset, keeping all 5 UQ scorer columns."""
    excluded = DATASET_EXCLUSIONS.get(dataset, [])
    raw, costs = {}, {}
    for m in MODEL_COST_ORDER:
        if m in excluded:
            continue
        path = DATA_DIR / f"{dataset}-{m}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).iloc[:2000].reset_index(drop=True)
        required = BASE_SCORERS + ["correct"]
        if not all(c in df.columns for c in required):
            continue
        raw[m]   = df
        costs[m] = compute_costs(df, m)
    return raw, costs


# ---------------------------------------------------------------------------
# Frontier and gain computation
# ---------------------------------------------------------------------------

def compute_cascade_frontier(
    s: np.ndarray,
    u_L: np.ndarray,
    u_H: np.ndarray,
    c_L_per: np.ndarray,
    c_H_per: np.ndarray,
    n_grid: int = N_THRESH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sweep threshold grid over scorer values.
    Returns (E[C(tau)], E[U(tau)]) arrays of length n_grid.
    Cascade cost: every query pays c_L; escalated queries also pay c_H.
    """
    tau_grid = np.linspace(s.min(), s.max(), n_grid)
    cs = np.empty(n_grid)
    qs = np.empty(n_grid)
    for i, tau in enumerate(tau_grid):
        escalate = s < tau
        qs[i] = np.where(escalate, u_H, u_L).mean()
        cs[i] = c_L_per.mean() + (escalate * c_H_per).mean()
    return cs, qs


def compute_gain(
    costs: np.ndarray,
    quality: np.ndarray,
    a_L: float,
    a_H: float,
    c_L: float,
    c_H_total: float,
) -> float:
    """
    Normalized area between cascade frontier and random-escalation diagonal.

    Diagonal: linear interpolation from (c_L, a_L) to (c_H_total, a_H),
    where c_H_total = c_L + E[C_H].
    Gain = area_above / ((c_H_total - c_L) * (a_H - a_L)).
    """
    rect_area = (c_H_total - c_L) * (a_H - a_L)
    if rect_area <= 0:
        return 0.0

    # Sort by cost and clip to valid cost range
    order = np.argsort(costs)
    c_s = costs[order]
    q_s = quality[order]

    mask = (c_s >= c_L - 1e-12) & (c_s <= c_H_total + 1e-12)
    c_s = c_s[mask]
    q_s = q_s[mask]

    if len(c_s) < 2:
        return 0.0

    diagonal = a_L + (a_H - a_L) * (c_s - c_L) / (c_H_total - c_L)
    area_above = float(np.trapz(q_s - diagonal, c_s))
    return area_above / rect_area


# ---------------------------------------------------------------------------
# Per-dataset computation
# ---------------------------------------------------------------------------

def run_dataset(dataset: str) -> pd.DataFrame:
    print(f"\n{'='*60}\n  {dataset}\n{'='*60}")
    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "results.parquet"
    version_path = out_dir / "methodology_version.txt"
    cache_current = (
        version_path.exists()
        and version_path.read_text().strip() == VOI_CACHE_VERSION
    )
    if cache_current and result_path.exists():
        print("  Already computed - skipping.")
        return pd.read_parquet(result_path)

    raw, costs = load_full_multi(dataset)

    # Filter rows where any available model has NaN in any required column.
    # Accuracy-based model/pair choices are made per split using calibration
    # labels only.
    valid_mask = np.ones(len(next(iter(raw.values()))), dtype=bool)
    for m in raw:
        valid_mask &= raw[m]["correct"].notna().values
        for sc in BASE_SCORERS:
            valid_mask &= raw[m][sc].notna().values
    valid_idx = np.where(valid_mask)[0]
    for m in raw:
        raw[m]   = raw[m].iloc[valid_idx].reset_index(drop=True)
        costs[m] = costs[m][valid_idx]
    n = len(valid_idx)
    print(f"  Valid rows after NaN filter: {n} / {valid_mask.size}")

    # Reference correct array for stratified splits (cheapest model)
    ref_model = next(iter(raw))
    ref_correct = raw[ref_model]["correct"].values.astype(int)

    records = []

    for split_i in range(N_SPLITS):
        cal_idx, test_idx = make_split(n, ref_correct, seed=SEED + split_i)
        split_models = non_dominated_models(raw, costs, idx=cal_idx)
        if len(split_models) < 2:
            continue
        cal_data = slice_data(raw, costs, split_models, cal_idx)
        pairs = valid_pairs_from_data(cal_data, split_models)

        for cheap, exp in pairs:
            # Calibration data for LR training
            df_cheap_cal = raw[cheap].iloc[cal_idx]
            X_cal = df_cheap_cal[BASE_SCORERS].values.astype(float)
            y_cal = df_cheap_cal["correct"].values.astype(int)

            # Test data
            df_cheap_test = raw[cheap].iloc[test_idx].copy()
            u_H           = raw[exp].iloc[test_idx]["correct"].values.astype(float)
            c_L_per       = costs[cheap][test_idx]
            c_H_per       = costs[exp][test_idx]

            u_L     = df_cheap_test["correct"].values.astype(float)
            y_test  = u_L  # binary correctness of cheap model on test
            a_L     = u_L.mean()
            a_H     = u_H.mean()
            c_L     = c_L_per.mean()
            c_H_total = c_L + c_H_per.mean()

            if a_H <= a_L or c_H_total <= c_L:
                continue

            # Train logistic regression on calibration set
            if len(np.unique(y_cal)) < 2:
                # Degenerate calibration set - ensemble falls back to mean negentropy
                df_cheap_test = df_cheap_test.copy()
                df_cheap_test["logreg_ensemble"] = \
                    df_cheap_test["mean_token_negentropy"].values
            else:
                lr = LogisticRegression(max_iter=1000, random_state=SEED)
                lr.fit(X_cal, y_cal)
                X_test = df_cheap_test[BASE_SCORERS].values.astype(float)
                df_cheap_test = df_cheap_test.copy()
                df_cheap_test["logreg_ensemble"] = lr.predict_proba(X_test)[:, 1]

            for scorer in ALL_SCORERS:
                s = df_cheap_test[scorer].values.astype(float)

                # AUROC for cheap-model correctness: high score means likely correct.
                if len(np.unique(y_test)) < 2:
                    cheap_auroc = 0.5
                else:
                    try:
                        cheap_auroc = float(roc_auc_score(y_test, s))
                    except Exception:
                        cheap_auroc = 0.5

                # AUROC for positive escalation benefit: low score should rank
                # queries where H is correct and L is incorrect above others.
                y_benefit = (u_H > u_L).astype(int)
                if len(np.unique(y_benefit)) < 2:
                    benefit_auroc = 0.5
                else:
                    try:
                        benefit_auroc = float(roc_auc_score(y_benefit, -s))
                    except Exception:
                        benefit_auroc = 0.5

                # Cascade frontier + gain
                cs, qs = compute_cascade_frontier(s, u_L, u_H, c_L_per, c_H_per)
                gain   = compute_gain(cs, qs, a_L, a_H, c_L, c_H_total)

                records.append({
                    "pair":   f"{cheap}→{exp}",
                    "scorer": scorer,
                    "split":  split_i,
                    "cheap_auroc": cheap_auroc,
                    "benefit_auroc": benefit_auroc,
                    "gain":   gain,
                })

        if (split_i + 1) % 10 == 0:
            print(f"  split {split_i+1}/{N_SPLITS}", flush=True)

    df = pd.DataFrame(records)
    df.to_parquet(result_path, index=False)
    version_path.write_text(VOI_CACHE_VERSION + "\n")

    # Summary: median and percentiles per (pair, scorer) cell
    def pct10(x): return float(np.percentile(x, 10))
    def pct90(x): return float(np.percentile(x, 90))

    summary = (
        df.groupby(["pair", "scorer"])
              .agg(
              median_cheap_auroc=("cheap_auroc", "median"),
              p10_cheap_auroc=("cheap_auroc", pct10),
              p90_cheap_auroc=("cheap_auroc", pct90),
              median_benefit_auroc=("benefit_auroc", "median"),
              p10_benefit_auroc=("benefit_auroc", pct10),
              p90_benefit_auroc=("benefit_auroc", pct90),
              median_gain=("gain", "median"),
              p10_gain=("gain", pct10),
              p90_gain=("gain", pct90),
          )
          .reset_index()
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(f"  Done. {N_SPLITS} splits x {len(ALL_SCORERS)} scorers"
          f" = {len(df)} records.")
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_scorer_ablation(all_results: dict[str, pd.DataFrame]):
    """
    1x5 scorer-choice ablation. Each point is the median gain across valid
    pair cells for a scorer; vertical bars show the 10th--90th percentile
    across pair cells.
    """
    fig, axes = plt.subplots(1, 5, figsize=(8.5, 2.5), sharey=True)
    x = np.arange(len(ALL_SCORERS))

    for ax, dataset in zip(axes, DATASETS):
        if dataset not in all_results:
            ax.set_visible(False)
            continue

        df = all_results[dataset]
        summary = (
            df.groupby(["pair", "scorer"])
              .agg(
                  median_benefit_auroc=("benefit_auroc", "median"),
                  median_gain=("gain", "median"),
              )
              .reset_index()
        )

        med, lo, hi, colors = [], [], [], []
        for scorer in ALL_SCORERS:
            sub = summary[summary["scorer"] == scorer]
            vals = sub["median_gain"].values
            med.append(float(np.median(vals)))
            lo.append(float(np.percentile(vals, 10)))
            hi.append(float(np.percentile(vals, 90)))
            colors.append(SCORER_COLORS[scorer])

        yerr = np.vstack([np.array(med) - np.array(lo), np.array(hi) - np.array(med)])
        ax.errorbar(
            x, med, yerr=yerr, fmt="none", ecolor="0.65",
            elinewidth=0.8, capsize=2, zorder=1,
        )
        ax.scatter(x, med, color=colors, s=28, edgecolors="white",
                   linewidths=0.4, zorder=2)
        ax.axhline(0, color="0.75", lw=0.6, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels([SCORER_LABELS[s] for s in ALL_SCORERS],
                           rotation=35, ha="right", fontsize=6.2)
        ax.set_ylabel("Gain over no-signal cascade", fontsize=8)
        ax.set_title(DATASET_LABELS[dataset], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.8)
    out_path = FIG_DIR / "figA1_voi_scatter.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out_path}")


def print_summary_table(all_results: dict[str, pd.DataFrame]):
    """Print scorer x dataset table of median gain (aggregated across pairs)."""
    rows = []
    for scorer in ALL_SCORERS:
        row = {"scorer": SCORER_LABELS[scorer]}
        for dataset in DATASETS:
            if dataset not in all_results:
                row[dataset] = "-"
                continue
            df = all_results[dataset]
            sub = df[df["scorer"] == scorer]
            agg = sub.groupby("pair")["gain"].median().mean()
            auroc_agg = sub.groupby("pair")["benefit_auroc"].median().mean()
            row[dataset] = f"{agg:.3f} ({auroc_agg:.2f})"
        rows.append(row)

    table = pd.DataFrame(rows).set_index("scorer")
    table.columns = [DATASET_LABELS[d] for d in DATASETS]
    print("\nMedian gain (median benefit-AUROC) per scorer x dataset, mean across pairs:")
    print(table.to_string())

    # Spearman r per dataset
    print("\nSpearman r(benefit-AUROC, gain) per dataset:")
    for dataset in DATASETS:
        if dataset not in all_results:
            continue
        df = all_results[dataset]
        summary = (
            df.groupby(["pair", "scorer"])
              .agg(
                  median_benefit_auroc=("benefit_auroc", "median"),
                  median_gain=("gain", "median"),
              )
              .reset_index()
        )
        pooled_r, pooled_p = spearmanr(summary["median_benefit_auroc"], summary["median_gain"])
        summary["benefit_auroc_dm"] = (
            summary["median_benefit_auroc"]
            - summary.groupby("pair")["median_benefit_auroc"].transform("mean")
        )
        summary["gain_dm"] = (
            summary["median_gain"]
            - summary.groupby("pair")["median_gain"].transform("mean")
        )
        within_r, within_p = spearmanr(summary["benefit_auroc_dm"], summary["gain_dm"])
        print(
            f"  {DATASET_LABELS[dataset]}: pooled r_s={pooled_r:.3f}, "
            f"pair-demeaned r_s={within_r:.3f}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = sys.argv[1:] or DATASETS
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for ds in datasets:
        all_results[ds] = run_dataset(ds)

    print_summary_table(all_results)
    plot_scorer_ablation(all_results)
    print("\nDone.")
