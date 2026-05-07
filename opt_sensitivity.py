"""
opt_sensitivity.py - Optimizer sensitivity for k-cascade comparison.

Tests whether the near-zero held-out k-cascade gaps depend on NSGA-II.
Compares three optimizers (NSGA-II, random search, TPE)
against the pairwise envelope across 50 splits on all four datasets.

TPE is scalarized: 20 weights x 100 trials = 2,000 trials total, matching
NSGA-II and random search budgets.

Outputs:
  results/optsens/{dataset}_{optimizer}_curves.npy  - (50, 500)
  results/optsens/{dataset}_envelope_curves.npy     - (50, 500)
  results/optsens/summary.csv
  figures/figA_opt_sensitivity.pdf

Usage:
    python3 opt_sensitivity.py [dataset ...]
"""

import sys, json, time
import os
import tempfile
import numpy as np
import pandas as pd
import optuna
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "llm_cascade_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d

from fig2_compute import (
    load_full, non_dominated_models, make_split, slice_data,
    make_objective_nd, pairwise_envelope_curve, valid_pairs_from_data,
    N_GRID, POP_SIZE, SEED, MAX_K, CACHE_VERSION,
)
from optuna_frontier import simulate_cascade

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR  = Path("results/optsens")
FIG_DIR  = Path("figures")
DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
DATASET_LABELS = {"mmlu": "MMLU", "triviaqa": "TriviaQA",
                  "math_hard": "MATH", "simpleqa": "SimpleQA",
                  "livecodebench": "LiveCodeBench"}

OPTIMIZERS  = ["nsga2", "random"]
N_SPLITS    = 50
N_TRIALS    = 2000
MID_IDX     = N_GRID // 2

OPT_STYLES = {
    "nsga2":  {"color": "coral",   "ls": "--", "lw": 1.2, "label": "NSGA-II"},
    "random": {"color": "#6ACC65", "ls": ":",  "lw": 1.2, "label": "Random"},
}


# ---------------------------------------------------------------------------
# Optimizer runners - each returns list of (sequence, thresholds) configs
# ---------------------------------------------------------------------------

def _extract_configs(pareto_trials, seqs, score_ranges):
    configs = []
    for t in pareto_trials:
        seq  = seqs[t.params["seq_idx"]]
        k    = len(seq)
        tau_norms  = [t.params[f"tau_{i}"] for i in range(MAX_K - 1)]
        thresholds = [
            score_ranges[seq[i]][0]
            + tau_norms[i] * (score_ranges[seq[i]][1] - score_ranges[seq[i]][0])
            for i in range(k - 1)
        ]
        configs.append((list(seq), thresholds))
    return configs


def run_nsga2(cal_data, models, score_ranges, split_seed):
    obj, seqs = make_objective_nd(cal_data, models, score_ranges)
    sampler   = optuna.samplers.NSGAIISampler(population_size=POP_SIZE,
                                              seed=split_seed)
    study = optuna.create_study(directions=["minimize", "minimize"],
                                sampler=sampler)
    study.optimize(obj, n_trials=N_TRIALS)
    return _extract_configs(study.best_trials, seqs, score_ranges)


def run_random(cal_data, models, score_ranges, split_seed):
    obj, seqs = make_objective_nd(cal_data, models, score_ranges)
    sampler   = optuna.samplers.RandomSampler(seed=split_seed)
    study = optuna.create_study(directions=["minimize", "minimize"],
                                sampler=sampler)
    study.optimize(obj, n_trials=N_TRIALS)
    return _extract_configs(study.best_trials, seqs, score_ranges)


def run_tpe(cal_data, models, score_ranges, split_seed, cost_scale: float = 1.0):
    """
    Single-objective TPE with scalarization sweep.
    Each of TPE_WEIGHTS weights gets N_TRIALS // TPE_WEIGHTS trials.
    Returns the best config per weight (contributing one test point each).

    cost_scale normalises c to [0, 1] so the weight sweep spans the frontier
    rather than being dominated by the quality term (which is O(1) while raw
    costs are O(1e-6)).
    """
    obj_nd, seqs        = make_objective_nd(cal_data, models, score_ranges)
    trials_per_w        = N_TRIALS // TPE_WEIGHTS
    weights             = np.linspace(0.05, 2.0, TPE_WEIGHTS)
    configs             = []

    for w in weights:
        def scal(trial, _w=w):
            c, neg_q = obj_nd(trial)
            return c / cost_scale + _w * neg_q   # both terms O(1)

        sampler = optuna.samplers.TPESampler(seed=split_seed)
        study   = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(scal, n_trials=trials_per_w)

        best = study.best_trial
        seq  = seqs[best.params["seq_idx"]]
        k    = len(seq)
        tau_norms  = [best.params[f"tau_{i}"] for i in range(MAX_K - 1)]
        thresholds = [
            score_ranges[seq[i]][0]
            + tau_norms[i] * (score_ranges[seq[i]][1] - score_ranges[seq[i]][0])
            for i in range(k - 1)
        ]
        configs.append((list(seq), thresholds))

    return configs


# ---------------------------------------------------------------------------
# Evaluate configs on test and extract Pareto frontier curve
# ---------------------------------------------------------------------------

def configs_to_curve(configs, test_data, cost_grid):
    if not configs:
        return np.full(len(cost_grid), np.nan)

    test_cs, test_qs = [], []
    for seq, thresholds in configs:
        c, q = simulate_cascade(seq, thresholds, test_data)
        test_cs.append(c)
        test_qs.append(q)

    cs = np.array(test_cs)
    qs = np.array(test_qs)

    # Keep only points within the cost grid (same domain as the envelope)
    in_range = (cs >= cost_grid[0] - 1e-10) & (cs <= cost_grid[-1] + 1e-10)
    cs, qs   = cs[in_range], qs[in_range]
    if len(cs) == 0:
        return np.full(len(cost_grid), np.nan)

    order = np.argsort(cs)
    cs, qs = cs[order], qs[order]
    qs     = np.maximum.accumulate(qs)

    f         = interp1d(cs, qs, bounds_error=False, fill_value=(np.nan, np.nan))
    interped  = f(cost_grid)
    interped  = np.maximum.accumulate(
        np.where(np.isnan(interped), -np.inf, interped))
    return np.where(interped == -np.inf, np.nan, interped)


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str):
    print(f"\n{'='*60}\n  {dataset}\n{'='*60}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check if all outputs exist
    paths = {
        opt: OUT_DIR / f"{dataset}_{opt}_curves.npy" for opt in OPTIMIZERS
    }
    paths["envelope"] = OUT_DIR / f"{dataset}_envelope_curves.npy"
    version_path = OUT_DIR / f"{dataset}_methodology_version.txt"
    cache_current = (
        version_path.exists()
        and version_path.read_text().strip() == CACHE_VERSION
    )
    if cache_current and all(p.exists() for p in paths.values()):
        print("  All cached - loading.")
        return {k: np.load(v) for k, v in paths.items()}

    raw, costs = load_full(dataset)

    # Filter NaN rows. Accuracy-based pool and pair choices are made inside
    # each split using calibration data only.
    valid_mask = np.ones(len(next(iter(raw.values()))), dtype=bool)
    for m in raw:
        valid_mask &= raw[m]["mean_token_negentropy"].notna().values
        valid_mask &= raw[m]["correct"].notna().values
    valid_idx = np.where(valid_mask)[0]
    for m in raw:
        raw[m]   = raw[m].iloc[valid_idx].reset_index(drop=True)
        costs[m] = costs[m][valid_idx]
    n = len(valid_idx)
    print(f"  Valid rows: {n}; pool selected from calibration split only.")

    # Cost grid from full-data mean costs. Costs are used only to define the
    # common x-axis, not for accuracy-based model/pair selection.
    full_mean_c = {m: costs[m].mean() for m in raw}
    cost_grid   = np.linspace(min(full_mean_c.values()),
                              max(full_mean_c.values()), N_GRID)

    ref_model = next(iter(raw))
    ref_correct = raw[ref_model]["correct"].values.astype(int)

    curves = {opt: np.full((N_SPLITS, N_GRID), np.nan) for opt in OPTIMIZERS}
    curves["envelope"] = np.full((N_SPLITS, N_GRID), np.nan)

    runners = {
        "nsga2":  run_nsga2,
        "random": run_random,
    }

    t0 = time.time()
    for i in range(N_SPLITS):
        cal_idx, test_idx = make_split(n, ref_correct, seed=SEED + i)
        split_models = non_dominated_models(raw, costs, idx=cal_idx)
        if len(split_models) < 2:
            continue
        score_ranges = {
            m: (float(raw[m]["mean_token_negentropy"].values[cal_idx].min()),
                float(raw[m]["mean_token_negentropy"].values[cal_idx].max()))
            for m in split_models
        }
        cal_data  = slice_data(raw, costs, split_models, cal_idx)
        test_data = slice_data(raw, costs, split_models, test_idx)
        cal_pairs = valid_pairs_from_data(cal_data, split_models)

        curves["envelope"][i] = pairwise_envelope_curve(
            test_data, split_models, score_ranges, cost_grid,
            pairs=cal_pairs)

        for opt, run_fn in runners.items():
            configs = run_fn(cal_data, split_models, score_ranges, split_seed=SEED + i)
            curves[opt][i] = configs_to_curve(configs, test_data, cost_grid)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (N_SPLITS - i - 1)
            print(f"  split {i+1}/{N_SPLITS}  ETA {eta:.0f}s", flush=True)

    for key, arr in curves.items():
        np.save(paths[key], arr)
    version_path.write_text(CACHE_VERSION + "\n")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    return curves


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(curves: dict) -> dict:
    env = curves["envelope"]
    rows = {}
    for opt in OPTIMIZERS:
        opt_c   = curves[opt]
        med_env = np.nanmedian(env,   axis=0)
        med_opt = np.nanmedian(opt_c, axis=0)

        # Gap: difference of cross-split medians averaged over a 20-point window
        # around the grid mid-point - matches fig2_compute.compute_diagnostics.
        win = slice(MID_IDX - 10, MID_IDX + 10)
        median_delta = float(np.nanmean(med_opt[win] - med_env[win]))

        env_bw = float(np.nanmean(
            np.nanpercentile(env,   90, axis=0)
            - np.nanpercentile(env,   10, axis=0)))
        opt_bw = float(np.nanmean(
            np.nanpercentile(opt_c, 90, axis=0)
            - np.nanpercentile(opt_c, 10, axis=0)))

        valid     = ~np.isnan(med_env) & ~np.isnan(med_opt)
        rows[opt] = {
            "median_delta":   median_delta,
            "bwr":            float(opt_bw / env_bw) if env_bw > 0 else float("nan"),
            "dominance_frac": float(np.mean(med_opt[valid] > med_env[valid]))
                              if valid.any() else float("nan"),
        }
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_sensitivity(all_curves: dict[str, dict]):
    fig, axes = plt.subplots(1, 5, figsize=(8.5, 2.2))

    for ax, dataset in zip(axes, DATASETS):
        if dataset not in all_curves:
            ax.set_visible(False)
            continue

        curves   = all_curves[dataset]
        cost_grid = np.linspace(0, 1, N_GRID)  # placeholder; scaled below

        # Recover approximate cost grid from non-NaN region of envelope
        env_med = np.nanmedian(curves["envelope"], axis=0)

        ax.plot(np.arange(N_GRID), env_med,
                color="steelblue", lw=1.5, label="Envelope", zorder=4)
        for opt in OPTIMIZERS:
            med = np.nanmedian(curves[opt], axis=0)
            ax.plot(np.arange(N_GRID), med, zorder=3, **OPT_STYLES[opt])

        ax.set_xlabel("Cost grid index", fontsize=7)
        ax.set_ylabel("E[accuracy]", fontsize=7)
        ax.set_title(DATASET_LABELS[dataset], fontsize=9)
        ax.tick_params(labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figA_opt_sensitivity.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_diag: dict):
    header = f"{'Optimizer':<12s}"
    for ds in DATASETS:
        if ds in all_diag:
            header += f"  {DATASET_LABELS[ds]:>18s}"
    print("\n" + header)
    sub = " " * 12 + "".join(
        f"  {'Delta':>7s}  {'BWR':>5s}"
        for ds in DATASETS if ds in all_diag)
    print(sub)

    for opt in OPTIMIZERS:
        row = f"{opt:<12s}"
        for ds in DATASETS:
            if ds not in all_diag:
                continue
            d = all_diag[ds][opt]
            row += (f"  {d['median_delta']:+.4f}"
                    f"  {d['bwr']:5.2f}")
        print(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = sys.argv[1:] or DATASETS

    all_curves = {}
    all_diag   = {}
    summary_rows = []

    for ds in datasets:
        curves = run_dataset(ds)
        all_curves[ds] = curves
        diag = compute_diagnostics(curves)
        all_diag[ds] = diag
        for opt, d in diag.items():
            summary_rows.append({"dataset": ds, "optimizer": opt, **d})

    df_summary = pd.DataFrame(summary_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(OUT_DIR / "summary.csv", index=False)

    print_summary(all_diag)
    plot_sensitivity(all_curves)
    print("\nDone.")
