"""
fig2_compute.py - Compute held-out frontier comparisons across 50 random
50/50 splits.

Outputs (per dataset):
  results/fig2/{dataset}_envelope_curves.npy    (50, N_GRID)
  results/fig2/{dataset}_fixed_chain_curves.npy (50, N_GRID)
  results/fig2/{dataset}_kcascade_curves.npy    (50, N_GRID)
  results/fig2/{dataset}_cost_grid.npy          (N_GRID,)
  results/fig2/{dataset}_model_coords.json      {model: [cost, acc]}
  results/fig2/{dataset}_diagnostics.json

Usage:
    python3 fig2_compute.py [dataset ...]
"""

import sys, json, time
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from itertools import combinations
from scipy.interpolate import interp1d

from optuna_frontier import (
    PRICE_PER_TOKEN, compute_costs, simulate_cascade, DATA_DIR, SCORER
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR  = Path("results/fig2")
N_SPLITS = 50
N_GRID   = 500
N_THRESH = 200
N_TRIALS = 2000
POP_SIZE = 100
MAX_K    = 4
SEED     = 42
CACHE_VERSION = "calibration-selected-pools-v1"

MODEL_COST_ORDER = sorted(PRICE_PER_TOKEN, key=lambda m: sum(PRICE_PER_TOKEN[m]))

# Models excluded per dataset (e.g. gpt-oss-20b dominates all others on livecodebench,
# so no cascade is useful; we exclude it to study cascade dynamics on that dataset).
DATASET_EXCLUSIONS: dict[str, list[str]] = {
    "livecodebench": ["gpt-oss-20b"],
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_full(dataset: str) -> tuple[dict, dict]:
    """Load all models for dataset. Returns (raw_dfs, per_query_costs)."""
    excluded = DATASET_EXCLUSIONS.get(dataset, [])
    raw, costs = {}, {}
    for m in MODEL_COST_ORDER:
        if m in excluded:
            continue
        path = DATA_DIR / f"{dataset}-{m}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).iloc[:2000].reset_index(drop=True)
        if SCORER not in df.columns or "correct" not in df.columns:
            continue
        raw[m]   = df
        costs[m] = compute_costs(df, m)
    return raw, costs


def non_dominated_models(raw: dict, costs: dict, idx: np.ndarray | None = None) -> list[str]:
    """Return cost-ordered non-dominated subset, optionally estimated on idx only."""
    mean_c = {m: costs[m][idx].mean() if idx is not None else costs[m].mean()
              for m in raw}
    mean_a = {
        m: raw[m]["correct"].values[idx].mean() if idx is not None
        else raw[m]["correct"].mean()
        for m in raw
    }
    dominated = {
        m for m in raw
        if any(mean_c[o] < mean_c[m] and mean_a[o] > mean_a[m]
               for o in raw if o != m)
    }
    nd = [m for m in MODEL_COST_ORDER if m in raw and m not in dominated]
    return nd


def valid_pairs_from_data(data: dict, models: list[str]) -> list[tuple[str, str]]:
    """Cost-ordered pairs whose expensive model is more accurate in the provided data."""
    mean_c = {m: data[m]["costs"].mean() for m in models}
    mean_a = {m: data[m]["correct"].mean() for m in models}
    models_by_cost = sorted(models, key=lambda m: mean_c[m])
    return [
        (L, H) for L, H in combinations(models_by_cost, 2)
        if mean_a[H] > mean_a[L]
    ]


def make_split(n: int, correct: np.ndarray, seed: int):
    rng  = np.random.default_rng(seed)
    idx0 = np.where(correct == 0)[0]
    idx1 = np.where(correct == 1)[0]
    n0   = len(idx0) // 2
    n1   = len(idx1) // 2
    cal  = np.sort(np.concatenate([
        rng.choice(idx0, n0, replace=False),
        rng.choice(idx1, n1, replace=False),
    ]))
    test = np.setdiff1d(np.arange(n), cal)
    return cal, test


def slice_data(raw: dict, costs: dict, models: list, idx: np.ndarray) -> dict:
    """Extract aligned per-query arrays for a subset of row indices."""
    return {
        m: {
            "scores":  raw[m][SCORER].values[idx],
            "correct": raw[m]["correct"].values[idx].astype(float),
            "costs":   costs[m][idx],
            "prompt":  raw[m]["prompt"].values[idx],
        }
        for m in models
    }


# ---------------------------------------------------------------------------
# Per-split pairwise envelope
# ---------------------------------------------------------------------------

def pairwise_envelope_curve(
    test_data: dict,
    models: list,
    score_ranges: dict,
    cost_grid: np.ndarray,
    pairs: list[tuple[str, str]] | None = None,
) -> np.ndarray:
    """Compute pairwise envelope on test_data, interpolated to cost_grid."""
    if pairs is None:
        pairs = valid_pairs_from_data(test_data, models)

    interped = []
    for L, H in pairs:
        s   = test_data[L]["scores"]
        u_L = test_data[L]["correct"]
        u_H = test_data[H]["correct"]
        # Use score range from full data for stable threshold grid
        tau_grid = np.linspace(score_ranges[L][0], score_ranges[L][1], N_THRESH)

        cs = [test_data[L]["costs"].mean()]
        qs = [u_L.mean()]
        for tau in tau_grid:
            esc = s < tau
            per_q_c = test_data[L]["costs"] + esc * test_data[H]["costs"]
            c = per_q_c.mean()
            q = np.where(esc, u_H, u_L).mean()
            if c <= test_data[H]["costs"].mean() + 1e-12:
                cs.append(c); qs.append(q)
        cs.append(test_data[H]["costs"].mean()); qs.append(u_H.mean())

        cs_arr, qs_arr = np.array(cs), np.array(qs)
        order = np.argsort(cs_arr)
        f = interp1d(cs_arr[order], qs_arr[order],
                     bounds_error=False, fill_value=(np.nan, np.nan))
        interped.append(f(cost_grid))

    if not interped:
        return np.full(len(cost_grid), np.nan)

    stacked = np.vstack(interped)
    env = np.nanmax(stacked, axis=0)
    env = np.maximum.accumulate(np.where(np.isnan(env), -np.inf, env))
    return np.where(env == -np.inf, np.nan, env)


# ---------------------------------------------------------------------------
# Per-split k-cascade
# ---------------------------------------------------------------------------

def make_objective_nd(cal_data: dict, models: list, score_ranges: dict):
    """Build Optuna objective restricted to the non-dominated pool, k<=MAX_K."""
    # Sort models by actual calibration mean cost so sequences are cost-ordered.
    models_sorted = sorted(models, key=lambda m: cal_data[m]["costs"].mean())
    all_sequences = []
    for k in range(2, min(MAX_K, len(models_sorted)) + 1):
        for combo in combinations(range(len(models_sorted)), k):
            all_sequences.append(tuple(models_sorted[i] for i in combo))

    def objective(trial: optuna.Trial):
        seq_idx = trial.suggest_categorical("seq_idx", list(range(len(all_sequences))))
        seq     = all_sequences[seq_idx]
        k       = len(seq)
        tau_norms = [trial.suggest_float(f"tau_{i}", 0.0, 1.0)
                     for i in range(MAX_K - 1)]
        thresholds = [
            score_ranges[seq[i]][0]
            + tau_norms[i] * (score_ranges[seq[i]][1] - score_ranges[seq[i]][0])
            for i in range(k - 1)
        ]
        c, q = simulate_cascade(list(seq), thresholds, cal_data)
        return c, -q

    return objective, all_sequences


def kcascade_curve(
    cal_data: dict,
    test_data: dict,
    models: list,
    score_ranges: dict,
    cost_grid: np.ndarray,
    split_seed: int,
) -> np.ndarray:
    """Run Optuna on cal, evaluate Pareto trials on test, interpolate to cost_grid."""
    obj, seqs = make_objective_nd(cal_data, models, score_ranges)

    sampler = optuna.samplers.NSGAIISampler(population_size=POP_SIZE,
                                            seed=split_seed)
    study = optuna.create_study(directions=["minimize", "minimize"],
                                sampler=sampler)
    study.optimize(obj, n_trials=N_TRIALS)

    # Re-evaluate on test
    test_cs, test_qs = [], []
    for t in study.best_trials:
        seq = seqs[t.params["seq_idx"]]
        k   = len(seq)
        tau_norms  = [t.params[f"tau_{i}"] for i in range(MAX_K - 1)]
        thresholds = [
            score_ranges[seq[i]][0]
            + tau_norms[i] * (score_ranges[seq[i]][1] - score_ranges[seq[i]][0])
            for i in range(k - 1)
        ]
        c, q = simulate_cascade(list(seq), thresholds, test_data)
        test_cs.append(c); test_qs.append(q)

    if not test_cs:
        return np.full(len(cost_grid), np.nan)

    cs = np.array(test_cs)
    qs = np.array(test_qs)
    # Pareto-filter test points
    order = np.argsort(cs)
    cs, qs = cs[order], qs[order]
    qs = np.maximum.accumulate(qs)

    f = interp1d(cs, qs, bounds_error=False, fill_value=(np.nan, np.nan))
    interped = f(cost_grid)
    # Enforce monotonicity
    interped = np.maximum.accumulate(np.where(np.isnan(interped), -np.inf, interped))
    return np.where(interped == -np.inf, np.nan, interped)


def fixed_chain_curve(
    cal_data: dict,
    test_data: dict,
    models: list,
    score_ranges: dict,
    cost_grid: np.ndarray,
    split_seed: int,
) -> np.ndarray:
    """Optimize thresholds for the full calibration-selected cost-ordered chain."""
    seq = tuple(sorted(models, key=lambda m: cal_data[m]["costs"].mean()))
    k = len(seq)
    if k < 2:
        return np.full(len(cost_grid), np.nan)

    def objective(trial: optuna.Trial):
        thresholds = [
            score_ranges[seq[i]][0]
            + trial.suggest_float(f"tau_{i}", 0.0, 1.0)
            * (score_ranges[seq[i]][1] - score_ranges[seq[i]][0])
            for i in range(k - 1)
        ]
        c, q = simulate_cascade(list(seq), thresholds, cal_data)
        return c, -q

    sampler = optuna.samplers.NSGAIISampler(population_size=POP_SIZE,
                                            seed=split_seed + 100_000)
    study = optuna.create_study(directions=["minimize", "minimize"],
                                sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS)

    test_cs, test_qs = [], []
    for t in study.best_trials:
        thresholds = [
            score_ranges[seq[i]][0]
            + t.params[f"tau_{i}"] * (score_ranges[seq[i]][1] - score_ranges[seq[i]][0])
            for i in range(k - 1)
        ]
        c, q = simulate_cascade(list(seq), thresholds, test_data)
        test_cs.append(c)
        test_qs.append(q)

    if not test_cs:
        return np.full(len(cost_grid), np.nan)

    cs = np.array(test_cs)
    qs = np.array(test_qs)
    order = np.argsort(cs)
    cs, qs = cs[order], qs[order]
    qs = np.maximum.accumulate(qs)

    f = interp1d(cs, qs, bounds_error=False, fill_value=(np.nan, np.nan))
    interped = f(cost_grid)
    interped = np.maximum.accumulate(np.where(np.isnan(interped), -np.inf, interped))
    return np.where(interped == -np.inf, np.nan, interped)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(
    env_curves: np.ndarray,
    kc_curves: np.ndarray,
    cost_grid: np.ndarray,
) -> dict:
    median_env = np.nanmedian(env_curves, axis=0)
    median_kc  = np.nanmedian(kc_curves,  axis=0)

    mid_idx = len(cost_grid) // 2
    gap     = float(np.nanmean(median_kc[mid_idx-10:mid_idx+10]
                               - median_env[mid_idx-10:mid_idx+10]))

    env_bw = float(np.nanmean(
        np.nanpercentile(env_curves, 90, axis=0)
        - np.nanpercentile(env_curves, 10, axis=0)
    ))
    kc_bw  = float(np.nanmean(
        np.nanpercentile(kc_curves, 90, axis=0)
        - np.nanpercentile(kc_curves, 10, axis=0)
    ))
    bw_ratio = kc_bw / env_bw if env_bw > 0 else np.nan

    valid = ~np.isnan(median_env) & ~np.isnan(median_kc)
    dom_frac = float(np.mean(median_kc[valid] > median_env[valid])) if valid.any() else np.nan

    return {
        "median_gap":         gap,
        "band_width_ratio":   bw_ratio,
        "dominance_fraction": dom_frac,
    }


# ---------------------------------------------------------------------------
# Dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str):
    print(f"\n{'='*60}\n  {dataset}\n{'='*60}")
    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already done
    version_path = out_dir / "methodology_version.txt"
    cache_current = (
        version_path.exists()
        and version_path.read_text().strip() == CACHE_VERSION
    )
    if cache_current and (out_dir / "envelope_curves.npy").exists() and \
       (out_dir / "fixed_chain_curves.npy").exists() and \
       (out_dir / "kcascade_curves.npy").exists():
        print("  Already computed - skipping.")
        return

    raw, costs = load_full(dataset)

    # Drop rows where any available model has NaN correct or NaN scorer
    # (e.g. SimpleQA judge failures). This uses missingness only; no
    # accuracy-based model/pair filtering is done until each calibration split.
    valid_mask = np.ones(len(next(iter(raw.values()))), dtype=bool)
    for m in raw:
        valid_mask &= raw[m][SCORER].notna().values
        valid_mask &= raw[m]["correct"].notna().values
    valid_idx = np.where(valid_mask)[0]
    for m in raw:
        raw[m]   = raw[m].iloc[valid_idx].reset_index(drop=True)
        costs[m] = costs[m][valid_idx]

    n           = len(valid_idx)
    ref_model   = next(m for m in MODEL_COST_ORDER if m in raw)
    ref_correct = raw[ref_model]["correct"].values.astype(int)
    print(f"  Valid rows after NaN filter: {n} / {valid_mask.size}")
    print("  Pool and valid pairs selected from calibration split only.")

    # Cost grid from full-data model mean costs. Costs are used only to define
    # a common x-axis; accuracy-based pool and pair choices are made per split
    # on calibration data below.
    full_mean_c = {m: costs[m].mean() for m in raw}
    grid_min = min(full_mean_c.values())
    grid_max = max(full_mean_c.values())
    cost_grid = np.linspace(grid_min, grid_max, N_GRID)
    np.save(out_dir / "cost_grid.npy", cost_grid)

    # Descriptive full-data model coordinates; not used for held-out selection.
    full_mean_a = {m: raw[m]["correct"].mean() for m in raw}
    model_coords = {m: [float(full_mean_c[m]), float(full_mean_a[m])] for m in raw}
    with open(out_dir / "model_coords.json", "w") as f:
        json.dump(model_coords, f, indent=2)

    env_curves   = np.full((N_SPLITS, N_GRID), np.nan)
    fixed_curves = np.full((N_SPLITS, N_GRID), np.nan)
    kc_curves    = np.full((N_SPLITS, N_GRID), np.nan)

    t0 = time.time()
    for i in range(N_SPLITS):
        cal_idx, test_idx = make_split(n, ref_correct, seed=SEED + i)
        split_models = non_dominated_models(raw, costs, idx=cal_idx)
        if len(split_models) < 2:
            continue

        score_ranges = {
            m: (float(raw[m][SCORER].values[cal_idx].min()),
                float(raw[m][SCORER].values[cal_idx].max()))
            for m in split_models
        }
        cal_data  = slice_data(raw, costs, split_models, cal_idx)
        test_data = slice_data(raw, costs, split_models, test_idx)
        cal_pairs = valid_pairs_from_data(cal_data, split_models)

        env_curves[i] = pairwise_envelope_curve(test_data, split_models,
                                                 score_ranges, cost_grid,
                                                 pairs=cal_pairs)
        fixed_curves[i] = fixed_chain_curve(cal_data, test_data, split_models,
                                            score_ranges, cost_grid,
                                            split_seed=SEED + i)
        kc_curves[i]  = kcascade_curve(cal_data, test_data, split_models,
                                        score_ranges, cost_grid,
                                        split_seed=SEED + i)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (N_SPLITS - i - 1)
        print(f"  split {i+1:2d}/{N_SPLITS}  "
              f"env_q={np.nanmedian(env_curves[i]):.4f}  "
              f"fix_q={np.nanmedian(fixed_curves[i]):.4f}  "
              f"kc_q={np.nanmedian(kc_curves[i]):.4f}  "
              f"ETA {eta:.0f}s", flush=True)

    np.save(out_dir / "envelope_curves.npy", env_curves)
    np.save(out_dir / "fixed_chain_curves.npy", fixed_curves)
    np.save(out_dir / "kcascade_curves.npy",  kc_curves)

    diag = compute_diagnostics(env_curves, kc_curves, cost_grid)
    diag["fixed_chain_vs_envelope"] = compute_diagnostics(env_curves, fixed_curves, cost_grid)
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    version_path.write_text(CACHE_VERSION + "\n")

    print(f"\n  Diagnostics: {diag}")
    print(f"  Total time: {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = sys.argv[1:] or ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        run_dataset(ds)
    print("\nDone.")
