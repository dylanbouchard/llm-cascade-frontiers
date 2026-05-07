"""
escalation_benefit.py - Estimate escalation benefit curves m_H(s) - m_L(s).

For each (pair, dataset) cell, fits smoothing splines across 50 random
50/50 calibration-test splits and saves the results as .npz files.

Outputs: results/escalation_benefit/{dataset}_{modelL}_{modelH}_{scorer}.npz
Each file contains:
    s_grid          (200,) evaluation grid
    spline_curves   (50, 200) per-split spline evaluations
    median_curve    (200,)
    band_lo         (200,) 10th percentile
    band_hi         (200,) 90th percentile
    isotonic_curve  (200,)
    diagnostics     dict stored via np.savez keyword args:
        dominance_frac, decreasing_frac, max_reversal

Usage:
    python3 escalation_benefit.py [dataset ...]
    python3 escalation_benefit.py mmlu triviaqa math_hard simpleqa
"""

import sys
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from scipy.interpolate import make_smoothing_spline
from sklearn.isotonic import IsotonicRegression

# Reuse infrastructure from cascade_core
from cascade_core import (
    DATA_DIR, MODEL_COST_ORDER, SCORERS, load_all_models, compute_costs
)

OUT_DIR  = Path("results/escalation_benefit")
N_SPLITS = 50
N_GRID   = 200
N_BINS   = 40
SEED     = 42


# ---------------------------------------------------------------------------
# Spline fitting
# ---------------------------------------------------------------------------

def fit_spline_direct(s: np.ndarray, benefit: np.ndarray, s_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(s)
    s_sorted = s[order]
    b_sorted = benefit[order].astype(float)
    # Deduplicate ties in s (average benefit)
    s_uniq, inv = np.unique(s_sorted, return_inverse=True)
    b_uniq = np.array([b_sorted[inv == i].mean() for i in range(len(s_uniq))])
    if len(s_uniq) < 4:
        return np.full(N_GRID, np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spline = make_smoothing_spline(s_uniq, b_uniq)
        return spline(s_grid)
    except Exception:
        # GCV failed (e.g. near-singular matrix on discrete scores); fall back to binned
        return fit_spline_binned(s, benefit, s_grid)


def fit_spline_binned(s: np.ndarray, benefit: np.ndarray, s_grid: np.ndarray,
                      n_bins: int = N_BINS) -> np.ndarray:
    bin_edges   = np.linspace(s.min(), s.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx     = np.clip(np.digitize(s, bin_edges) - 1, 0, n_bins - 1)

    bin_means  = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = bin_idx == i
        if mask.sum() > 0:
            bin_means[i]  = benefit[mask].astype(float).mean()
            bin_counts[i] = mask.sum()

    valid = bin_counts > 0
    if valid.sum() < 4:
        return np.full(N_GRID, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spline = make_smoothing_spline(bin_centers[valid], bin_means[valid])
    return spline(s_grid)


def fit_spline(s: np.ndarray, benefit: np.ndarray, s_grid: np.ndarray,
               use_binned: bool) -> np.ndarray:
    if use_binned:
        return fit_spline_binned(s, benefit, s_grid)
    return fit_spline_direct(s, benefit, s_grid)


def is_wiggly(curves: np.ndarray, threshold: float = 0.3) -> bool:
    """Return True if median curve has too many sign changes (jagged)."""
    median = np.nanmedian(curves, axis=0)
    diffs  = np.diff(median)
    sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
    return sign_changes / len(diffs) > threshold


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(median_curve: np.ndarray) -> dict:
    dominance_frac   = float(np.mean(median_curve > 0))
    diffs            = np.diff(median_curve)
    decreasing_frac  = float(np.mean(diffs <= 0))
    max_reversal     = float(np.max(np.maximum(diffs, 0)))
    return {
        "dominance_frac":  dominance_frac,
        "decreasing_frac": decreasing_frac,
        "max_reversal":    max_reversal,
    }


# ---------------------------------------------------------------------------
# Main cell computation
# ---------------------------------------------------------------------------

def compute_cell(
    s_L: np.ndarray,
    u_L: np.ndarray,
    u_H: np.ndarray,
    scorer: str,
    dataset: str,
    model_L: str,
    model_H: str,
) -> dict:
    n      = len(s_L)
    rng    = np.random.default_rng(SEED)
    s_min, s_max = s_L.min(), s_L.max()
    s_grid = np.linspace(s_min, s_max, N_GRID)
    benefit = (u_H - u_L).astype(float)

    # Generate 50 random 50/50 splits
    splits = []
    for _ in range(N_SPLITS):
        perm     = rng.permutation(n)
        test_idx = perm[n // 2:]
        splits.append(test_idx)

    # First pass: try direct splines on first 5 splits to decide approach
    probe_curves = []
    for test_idx in splits[:5]:
        curve = fit_spline_direct(s_L[test_idx], benefit[test_idx], s_grid)
        probe_curves.append(curve)
    probe_arr = np.array(probe_curves)
    use_binned = is_wiggly(probe_arr[~np.isnan(probe_arr).any(axis=1)])

    method = "binned" if use_binned else "direct"

    # Full 50-split pass
    spline_curves = []
    for test_idx in splits:
        curve = fit_spline(s_L[test_idx], benefit[test_idx], s_grid, use_binned)
        spline_curves.append(curve)
    spline_curves = np.array(spline_curves)  # (50, 200)

    # Replace any nan rows with interpolated values from non-nan rows
    valid_rows = ~np.isnan(spline_curves).any(axis=1)
    if valid_rows.sum() < 2:
        print(f"  WARNING: fewer than 2 valid splits for {dataset}/{model_L}->{model_H}/{scorer}")

    median_curve = np.nanmedian(spline_curves, axis=0)
    band_lo      = np.nanpercentile(spline_curves, 10, axis=0)
    band_hi      = np.nanpercentile(spline_curves, 90, axis=0)

    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    isotonic_curve = iso.fit_transform(s_grid, median_curve)

    diagnostics = compute_diagnostics(median_curve)

    return {
        "s_grid":        s_grid,
        "spline_curves": spline_curves,
        "median_curve":  median_curve,
        "band_lo":       band_lo,
        "band_hi":       band_hi,
        "isotonic_curve": isotonic_curve,
        "diagnostics":   diagnostics,
        "method":        method,
    }


# ---------------------------------------------------------------------------
# Dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, scorers: list[str] = SCORERS):
    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset}")
    print(f"{'='*60}")

    data = load_all_models(dataset)
    models = list(data.keys())
    # Sort by actual per-query cost (token-price-sum ordering can differ)
    mean_costs = {m: compute_costs(data[m], m).mean() for m in models}
    models_by_cost = sorted(models, key=lambda m: mean_costs[m])
    pairs = list(combinations(models_by_cost, 2))

    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for scorer in scorers:
        for model_L, model_H in pairs:
            df_L = data[model_L]
            df_H = data[model_H]

            # Build aligned arrays, drop any rows with missing data
            valid = (
                df_L[scorer].notna() &
                df_L["correct"].notna() &
                df_H["correct"].notna()
            )
            if valid.sum() < 200:
                print(f"  SKIP {model_L}->{model_H}/{scorer}: only {valid.sum()} valid rows")
                continue

            s_L = df_L.loc[valid, scorer].to_numpy()
            u_L = df_L.loc[valid, "correct"].to_numpy().astype(float)
            u_H = df_H.loc[valid, "correct"].to_numpy().astype(float)

            fname = out_dir / f"{dataset}_{model_L}_{model_H}_{scorer}.npz"
            if fname.exists():
                print(f"  skip (exists): {fname.name}")
                continue

            print(f"  {model_L} -> {model_H}  [{scorer}]", end="", flush=True)
            result = compute_cell(s_L, u_L, u_H, scorer, dataset, model_L, model_H)

            d = result["diagnostics"]
            print(f"  dom={d['dominance_frac']:.2f}  dec={d['decreasing_frac']:.2f}"
                  f"  maxrev={d['max_reversal']:.3f}  method={result['method']}")

            np.savez(
                fname,
                s_grid        = result["s_grid"],
                spline_curves = result["spline_curves"],
                median_curve  = result["median_curve"],
                band_lo       = result["band_lo"],
                band_hi       = result["band_hi"],
                isotonic_curve= result["isotonic_curve"],
                dominance_frac  = d["dominance_frac"],
                decreasing_frac = d["decreasing_frac"],
                max_reversal    = d["max_reversal"],
                method          = result["method"],
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = sys.argv[1:] or ["mmlu", "triviaqa", "simpleqa", "math_hard", "livecodebench"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        run_dataset(ds, scorers=["mean_token_negentropy"])
    print("\nDone.")
