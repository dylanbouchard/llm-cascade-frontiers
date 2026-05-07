"""
grid_sensitivity.py - Threshold grid resolution sensitivity (Appendix).

Tests whether the 200-point threshold grid used in the main experiments
accurately captures the pairwise envelope. Runs the envelope pipeline at
four resolutions (50, 100, 200, 500) with everything else fixed.

Outputs:
  results/gridsens/{dataset}_{resolution}_curves.npy  - (50, 500)
  results/gridsens/summary.csv
  figures/figA_grid_sensitivity.pdf

Usage:
    python3 grid_sensitivity.py [dataset ...]
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
from itertools import combinations
from scipy.interpolate import interp1d

from fig2_compute import (
    load_full, non_dominated_models, make_split, slice_data,
    valid_pairs_from_data, SEED, CACHE_VERSION,
)

OUT_DIR  = Path("results/gridsens")
FIG_DIR  = Path("figures")
DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
DATASET_LABELS = {"mmlu": "MMLU", "triviaqa": "TriviaQA",
                  "math_hard": "MATH", "simpleqa": "SimpleQA",
                  "livecodebench": "LiveCodeBench"}

RESOLUTIONS  = [50, 100, 200, 500]
N_SPLITS     = 50
COMMON_GRID  = 500   # fixed cost-grid size for interpolation

RES_COLORS = {50: "#D65F5F", 100: "#C4AD66", 200: "#4878CF", 500: "#1A1A1A"}
RES_ALPHA  = {50: 0.7, 100: 0.7, 200: 0.85, 500: 0.9}


# ---------------------------------------------------------------------------
# Core: envelope at a given resolution
# ---------------------------------------------------------------------------

def envelope_at_resolution(
    test_data: dict,
    models: list,
    score_ranges: dict,
    cost_grid: np.ndarray,
    n_thresh: int,
    pairs: list[tuple[str, str]] | None = None,
) -> np.ndarray:
    """
    Compute pairwise envelope on test_data using n_thresh threshold points,
    interpolated onto cost_grid.
    """
    if pairs is None:
        pairs = valid_pairs_from_data(test_data, models)

    interped = []
    for L, H in pairs:
        s    = test_data[L]["scores"]
        u_L  = test_data[L]["correct"]
        u_H  = test_data[H]["correct"]
        c_L  = test_data[L]["costs"]
        c_H  = test_data[H]["costs"]

        tau_grid = np.linspace(score_ranges[L][0], score_ranges[L][1], n_thresh)

        cs = [c_L.mean()]
        qs = [u_L.mean()]
        for tau in tau_grid:
            esc = s < tau
            c   = c_L.mean() + (esc * c_H).mean()
            q   = np.where(esc, u_H, u_L).mean()
            if c <= c_L.mean() + c_H.mean() + 1e-12:
                cs.append(c); qs.append(q)
        cs.append(c_L.mean() + c_H.mean()); qs.append(u_H.mean())

        cs_arr = np.array(cs); qs_arr = np.array(qs)
        order  = np.argsort(cs_arr)
        f = interp1d(cs_arr[order], qs_arr[order],
                     bounds_error=False, fill_value=(np.nan, np.nan))
        interped.append(f(cost_grid))

    if not interped:
        return np.full(len(cost_grid), np.nan)

    env = np.nanmax(np.vstack(interped), axis=0)
    env = np.maximum.accumulate(np.where(np.isnan(env), -np.inf, env))
    return np.where(env == -np.inf, np.nan, env)


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str):
    print(f"\n{'='*60}\n  {dataset}\n{'='*60}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw, costs = load_full(dataset)

    valid_mask = np.ones(len(next(iter(raw.values()))), dtype=bool)
    for m in raw:
        valid_mask &= raw[m]["mean_token_negentropy"].notna().values
        valid_mask &= raw[m]["correct"].notna().values
    valid_idx = np.where(valid_mask)[0]
    for m in raw:
        raw[m]   = raw[m].iloc[valid_idx].reset_index(drop=True)
        costs[m] = costs[m][valid_idx]
    n = len(valid_idx)

    ref_model = next(iter(raw))
    ref_correct  = raw[ref_model]["correct"].values.astype(int)

    # Fixed common cost grid from full-data model costs; accuracy-based pool
    # and pair choices are made on calibration data in each split.
    mean_c = {m: costs[m].mean() for m in raw}
    cost_grid = np.linspace(min(mean_c.values()), max(mean_c.values()), COMMON_GRID)

    curves = {}  # res -> (N_SPLITS, COMMON_GRID)

    for res in RESOLUTIONS:
        out_path = OUT_DIR / f"{dataset}_{res}_curves.npy"
        version_path = OUT_DIR / f"{dataset}_{res}_methodology_version.txt"
        cache_current = (
            version_path.exists()
            and version_path.read_text().strip() == CACHE_VERSION
        )
        if cache_current and out_path.exists():
            print(f"  n_tau={res}: loading cached.")
            curves[res] = np.load(out_path)
            continue

        mat = np.full((N_SPLITS, COMMON_GRID), np.nan)
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
            cal_data = slice_data(raw, costs, split_models, cal_idx)
            test_data = slice_data(raw, costs, split_models, test_idx)
            cal_pairs = valid_pairs_from_data(cal_data, split_models)
            mat[i]      = envelope_at_resolution(
                test_data, split_models, score_ranges, cost_grid,
                n_thresh=res, pairs=cal_pairs)

        np.save(out_path, mat)
        version_path.write_text(CACHE_VERSION + "\n")
        curves[res] = mat
        print(f"  n_tau={res}: done.", flush=True)

    return cost_grid, curves


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_summary(curves: dict) -> dict:
    """Compare each resolution against n_tau=500 (reference)."""
    ref = curves[500]
    rows = []
    for res in RESOLUTIONS:
        diff = curves[res] - ref  # (50, 500), negative means coarser is worse
        rows.append({
            "resolution": res,
            "mean_abs_diff": float(np.nanmean(np.abs(diff))),
            "max_abs_diff":  float(np.nanmax(np.abs(diff))),
            "median_diff":   float(np.nanmedian(diff)),
            "p90_abs_diff":  float(np.nanpercentile(np.abs(diff), 90)),
        })
    return rows


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_overlay(all_curves: dict[str, dict], all_grids: dict[str, np.ndarray]):
    fig, axes = plt.subplots(1, 5, figsize=(8.5, 2.2))

    for ax, dataset in zip(axes, DATASETS):
        if dataset not in all_curves:
            ax.set_visible(False)
            continue

        grid   = all_grids[dataset]
        curves = all_curves[dataset]

        # Plot reference (500) first so coloured lines render on top
        plot_order = [500, 50, 100, 200]
        zorders    = {500: 1, 50: 2, 100: 3, 200: 4}
        lws        = {500: 1.2, 200: 1.8, 100: 1.4, 50: 1.4}
        alphas     = {500: 0.55, 200: 0.95, 100: 0.85, 50: 0.85}
        linestyles = {500: "-", 200: "--", 100: (0, (3, 1)), 50: ":"}

        for res in plot_order:
            med = np.nanmedian(curves[res], axis=0)
            ax.plot(grid * 1e6, med,
                    color=RES_COLORS[res], alpha=alphas[res],
                    lw=lws[res], ls=linestyles[res],
                    label=f"$n_{{\\tau}}={res}$", zorder=zorders[res])

        ax.set_xlabel(r"E[cost] ($\mu$\$)", fontsize=7)
        ax.set_ylabel("E[accuracy]", fontsize=7)
        ax.set_title(DATASET_LABELS[dataset], fontsize=9)
        ax.tick_params(labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout(pad=0.8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figA_grid_sensitivity.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = sys.argv[1:] or DATASETS

    all_grids  = {}
    all_curves = {}
    summary_rows = []

    for ds in datasets:
        grid, curves = run_dataset(ds)
        all_grids[ds]  = grid
        all_curves[ds] = curves

        rows = compute_summary(curves)
        for r in rows:
            r["dataset"] = ds
        summary_rows.extend(rows)

    # Save summary CSV
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(OUT_DIR / "summary.csv", index=False)

    # Print table
    print(f"\n{'Resolution':<12s}" + "".join(
        f"  {DATASET_LABELS[ds]:>18s}" for ds in datasets))
    print(" " * 12 + "".join(
        f"  {'mean|d|':>8s}  {'max|d|':>7s}" for _ in datasets))
    for res in RESOLUTIONS:
        row = f"{res:<12d}"
        for ds in datasets:
            r = next(x for x in summary_rows if x["dataset"] == ds and x["resolution"] == res)
            if res == 500:
                row += f"  {'(ref)':>8s}  {'(ref)':>7s}"
            else:
                row += f"  {r['mean_abs_diff']:>8.5f}  {r['max_abs_diff']:>7.5f}"
        print(row)

    plot_overlay(all_curves, all_grids)
    print("\nDone.")
