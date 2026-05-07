"""
cal_sensitivity.py - Appendix A7: Calibration-size sensitivity for the
k-cascade comparison.

Tests whether near-zero k-cascade gaps relative to the pairwise envelope
are driven by insufficient calibration data. Runs the Figure 2 pipeline
at four calibration fractions (50/70/80/90%) across all four datasets.

Outputs per (dataset, frac):
  results/cal_sensitivity/{dataset}_{label}_envelope.npy   (N_SPLITS, N_GRID)
  results/cal_sensitivity/{dataset}_{label}_kcascade.npy   (N_SPLITS, N_GRID)
  results/cal_sensitivity/{dataset}_{label}_diagnostics.json

Figures:
  figures/figA7_cal_sensitivity.pdf   - 1x4 Delta vs cal-fraction with error bars

Usage:
    python3 cal_sensitivity.py [dataset ...]
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
    load_full, non_dominated_models, slice_data, valid_pairs_from_data,
    pairwise_envelope_curve, kcascade_curve,
    N_GRID, N_THRESH, POP_SIZE, SEED, CACHE_VERSION,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR  = Path("results/cal_sensitivity")
FIG_DIR  = Path("figures")
DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
DATASET_LABELS = {"mmlu": "MMLU", "triviaqa": "TriviaQA",
                  "math_hard": "MATH", "simpleqa": "SimpleQA",
                  "livecodebench": "LiveCodeBench"}

CAL_FRACS = [0.50, 0.70, 0.80, 0.90]
FRAC_LABELS = {0.50: "50/50", 0.70: "70/30", 0.80: "80/20", 0.90: "90/10"}
FRAC_AXIS_LABELS = {0.50: "50", 0.70: "70", 0.80: "80", 0.90: "90"}

N_SPLITS = 50
N_TRIALS = 2000
MID_IDX  = N_GRID // 2   # index used for per-split Delta computation


def frac_label(frac: float) -> str:
    return FRAC_LABELS[frac].replace("/", "")   # "5050", "7030", etc.


# ---------------------------------------------------------------------------
# Split generation (simple random permutation, per spec)
# ---------------------------------------------------------------------------

def make_split_frac(n: int, frac: float, seed: int):
    rng   = np.random.default_rng(seed)
    perm  = rng.permutation(n)
    n_cal = int(n * frac)
    return perm[:n_cal], perm[n_cal:]


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str):
    print(f"\n{'='*60}\n  {dataset}\n{'='*60}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw, costs = load_full(dataset)

    # Filter NaN rows (consistent with fig2_compute). Accuracy-based pool and
    # pair choices are made inside each split using calibration data only.
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

    full_mean_c = {m: costs[m].mean() for m in raw}
    cost_grid   = np.linspace(min(full_mean_c.values()),
                              max(full_mean_c.values()), N_GRID)

    all_diagnostics = {}

    for frac in CAL_FRACS:
        label    = frac_label(frac)
        env_path = OUT_DIR / f"{dataset}_{label}_envelope.npy"
        kc_path  = OUT_DIR / f"{dataset}_{label}_kcascade.npy"
        diag_path = OUT_DIR / f"{dataset}_{label}_diagnostics.json"
        version_path = OUT_DIR / f"{dataset}_{label}_methodology_version.txt"
        cache_current = (
            version_path.exists()
            and version_path.read_text().strip() == CACHE_VERSION
        )

        if cache_current and env_path.exists() and kc_path.exists():
            print(f"  {FRAC_LABELS[frac]}: already computed - loading.")
            env_curves = np.load(env_path)
            kc_curves  = np.load(kc_path)
        else:
            env_curves = np.full((N_SPLITS, N_GRID), np.nan)
            kc_curves  = np.full((N_SPLITS, N_GRID), np.nan)

            t0 = time.time()
            for i in range(N_SPLITS):
                cal_idx, test_idx = make_split_frac(n, frac, seed=SEED + i)
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

                env_curves[i] = pairwise_envelope_curve(
                    test_data, split_models, score_ranges, cost_grid,
                    pairs=cal_pairs)
                kc_curves[i]  = kcascade_curve(
                    cal_data, test_data, split_models, score_ranges, cost_grid,
                    split_seed=SEED + i)

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (N_SPLITS - i - 1)
                    print(f"    {FRAC_LABELS[frac]}  split {i+1}/{N_SPLITS}"
                          f"  ETA {eta:.0f}s", flush=True)

            np.save(env_path, env_curves)
            np.save(kc_path,  kc_curves)
            version_path.write_text(CACHE_VERSION + "\n")
            print(f"  {FRAC_LABELS[frac]}: done in {time.time()-t0:.1f}s")

        diag = _compute_diagnostics(env_curves, kc_curves)
        all_diagnostics[FRAC_LABELS[frac]] = diag
        with open(diag_path, "w") as f:
            json.dump(diag, f, indent=2)

    return all_diagnostics


def _compute_diagnostics(env_curves: np.ndarray, kc_curves: np.ndarray) -> dict:
    env_at_mid = env_curves[:, MID_IDX]
    kc_at_mid  = kc_curves[:, MID_IDX]
    deltas     = kc_at_mid - env_at_mid

    env_bw  = float(np.nanpercentile(env_at_mid, 90) - np.nanpercentile(env_at_mid, 10))
    kc_bw   = float(np.nanpercentile(kc_at_mid,  90) - np.nanpercentile(kc_at_mid,  10))
    bwr     = kc_bw / env_bw if env_bw > 0 else float("nan")

    med_env = np.nanmedian(env_curves, axis=0)
    med_kc  = np.nanmedian(kc_curves,  axis=0)
    valid   = ~np.isnan(med_env) & ~np.isnan(med_kc)
    dom_frac = float(np.mean(med_kc[valid] > med_env[valid])) if valid.any() else float("nan")

    return {
        "median_delta":      float(np.nanmedian(deltas)),
        "p10_delta":         float(np.nanpercentile(deltas, 10)),
        "p90_delta":         float(np.nanpercentile(deltas, 90)),
        "band_width_ratio":  bwr,
        "dominance_frac":    dom_frac,
        "deltas":            deltas.tolist(),
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_sensitivity(all_diag: dict[str, dict]):
    """
    1x4 panels: Delta (k-cascade minus envelope at median cost) vs
    calibration fraction, with 10th-90th percentile error bars.
    """
    fig, axes = plt.subplots(1, 5, figsize=(8.5, 2.2))

    for ax, dataset in zip(axes, DATASETS):
        if dataset not in all_diag:
            ax.set_visible(False)
            continue

        fracs   = CAL_FRACS
        x_pos   = np.arange(len(fracs))
        medians, lo_bars, hi_bars = [], [], []

        for frac in fracs:
            d = all_diag[dataset][FRAC_LABELS[frac]]
            med = d["median_delta"]
            medians.append(med)
            lo_bars.append(med - d["p10_delta"])
            hi_bars.append(d["p90_delta"] - med)

        ax.errorbar(x_pos, medians, yerr=[lo_bars, hi_bars],
                    fmt="o-", capsize=3, color="coral", ms=4, lw=1.2)
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.set_xlabel("Cal. %", fontsize=8)
        ax.set_ylabel(r"$\Delta$ (k-casc.$-$env.)", fontsize=8)
        ax.set_title(DATASET_LABELS[dataset], fontsize=9)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([FRAC_AXIS_LABELS[f] for f in fracs], fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figA7_cal_sensitivity.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_diag: dict[str, dict]):
    header = f"{'':8s}"
    for ds in DATASETS:
        if ds in all_diag:
            header += f"  {DATASET_LABELS[ds]:>22s}"
    print("\n" + header)
    sub = f"{'':8s}" + "".join(
        f"  {'Delta':>6s}  {'BWR':>5s}"
        for ds in DATASETS if ds in all_diag
    )
    print(sub)

    for frac in CAL_FRACS:
        row = f"{FRAC_LABELS[frac]:8s}"
        for ds in DATASETS:
            if ds not in all_diag:
                continue
            d = all_diag[ds][FRAC_LABELS[frac]]
            row += (f"  {d['median_delta']:+.4f}"
                    f"  {d['band_width_ratio']:5.2f}")
        print(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = sys.argv[1:] or DATASETS

    all_diag = {}
    for ds in datasets:
        all_diag[ds] = run_dataset(ds)

    print_summary(all_diag)
    plot_sensitivity(all_diag)
    print("\nDone.")
