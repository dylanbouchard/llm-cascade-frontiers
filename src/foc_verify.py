"""
foc_verify.py - Appendix: FOC Verification.

For each NSGA-II Pareto-optimal k=2 trial, checks that the empirical
two-model first-order condition holds:
    (m_H(tau*) - m_L(tau*)) / c_H  ≈  d U / d C  at the optimized threshold.

Left-hand side (analytical slope): estimated from conditional accuracy
  binned on calibration data.
Right-hand side (numerical slope): finite difference of the per-pair
  threshold-sweep frontier on test data.

If these agree, the optimizer finds FOC-satisfying stationary points.

Outputs per dataset (in results/foc/):
  {dataset}_slopes.npz          - analytical + numerical slope arrays
  {dataset}_agreement_gaps.npy  - trial_quality - frontier_interp(trial_cost)
  {dataset}_residuals.npy       - normalised (analytical - numerical) / analytical

Combined figure:
  figures/figA_foc_verification.pdf  - 1x4 analytical vs numerical slope scatter

Usage:
    python3 foc_verify.py [dataset ...]
"""

import sys, json
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
from itertools import combinations
from scipy.interpolate import interp1d

from fig2_compute import (
    load_full, non_dominated_models, make_split, slice_data,
    make_objective_nd, N_THRESH, POP_SIZE, SEED, MAX_K,
    CACHE_VERSION,
)
from optuna_frontier import simulate_cascade

optuna.logging.set_verbosity(optuna.logging.WARNING)

OUT_DIR  = Path("results/foc")
FIG_DIR  = Path("figures")
DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
DATASET_LABELS = {"mmlu": "MMLU", "triviaqa": "TriviaQA",
                  "math_hard": "MATH", "simpleqa": "SimpleQA",
                  "livecodebench": "LiveCodeBench"}
N_SPLITS = 50
N_TRIALS = 2000
N_BINS   = 30
N_THRESH_PAIR = 200


# ---------------------------------------------------------------------------
# Per-pair test frontier
# ---------------------------------------------------------------------------

def per_pair_frontier(cheap, exp, test_data, score_ranges):
    s     = test_data[cheap]["scores"]
    u_L   = test_data[cheap]["correct"]
    u_H   = test_data[exp]["correct"]
    c_L   = test_data[cheap]["costs"]
    c_H   = test_data[exp]["costs"]

    tau_grid = np.linspace(score_ranges[cheap][0], score_ranges[cheap][1],
                           N_THRESH_PAIR)
    cs = [c_L.mean()]
    qs = [u_L.mean()]
    for tau in tau_grid:
        esc  = s < tau
        cs.append(c_L.mean() + (esc * c_H).mean())
        qs.append(np.where(esc, u_H, u_L).mean())
    cs.append(c_L.mean() + c_H.mean())
    qs.append(u_H.mean())

    cs_arr = np.array(cs)
    qs_arr = np.array(qs)
    order  = np.argsort(cs_arr)
    return cs_arr[order], qs_arr[order]


def numerical_slope_at(cs, qs, target_cost):
    """Central finite difference of (cs, qs) at the point nearest target_cost."""
    idx = int(np.argmin(np.abs(cs - target_cost)))
    if 0 < idx < len(cs) - 1 and (cs[idx + 1] - cs[idx - 1]) > 0:
        return float((qs[idx + 1] - qs[idx - 1]) / (cs[idx + 1] - cs[idx - 1]))
    return np.nan


# ---------------------------------------------------------------------------
# Conditional accuracy estimation from calibration
# ---------------------------------------------------------------------------

def estimate_m_at_tau(tau, s_cal, corr_L_cal, corr_H_cal):
    """
    Bin s_cal into N_BINS bins, compute mean correctness per bin for L and H,
    then linearly interpolate to tau.
    """
    lo, hi = s_cal.min(), s_cal.max()
    if hi <= lo:
        return np.nan, np.nan

    edges   = np.linspace(lo, hi, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bidx    = np.clip(np.digitize(s_cal, edges) - 1, 0, N_BINS - 1)

    m_L = np.full(N_BINS, np.nan)
    m_H = np.full(N_BINS, np.nan)
    for b in range(N_BINS):
        mask = bidx == b
        if mask.sum() > 3:
            m_L[b] = corr_L_cal[mask].mean()
            m_H[b] = corr_H_cal[mask].mean()

    valid = ~np.isnan(m_L) & ~np.isnan(m_H)
    if valid.sum() < 2:
        return np.nan, np.nan

    f_L = interp1d(centers[valid], m_L[valid],
                   bounds_error=False, fill_value="extrapolate")
    f_H = interp1d(centers[valid], m_H[valid],
                   bounds_error=False, fill_value="extrapolate")
    return float(f_L(tau)), float(f_H(tau))


# ---------------------------------------------------------------------------
# Per-dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str) -> dict:
    print(f"\n{'='*60}\n  {dataset}\n{'='*60}")
    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    slopes_path = out_dir / "slopes.npz"
    version_path = out_dir / "methodology_version.txt"
    cache_current = (
        version_path.exists()
        and version_path.read_text().strip() == CACHE_VERSION
    )
    if cache_current and slopes_path.exists():
        print("  Already computed - loading.")
        npz = np.load(slopes_path)
        return {
            "analytical": npz["analytical"],
            "numerical":  npz["numerical"],
            "gaps":        np.load(out_dir / "agreement_gaps.npy"),
            "residuals":   np.load(out_dir / "residuals.npy"),
        }

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
    print(f"  Valid rows: {n}; pool selected from calibration split only.")

    ref_model = next(iter(raw))
    ref_correct = raw[ref_model]["correct"].values.astype(int)

    analytical_slopes, numerical_slopes, agreement_gaps = [], [], []

    for split_i in range(N_SPLITS):
        cal_idx, test_idx = make_split(n, ref_correct, seed=SEED + split_i)
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

        # Run NSGA-II on calibration
        obj, seqs = make_objective_nd(cal_data, split_models, score_ranges)
        sampler = optuna.samplers.NSGAIISampler(population_size=POP_SIZE,
                                                seed=SEED + split_i)
        study = optuna.create_study(directions=["minimize", "minimize"],
                                    sampler=sampler)
        study.optimize(obj, n_trials=N_TRIALS)

        # Extract k=2 Pareto trials
        k2_trials = [t for t in study.best_trials
                     if len(seqs[t.params["seq_idx"]]) == 2]

        for t in k2_trials:
            seq   = seqs[t.params["seq_idx"]]
            cheap, exp = seq[0], seq[1]

            # Reconstruct threshold from normalised param
            tau_norm = t.params["tau_0"]
            lo, hi   = score_ranges[cheap]
            tau_star = lo + tau_norm * (hi - lo)

            # Test-evaluated (cost, quality) for this trial
            thresholds = [tau_star]
            test_cost, test_quality = simulate_cascade(
                list(seq), thresholds, test_data)

            # Per-pair test frontier and numerical slope
            fc, fq = per_pair_frontier(cheap, exp, test_data, score_ranges)
            num_sl = numerical_slope_at(fc, fq, test_cost)

            # Frontier agreement gap
            f_interp  = interp1d(fc, fq, bounds_error=False, fill_value=np.nan)
            gap       = float(test_quality - float(f_interp(test_cost)))

            # Conditional accuracy from calibration → analytical slope
            s_cal      = cal_data[cheap]["scores"]
            corr_L_cal = cal_data[cheap]["correct"]
            corr_H_cal = cal_data[exp]["correct"]
            m_L_tau, m_H_tau = estimate_m_at_tau(
                tau_star, s_cal, corr_L_cal, corr_H_cal)

            c_H_mean = test_data[exp]["costs"].mean()
            if np.isnan(m_L_tau) or np.isnan(m_H_tau) or c_H_mean <= 0:
                continue

            anal_sl = (m_H_tau - m_L_tau) / c_H_mean

            analytical_slopes.append(anal_sl)
            numerical_slopes.append(num_sl)
            agreement_gaps.append(gap)

        if (split_i + 1) % 10 == 0:
            print(f"  split {split_i+1}/{N_SPLITS}  "
                  f"k2_trials={len(k2_trials)}  "
                  f"collected={len(analytical_slopes)}", flush=True)

    anal  = np.array(analytical_slopes)
    num   = np.array(numerical_slopes)
    gaps  = np.array(agreement_gaps)

    # Normalised FOC residual: (anal - num) / |anal|, dropping NaN
    valid   = ~np.isnan(anal) & ~np.isnan(num) & (np.abs(anal) > 1e-9)
    resid   = np.full(len(anal), np.nan)
    resid[valid] = (anal[valid] - num[valid]) / np.abs(anal[valid])

    np.savez(out_dir / "slopes.npz", analytical=anal, numerical=num)
    np.save(out_dir / "agreement_gaps.npy", gaps)
    np.save(out_dir / "residuals.npy", resid)
    version_path.write_text(CACHE_VERSION + "\n")

    _print_summary(dataset, anal, num, gaps, resid)
    return {"analytical": anal, "numerical": num, "gaps": gaps, "residuals": resid}


def _print_summary(dataset, anal, num, gaps, resid):
    valid_sl = ~np.isnan(anal) & ~np.isnan(num)
    valid_r  = ~np.isnan(resid)
    valid_g  = ~np.isnan(gaps)
    print(f"\n  {DATASET_LABELS[dataset]} summary ({valid_sl.sum()} valid slope pairs):")
    if valid_sl.sum() > 0:
        print(f"    Analytical slope: median={np.nanmedian(anal):.4f}, "
              f"IQR=[{np.nanpercentile(anal,25):.4f}, {np.nanpercentile(anal,75):.4f}]")
        print(f"    Numerical slope:  median={np.nanmedian(num):.4f}, "
              f"IQR=[{np.nanpercentile(num,25):.4f}, {np.nanpercentile(num,75):.4f}]")
    if valid_r.sum() > 0:
        print(f"    |FOC residual|: median={np.nanmedian(np.abs(resid[valid_r])):.4f}, "
              f"90th={np.nanpercentile(np.abs(resid[valid_r]), 90):.4f}")
    if valid_g.sum() > 0:
        print(f"    |Agreement gap|: median={np.nanmedian(np.abs(gaps[valid_g])):.5f}, "
              f"90th={np.nanpercentile(np.abs(gaps[valid_g]), 90):.5f}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_foc(all_results: dict[str, dict]):
    fig, axes = plt.subplots(1, 5, figsize=(8.5, 2.2))

    for ax, dataset in zip(axes, DATASETS):
        if dataset not in all_results:
            ax.set_visible(False)
            continue

        anal = all_results[dataset]["analytical"]
        num  = all_results[dataset]["numerical"]
        valid = ~np.isnan(anal) & ~np.isnan(num)

        ax.scatter(anal[valid], num[valid],
                   s=6, alpha=0.25, color="steelblue", edgecolors="none", zorder=3)

        # Identity line over the data range
        if valid.sum() > 0:
            lo = min(np.nanmin(anal[valid]), np.nanmin(num[valid]))
            hi = max(np.nanmax(anal[valid]), np.nanmax(num[valid]))
            pad = 0.05 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    "k--", lw=0.8, alpha=0.6, zorder=2)

        med_resid = np.nanmedian(np.abs(
            all_results[dataset]["residuals"][
                ~np.isnan(all_results[dataset]["residuals"])]))

        ax.text(0.97, 0.05, f"med $|r|={med_resid:.2f}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)

        ax.set_xlabel(r"Analytical slope $(m_H\!-\!m_L)/c_H$", fontsize=6.5)
        ax.set_ylabel("Numerical slope", fontsize=6.5)
        ax.set_title(DATASET_LABELS[dataset], fontsize=9)
        ax.tick_params(labelsize=6.5)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.8)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figA_foc_verification.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {out}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_table(all_results: dict[str, dict]):
    print(f"\n{'Metric':<22s}" + "".join(f"  {DATASET_LABELS[ds]:>12s}"
                                          for ds in DATASETS if ds in all_results))
    for label, key, transform in [
        ("Median |FOC resid|",  "residuals", lambda x: np.nanmedian(np.abs(x[~np.isnan(x)]))),
        ("90th  |FOC resid|",   "residuals", lambda x: np.nanpercentile(np.abs(x[~np.isnan(x)]), 90)),
        ("Median |agree gap|",  "gaps",      lambda x: np.nanmedian(np.abs(x[~np.isnan(x)]))),
        ("90th  |agree gap|",   "gaps",      lambda x: np.nanpercentile(np.abs(x[~np.isnan(x)]), 90)),
    ]:
        row = f"{label:<22s}"
        for ds in DATASETS:
            if ds not in all_results:
                continue
            arr = all_results[ds][key]
            val = transform(arr) if len(arr) > 0 else float("nan")
            row += f"  {val:>12.4f}"
        print(row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = sys.argv[1:] or DATASETS

    all_results = {}
    for ds in datasets:
        all_results[ds] = run_dataset(ds)

    print_table(all_results)
    plot_foc(all_results)
    print("\nDone.")
