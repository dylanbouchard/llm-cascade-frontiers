"""
Concavity check for representative two-model threshold frontiers.

For each dataset and split, this script evaluates the representative envelope
pair's realized held-out cost-quality frontier from a threshold sweep, computes
the concave hull of the Paretoized frontier, and reports how closely the
frontier tracks its hull.

Outputs:
  results/concavity_check/per_split.csv
  results/concavity_check/summary.csv
  results/concavity_check/summary.json

Usage:
  .venv/bin/python3 concavity_check.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from fig2_compute import (
    N_SPLITS,
    N_THRESH,
    SEED,
    load_full,
    make_split,
    MODEL_COST_ORDER,
)
from figures import _select_representative_pair
from optuna_frontier import SCORER


OUT_DIR = Path("results/concavity_check")
DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
EPS_FRAC = 0.01
N_COST_BINS = 10


def filter_valid_rows(raw: dict[str, pd.DataFrame], costs: dict[str, np.ndarray]):
    valid = np.ones(len(next(iter(raw.values()))), dtype=bool)
    for model in raw:
        valid &= raw[model][SCORER].notna().values
        valid &= raw[model]["correct"].notna().values
    idx = np.where(valid)[0]
    for model in raw:
        raw[model] = raw[model].iloc[idx].reset_index(drop=True)
        costs[model] = costs[model][idx]
    return raw, costs, valid


def representative_pair(dataset: str) -> tuple[str, str]:
    pair = _select_representative_pair(dataset)
    cheap, exp = pair.split("→")
    return cheap, exp


def pareto_frontier(cost: np.ndarray, quality: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"cost": cost, "quality": quality})
    df = df.groupby("cost", as_index=False)["quality"].max().sort_values("cost")
    x = df["cost"].to_numpy(dtype=float)
    y = np.maximum.accumulate(df["quality"].to_numpy(dtype=float))
    keep = np.concatenate(([True], np.diff(x) > 1e-15))
    return x[keep], y[keep]


def concave_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Least concave majorant evaluated at x.

    Assumes x is strictly increasing and y is the Paretoized frontier. The
    returned hull is the upper piecewise-linear envelope with non-increasing
    slopes.
    """
    if len(x) <= 2:
        return y.copy()

    knots: list[int] = []
    for i in range(len(x)):
        knots.append(i)
        while len(knots) >= 3:
            a, b, c = knots[-3], knots[-2], knots[-1]
            slope_ab = (y[b] - y[a]) / (x[b] - x[a])
            slope_bc = (y[c] - y[b]) / (x[c] - x[b])
            if slope_ab + 1e-15 < slope_bc:
                knots.pop(-2)
            else:
                break
    return np.interp(x, x[knots], y[knots])


def binned_frontier(x: np.ndarray, y: np.ndarray, n_bins: int = N_COST_BINS):
    """Average quality within cost bins and Paretoize the resulting frontier."""
    if len(x) < 3:
        return x, y
    edges = np.linspace(float(np.min(x)), float(np.max(x)), n_bins + 1)
    centers, qualities = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (x >= lo) & (x <= hi)
        else:
            mask = (x >= lo) & (x < hi)
        if np.any(mask):
            centers.append(float(np.mean(x[mask])))
            qualities.append(float(np.mean(y[mask])))
    if len(centers) < 2:
        return x, y
    return pareto_frontier(np.asarray(centers), np.asarray(qualities))


def concavity_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    hull = concave_hull(x, y)
    gap = np.maximum(hull - y, 0.0)
    quality_range = float(np.nanmax(y) - np.nanmin(y))
    epsilon = EPS_FRAC * quality_range
    cost_range = float(np.nanmax(x) - np.nanmin(x))
    if cost_range <= 0 or len(x) < 2:
        concave_share = float("nan")
        weighted_share = float("nan")
    else:
        concave_share = float(np.mean(gap <= epsilon + 1e-15))
        dx = np.diff(x)
        seg_ok = (gap[:-1] <= epsilon + 1e-15) & (gap[1:] <= epsilon + 1e-15)
        weighted_share = float(np.sum(dx[seg_ok]) / cost_range)
    return {
        "concave_share": concave_share,
        "cost_weighted_concave_share": weighted_share,
        "max_gap": float(np.nanmax(gap)),
        "mean_gap": float(np.nanmean(gap)),
        "epsilon": epsilon,
        "quality_range": quality_range,
        "cost_range": cost_range,
        "n_frontier_points": len(x),
    }


def frontier_for_split(
    raw: dict[str, pd.DataFrame],
    costs: dict[str, np.ndarray],
    cheap: str,
    exp: str,
    cal_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    s_cal = raw[cheap][SCORER].values[cal_idx]
    s = raw[cheap][SCORER].values[test_idx]
    u_l = raw[cheap]["correct"].values[test_idx].astype(float)
    u_h = raw[exp]["correct"].values[test_idx].astype(float)
    c_l = costs[cheap][test_idx]
    c_h = costs[exp][test_idx]

    tau_grid = np.concatenate(
        ([-np.inf], np.linspace(float(s_cal.min()), float(s_cal.max()), N_THRESH), [np.inf])
    )
    c_vals = np.empty(len(tau_grid))
    q_vals = np.empty(len(tau_grid))
    for j, tau in enumerate(tau_grid):
        esc = s < tau
        c_vals[j] = float((c_l + esc.astype(float) * c_h).mean())
        q_vals[j] = float(np.where(esc, u_h, u_l).mean())
    return pareto_frontier(c_vals, q_vals)


def run_dataset(dataset: str) -> list[dict]:
    print(f"\n{'=' * 60}\n  {dataset}\n{'=' * 60}")
    raw, costs = load_full(dataset)
    raw, costs, valid_mask = filter_valid_rows(raw, costs)
    cheap, exp = representative_pair(dataset)
    n = len(next(iter(raw.values())))
    ref_model = next(model for model in MODEL_COST_ORDER if model in raw)
    ref_correct = raw[ref_model]["correct"].values.astype(int)
    print(f"  Pair: {cheap} -> {exp}")
    print(f"  Valid rows after NaN filter: {n} / {valid_mask.size}")

    records = []
    for split_i in range(N_SPLITS):
        cal_idx, test_idx = make_split(n, ref_correct, seed=SEED + split_i)
        x, y = frontier_for_split(raw, costs, cheap, exp, cal_idx, test_idx)
        metrics = concavity_metrics(x, y)
        xb, yb = binned_frontier(x, y)
        binned_metrics = concavity_metrics(xb, yb)

        record = {
            "dataset": dataset,
            "split": split_i,
            "pair": f"{cheap}→{exp}",
        }
        record.update(metrics)
        record.update({f"binned_{k}": v for k, v in binned_metrics.items()})
        records.append(record)
        if (split_i + 1) % 10 == 0:
            print(f"  split {split_i + 1:2d}/{N_SPLITS}", flush=True)
    return records


def bootstrap_ci(values: np.ndarray, n_boot: int = 200, seed: int = SEED):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = rng.choice(values, size=len(values), replace=True).mean()
    return (
        float(np.median(values)),
        float(np.percentile(values, 10)),
        float(np.percentile(values, 90)),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "concave_share",
        "cost_weighted_concave_share",
        "max_gap",
        "mean_gap",
        "binned_concave_share",
        "binned_cost_weighted_concave_share",
        "binned_max_gap",
        "binned_mean_gap",
        "epsilon",
        "binned_epsilon",
        "quality_range",
        "n_frontier_points",
        "binned_n_frontier_points",
    ]
    for dataset, sub in df.groupby("dataset"):
        row = {
            "dataset": dataset,
            "pair": sub["pair"].iloc[0],
            "n_splits": len(sub),
        }
        for metric in metrics:
            med, p10, p90, lo, hi = bootstrap_ci(sub[metric].values)
            row[f"{metric}_median"] = med
            row[f"{metric}_p10"] = p10
            row[f"{metric}_p90"] = p90
            row[f"{metric}_mean_ci_lo"] = lo
            row[f"{metric}_mean_ci_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def main(args: list[str]) -> None:
    datasets = args or DATASETS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []
    for dataset in datasets:
        path = OUT_DIR / f"{dataset}_per_split.csv"
        if path.exists():
            print(f"\n{dataset}: loading cached results.")
            records = pd.read_csv(path).to_dict("records")
        else:
            records = run_dataset(dataset)
            pd.DataFrame(records).to_csv(path, index=False)
        all_records.extend(records)

    per_split = pd.DataFrame(all_records)
    per_split.to_csv(OUT_DIR / "per_split.csv", index=False)
    summary = summarize(per_split)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(json.loads(summary.to_json(orient="records")), f, indent=2)

    cols = [
        "dataset",
        "pair",
        "concave_share_median",
        "concave_share_p10",
        "concave_share_p90",
        "max_gap_median",
        "max_gap_p10",
        "max_gap_p90",
        "binned_concave_share_median",
        "binned_max_gap_median",
        "epsilon_median",
    ]
    print("\nSummary:")
    print(summary[cols].round(4).to_string(index=False))
    print(f"\nSaved {OUT_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:])
