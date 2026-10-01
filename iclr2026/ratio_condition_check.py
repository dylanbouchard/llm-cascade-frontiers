"""Empirical check for decreasing benefit-cost ratio.

For each pair, estimate

    (m_H(s) - m_L(s)) / E[C_H | s_L = s]

over cheap-model confidence bins and test whether it is weakly decreasing in
the cheap model confidence score. This is the empirical counterpart to a
variable-cost concavity condition.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cascade_core import MODEL_COST_ORDER, compute_costs, load_all_models
from figures import _select_representative_pair


OUT_DIR = Path("results/ratio_condition")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
SCORER = "mean_token_negentropy"
N_BINS = 20
N_SPLITS = 50
SEED = 42


def binned_ratio(
    s: np.ndarray,
    u_l: np.ndarray,
    u_h: np.ndarray,
    c_h: np.ndarray,
    idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_i = s[idx]
    benefit_i = (u_h[idx] - u_l[idx]).astype(float)
    c_h_i = c_h[idx].astype(float)
    valid = np.isfinite(s_i) & np.isfinite(benefit_i) & np.isfinite(c_h_i) & (c_h_i > 0)
    s_i = s_i[valid]
    benefit_i = benefit_i[valid]
    c_h_i = c_h_i[valid]
    if len(s_i) < N_BINS:
        empty = np.full(N_BINS, np.nan)
        return empty, empty, empty, empty

    quantiles = np.linspace(0, 1, N_BINS + 1)
    edges = np.quantile(s_i, quantiles)
    edges = np.unique(edges)
    if len(edges) < 4:
        empty = np.full(N_BINS, np.nan)
        return empty, empty, empty, empty

    centers, benefits, gammas, ratios = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == edges[-1]:
            mask = (s_i >= lo) & (s_i <= hi)
        else:
            mask = (s_i >= lo) & (s_i < hi)
        if mask.sum() < 10:
            continue
        b = float(np.mean(benefit_i[mask]))
        g = float(np.mean(c_h_i[mask]))
        centers.append(float(np.mean(s_i[mask])))
        benefits.append(b)
        gammas.append(g)
        ratios.append(b / g if g > 0 else np.nan)
    return np.array(centers), np.array(benefits), np.array(gammas), np.array(ratios)


def diagnostics_for_pair(df_l: pd.DataFrame, df_h: pd.DataFrame, c_h: np.ndarray) -> dict:
    valid = (
        df_l[SCORER].notna().values
        & df_l["correct"].notna().values
        & df_h["correct"].notna().values
        & np.isfinite(c_h)
    )
    idx_all = np.where(valid)[0]
    s = df_l[SCORER].to_numpy(dtype=float)
    u_l = df_l["correct"].to_numpy(dtype=float)
    u_h = df_h["correct"].to_numpy(dtype=float)

    rng = np.random.default_rng(SEED)
    split_records = []
    curves = []
    for split in range(N_SPLITS):
        idx = rng.choice(idx_all, size=len(idx_all) // 2, replace=False)
        centers, benefits, gammas, ratios = binned_ratio(s, u_l, u_h, c_h, idx)
        ok = np.isfinite(ratios)
        if ok.sum() < 3:
            continue
        order = np.argsort(centers[ok])
        ratio_ordered = ratios[ok][order]
        benefit_ordered = benefits[ok][order]
        gamma_ordered = gammas[ok][order]
        diffs = np.diff(ratio_ordered)
        rho, _ = spearmanr(np.arange(len(ratio_ordered)), ratio_ordered)
        split_records.append({
            "split": split,
            "ratio_decreasing_frac": float(np.mean(diffs <= 0)),
            "ratio_max_reversal": float(np.max(np.maximum(diffs, 0))),
            "ratio_spearman": float(rho),
            "benefit_decreasing_frac": float(np.mean(np.diff(benefit_ordered) <= 0)),
            "gamma_decreasing_frac": float(np.mean(np.diff(gamma_ordered) <= 0)),
            "n_bins": int(ok.sum()),
        })
        curves.append(ratio_ordered)

    if not split_records:
        return {}
    split_df = pd.DataFrame(split_records)
    return {
        "ratio_decreasing_frac_median": float(split_df["ratio_decreasing_frac"].median()),
        "ratio_decreasing_frac_p10": float(split_df["ratio_decreasing_frac"].quantile(0.10)),
        "ratio_decreasing_frac_p90": float(split_df["ratio_decreasing_frac"].quantile(0.90)),
        "ratio_max_reversal_median": float(split_df["ratio_max_reversal"].median()),
        "ratio_spearman_median": float(split_df["ratio_spearman"].median()),
        "share_ratio_spearman_negative": float(np.mean(split_df["ratio_spearman"] < 0)),
        "benefit_decreasing_frac_median": float(split_df["benefit_decreasing_frac"].median()),
        "gamma_decreasing_frac_median": float(split_df["gamma_decreasing_frac"].median()),
        "n_valid_splits": int(len(split_df)),
    }


def run_dataset(dataset: str) -> list[dict]:
    data = load_all_models(dataset)
    costs = {model: compute_costs(df, model) for model, df in data.items()}
    mean_cost = {model: float(costs[model].mean()) for model in data}
    mean_acc = {model: float(data[model]["correct"].mean()) for model in data}
    models_by_cost = sorted(data, key=lambda model: mean_cost[model])
    rep_pair = _select_representative_pair(dataset).replace("→", "->")

    rows = []
    for cheap, expensive in combinations(models_by_cost, 2):
        if mean_acc[expensive] <= mean_acc[cheap]:
            continue
        stats = diagnostics_for_pair(data[cheap], data[expensive], costs[expensive])
        if not stats:
            continue
        pair = f"{cheap}->{expensive}"
        rows.append({
            "dataset": dataset,
            "pair": pair,
            "cheap_model": cheap,
            "expensive_model": expensive,
            "is_representative": pair == rep_pair,
            "mean_acc_gain": mean_acc[expensive] - mean_acc[cheap],
            "mean_cost_ratio": mean_cost[cheap] / mean_cost[expensive],
            **stats,
        })
    return rows


def main():
    all_rows = []
    for dataset in DATASETS:
        print(f"Running {dataset}", flush=True)
        all_rows.extend(run_dataset(dataset))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "pair_ratio_condition.csv", index=False)

    summary = (
        df.groupby("dataset")
        .agg(
            n_pairs=("pair", "count"),
            median_ratio_dec=("ratio_decreasing_frac_median", "median"),
            p10_ratio_dec=("ratio_decreasing_frac_median", lambda x: np.quantile(x, 0.10)),
            p90_ratio_dec=("ratio_decreasing_frac_median", lambda x: np.quantile(x, 0.90)),
            median_benefit_dec=("benefit_decreasing_frac_median", "median"),
            median_gamma_dec=("gamma_decreasing_frac_median", "median"),
            median_ratio_spearman=("ratio_spearman_median", "median"),
            share_pairs_negative_spearman=("ratio_spearman_median", lambda x: np.mean(x < 0)),
            share_ratio_dec_ge_090=("ratio_decreasing_frac_median", lambda x: np.mean(x >= 0.90)),
            share_ratio_dec_ge_095=("ratio_decreasing_frac_median", lambda x: np.mean(x >= 0.95)),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "dataset_ratio_condition_summary.csv", index=False)
    rep = df[df["is_representative"]].copy()
    rep.to_csv(OUT_DIR / "representative_ratio_condition.csv", index=False)

    print("\nRepresentative pairs")
    print(rep[[
        "dataset", "pair", "ratio_decreasing_frac_median",
        "ratio_decreasing_frac_p10", "ratio_decreasing_frac_p90",
        "ratio_spearman_median", "share_ratio_spearman_negative",
        "benefit_decreasing_frac_median", "gamma_decreasing_frac_median",
    ]].round(3).to_string(index=False))
    print("\nAll valid pairs summary")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
