"""
cost_variability.py - Appendix: cost/score independence diagnostics.

For each dataset and cost-ordered pair (L,H), computes Spearman r between the
cheap model's confidence score s_L and the expensive model's realized cost C_H.
This directly checks the score-independent expected escalation cost condition
used in Proposition 2.

Saves:
  results/cost_score_correlations.csv          - pair-level diagnostics
  results/cost_score_correlation_summary.csv   - dataset-level summaries

Usage:
    python3 cost_variability.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from cascade_core import PRICE_PER_TOKEN, DATA_DIR, DATASET_EXCLUSIONS, _enc

DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
DATASET_LABELS = {"mmlu": "MMLU", "triviaqa": "TriviaQA",
                  "math_hard": "MATH", "simpleqa": "SimpleQA",
                  "livecodebench": "LiveCodeBench"}

MODEL_ORDER = [
    "llama-3.1-8b", "qwen2.5-7b", "gpt-4o-mini", "gpt-oss-20b",
    "deepseek-v3", "llama-3.3-70b", "gpt-4o", "MiniMax-M2.7",
]
MODEL_LABELS = {
    "gpt-4o-mini":   "GPT-4o-mini",
    "llama-3.1-8b":  "Llama-3.1-8B",
    "qwen2.5-7b":    "Qwen2.5-7B",
    "deepseek-v3":   "DeepSeek-V3",
    "llama-3.3-70b": "Llama-3.3-70B",
    "gpt-4o":        "GPT-4o",
    "gpt-oss-20b":   "GPT-oss-20B",
    "MiniMax-M2.7":  "MiniMax-M2.7",
}

RESULT_DIR = Path("results")
SCORER = "mean_token_negentropy"


def compute_costs_from_tokens(df: pd.DataFrame, model: str, n_in: np.ndarray) -> np.ndarray:
    p_in, p_out = PRICE_PER_TOKEN[model]
    n_out = df["logprob"].apply(len).to_numpy()
    return n_in * p_in + n_out * p_out


def run():
    pair_records = []
    cv_records = []

    for dataset in DATASETS:
        frames = {}
        excluded = set(DATASET_EXCLUSIONS.get(dataset, []))

        for model in [m for m in MODEL_ORDER if m not in excluded]:
            path = DATA_DIR / f"{dataset}-{model}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path).iloc[:2000].reset_index(drop=True)
            if SCORER not in df.columns:
                continue
            frames[model] = df

        if not frames:
            continue

        prompt_source = next(iter(frames.values()))
        n_in = prompt_source["prompt"].apply(lambda p: len(_enc.encode(p))).to_numpy()

        for model, df in frames.items():
            costs = compute_costs_from_tokens(df, model, n_in)
            cv = float(np.nanstd(costs) / np.nanmean(costs)) if np.nanmean(costs) > 0 else np.nan
            cv_records.append({
                "dataset": dataset,
                "model": model,
                "cv": cv,
                "mean_cost_usd": float(np.nanmean(costs)),
            })

        available = [m for m in MODEL_ORDER if m in frames]
        costs_by_model = {m: compute_costs_from_tokens(frames[m], m, n_in) for m in available}

        for i, cheap in enumerate(available):
            scores = frames[cheap][SCORER].to_numpy(dtype=float)
            for expensive in available[i + 1:]:
                expensive_cost = costs_by_model[expensive]
                valid = np.isfinite(scores) & np.isfinite(expensive_cost)
                rho, pval = (
                    spearmanr(scores[valid], expensive_cost[valid])
                    if valid.sum() > 1 else (np.nan, np.nan)
                )
                pair_records.append({
                    "dataset": dataset,
                    "cheap_model": cheap,
                    "expensive_model": expensive,
                    "spearman_score_cost": float(rho),
                    "abs_spearman_score_cost": float(abs(rho)),
                    "p_value": float(pval),
                    "n": int(valid.sum()),
                })

    pair_df = pd.DataFrame(pair_records)
    cv_df = pd.DataFrame(cv_records)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(RESULT_DIR / "cost_score_correlations.csv", index=False)
    cv_df.to_csv(RESULT_DIR / "cost_variability.csv", index=False)

    summary = (
        pair_df
        .groupby("dataset")
        .agg(
            n_pairs=("spearman_score_cost", "count"),
            median_abs_r=("abs_spearman_score_cost", "median"),
            p90_abs_r=("abs_spearman_score_cost", lambda x: np.nanpercentile(x, 90)),
            max_abs_r=("abs_spearman_score_cost", "max"),
            share_abs_r_lt_0_10=("abs_spearman_score_cost", lambda x: np.nanmean(x < 0.10)),
            share_abs_r_lt_0_20=("abs_spearman_score_cost", lambda x: np.nanmean(x < 0.20)),
        )
        .reset_index()
    )
    summary.to_csv(RESULT_DIR / "cost_score_correlation_summary.csv", index=False)

    print("\nPair-level Spearman r(s_L, C_H) summary:")
    display = summary.copy()
    display["dataset"] = display["dataset"].map(DATASET_LABELS)
    print(display.round(3).to_string(index=False))

    print("\nPer-model CV of realized cost:")
    cv_table = cv_df.pivot(index="model", columns="dataset", values="cv")
    cv_table = cv_table.reindex(index=[m for m in MODEL_ORDER if m in cv_table.index],
                                columns=DATASETS)
    cv_table.columns = [DATASET_LABELS[d] for d in cv_table.columns]
    cv_table.index = [MODEL_LABELS[m] for m in cv_table.index]
    print(cv_table.round(3).to_string())

    return pair_df, summary


if __name__ == "__main__":
    run()
