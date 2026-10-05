"""Pairwise correlations of mean token negentropy across model outputs.

Reads only prompts and the primary confidence scorer from each response file.
Each pair uses queries with finite scores for both models. Run from the project
root with ``.venv/bin/python confidence_score_correlations.py``.
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data/output_data")
OUT_DIR = Path("results/confidence_score_correlations")
DATASETS = ("mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench")
N_ROWS = {"mmlu": 2000, "triviaqa": 2000, "math_hard": 2000,
          "simpleqa": 2000, "livecodebench": 1055}
SCORER = "mean_token_negentropy"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = []
    summary = []

    for dataset in DATASETS:
        paths = sorted(DATA_DIR.glob(f"{dataset}-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No response files for {dataset}")

        scores = {}
        reference_prompts = None
        for path in paths:
            model = path.name.removeprefix(f"{dataset}-").removesuffix(".parquet")
            df = pd.read_parquet(path, columns=["prompt", SCORER]).iloc[:N_ROWS[dataset]]
            prompts = df["prompt"].to_numpy()
            if reference_prompts is None:
                reference_prompts = prompts
            elif not np.array_equal(reference_prompts, prompts):
                raise ValueError(f"Prompts are not row aligned in {path}")
            scores[model] = pd.to_numeric(df[SCORER], errors="coerce").to_numpy(dtype=float)

        frame = pd.DataFrame(scores)
        finite = frame.replace([np.inf, -np.inf], np.nan)
        corr = finite.corr(method="pearson", min_periods=2)
        corr.to_csv(OUT_DIR / f"{dataset}_pearson_matrix.csv", float_format="%.6f")

        rows = []
        for model_a, model_b in combinations(frame.columns, 2):
            pair = finite[[model_a, model_b]].dropna()
            rows.append({
                "dataset": dataset,
                "model_a": model_a,
                "model_b": model_b,
                "n_pair": len(pair),
                "pearson_r": pair[model_a].corr(pair[model_b], method="pearson"),
                "spearman_rho": pair[model_a].corr(pair[model_b], method="spearman"),
            })
        pairs = pd.DataFrame(rows)
        pairs.to_csv(OUT_DIR / f"{dataset}_pairs.csv", index=False, float_format="%.6f")
        all_pairs.append(pairs)

        values = pairs["pearson_r"]
        summary.append({
            "dataset": dataset,
            "n_queries": len(frame),
            "n_models": len(scores),
            "n_pairs": len(pairs),
            "min_finite_per_model": int(finite.notna().sum().min()),
            "max_finite_per_model": int(finite.notna().sum().max()),
            "min_pair_n": int(pairs["n_pair"].min()),
            "median_pearson_r": values.median(),
            "min_pearson_r": values.min(),
            "max_pearson_r": values.max(),
            "median_spearman_rho": pairs["spearman_rho"].median(),
        })

    pd.concat(all_pairs, ignore_index=True).to_csv(
        OUT_DIR / "all_pairs.csv", index=False, float_format="%.6f"
    )
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False, float_format="%.6f")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
