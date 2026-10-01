"""Simulate confidence scores with fixed cross-model correlation and AUROC.

For each dataset and target AUROC, write one Parquet file containing a row index
and eight synthetic confidence columns. The rows are aligned with the source
response files. Missing correctness labels are ignored for AUROC calculation.

The construction writes S = Y diag(a) + E L.T, where each column of Y is a
standardized correctness label, E has identity sample covariance and is
orthogonal to Y, and L L.T = C - diag(a) corr(Y) diag(a). Thus the sample
Pearson correlation of S is exactly the requested matrix C. The coefficients
in a are calibrated sequentially to the requested empirical AUROC.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.stats import norm, rankdata


DATA_DIR = Path("data/output_data")
CORR_DIR = Path("results/confidence_score_correlations")
OUT_DIR = Path("results/simulated_confidence")
DATASETS = ("mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench")
TARGETS = (0.5, 0.6, 0.7, 0.8, 0.9)
N_ROWS = {"mmlu": 2000, "triviaqa": 2000, "math_hard": 2000,
          "simpleqa": 2000, "livecodebench": 1055}
SEED = 20260917
AUROC_TOL = 0.001
CORR_TOL = 1e-10


def empirical_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    valid = np.isfinite(labels)
    scores = scores[valid]
    positive = labels[valid] == 1
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def load_dataset(dataset: str) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    target = pd.read_csv(CORR_DIR / f"{dataset}_pearson_matrix.csv", index_col=0)
    models = target.columns.tolist()
    if target.index.tolist() != models:
        raise ValueError(f"Mismatched correlation row and column order for {dataset}")
    corr = target.to_numpy(dtype=float)
    if not np.allclose(corr, corr.T, atol=1e-12) or np.linalg.eigvalsh(corr).min() <= 0:
        raise ValueError(f"Correlation matrix is not positive definite for {dataset}")

    labels = []
    reference_prompts = None
    for model in models:
        path = DATA_DIR / f"{dataset}-{model}.parquet"
        df = pd.read_parquet(path, columns=["prompt", "correct"]).iloc[:N_ROWS[dataset]]
        if len(df) != N_ROWS[dataset]:
            raise ValueError(f"Unexpected number of rows in {path}")
        prompts = df["prompt"].to_numpy()
        if reference_prompts is None:
            reference_prompts = prompts
        elif not np.array_equal(reference_prompts, prompts):
            raise ValueError(f"Prompts are not aligned in {path}")
        labels.append(pd.to_numeric(df["correct"], errors="coerce").to_numpy(float))

    all_labels = np.column_stack(labels)
    row_index = np.arange(N_ROWS[dataset])
    outcomes = all_labels
    if not np.isin(outcomes[np.isfinite(outcomes)], [0.0, 1.0]).all():
        raise ValueError(f"Nonbinary correctness labels in {dataset}")
    positives = np.nansum(outcomes, axis=0)
    counts = np.isfinite(outcomes).sum(axis=0)
    if ((positives == 0) | (positives == counts)).any():
        raise ValueError(f"AUROC undefined for at least one model in {dataset}")
    return row_index, models, corr, outcomes


def orthogonal_noise(outcomes: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k = outcomes.shape
    means = np.nanmean(outcomes, axis=0)
    filled = np.where(np.isfinite(outcomes), outcomes, means)
    centered = filled - filled.mean(axis=0)
    y = centered / centered.std(axis=0, ddof=1)
    q, _ = np.linalg.qr(np.column_stack([np.ones(n), y]), mode="reduced")
    raw = rng.standard_normal((n, k))
    residual = raw - q @ (q.T @ raw)
    cov = residual.T @ residual / (n - 1)
    chol = np.linalg.cholesky(cov)
    noise = residual @ solve_triangular(chol.T, np.eye(k), lower=False)
    label_corr = y.T @ y / (n - 1)
    return y, noise, label_corr


def make_scores(a: np.ndarray, y: np.ndarray, noise: np.ndarray,
                label_corr: np.ndarray, target_corr: np.ndarray) -> np.ndarray:
    residual_corr = target_corr - a[:, None] * label_corr * a[None, :]
    chol = np.linalg.cholesky(residual_corr)
    return y * a + noise @ chol.T


def calibrate(target_auc: float, y: np.ndarray, noise: np.ndarray,
              label_corr: np.ndarray, target_corr: np.ndarray,
              outcomes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = outcomes.shape[1]
    p = np.nanmean(outcomes, axis=0)
    z = norm.ppf(target_auc)
    a = z * np.sqrt(2 * p * (1 - p)) / np.sqrt(1 + 2 * z * z * p * (1 - p))
    if np.linalg.eigvalsh(target_corr - a[:, None] * label_corr * a[None, :]).min() <= 0:
        raise RuntimeError("Analytic AUROC initialization violates the correlation constraint")

    # Calibrate all coefficients jointly by repeated coordinate sweeps. Starting
    # all coefficients near their targets preserves the covariance budget that
    # a one-model-at-a-time construction can exhaust prematurely.
    for sweep in range(30):
        for j in range(k):
            labels = outcomes[:, j]

            def feasible(value: float) -> bool:
                trial = a.copy()
                trial[j] = value
                try:
                    np.linalg.cholesky(target_corr - trial[:, None] * label_corr * trial[None, :])
                    return True
                except np.linalg.LinAlgError:
                    return False

            def score_auc(value: float) -> float:
                trial = a.copy()
                trial[j] = value
                scores = make_scores(trial, y, noise, label_corr, target_corr)
                return empirical_auc(scores[:, j], labels)

            def feasible_limit(bound: float) -> float:
                current = a[j]
                if feasible(bound):
                    return bound
                inside, outside = current, bound
                for _ in range(40):
                    mid = (inside + outside) / 2
                    if feasible(mid):
                        inside = mid
                    else:
                        outside = mid
                return current + 0.99999999 * (inside - current)

            lower, upper = feasible_limit(-0.999), feasible_limit(0.999)
            auc_lower, auc_upper = score_auc(lower), score_auc(upper)
            if target_auc <= auc_lower:
                a[j] = lower
                continue
            if target_auc >= auc_upper:
                a[j] = upper
                continue
            best = (abs(score_auc(a[j]) - target_auc), a[j])
            for _ in range(45):
                mid = (lower + upper) / 2
                auc = score_auc(mid)
                if abs(auc - target_auc) < best[0]:
                    best = (abs(auc - target_auc), mid)
                if auc < target_auc:
                    lower = mid
                else:
                    upper = mid
            a[j] = best[1]

        current = make_scores(a, y, noise, label_corr, target_corr)
        aucs = np.array([empirical_auc(current[:, j], outcomes[:, j])
                         for j in range(k)])
        if np.max(np.abs(aucs - target_auc)) <= AUROC_TOL:
            break
    else:
        raise RuntimeError(
            f"AUROC calibration did not converge at {target_auc:.1f}: "
            f"achieved {np.round(aucs, 4).tolist()}"
        )

    scores = make_scores(a, y, noise, label_corr, target_corr)
    # Positive affine transformations preserve both Pearson correlation and AUROC.
    scores = (scores - scores.min(axis=0)) / (scores.max(axis=0) - scores.min(axis=0))
    achieved_corr = np.corrcoef(scores, rowvar=False)
    achieved_auc = np.array([
        empirical_auc(scores[:, j], outcomes[:, j]) for j in range(k)
    ])
    if np.max(np.abs(achieved_corr - target_corr)) > CORR_TOL:
        raise RuntimeError("Simulated correlations missed the target")
    if np.max(np.abs(achieved_auc - target_auc)) > AUROC_TOL:
        raise RuntimeError("Simulated AUROCs missed the target")
    return scores, achieved_auc, achieved_corr


def run_dataset(dataset: str) -> list[dict]:
    row_index, models, corr, outcomes = load_dataset(dataset)
    rng = np.random.default_rng(SEED + DATASETS.index(dataset))
    y, noise, label_corr = orthogonal_noise(outcomes, rng)
    dataset_dir = OUT_DIR / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in TARGETS:
        scores, aucs, achieved_corr = calibrate(
            target, y, noise, label_corr, corr, outcomes
        )
        frame = pd.DataFrame(scores, columns=models)
        frame.insert(0, "row_index", row_index)
        filename = f"auroc_{target:.1f}.parquet"
        frame.to_parquet(dataset_dir / filename, index=False)
        rows.extend({"dataset": dataset, "target_auroc": target, "model": model,
                     "achieved_auroc": float(aucs[j]), "n_rows": len(row_index),
                     "max_abs_correlation_error": float(np.max(np.abs(achieved_corr - corr)))}
                    for j, model in enumerate(models))
        print(f"{dataset} X={target:.1f}: n={len(row_index)}, "
              f"max AUROC error={np.max(np.abs(aucs-target)):.6f}, "
              f"max correlation error={np.max(np.abs(achieved_corr-corr)):.2e}", flush=True)

    metadata = {"dataset": dataset, "seed": SEED + DATASETS.index(dataset),
                "models": models, "target_aurocs": TARGETS,
                "input_rows": N_ROWS[dataset], "retained_rows": len(row_index),
                "labeled_rows_by_model": {model: int(np.isfinite(outcomes[:, j]).sum())
                                          for j, model in enumerate(models)},
                "correlation_target_file": str(CORR_DIR / f"{dataset}_pearson_matrix.csv"),
                "score_range": [0, 1], "auroc_tolerance": AUROC_TOL,
                "correlation_tolerance": CORR_TOL}
    (dataset_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(
        dataset_dir / "achieved_aurocs.csv", index=False, float_format="%.9f"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", choices=DATASETS)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets or DATASETS:
        run_dataset(dataset)
    summaries = [OUT_DIR / d / "achieved_aurocs.csv" for d in DATASETS]
    pd.concat((pd.read_csv(path) for path in summaries if path.exists()),
              ignore_index=True).to_csv(
                  OUT_DIR / "achieved_aurocs.csv", index=False, float_format="%.9f"
              )


if __name__ == "__main__":
    main()
