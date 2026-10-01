"""
response_scorer_compute.py

FrugalGPT-style post-generation correctness scorer ablation.

For each cheap model in a calibration-selected pair, embed the prompt plus the
cheap model response with the same frozen sentence-transformer used by the
diagnostic router (all-MiniLM-L6-v2). On each calibration split, train a
logistic regression to predict cheap-model correctness from these embeddings.
Use the fitted probability as the cascade deferral score on the held-out split,
then compute the pairwise envelope over calibration-admissible pairs.

Outputs per dataset:
  results/response_scorer/{dataset}/response_scorer_curves.npy
  results/response_scorer/{dataset}/response_scorer_aurocs.npy
  results/response_scorer/{dataset}/diagnostics.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fig2_compute import (
    CACHE_VERSION,
    N_GRID,
    N_SPLITS,
    N_THRESH,
    OUT_DIR as FIG2_OUT_DIR,
    SEED,
    load_full,
    make_split,
    non_dominated_models,
    slice_data,
    valid_pairs_from_data,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_DIR = Path("data/response_embeddings")
OUT_DIR = Path("results/response_scorer")
CACHE_VERSION_RESPONSE = CACHE_VERSION + "-prompt-response-logreg-v2"

DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]


def response_text(prompt: object, response: object) -> str:
    prompt_s = "" if pd.isna(prompt) else str(prompt)
    response_s = "" if pd.isna(response) else str(response)
    return f"Prompt:\n{prompt_s}\n\nResponse:\n{response_s}"


def embedding_path(dataset: str, model: str) -> Path:
    return EMB_DIR / f"{dataset}-{model}.parquet"


def ensure_response_embeddings(dataset: str, raw: dict[str, pd.DataFrame]) -> dict[str, np.ndarray]:
    """Load or create prompt+response embeddings for each model."""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for m in raw:
        path = embedding_path(dataset, m)
        if not path.exists():
            missing.append(m)
            continue
        if len(pd.read_parquet(path)) != len(raw[m]):
            missing.append(m)

    if missing:
        from sentence_transformers import SentenceTransformer

        print(f"  Embedding {len(missing)} model response files with {MODEL_NAME}...")
        encoder = SentenceTransformer(MODEL_NAME)
        for model in missing:
            texts = [
                response_text(p, r)
                for p, r in zip(raw[model]["prompt"].values, raw[model]["response"].values)
            ]
            emb = encoder.encode(
                texts,
                batch_size=64,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).astype(np.float32)
            pd.DataFrame({"embedding": list(emb)}).to_parquet(embedding_path(dataset, model))
            print(f"    saved {embedding_path(dataset, model)}")

    embeddings = {}
    for model in raw:
        df = pd.read_parquet(embedding_path(dataset, model))
        embeddings[model] = np.vstack(df["embedding"].values).astype(np.float32)
    return embeddings


def compute_response_scorer_envelope(
    scores_by_model: dict[str, np.ndarray],
    test_data: dict,
    models: list[str],
    cost_grid: np.ndarray,
    pairs: list[tuple[str, str]],
) -> np.ndarray:
    interped = []
    mean_c = {m: test_data[m]["costs"].mean() for m in models}

    for cheap, exp in pairs:
        s = scores_by_model[cheap]
        u_l = test_data[cheap]["correct"].astype(float)
        u_h = test_data[exp]["correct"].astype(float)
        c_l = test_data[cheap]["costs"]
        c_h = test_data[exp]["costs"]
        c_h_mean = mean_c[exp]

        tau_grid = np.linspace(float(np.nanmin(s)), float(np.nanmax(s)), N_THRESH)
        cs = [mean_c[cheap]]
        qs = [float(u_l.mean())]
        for tau in tau_grid:
            esc = s < tau
            per_q_c = c_l + esc.astype(float) * c_h
            c = float(per_q_c.mean())
            q = float(np.where(esc, u_h, u_l).mean())
            if c <= c_h_mean + 1e-12:
                cs.append(c)
                qs.append(q)
        cs.append(c_h_mean)
        qs.append(float(u_h.mean()))

        cs_arr = np.asarray(cs)
        qs_arr = np.asarray(qs)
        order = np.argsort(cs_arr)
        f = interp1d(
            cs_arr[order],
            qs_arr[order],
            bounds_error=False,
            fill_value=(np.nan, np.nan),
        )
        interped.append(f(cost_grid))

    if not interped:
        return np.full(len(cost_grid), np.nan)

    stacked = np.vstack(interped)
    env = np.full(stacked.shape[1], np.nan)
    valid_cols = ~np.all(np.isnan(stacked), axis=0)
    if valid_cols.any():
        env[valid_cols] = np.nanmax(stacked[:, valid_cols], axis=0)
    env = np.maximum.accumulate(np.where(np.isnan(env), -np.inf, env))
    return np.where(env == -np.inf, np.nan, env)


def curve_area(cost_grid: np.ndarray, curve: np.ndarray) -> float:
    valid = ~np.isnan(curve)
    if valid.sum() < 2:
        return float("nan")
    return float(np.trapz(curve[valid], cost_grid[valid]))


def run_dataset(dataset: str) -> dict:
    print(f"\n{'=' * 60}\n  {dataset}\n{'=' * 60}")
    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "response_scorer_curves.npy"
    auroc_path = out_dir / "response_scorer_aurocs.npy"
    version_path = out_dir / "methodology_version.txt"
    diag_path = out_dir / "diagnostics.json"
    cache_current = (
        version_path.exists()
        and version_path.read_text().strip() == CACHE_VERSION_RESPONSE
        and result_path.exists()
        and auroc_path.exists()
        and diag_path.exists()
    )
    if cache_current:
        print("  Already computed, loading cache.")
        with open(diag_path) as f:
            return json.load(f)

    fig2_dir = FIG2_OUT_DIR / dataset
    if not (fig2_dir / "cost_grid.npy").exists():
        raise FileNotFoundError(f"Run fig2_compute.py {dataset} first.")
    cost_grid = np.load(fig2_dir / "cost_grid.npy")
    env_curves = np.load(fig2_dir / "envelope_curves.npy")

    raw, costs = load_full(dataset)
    valid_mask = np.ones(len(next(iter(raw.values()))), dtype=bool)
    for model in raw:
        valid_mask &= raw[model]["correct"].notna().values
    valid_idx = np.where(valid_mask)[0]
    for model in raw:
        raw[model] = raw[model].iloc[valid_idx].reset_index(drop=True)
        costs[model] = costs[model][valid_idx]
    n = len(valid_idx)
    print(f"  Valid rows after correctness filter: {n} / {valid_mask.size}")

    embeddings = ensure_response_embeddings(dataset, raw)
    ref_correct = next(iter(raw.values()))["correct"].values.astype(int)

    curves = np.full((N_SPLITS, N_GRID), np.nan)
    aurocs = np.full((N_SPLITS, len(raw)), np.nan)
    model_order = list(raw)

    t0 = time.time()
    for split_i in range(N_SPLITS):
        cal_idx, test_idx = make_split(n, ref_correct, seed=SEED + split_i)
        split_models = non_dominated_models(raw, costs, idx=cal_idx)
        if len(split_models) < 2:
            continue

        cal_data = slice_data(raw, costs, split_models, cal_idx)
        test_data = slice_data(raw, costs, split_models, test_idx)
        pairs = valid_pairs_from_data(cal_data, split_models)
        if not pairs:
            continue

        scores_by_model = {}
        for model in split_models:
            x_cal = embeddings[model][cal_idx]
            x_test = embeddings[model][test_idx]
            y_cal = cal_data[model]["correct"].astype(int)
            y_test = test_data[model]["correct"].astype(int)

            if len(np.unique(y_cal)) < 2:
                score = np.full(len(test_idx), float(y_cal.mean()))
            else:
                clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=SEED)
                clf.fit(x_cal, y_cal)
                score = clf.predict_proba(x_test)[:, 1]
            scores_by_model[model] = score

            if len(np.unique(y_test)) >= 2:
                try:
                    aurocs[split_i, model_order.index(model)] = roc_auc_score(y_test, score)
                except ValueError:
                    pass

        curves[split_i] = compute_response_scorer_envelope(
            scores_by_model=scores_by_model,
            test_data=test_data,
            models=split_models,
            cost_grid=cost_grid,
            pairs=pairs,
        )

        if (split_i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  split {split_i + 1:2d}/{N_SPLITS}  elapsed={elapsed:5.1f}s")

    np.save(result_path, curves)
    np.save(auroc_path, aurocs)

    med_resp = np.nanmedian(curves, axis=0)
    med_env = np.nanmedian(env_curves, axis=0)
    valid = ~np.isnan(med_resp) & ~np.isnan(med_env)
    area_delta = curve_area(cost_grid, med_resp) - curve_area(cost_grid, med_env)
    point_delta = med_resp[valid] - med_env[valid]
    diagnostics = {
        "dataset": dataset,
        "methodology_version": CACHE_VERSION_RESPONSE,
        "model_order": model_order,
        "mean_model_auroc": {
            model: float(np.nanmean(aurocs[:, j]))
            for j, model in enumerate(model_order)
            if not np.all(np.isnan(aurocs[:, j]))
        },
        "response_scorer_vs_mean_negentropy": {
            "area_delta": float(area_delta),
            "median_point_delta": float(np.nanmedian(point_delta)) if valid.any() else float("nan"),
            "p10_point_delta": float(np.nanpercentile(point_delta, 10)) if valid.any() else float("nan"),
            "p90_point_delta": float(np.nanpercentile(point_delta, 90)) if valid.any() else float("nan"),
            "share_cost_grid_above": float(np.nanmean(point_delta > 1e-12)) if valid.any() else float("nan"),
            "max_point_delta": float(np.nanmax(point_delta)) if valid.any() else float("nan"),
        },
    }

    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    version_path.write_text(CACHE_VERSION_RESPONSE)
    print(json.dumps(diagnostics["response_scorer_vs_mean_negentropy"], indent=2))
    return diagnostics


def main(args: list[str]) -> None:
    datasets = args or DATASETS
    all_diags = [run_dataset(ds) for ds in datasets]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(all_diags, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
