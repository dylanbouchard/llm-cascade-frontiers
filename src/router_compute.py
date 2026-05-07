"""
router_compute.py - Embedding-based pre-generation k-model router
                    AND embedding-based cascade envelope.

For each dataset, trains one logistic regression per non-dominated model
(predicting correctness from query embeddings) and produces two curves:

  1. k-model router: sweep tradeoff weight w, each query routed to
     argmax_j [P_j(x) - w * cbar_j^cal]. One model called per query;
     realized held-out token cost is used only after dispatch for evaluation.

  2. Embedding cascade: same P(L correct | embedding) as the deferral
     signal instead of mean_token_negentropy. Cheap model always called;
     escalation decision uses the pre-generation embedding score.
     Pairwise envelope over all valid pairs.

Both use the same 50 splits as fig2_compute.py for a fair comparison.

Outputs (per dataset, written to results/fig2/{dataset}/):
  router_curves.npy          (50, N_GRID)
  emb_cascade_curves.npy     (50, N_GRID)
  router_aurocs.npy          (50, k)
  router_diagnostics.json    comparisons vs. standard cascade envelope

Usage:
    python3 router_compute.py [dataset ...]
"""

import sys, json, time
import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.interpolate import interp1d

from fig2_compute import (
    load_full, non_dominated_models, make_split,
    valid_pairs_from_data, slice_data,
    N_SPLITS, N_GRID, SEED, OUT_DIR, CACHE_VERSION,
)
from optuna_frontier import SCORER

EMB_DIR  = Path("data")
N_WEIGHTS = 200
N_THRESH  = 200   # threshold sweep resolution for emb cascade
ROUTER_CACHE_VERSION = CACHE_VERSION + "-router-calmean-cost-v1"


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embeddings(dataset: str, valid_idx: np.ndarray) -> np.ndarray:
    """Load frozen sentence-transformer embeddings, aligned to valid_idx."""
    path = EMB_DIR / f"{dataset}_embeddings.parquet"
    df   = pd.read_parquet(path)
    emb  = np.vstack(df["embedding"].values).astype(np.float32)
    return emb[valid_idx]


# ---------------------------------------------------------------------------
# Per-model logistic regression routers
# ---------------------------------------------------------------------------

def train_routers(X_cal: np.ndarray, correct_cal: dict, models: list) -> dict:
    routers = {}
    for m in models:
        y = correct_cal[m]
        if y.sum() == 0 or y.sum() == len(y):
            routers[m] = None
            continue
        lr = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs", random_state=42,
        )
        lr.fit(X_cal, y)
        routers[m] = lr
    return routers


def predict_proba(routers: dict, X_test: np.ndarray, models: list) -> np.ndarray:
    """Return (n_test, k) matrix of predicted P(correct)."""
    scores = np.zeros((X_test.shape[0], len(models)), dtype=np.float64)
    for j, m in enumerate(models):
        if routers[m] is None:
            scores[:, j] = 0.5
        else:
            scores[:, j] = routers[m].predict_proba(X_test)[:, 1]
    return scores


# ---------------------------------------------------------------------------
# k-model router frontier
# ---------------------------------------------------------------------------

def compute_router_frontier(
    scores: np.ndarray,
    correct_test: dict,
    cost_test: dict,
    dispatch_costs: dict,
    models: list,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep w: route to argmax_j P_j(x) - w * cbar_j^cal.

    Dispatch costs are calibration mean costs known before generation.
    Realized test costs are used only to evaluate the cost of the selected
    model after dispatch.
    """
    n_test         = scores.shape[0]
    eval_cost_matrix = np.column_stack([cost_test[m] for m in models])
    dispatch_cost_vec = np.array([dispatch_costs[m] for m in models], dtype=np.float64)
    correct_matrix = np.column_stack([correct_test[m].astype(float) for m in models])

    w_grid    = np.concatenate([[0.0], np.logspace(-4, 4, N_WEIGHTS - 2), [1e6]])
    f_costs   = np.zeros(len(w_grid))
    f_quality = np.zeros(len(w_grid))
    for idx, w in enumerate(w_grid):
        chosen         = np.argmax(scores - w * dispatch_cost_vec[None, :], axis=1)
        f_costs[idx]   = eval_cost_matrix[np.arange(n_test), chosen].mean()
        f_quality[idx] = correct_matrix[np.arange(n_test), chosen].mean()

    return f_costs, f_quality, w_grid


def extract_pareto(costs: np.ndarray, quality: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    order    = np.argsort(costs)
    c_sorted = costs[order]
    q_sorted = quality[order]
    pareto_c = [c_sorted[0]]
    pareto_q = [q_sorted[0]]
    max_q    = q_sorted[0]
    for i in range(1, len(c_sorted)):
        if q_sorted[i] > max_q:
            pareto_c.append(c_sorted[i])
            pareto_q.append(q_sorted[i])
            max_q = q_sorted[i]
    return np.array(pareto_c), np.array(pareto_q)


# ---------------------------------------------------------------------------
# Embedding-based cascade envelope
# ---------------------------------------------------------------------------

def emb_cascade_envelope_curve(
    emb_scores: np.ndarray,    # (n_test, k)  P(model correct | embedding)
    correct_test: dict,
    cost_test: dict,
    models: list,
    cost_grid: np.ndarray,
    pairs: list[tuple[str, str]] | None = None,
) -> np.ndarray:
    """
    Pairwise cascade envelope using P(L correct | embedding) as the deferral
    signal.  Mirrors pairwise_envelope_curve from fig2_compute but substitutes
    the embedding-predicted probability for mean_token_negentropy.
    Escalation rule: escalate when P(L correct) < tau.
    """
    mean_c = {m: cost_test[m].mean() for m in models}
    if pairs is None:
        mean_a = {m: correct_test[m].mean() for m in models}
        models_by_cost = sorted(models, key=lambda m: mean_c[m])
        pairs = [
            (L, H) for L, H in combinations(models_by_cost, 2)
            if mean_a[H] > mean_a[L]
        ]

    interped = []
    for L, H in pairs:
        j_L  = models.index(L)
        s    = emb_scores[:, j_L]          # P(L correct | embedding)
        u_L  = correct_test[L].astype(float)
        u_H  = correct_test[H].astype(float)
        c_H_mean = cost_test[H].mean()

        tau_grid = np.linspace(float(s.min()), float(s.max()), N_THRESH)
        cs = [mean_c[L]]
        qs = [u_L.mean()]
        for tau in tau_grid:
            esc     = s < tau
            per_q_c = cost_test[L] + esc.astype(float) * cost_test[H]
            c       = per_q_c.mean()
            q       = np.where(esc, u_H, u_L).mean()
            if c <= c_H_mean + 1e-12:
                cs.append(c); qs.append(q)
        cs.append(c_H_mean); qs.append(u_H.mean())

        cs_arr = np.array(cs)
        qs_arr = np.array(qs)
        order  = np.argsort(cs_arr)
        f = interp1d(cs_arr[order], qs_arr[order],
                     bounds_error=False, fill_value=(np.nan, np.nan))
        interped.append(f(cost_grid))

    if not interped:
        return np.full(len(cost_grid), np.nan)

    stacked = np.vstack(interped)
    env = np.nanmax(stacked, axis=0)
    env = np.maximum.accumulate(np.where(np.isnan(env), -np.inf, env))
    return np.where(env == -np.inf, np.nan, env)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _gap_stats(test_curves: np.ndarray, ref_curves: np.ndarray) -> dict:
    """
    Compute median_gap, band_width_ratio, dominance_fraction for
    test_curves vs ref_curves, both (N_SPLITS, N_GRID).
    """
    med_test = np.nanmedian(test_curves, axis=0)
    med_ref  = np.nanmedian(ref_curves,  axis=0)
    valid    = ~np.isnan(med_test) & ~np.isnan(med_ref)
    vi       = np.where(valid)[0]
    mid      = int(vi[len(vi) // 2]) if vi.size > 0 else 0

    delta = float(np.nanmedian(test_curves[:, mid] - ref_curves[:, mid]))

    ref_bw  = float(np.nanpercentile(ref_curves[:,  mid], 90)
                  - np.nanpercentile(ref_curves[:,  mid], 10))
    test_bw = float(np.nanpercentile(test_curves[:, mid], 90)
                  - np.nanpercentile(test_curves[:, mid], 10))
    bwr = test_bw / ref_bw if ref_bw > 0 else float("nan")

    dom = (float(np.mean(med_test[valid] > med_ref[valid]))
           if valid.any() else float("nan"))

    return {"median_gap": delta, "band_width_ratio": bwr, "dominance_fraction": dom}


def compute_diagnostics(
    router_curves: np.ndarray,
    emb_cas_curves: np.ndarray,
    env_curves: np.ndarray,
    models: list,
    aurocs: np.ndarray,
) -> dict:
    return {
        "router_vs_cascade":      _gap_stats(router_curves,  env_curves),
        "emb_cascade_vs_cascade": _gap_stats(emb_cas_curves, env_curves),
        "router_vs_emb_cascade":  _gap_stats(router_curves,  emb_cas_curves),
        "model_aurocs": {
            m: float(np.nanmean(aurocs[:, j]))
            for j, m in enumerate(models)
        },
    }


# ---------------------------------------------------------------------------
# Dataset runner
# ---------------------------------------------------------------------------

def run_dataset(dataset: str) -> None:
    print(f"\n{'='*60}\n  {dataset}\n{'='*60}")
    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    fig2_version_path = out_dir / "methodology_version.txt"
    if not fig2_version_path.exists() or fig2_version_path.read_text().strip() != CACHE_VERSION:
        raise RuntimeError(
            f"Run fig2_compute.py {dataset} first; router comparison needs "
            "calibration-selected envelope curves."
        )

    version_path = out_dir / "router_methodology_version.txt"
    cache_current = (
        version_path.exists()
        and version_path.read_text().strip() == ROUTER_CACHE_VERSION
    )
    if cache_current and (out_dir / "emb_cascade_curves.npy").exists():
        print("  Already computed - skipping.")
        return

    # ---- Load cascade data (mirrors fig2_compute.run_dataset exactly) -------
    raw, costs = load_full(dataset)

    valid_mask = np.ones(len(next(iter(raw.values()))), dtype=bool)
    for m in raw:
        valid_mask &= raw[m][SCORER].notna().values
        valid_mask &= raw[m]["correct"].notna().values
    valid_idx = np.where(valid_mask)[0]
    for m in raw:
        raw[m]   = raw[m].iloc[valid_idx].reset_index(drop=True)
        costs[m] = costs[m][valid_idx]

    n           = len(valid_idx)
    ref_model   = next(iter(raw))
    ref_correct = raw[ref_model]["correct"].values.astype(int)
    print(f"  Valid rows: {n}; pool selected from calibration split only.")

    embeddings = load_embeddings(dataset, valid_idx)
    cost_grid  = np.load(out_dir / "cost_grid.npy")

    # ---- Per-split loop ------------------------------------------------------
    k               = len(raw)
    router_curves   = np.full((N_SPLITS, N_GRID), np.nan)
    emb_cas_curves  = np.full((N_SPLITS, N_GRID), np.nan)
    aurocs          = np.full((N_SPLITS, k),      np.nan)
    auroc_models    = list(raw.keys())

    t0 = time.time()
    for i in range(N_SPLITS):
        cal_idx, test_idx = make_split(n, ref_correct, seed=SEED + i)

        X_cal  = embeddings[cal_idx]
        X_test = embeddings[test_idx]

        split_models = non_dominated_models(raw, costs, idx=cal_idx)
        if len(split_models) < 2:
            continue

        correct_cal  = {m: raw[m]["correct"].values[cal_idx].astype(float)  for m in split_models}
        correct_test = {m: raw[m]["correct"].values[test_idx].astype(float) for m in split_models}
        dispatch_costs = {m: float(costs[m][cal_idx].mean())                 for m in split_models}
        cost_test    = {m: costs[m][test_idx]                                for m in split_models}

        routers = train_routers(X_cal, correct_cal, split_models)
        scores  = predict_proba(routers, X_test, split_models)   # (n_test, k)

        # AUROC
        for j, m in enumerate(split_models):
            y = correct_test[m]
            if 0 < y.sum() < len(y):
                aurocs[i, auroc_models.index(m)] = roc_auc_score(y, scores[:, j])

        # k-model router
        f_costs, f_quality, _ = compute_router_frontier(
            scores, correct_test, cost_test, dispatch_costs, split_models)
        p_costs, p_quality = extract_pareto(f_costs, f_quality)
        if len(p_costs) >= 2:
            f_interp = interp1d(p_costs, p_quality,
                                bounds_error=False, fill_value=np.nan)
            router_curves[i] = f_interp(cost_grid)

        # Embedding-based cascade envelope
        cal_data = slice_data(raw, costs, split_models, cal_idx)
        cal_pairs = valid_pairs_from_data(cal_data, split_models)
        emb_cas_curves[i] = emb_cascade_envelope_curve(
            scores, correct_test, cost_test, split_models, cost_grid,
            pairs=cal_pairs)

        elapsed = time.time() - t0
        eta     = elapsed / (i + 1) * (N_SPLITS - i - 1)
        print(
            f"  split {i+1:2d}/{N_SPLITS}  "
            f"router_q={np.nanmedian(router_curves[i]):.4f}  "
            f"emb_cas_q={np.nanmedian(emb_cas_curves[i]):.4f}  "
            f"auroc={np.nanmean(aurocs[i]):.4f}  "
            f"ETA {eta:.0f}s",
            flush=True,
        )

    # ---- Save ----------------------------------------------------------------
    np.save(out_dir / "router_curves.npy",      router_curves)
    np.save(out_dir / "emb_cascade_curves.npy", emb_cas_curves)
    np.save(out_dir / "router_aurocs.npy",      aurocs)

    env_curves = np.load(out_dir / "envelope_curves.npy")
    diag       = compute_diagnostics(
        router_curves, emb_cas_curves, env_curves, auroc_models, aurocs)

    with open(out_dir / "router_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)
    version_path.write_text(ROUTER_CACHE_VERSION + "\n")

    print(f"\n  Diagnostics:")
    for k_name, v in diag.items():
        print(f"    {k_name}: {v}")
    print(f"  Total time: {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    datasets = sys.argv[1:] or [
        "mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"
    ]
    for ds in datasets:
        run_dataset(ds)
