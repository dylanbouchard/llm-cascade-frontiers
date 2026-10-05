"""Descriptive full-sample oracle comparison of pairwise and depth-3 cascades.

This analysis intentionally uses every available example for both policy search
and evaluation.  It is therefore not a deployment estimate.  Its purpose is to
ask whether the observed model pool contains any empirical value of depth once
calibration/test selection variance is removed.

For a fixed cost-ordered triple A -> B -> C, all empirically distinct threshold
pairs can be evaluated exactly.  If r_A and r_B index the score ranks below the
two thresholds, then

  Q = sum(U_A) + sum_{s_A < tau_A}(U_B-U_A)
                 + sum_{s_A < tau_A, s_B < tau_B}(U_C-U_B),

and cost has the analogous prefix-sum form.  Two-dimensional cumulative sums
therefore recover the exact empirical three-stage frontier without stochastic
optimization.  The reported up-to-three class is the union of all single-model,
two-model, and three-model policies, so it nests the pairwise class by
construction.

Outputs:
  results/oracle_depth/{dataset}/frontiers.npz
  results/oracle_depth/{dataset}/summary.json
  results/oracle_depth/summary.csv
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fig2_compute import MODEL_COST_ORDER, SCORER, load_full


DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
DISPLAY_NAMES = {
    "mmlu": "MMLU",
    "triviaqa": "TriviaQA",
    "math_hard": "MATH",
    "simpleqa": "SimpleQA",
    "livecodebench": "LiveCodeBench",
}
OUT_DIR = Path("results/oracle_depth")
N_BUDGETS = 10001
ROW_CHUNK = 128
CACHE_VERSION = "full-sample-exact-up-to-three-v2-10001-budgets"


def clean_data(dataset: str):
    raw, costs = load_full(dataset)
    valid = np.ones(len(next(iter(raw.values()))), dtype=bool)
    for model in raw:
        valid &= raw[model][SCORER].notna().to_numpy()
        valid &= raw[model]["correct"].notna().to_numpy()
    idx = np.flatnonzero(valid)
    raw = {m: raw[m].iloc[idx].reset_index(drop=True) for m in raw}
    costs = {m: np.asarray(costs[m][idx], dtype=float) for m in raw}
    return raw, costs, int(valid.sum()), int(len(valid))


def add_candidates(
    frontier: np.ndarray,
    budgets: np.ndarray,
    costs: np.ndarray,
    qualities: np.ndarray,
) -> None:
    """Add candidate policies to an exact budget-grid value function."""
    c = np.asarray(costs, dtype=float).ravel()
    q = np.asarray(qualities, dtype=float).ravel()
    valid = (
        np.isfinite(c)
        & np.isfinite(q)
        & (c <= budgets[-1] + 1e-12)
    )
    if not valid.any():
        return
    # A candidate first becomes affordable at this grid location.  A final
    # cumulative maximum below propagates it to all larger budgets.
    loc = np.searchsorted(budgets, c[valid], side="left")
    in_grid = loc < len(budgets)
    np.maximum.at(frontier, loc[in_grid], q[valid][in_grid])


def score_ranks(scores: np.ndarray) -> tuple[np.ndarray, int]:
    """Return ascending tie-preserving ranks and the number of score groups."""
    _, inverse = np.unique(np.asarray(scores, dtype=float), return_inverse=True)
    return inverse.astype(np.int32, copy=False), int(inverse.max()) + 1


def prefix_1d(rank: np.ndarray, n_rank: int, weights: np.ndarray) -> np.ndarray:
    out = np.empty(n_rank + 1, dtype=float)
    out[0] = 0.0
    out[1:] = np.cumsum(np.bincount(rank, weights=weights, minlength=n_rank))
    return out


def prefix_2d(
    rank_a: np.ndarray,
    n_a: int,
    rank_b: np.ndarray,
    n_b: int,
    weights: np.ndarray,
) -> np.ndarray:
    # Offset both ranks by one so entry [r_a, r_b] sums observations satisfying
    # rank_a < r_a and rank_b < r_b, including r=0 (never escalate).
    out = np.zeros((n_a + 1, n_b + 1), dtype=float)
    np.add.at(out, (rank_a + 1, rank_b + 1), weights)
    np.cumsum(out, axis=0, out=out)
    np.cumsum(out, axis=1, out=out)
    return out


def run_dataset(dataset: str) -> dict:
    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    version_path = out_dir / "methodology_version.txt"
    summary_path = out_dir / "summary.json"
    frontier_path = out_dir / "frontiers.npz"
    if (
        version_path.exists()
        and version_path.read_text().strip() == CACHE_VERSION
        and summary_path.exists()
        and frontier_path.exists()
    ):
        print(f"{dataset}: loading cached result", flush=True)
        return json.loads(summary_path.read_text())

    raw, per_query_cost, n_valid, n_original = clean_data(dataset)
    mean_cost = {m: float(per_query_cost[m].mean()) for m in raw}
    mean_quality = {m: float(raw[m]["correct"].mean()) for m in raw}
    models = sorted(raw, key=mean_cost.get)
    lowest = min(models, key=mean_cost.get)
    highest_quality = max(models, key=mean_quality.get)
    budgets = np.linspace(mean_cost[lowest], mean_cost[highest_quality], N_BUDGETS)

    single_frontier = np.full(N_BUDGETS, -np.inf)
    pair_frontier = np.full(N_BUDGETS, -np.inf)
    triple_frontier = np.full(N_BUDGETS, -np.inf)
    for model in models:
        add_candidates(
            single_frontier,
            budgets,
            np.asarray([mean_cost[model]]),
            np.asarray([mean_quality[model]]),
        )

    # Exact empirical two-stage policies over every cost-ordered pair.
    for cheap, expensive in combinations(models, 2):
        rank_a, n_a = score_ranks(raw[cheap][SCORER].to_numpy())
        base_cost = float(per_query_cost[cheap].sum())
        base_quality = float(raw[cheap]["correct"].sum())
        c = base_cost + prefix_1d(rank_a, n_a, per_query_cost[expensive])
        q = base_quality + prefix_1d(
            rank_a,
            n_a,
            raw[expensive]["correct"].to_numpy(dtype=float)
            - raw[cheap]["correct"].to_numpy(dtype=float),
        )
        add_candidates(pair_frontier, budgets, c / n_valid, q / n_valid)

    # Exact empirical three-stage policies over every cost-ordered triple.
    t0 = time.time()
    triples = list(combinations(models, 3))
    for triple_idx, (first, second, third) in enumerate(triples):
        rank_a, n_a = score_ranks(raw[first][SCORER].to_numpy())
        rank_b, n_b = score_ranks(raw[second][SCORER].to_numpy())
        prefix_ab_quality = prefix_1d(
            rank_a,
            n_a,
            raw[second]["correct"].to_numpy(dtype=float)
            - raw[first]["correct"].to_numpy(dtype=float),
        )
        prefix_b_cost = prefix_1d(rank_a, n_a, per_query_cost[second])
        prefix_bc_quality = prefix_2d(
            rank_a,
            n_a,
            rank_b,
            n_b,
            raw[third]["correct"].to_numpy(dtype=float)
            - raw[second]["correct"].to_numpy(dtype=float),
        )
        prefix_c_cost = prefix_2d(
            rank_a,
            n_a,
            rank_b,
            n_b,
            per_query_cost[third],
        )
        base_cost = float(per_query_cost[first].sum())
        base_quality = float(raw[first]["correct"].sum())

        for start in range(0, n_a + 1, ROW_CHUNK):
            stop = min(start + ROW_CHUNK, n_a + 1)
            c = (
                base_cost
                + prefix_b_cost[start:stop, None]
                + prefix_c_cost[start:stop]
            ) / n_valid
            q = (
                base_quality
                + prefix_ab_quality[start:stop, None]
                + prefix_bc_quality[start:stop]
            ) / n_valid
            add_candidates(triple_frontier, budgets, c, q)

        elapsed = time.time() - t0
        eta = elapsed / (triple_idx + 1) * (len(triples) - triple_idx - 1)
        print(
            f"{dataset}: triple {triple_idx + 1:02d}/{len(triples)} "
            f"({first}->{second}->{third}), ETA {eta:.0f}s",
            flush=True,
        )

    # Each class includes standalone endpoints.  The up-to-three class also
    # includes the pair class explicitly, preserving the required nesting.
    pair_frontier = np.maximum(single_frontier, pair_frontier)
    up_to_three_frontier = np.maximum(pair_frontier, triple_frontier)
    pair_frontier = np.maximum.accumulate(pair_frontier)
    up_to_three_frontier = np.maximum.accumulate(up_to_three_frontier)
    pair_frontier[pair_frontier == -np.inf] = np.nan
    up_to_three_frontier[up_to_three_frontier == -np.inf] = np.nan

    valid = np.isfinite(pair_frontier) & np.isfinite(up_to_three_frontier)
    gap = up_to_three_frontier[valid] - pair_frontier[valid]
    # Numerical subtraction can produce signed zeros around exact ties.
    gap = np.maximum(gap, 0.0)
    valid_budgets = budgets[valid]
    max_idx = int(np.argmax(gap))
    summary = {
        "dataset": dataset,
        "display_name": DISPLAY_NAMES[dataset],
        "n_examples": n_valid,
        "n_original_examples": n_original,
        "n_models": len(models),
        "n_pairs": len(list(combinations(models, 2))),
        "n_triples": len(triples),
        "models": models,
        "budget_min": float(valid_budgets[0]),
        "budget_max": float(valid_budgets[-1]),
        "max_gap": float(gap[max_idx]),
        "budget_at_max_gap": float(valid_budgets[max_idx]),
        "mean_gap": float(gap.mean()),
        "fraction_gap_gt_0_005": float(np.mean(gap > 0.005)),
        "fraction_gap_gt_0_010": float(np.mean(gap > 0.010)),
    }
    np.savez_compressed(
        frontier_path,
        budgets=budgets,
        pair_frontier=pair_frontier,
        up_to_three_frontier=up_to_three_frontier,
        gap=up_to_three_frontier - pair_frontier,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    version_path.write_text(CACHE_VERSION + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(datasets: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = [run_dataset(dataset) for dataset in datasets]
    pd.DataFrame(summaries).to_csv(OUT_DIR / "summary.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or DATASETS)
