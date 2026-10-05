"""Exact held-out S2-S4 with next-model Diff-01 response-embedding regressions."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from continuation_benefit_compute import evaluate, fit_linear, predict
from exact_heldout_depth import FRACTIONS, encode
from next_model_deep_search import merge, search
from oracle_depth_analysis import prefix_1d, prefix_2d, score_ranks
from scorer_depth_compute import prepare
from voi_compute import DATASETS

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results/next_model_diff01_full"
ARMS = ("embedding_diff01_ridge",)
SOURCES = (
    "next_model_embedding_diff01.py",
    "test_next_model_embedding_diff01.py",
    "next_model_deep_search.py",
    "next_model_depth_kernel.cpp",
    "continuation_benefit_compute.py",
    "scorer_depth_compute.py",
    "response_scorer_compute.py",
    "exact_heldout_depth.py",
    "oracle_depth_analysis.py",
    "fig2_compute.py",
    "optuna_frontier.py",
    "voi_compute.py",
)
INPUT_CACHE = {}


def fit_embedding_edges(cal, arm):
    """Fit each current model once, jointly across all downstream targets."""
    models = sorted(cal, key=lambda model: (cal[model]["c"].mean(), model))
    edges = {}
    if arm != "embedding_diff01_ridge":
        raise ValueError(arm)
    for index, current in enumerate(models[:-1]):
        downstream = models[index + 1 :]
        targets = np.column_stack(
            [cal[next_model]["y"] - cal[current]["y"] for next_model in downstream]
        )
        coefficients, intercepts = fit_linear(cal[current]["x"], targets)
        for column, next_model in enumerate(downstream):
            edges[current, next_model] = {
                "coef": (-coefficients[:, column]).tolist(),
                "intercept": float(-intercepts[column]),
            }
    return models, edges


def cutoffs(score):
    unique = np.unique(score)
    return np.r_[-np.inf, unique[1:], np.inf]


def select_three(cal, budgets, models, edges):
    """Exact nested S1/S2/S3 selection for prefit transition scores."""
    n = len(cal[models[0]]["y"])
    scores = {
        edge: predict(cal[edge[0]]["x"], fitted)
        for edge, fitted in edges.items()
    }
    ranks = {edge: score_ranks(score) for edge, score in scores.items()}
    cuts = {edge: cutoffs(score) for edge, score in scores.items()}
    best = np.full(n + 1, np.inf)
    policies = [None] * (n + 1)
    frozen = {}

    def add(sequence, costs, quality, start=0):
        flat_cost = np.asarray(costs).ravel() / n
        flat_quality = np.rint(np.asarray(quality).ravel()).astype(int)
        eligible = np.flatnonzero(flat_cost <= budgets[-1] + 1e-12)
        local = np.full(n + 1, np.inf)
        np.minimum.at(local, flat_quality[eligible], flat_cost[eligible])
        changed = np.flatnonzero(local < best)
        if not len(changed):
            return
        tied = eligible[flat_cost[eligible] == local[flat_quality[eligible]]]
        witness = np.full(n + 1, len(flat_cost), dtype=int)
        np.minimum.at(witness, flat_quality[tied], tied)
        for value in changed:
            flat_index = int(witness[value])
            stages = []
            if len(sequence) == 2:
                edge = tuple(sequence)
                stages = [{
                    "predictor": edges[edge],
                    "threshold": float(cuts[edge][flat_index]),
                }]
            elif len(sequence) == 3:
                first_index, second_index = np.unravel_index(flat_index, np.shape(costs))
                for edge, threshold_index in (
                    (tuple(sequence[:2]), start + first_index),
                    (tuple(sequence[1:]), second_index),
                ):
                    stages.append({
                        "predictor": edges[edge],
                        "threshold": float(cuts[edge][threshold_index]),
                    })
            policies[value] = {
                "models": list(sequence),
                "stages": stages,
                "cal_cost": float(local[value]),
                "cal_quality": float(value / n),
            }
        best[changed] = local[changed]

    for depth in (1, 2, 3):
        for sequence in combinations(models, depth):
            first = sequence[0]
            base_cost = cal[first]["c"].sum()
            base_quality = cal[first]["y"].sum()
            if depth == 1:
                add(sequence, [base_cost], [base_quality])
                continue
            second = sequence[1]
            rank_first, count_first = ranks[first, second]
            pair_cost = base_cost + prefix_1d(
                rank_first, count_first, cal[second]["c"]
            )
            pair_quality = base_quality + prefix_1d(
                rank_first,
                count_first,
                cal[second]["y"] - cal[first]["y"],
            )
            if depth == 2:
                add(sequence, pair_cost, pair_quality)
                continue
            third = sequence[2]
            rank_second, count_second = ranks[second, third]
            continuation_cost = prefix_2d(
                rank_first,
                count_first,
                rank_second,
                count_second,
                cal[third]["c"],
            )
            continuation_quality = prefix_2d(
                rank_first,
                count_first,
                rank_second,
                count_second,
                cal[third]["y"] - cal[second]["y"],
            )
            for start in range(0, count_first + 1, 128):
                stop = min(start + 128, count_first + 1)
                add(
                    sequence,
                    pair_cost[start:stop, None] + continuation_cost[start:stop],
                    pair_quality[start:stop, None] + continuation_quality[start:stop],
                    start,
                )
        selected = []
        for budget in budgets:
            feasible = np.flatnonzero(best <= budget + 1e-12)
            if not len(feasible):
                raise ValueError("No feasible calibration policy")
            selected.append(policies[int(feasible[-1])])
        frozen[str(depth)] = selected
    return frozen


def prepare_input(dataset, output):
    raw, costs, rows, embeddings, fingerprint = prepare(dataset, response=True)
    if len(raw) != 8 or set(raw) != set(embeddings):
        raise ValueError(f"{dataset}: expected eight embedding-backed models")
    path = output / "inputs" / f"{dataset}.npz"
    with path.open("xb") as handle:
        np.savez(
            handle,
            models=list(raw),
            rows=rows,
            fingerprint=fingerprint,
            x=np.asarray([embeddings[model] for model in raw], dtype=np.float32),
            y=np.asarray([raw[model].correct.to_numpy(float) for model in raw]),
            c=np.asarray([costs[model] for model in raw]),
        )
    print(f"{dataset}: prepared {len(rows)} rows and eight embedding caches", flush=True)


def run_split(dataset, arm, split, output):
    key = str(output / "inputs" / f"{dataset}.npz")
    if key not in INPUT_CACHE:
        with np.load(key) as arrays:
            INPUT_CACHE[key] = {name: arrays[name] for name in arrays.files}
    arrays = INPUT_CACHE[key]
    rows = arrays["rows"]
    destination = output / dataset / arm
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / f"split_{split:02d}.csv").exists():
        return
    baseline_path = BASE / dataset / "diff01_ridge" / f"split_{split:02d}.json"
    baseline_bytes = baseline_path.read_bytes()
    baseline = json.loads(baseline_bytes)

    def sliced(original_indices):
        index = np.searchsorted(rows, np.asarray(original_indices))
        np.testing.assert_array_equal(rows[index], original_indices)
        return {
            model: {
                name: arrays[name][model_index, index]
                for name in ("x", "y", "c")
            }
            for model_index, model in enumerate(arrays["models"])
        }

    calibration = sliced(baseline["cal_idx"])
    models, edges = fit_embedding_edges(calibration, arm)
    budgets = np.asarray(baseline["budgets"])
    frozen = select_three(calibration, budgets, models, edges)
    selected = list(frozen["3"])
    stats = {}
    checkpoint = destination / f"split_{split:02d}_s4_chains"
    checkpoint.mkdir(exist_ok=False)
    started = time.perf_counter()
    for chain_index, sequence in enumerate(combinations(models, 4)):
        best, witness, cuts, chain_stats = search(
            calibration, sequence, edges, budgets, selected
        )
        if not chain_stats["complete"]:
            raise RuntimeError(f"Incomplete chain {dataset}/{arm}/{split}/{chain_index}")
        selected = merge(selected, budgets, sequence, edges, best, witness, cuts)
        stats[str(chain_index)] = chain_stats
        np.savez_compressed(
            checkpoint / f"chain_{chain_index:02d}.npz",
            best=best,
            witness=witness,
            models=sequence,
            stats=json.dumps(chain_stats),
        )
    frozen["4"] = selected
    for depth in ("1", "2", "3", "4"):
        unique = {json.dumps(encode(policy), sort_keys=True): policy for policy in frozen[depth]}
        for policy in unique.values():
            np.testing.assert_allclose(
                evaluate(policy, calibration)[:2],
                [policy["cal_cost"], policy["cal_quality"]],
                atol=1e-12,
            )
    manifest = {
        "dataset": dataset,
        "arm": arm,
        "split": split,
        "seed": baseline["seed"],
        "embedding_data_sha256": str(arrays["fingerprint"]),
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "pool": list(arrays["models"]),
        "embedding_dimension": int(arrays["x"].shape[-1]),
        "cal_idx": baseline["cal_idx"],
        "test_idx": baseline["test_idx"],
        "budgets": baseline["budgets"],
        "frozen": frozen,
        "s4_search_stats": stats,
    }
    with (destination / f"split_{split:02d}.json").open("x") as handle:
        json.dump(encode(manifest), handle, allow_nan=False)

    test = sliced(baseline["test_idx"])
    records = []
    memo = {}
    for budget_index, budget in enumerate(budgets):
        record = {
            "dataset": dataset,
            "arm": arm,
            "split": split,
            "fraction": FRACTIONS[budget_index],
            "budget": budget,
        }
        for depth in (1, 2, 3, 4):
            policy = frozen[str(depth)][budget_index]
            policy_key = json.dumps(encode(policy), sort_keys=True)
            if policy_key not in memo:
                memo[policy_key] = evaluate(policy, test)
            cost, accuracy, calls = memo[policy_key]
            values = {
                "cost": cost,
                "accuracy": accuracy,
                "mean_calls": calls,
                "depth": len(policy["models"]),
                "cal_cost": policy["cal_cost"],
                "cal_accuracy": policy["cal_quality"],
                "overshoot": cost - budget,
            }
            record.update({f"s{depth}_{name}": value for name, value in values.items()})
        for depth in (2, 3, 4):
            if record[f"s{depth}_cal_accuracy"] < record[f"s{depth-1}_cal_accuracy"] - 1e-12:
                raise AssertionError("Calibration nesting failed")
        records.append(record)
    pd.DataFrame(records).to_csv(
        destination / f"split_{split:02d}.csv", index=False, mode="x"
    )
    print(
        f"{dataset} {arm} {split + 1}/50: {time.perf_counter() - started:.1f}s",
        flush=True,
    )


def summarize(output, datasets, arms, splits):
    records = []
    for dataset in datasets:
        for arm in arms:
            for split in range(splits):
                frame = pd.read_csv(output / dataset / arm / f"split_{split:02d}.csv")
                record = {"dataset": dataset, "arm": arm, "split": split}
                for depth in (1, 2, 3, 4):
                    record[f"s{depth}_accuracy"] = frame[f"s{depth}_accuracy"].mean()
                    record[f"s{depth}_cost"] = frame[f"s{depth}_cost"].mean()
                    record[f"s{depth}_gain_over_s1_pp"] = 100 * (
                        frame[f"s{depth}_accuracy"] - frame.s1_accuracy
                    ).mean()
                record["s3_minus_s2_pp"] = 100 * (frame.s3_accuracy - frame.s2_accuracy).mean()
                record["s4_minus_s2_pp"] = 100 * (frame.s4_accuracy - frame.s2_accuracy).mean()
                record["s4_minus_s3_pp"] = 100 * (frame.s4_accuracy - frame.s3_accuracy).mean()
                records.append(record)
    split_frame = pd.DataFrame(records)
    split_frame.to_csv(output / "split_summary.csv", index=False, mode="x")
    summary = split_frame.groupby(["dataset", "arm"]).mean(numeric_only=True)
    summary.to_csv(output / "summary.csv", mode="x")
    print(summary[["s3_minus_s2_pp", "s4_minus_s2_pp", "s4_minus_s3_pp"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--out", type=Path, default=Path("results/next_model_embedding_diff01_s4")
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source_snapshot = args.out / "executed_source"
    source_snapshot.mkdir()
    for source in SOURCES:
        shutil.copy2(ROOT / source, source_snapshot / source)
    provenance = {
        "datasets": args.datasets,
        "arms": args.arms,
        "splits": args.splits,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "regression": "standardized ridge, alpha=1, unpenalized intercept",
        "target": "U_next-U_current, negative prediction as score",
        "source_sha256": {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
            for source in SOURCES
        },
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.out / "inputs").mkdir()
    with ProcessPoolExecutor(max_workers=min(args.workers, 3)) as pool:
        jobs = [pool.submit(prepare_input, dataset, args.out) for dataset in args.datasets]
        for job in jobs:
            job.result()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(run_split, dataset, arm, split, args.out)
            for split in range(args.splits)
            for dataset in args.datasets
            for arm in args.arms
        ]
        for job in jobs:
            job.result()
    summarize(args.out, args.datasets, args.arms, args.splits)
