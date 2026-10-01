"""Extend the frozen embedding Diff-01 experiment from exact S4 to exact S5."""
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

from exact_heldout_depth import encode
from next_model_deep_search import cutoffs, merge, predict, search
from next_model_embedding_diff01 import evaluate, fit_embedding_edges
from voi_compute import DATASETS

ROOT = Path(__file__).resolve().parent
PARENT = ROOT / "results/next_model_embedding_diff01_s4_full"
ARM = "embedding_diff01_ridge"
SOURCES = (
    "next_model_embedding_diff01_s5.py",
    "next_model_embedding_diff01.py",
    "next_model_deep_search.py",
    "next_model_depth_kernel.cpp",
    "continuation_benefit_compute.py",
)
INPUT_CACHE = {}


def load_input(dataset):
    key = str(PARENT / "inputs" / f"{dataset}.npz")
    if key not in INPUT_CACHE:
        with np.load(key) as arrays:
            INPUT_CACHE[key] = {name: arrays[name] for name in arrays.files}
    return INPUT_CACHE[key]


def run_split(dataset, split, output):
    destination = output / dataset / ARM
    destination.mkdir(parents=True, exist_ok=True)
    final_csv = destination / f"split_{split:02d}.csv"
    if final_csv.exists():
        return

    arrays = load_input(dataset)
    rows = arrays["rows"]
    parent_path = PARENT / dataset / ARM / f"split_{split:02d}.json"
    parent_bytes = parent_path.read_bytes()
    parent = json.loads(parent_bytes)

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

    calibration = sliced(parent["cal_idx"])
    models, edges = fit_embedding_edges(calibration, ARM)
    budgets = np.asarray(parent["budgets"])
    selected = list(parent["frozen"]["4"])
    checkpoint = destination / f"split_{split:02d}_s5_chains"
    checkpoint.mkdir(exist_ok=True)
    stats = {}
    started = time.perf_counter()

    for chain_index, sequence in enumerate(combinations(models, 5)):
        path = checkpoint / f"chain_{chain_index:02d}.npz"
        incumbent_hash = hashlib.sha256(
            json.dumps(encode(selected), sort_keys=True).encode()
        ).hexdigest()
        if path.exists():
            with np.load(path) as saved:
                if str(saved["incumbent_hash"]) != incumbent_hash:
                    raise ValueError(f"Checkpoint incumbent mismatch: {path}")
                best = saved["best"]
                witness = saved["witness"]
                chain_stats = json.loads(str(saved["stats"]))
            cuts = [
                cutoffs(predict(calibration[a]["x"], edges[a, b]))
                for a, b in zip(sequence[:-1], sequence[1:])
            ]
        else:
            chain_started = time.perf_counter()
            best, witness, cuts, chain_stats = search(
                calibration, sequence, edges, budgets, selected
            )
            chain_stats["seconds"] = time.perf_counter() - chain_started
            if not chain_stats["complete"]:
                raise RuntimeError(
                    f"Incomplete chain {dataset}/{split}/{chain_index}: {chain_stats}"
                )
            temporary = path.with_suffix(".partial.npz")
            np.savez_compressed(
                temporary,
                best=best,
                witness=witness,
                models=sequence,
                stats=json.dumps(chain_stats),
                incumbent_hash=incumbent_hash,
            )
            temporary.rename(path)
        selected = merge(selected, budgets, sequence, edges, best, witness, cuts)
        stats[str(chain_index)] = chain_stats

    unique = {json.dumps(encode(policy), sort_keys=True): policy for policy in selected}
    for policy in unique.values():
        np.testing.assert_allclose(
            evaluate(policy, calibration)[:2],
            [policy["cal_cost"], policy["cal_quality"]],
            atol=1e-12,
        )
    for s4_policy, s5_policy in zip(parent["frozen"]["4"], selected):
        if s5_policy["cal_quality"] < s4_policy["cal_quality"] - 1e-12:
            raise AssertionError("Calibration nesting failed")

    manifest = {
        "dataset": dataset,
        "arm": ARM,
        "split": split,
        "parent": str(parent_path.relative_to(ROOT)),
        "parent_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "pool": list(arrays["models"]),
        "embedding_dimension": int(arrays["x"].shape[-1]),
        "cal_idx": parent["cal_idx"],
        "test_idx": parent["test_idx"],
        "budgets": parent["budgets"],
        "frozen_s5": selected,
        "s5_search_stats": stats,
    }
    with (destination / f"split_{split:02d}.json").open("x") as handle:
        json.dump(encode(manifest), handle, allow_nan=False)

    test = sliced(parent["test_idx"])
    frame = pd.read_csv(PARENT / dataset / ARM / f"split_{split:02d}.csv")
    records = []
    memo = {}
    for budget, policy in zip(budgets, selected):
        policy_key = json.dumps(encode(policy), sort_keys=True)
        if policy_key not in memo:
            memo[policy_key] = evaluate(policy, test)
        cost, accuracy, calls = memo[policy_key]
        records.append(
            {
                "s5_cost": cost,
                "s5_accuracy": accuracy,
                "s5_mean_calls": calls,
                "s5_depth": len(policy["models"]),
                "s5_cal_cost": policy["cal_cost"],
                "s5_cal_accuracy": policy["cal_quality"],
                "s5_overshoot": cost - budget,
            }
        )
    result = pd.concat([frame, pd.DataFrame(records)], axis=1)
    if not (result.s5_cal_accuracy >= result.s4_cal_accuracy - 1e-12).all():
        raise AssertionError("Calibration nesting failed after serialization")
    result.to_csv(final_csv, index=False, mode="x")
    print(
        f"{dataset} {split + 1}/50 S5: {time.perf_counter() - started:.1f}s",
        flush=True,
    )


def summarize(output, datasets, splits):
    records = []
    for dataset in datasets:
        for split in range(splits):
            frame = pd.read_csv(output / dataset / ARM / f"split_{split:02d}.csv")
            records.append(
                {
                    "dataset": dataset,
                    "arm": ARM,
                    "split": split,
                    "s5_accuracy": frame.s5_accuracy.mean(),
                    "s5_cost": frame.s5_cost.mean(),
                    "s5_gain_over_s1_pp": 100
                    * (frame.s5_accuracy - frame.s1_accuracy).mean(),
                    "s5_minus_s2_pp": 100
                    * (frame.s5_accuracy - frame.s2_accuracy).mean(),
                    "s5_minus_s4_pp": 100
                    * (frame.s5_accuracy - frame.s4_accuracy).mean(),
                }
            )
    split_frame = pd.DataFrame(records)
    split_frame.to_csv(output / "split_summary.csv", index=False, mode="x")
    summary = split_frame.groupby(["dataset", "arm"]).mean(numeric_only=True)
    summary.to_csv(output / "summary.csv", mode="x")
    print(summary[["s5_minus_s2_pp", "s5_minus_s4_pp"]].to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    parser.add_argument("--splits", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/next_model_embedding_diff01_s5_full",
    )
    args = parser.parse_args()
    provenance = {
        "datasets": args.datasets,
        "splits": args.splits,
        "arm": ARM,
        "parent": str(PARENT.relative_to(ROOT)),
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "regression": "standardized ridge, alpha=1, unpenalized intercept",
        "target": "U_next-U_current, negative prediction as score",
        "source_sha256": {
            source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()
            for source in SOURCES
        },
    }
    if args.resume:
        if json.loads((args.out / "provenance.json").read_text()) != provenance:
            raise ValueError("Resume provenance mismatch")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        snapshot = args.out / "executed_source"
        snapshot.mkdir()
        for source in SOURCES:
            shutil.copy2(ROOT / source, snapshot / source)
        (args.out / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [
            pool.submit(run_split, dataset, split, args.out)
            for split in range(args.splits)
            for dataset in args.datasets
        ]
        for job in jobs:
            job.result()
    summarize(args.out, args.datasets, args.splits)
