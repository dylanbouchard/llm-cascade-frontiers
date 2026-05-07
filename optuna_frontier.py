"""
Estimate the k-model cascade Pareto frontier using Optuna NSGA-II.

For each trial, Optuna jointly selects:
  - A model sequence (ordered subset of available models by cost)
  - Thresholds τ_1, ..., τ_{k-1} for escalation decisions

Objectives (multi-objective):
  - Minimize E[C] (expected cost per query)
  - Maximize E[U] (expected accuracy)

All evaluations use pre-collected data - no API calls.
The resulting Pareto front is compared against the 2-model global frontier.

Usage:
    python3 optuna_frontier.py mmlu
    python3 optuna_frontier.py mmlu triviaqa math_hard
"""

import sys
import os
import tempfile
import numpy as np
import pandas as pd
import optuna
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "llm_cascade_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tiktoken
from pathlib import Path
from itertools import combinations
from scipy.interpolate import interp1d

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/output_data")
SCORER   = "mean_token_negentropy"
N_TRIALS = 2000
MAX_K    = 6    # maximum cascade length (use all 6 models)
CAL_FRAC = 0.5  # fraction used for calibration; metrics reported on test set
SEED     = 42

PRICE_PER_TOKEN = {
    "gpt-4o-mini":   (0.15e-6,  0.60e-6),
    "gpt-4o":        (2.50e-6, 10.00e-6),
    "llama-3.1-8b":  (0.10e-6,  0.10e-6),
    "llama-3.3-70b": (0.88e-6,  0.88e-6),
    "qwen2.5-7b":    (0.30e-6,  0.30e-6),
    "deepseek-v3":   (0.60e-6,  1.70e-6),
    "gpt-oss-20b":   (0.05e-6,  0.20e-6),
    "MiniMax-M2.7":  (0.30e-6,  1.20e-6),
}

_enc = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def compute_costs(df: pd.DataFrame, model: str) -> np.ndarray:
    p_in, p_out = PRICE_PER_TOKEN[model]
    n_in  = df["prompt"].apply(lambda x: len(_enc.encode(x))).values
    n_out = df["logprob"].apply(len).values
    return n_in * p_in + n_out * p_out


def load_dataset(dataset: str, n_rows: int = 2000) -> tuple[dict, dict]:
    """
    Load all available model data for a dataset, aligned by row.
    Applies a stratified 50/50 cal/test split; returns (cal_data, test_data).
    Each is a dict: model_name -> {scores, correct, costs, prompt}.
    """
    models_by_cost = sorted(
        PRICE_PER_TOKEN.keys(),
        key=lambda m: sum(PRICE_PER_TOKEN[m])
    )
    raw = {}
    for model in models_by_cost:
        path = DATA_DIR / f"{dataset}-{model}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "correct" not in df.columns or SCORER not in df.columns:
            continue
        if n_rows and n_rows < len(df):
            df = df.iloc[:n_rows].reset_index(drop=True)
        raw[model] = df

    if not raw:
        return {}, {}

    # Stratified cal/test split using first model's correctness labels
    ref_correct = next(iter(raw.values()))["correct"].values
    rng   = np.random.default_rng(SEED)
    idx_0 = np.where(ref_correct == 0)[0]
    idx_1 = np.where(ref_correct == 1)[0]
    n0    = int(len(idx_0) * CAL_FRAC)
    n1    = int(len(idx_1) * CAL_FRAC)
    cal_idx  = np.sort(np.concatenate([
        rng.choice(idx_0, n0, replace=False),
        rng.choice(idx_1, n1, replace=False),
    ]))
    test_idx = np.setdiff1d(np.arange(len(ref_correct)), cal_idx)

    def _extract(df, model, idx):
        return {
            "scores":  df[SCORER].values[idx],
            "correct": df["correct"].values[idx].astype(float),
            "costs":   compute_costs(df, model)[idx],
            "prompt":  df["prompt"].values[idx],
        }

    cal_data  = {m: _extract(raw[m], m, cal_idx)  for m in raw}
    test_data = {m: _extract(raw[m], m, test_idx) for m in raw}
    return cal_data, test_data


def verify_alignment(data: dict):
    """Assert all models share the same prompts (same rows)."""
    models = list(data.keys())
    if len(models) < 2:
        return
    ref = data[models[0]]["prompt"]
    for m in models[1:]:
        assert np.array_equal(ref, data[m]["prompt"]), \
            f"Row mismatch between {models[0]} and {m}"

# ---------------------------------------------------------------------------
# Cascade simulation
# ---------------------------------------------------------------------------

def simulate_cascade(
    model_sequence: list[str],
    thresholds: list[float],
    data: dict,
) -> tuple[float, float]:
    """
    Simulate a k-model cascade on all queries.

    Each model in the sequence always runs for queries that haven't been
    handled yet (cost accumulates). A query is handled at model i if
    score_i >= tau_i (don't escalate). The last model handles everything
    remaining.

    Returns (mean_cost, mean_quality).
    """
    k = len(model_sequence)
    n = len(data[model_sequence[0]]["costs"])

    cumulative_cost = np.zeros(n)
    quality         = np.zeros(n)
    handled         = np.zeros(n, dtype=bool)

    for i, model in enumerate(model_sequence):
        d = data[model]
        # Every unhandled query incurs this model's cost
        cumulative_cost += (~handled).astype(float) * d["costs"]

        if i < k - 1:
            score    = d["scores"]
            escalate = (~handled) & (score < thresholds[i])
            stay     = (~handled) & ~escalate
            quality[stay]  = d["correct"][stay]
            handled[stay]  = True
            # Queries that escalate continue to next model
        else:
            # Last model handles all remaining
            quality[~handled] = d["correct"][~handled]
            handled[:] = True

    return cumulative_cost.mean(), quality.mean()

# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(data: dict):
    available  = list(data.keys())   # already sorted by cost
    n_avail    = len(available)

    # Pre-enumerate all valid model sequences (cost-ordered subsets)
    all_sequences = []
    for k in range(2, min(MAX_K, n_avail) + 1):
        for combo in combinations(range(n_avail), k):
            all_sequences.append(tuple(available[i] for i in combo))

    seq_indices = list(range(len(all_sequences)))

    # Score ranges for threshold normalisation
    score_ranges = {
        m: (float(data[m]["scores"].min()), float(data[m]["scores"].max()))
        for m in available
    }

    def objective(trial: optuna.Trial):
        seq_idx       = trial.suggest_categorical("seq_idx", seq_indices)
        model_sequence = all_sequences[seq_idx]
        k             = len(model_sequence)

        # Suggest MAX_K-1 normalised thresholds in [0,1] - always defined
        # regardless of k so NSGA-II sees a fixed-dimensional space.
        tau_norms = [
            trial.suggest_float(f"tau_{i}", 0.0, 1.0)
            for i in range(MAX_K - 1)
        ]

        # Rescale to actual score ranges for the active models
        thresholds = [
            score_ranges[model_sequence[i]][0]
            + tau_norms[i] * (score_ranges[model_sequence[i]][1]
                              - score_ranges[model_sequence[i]][0])
            for i in range(k - 1)
        ]

        mean_cost, mean_qual = simulate_cascade(list(model_sequence), thresholds, data)
        return mean_cost, -mean_qual   # minimise both

    return objective, all_sequences

# ---------------------------------------------------------------------------
# 2-model baseline frontier (for comparison)
# ---------------------------------------------------------------------------

def two_model_frontier(data: dict, n_thresh: int = 200, n_grid: int = 2000):
    """Recompute the 2-model global Pareto frontier from data."""
    available  = list(data.keys())
    mean_cost  = {m: data[m]["costs"].mean() for m in available}
    mean_acc   = {m: data[m]["correct"].mean() for m in available}

    valid_pairs = [
        (cheap, exp)
        for cheap in available for exp in available
        if mean_cost[cheap] < mean_cost[exp] and mean_acc[exp] > mean_acc[cheap]
    ]

    all_curves = []
    for cheap, exp in valid_pairs:
        s   = data[cheap]["scores"]
        u_L = data[cheap]["correct"].astype(float)
        u_H = data[exp]["correct"].astype(float)
        c_L = data[cheap]["costs"].mean()
        c_H = data[exp]["costs"].mean()

        thresholds = np.linspace(s.min(), s.max(), n_thresh)
        cs, qs = [c_L], [u_L.mean()]
        for tau in thresholds:
            esc = s < tau
            p   = esc.mean()
            c   = c_L + p * c_H
            if c <= c_H:
                cs.append(c)
                qs.append(np.where(esc, u_H, u_L).mean())
        cs.append(c_H); qs.append(u_H.mean())
        all_curves.append((np.array(cs), np.array(qs)))

    if not all_curves:
        return np.array([]), np.array([])
    min_c = min(c.min() for c, _ in all_curves)
    max_c = max(c.max() for c, _ in all_curves)
    grid  = np.linspace(min_c, max_c, n_grid)

    interped = [
        interp1d(c, q, bounds_error=False, fill_value=(np.nan, np.nan))(grid)
        for c, q in all_curves
    ]
    env = np.nanmax(interped, axis=0)
    env = np.maximum.accumulate(np.where(np.isnan(env), -np.inf, env))
    env = np.where(env == -np.inf, np.nan, env)
    q_max = np.nanmax(env)
    idx   = np.where(~np.isnan(env) & (env >= q_max - 1e-12))[0]
    if len(idx):
        grid = grid[:idx[0] + 1]
        env  = env[:idx[0] + 1]
    return grid, env

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset: str):
    print(f"\n=== {dataset} ===")
    cal_data, test_data = load_dataset(dataset)
    # Optuna searches on cal set; final metrics reported on test set
    data = cal_data
    verify_alignment(data)

    available = list(data.keys())
    print(f"Models: {available}")

    # --- Optuna k-model frontier ---
    objective_fn, all_sequences = make_objective(data)
    print(f"Search space: {len(all_sequences)} model sequences, "
          f"{N_TRIALS} trials, NSGA-II")

    sampler = optuna.samplers.NSGAIISampler(population_size=100, seed=42)
    study   = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=sampler,
    )
    study.optimize(objective_fn, n_trials=N_TRIALS, show_progress_bar=True)

    # --- Re-evaluate Pareto solutions on held-out TEST set ---
    # Score ranges on cal set (used for threshold rescaling during search)
    cal_score_ranges = {
        m: (float(cal_data[m]["scores"].min()), float(cal_data[m]["scores"].max()))
        for m in available
    }
    test_costs, test_quals = [], []
    for t in study.best_trials:
        seq = all_sequences[t.params["seq_idx"]]
        k   = len(seq)
        tau_norms = [t.params[f"tau_{i}"] for i in range(MAX_K - 1)]
        thresholds = [
            cal_score_ranges[seq[i]][0]
            + tau_norms[i] * (cal_score_ranges[seq[i]][1]
                              - cal_score_ranges[seq[i]][0])
            for i in range(k - 1)
        ]
        c, q = simulate_cascade(list(seq), thresholds, test_data)
        test_costs.append(c)
        test_quals.append(q)

    pk_costs = np.array(test_costs)
    pk_quals = np.array(test_quals)

    # Sort and enforce monotonicity on test-set evaluations
    order    = np.argsort(pk_costs)
    pk_costs = pk_costs[order]
    pk_quals = pk_quals[order]
    pk_quals = np.maximum.accumulate(pk_quals)

    # --- Save Pareto front to parquet ---
    out_dir = Path("results/kmodel")
    out_dir.mkdir(parents=True, exist_ok=True)
    pareto_df = pd.DataFrame({"cost": pk_costs, "quality": pk_quals})
    pareto_df.to_parquet(out_dir / f"{dataset}_pareto.parquet", index=False)
    print(f"Saved test-set Pareto front to {out_dir / f'{dataset}_pareto.parquet'}")

    # --- 2-model baseline (evaluated on test set for fair comparison) ---
    print("Computing 2-model baseline frontier ...")
    base_c, base_q = two_model_frontier(test_data)

    # --- Print summary of best sequences found ---
    print("\nTop Pareto-optimal sequences:")
    seq_counts = {}
    for t in study.best_trials:
        seq = all_sequences[t.params["seq_idx"]]
        seq_counts[seq] = seq_counts.get(seq, 0) + 1
    for seq, cnt in sorted(seq_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {' → '.join(seq)}  ({cnt} Pareto trials)")

    # --- Quality gap at median budget ---
    med_cost = np.median(base_c)
    f2 = interp1d(base_c, base_q, bounds_error=False, fill_value=np.nan)
    fk = interp1d(pk_costs, pk_quals, bounds_error=False, fill_value=np.nan)
    q2 = float(f2(med_cost))
    qk = float(fk(med_cost))
    print(f"\nAt median 2-model cost ({med_cost*1e6:.1f} µ$): "
          f"2-model acc={q2:.4f}, k-model acc={qk:.4f}, "
          f"Δ={qk-q2:+.4f}")


if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["mmlu"]
    for ds in datasets:
        run_dataset(ds)
