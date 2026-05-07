"""
figures.py - Generate all paper figures from cascade_core.py results.

Figures:
  Fig 1: Global Pareto frontiers - 2x2 grid (4 datasets), all pair curves +
          bootstrap envelope bands + individual model coordinates + switching points.
  Fig 2: Envelope vs k-cascade vs learned router (per dataset).
  Fig 3: Escalation benefit curves - m_H(τ)−m_L(τ) vs τ, isotonic fit,
          one representative pair per dataset.
  Fig shadow: Shadow price λ*(τ) along the frontier for representative pairs.

Usage:
    python3 figures.py              # regenerates bundled cached figures
    python3 figures.py fig2
    python3 figures.py fig3
    python3 figures.py figA5
"""

import json
import os
import tempfile
import warnings
import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "llm_cascade_mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from scipy.interpolate import interp1d

warnings.filterwarnings(
    "ignore",
    message="All-NaN slice encountered",
    category=RuntimeWarning,
)

RESULTS_DIR  = Path("results/cascade_core")
FIG2_DIR     = Path("results/fig2")
ESC_BEN_DIR  = Path("results/escalation_benefit")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
DATASET_LABELS = {
    "mmlu":          "MMLU",
    "triviaqa":      "TriviaQA",
    "math_hard":     "MATH (levels 3-5)",
    "simpleqa":      "SimpleQA",
    "livecodebench": "LiveCodeBench",
}

# Colour palette: assign one colour per cheap model
CHEAP_MODEL_COLORS = {
    "llama-3.1-8b":  "#4878CF",   # blue
    "qwen2.5-7b":    "#6ACC65",   # green
    "gpt-4o-mini":   "#D65F5F",   # red
    "llama-3.3-70b": "#B47CC7",   # purple
    "deepseek-v3":   "#C4AD66",   # tan
    "gpt-4o":        "#77BEDB",   # light blue
    "gpt-oss-20b":   "#F28E2B",   # orange
    "MiniMax-M2.7":  "#59A14F",   # dark green
}

# Marker shape per model for individual model coordinates
MODEL_MARKERS = {
    "llama-3.1-8b":  "o",
    "qwen2.5-7b":    "s",
    "gpt-4o-mini":   "^",
    "llama-3.3-70b": "D",
    "deepseek-v3":   "P",
    "gpt-4o":        "*",
    "gpt-oss-20b":   "h",
    "MiniMax-M2.7":  "X",
}

MODEL_DISPLAY = {
    "llama-3.1-8b":  "Llama 3.1-8B",
    "qwen2.5-7b":    "Qwen2.5-7B",
    "gpt-4o-mini":   "GPT-4o-mini",
    "llama-3.3-70b": "Llama 3.3-70B",
    "deepseek-v3":   "DeepSeek-V3",
    "gpt-4o":        "GPT-4o",
    "gpt-oss-20b":   "GPT-oss-20B",
    "MiniMax-M2.7":  "MiniMax-M2.7",
}


# Scorer display names
SCORER_LABELS = {
    "sequence_probability":  "Seq. prob.",
    "min_probability":       "Min prob.",
    "min_token_negentropy":  "Min negentropy",
    "mean_token_negentropy": "Mean negentropy",
    "probability_margin":    "Prob. margin",
}

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       8,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi":      150,
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(dataset: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(RESULTS_DIR / dataset / f"{name}.parquet")


def clean_switching_points(env_df: pd.DataFrame, min_run: int = 40) -> pd.DataFrame:
    """
    Suppress noisy pair flickers from the envelope by requiring each pair to
    dominate for at least `min_run` consecutive grid points before recording
    a transition.
    """
    pairs = env_df["best_pair"].values
    costs = env_df["cost"].values

    # Run-length encode
    runs, i = [], 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j] == pairs[i]:
            j += 1
        runs.append({"pair": pairs[i], "start": costs[i],
                     "end": costs[j - 1], "length": j - i})
        i = j

    long_runs = [r for r in runs if r["length"] >= min_run]

    switches = []
    for k in range(1, len(long_runs)):
        if long_runs[k]["pair"] != long_runs[k - 1]["pair"]:
            switches.append({
                "cost":      long_runs[k]["start"],
                "from_pair": long_runs[k - 1]["pair"],
                "to_pair":   long_runs[k]["pair"],
            })
    return pd.DataFrame(switches) if switches else \
        pd.DataFrame(columns=["cost", "from_pair", "to_pair"])


def pair_color(pair_name: str) -> str:
    cheap = pair_name.split("→")[0]
    return CHEAP_MODEL_COLORS.get(cheap, "#888888")


def pair_linestyle(pair_name: str, all_pairs: list[str]) -> str:
    """Vary line style within the same cheap model to distinguish expensive targets."""
    cheap = pair_name.split("→")[0]
    same_cheap = [p for p in all_pairs if p.startswith(cheap + "→")]
    styles = ["-", "--", ":", "-."]
    idx = same_cheap.index(pair_name) if pair_name in same_cheap else 0
    return styles[idx % len(styles)]


# ---------------------------------------------------------------------------
# Full-sample envelope + sampled-split CI bands
# ---------------------------------------------------------------------------

def _full_sample_envelope(
    dataset: str,
    n_grid: int = 500,
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]],
           dict[str, tuple[float, float]]]:
    """
    Compute the envelope using ALL 2000 queries (no cal/test split).
    Returns (cost_grid, envelope_quality, curve_arrays, model_coords) with
    costs in dollars per query.
    """
    from cascade_core import (load_all_models, compute_costs,
                               compute_frontier, compute_envelope)
    from itertools import permutations

    SCORER    = "mean_token_negentropy"
    all_dfs   = load_all_models(dataset)
    all_costs = {m: compute_costs(df, m) for m, df in all_dfs.items()}

    mean_cost = {m: all_costs[m].mean() for m in all_dfs}
    mean_acc  = {m: all_dfs[m]["correct"].mean() for m in all_dfs}

    # Drop strictly dominated models: ∃ other model with lower cost AND higher accuracy
    dominated = {
        m for m in all_dfs
        if any(mean_cost[o] < mean_cost[m] and mean_acc[o] > mean_acc[m]
               for o in all_dfs if o != m)
    }
    non_dominated = set(all_dfs.keys()) - dominated

    valid_pairs = [
        (c, e) for c, e in permutations(list(non_dominated), 2)
        if mean_cost[c] < mean_cost[e] and mean_acc[e] > mean_acc[c]
    ]

    # Fix 4: compute model_coords directly from population stats, not curve endpoints.
    # Curve endpoints depend on s.min()/s.max() of the specific pair and get
    # overwritten when a model appears as both cheap and expensive across pairs.
    model_coords: dict[str, tuple[float, float]] = {
        m: (mean_cost[m], mean_acc[m]) for m in non_dominated
    }

    curve_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for cheap, exp in valid_pairs:
        s        = all_dfs[cheap][SCORER].values
        tau_grid = np.linspace(s.min(), s.max(), 200)
        costs_L  = all_costs[cheap]
        costs_H  = all_costs[exp]
        curve    = compute_frontier(all_dfs[cheap], all_dfs[exp],
                                    costs_L, costs_H, SCORER, tau_grid)
        c_arr = curve["E_C"].values   # dollars per query
        q_arr = curve["E_U"].values
        # Fix 5: assert cost is monotone so compute_envelope's interp1d is safe.
        assert np.all(np.diff(c_arr) >= -1e-12), \
            f"Non-monotone cost curve for {cheap}→{exp} on {dataset}"
        curve_arrays[f"{cheap}→{exp}"] = (c_arr, q_arr)

    g, q, _ = compute_envelope(curve_arrays, n_grid=n_grid)
    return g, q, curve_arrays, model_coords


def _sampled_split_ci(
    dataset:  str,
    cost_grid_us: np.ndarray,
    n_splits: int = 50,
    seed:     int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate 95% CI half-width at each cost grid point using n_splits
    independent random 50/50 stratified splits of the 2000-query pool.
    Returns (lower, upper) arrays aligned to cost_grid_us.
    CI = full_sample_envelope ± t_{0.975, n-1} * SE across splits.
    """
    from cascade_core import (load_all_models, compute_costs, cal_test_split,
                               compute_frontier, compute_envelope)
    from itertools import permutations
    from scipy.stats import t as t_dist

    SCORER    = "mean_token_negentropy"
    all_dfs   = load_all_models(dataset)
    all_costs = {m: compute_costs(df, m) for m, df in all_dfs.items()}

    mean_cost_full = {m: all_costs[m].mean() for m in all_dfs}
    mean_acc_full  = {m: all_dfs[m]["correct"].mean() for m in all_dfs}
    base_pairs = [
        (c, e) for c, e in permutations(list(all_dfs.keys()), 2)
        if mean_cost_full[c] < mean_cost_full[e]
        and mean_acc_full[e] > mean_acc_full[c]
    ]

    n_grid      = len(cost_grid_us)
    n           = len(next(iter(all_dfs.values())))
    ref_correct = next(iter(all_dfs.values()))["correct"].values
    split_envs  = np.full((n_splits, n_grid), np.nan)

    for b in range(n_splits):
        cal_idx, test_idx = cal_test_split(n, ref_correct, seed=seed + b + 1)
        cal   = {m: df.iloc[cal_idx].reset_index(drop=True)  for m, df in all_dfs.items()}
        test  = {m: df.iloc[test_idx].reset_index(drop=True) for m, df in all_dfs.items()}
        c_cal = {m: all_costs[m][cal_idx]  for m in all_dfs}
        c_tst = {m: all_costs[m][test_idx] for m in all_dfs}

        mc_b = {m: c_tst[m].mean() for m in all_dfs}
        ma_b = {m: test[m]["correct"].mean() for m in all_dfs}
        valid_b = [(c, e) for c, e in base_pairs
                   if mc_b[c] < mc_b[e] and ma_b[e] > ma_b[c]]
        if not valid_b:
            continue

        curve_arrays = {}
        for cheap, exp in valid_b:
            s_cal    = cal[cheap][SCORER].values
            tau_grid = np.linspace(s_cal.min(), s_cal.max(), 200)
            curve    = compute_frontier(test[cheap], test[exp],
                                        c_tst[cheap], c_tst[exp], SCORER, tau_grid)
            curve_arrays[f"{cheap}→{exp}"] = (
                curve["E_C"].values, curve["E_U"].values)

        g, q, _ = compute_envelope(curve_arrays, n_grid=n_grid)
        valid = ~np.isnan(q)
        if valid.sum() > 1:
            f = interp1d(g[valid] * 1e3, q[valid], bounds_error=False,
                         fill_value=(np.nan, np.nan))
            split_envs[b] = f(cost_grid_us)

    n_valid = np.sum(~np.isnan(split_envs), axis=0)
    std_env = np.nanstd(split_envs, axis=0, ddof=1)
    se_env  = np.where(n_valid > 1, std_env / np.sqrt(n_valid), np.nan)
    t_crit  = t_dist.ppf(0.975, df=np.maximum(n_valid - 1, 1))
    radius  = t_crit * se_env
    return radius


# ---------------------------------------------------------------------------
# Figure 1 - Global Pareto frontiers
# ---------------------------------------------------------------------------

def fig1_pareto_frontiers(save: bool = True):
    fig, axes = plt.subplots(1, 5, figsize=(16.9, 3.2))
    axes = axes.flatten()

    # Collect which models appear (after dominance filter) across all datasets,
    # preserving MODEL_MARKERS order.
    appearing_models: set[str] = set()

    for ax, dataset in zip(axes, DATASETS):
        cost_grid, env_q, curve_arrays, model_coords = _full_sample_envelope(dataset)
        appearing_models |= set(model_coords.keys())

        # All pair curves in muted gray - focus is on the envelope, not individual pairs
        for c_arr, q_arr in curve_arrays.values():
            ax.plot(c_arr, q_arr, color="#BBBBBB", lw=0.75, alpha=0.6)

        # Global envelope
        ax.plot(cost_grid, env_q, color="black", lw=1.6, ls="--",
                alpha=0.9, zorder=5)

        # Switching points (precomputed envelope is stored in dollars)
        f_env  = interp1d(cost_grid[~np.isnan(env_q)],
                          env_q[~np.isnan(env_q)],
                          bounds_error=False, fill_value=np.nan)
        env_df = load(dataset, "envelope")
        for _, sw in clean_switching_points(env_df).iterrows():
            sw_c = float(sw["cost"])   # dollars → dollars (no conversion needed)
            sw_q = float(f_env(sw_c))
            if not np.isnan(sw_q):
                ax.scatter(sw_c, sw_q, color="black", s=30, zorder=6,
                           marker="D", linewidths=0.4, edgecolors="white")

        # Individual model coordinates - unique color + shape per model
        for mdl, (mc, mq) in model_coords.items():
            ax.scatter(mc, mq,
                       color=CHEAP_MODEL_COLORS[mdl],
                       marker=MODEL_MARKERS[mdl],
                       s=55, zorder=7, edgecolors="white", linewidths=0.6)

        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Expected cost per query (\\$)")
        ax.set_ylabel("Expected accuracy")
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.set_xlim(left=0)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # Legend: one entry per model (color + shape), plus envelope + switching point
    model_handles = [
        Line2D([0], [0],
               color=CHEAP_MODEL_COLORS[m], marker=MODEL_MARKERS[m],
               ms=5, lw=0, markeredgecolor="white", markeredgewidth=0.5,
               label=MODEL_DISPLAY[m])
        for m in MODEL_MARKERS   # fixed order from MODEL_MARKERS dict
        if m in appearing_models
    ]
    meta_handles = [
        Line2D([0], [0], color="black", lw=1.6, ls="--", label="Global envelope"),
        Line2D([0], [0], color="black", marker="D", ms=4, lw=0,
               markeredgecolor="white", markeredgewidth=0.4, label="Switching point"),
    ]
    fig.legend(handles=model_handles + meta_handles,
               loc="lower center", ncol=4, frameon=True,
               bbox_to_anchor=(0.5, -0.16), fontsize=6.5)

    fig.suptitle("Cost–Quality Pareto Frontiers", fontsize=10, y=1.01)
    plt.tight_layout(rect=(0, 0.12, 1, 0.98))
    path = FIGURES_DIR / "fig1_pareto_frontiers.pdf"
    if save:
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
        print(f"Saved {path}")
    return fig


def _select_representative_pair(dataset: str) -> str:
    """
    Select the pair that contributes the most cost-range coverage on the
    global pairwise envelope (longest contiguous segment on the envelope).
    This is the pair the paper's structural conditions are most relevant for.
    """
    env_df = load(dataset, "envelope")
    if env_df.empty or "best_pair" not in env_df.columns:
        # Fallback: pair with highest quality ceiling in non-dominated pool
        _, _, curve_arrays, _ = _full_sample_envelope(dataset)
        best = max(curve_arrays, key=lambda p: curve_arrays[p][1].max())
        return best

    # Count grid points dominated by each pair (after denoising)
    sw_df  = clean_switching_points(env_df)
    pairs  = env_df["best_pair"].values
    counts = {}
    for p in pairs:
        if p and not (isinstance(p, float) and np.isnan(p)):
            counts[p] = counts.get(p, 0) + 1
    if not counts:
        _, _, curve_arrays, _ = _full_sample_envelope(dataset)
        return max(curve_arrays, key=lambda p: curve_arrays[p][1].max())
    return max(counts, key=counts.__getitem__)


# Fixed from the manuscript analyses so figure regeneration does not require
# the omitted full cascade-core parquet cache.
REPRESENTATIVE_PAIRS: dict[str, str] = {
    "mmlu": "gpt-4o-mini→gpt-oss-20b",
    "triviaqa": "gpt-4o-mini→llama-3.3-70b",
    "math_hard": "gpt-oss-20b→deepseek-v3",
    "simpleqa": "deepseek-v3→gpt-4o",
    "livecodebench": "llama-3.3-70b→deepseek-v3",
}


# ---------------------------------------------------------------------------
# Figure 2 helpers - k-cascade and learned router splits
# ---------------------------------------------------------------------------

def _nondominated_pool(dataset: str):
    """
    Return the full-dataset non-dominated pool (model names + precomputed costs).
    Uses same filter as _full_sample_envelope so Figure 2 and Figure 1 are aligned.
    """
    from cascade_core import load_all_models, compute_costs
    all_dfs   = load_all_models(dataset)
    all_costs = {m: compute_costs(df, m) for m, df in all_dfs.items()}
    mean_cost = {m: all_costs[m].mean() for m in all_dfs}
    mean_acc  = {m: all_dfs[m]["correct"].mean() for m in all_dfs}
    dominated = {
        m for m in all_dfs
        if any(mean_cost[o] < mean_cost[m] and mean_acc[o] > mean_acc[m]
               for o in all_dfs if o != m)
    }
    pool = sorted(set(all_dfs.keys()) - dominated,
                  key=lambda m: mean_cost[m])
    return pool, all_dfs, all_costs


def _fig2_envelope_splits(
    dataset: str,
    cost_grid: np.ndarray,
    n_splits: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """
    Pairwise envelope quality evaluated on held-out test half, over n_splits
    random 50/50 stratified splits.  Returns (n_splits, len(cost_grid)) matrix.
    """
    from cascade_core import (compute_frontier, compute_envelope, cal_test_split)
    from itertools import permutations

    SCORER = "mean_token_negentropy"
    pool, all_dfs, all_costs = _nondominated_pool(dataset)
    n           = len(next(iter(all_dfs.values())))
    ref_correct = next(iter(all_dfs.values()))["correct"].values

    qual_matrix = np.full((n_splits, len(cost_grid)), np.nan)

    for b in range(n_splits):
        cal_idx, test_idx = cal_test_split(n, ref_correct, seed=seed + b)

        # Threshold calibration grid from cal set; evaluate on test set
        test_dfs   = {m: all_dfs[m].iloc[test_idx].reset_index(drop=True)
                      for m in pool}
        test_costs = {m: all_costs[m][test_idx] for m in pool}

        mean_c_tst = {m: test_costs[m].mean() for m in pool}
        mean_a_tst = {m: test_dfs[m]["correct"].mean() for m in pool}
        valid_pairs = [
            (c, e) for c, e in permutations(pool, 2)
            if mean_c_tst[c] < mean_c_tst[e] and mean_a_tst[e] > mean_a_tst[c]
        ]
        if not valid_pairs:
            continue

        curve_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for cheap, exp in valid_pairs:
            # Calibrate threshold grid on cal split, sweep on test
            s_cal    = all_dfs[cheap].iloc[cal_idx][SCORER].values
            tau_grid = np.linspace(s_cal.min(), s_cal.max(), 200)
            curve    = compute_frontier(test_dfs[cheap], test_dfs[exp],
                                        test_costs[cheap], test_costs[exp],
                                        SCORER, tau_grid)
            curve_arrays[f"{cheap}→{exp}"] = (
                curve["E_C"].values, curve["E_U"].values)

        g, q, _ = compute_envelope(curve_arrays, n_grid=500)
        valid = ~np.isnan(q)
        if valid.sum() > 1:
            f = interp1d(g[valid], q[valid], bounds_error=False,
                         fill_value=(np.nan, np.nan))
            qual_matrix[b] = f(cost_grid)

    return qual_matrix


def _fig2_kmodel_splits(
    dataset: str,
    cost_grid: np.ndarray,
    n_splits: int = 50,
    n_trials: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """
    k-model cascade quality on held-out test half over n_splits splits.
    Optuna NSGA-II calibrated on cal set, evaluated on test set.
    Returns (n_splits, len(cost_grid)) matrix.
    """
    import optuna
    from itertools import combinations
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    SCORER = "mean_token_negentropy"
    MAX_K  = 4
    pool, all_dfs, all_costs = _nondominated_pool(dataset)
    n           = len(next(iter(all_dfs.values())))
    ref_correct = next(iter(all_dfs.values()))["correct"].values

    # Pre-enumerate all valid model sequences from the non-dominated pool
    all_sequences = []
    for k in range(2, MAX_K + 1):
        for combo in combinations(range(len(pool)), k):
            all_sequences.append(tuple(pool[i] for i in combo))
    seq_indices = list(range(len(all_sequences)))

    qual_matrix = np.full((n_splits, len(cost_grid)), np.nan)

    for b in range(n_splits):
        from cascade_core import cal_test_split
        cal_idx, test_idx = cal_test_split(n, ref_correct, seed=seed + b)

        cal_data = {
            m: {
                "scores":  all_dfs[m].iloc[cal_idx][SCORER].values,
                "correct": all_dfs[m].iloc[cal_idx]["correct"].values.astype(float),
                "costs":   all_costs[m][cal_idx],
            }
            for m in pool
        }
        test_data = {
            m: {
                "scores":  all_dfs[m].iloc[test_idx][SCORER].values,
                "correct": all_dfs[m].iloc[test_idx]["correct"].values.astype(float),
                "costs":   all_costs[m][test_idx],
            }
            for m in pool
        }

        # Threshold ranges from cal set
        score_ranges = {
            m: (float(cal_data[m]["scores"].min()),
                float(cal_data[m]["scores"].max()))
            for m in pool
        }

        def _simulate(model_sequence, thresholds, data):
            k2 = len(model_sequence)
            n2 = len(data[model_sequence[0]]["costs"])
            cum_cost = np.zeros(n2)
            quality  = np.zeros(n2)
            handled  = np.zeros(n2, dtype=bool)
            for i, m in enumerate(model_sequence):
                cum_cost += (~handled).astype(float) * data[m]["costs"]
                if i < k2 - 1:
                    esc = (~handled) & (data[m]["scores"] < thresholds[i])
                    stay = (~handled) & ~esc
                    quality[stay] = data[m]["correct"][stay]
                    handled[stay] = True
                else:
                    quality[~handled] = data[m]["correct"][~handled]
                    handled[:] = True
            return cum_cost.mean(), quality.mean()

        def objective(trial):
            si   = trial.suggest_categorical("si", seq_indices)
            seq  = all_sequences[si]
            k2   = len(seq)
            taus = [
                score_ranges[seq[i]][0]
                + trial.suggest_float(f"t{i}", 0.0, 1.0)
                * (score_ranges[seq[i]][1] - score_ranges[seq[i]][0])
                for i in range(k2 - 1)
            ]
            c, q = _simulate(list(seq), taus, cal_data)
            return c, -q

        sampler = optuna.samplers.NSGAIISampler(population_size=50, seed=seed + b)
        study   = optuna.create_study(
            directions=["minimize", "minimize"], sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Re-evaluate best trials on test set
        pk_c, pk_q = [], []
        for t in study.best_trials:
            si  = t.params["si"]
            seq = all_sequences[si]
            k2  = len(seq)
            taus = [
                score_ranges[seq[i]][0]
                + t.params[f"t{i}"]
                * (score_ranges[seq[i]][1] - score_ranges[seq[i]][0])
                for i in range(k2 - 1)
            ]
            c2, q2 = _simulate(list(seq), taus, test_data)
            pk_c.append(c2); pk_q.append(q2)

        if not pk_c:
            continue
        pk_c = np.array(pk_c); pk_q = np.array(pk_q)
        order = np.argsort(pk_c)
        pk_c  = pk_c[order]; pk_q = pk_q[order]
        pk_q  = np.maximum.accumulate(pk_q)

        # Interpolate onto cost_grid
        if len(pk_c) > 1:
            f = interp1d(pk_c, pk_q, bounds_error=False,
                         fill_value=(np.nan, np.nan))
            qual_matrix[b] = f(cost_grid)

    return qual_matrix


def _fig2_router_splits(
    dataset: str,
    cost_grid: np.ndarray,
    n_splits: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """
    Learned routing (logistic regression on 5 UQ features from cheap model)
    per pair, envelope taken over all valid pairs, evaluated on test half.
    Returns (n_splits, len(cost_grid)) matrix.
    """
    from cascade_core import cal_test_split
    from itertools import permutations
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    SCORERS = ["mean_token_negentropy", "min_token_negentropy",
               "min_probability", "sequence_probability", "probability_margin"]

    pool, all_dfs, all_costs = _nondominated_pool(dataset)
    n           = len(next(iter(all_dfs.values())))
    ref_correct = next(iter(all_dfs.values()))["correct"].values

    qual_matrix = np.full((n_splits, len(cost_grid)), np.nan)

    for b in range(n_splits):
        cal_idx, test_idx = cal_test_split(n, ref_correct, seed=seed + b)

        test_costs = {m: all_costs[m][test_idx] for m in pool}
        mean_c_tst = {m: test_costs[m].mean() for m in pool}
        mean_a_tst = {m: all_dfs[m].iloc[test_idx]["correct"].mean() for m in pool}

        valid_pairs = [
            (c, e) for c, e in permutations(pool, 2)
            if mean_c_tst[c] < mean_c_tst[e] and mean_a_tst[e] > mean_a_tst[c]
        ]
        if not valid_pairs:
            continue

        pair_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for cheap, exp in valid_pairs:
            # Build feature matrices from cheap model UQ scores
            X_cal = all_dfs[cheap].iloc[cal_idx][SCORERS].values
            y_cal = (1 - all_dfs[cheap].iloc[cal_idx]["correct"].values)  # 1 = escalate
            X_tst = all_dfs[cheap].iloc[test_idx][SCORERS].values

            # Skip pair if calibration set is too small or all one class
            if y_cal.sum() < 5 or y_cal.sum() > len(y_cal) - 5:
                continue

            scaler = StandardScaler()
            X_cal_s = scaler.fit_transform(X_cal)
            X_tst_s = scaler.transform(X_tst)

            try:
                clf = LogisticRegression(max_iter=500, random_state=seed + b)
                clf.fit(X_cal_s, y_cal)
                p_esc = clf.predict_proba(X_tst_s)[:, 1]
            except Exception:
                continue

            # Sweep decision threshold to trace cost-quality curve
            u_L = all_dfs[cheap].iloc[test_idx]["correct"].values.astype(float)
            u_H = all_dfs[exp].iloc[test_idx]["correct"].values.astype(float)
            c_L = test_costs[cheap]
            c_H = test_costs[exp]

            thresholds = np.linspace(0.0, 1.0, 200)
            cs, qs = [], []
            for thresh in thresholds:
                esc  = p_esc >= thresh
                p    = esc.mean()
                c    = c_L.mean() + p * c_H.mean()
                q    = np.where(esc, u_H, u_L).mean()
                cs.append(c); qs.append(q)
            # Single-model endpoints
            cs = [c_L.mean()] + cs + [c_L.mean() + c_H.mean()]
            qs = [u_L.mean()] + qs + [u_H.mean()]
            cs_arr = np.array(cs); qs_arr = np.array(qs)
            order  = np.argsort(cs_arr)
            pair_curves.append((cs_arr[order], qs_arr[order]))

        if not pair_curves:
            continue

        # Envelope over all pairs
        min_c = min(c.min() for c, _ in pair_curves)
        max_c = max(c.max() for c, _ in pair_curves)
        g     = np.linspace(min_c, max_c, 500)
        interped = [
            interp1d(c, q, bounds_error=False, fill_value=(np.nan, np.nan))(g)
            for c, q in pair_curves
        ]
        env = np.nanmax(interped, axis=0)
        env = np.maximum.accumulate(np.where(np.isnan(env), -np.inf, env))
        env = np.where(env == -np.inf, np.nan, env)

        valid = ~np.isnan(env)
        if valid.sum() > 1:
            f = interp1d(g[valid], env[valid], bounds_error=False,
                         fill_value=(np.nan, np.nan))
            qual_matrix[b] = f(cost_grid)

    return qual_matrix


# ---------------------------------------------------------------------------
# Figure 2 - Envelope vs cascades vs learned router
# ---------------------------------------------------------------------------

def fig2_kmodel_vs_envelope(save: bool = True):
    """
    2x2 grid (one panel per dataset). Reads precomputed results from
    results/fig2/ rather than re-running Optuna.

    - Black dashed: pairwise envelope (deterministic, full dataset)
    - Blue + band:  pairwise envelope median ± 10-90th pct (50 splits)
    - Gray + band:  full fixed-chain cascade median ± 10-90th pct
    - Red + band:   subsequence cascade median ± 10-90th pct (Optuna NSGA-II)

    Shows median held-out frontiers with 10th--90th percentile bands.
    """
    ENV_COLOR    = "#4878CF"   # blue
    FIXED_COLOR  = "#6F6F6F"   # gray
    KMODEL_COLOR = "#D65F5F"   # red
    ROUTER_COLOR = "#27AE60"   # green

    fig, axes = plt.subplots(1, 5, figsize=(16.9, 3.0))
    axes = axes.flatten()

    for ax, dataset in zip(axes, DATASETS):
        ds_dir = FIG2_DIR / dataset
        cost_grid  = np.load(ds_dir / "cost_grid.npy")
        env_curves = np.load(ds_dir / "envelope_curves.npy")   # (50, N_GRID)
        fixed_curves = np.load(ds_dir / "fixed_chain_curves.npy")
        kc_curves     = np.load(ds_dir / "kcascade_curves.npy")
        router_curves = np.load(ds_dir / "router_curves.npy")
        for mat, color, label_base, ls in [
            (env_curves,    ENV_COLOR,    "Optimal pair",               "-"),
            (fixed_curves,  FIXED_COLOR,  r"Full fixed chain",           "-."),
            (kc_curves,     KMODEL_COLOR, r"Optimal subsequence",        "--"),
            (router_curves, ROUTER_COLOR, r"Diagnostic learned router", ":"),
        ]:
            med   = np.nanmedian(mat, axis=0)
            lo    = np.nanpercentile(mat, 10, axis=0)
            hi    = np.nanpercentile(mat, 90, axis=0)
            valid = ~np.isnan(med) & ~np.isnan(lo) & ~np.isnan(hi)
            if valid.sum() < 2:
                continue
            ax.fill_between(cost_grid[valid], lo[valid], hi[valid],
                            alpha=0.20, color=color)
            ax.plot(cost_grid[valid], med[valid],
                    color=color, lw=1.3, ls=ls, label=label_base)

        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Expected cost per query (\\$)")
        ax.set_ylabel("Expected accuracy")
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.set_xlim(left=0)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    from matplotlib.legend_handler import HandlerTuple
    legend_handles = [
        (mpatches.Patch(facecolor=ENV_COLOR,    alpha=0.25),
         Line2D([0], [0], color=ENV_COLOR,    lw=1.3, ls="-")),
        (mpatches.Patch(facecolor=FIXED_COLOR, alpha=0.25),
         Line2D([0], [0], color=FIXED_COLOR, lw=1.3, ls="-.")),
        (mpatches.Patch(facecolor=KMODEL_COLOR, alpha=0.25),
         Line2D([0], [0], color=KMODEL_COLOR, lw=1.3, ls="--")),
        (mpatches.Patch(facecolor=ROUTER_COLOR, alpha=0.25),
         Line2D([0], [0], color=ROUTER_COLOR, lw=1.3, ls=":")),
    ]
    legend_labels = [
        r"Optimal pair (med. $\pm$ 10--90th pct.)",
        r"Full fixed chain (med. $\pm$ 10--90th pct.)",
        r"Optimal subsequence (med. $\pm$ 10--90th pct.)",
        r"Diagnostic learned router (med. $\pm$ 10--90th pct.)",
    ]
    plt.tight_layout()
    fig.legend(
        handles=legend_handles, labels=legend_labels,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
        loc="lower center", ncol=4, frameon=True,
        bbox_to_anchor=(0.5, -0.12), fontsize=6.5,
    )
    path = FIGURES_DIR / "fig2_kmodel_vs_envelope.pdf"
    if save:
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
        print(f"Saved {path}")
    return fig


def fig3_escalation_benefit(save: bool = True):
    """
    1x4 row: escalation benefit m_H - m_L vs. confidence score s_L.
    Reads pre-computed smoothing-spline results from results/escalation_benefit/.

    Per panel:
    - Blue band: 10-90th pct. across 50 splits (per-split spline curves)
    - Blue line: median spline curve
    - Red dashed: isotonic non-increasing regression on the median
    - Green shading: regions where median gain > 0 (expected-dominance region)
    - Annotation: dom (expected-dominance fraction) and dec (decreasing-benefit fraction)
    """
    BAND_COLOR = "#4878CF"
    ISO_COLOR  = "#D65F5F"
    POS_COLOR  = "#6ACC65"

    fig, axes = plt.subplots(1, 5, figsize=(16.9, 2.8))
    axes = axes.flatten()

    for ax, ds in zip(axes, DATASETS):
        pair = (REPRESENTATIVE_PAIRS.get(ds) or _select_representative_pair(ds))
        cheap, exp = pair.split("→")

        npz_path = ESC_BEN_DIR / ds / f"{ds}_{cheap}_{exp}_mean_token_negentropy.npz"
        if not npz_path.exists():
            ax.set_visible(False)
            continue

        npz = np.load(npz_path, allow_pickle=True)
        s_grid       = npz["s_grid"]
        median_curve = npz["median_curve"]
        band_lo      = npz["band_lo"]
        band_hi      = npz["band_hi"]
        isotonic     = npz["isotonic_curve"]
        dom          = float(npz["dominance_frac"])
        dec          = float(npz["decreasing_frac"])

        valid = ~np.isnan(median_curve)

        # Shade regions where median gain > 0
        in_pos, seg_start = False, None
        x_v = s_grid[valid]
        g_v = median_curve[valid]
        dx  = (x_v[-1] - x_v[0]) / len(x_v) / 2 if len(x_v) > 1 else 0
        for xi, gi in zip(x_v, g_v):
            if gi > 0 and not in_pos:
                seg_start, in_pos = xi - dx, True
            elif gi <= 0 and in_pos:
                ax.axvspan(seg_start, xi + dx, alpha=0.08, color=POS_COLOR, lw=0, zorder=0)
                in_pos = False
        if in_pos:
            ax.axvspan(seg_start, x_v[-1] + dx, alpha=0.08, color=POS_COLOR, lw=0, zorder=0)

        ax.fill_between(s_grid[valid], band_lo[valid], band_hi[valid],
                        alpha=0.25, color=BAND_COLOR, zorder=2)
        ax.plot(s_grid[valid], median_curve[valid],
                color=BAND_COLOR, lw=1.4, zorder=3)
        ax.plot(s_grid[valid], isotonic[valid],
                color=ISO_COLOR, lw=1.3, ls="--", zorder=4)
        ax.axhline(0, color="gray", lw=0.6, ls=":", zorder=1)

        ax.set_title(
            f"{DATASET_LABELS[ds]}: "
            f"{MODEL_DISPLAY.get(cheap, cheap)}$\\to${MODEL_DISPLAY.get(exp, exp)}",
            fontsize=7.5,
        )
        ax.set_xlabel("Confidence score $s_L$", fontsize=7.5)
        ax.set_ylabel(r"$\hat{m}_H - \hat{m}_L$", fontsize=7.5)
        ax.grid(True, alpha=0.2, lw=0.5)

        ann = f"dom={dom:.0%}  dec={dec:.0%}"
        ax.text(0.97, 0.97, ann,
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                          alpha=0.85, lw=0.5))

    handles = [
        mpatches.Patch(facecolor=POS_COLOR, alpha=0.15,
                       label=r"$m_H > m_L$ region"),
        mpatches.Patch(facecolor=BAND_COLOR, alpha=0.3,
                       label="10-90th pct. (50 splits)"),
        Line2D([0], [0], color=BAND_COLOR, lw=1.4, label="Median spline"),
        Line2D([0], [0], color=ISO_COLOR, lw=1.3, ls="--",
               label="Isotonic (non-incr.) fit"),
        Line2D([0], [0], color="gray", lw=0.6, ls=":", label="Zero"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True,
               bbox_to_anchor=(0.5, -0.03), fontsize=6.5)
    fig.suptitle(
        r"Escalation Benefit $\hat{m}_H - \hat{m}_L$ vs.\ Confidence",
        fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "fig3_escalation_benefit.pdf"
    if save:
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
        print(f"Saved {path}")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Figure 3 (old) - Signal quality: oracle-normalised AUC vs AUROC
# ---------------------------------------------------------------------------

def fig3_signal_quality(save: bool = True):
    """
    Left: scatter of oracle-normalised AUC vs. AUROC across all (pair, dataset) cells.
    Right: heatmap of best oracle-normalised AUC per cheap model x dataset.
    """
    # Collect (pair, dataset, auroc, oracle_norm_auc)
    rows = []
    for ds in DATASETS:
        fauc     = load(ds, "frontier_auc")
        auroc_df = load(ds, "auroc")
        auroc_by_model = (
            auroc_df[auroc_df["scorer"] == "mean_token_negentropy"]
            .set_index("model")["auroc"]
        )
        for _, row in fauc.iterrows():
            cheap = row["pair"].split("→")[0]
            rows.append({
                "dataset":        ds,
                "pair":           row["pair"],
                "auroc":          auroc_by_model.get(cheap, np.nan),
                "oracle_norm_auc": row["oracle_norm_auc"],
            })
    df = pd.DataFrame(rows).dropna(subset=["auroc", "oracle_norm_auc"])

    ds_colors = {
        "mmlu":      "#4878CF",
        "triviaqa":  "#D65F5F",
        "math_hard": "#6ACC65",
        "simpleqa":  "#B47CC7",
    }

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.2))

    # --- Left: scatter ---
    ax = axes[0]
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        ax.scatter(sub["auroc"], sub["oracle_norm_auc"],
                   color=ds_colors[ds], s=22, alpha=0.75,
                   label=DATASET_LABELS[ds], zorder=3)
    # Pooled OLS line
    x_all = df["auroc"].values
    y_all = df["oracle_norm_auc"].values
    if len(x_all) > 2:
        m_ols, b_ols = np.polyfit(x_all, y_all, 1)
        xr = np.linspace(x_all.min(), x_all.max(), 100)
        ax.plot(xr, m_ols * xr + b_ols, color="black", lw=1.0, ls="--",
                label="OLS fit (pooled)", zorder=2)
    ax.set_xlabel("AUROC of $s_L$ vs. cheap-model correctness")
    ax.set_ylabel("Oracle-normalised AUC")
    ax.set_title("Frontier quality vs. signal quality")
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.2, lw=0.5)

    # --- Right: heatmap (max oracle_norm_auc per cheap x dataset) ---
    oracle_rows = []
    for ds in DATASETS:
        fauc = load(ds, "frontier_auc")
        fauc = fauc.copy()
        fauc["cheap"]   = fauc["pair"].apply(lambda p: p.split("→")[0])
        fauc["dataset"] = ds
        oracle_rows.append(fauc[["cheap", "dataset", "oracle_norm_auc"]])
    oracle_all    = pd.concat(oracle_rows, ignore_index=True)
    oracle_pivot  = (
        oracle_all.dropna(subset=["oracle_norm_auc"])
        .groupby(["cheap", "dataset"])["oracle_norm_auc"]
        .max()
        .unstack("dataset")
        .reindex(columns=DATASETS)
    )
    oracle_pivot = oracle_pivot.loc[
        oracle_pivot.mean(axis=1).sort_values(ascending=False).index
    ]

    ax = axes[1]
    im = ax.imshow(oracle_pivot.values, aspect="auto",
                   cmap="YlGn", vmin=0.0, vmax=0.6)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS],
                       rotation=30, ha="right")
    ax.set_yticks(range(len(oracle_pivot)))
    ax.set_yticklabels(oracle_pivot.index)
    ax.set_title("Best oracle-norm. AUC\nper cheap model x dataset")
    for i in range(oracle_pivot.shape[0]):
        for j in range(oracle_pivot.shape[1]):
            v = oracle_pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="black" if v < 0.35 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Oracle-norm. AUC")

    fig.suptitle("UQ Signal Quality and Cascade Frontier Quality", fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "fig3_signal_quality.pdf"
    if save:
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
        print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Figure: shadow prices λ*(τ) along the frontier
# ---------------------------------------------------------------------------

def fig_shadow_prices(dataset: str = "triviaqa", save: bool = True):
    """
    Plot λ_P2(τ) = (m_H(τ)−m_L(τ)) / c_H vs. confidence score for
    three representative pairs, illustrating how the shadow price varies
    with threshold.
    """
    sp_df = load(dataset, "shadow_prices")

    # Pick three pairs: cheap→mid, cheap→expensive, mid→expensive
    pairs_ranked = (
        sp_df.dropna(subset=["lambda_P2"])
        .groupby("pair")["lambda_P2"]
        .apply(lambda g: g.std() / g.mean() if g.mean() > 0 else 0)
        .sort_values(ascending=False)
    )
    # Take up to 3 pairs with most relative variation
    selected_pairs = pairs_ranked.index[:3].tolist()

    fig, ax = plt.subplots(figsize=(4.5, 2.8))

    for pair in selected_pairs:
        sub = (sp_df[sp_df["pair"] == pair]
               .sort_values("bin_center")
               .dropna(subset=["lambda_P2"]))
        if sub.empty:
            continue
        x = sub["bin_center"].values
        # λ_P2 has units quality-improvement per dollar; express per µ$
        y = sub["lambda_P2"].values * 1e-6
        ax.plot(x, y, "o-", color=pair_color(pair), ms=3.5, lw=1.2,
                label=pair.replace("→", " → "))

    ax.set_xlabel("Confidence score $s_L$")
    ax.set_ylabel(r"$\hat{\lambda}^*_{P2}$ (quality per µ\$)")
    ax.set_title(f"Shadow Price $\\hat{{\\lambda}}^*_{{P2}}(\\tau)$ - {DATASET_LABELS[dataset]}")
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.2, lw=0.5)
    plt.tight_layout()

    path = FIGURES_DIR / f"fig_shadow_{dataset}.pdf"
    if save:
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
        print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Figure: k-model cascade vs 2-model envelope
# ---------------------------------------------------------------------------

def fig_kmodel_comparison(save: bool = True):
    """
    For each dataset, overlay the k-model Pareto front (from optuna_frontier.py
    saved results) on the 2-model global envelope.  Reads
    results/kmodel/{dataset}_pareto.parquet.
    """
    KMODEL_DIR = Path("results/kmodel")
    available  = [ds for ds in DATASETS
                  if (KMODEL_DIR / f"{ds}_pareto.parquet").exists()]
    if not available:
        print("No k-model results found; skipping fig_kmodel_comparison.")
        return None

    ncols = min(len(available), 2)
    nrows = (len(available) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6.75 if ncols == 2 else 3.5,
                                       3.0 * nrows))
    if len(available) == 1:
        axes = np.array([[axes]])
    axes = np.array(axes).reshape(nrows, ncols)

    for ax, ds in zip(axes.flatten(), available):
        env_df = load(ds, "envelope")
        pk     = pd.read_parquet(KMODEL_DIR / f"{ds}_pareto.parquet")

        env_c = env_df["cost"].values * 1e3   # dollars → $/1000 queries
        env_q = env_df["quality"].values
        ax.plot(env_c, env_q, color="steelblue", lw=1.8, ls="--",
                label="Best 2-model envelope")

        pk_c = pk["cost"].values * 1e3   # dollars → $/1000 queries
        pk_q = pk["quality"].values
        order = np.argsort(pk_c)
        ax.scatter(pk_c[order], pk_q[order], s=12, alpha=0.5,
                   color="tomato", zorder=3)
        # Monotone step envelope of k-model Pareto
        pk_q_mono = np.maximum.accumulate(pk_q[order])
        ax.step(pk_c[order], pk_q_mono, where="post", color="tomato",
                lw=1.6, label=r"$k$-model frontier ($k\leq6$)")

        ax.set_title(DATASET_LABELS[ds])
        ax.set_xlabel("Expected cost per 1000 queries (\\$)")
        ax.set_ylabel("Expected accuracy")
        ax.legend(fontsize=6.5)
        ax.grid(True, alpha=0.2, lw=0.5)
        ax.set_xlim(left=0)

    for ax in axes.flatten()[len(available):]:
        ax.set_visible(False)

    fig.suptitle(r"$k$-Model Cascade vs.\ Best 2-Model Envelope", fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "fig_kmodel_comparison.pdf"
    if save:
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
        print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Figure 4 - UQ signal informativeness
# ---------------------------------------------------------------------------

def fig4_uq_informativeness(save: bool = True):
    """
    Two heatmaps side-by-side:
      Left:  AUROC per cheap model x dataset (scorer = best per model per dataset)
      Right: Oracle-normalized AUC per pair x dataset (scorer = best)
    """
    # --- AUROC heatmap: best scorer per (model, dataset) ---
    auroc_rows = []
    for ds in DATASETS:
        adf = load(ds, "auroc")
        # Best AUROC per model
        adf_clean = adf.dropna(subset=["auroc"])
        idx = adf_clean.groupby("model")["auroc"].idxmax()
        best = adf_clean.loc[idx].reset_index(drop=True)
        best["dataset"] = ds
        auroc_rows.append(best)
    auroc_all = pd.concat(auroc_rows, ignore_index=True)

    # Oracle-normalized AUC: average over all pairs for each (cheap_model, dataset)
    oracle_rows = []
    for ds in DATASETS:
        fauc = load(ds, "frontier_auc")
        fauc["cheap"] = fauc["pair"].apply(lambda p: p.split("→")[0])
        fauc["dataset"] = ds
        oracle_rows.append(fauc[["cheap", "dataset", "oracle_norm_auc"]])
    oracle_all = pd.concat(oracle_rows, ignore_index=True)
    oracle_pivot = (
        oracle_all.dropna(subset=["oracle_norm_auc"])
        .groupby(["cheap", "dataset"])["oracle_norm_auc"]
        .mean()
        .unstack("dataset")
        .reindex(columns=DATASETS)
    )

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 3.2))

    # --- Left: AUROC heatmap ---
    ax = axes[0]
    auroc_pivot = (
        auroc_all.groupby(["model", "dataset"])["auroc"]
        .max()
        .unstack("dataset")
        .reindex(columns=DATASETS)
    )
    # Order models by mean AUROC descending
    auroc_pivot = auroc_pivot.loc[
        auroc_pivot.mean(axis=1).sort_values(ascending=False).index
    ]
    im = ax.imshow(auroc_pivot.values, aspect="auto",
                   cmap="YlOrRd", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], rotation=30, ha="right")
    ax.set_yticks(range(len(auroc_pivot)))
    ax.set_yticklabels(auroc_pivot.index)
    ax.set_title("Best AUROC per model x dataset")
    for i in range(auroc_pivot.shape[0]):
        for j in range(auroc_pivot.shape[1]):
            v = auroc_pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5, color="black" if v < 0.8 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8, label="AUROC")

    # --- Right: oracle-normalized AUC heatmap ---
    ax = axes[1]
    if not oracle_pivot.empty:
        oracle_pivot = oracle_pivot.loc[
            oracle_pivot.mean(axis=1).sort_values(ascending=False).index
        ]
        im2 = ax.imshow(oracle_pivot.values, aspect="auto",
                        cmap="YlGn", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(DATASETS)))
        ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS],
                           rotation=30, ha="right")
        ax.set_yticks(range(len(oracle_pivot)))
        ax.set_yticklabels(oracle_pivot.index)
        ax.set_title("Oracle-normalized AUC (mean over pairs)")
        for i in range(oracle_pivot.shape[0]):
            for j in range(oracle_pivot.shape[1]):
                v = oracle_pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color="black" if v < 0.6 else "white")
        fig.colorbar(im2, ax=ax, shrink=0.8, label="Oracle-norm. AUC")
    else:
        ax.set_visible(False)

    fig.suptitle("UQ Signal Informativeness", fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / "fig4_uq_informativeness.pdf"
    if save:
        fig.savefig(path, bbox_inches="tight")
        fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=200)
        print(f"Saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Appendix Figure A5 - Per-dataset all-pairs escalation benefit
# ---------------------------------------------------------------------------

def fig_escalation_appendix(dataset: str, save: bool = True):
    """
    One figure per dataset: grid of escalation benefit curves m_H(s)-m_L(s)
    for all pairs, read from results/escalation_benefit/{dataset}/*.npz.
    Each panel shows median curve, 10-90th pct band, and isotonic fit.
    Annotates with dom (expected-dominance fraction) and dec (decreasing fraction).
    """
    BAND_COLOR = "#4878CF"
    ISO_COLOR  = "#D65F5F"

    ds_dir = ESC_BEN_DIR / dataset
    npz_files = sorted(ds_dir.glob(f"{dataset}_*_mean_token_negentropy.npz"))
    if not npz_files:
        print(f"  No escalation benefit files for {dataset} - skipping.")
        return None

    n_pairs = len(npz_files)
    ncols   = 5
    nrows   = (n_pairs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.5, nrows * 2.0),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, fpath in enumerate(npz_files):
        ax  = axes_flat[i]
        npz = np.load(fpath, allow_pickle=True)

        s_grid       = npz["s_grid"]
        median_curve = npz["median_curve"]
        band_lo      = npz["band_lo"]
        band_hi      = npz["band_hi"]
        isotonic     = npz["isotonic_curve"]
        dom          = float(npz["dominance_frac"])
        dec          = float(npz["decreasing_frac"])

        valid = ~np.isnan(median_curve)
        if valid.sum() > 1:
            ax.fill_between(s_grid[valid], band_lo[valid], band_hi[valid],
                            alpha=0.25, color=BAND_COLOR)
            ax.plot(s_grid[valid], median_curve[valid],
                    color=BAND_COLOR, lw=1.2)
            ax.plot(s_grid[valid], isotonic[valid],
                    color=ISO_COLOR, lw=1.0, ls="--")

        ax.axhline(0, color="black", lw=0.6, ls=":")

        # Pair label: extract cheap→exp from filename
        stem  = fpath.stem  # {dataset}_{cheap}_{exp}_{scorer}
        parts = stem.split("_")
        # scorer is last token, dataset prefix varies - extract cheap→exp
        scorer_token = "mean_token_negentropy"
        inner = stem[len(dataset) + 1 : -(len(scorer_token) + 1)]
        ax.set_title(inner.replace("_", "\n→", 1), fontsize=5.5, pad=2)
        ax.text(0.97, 0.97, f"dom={dom:.2f}\ndec={dec:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=5)
        ax.tick_params(labelsize=5)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(n_pairs, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"Escalation benefit $m_H(s)-m_L(s)$ - {DATASET_LABELS.get(dataset, dataset)}",
                 fontsize=9)
    plt.tight_layout()
    path = FIGURES_DIR / f"figA5_escalation_{dataset}.pdf"
    if save:
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved {path}")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Table 1 - Summary comparison
# ---------------------------------------------------------------------------

def _cost_at_quality(curves: np.ndarray, cost_grid: np.ndarray,
                     q_target: float) -> float:
    """First cost on the median curve that reaches q_target; NaN if not reached."""
    med = np.nanmedian(curves, axis=0)
    idx = np.where(med >= q_target)[0]
    return float(cost_grid[idx[0]]) if len(idx) > 0 else float("nan")


def _gain_over_random(
    curves: np.ndarray,
    cost_grid: np.ndarray,
    c_low: float,
    q_low: float,
    c_high: float,
    q_high: float,
) -> float:
    """Normalized area above random routing for the median held-out frontier."""
    rect_area = (c_high - c_low) * (q_high - q_low)
    if rect_area <= 0:
        return float("nan")

    med = np.nanmedian(curves, axis=0)
    valid = (
        ~np.isnan(med)
        & (cost_grid >= c_low - 1e-12)
        & (cost_grid <= c_high + 1e-12)
    )
    if valid.sum() < 2:
        return float("nan")

    c = cost_grid[valid]
    q = med[valid]
    diagonal = q_low + (q_high - q_low) * (c - c_low) / (c_high - c_low)
    area_above = float(np.trapz(q - diagonal, c))
    return area_above / rect_area


def table1_summary(save: bool = True) -> str:
    """
    Table 1: method comparison across all datasets.

    Rows: Always-expensive, Cascade envelope, full fixed chain, optimal
    subsequence, diagnostic learned router.
    Per-dataset columns: normalized gain over random routing, CR@90% quality.
      Gain = normalized area above the random-routing baseline
      CR%  = cost reduction at 90% of ceiling quality vs. highest-accuracy single model

    Writes tables/table1_summary.tex; returns the LaTeX string.
    """
    tables_dir = Path("tables")
    tables_dir.mkdir(exist_ok=True)

    rows_data: dict[str, dict] = {m: {} for m in
                                  ["always_exp", "envelope", "fixed", "kcascade", "router"]}

    for ds in DATASETS:
        ds_dir = FIG2_DIR / ds
        cost_grid     = np.load(ds_dir / "cost_grid.npy")
        env_curves    = np.load(ds_dir / "envelope_curves.npy")
        fixed_curves  = np.load(ds_dir / "fixed_chain_curves.npy")
        kc_curves     = np.load(ds_dir / "kcascade_curves.npy")
        router_curves = np.load(ds_dir / "router_curves.npy")
        with open(ds_dir / "diagnostics.json")        as f: diag  = json.load(f)
        with open(ds_dir / "router_diagnostics.json") as f: rdiag = json.load(f)
        with open(ds_dir / "model_coords.json")       as f: mc    = json.load(f)

        low_model = min(mc.items(), key=lambda kv: kv[1][0])
        best_model = max(mc.items(), key=lambda kv: kv[1][1])
        c_low = float(low_model[1][0])
        q_low = float(low_model[1][1])
        c_exp = float(best_model[1][0])
        q_max = float(best_model[1][1])
        q_90  = 0.9 * q_max

        def cr90(curves):
            c = _cost_at_quality(curves, cost_grid, q_90)
            return (c_exp - c) / c_exp * 100.0 if not np.isnan(c) else float("nan")

        def gain(curves):
            return _gain_over_random(curves, cost_grid, c_low, q_low, c_exp, q_max)

        rows_data["always_exp"][ds] = {"gain": None,                "cr90": 0.0}
        rows_data["envelope"][ds]   = {"gain": gain(env_curves),    "cr90": cr90(env_curves)}
        rows_data["fixed"][ds]      = {"gain": gain(fixed_curves),  "cr90": cr90(fixed_curves)}
        rows_data["kcascade"][ds]   = {"gain": gain(kc_curves),     "cr90": cr90(kc_curves)}
        rows_data["router"][ds]     = {"gain": gain(router_curves), "cr90": cr90(router_curves)}

    method_labels = {
        "always_exp": r"Always-expensive",
        "envelope":   r"Optimal pair",
        "fixed":      r"Full fixed chain",
        "kcascade":   r"Optimal subsequence",
        "router":     r"Diagnostic learned router",
    }

    n_ds       = len(DATASETS)
    col_spec   = "l " + " ".join(["rr"] * n_ds)
    ds_headers = " & ".join(
        r"\multicolumn{2}{c}{" + DATASET_LABELS[ds] + "}" for ds in DATASETS)
    cmidrules  = " ".join(
        r"\cmidrule(lr){" + f"{2+2*i}-{3+2*i}" + "}" for i in range(n_ds))

    lines = [
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"& " + ds_headers + r" \\",
        cmidrules,
        r"Method & " + " & ".join([r"Gain & CR\,\%"] * n_ds) + r" \\",
        r"\midrule",
    ]
    best_gain = {}
    best_cr90 = {}
    for ds in DATASETS:
        gain_vals = [
            rows_data[m][ds]["gain"]
            for m in rows_data
            if rows_data[m][ds]["gain"] is not None
            and not np.isnan(rows_data[m][ds]["gain"])
        ]
        cr_vals = [
            rows_data[m][ds]["cr90"]
            for m in rows_data
            if not np.isnan(rows_data[m][ds]["cr90"])
        ]
        best_gain[ds] = max(gain_vals) if gain_vals else float("nan")
        best_cr90[ds] = max(cr_vals) if cr_vals else float("nan")

    def fmt_best(value: float, best: float, digits: int) -> str:
        s = f"{value:.{digits}f}"
        if not np.isnan(best) and np.isclose(value, best, rtol=0.0, atol=5e-4):
            return r"\textbf{" + s + "}"
        return s

    for m in ["always_exp", "envelope", "fixed", "kcascade", "router"]:
        cells = []
        for ds in DATASETS:
            d = rows_data[m][ds]
            gain_str  = "---" if d["gain"] is None or np.isnan(d["gain"]) else fmt_best(d["gain"], best_gain[ds], 3)
            cr_str    = "---" if np.isnan(d["cr90"]) else fmt_best(d["cr90"], best_cr90[ds], 1)
            cells.extend([gain_str, cr_str])
        lines.append(method_labels[m] + " & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)

    path = tables_dir / "table1_summary.tex"
    if save:
        path.write_text(tex)
        print(f"Saved {path}")
    return tex


# ---------------------------------------------------------------------------
# Appendix Table A2 - Signal decomposition
# ---------------------------------------------------------------------------

def tableA_signal_decomp(save: bool = True) -> str:
    """
    Appendix Table A2: signal decomposition.

    Three rows: UQ cascade (reference), embedding cascade, k-model router.
    Per-dataset columns: Δ vs. UQ cascade, mean AUROC of routing signal.

    Separates the structural advantage (can skip cheap model) from the
    signal quality advantage (embedding vs. post-generation UQ).

    Writes tables/tableA_signal_decomp.tex; returns the LaTeX string.
    """
    tables_dir = Path("tables")
    tables_dir.mkdir(exist_ok=True)

    rows_data: dict[str, dict] = {}

    for ds in DATASETS:
        ds_dir = FIG2_DIR / ds
        with open(ds_dir / "router_diagnostics.json") as f:
            rdiag = json.load(f)

        # UQ cascade AUROC: mean_token_negentropy AUROC averaged over pool models
        try:
            auroc_df = pd.read_parquet(RESULTS_DIR / ds / "auroc.parquet")
            uq_auroc = float(
                auroc_df[auroc_df["scorer"] == "mean_token_negentropy"]["auroc"].mean())
        except Exception:
            uq_auroc = float("nan")

        mean_emb_auroc = float(np.nanmean(list(rdiag["model_aurocs"].values())))

        for m in ["uq_cascade", "emb_cascade", "router"]:
            rows_data.setdefault(m, {})

        rows_data["uq_cascade"][ds]  = {"delta": 0.0,  "auroc": uq_auroc}
        rows_data["emb_cascade"][ds] = {
            "delta": rdiag["emb_cascade_vs_cascade"]["median_gap"],
            "auroc": mean_emb_auroc,
        }
        rows_data["router"][ds]      = {
            "delta": rdiag["router_vs_cascade"]["median_gap"],
            "auroc": mean_emb_auroc,
        }

    method_labels = {
        "uq_cascade":  r"UQ cascade (envelope)",
        "emb_cascade": r"Embedding cascade",
        "router":      r"Diagnostic learned router",
    }

    n_ds       = len(DATASETS)
    col_spec   = "l " + " ".join(["rr"] * n_ds)
    ds_headers = " & ".join(
        r"\multicolumn{2}{c}{" + DATASET_LABELS[ds] + "}" for ds in DATASETS)
    cmidrules  = " ".join(
        r"\cmidrule(lr){" + f"{2+2*i}-{3+2*i}" + "}" for i in range(n_ds))

    lines = [
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"& " + ds_headers + r" \\",
        cmidrules,
        r"Method & " + " & ".join([r"$\Delta$ & AUROC"] * n_ds) + r" \\",
        r"\midrule",
    ]
    for m in ["uq_cascade", "emb_cascade", "router"]:
        cells = []
        for ds in DATASETS:
            d = rows_data[m][ds]
            delta_str = f"${d['delta']:+.4f}$"
            auroc_str = f"{d['auroc']:.3f}" if not np.isnan(d["auroc"]) else "---"
            cells.extend([delta_str, auroc_str])
        lines.append(method_labels[m] + " & " + " & ".join(cells) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(lines)

    path = tables_dir / "tableA_signal_decomp.tex"
    if save:
        path.write_text(tex)
        print(f"Saved {path}")
    return tex


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "bundled"

    if target in ("all", "fig1"):
        print("Generating Figure 1...")
        fig1_pareto_frontiers()
    if target in ("all", "bundled", "fig2"):
        print("Generating Figure 2 (k-cascade vs envelope)...")
        fig2_kmodel_vs_envelope()
    if target in ("all", "bundled", "fig3"):
        print("Generating Figure 3 (escalation benefit)...")
        fig3_escalation_benefit()
    if target in ("all", "bundled", "figA5"):
        print("Generating Appendix Figure A5 (per-dataset escalation benefit)...")
        for ds in DATASETS:
            print(f"  {ds}...")
            fig_escalation_appendix(ds)
    if target in ("all", "table1"):
        print("Generating Table 1 (summary)...")
        table1_summary()
    if target in ("all", "tableA2"):
        print("Generating Appendix Table A2 (signal decomposition)...")
        tableA_signal_decomp()
    print("Done.")
