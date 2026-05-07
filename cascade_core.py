"""
cascade_core.py - 2-model cascade analysis for all datasets, scorers, and pairs.

All metrics are computed on a held-out test set; thresholds are calibrated on a
separate calibration set.

Outputs written to results/cascade_core/{dataset}/ as parquet files:
  frontiers.parquet       - (tau, E_C, E_U, sigma_C, p_esc) per pair x scorer
  envelope.parquet        - upper-left envelope + best_pair per grid point
  switching_points.parquet
  auroc.parquet           - AUROC per cheap model x scorer
  escalation_gains.parquet - m_H(tau)-m_L(tau) per bin, per pair x scorer
  isotonic_tests.parquet  - monotonicity test result per pair
  shadow_prices.parquet   - empirical lambda* per bin, per pair
  risk_frontiers.parquet  - risk-adjusted effective cost for delta=0.05, 0.10
  frontier_auc.parquet    - rectangle-AUC and oracle-normalized AUC per pair x scorer

Usage:
    python3 cascade_core.py [dataset ...]
    python3 cascade_core.py mmlu triviaqa math_hard simpleqa
"""

import sys
import numpy as np
import pandas as pd
import tiktoken
from pathlib import Path
from itertools import permutations
from scipy.interpolate import interp1d
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = Path("data/output_data")
RESULTS_DIR = Path("results/cascade_core")

SCORERS = [
    "sequence_probability",
    "min_probability",
    "min_token_negentropy",
    "mean_token_negentropy",
    "probability_margin",
]

# Input and output price per token ($/token)
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

# Ascending cost order for the 6-model cascade pool
MODEL_COST_ORDER = sorted(PRICE_PER_TOKEN, key=lambda m: sum(PRICE_PER_TOKEN[m]))

# Models excluded per dataset (gpt-oss-20b dominates all others on livecodebench).
DATASET_EXCLUSIONS: dict[str, list[str]] = {
    "livecodebench": ["gpt-oss-20b"],
}

_enc = tiktoken.get_encoding("cl100k_base")

N_THRESH  = 200   # threshold sweep resolution
N_BINS    = 20    # bins for m_H - m_L estimation
N_BOOT    = 1000  # bootstrap resamples
N_GRID    = 2000  # cost-grid resolution for envelope
CAL_FRAC  = 0.5
SEED      = 42

# ---------------------------------------------------------------------------
# Cost computation
# ---------------------------------------------------------------------------

def compute_costs(df: pd.DataFrame, model: str) -> np.ndarray:
    """Per-query cost in dollars: input tokens (tiktoken) + output tokens (logprob length)."""
    p_in, p_out = PRICE_PER_TOKEN[model]
    n_in  = df["prompt"].apply(lambda p: len(_enc.encode(p))).to_numpy()
    n_out = df["logprob"].apply(len).to_numpy()
    return n_in * p_in + n_out * p_out

# ---------------------------------------------------------------------------
# Data loading and splitting
# ---------------------------------------------------------------------------

def load_all_models(dataset: str, n_rows: int = 2000) -> dict[str, pd.DataFrame]:
    """Load all available model parquets for a dataset, verify row alignment."""
    excluded = DATASET_EXCLUSIONS.get(dataset, [])
    data = {}
    for model in MODEL_COST_ORDER:
        if model in excluded:
            continue
        path = DATA_DIR / f"{dataset}-{model}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if n_rows and n_rows < len(df):
            df = df.iloc[:n_rows].reset_index(drop=True)
        data[model] = df

    models = list(data.keys())
    ref = data[models[0]]["prompt"].values
    for m in models[1:]:
        assert np.array_equal(ref, data[m]["prompt"].values), \
            f"Row mismatch: {models[0]} vs {m}"
    return data


def cal_test_split(
    n: int, correct: np.ndarray, cal_frac: float = CAL_FRAC, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split on binary correctness label. Returns (cal_idx, test_idx)."""
    rng   = np.random.default_rng(seed)
    idx_0 = np.where(correct == 0)[0]
    idx_1 = np.where(correct == 1)[0]
    n0    = int(len(idx_0) * cal_frac)
    n1    = int(len(idx_1) * cal_frac)
    cal_idx = np.sort(np.concatenate([
        rng.choice(idx_0, n0, replace=False),
        rng.choice(idx_1, n1, replace=False),
    ]))
    test_idx = np.setdiff1d(np.arange(n), cal_idx)
    return cal_idx, test_idx

# ---------------------------------------------------------------------------
# 2-model cascade frontier
# ---------------------------------------------------------------------------

def compute_frontier(
    cheap_test: pd.DataFrame,
    exp_test: pd.DataFrame,
    costs_L: np.ndarray,
    costs_H: np.ndarray,
    scorer: str,
    tau_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Sweep tau_grid on the test set and compute (E_C, E_U, sigma_C, p_esc) at each point.
    Per-query cost uses actual token counts (variable cost); sigma_C is the empirical
    std of C(x;tau) = cost_L(x) + 1[s_L < tau]*cost_H(x).
    Drops points where E_C > mean(costs_H) (dominated by the high-cost single-model endpoint).
    Returns DataFrame sorted by E_C.
    """
    valid = (cheap_test[scorer].notna() & cheap_test["correct"].notna()
             & exp_test["correct"].notna()).values
    s       = cheap_test[scorer].values[valid]
    u_L     = cheap_test["correct"].values[valid].astype(float)
    u_H     = exp_test["correct"].values[valid].astype(float)
    costs_L = costs_L[valid]
    costs_H = costs_H[valid]
    c_H_mean = costs_H.mean()

    rows = []
    for tau in tau_grid:
        esc = s < tau
        p   = esc.mean()
        per_query_c = costs_L + esc.astype(float) * costs_H
        E_C     = per_query_c.mean()
        sigma_C = per_query_c.std()
        E_U     = np.where(esc, u_H, u_L).mean()
        if E_C <= c_H_mean:
            rows.append({"tau": tau, "E_C": E_C, "E_U": E_U,
                         "sigma_C": sigma_C, "p_esc": float(p)})

    if rows:
        df = pd.DataFrame(rows).sort_values("E_C").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["tau", "E_C", "E_U", "sigma_C", "p_esc"])

    # Prepend low-cost and append high-cost single-model endpoints
    always_cheap = {"tau": float(tau_grid[0]),
                    "E_C": float(costs_L.mean()), "E_U": float(u_L.mean()),
                    "sigma_C": 0.0, "p_esc": 0.0}
    always_exp   = {"tau": float(tau_grid[-1]),
                    "E_C": float(c_H_mean), "E_U": float(u_H.mean()),
                    "sigma_C": 0.0, "p_esc": 1.0}
    return pd.concat(
        [pd.DataFrame([always_cheap]), df, pd.DataFrame([always_exp])],
        ignore_index=True,
    )


def random_routing_baseline(
    c_L: float, c_H: float, u_L: float, u_H: float, n: int = 100,
) -> pd.DataFrame:
    """Straight line from (c_L, u_L) to (c_H, u_H): no signal, random escalation."""
    ps = np.linspace(0.0, 1.0, n)
    return pd.DataFrame({
        "E_C": c_L + ps * (c_H - c_L),
        "E_U": u_L + ps * (u_H - u_L),
    })

# ---------------------------------------------------------------------------
# Global envelope and switching points
# ---------------------------------------------------------------------------

def compute_envelope(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    n_grid: int = N_GRID,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Upper-left envelope over all pair curves.
    Input: dict of pair_name -> (cost_array, quality_array).
    Returns: (cost_grid, envelope_quality, best_pair_per_grid_point).
    """
    pair_names = list(curves.keys())
    min_c = min(c.min() for c, _ in curves.values())
    max_c = max(c.max() for c, _ in curves.values())
    grid  = np.linspace(min_c, max_c, n_grid)

    interped = []
    for name in pair_names:
        c, q = curves[name]
        order = np.argsort(c)
        f = interp1d(c[order], q[order], bounds_error=False,
                     fill_value=(np.nan, np.nan))
        interped.append(f(grid))

    stacked = np.vstack(interped)  # (n_pairs, n_grid)
    with np.errstate(all="ignore"):
        best_idx = np.where(
            np.all(np.isnan(stacked), axis=0),
            -1,
            np.nanargmax(stacked, axis=0),
        )
    envelope = np.nanmax(stacked, axis=0)
    # Enforce monotonicity (quality non-decreasing in cost)
    envelope = np.maximum.accumulate(
        np.where(np.isnan(envelope), -np.inf, envelope)
    )
    envelope = np.where(envelope == -np.inf, np.nan, envelope)
    # Truncate at first point reaching maximum quality
    q_max     = np.nanmax(envelope)
    first_max = np.where(~np.isnan(envelope) & (envelope >= q_max - 1e-12))[0]
    if len(first_max):
        i = first_max[0] + 1
        grid, envelope, best_idx = grid[:i], envelope[:i], best_idx[:i]

    best_pairs = [pair_names[i] if i >= 0 else None for i in best_idx]
    return grid, envelope, best_pairs


def find_switching_points(
    cost_grid: np.ndarray,
    best_pairs: list[str],
) -> list[dict]:
    """Transitions in the optimal pair along the global envelope."""
    switches = []
    for i in range(1, len(best_pairs)):
        if best_pairs[i] != best_pairs[i - 1] and best_pairs[i] is not None:
            switches.append({
                "cost":      float(cost_grid[i]),
                "from_pair": best_pairs[i - 1],
                "to_pair":   best_pairs[i],
            })
    return switches

# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_frontier(
    cheap_test: pd.DataFrame,
    exp_test: pd.DataFrame,
    costs_L: np.ndarray,
    costs_H: np.ndarray,
    scorer: str,
    tau_grid: np.ndarray,
    n_boot: int = N_BOOT,
    n_grid: int = 500,
    seed: int = SEED,
) -> dict:
    """
    Bootstrap 95% CI on the cascade curve by resampling the test set (with replacement).
    Returns dict: cost_grid, lower (2.5%), median (50%), upper (97.5%).
    """
    rng = np.random.default_rng(seed)
    n   = len(cheap_test)
    c_L = float(costs_L.mean())
    c_H = float(costs_H.mean())
    cost_grid = np.linspace(c_L, c_H, n_grid)

    boot_quals = np.full((n_boot, n_grid), np.nan)
    for b in range(n_boot):
        idx   = rng.integers(0, n, size=n)
        curve = compute_frontier(
            cheap_test.iloc[idx].reset_index(drop=True),
            exp_test.iloc[idx].reset_index(drop=True),
            costs_L[idx], costs_H[idx],
            scorer, tau_grid,
        )
        c_arr = curve["E_C"].values
        q_arr = curve["E_U"].values
        order = np.argsort(c_arr)
        f = interp1d(c_arr[order], q_arr[order],
                     bounds_error=False, fill_value=(np.nan, np.nan))
        boot_quals[b] = f(cost_grid)

    return {
        "cost_grid": cost_grid,
        "lower":     np.nanpercentile(boot_quals,  2.5, axis=0),
        "median":    np.nanpercentile(boot_quals, 50.0, axis=0),
        "upper":     np.nanpercentile(boot_quals, 97.5, axis=0),
    }

# ---------------------------------------------------------------------------
# Condition verification
# ---------------------------------------------------------------------------

def compute_escalation_gain(
    cheap_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    scorer: str,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """
    Estimate m_H(tau) - m_L(tau) by binning s_L into equal-frequency bins.
    Returns DataFrame: bin_center, m_L, m_H, gain, gain_se, n.
    """
    valid = (cheap_df[scorer].notna() & cheap_df["correct"].notna()
             & exp_df["correct"].notna()).values
    s   = cheap_df[scorer].values[valid]
    u_L = cheap_df["correct"].values[valid].astype(float)
    u_H = exp_df["correct"].values[valid].astype(float)

    if len(s) < n_bins:
        return pd.DataFrame(columns=["bin_center", "m_L", "m_H", "gain", "gain_se", "n"])

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges     = np.percentile(s, quantiles)
    edges[-1] += 1e-9

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (s >= lo) & (s < hi)
        if mask.sum() < 5:
            continue
        m_L = u_L[mask].mean()
        m_H = u_H[mask].mean()
        n   = int(mask.sum())
        # SE via delta method for difference of two independent proportions
        se = np.sqrt(m_H * (1 - m_H) / n + m_L * (1 - m_L) / n)
        rows.append({
            "bin_center": float((lo + hi) / 2),
            "m_L": float(m_L), "m_H": float(m_H),
            "gain": float(m_H - m_L), "gain_se": float(se), "n": n,
        })
    return pd.DataFrame(rows)


def test_monotone_nonincreasing(gain_df: pd.DataFrame) -> dict:
    """
    Fit non-increasing isotonic regression to gain vs bin_center.
    max_reversal: largest positive deviation of raw gain from isotonic fit.
    passes: True if max_reversal < 0.02.
    Returns a degenerate result if gain_df has fewer than 3 rows.
    """
    if len(gain_df) < 3:
        return {
            "bin_centers": np.array([]), "raw_gain": np.array([]),
            "isotonic_fit": np.array([]), "max_reversal": np.nan, "passes": False,
        }
    x     = gain_df["bin_center"].values
    y     = gain_df["gain"].values
    y_fit = IsotonicRegression(increasing=False).fit_transform(x, y)
    reversals = np.maximum(0.0, y - y_fit)
    return {
        "bin_centers":  x,
        "raw_gain":     y,
        "isotonic_fit": y_fit,
        "max_reversal": float(reversals.max()),
        "passes":       bool(reversals.max() < 0.02),
    }

# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------

def compute_auroc(df: pd.DataFrame, scorer: str) -> float:
    mask = df[scorer].notna() & df["correct"].notna()
    if mask.sum() < 10 or df.loc[mask, "correct"].nunique() < 2:
        return np.nan
    return float(roc_auc_score(df.loc[mask, "correct"].values,
                                df.loc[mask, scorer].values))


def compute_all_aurocs(df: pd.DataFrame) -> dict[str, float]:
    return {s: compute_auroc(df, s) for s in SCORERS if s in df.columns}

# ---------------------------------------------------------------------------
# Shadow prices
# ---------------------------------------------------------------------------

def compute_shadow_prices(
    cheap_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    scorer: str,
    c_H_mean: float,
    n_bins: int = N_BINS,
) -> pd.DataFrame:
    """
    Empirical lambda*_P2 = (m_H(tau) - m_L(tau)) / c_H_mean per bin.
    Equals the slope of the frontier map C -> U*(C) at each operating point.
    lambda*_P1 = 1 / lambda*_P2 (reciprocal shadow price).
    Returns empty DataFrame if insufficient data for binning.
    """
    df = compute_escalation_gain(cheap_df, exp_df, scorer, n_bins)
    if df.empty:
        return df
    df["lambda_P2"] = df["gain"] / c_H_mean
    df["lambda_P1"] = np.where(df["gain"] > 1e-9, c_H_mean / df["gain"], np.nan)
    return df

# ---------------------------------------------------------------------------
# Frontier AUC metrics
# ---------------------------------------------------------------------------

def compute_frontier_auc(
    curve: pd.DataFrame,
    c_L: float, c_H: float,
    u_L: float, u_H: float,
    oracle_curve: pd.DataFrame | None = None,
) -> dict:
    """
    Rectangle-AUC: area between cascade curve and random-routing baseline,
    normalized by (c_H - c_L) * (u_H - u_L).

    Oracle-normalized AUC: ratio of cascade AUC to oracle AUC, where the oracle
    escalates exactly when cheap is wrong and expensive is right.
    oracle_curve should be the frontier computed with the oracle scorer.
    """
    baseline = random_routing_baseline(c_L, c_H, u_L, u_H, n=500)
    f_base   = interp1d(baseline["E_C"], baseline["E_U"],
                        bounds_error=False, fill_value=(u_L, u_H))

    c_arr = curve["E_C"].values
    q_arr = curve["E_U"].values
    order = np.argsort(c_arr)
    c_arr, q_arr = c_arr[order], q_arr[order]

    # Area above baseline under the cascade curve (trapezoid rule)
    base_at_c = f_base(c_arr)
    auc_above = np.trapz(np.maximum(q_arr - base_at_c, 0), c_arr)
    normalizer = (c_H - c_L) * (u_H - u_L)
    rect_auc   = float(auc_above / normalizer) if normalizer > 0 else np.nan

    oracle_norm_auc = np.nan
    if oracle_curve is not None:
        oc   = oracle_curve["E_C"].values
        oq   = oracle_curve["E_U"].values
        oo   = np.argsort(oc)
        oc, oq = oc[oo], oq[oo]
        oracle_auc_above = np.trapz(np.maximum(oq - f_base(oc), 0), oc)
        if oracle_auc_above > 1e-12:
            oracle_norm_auc = float(auc_above / oracle_auc_above)

    return {"rect_auc": rect_auc, "oracle_norm_auc": oracle_norm_auc}


def compute_oracle_frontier(
    cheap_test: pd.DataFrame,
    exp_test: pd.DataFrame,
    costs_L: np.ndarray,
    costs_H: np.ndarray,
) -> pd.DataFrame:
    """
    Oracle cascade: escalate exactly when cheap is wrong AND expensive is right.
    Best achievable frontier for any binary confidence signal.

    Escalation fires when score < tau. So queries that SHOULD escalate need
    score = 0; queries that should NOT escalate need score = 1. The oracle
    score is therefore the INVERSE of the escalation indicator:
        oracle_score = 1 - 1[U_H=1 AND U_L=0]
    A single threshold tau=0.5 separates the two groups exactly.
    """
    should_escalate = (
        (exp_test["correct"].values == 1) &
        (cheap_test["correct"].values == 0)
    ).astype(float)
    oracle_score = 1.0 - should_escalate   # 0 → escalate, 1 → don't

    cheap_aug = cheap_test.copy()
    cheap_aug["_oracle"] = oracle_score

    # Binary signal only needs a single interior threshold
    tau_grid = np.array([0.5])
    return compute_frontier(cheap_aug, exp_test, costs_L, costs_H,
                             "_oracle", tau_grid)

# ---------------------------------------------------------------------------
# Risk-adjusted frontier
# ---------------------------------------------------------------------------

def compute_risk_frontier(
    cheap_test: pd.DataFrame,
    exp_test: pd.DataFrame,
    costs_L: np.ndarray,
    costs_H: np.ndarray,
    scorer: str,
    tau_grid: np.ndarray,
    delta: float,
) -> pd.DataFrame:
    """
    Risk-adjusted effective cost = E[C] + z_delta * sigma_C.
    sigma_C is the empirical std of C(x;tau) from actual per-query costs.
    """
    curve = compute_frontier(cheap_test, exp_test, costs_L, costs_H,
                              scorer, tau_grid)
    z = float(norm.ppf(1.0 - delta))
    curve[f"eff_cost_{delta}"] = curve["E_C"] + z * curve["sigma_C"]
    return curve

# ---------------------------------------------------------------------------
# Main: run all analyses for one dataset
# ---------------------------------------------------------------------------

def run_dataset(dataset: str, n_rows: int = 2000):
    print(f"\n{'='*60}\n{dataset.upper()}\n{'='*60}")
    out_dir = RESULTS_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all 6 model parquets
    all_dfs = load_all_models(dataset, n_rows=n_rows)
    all_costs = {m: compute_costs(df, m) for m, df in all_dfs.items()}

    # Stratified cal/test split (same indices for all models)
    ref_model  = MODEL_COST_ORDER[0] if MODEL_COST_ORDER[0] in all_dfs \
                 else list(all_dfs.keys())[0]
    cal_idx, test_idx = cal_test_split(
        len(all_dfs[ref_model]),
        all_dfs[ref_model]["correct"].values,
    )
    cal   = {m: df.iloc[cal_idx].reset_index(drop=True)  for m, df in all_dfs.items()}
    test  = {m: df.iloc[test_idx].reset_index(drop=True) for m, df in all_dfs.items()}
    c_cal = {m: all_costs[m][cal_idx]  for m in all_dfs}
    c_tst = {m: all_costs[m][test_idx] for m in all_dfs}

    mean_cost = {m: all_costs[m].mean() for m in all_dfs}
    mean_acc  = {m: all_dfs[m]["correct"].mean() for m in all_dfs}

    print(f"Split: {len(cal_idx)} cal / {len(test_idx)} test")
    print("\nModel summary:")
    for m in MODEL_COST_ORDER:
        if m in mean_cost:
            print(f"  {m:20s}  cost={mean_cost[m]*1e6:.2f} µ$  acc={mean_acc[m]:.3f}")

    # Valid pairs: cheaper expected cost AND higher accuracy
    valid_pairs = sorted(
        {(cheap, exp)
         for cheap, exp in permutations(list(all_dfs.keys()), 2)
         if mean_cost[cheap] < mean_cost[exp] and mean_acc[exp] > mean_acc[cheap]},
        key=lambda p: (mean_cost[p[0]], mean_cost[p[1]]),
    )
    print(f"\n{len(valid_pairs)} valid pairs")

    # ------------------------------------------------------------------
    # AUROC for all scorers on all cheap models
    # ------------------------------------------------------------------
    auroc_rows = []
    for model in all_dfs:
        for s in SCORERS:
            if s in all_dfs[model].columns:
                auroc_rows.append({
                    "model": model, "scorer": s,
                    "auroc": compute_auroc(all_dfs[model], s),
                })
    auroc_df = pd.DataFrame(auroc_rows)
    auroc_df.to_parquet(out_dir / "auroc.parquet", index=False)

    # Fixed scorer for all cascade frontier computations
    CASCADE_SCORER = "mean_token_negentropy"
    best_scorer = {m: CASCADE_SCORER for m in all_dfs}
    print(f"\nUsing fixed cascade scorer: {CASCADE_SCORER}")

    # ------------------------------------------------------------------
    # Frontier computation for each pair x best scorer
    # ------------------------------------------------------------------
    frontier_rows, gain_rows, iso_rows, shadow_rows = [], [], [], []
    risk_rows, auc_rows = [], []
    curve_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for cheap, exp in valid_pairs:
        pair = f"{cheap}→{exp}"
        scorer = best_scorer.get(cheap, "mean_token_negentropy")

        # tau_grid from calibration set score range
        s_cal    = cal[cheap][scorer].values
        tau_grid = np.linspace(s_cal.min(), s_cal.max(), N_THRESH)

        # --- Pareto frontier ---
        curve = compute_frontier(
            test[cheap], test[exp], c_tst[cheap], c_tst[exp], scorer, tau_grid,
        )
        curve["pair"] = pair
        curve["scorer"] = scorer
        frontier_rows.append(curve)
        curve_arrays[pair] = (curve["E_C"].values, curve["E_U"].values)

        # --- Condition verification ---
        gain_df = compute_escalation_gain(test[cheap], test[exp], scorer)
        gain_df["pair"] = pair
        gain_rows.append(gain_df)

        iso = test_monotone_nonincreasing(gain_df)
        iso_rows.append({
            "pair": pair, "max_reversal": iso["max_reversal"],
            "passes": iso["passes"],
        })

        # --- Shadow prices ---
        sp = compute_shadow_prices(
            test[cheap], test[exp], scorer, c_tst[exp].mean()
        )
        sp["pair"] = pair
        shadow_rows.append(sp)

        # --- Risk-adjusted frontiers ---
        for delta in [0.05, 0.10]:
            rf = compute_risk_frontier(
                test[cheap], test[exp], c_tst[cheap], c_tst[exp],
                scorer, tau_grid, delta,
            )
            rf["pair"]  = pair
            rf["delta"] = delta
            risk_rows.append(rf)

        # --- Oracle frontier and AUC metrics ---
        oracle_curve = compute_oracle_frontier(
            test[cheap], test[exp], c_tst[cheap], c_tst[exp],
        )
        c_L = float(c_tst[cheap].mean())
        c_H = float(c_tst[exp].mean())
        u_L = float(test[cheap]["correct"].mean())
        u_H = float(test[exp]["correct"].mean())
        auc_metrics = compute_frontier_auc(curve, c_L, c_H, u_L, u_H, oracle_curve)
        auc_rows.append({"pair": pair, "scorer": scorer, **auc_metrics})

    frontier_df = pd.concat(frontier_rows, ignore_index=True)
    gain_df_all = pd.concat(gain_rows,     ignore_index=True)
    iso_df      = pd.DataFrame(iso_rows)
    shadow_df   = pd.concat(shadow_rows,   ignore_index=True)
    risk_df     = pd.concat(risk_rows,     ignore_index=True)
    auc_df      = pd.DataFrame(auc_rows)

    frontier_df.to_parquet(out_dir / "frontiers.parquet",         index=False)
    gain_df_all.to_parquet(out_dir / "escalation_gains.parquet",  index=False)
    iso_df.to_parquet(     out_dir / "isotonic_tests.parquet",    index=False)
    shadow_df.to_parquet(  out_dir / "shadow_prices.parquet",     index=False)
    risk_df.to_parquet(    out_dir / "risk_frontiers.parquet",    index=False)
    auc_df.to_parquet(     out_dir / "frontier_auc.parquet",      index=False)

    # ------------------------------------------------------------------
    # Global envelope and switching points
    # ------------------------------------------------------------------
    cost_grid, envelope, best_pairs = compute_envelope(curve_arrays)
    switches = find_switching_points(cost_grid, best_pairs)

    env_df = pd.DataFrame({
        "cost": cost_grid, "quality": envelope, "best_pair": best_pairs,
    })
    switch_df = pd.DataFrame(switches) if switches else \
        pd.DataFrame(columns=["cost", "from_pair", "to_pair"])

    env_df.to_parquet(    out_dir / "envelope.parquet",         index=False)
    switch_df.to_parquet( out_dir / "switching_points.parquet", index=False)

    # ------------------------------------------------------------------
    # Summary printout
    # ------------------------------------------------------------------
    print(f"\nIsotonic tests (Assumption 2 - decreasing escalation benefit):")
    print(iso_df.to_string(index=False))

    print(f"\nSwitching points ({len(switches)}):")
    print(switch_df.to_string(index=False))

    print(f"\nFrontier AUC:")
    print(auc_df.to_string(index=False))

    print(f"\nResults written to {out_dir}/")
    return {
        "frontiers":  frontier_df,
        "envelope":   env_df,
        "switches":   switch_df,
        "auroc":      auroc_df,
        "gains":      gain_df_all,
        "isotonic":   iso_df,
        "shadow":     shadow_df,
        "risk":       risk_df,
        "auc":        auc_df,
    }


if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else [
        "mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"
    ]
    for ds in datasets:
        run_dataset(ds)
