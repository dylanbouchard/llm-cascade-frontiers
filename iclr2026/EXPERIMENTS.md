# Included experiment map

Scope is the active ICLR source, including recursive appendix inputs. Section
names are used rather than unstable figure and appendix numbers.
All scripts below are copied unless identified as a packaging helper.

| Included analysis | Manuscript source | Computation and reporting | Numerical outputs |
|---|---|---|---|
| Exact S1-S5, mean token negentropy | `experiments.tex`, `exact_depth_appendix.tex` | `exact_heldout_depth.py`, `exact_heldout_four.py`, `exact_heldout_five.py`, C++ kernels, `iclr/exact_depth_report.py` | `results/exact_heldout_{depth,four,five}/`, `iclr/results/exact_depth/` |
| Seven-scorer S2-S5 and gain over S1 | `experiments.tex`, `voi_appendix.tex` | `scorer_depth_compute.py`, `experiments/scorer_s4_20260916/run.py`, `experiments/scorer_s5_20260916/run.py`, `scorer_depth_report.py`, helper `render_current_outputs.py` | `results/scorer_depth/`, each scorer-extension `results/`, main `table_exact_depth.tex` |
| Nominal depth, reach rates, budget thirds, feasibility, costs, search timing | `exact_depth_appendix.tex` | `iclr/exact_depth_report.py`, `audit_heldout_four.py`, `audit_heldout_five.py` | Included `table_exact_*` tables, per-split and budget-resolved summaries, `fig_exact_depth.pdf` |
| FOC-guided optimization S2-S5 and matched S3/S4 timing | `experiments.tex`, `foc_optimizer_appendix.tex` | `foc_pairwise_compare.py`, `foc_subsequence_compare.py`, `foc_subsequence_kernel.cpp`, `foc_subsequence_benchmark.py`, `foc_five_compare.py`, `foc_five_report.py`, `iclr/foc_optimizer_tables.py` | `results/foc_pairwise/`, `results/foc_subsequence/`, `results/foc_five/`, held-out and timing tables |
| Benefit-AUROC for seven scorers | `voi_appendix.tex` | `voi_compute.py`, response embedding support | `results/voi/{dataset}/results.parquet` and `summary.csv`. Paper averages pair-specific split medians over valid pairs. |
| Synthetic correctness AUROC 0.5-0.9, S1-S4 | `experiments.tex`, `synthetic_score_appendix.tex` | `confidence_score_correlations.py`, `simulate_correlated_confidence.py`, `simulated_signal_depth.py`, `simulated_signal_depth_four.py`, their report/audit scripts | `results/confidence_score_correlations/`, `results/simulated_confidence/`, `results/simulated_signal_depth{,_four}/` |
| Observed versus synthetic frontier plot | `experiments.tex` | `iclr/frontier_figure.py --main-only` | `iclr/figures/fig_exact_frontiers.pdf` |
| Synthetic signal-quality plot and matched AUROC diagnostics | `experiments.tex`, `synthetic_score_appendix.tex` | `simulated_signal_depth_report.py`, `simulated_signal_depth_four_report.py`, helper `render_current_outputs.py` | `table_synthetic_diagnostics.tex`, `fig_signal_quality.pdf`, input AUROC CSVs |
| Five additional independent synthetic draws, targets 0.8 and 0.9 | `synthetic_score_appendix.tex` | `synthetic_score_draws.py`, `synthetic_score_draws_tables.py` | `results/synthetic_score_draws/`, draw-mean and split-variability tables |
| Shared query difficulty, weights 0, 0.5, maximum feasible | `experiments.tex`, `shared_difficulty_appendix.tex` | `shared_difficulty_depth.py`, `shared_difficulty_report.py`, `iclr/shared_difficulty_tables.py` | `results/shared_difficulty_depth/`, main, gain and stage diagnostic tables |
| AUROC identity diagnostics for observed scorers | `shared_difficulty_appendix.tex` | `observed_stage_diagnostics.py`, `iclr/shared_difficulty_tables.py` | `results/shared_difficulty_depth/observed/`, `table_shared_difficulty_observed.tex` |
| Nine synthetic price spacings, observed and AUROC-0.9 scores | `experiments.tex`, `cost_counterfactual_appendix.tex` | `synthetic_cost_depth.py`, `synthetic_cost_full_confirmation.py`, `synthetic_cost_auroc_depth.py`, `synthetic_cost_depth_report.py` | `results/synthetic_cost_depth/`, `results/synthetic_cost_auroc09_depth/`, fitted parameters in dataset manifests, two cost-depth PDFs |
| Dataset grading and single-model operating points | `grading_appendix.tex`, `model_descriptives_appendix.tex` | `grade_*.py`, `cascade_core.py`, cost computation and source loaders | Recorded `correct` columns, `results/cost_variability.csv`, endpoint statistics from model loaders. `table_model_descriptives.tex` is a manually typeset snapshot. See DATA.md for grading limits. |
| Full-sample exact depth-three oracle | `oracle_depth_appendix.tex` | `oracle_depth_analysis.py` | `results/oracle_depth/`, per-dataset frontier and summary files. Values are manually typeset in the paper. |
| Escalation benefit, representative pairs and all 28 pairs | `escalation_appendix.tex` | `escalation_benefit.py`, `iclr/escalation_figure.py`, `figures.fig_escalation_appendix` via helper `render_all_pairs.py` | `results/escalation_benefit/`, representative PDF and `figures/figA5_escalation_{dataset}.pdf` |
| Realized frontier concavity | `concavity_check_appendix.tex` | `concavity_check.py`, `cascade_core.py` for representative-pair prerequisites | `results/concavity_check/{per_split,summary}.csv`, manually typeset `table_concavity_check.tex` |
| Token-price, cost-score and benefit-cost ratio diagnostics | `cost_variability_appendix.tex` | `cascade_core.PRICE_PER_TOKEN`, `cost_variability.py`, `ratio_condition_check.py` | `results/cost_score_correlations.csv`, `results/cost_score_correlation_summary.csv`, `results/ratio_condition/`, manually typeset ratio table |
| Fixed-test calibration learning curve | `cal_sensitivity_appendix.tex` | `calibration_learning_curve.py`, `calibration_learning_curve_report.py` | `results/calibration_learning_curve/`, learning-curve table and `figA7_cal_sensitivity.pdf` |

For exact depth, scorer depth, learning curves, oracle, benefit-AUROC, escalation,
and concavity, LiveCodeBench uses the separate eight-model source tree selected
by `manuscript_sources.py`. Synthetic, shared-difficulty, and synthetic-price
experiments use the current root loaders, which also retain all eight models.

The theory sections and recursively included two-model geometry and signal-value
proofs do not introduce additional empirical experiments.

## Excluded experiment entry points

The active master does not include the old router/deployment comparison,
heuristic NSGA-II versus random-search sensitivity, threshold-grid sensitivity,
old calibration-fraction sensitivity, fixed-chain comparisons, frozen-stage
removal, continuation-benefit learning, intermediate-stage-only cost discount,
or full-sample four/five-model pilot analyses. These are not reproduction stages.
The all-scorer frontier appendix and margin-frequency table are also not active
includes. Shared helper modules can contain functions from older experiments,
and the unbounded five-model kernel remains as a small-fixture test comparator.

## Diff-01 scores

Both new scorer implementations and their tests are included. `reproduce.py diff01` runs both scores through S5. `diff01_report.py` extends the main and depth tables. See `DIFF01.md` for output paths, aggregation, and the explicit benefit-AUROC cache selection.
