"""Run copied experiment scripts in dependency order. Default is a dry run."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
LCB = ROOT / 'experiments/livecodebench_8model_20260912'
DATASETS = ['mmlu', 'triviaqa', 'math_hard', 'simpleqa', 'livecodebench']
TESTS = ['test_exact_heldout_depth', 'test_exact_four_model', 'test_exact_heldout_four',
         'test_exact_five', 'test_exact_heldout_five', 'test_foc_pairwise_compare',
         'test_foc_subsequence_compare', 'test_calibration_learning_curve',
         'test_scorer_depth_compute', 'test_shared_difficulty_depth',
         'test_synthetic_score_draws', 'test_synthetic_cost_depth',
         'test_simulated_signal_depth_four', 'test_next_model_diff01',
         'test_next_model_embedding_diff01', 'test_next_model_deep_search',
         'test_next_model_deep_pruning']


def commands(stage, workers):
    jobs = []
    def add(script, *args, cwd=ROOT):
        jobs.append((cwd, [sys.executable, '-B', str(cwd / script), *map(str, args)]))
    if stage == 'tests':
        return [(ROOT, [sys.executable, '-B', '-m', 'unittest', '-v', *TESTS])]
    if stage == 'embeddings':
        add('prepare_embeddings.py')
    if stage == 'exact':
        for cwd, ds in [(ROOT, DATASETS[:-1]), (LCB, ['livecodebench'])]:
            add('exact_heldout_depth.py', *ds, cwd=cwd)
            add('exact_heldout_four.py', *ds, cwd=cwd)
            add('exact_heldout_five.py', *ds, '--workers', workers, cwd=cwd)
    if stage == 'scorers':
        for cwd, ds in [(ROOT, DATASETS[:-1]), (LCB, ['livecodebench'])]:
            add('scorer_depth_compute.py', *ds, '--workers', workers, cwd=cwd)
        for depth in (4, 5):
            add(f'experiments/scorer_s{depth}_20260916/run.py', '--workers', workers)
        add('experiments/scorer_s4_20260916/summarize.py')
        add('prepare_report_inputs.py')
        add('scorer_depth_report.py')
    if stage == 'foc':
        add('foc_pairwise_compare.py')
        add('foc_subsequence_compare.py', '--verify-chains')
        add('foc_subsequence_benchmark.py')
        add('foc_five_compare.py')
        add('foc_five_report.py')
        add('iclr/foc_optimizer_tables.py')
    if stage == 'diff01':
        add('next_model_diff01.py', '--workers', workers, '--out', 'results/next_model_diff01_full')
        add('next_model_diff01_deep.py', '--workers', workers, '--arms', 'diff01_ridge',
            '--max-depth', 4, '--out', 'results/next_model_diff01_s4_full')
        add('next_model_diff01_deep.py', '--workers', workers, '--arms', 'diff01_ridge',
            '--max-depth', 5, '--out', 'results/next_model_diff01_s5_full')
        add('next_model_embedding_diff01.py', '--workers', workers,
            '--out', 'results/next_model_embedding_diff01_s4_full')
        add('next_model_embedding_diff01_s5.py', '--workers', workers,
            '--out', 'results/next_model_embedding_diff01_s5_full')
    if stage == 'synthetic':
        add('confidence_score_correlations.py')
        add('simulate_correlated_confidence.py', *DATASETS)
        add('simulated_signal_depth.py', '--workers', workers)
        add('simulated_signal_depth_four.py', '--workers', workers)
        add('simulated_signal_depth_four_report.py')
        add('simulated_signal_depth_report.py')
    if stage == 'draws':
        add('synthetic_score_draws.py', '--workers', workers)
        add('synthetic_score_draws_tables.py')
    if stage == 'difficulty':
        add('shared_difficulty_depth.py', '--workers', workers)
        add('shared_difficulty_report.py')
        add('observed_stage_diagnostics.py')
        add('iclr/shared_difficulty_tables.py')
    if stage == 'costs':
        add('synthetic_cost_depth.py', '--prepare-only')
        add('synthetic_cost_full_confirmation.py', '--workers', workers)
        add('synthetic_cost_auroc_depth.py', '--workers', workers)
        for name in ['synthetic_cost_depth', 'synthetic_cost_auroc09_depth']:
            add('synthetic_cost_depth_report.py', '--out', f'results/{name}')
    if stage == 'diagnostics':
        # These older diagnostics keep their own pair-selection conventions.
        for cwd, ds in [(ROOT, DATASETS[:-1]), (LCB, ['livecodebench'])]:
            add('cascade_core.py', *(DATASETS if cwd == ROOT else ds), cwd=cwd)
            for script in ['oracle_depth_analysis.py', 'voi_compute.py',
                           'escalation_benefit.py', 'concavity_check.py']:
                add(script, *ds, cwd=cwd)
        add('cost_variability.py')
        add('ratio_condition_check.py')
        add('iclr/escalation_figure.py')
        add('render_all_pairs.py')
    if stage == 'learning':
        for cwd, ds in [(ROOT, DATASETS[:-1]), (LCB, ['livecodebench'])]:
            add('calibration_learning_curve.py', *ds, '--workers', workers, cwd=cwd)
        add('calibration_learning_curve_report.py')
    if stage == 'reports':
        add('iclr/exact_depth_report.py')
        add('iclr/frontier_figure.py', '--main-only')
        add('render_current_outputs.py')
    return jobs


def main():
    stages = ['embeddings', 'exact', 'scorers', 'diff01', 'foc', 'synthetic', 'draws',
              'difficulty', 'costs', 'diagnostics', 'learning', 'reports']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'tests', *stages])
    parser.add_argument('--run', action='store_true', help='Execute rather than print commands.')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('workers must be between 1 and 8')
    if args.run:
        for directory in ['iclr/figures', 'iclr/sections', 'iclr/tables', 'figures', 'results']:
            (ROOT/directory).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault('MPLBACKEND', 'Agg')
    env.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[key] = '1'
    for stage in stages if args.stage == 'all' else [args.stage]:
        for cwd, cmd in commands(stage, args.workers):
            print(f'[{cwd.relative_to(ROOT)}] {shlex.join(cmd)}', flush=True)
            if args.run:
                subprocess.run(cmd, cwd=cwd, env=env, check=True)

if __name__ == '__main__':
    main()
