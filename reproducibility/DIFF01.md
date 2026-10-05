# Diff-01 scorer extension

Two additional deferral scores use standardized ridge regression with penalty
1 and an unpenalized intercept. The target for each ordered transition is
`U_next - U_current`, taking values -1, 0, and +1. The negative prediction
is the confidence score. In a longer sequence, next means the immediately
following model, independent of later thresholds. All fitting and feature
standardization use calibration rows only.

`diff01_ridge` uses the current model's five UQ features.
`embedding_diff01_ridge` uses the same frozen 384-dimensional
`sentence-transformers/all-MiniLM-L6-v2` prompt-and-response embeddings as
the response correctness classifier. It changes the prediction target and
uses ridge regression. It is one scorer family with transition-specific fits.

## Reproduction

After importing recorded responses and preparing embeddings, run:

```sh
python reproduce.py diff01
python reproduce.py diff01 --run --workers 4
```

The stage runs UQ S2/S3, UQ S4/S5, embedding S2-S4, and embedding S5, in that
order. The UQ S2/S3 source also computes its original matched correctness-ridge
and negentropy controls. Only the Diff-01 arm is extended to S4/S5 by this stage.
All analyses use eight models, 50 matched splits, and 500 assigned budgets.
No LLM generation is performed.

The UQ S5 stage writes to `results/next_model_diff01_s5_full` and performs
exact S4/S5 search from the UQ S3 baseline. Both S5 scores are included in
the final manuscript. Reporting requires all 50 split files per dataset.
Partial runs are never averaged as full runs.
The scripts-only package contains no empirical caches.

The deep UQ and embedding S5 runners accept `--resume` with the same arguments
and output directory to reuse chain checkpoints. Do not launch duplicate
processes against one output directory. The initial UQ and embedding S4 runners
require a new output directory. Original result directories are never replaced.

## Reporting and validation

`render_current_outputs.py` adds both Diff-01 columns to the main table and
both rows to the third-stage accuracy/cost table. It uses trapezoidal integration
over normalized budget, matching the existing supplement reporter. The original
run summaries use an arithmetic mean over 500 positions, so last digits can
differ slightly. Split percentiles describe split variability.

The benefit-AUROC rows are computed separately from threshold search:

```sh
python diff01_benefit_auroc.py --pair-cache results/voi
```

This helper uses pair/split eligibility from the specified reference cache and
training/test rows from the Diff-01 manifests. It writes per-pair measurements
and the mean of pair-specific split medians to a new directory. The reference
cache identity matters: the original appended rows used root `results/voi`,
whereas the main supplemental diagnostic maps LiveCodeBench to the separate
eight-model tree. Changing that eligibility cache may change the reported
values. The helper therefore requires an explicit cache path. The existing
seven-scorer AUROC runner remains unchanged. A full empirical rerun of this
helper in the package has not been performed.

The shared-difficulty observed-policy diagnostic intentionally remains limited
to the seven confidence-based scores. The Diff-01 fits are not included in that
diagnostic. No overfitting analysis is added.

Four included test modules cover Diff-01 targets, ridge fitting, exact search,
and S4/S5 pruning. Run `python reproduce.py tests --run`.
