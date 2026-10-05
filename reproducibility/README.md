# Reproducibility code

Code supplement for **Is Escalation Worth It? A Decision-Theoretic Study of Depth in LLM Cascades**.
Scope follows the uncommented, recursive includes of the working manuscript
at packaging time. `EXPERIMENTS.md` maps every included empirical analysis to
its source and output. Manuscript paths in the inventory are normalized aliases
for the original source paths.

This is a **scripts-only supplement**. Recorded response parquets, embeddings,
large result caches, model weights, and API credentials are not included.
Full empirical reproduction requires the inputs described in `DATA.md`.
The existing small synthetic unit tests run without response data.

## Installation and checks

Use Python 3.12 on macOS or Linux and install `clang++` with C++17 support.
The existing code compiles the exact-search and FOC kernels on first use into
`/tmp`. No precompiled binaries are distributed.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_package.py
.venv/bin/python reproduce.py tests --run
```

The first `tiktoken` import can download its public `cl100k_base` vocabulary.
For offline execution, prepopulate a cache and set `TIKTOKEN_CACHE_DIR`.
`requirements.txt` records direct versions from the local analysis environment,
not a complete historical environment lock. See `VALIDATION.md` for checks
actually performed on this package.

## Inputs and execution

Run all commands from this folder. Given a separate copy of the original data
repository, first inspect its required files, then copy the inputs explicitly.
The helper does not copy old result caches or overwrite existing input files.

```sh
.venv/bin/python import_data.py /path/to/data_repository
.venv/bin/python import_data.py /path/to/data_repository --copy
.venv/bin/python reproduce.py all
```

`reproduce.py` prints the ordered commands by default. Add `--run` to execute.
It stops on the first failed subprocess. Each computation uses the copied
source, its original statistical settings, and fresh outputs within this folder.

```sh
.venv/bin/python reproduce.py all --run --workers 4
```

The complete suite is computationally substantial, especially exact depth-five
search for the seven original scores and both Diff-01 scores through S5.
No reliable full-suite runtime or peak-memory estimate
is available. The wrapper runs experiment families sequentially and defaults to
four workers where the underlying runner supports a worker setting. Some copied
runners have their own fixed concurrency. Do not run multiple instances against
the same output directory. A timed-out search is incomplete.

Stages can also run separately, in this order:

| Stage | Prerequisites |
|---|---|
| `embeddings` | Recorded responses. Optional generation requirements if embeddings are missing. |
| `exact` | Recorded responses. Computes S1 through S5 with mean negentropy. |
| `scorers` | Responses, embeddings, exact outputs. Computes the seven original scorers through S5. |
| `diff01` | Responses and embeddings. Both Diff-01 scores through S5. |
| `foc` | Exact outputs. Includes the sequential split-0 timing benchmark. |
| `synthetic` | Responses. Correlations, score construction, S1 through S4, diagnostics. |
| `draws` | Synthetic inputs. Five independent draws at AUROC 0.8 and 0.9. |
| `difficulty` | Synthetic and scorer outputs, response embeddings. |
| `costs` | Responses and synthetic scores. All nine price settings plus observed prices. |
| `diagnostics` | Responses and response embeddings. Includes all-pair benefit plots. |
| `learning` | Responses and exact outputs for the matched report. |
| `reports` | All prior stages. Regenerates the current main table and plotted results. |

An example is `python reproduce.py exact --run`. Direct script invocation permits
smaller pilot runs via the original `--splits` or dataset arguments where
available. Reporters generally require all 50 splits. A pilot is not a paper
replication and should run in a separate copy of the supplement.

## Source provenance and reporting

`SOURCE_MANIFEST.json` lists normalized source paths, original SHA-256,
packaged SHA-256, role, and packaging changes for every copied source and table.
Numerical implementations are copied from the existing source. A small set of
scorer-extension scripts had author-specific absolute roots replaced with paths
relative to this folder. Their recorded code hashes will therefore differ from
historical caches. Fresh runs preserve the original fingerprint checks.

The separate `experiments/livecodebench_8model_20260912` tree preserves the
manuscript's eight-model source mapping. It is required by report loaders.
Never substitute older seven-model LiveCodeBench results.

New packaging helpers are `reproduce.py`, `import_data.py`, `prepare_embeddings.py`,
`prepare_report_inputs.py`, `render_all_pairs.py`, `render_current_outputs.py`,
`diff01_report.py`, and `verify_package.py`.
They handle orchestration, input copying, validation, or presentation.
`render_current_outputs.py` assembles the nine-score main table and S1-S4
signal plot from the copied validated loaders and summary outputs because the
older report entry points produce superseded layouts. Figure styling can differ
from the submitted PDF. Table snapshots under `paper/tables` are reference values
and, for some original reporters, templates. Their presence is not evidence that
an experiment has been rerun. Numeric outputs for manually typeset appendix
tables are identified in `EXPERIMENTS.md`.

`MANUSCRIPT_INVENTORY.json` records every active TeX dependency, its checksum,
headings, and included table/figure references. Manuscript prose, bibliography,
compiled paper, excluded experiments, and private configuration are not bundled.
Some shared source modules still contain unused older experiment functions.
The wrapper invokes only the included analyses and their prerequisites.

The scorer-table reporter expects one combined result directory. The wrapper
uses `prepare_report_inputs.py` to copy only the eight-model LiveCodeBench
scorer and S1-S3 outputs into that layout, rejecting conflicting files.

See `DIFF01.md` for the two learned benefit scores and reporting limits.
