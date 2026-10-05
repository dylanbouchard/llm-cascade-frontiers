# Packaging validation

The following checks were completed on 2026-09-23 with Python 3.12 and the direct
analysis-library versions listed in `requirements.txt`.

- Verified checksums for all 136 copied source and table files against the
  packaging manifest, including the documented path substitutions and cleared
  generation notebook outputs.
- Parsed all 113 packaged Python files and resolved all 57 subprocess entry
  points in the complete dry-run plan.
- Ran 34 existing data-free unit tests successfully. Coverage includes exact
  S1-S5 selection against brute force, ties and endpoint policies, bounded
  pruning, calibration/test separation, FOC candidate and local search checks,
  nested calibration samples, scorer fitting, synthetic-score draws,
  shared-difficulty diagnostics, price reconstruction, and S4 cache replay.
- Copied the package to a separate temporary directory and reran the same 34
  tests successfully to check independence from the original checkout location.
- Confirmed that all 40 required response parquets exist in the original data
  repository and have the columns documented in `DATA.md`. Data were inspected
  read-only and were not added to this scripts-only package.
- Checked the packaged source for author-specific home-directory paths and
  common credential-literal patterns. Notebook outputs and execution metadata
  were removed. No environment files or compiled binaries were copied.

The public tokenizer vocabulary was downloaded to a temporary cache for these
tests. The C++ kernels were exercised through the existing tests. No new LLM
calls, paid grading calls, or full empirical reruns were performed.

Full 50-split runs, every reporting entry point against freshly generated
results, and installation in a newly resolved environment have **not** been
validated end to end. Static path checks and synthetic tests do not establish
reproduction of the paper's numerical results. The price-protocol and data
provenance limits identified during packaging are recorded in `DATA.md`.

To repeat the checks, run `python verify_package.py` and
`python reproduce.py tests --run`. The archival historical-source test requires
external seven-model and eight-model caches and is intentionally not in the
standard test command.

## Diff-01 package update (2026-09-25)

Added both Diff-01 implementations, exact S4/S5 support, and reporting helpers.
The complete data-free suite passed 44 tests on 2026-09-25. Static verification
passed for 147 copied-file hashes, 125 Python files, and 62 stage commands.
No full empirical rerun was performed. UQ S5 remains pending and is an
optional stage. The AUROC helper requires explicit reference-cache selection
as documented in DIFF01.md.

## Final manuscript update (2026-09-30)

The working manuscript now includes completed UQ Diff-01 S5 results. The
reproduction plan includes that stage, and the public source and table
snapshots were refreshed from the final working manuscript. The historical
validation entries above describe earlier package versions. No full empirical
rerun was performed for this update.
