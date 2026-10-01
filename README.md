# Is Escalation Worth It? Reproducibility Code

This repository contains experiment code for **Is Escalation Worth It? A Decision-Theoretic Study of Depth in LLM Cascades**.

The final ICLR manuscript studies exact selection of threshold cascades with up
to five models, nine deferral scores, an optimization method guided by the
first-order conditions, and counterfactual score and price experiments. The
current code supplement is in [`iclr2026/`](iclr2026/README.md). Its
[`EXPERIMENTS.md`](iclr2026/EXPERIMENTS.md) maps each active empirical analysis
to its scripts and outputs.

## Getting started

The supplement is a scripts-only release. It excludes the large graded
model-response files, embeddings, and full result caches. See
[`iclr2026/DATA.md`](iclr2026/DATA.md) for input requirements and
[`iclr2026/VALIDATION.md`](iclr2026/VALIDATION.md) for completed checks.

```sh
cd iclr2026
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify_package.py
.venv/bin/python reproduce.py tests --run
.venv/bin/python reproduce.py all
```

The last command prints the full computation plan. Add `--run` after supplying
the required inputs to execute it. Full reproduction is computationally
substantial and uses the eight-model data described in the paper.

## Earlier release

The `src/` directory, top-level `DATA.md`, `REPRODUCIBILITY.md`, and
`requirements.txt` preserve the earlier pairwise-envelope reproducibility
release. The cached results and figures there do not reproduce the final ICLR
paper. Use `iclr2026/` for the current experiments.

## License

Code is released under the Apache License 2.0. Dataset records remain subject
to their original licenses and terms.
