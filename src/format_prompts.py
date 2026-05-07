"""
Format prompts for each dataset and save as parquet.

UQLM expects a list of prompt strings. Each parquet file contains one row
per example with the formatted prompt and everything needed to compute U(y)
at scoring time.

Output:
    data/prompts/math_hard.parquet
    data/prompts/mmlu.parquet
    data/prompts/simpleqa.parquet
    data/prompts/gpqa.parquet
    data/prompts/triviaqa.parquet
    data/prompts/livecodebench.parquet   (regenerated from the indices manifest;
                                          not redistributed in this packet)

Install dependencies:
    pip install datasets pandas pyarrow
"""

import random
from pathlib import Path

import pandas as pd

import re

from load_datasets import (
    load_gpqa,
    load_livecodebench,
    load_math_hard,
    load_mmlu,
    load_simpleqa,
    load_triviaqa,
)

SEED = 42
OUTPUT_DIR = Path("data/prompts")
MCQA_LABELS = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# MATH (levels 4-5)
# ---------------------------------------------------------------------------

def _extract_boxed(solution: str) -> str:
    """Extract the content of the last \\boxed{} in a solution string."""
    matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", solution)
    return matches[-1].strip() if matches else solution.strip()


def format_math_hard() -> pd.DataFrame:
    """
    Columns:
      - prompt        : formatted prompt string
      - ground_truth  : answer extracted from \\boxed{} in solution
      - subject       : math subject (for subgroup analysis)
      - level         : difficulty level string e.g. 'Level 4'
    """
    examples = load_math_hard()
    rows = []
    for idx, ex in enumerate(examples):
        prompt = (
            "Solve the following math problem. "
            "Put your final answer inside \\boxed{}.\n\n"
            f"Problem: {ex['question']}"
        )
        rows.append({
            "id":           idx,
            "prompt":       prompt,
            "ground_truth": _extract_boxed(ex["solution"]),
            "subject":      ex["subject"],
            "level":        ex["level"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TriviaQA
# ---------------------------------------------------------------------------

def format_triviaqa() -> pd.DataFrame:
    """
    Columns:
      - prompt          : formatted factual question
      - ground_truth    : primary answer string
      - answer_aliases  : pipe-separated list of all acceptable answers
                          (use for normalized exact-match scoring)
    """
    examples = load_triviaqa()
    rows = []
    for idx, ex in enumerate(examples):
        prompt = (
            "Answer the following question as concisely as possible. "
            "Give only the answer, no explanation.\n\n"
            f"Question: {ex['question']}"
        )
        rows.append({
            "id":             idx,
            "prompt":         prompt,
            "ground_truth":   ex["answer"],
            "answer_aliases": " | ".join(ex["answer_aliases"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

def format_mmlu() -> pd.DataFrame:
    """
    Columns:
      - prompt          : formatted multiple-choice prompt
      - ground_truth    : correct answer label ('A', 'B', 'C', or 'D')
      - correct_index   : integer index of correct choice (0-3)
      - choices         : pipe-separated choice texts for reference
    """
    examples = load_mmlu()
    rows = []
    for idx, ex in enumerate(examples):
        choices_text = "\n".join(
            f"{label}. {text}"
            for label, text in zip(MCQA_LABELS, ex["choices"])
        )
        prompt = (
            f"Question: {ex['question']}\n\n"
            f"{choices_text}\n\n"
            "Answer with only the letter of the correct option (A, B, C, or D)."
        )
        rows.append({
            "id":             idx,
            "prompt":         prompt,
            "ground_truth":   MCQA_LABELS[ex["answer"]],
            "correct_index":  ex["answer"],
            "choices":        " | ".join(ex["choices"]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SimpleQA
# ---------------------------------------------------------------------------

def format_simpleqa() -> pd.DataFrame:
    """
    Columns:
      - prompt        : formatted factual question
      - ground_truth  : correct short answer string

    U(y) requires OpenAI's grader prompt to classify responses as
    'correct', 'incorrect', or 'not_attempted'. ground_truth is the
    reference answer passed to the grader.
    """
    examples = load_simpleqa()
    rows = []
    for idx, ex in enumerate(examples):
        prompt = (
            "Answer the following question as concisely as possible. "
            "If you are not sure, give your best guess.\n\n"
            f"Question: {ex['question']}"
        )
        rows.append({
            "id":            idx,
            "prompt":        prompt,
            "ground_truth":  ex["answer"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GPQA
# ---------------------------------------------------------------------------

def format_gpqa() -> pd.DataFrame:
    """
    Columns:
      - prompt           : formatted multiple-choice prompt (options shuffled)
      - ground_truth     : correct answer label after shuffling ('A'-'D')
      - correct_answer   : raw correct answer text (for verification)
      - choices          : pipe-separated shuffled choice texts for reference

    Options are shuffled with a fixed seed for reproducibility.
    """
    rng = random.Random(SEED)
    examples = load_gpqa()
    rows = []
    for idx, ex in enumerate(examples):
        options = [ex["correct_answer"]] + ex["incorrect_answers"]
        rng.shuffle(options)
        correct_label = MCQA_LABELS[options.index(ex["correct_answer"])]
        choices_text = "\n".join(
            f"{label}. {text}"
            for label, text in zip(MCQA_LABELS, options)
        )
        prompt = (
            f"Question: {ex['question']}\n\n"
            f"{choices_text}\n\n"
            "Answer with only the letter of the correct option (A, B, C, or D)."
        )
        rows.append({
            "id":             idx,
            "prompt":         prompt,
            "ground_truth":   correct_label,
            "correct_answer": ex["correct_answer"],
            "choices":        " | ".join(options),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LiveCodeBench
# ---------------------------------------------------------------------------

_LCB_PROMPT_CALLABLE = """
You are an expert Python programmer. You always return complete, executable Python code.

Your task:
- Read the problem description.
- Complete the method inside the starter code.
- Return only valid Python code with no explanations or markdown.

Problem:
{question_content}

Starter code:
{starter_code}

Guidelines:
- Keep the class name and method signature exactly as provided.
- NEVER rename the function or modify its arguments.
- NEVER return only the function body.
- Do not add print statements or extra text.
- Just return the completed Python solution.
"""

_LCB_PROMPT_STDIO = """
You are an expert Python programmer. You always return complete, executable Python code.

Your task:
- Read ALL input from standard input (stdin).
- Produce the required output to standard output (stdout) ONLY.
- Return only valid Python code with no explanations or markdown.

Problem:
{question_content}

Guidelines:
- Parse input exactly as described (use input() or sys.stdin).
- Print outputs exactly as specified (correct order, spacing, and newlines).
- Do NOT print any extra text (no debug logs, prompts, or explanations).
- Do NOT read or write files, and do NOT use network access.
- Use only the Python standard library.
- Ensure the program terminates promptly and handles edge cases within time limits.

Just return the completed Python solution.
""".strip()


def format_livecodebench() -> pd.DataFrame:
    """
    Columns:
      - id                : int  - row index in the slice
      - question_id       : str  - upstream LCB id (e.g. '1873_A')
      - prompt            : str  - formatted code-generation prompt
      - platform          : str  - source platform
      - starter_code      : str  - starter snippet (empty for stdio problems)
      - public_test_cases : str  - JSON list of test cases (used by grader)
      - metadata          : str  - JSON metadata (used by grader)
      - difficulty        : str  - 'easy' / 'medium' / 'hard'

    Prompts use the leetcode/callable template when starter_code is provided,
    and the I/O (stdio) template otherwise. Templates match the strings used to
    generate the responses scored in the paper.

    The benchmark itself is loaded from HuggingFace and filtered to the
    question_ids in data/prompts/livecodebench_indices.json; this packet does
    not redistribute LiveCodeBench data.
    """
    examples = load_livecodebench()
    rows = []
    for idx, ex in enumerate(examples):
        if ex["platform"] == "leetcode":
            prompt = _LCB_PROMPT_CALLABLE.format(
                question_content=ex["question_content"],
                starter_code=ex["starter_code"],
            )
        else:
            prompt = _LCB_PROMPT_STDIO.format(question_content=ex["question_content"])
        rows.append({
            "id":                idx,
            "question_id":       ex["question_id"],
            "prompt":            prompt,
            "platform":          ex["platform"],
            "starter_code":      ex["starter_code"],
            "public_test_cases": ex["public_test_cases"],
            "metadata":          ex["metadata"],
            "difficulty":        ex["difficulty"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "math_hard":     format_math_hard,
        "mmlu":          format_mmlu,
        "simpleqa":      format_simpleqa,
        "gpqa":          format_gpqa,
        "triviaqa":      format_triviaqa,
        "livecodebench": format_livecodebench,
    }

    for name, fmt_fn in datasets.items():
        print(f"Formatting {name}...")
        df = fmt_fn()
        out_path = OUTPUT_DIR / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {len(df):,} examples -> {out_path}")

    print("\nDone.")
