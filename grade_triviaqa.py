"""
Grade TriviaQA responses by normalized exact match against all accepted aliases.

Normalization (following the TriviaQA eval standard):
  - Lowercase
  - Strip leading/trailing whitespace
  - Remove punctuation
  - Collapse multiple spaces
  - Remove articles: a, an, the

The prediction is correct if its normalized form matches the normalized form
of ANY alias in the pipe-separated answer_aliases field.

answer_aliases is joined from data/prompts/triviaqa.parquet via prompt key,
since it was not carried through to the output parquets.

Usage:
    python3 grade_triviaqa.py
"""

import re
import string
from pathlib import Path

import pandas as pd

DATA_DIR    = Path("data/output_data")
PROMPTS_DIR = Path("data/prompts")


# ---------------------------------------------------------------------------
# Normalization (TriviaQA standard)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def grade_response(response: str, aliases_str: str) -> int:
    """
    Return 1 if the normalized response matches any normalized alias, else 0.
    aliases_str is the pipe-separated aliases field from the prompts parquet.
    """
    pred_norm = _normalize(response)
    aliases   = [_normalize(a) for a in aliases_str.split(" | ")]
    return int(pred_norm in aliases)


def grade_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["correct"] = [
        grade_response(row.response, row.answer_aliases)
        for row in df.itertuples()
    ]
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load aliases from prompts parquet and join by prompt text
    prompts = pd.read_parquet(PROMPTS_DIR / "triviaqa.parquet")[["prompt", "answer_aliases"]]

    files = sorted(DATA_DIR.glob("triviaqa-*.parquet"))
    for path in files:
        df = pd.read_parquet(path)

        if "correct" in df.columns:
            print(f"{path.name}: already graded (acc={df['correct'].mean():.3f}), skipping.")
            continue

        # Join aliases
        df = df.merge(prompts, on="prompt", how="left")
        n_missing = df["answer_aliases"].isna().sum()
        if n_missing:
            print(f"  [warn] {n_missing} rows missing aliases after join")

        df = grade_df(df)
        df = df.drop(columns=["answer_aliases"])  # don't bloat output parquet

        acc = df["correct"].mean()
        print(f"{path.name}: acc={acc:.3f}")
        df.to_parquet(path, index=False)

    print("\nDone.")
