"""
Grade MMLU responses by exact letter match (A/B/C/D).

Extracts the first A-D letter from each response. Most models return
a bare letter; some wrap it in a sentence, so we scan for the first
occurrence of a standalone A, B, C, or D.

Usage:
    from grade_mmlu import grade_df

    df = pd.read_parquet("mmlu-gpt-4o-mini.parquet")
    df = grade_df(df)
    print(df["correct"].mean())
"""

import re
import glob
from pathlib import Path

import pandas as pd


def extract_letter(text: str) -> str | None:
    """
    Extract the first standalone A/B/C/D letter from a response string.
    Returns None if no letter is found.
    """
    if not isinstance(text, str):
        return None
    # Match a standalone letter (word boundary on both sides)
    m = re.search(r"\b([A-D])\b", text.strip(), re.IGNORECASE)
    return m.group(1).upper() if m else None


def grade_df(
    df: pd.DataFrame,
    response_col: str = "response",
    gt_col: str = "ground_truth",
) -> pd.DataFrame:
    """
    Grade an MMLU DataFrame.

    Adds columns:
      - predicted_answer : extracted letter (or None if unparseable)
      - correct          : 1 if matches ground truth, 0 otherwise
    """
    df = df.copy()
    df["predicted_answer"] = df[response_col].apply(extract_letter)
    df["correct"] = (df["predicted_answer"] == df[gt_col].str.strip().str.upper()).astype(int)
    return df


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data" / "output_data"
    files = sorted(data_dir.glob("mmlu-*.parquet"))

    for path in files:
        df = pd.read_parquet(path)
        df = grade_df(df)
        acc = df["correct"].mean()
        unparsed = df["predicted_answer"].isna().sum()
        print(f"{path.name}: accuracy={acc:.3f}  unparsed={unparsed}/{len(df)}")
        df.to_parquet(path, index=False)

    print("\nDone. Graded columns written back to each parquet.")
