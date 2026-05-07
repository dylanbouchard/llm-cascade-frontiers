"""
Grade responses for gpt-oss-20b and MiniMax-M2.7 across all four datasets.

Skips files that already have a 'correct' column.
SimpleQA uses the Gemini LLM judge (requires GOOGLE_API_KEY in env).

Usage:
    python3 grade_new_models.py
    python3 grade_new_models.py mmlu triviaqa  # subset
"""

import sys
import asyncio
from pathlib import Path

import pandas as pd

from grade_mmlu import grade_df as grade_mmlu
from grade_triviaqa import grade_df as grade_triviaqa
from grade_math import grade_df as grade_math
from grade_simpleqa import grade_df_async

DATA_DIR    = Path("data/output_data")
PROMPTS_DIR = Path("data/prompts")

NEW_MODELS = ["gpt-oss-20b", "MiniMax-M2.7"]
ALL_DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa"]


def run_deterministic(datasets: list[str]):
    for dataset in datasets:
        if dataset == "simpleqa":
            continue
        for model in NEW_MODELS:
            path = DATA_DIR / f"{dataset}-{model}.parquet"
            if not path.exists():
                print(f"  [missing] {path.name}")
                continue
            df = pd.read_parquet(path)
            if "correct" in df.columns:
                print(f"  [skip] {path.name} already graded (acc={df['correct'].mean():.3f})")
                continue

            print(f"  Grading {path.name} ...", end=" ", flush=True)
            if dataset == "mmlu":
                df = grade_mmlu(df)
            elif dataset == "triviaqa":
                prompts = pd.read_parquet(PROMPTS_DIR / "triviaqa.parquet")[["prompt", "answer_aliases"]]
                df = df.merge(prompts, on="prompt", how="left")
                df = grade_triviaqa(df)
                df = df.drop(columns=["answer_aliases"])
            elif dataset == "math_hard":
                df = grade_math(df, response_col="response")

            acc = df["correct"].mean()
            print(f"acc={acc:.3f}")
            df.to_parquet(path, index=False)


async def run_simpleqa(datasets: list[str]):
    if "simpleqa" not in datasets:
        return
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    for model in NEW_MODELS:
        path = DATA_DIR / f"simpleqa-{model}.parquet"
        if not path.exists():
            print(f"  [missing] {path.name}")
            continue
        df = pd.read_parquet(path)
        if "correct" in df.columns:
            print(f"  [skip] {path.name} already graded (acc={df['correct'].mean():.3f})")
            continue

        print(f"\n  Grading {path.name} ...")
        df = await grade_df_async(df, llm)
        n_nan = df["correct"].isna().sum()
        acc = df["correct"].mean()
        print(f"  {path.name}: acc={acc:.3f}  ({n_nan} grading failures)")
        df.to_parquet(path, index=False)


async def main():
    datasets = sys.argv[1:] if sys.argv[1:] else ALL_DATASETS

    print("=== Deterministic graders (MMLU, TriviaQA, MATH) ===")
    run_deterministic(datasets)

    print("\n=== SimpleQA (LLM judge) ===")
    await run_simpleqa(datasets)

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
