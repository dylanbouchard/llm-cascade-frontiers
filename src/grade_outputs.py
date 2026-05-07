"""
Grade generated response parquet files for all paper datasets.

Expected inputs live in data/output_data/{dataset}-{model}.parquet. The script
adds a binary correct column in place.

Examples:
    python3 grade_outputs.py
    python3 grade_outputs.py --dataset livecodebench
"""

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from grade_livecodebench import grade_df as grade_livecodebench
from grade_math import grade_df as grade_math
from grade_mmlu import grade_df as grade_mmlu
from grade_simpleqa import grade_df_async as grade_simpleqa_async
from grade_triviaqa import grade_df as grade_triviaqa


DATA_DIR = Path("data/output_data")
PROMPT_DIR = Path("data/prompts")
DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]


def output_files(dataset: str) -> list[Path]:
    return sorted(DATA_DIR.glob(f"{dataset}-*.parquet"))


def grade_deterministic(dataset: str, df: pd.DataFrame) -> pd.DataFrame:
    if dataset == "mmlu":
        return grade_mmlu(df)
    if dataset == "math_hard":
        return grade_math(df, response_col="response")
    if dataset == "triviaqa":
        prompts = pd.read_parquet(PROMPT_DIR / "triviaqa.parquet")[["prompt", "answer_aliases"]]
        if "answer_aliases" not in df.columns:
            df = df.merge(prompts, on="prompt", how="left")
        out = grade_triviaqa(df)
        return out
    if dataset == "livecodebench":
        return grade_livecodebench(df)
    raise ValueError(f"No deterministic grader for {dataset}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    simpleqa_llm = None

    for dataset in datasets:
        for path in output_files(dataset):
            df = pd.read_parquet(path)
            if "correct" in df.columns and not args.overwrite:
                print(f"[skip] {path.name} already graded (acc={df['correct'].mean():.3f})")
                continue
            print(f"[grade] {path.name}")
            if dataset == "simpleqa":
                if simpleqa_llm is None:
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    simpleqa_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
                out = await grade_simpleqa_async(df, simpleqa_llm)
            else:
                out = grade_deterministic(dataset, df)
            out.to_parquet(path, index=False)
            print(f"  acc={out['correct'].mean():.3f}")


if __name__ == "__main__":
    asyncio.run(main())
