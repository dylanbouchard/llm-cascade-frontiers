"""
Grade SimpleQA responses using Gemini 2.5 Flash as an LLM judge.

Grading rubric (from OpenAI SimpleQA paper):
  A: CORRECT       -> correct = 1
  B: INCORRECT     -> correct = 0
  C: NOT_ATTEMPTED -> correct = 0

Processes in batches with tqdm progress. Uses LangChain ainvoke for
compatibility with the rest of the project.

Usage:
    python3 grade_simpleqa.py
"""

import asyncio
import re
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR       = Path("data/output_data")
MODEL          = "gemini-2.5-flash"
BATCH_SIZE     = 20     # rows per batch
MAX_CONCURRENT = 3      # parallel requests within each batch

GRADER_PROMPT = """\
Your job is to look at a question, a gold target answer, and a predicted answer, \
and then assign a grade of CORRECT, INCORRECT, or NOT_ATTEMPTED.

Use the following guidelines:
- CORRECT: the predicted answer matches the gold target (allow minor variations in phrasing, \
spelling, or formatting as long as the factual content is equivalent).
- NOT_ATTEMPTED: the predicted answer explicitly declines to answer, says it doesn't know, \
or gives no substantive response.
- INCORRECT: the predicted answer is factually wrong.

Question: {question}
Gold target: {gold}
Predicted answer: {predicted}

Reply with exactly one word: CORRECT, INCORRECT, or NOT_ATTEMPTED."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_question(prompt: str) -> str:
    m = re.search(r"Question:\s*(.+)", prompt, re.DOTALL)
    return m.group(1).strip() if m else prompt.strip()


def _parse_grade(text: str | None) -> float:
    if not text:
        return float("nan")
    t = text.strip().upper()
    if t.startswith("CORRECT"):
        return 1.0
    if t.startswith("INCORRECT") or t.startswith("NOT_ATTEMPTED"):
        return 0.0
    return float("nan")


async def _grade_one(llm, sem, question: str, gold: str, predicted: str) -> float:
    prompt = GRADER_PROMPT.format(question=question, gold=gold, predicted=predicted)
    for attempt in range(6):
        try:
            async with sem:
                response = await llm.ainvoke([HumanMessage(content=prompt)])
            return _parse_grade(response.content)
        except Exception as e:
            if attempt == 5:
                return float("nan")
            msg = str(e)
            retry_match = re.search(r"retry[^\d]*(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
            wait = float(retry_match.group(1)) + 1 if retry_match else 2 ** attempt
            await asyncio.sleep(wait)


async def grade_df_async(df: pd.DataFrame, llm) -> pd.DataFrame:
    df = df.copy()
    sem = asyncio.Semaphore(MAX_CONCURRENT)

    questions  = df["prompt"].apply(_extract_question).tolist()
    golds      = df["ground_truth"].tolist()
    predicteds = df["response"].tolist()

    results = [None] * len(df)
    n_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

    with tqdm(total=len(df), desc="grading") as pbar:
        for batch_idx in range(n_batches):
            lo = batch_idx * BATCH_SIZE
            hi = min(lo + BATCH_SIZE, len(df))

            tasks = [
                _grade_one(llm, sem, questions[i], golds[i], predicteds[i])
                for i in range(lo, hi)
            ]
            batch_results = await asyncio.gather(*tasks)
            for i, r in enumerate(batch_results):
                results[lo + i] = r

            pbar.update(hi - lo)
            n_done = hi
            n_correct = sum(1 for r in results[:n_done] if r == 1.0)
            n_nan     = sum(1 for r in results[:n_done] if r is None or (r != r))
            pbar.set_postfix(acc=f"{n_correct/n_done:.3f}", failures=n_nan)

    df["correct"] = results
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    llm = ChatGoogleGenerativeAI(
        model=MODEL,
        temperature=0.0,
    )

    files = sorted(DATA_DIR.glob("simpleqa-*.parquet"))
    for path in files:
        df = pd.read_parquet(path)
        if "correct" in df.columns:
            print(f"{path.name}: already graded (acc={df['correct'].mean():.3f}), skipping.")
            continue

        print(f"\n{path.name}: grading {len(df)} responses ...")
        df = await grade_df_async(df, llm)
        n_nan = df["correct"].isna().sum()
        acc   = df["correct"].mean()
        print(f"{path.name}: done  acc={acc:.3f}  ({n_nan} grading failures)")
        df.to_parquet(path, index=False)


if __name__ == "__main__":
    asyncio.run(main())
