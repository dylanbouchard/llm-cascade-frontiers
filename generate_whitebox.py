"""
Generate model responses and white-box UQ scores with UQLM WhiteBoxUQ.

Outputs one parquet per dataset/model under data/output_data/:
    {dataset}-{model}.parquet

The script uses the formatted prompt/source records in data/prompts/. It
requires provider credentials for the requested model backend.

Examples:
    python3 generate_whitebox.py --dataset mmlu --model gpt-4o-mini
    python3 generate_whitebox.py --dataset all --model all --batch-size 100
"""

import argparse
import asyncio
import inspect
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "llm_cascade_mpl"))
from uqlm import WhiteBoxUQ

from models import MODELS, MODEL_SPECS


PROMPT_DIR = Path("data/prompts")
OUTPUT_DIR = Path("data/output_data")

DATASETS = ["mmlu", "triviaqa", "math_hard", "simpleqa", "livecodebench"]
SCORERS = [
    "sequence_probability",
    "min_probability",
    "min_token_negentropy",
    "mean_token_negentropy",
    "probability_margin",
]


def livecodebench_prompt(row: pd.Series) -> str:
    starter = row.get("starter_code")
    starter_block = ""
    if isinstance(starter, str) and starter.strip():
        starter_block = f"\nStarter code:\n{starter.strip()}\n"
    return (
        "You are an expert Python programmer. You always return complete, "
        "executable Python code.\n\n"
        "Your task:\n"
        "- Read ALL input from standard input (stdin).\n"
        "- Produce the required output to standard output (stdout) ONLY.\n"
        "- Return only valid Python code with no explanations or markdown.\n\n"
        f"Problem:\n{row['question_content']}\n"
        f"{starter_block}\n"
        "Guidelines:\n"
        "- Parse input exactly as described (use input() or sys.stdin).\n"
        "- Print outputs exactly as specified (correct order, spacing, and newlines).\n"
        "- Do NOT print any extra text (no debug logs, prompts, or explanations).\n"
        "- Do NOT read or write files, and do NOT use network access.\n"
        "- Use only the Python standard library.\n"
    )


def load_prompt_frame(dataset: str) -> pd.DataFrame:
    path = PROMPT_DIR / f"{dataset}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    df = pd.read_parquet(path)
    if "prompt" not in df.columns:
        if dataset != "livecodebench":
            raise ValueError(f"{path} has no prompt column")
        df = df.copy()
        df["prompt"] = df.apply(livecodebench_prompt, axis=1)
    return df


def top_k_for_model(model_name: str) -> int:
    # Together-hosted Llama/Qwen endpoints expose fewer top logprobs.
    if any(key in model_name for key in ["llama", "qwen"]):
        return 5
    return 15


async def maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


async def generate_dataset_model(
    dataset: str,
    model_name: str,
    batch_size: int,
    max_calls_per_min: int,
    overwrite: bool,
    sleep_seconds: float,
):
    df = load_prompt_frame(dataset)
    prompts = df["prompt"].tolist()
    out_path = OUTPUT_DIR / f"{dataset}-{model_name}.parquet"
    batch_dir = OUTPUT_DIR / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} exists")
        return

    llm = MODELS[model_name]
    uq = WhiteBoxUQ(
        llm=llm,
        scorers=SCORERS,
        max_calls_per_min=max_calls_per_min,
        top_k_logprobs=top_k_for_model(model_name),
    )

    parts = []
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_idx in range(n_batches):
        lo = batch_idx * batch_size
        hi = min(lo + batch_size, len(prompts))
        batch_path = batch_dir / f"{dataset}-{model_name}-batch-{batch_idx:04d}.parquet"
        if batch_path.exists() and not overwrite:
            print(f"  batch {batch_idx:04d}: cached")
            parts.append(pd.read_parquet(batch_path))
            continue

        print(f"  batch {batch_idx:04d}: rows {lo}-{hi - 1}", flush=True)
        result = await maybe_await(uq.generate_and_score(prompts=prompts[lo:hi]))
        batch_df = result.to_df()
        batch_df.to_parquet(batch_path, index=False)
        parts.append(batch_df)
        if sleep_seconds > 0 and batch_idx < n_batches - 1:
            time.sleep(sleep_seconds)

    out = pd.concat(parts, ignore_index=True)
    meta_cols = [c for c in df.columns if c != "prompt"]
    for col in meta_cols:
        if col not in out.columns:
            out[col] = df[col].values[: len(out)]
    out.to_parquet(out_path, index=False)
    print(f"[saved] {out_path} ({len(out):,} rows)")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--model", choices=list(MODEL_SPECS) + ["all"], default="all")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-calls-per-min", type=int, default=15)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    models = list(MODEL_SPECS) if args.model == "all" else [args.model]

    for dataset in datasets:
        for model_name in models:
            print(f"\n=== {dataset} | {model_name} ===")
            await generate_dataset_model(
                dataset,
                model_name,
                args.batch_size,
                args.max_calls_per_min,
                args.overwrite,
                args.sleep_seconds,
            )


if __name__ == "__main__":
    asyncio.run(main())
