"""
Grade LiveCodeBench-style Python responses on included public tests.

This grader executes generated Python in a restricted subprocess with a timeout
and compares stdout to the expected public-test output. It adds:
    correct : 1 if all public tests pass, else 0
    stderr  : subprocess stderr or failure reason

Examples:
    python3 grade_livecodebench.py
    python3 grade_livecodebench.py data/output_data/livecodebench-gpt-4o-mini.parquet
"""

import json
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/output_data")
PROMPT_PATH = Path("data/prompts/livecodebench.parquet")
TIMEOUT_SECONDS = 6


def _loads_maybe(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def extract_python(response: str) -> str:
    if not isinstance(response, str):
        return ""
    match = re.search(r"```(?:python)?\s*(.*?)```", response, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else response.strip()


def normalize_output(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def run_one(code: str, test_input: str, expected_output: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solution.py"
        path.write_text(code)
        try:
            proc = subprocess.run(
                [sys.executable, str(path)],
                input=test_input,
                text=True,
                capture_output=True,
                timeout=TIMEOUT_SECONDS,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode != 0:
        return False, proc.stderr[-1000:]
    passed = normalize_output(proc.stdout) == normalize_output(expected_output)
    return passed, "" if passed else f"wrong output: {proc.stdout[-1000:]}"


def grade_response(response: str, public_test_cases) -> tuple[int, str]:
    code = extract_python(response)
    tests = _loads_maybe(public_test_cases)
    if not code or not tests:
        return 0, "missing code or tests"
    failures = []
    for test in tests:
        ok, err = run_one(code, test.get("input", ""), test.get("output", ""))
        if not ok:
            failures.append(err)
    return (0 if failures else 1), " | ".join(failures[:3])


def grade_df(df: pd.DataFrame) -> pd.DataFrame:
    if "public_test_cases" not in df.columns:
        prompts = pd.read_parquet(PROMPT_PATH)[["public_test_cases"]].reset_index(drop=True)
        df = df.reset_index(drop=True).join(prompts)
    results = [
        grade_response(row.response, row.public_test_cases)
        for row in df.itertuples(index=False)
    ]
    out = df.copy()
    out["correct"], out["stderr"] = zip(*results)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", help="LiveCodeBench response parquet files")
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    if not paths:
        paths = sorted(DATA_DIR.glob("livecodebench-*.parquet"))
    for path in paths:
        df = pd.read_parquet(path)
        if "correct" in df.columns:
            print(f"{path.name}: already graded (acc={df['correct'].mean():.3f}), skipping.")
            continue
        print(f"{path.name}: grading {len(df):,} responses")
        out = grade_df(df)
        out.to_parquet(path, index=False)
        print(f"  acc={out['correct'].mean():.3f}")


if __name__ == "__main__":
    main()
