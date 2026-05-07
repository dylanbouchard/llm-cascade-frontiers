"""
Grade math responses against reference boxed answers.

Grading pipeline:
  1. Extract \\boxed{} content from model response (falls back to full response)
  2. Try symbolic equivalence via sympy (handles \\frac{1}{2} == 0.5, etc.)
  3. Fall back to normalized string match

Usage:
    from grade_math import grade_df

    df = pd.read_parquet("math_hard-gpt-4o-mini.parquet")
    df = grade_df(df)         # adds 'predicted_answer' and 'correct' columns
    print(df["correct"].mean())
"""

import re
import unicodedata

import pandas as pd


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str:
    """
    Extract the content of the last \\boxed{} in a string.
    Uses brace-depth tracking with a 500-char scan limit per occurrence
    to handle arbitrary nesting without catastrophic backtracking.
    Falls back to extracting the last number/expression in the text.
    """
    results = []
    i = 0
    while i < len(text):
        idx = text.find(r'\boxed{', i)
        if idx == -1:
            break
        start = idx + len(r'\boxed{')
        depth = 1
        j = start
        limit = min(len(text), start + 500)  # guard against malformed LaTeX
        while j < limit and depth > 0:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start:j - 1])
        i = idx + 1

    if results:
        return results[-1].strip()

    # No boxed answer: grab the last standalone number or simple expression
    number_matches = re.findall(r"[-+]?\d+(?:[./]\d+)?", text)
    if number_matches:
        return number_matches[-1].strip()

    return text.strip()


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_str(s: str) -> str:
    """Lowercase, strip whitespace, remove unicode punctuation variants."""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _strip_latex(s: str) -> str:
    """Remove common LaTeX wrappers to aid string fallback matching."""
    s = re.sub(r"\\left|\\right|\\,|\\!", "", s)
    s = re.sub(r"\$", "", s)
    s = s.replace("{", "").replace("}", "")
    return s.strip()


# ---------------------------------------------------------------------------
# Symbolic equivalence via sympy
# ---------------------------------------------------------------------------

def _sympy_equal(pred: str, ref: str) -> bool | None:
    """
    Return True if pred and ref are symbolically equal, False if not,
    None if sympy cannot parse either expression.

    Tries LaTeX parsing first, then sympify for plain expressions.
    Also checks numeric equality (handles 0.5 == \\frac{1}{2}).
    Times out after 5 seconds to prevent sympy.simplify() from hanging.
    """
    import signal

    def _timeout_handler(signum, frame):
        raise TimeoutError("sympy timeout")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(5)
    try:
        from sympy import simplify, N
        from sympy.parsing.latex import parse_latex
        from sympy import sympify

        def _parse(s: str):
            try:
                return parse_latex(s)
            except Exception:
                return sympify(s)

        pred_expr = _parse(pred)
        ref_expr  = _parse(ref)

        # Symbolic difference
        diff = simplify(pred_expr - ref_expr)
        if diff == 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            return True

        # Numeric fallback (catches 0.5 == 1/2 when simplify misses it)
        try:
            if abs(float(N(pred_expr)) - float(N(ref_expr))) < 1e-9:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                return True
        except Exception:
            pass

        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return False
    except Exception:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        return None


# ---------------------------------------------------------------------------
# Per-response grader
# ---------------------------------------------------------------------------

def grade_response(response: str, ground_truth: str) -> tuple[str, int]:
    """
    Grade a single response against the ground truth answer.

    Returns:
        (predicted_answer, correct)
        predicted_answer : extracted boxed content (or full response if no box)
        correct          : 1 if correct, 0 otherwise
    """
    pred = extract_boxed(response)
    ref  = extract_boxed(ground_truth) if r'\boxed{' in ground_truth else ground_truth.strip()

    # 1. Symbolic equivalence
    result = _sympy_equal(pred, ref)
    if result is True:
        return pred, 1
    if result is False:
        return pred, 0

    # 2. Normalized string match (sympy failed to parse)
    pred_norm = _normalize_str(_strip_latex(pred))
    ref_norm  = _normalize_str(_strip_latex(ref))
    correct   = int(pred_norm == ref_norm)
    return pred, correct


# ---------------------------------------------------------------------------
# DataFrame-level grader
# ---------------------------------------------------------------------------

def grade_df(df: pd.DataFrame, response_col: str = "responses", gt_col: str = "ground_truth") -> pd.DataFrame:
    """
    Grade a DataFrame of math responses.

    Expects columns:
      - `response_col`  : model response strings
      - `gt_col`        : reference answer strings (boxed content from solution)

    Adds columns:
      - predicted_answer : extracted boxed answer from the model response
      - correct          : 1 if correct, 0 otherwise

    Returns a copy of the DataFrame with the new columns added.
    """
    from tqdm import tqdm
    df = df.copy()
    results = [
        grade_response(row[response_col], row[gt_col])
        for _, row in tqdm(df.iterrows(), total=len(df), desc="grading")
    ]
    df["predicted_answer"], df["correct"] = zip(*results)
    return df


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        # (response,                         ground_truth,  expected)
        (r"The answer is \boxed{\frac{1}{2}}", r"\frac{1}{2}", 1),
        (r"\boxed{0.5}",                       r"\frac{1}{2}", 1),
        (r"\boxed{2}",                         r"2",           1),
        (r"\boxed{3}",                         r"2",           0),
        (r"\boxed{\sqrt{2}}",                  r"\sqrt{2}",    1),
        (r"The answer is 42.",                 r"42",          1),  # no boxed fallback
        (r"\boxed{x^2 + 1}",                   r"1 + x^2",     1),
    ]

    print(f"{'Response':<40} {'GT':<15} {'Expected':>8} {'Got':>5}")
    print("-" * 75)
    for response, gt, expected in cases:
        pred, correct = grade_response(response, gt)
        status = "OK" if correct == expected else "FAIL"
        print(f"{response:<40} {gt:<15} {expected:>8} {correct:>5}  {status}")
