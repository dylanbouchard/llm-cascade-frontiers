"""
Load benchmark datasets from Hugging Face Hub for the cascade experiments.

Datasets:
  - MATH         (EleutherAI/hendrycks_math)            : Math reasoning, exact match on boxed answer
  - MMLU         (cais/mmlu)                            : Multiple choice, broad knowledge
  - SimpleQA     (basicv8vc/SimpleQA)                   : Factual recall, correct/incorrect/not-attempted
  - GPQA         (Idavidrein/gpqa)                      : Graduate-level science, multiple choice
  - TriviaQA     (mandarjoshi/trivia_qa)                : Factual recall, exact match with aliases
  - LiveCodeBench (livecodebench/code_generation_lite)  : Code generation, unit-test correctness;
                                                          loaded from HuggingFace and filtered to the
                                                          question_ids in
                                                          data/prompts/livecodebench_indices.json
                                                          to reconstruct the slice used in the paper.

Install dependencies:
  pip install datasets
"""

from datasets import load_dataset


def load_math_hard(levels: tuple = (3, 4, 5)) -> list[dict]:
    """
    Loads the full Hendrycks MATH test set filtered to the specified difficulty
    levels (default: 3, 4, and 5), giving ~3,800 problems across all subjects.

    Fields used:
      - problem  : str - the math problem
      - solution : str - full solution; final answer is in \\boxed{}
      - level    : str - difficulty level string e.g. 'Level 4'
      - type     : str - math subject (Algebra, Geometry, etc.)

    U(y): exact match on the answer extracted from \\boxed{} in the solution.
    extract_boxed_answer() handles this at scoring time.
    """
    from datasets import concatenate_datasets

    subjects = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]
    level_strings = {f"Level {l}" for l in levels}

    all_test = [load_dataset("EleutherAI/hendrycks_math", s)["test"] for s in subjects]
    full_test = concatenate_datasets(all_test)

    return [
        {
            "question": row["problem"],
            "solution": row["solution"],
            "subject":  row["type"],
            "level":    row["level"],
        }
        for row in full_test
        if row["level"] in level_strings
    ]


def load_triviaqa(split: str = "validation") -> list[dict]:
    """
    Fields used:
      - question        : str       - the trivia question
      - answer.value    : str       - primary answer string
      - answer.aliases  : list[str] - all acceptable answer strings

    U(y): normalized exact match against any alias.

    Uses rc.nocontext config (no passage provided) to test parametric
    knowledge only - consistent with SimpleQA and MATH-500.
    Split: 'validation' (test answers are withheld on the leaderboard).
    """
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
    return [
        {
            "question":      row["question"],
            "answer":        row["answer"]["value"],
            "answer_aliases": row["answer"]["aliases"],
        }
        for row in ds
    ]


def load_gsm8k(split: str = "test") -> list[dict]:
    """
    Fields used:
      - question : str   - the math problem prompt
      - answer   : str   - chain-of-thought ending with '#### <final_answer>'

    U(y): exact match on the integer after '####'.
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    return [{"question": row["question"], "answer": row["answer"]} for row in ds]


def load_mmlu(split: str = "test", subject: str = "all") -> list[dict]:
    """
    Fields used:
      - question : str       - the question text
      - choices  : list[str] - four answer options [A, B, C, D]
      - answer   : int       - index of correct choice (0-3)

    U(y): 1 if model selects choices[answer], else 0.

    Set subject='all' to load all 57 subjects, or pass a specific subject name
    (e.g. 'abstract_algebra') to load a single subject.
    """
    ds = load_dataset("cais/mmlu", subject, split=split)
    return [
        {
            "question": row["question"],
            "choices": row["choices"],
            "answer": row["answer"],
        }
        for row in ds
    ]


def load_simpleqa(split: str = "test") -> list[dict]:
    """
    Fields used:
      - problem  : str - the factual question
      - answer   : str - the ground truth short answer

    U(y): graded as 'correct', 'incorrect', or 'not_attempted' using
    OpenAI's grader prompt (see SimpleQA paper). For binary U(y), map
    correct -> 1, incorrect/not_attempted -> 0.

    Note: verify the dataset identifier at https://huggingface.co/datasets/openai/SimpleQA
    if this raises a FileNotFoundError.
    """
    ds = load_dataset("basicv8vc/SimpleQA", split=split)
    return [{"question": row["problem"], "answer": row["answer"]} for row in ds]


def load_gpqa(subset: str = "gpqa_diamond", split: str = "train") -> list[dict]:
    """
    Fields used:
      - Question                    : str - the question text
      - Correct Answer              : str - correct answer option text
      - Incorrect Answer 1/2/3      : str - distractor options

    U(y): 1 if model selects the correct answer option, else 0.

    Subsets:
      - gpqa_diamond  (198 questions) : hardest, expert-validated
      - gpqa_main     (448 questions) : full set
      - gpqa_extended (546 questions) : includes lower-quality items

    Note: GPQA has no official test split; 'train' contains all examples.
    Use your own calibration/test split.
    """
    ds = load_dataset("Idavidrein/gpqa", subset, split=split)
    return [
        {
            "question": row["Question"],
            "correct_answer": row["Correct Answer"],
            "incorrect_answers": [
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ],
        }
        for row in ds
    ]


def load_livecodebench(indices_path: str = "data/prompts/livecodebench_indices.json") -> list[dict]:
    """
    Load the LiveCodeBench slice used in the paper. The benchmark is not redistributed
    in this packet; this function reloads it from the HuggingFace Hub
    (livecodebench/code_generation_lite, test split) and filters to the question_ids
    in indices_path, preserving the original ordering.

    Fields used:
      - question_id        : str  - unique problem identifier (e.g. '1873_A')
      - question_content   : str  - problem statement
      - platform           : str  - source platform ('codeforces', 'leetcode', etc.)
      - starter_code       : str  - any provided code stub
      - public_test_cases  : str  - JSON list of input/output test cases
      - metadata           : str  - JSON metadata
      - difficulty         : str  - 'easy' / 'medium' / 'hard'

    U(y): unit-test pass rate.
    """
    import json

    with open(indices_path) as f:
        manifest = json.load(f)
    target_ids = manifest["question_ids"]

    ds = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
    rows_by_id = {row["question_id"]: row for row in ds}

    return [
        {
            "question_id":       rows_by_id[qid]["question_id"],
            "question_content":  rows_by_id[qid]["question_content"],
            "platform":          rows_by_id[qid]["platform"],
            "starter_code":      rows_by_id[qid]["starter_code"],
            "public_test_cases": rows_by_id[qid]["public_test_cases"],
            "metadata":          rows_by_id[qid]["metadata"],
            "difficulty":        rows_by_id[qid]["difficulty"],
        }
        for qid in target_ids
    ]


def load_all() -> dict[str, list[dict]]:
    print("Loading GSM8K...")
    gsm8k = load_gsm8k()
    print(f"  {len(gsm8k)} examples")

    print("Loading MMLU...")
    mmlu = load_mmlu()
    print(f"  {len(mmlu)} examples")

    print("Loading SimpleQA...")
    simpleqa = load_simpleqa()
    print(f"  {len(simpleqa)} examples")

    print("Loading GPQA (diamond)...")
    gpqa = load_gpqa()
    print(f"  {len(gpqa)} examples")

    return {"gsm8k": gsm8k, "mmlu": mmlu, "simpleqa": simpleqa, "gpqa": gpqa}


if __name__ == "__main__":
    datasets = load_all()

    print("\n--- Sample entries ---")

    print("\nGSM8K:")
    print(f"  Q: {datasets['gsm8k'][0]['question'][:80]}...")
    print(f"  A: {datasets['gsm8k'][0]['answer'][-40:]}")

    print("\nMMLU:")
    ex = datasets["mmlu"][0]
    print(f"  Q: {ex['question'][:80]}...")
    print(f"  Choices: {ex['choices']}")
    print(f"  Correct: {ex['answer']} ({ex['choices'][ex['answer']]})")

    print("\nSimpleQA:")
    print(f"  Q: {datasets['simpleqa'][0]['question'][:80]}...")
    print(f"  A: {datasets['simpleqa'][0]['answer']}")

    print("\nGPQA:")
    ex = datasets["gpqa"][0]
    print(f"  Q: {ex['question'][:80]}...")
    print(f"  Correct: {ex['correct_answer'][:60]}")
