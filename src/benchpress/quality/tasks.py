"""Task accuracy probes: MMLU, HellaSwag, TruthfulQA."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from benchpress.backends.base import Backend

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    name: str
    n_correct: int
    n_total: int

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n_total if self.n_total > 0 else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_IDX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
_LETTER_TO_IDX = {v: k for k, v in _IDX_TO_LETTER.items()}


def _extract_letter(text: str) -> str | None:
    """Return the first A/B/C/D found in *text*, or None."""
    text = text.strip()
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else None


def _choices_block(choices: list[str]) -> str:
    return "\n".join(f"{_IDX_TO_LETTER[i]}) {c}" for i, c in enumerate(choices[:4]))


def _ask(backend: Backend, prompt: str) -> str | None:
    result = backend.generate(prompt, max_tokens=16, temperature=0.0)
    return _extract_letter(result.output)


# ---------------------------------------------------------------------------
# Individual task runners
# ---------------------------------------------------------------------------

# Balanced subject sample covering easy and hard topics across MMLU categories
_MMLU_SUBJECTS = [
    "high_school_geography",
    "high_school_psychology",
    "high_school_computer_science",
    "high_school_biology",
    "college_biology",
    "college_computer_science",
    "world_religions",
    "moral_scenarios",
    "prehistory",
    "sociology",
]

_MMLU_FEW_SHOT = """\
The following are multiple choice questions. Answer with only the letter.

Question: What is the capital of France?
A) Berlin
B) Madrid
C) Paris
D) Rome
Answer: C

Question: Which data structure uses LIFO ordering?
A) Queue
B) Stack
C) Tree
D) Graph
Answer: B

Question: What is 15% of 200?
A) 25
B) 30
C) 35
D) 40
Answer: B

"""


def _eval_mmlu(
    backend: Backend,
    n: int,
    progress_cb: Callable[[int, int, str], None] | None,
) -> TaskResult:
    from datasets import load_dataset
    import random

    examples: list[dict] = []
    per_subject = max(1, n // len(_MMLU_SUBJECTS))

    for subj in _MMLU_SUBJECTS:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=False)
            rows = list(ds)
            random.seed(42)
            random.shuffle(rows)
            examples.extend(rows[:per_subject])
        except Exception:
            continue
        if len(examples) >= n:
            break

    examples = examples[:n]
    n_correct = 0

    for i, ex in enumerate(examples):
        prompt = (
            _MMLU_FEW_SHOT
            + f"Question: {ex['question']}\n"
            f"{_choices_block(ex['choices'])}\nAnswer:"
        )
        predicted = _ask(backend, prompt)
        if predicted == _IDX_TO_LETTER.get(ex["answer"]):
            n_correct += 1
        if progress_cb:
            progress_cb(i + 1, len(examples), "MMLU")

    return TaskResult(name="MMLU", n_correct=n_correct, n_total=len(examples))


def _eval_hellaswag(
    backend: Backend,
    n: int,
    progress_cb: Callable[[int, int, str], None] | None,
) -> TaskResult:
    from datasets import load_dataset

    ds = load_dataset(
        "Rowan/hellaswag", split=f"validation[:{n}]", trust_remote_code=False
    )
    examples = list(ds)
    n_correct = 0

    for i, ex in enumerate(examples):
        prompt = (
            "Choose the most likely continuation. Answer with only the letter A, B, C, or D.\n\n"
            f"Context: {ex['ctx']}\n"
            f"{_choices_block(ex['endings'])}\nAnswer:"
        )
        predicted = _ask(backend, prompt)
        if predicted == _IDX_TO_LETTER.get(int(ex["label"])):
            n_correct += 1
        if progress_cb:
            progress_cb(i + 1, len(examples), "HellaSwag")

    return TaskResult(name="HellaSwag", n_correct=n_correct, n_total=len(examples))


def _eval_truthfulqa(
    backend: Backend,
    n: int,
    progress_cb: Callable[[int, int, str], None] | None,
) -> TaskResult:
    from datasets import load_dataset

    ds = load_dataset(
        "truthfulqa/truthful_qa",
        "multiple_choice",
        split=f"validation[:{n}]",
        trust_remote_code=False,
    )
    examples = list(ds)
    n_correct = 0

    for i, ex in enumerate(examples):
        mc1 = ex["mc1_targets"]
        choices = mc1["choices"][:4]
        labels = mc1["labels"][:4]
        correct_idx = labels.index(1) if 1 in labels else 0

        prompt = (
            "Answer with only the letter A, B, C, or D.\n\n"
            f"Question: {ex['question']}\n"
            f"{_choices_block(choices)}\nAnswer:"
        )
        predicted = _ask(backend, prompt)
        if predicted == _IDX_TO_LETTER.get(correct_idx):
            n_correct += 1
        if progress_cb:
            progress_cb(i + 1, len(examples), "TruthfulQA")

    return TaskResult(name="TruthfulQA", n_correct=n_correct, n_total=len(examples))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_RUNNERS = {
    "mmlu": _eval_mmlu,
    "hellaswag": _eval_hellaswag,
    "truthfulqa": _eval_truthfulqa,
}


def run_task_probes(
    backend: Backend,
    tasks: list[str] | None = None,
    n_per_task: int = 50,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> list[TaskResult]:
    """Run the specified task accuracy probes and return results.

    Args:
        tasks: Which tasks to run. Defaults to all three.
        n_per_task: Number of examples per task.
        progress_cb: Called as (done, total, task_name) after each example.
    """
    if tasks is None:
        tasks = list(_RUNNERS)

    results: list[TaskResult] = []
    for name in tasks:
        if name not in _RUNNERS:
            raise ValueError(f"Unknown task '{name}'. Choose from: {list(_RUNNERS)}")
        results.append(_RUNNERS[name](backend, n_per_task, progress_cb))

    return results
