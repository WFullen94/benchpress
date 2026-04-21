"""Tests for Phase 2 quality evaluation module."""

import math
import pytest

from benchpress.quality.tasks import TaskResult, _extract_letter
from benchpress.quality.scorer import QualityResult


# ---------------------------------------------------------------------------
# _extract_letter
# ---------------------------------------------------------------------------

def test_extract_letter_first_char():
    assert _extract_letter("A) something") == "A"
    assert _extract_letter("B") == "B"
    assert _extract_letter("c answer") == "C"


def test_extract_letter_embedded():
    assert _extract_letter("The answer is B.") == "B"
    assert _extract_letter("I think D is correct.") == "D"


def test_extract_letter_none():
    assert _extract_letter("") is None
    assert _extract_letter("The answer is X.") is None
    assert _extract_letter("  ") is None


# ---------------------------------------------------------------------------
# TaskResult
# ---------------------------------------------------------------------------

def test_task_result_accuracy():
    tr = TaskResult(name="MMLU", n_correct=40, n_total=50)
    assert tr.accuracy == pytest.approx(0.8)


def test_task_result_zero_total():
    tr = TaskResult(name="MMLU", n_correct=0, n_total=0)
    assert tr.accuracy == 0.0


# ---------------------------------------------------------------------------
# QualityResult
# ---------------------------------------------------------------------------

def _make_result(ppl=None, accuracies=None):
    tasks = []
    if accuracies:
        for i, acc in enumerate(accuracies):
            n_total = 50
            n_correct = round(acc * n_total)
            tasks.append(TaskResult(name=f"Task{i}", n_correct=n_correct, n_total=n_total))
    return QualityResult(
        model="test-model",
        backend="mlx",
        hardware="M3",
        timestamp="2026-04-20T00:00:00",
        perplexity=ppl,
        tasks=tasks,
    )


def test_mean_accuracy_no_tasks():
    r = _make_result()
    assert r.mean_accuracy is None


def test_mean_accuracy_single():
    r = _make_result(accuracies=[0.7])
    assert r.mean_accuracy == pytest.approx(0.7, abs=0.02)


def test_mean_accuracy_multiple():
    r = _make_result(accuracies=[0.6, 0.8])
    assert r.mean_accuracy == pytest.approx(0.7, abs=0.02)


def test_quality_score_accuracy_only():
    r = _make_result(accuracies=[0.8])
    assert r.quality_score == pytest.approx(0.8, abs=0.02)


def test_quality_score_with_perplexity():
    r = _make_result(ppl=10.0, accuracies=[0.8])
    # ppl_score = 1 - log(10)/log(1000) = 1 - 1/3 ≈ 0.667
    ppl_score = 1.0 - math.log(10) / math.log(1000)
    expected = 0.5 * 0.8 + 0.5 * ppl_score
    assert r.quality_score == pytest.approx(expected, abs=0.01)


def test_quality_score_none_when_no_data():
    r = _make_result()
    assert r.quality_score is None


def test_quality_score_perplexity_only():
    r = _make_result(ppl=20.0)
    ppl_score = 1.0 - math.log(20) / math.log(1000)
    assert r.quality_score == pytest.approx(ppl_score, abs=0.01)
