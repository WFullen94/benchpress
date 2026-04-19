"""Tests for InferenceResult metric calculations."""

import pytest
from benchpress.metrics import InferenceResult


def _make_result(
    output_tokens: int = 10,
    first_token_time: float = 0.1,
    total_time: float = 1.1,
    timestamps: list[float] | None = None,
) -> InferenceResult:
    if timestamps is None:
        timestamps = [0.1 + i * 0.1 for i in range(output_tokens)]
    return InferenceResult(
        prompt="test",
        output="x" * output_tokens,
        prompt_tokens=5,
        output_tokens=output_tokens,
        first_token_time=first_token_time,
        total_time=total_time,
        token_timestamps=timestamps,
    )


def test_ttft():
    r = _make_result(first_token_time=0.05)
    assert r.ttft == pytest.approx(0.05)


def test_tokens_per_second_basic():
    r = _make_result(output_tokens=9, first_token_time=0.1, total_time=1.0)
    # decode_time = 1.0 - 0.1 = 0.9, tokens = 8 (exclude first)
    assert r.tokens_per_second == pytest.approx(8 / 0.9, rel=1e-3)


def test_tokens_per_second_zero_if_single_token():
    r = _make_result(output_tokens=1, first_token_time=0.5, total_time=0.5)
    assert r.tokens_per_second == 0.0


def test_inter_token_latencies_count():
    timestamps = [0.1, 0.2, 0.35, 0.5]
    r = _make_result(output_tokens=4, timestamps=timestamps)
    itls = r.inter_token_latencies
    assert len(itls) == 3
    assert itls[0] == pytest.approx(0.1)
    assert itls[1] == pytest.approx(0.15)


def test_mean_tbt():
    timestamps = [0.1, 0.2, 0.3]
    r = _make_result(output_tokens=3, timestamps=timestamps)
    assert r.mean_tbt == pytest.approx(0.1)


def test_no_timestamps_returns_empty_itl():
    r = _make_result(output_tokens=5, timestamps=[])
    assert r.inter_token_latencies == []
    assert r.mean_tbt == 0.0
