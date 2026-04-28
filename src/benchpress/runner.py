"""Benchmark runner: drives repeated inference runs and computes statistics."""

from __future__ import annotations

import datetime
import time
from typing import Callable

from benchpress.backends.base import Backend
from benchpress.metrics import BenchmarkStats, InferenceResult
from benchpress.stats import bootstrap_ci, summarize
from benchpress.hardware import hardware_summary

DEFAULT_PROMPTS = [
    "Explain the difference between a transformer and an RNN in two sentences.",
    "What are three advantages of Apple Silicon over traditional x86 CPUs?",
    "Write a Python function that computes the nth Fibonacci number iteratively.",
    "Summarize the key ideas behind attention mechanisms in neural networks.",
    "What is the capital of France and what is it known for?",
]


def run_benchmark(
    backend: Backend,
    model: str,
    prompts: list[str] | None = None,
    n_runs: int = 5,
    warmup_runs: int = 1,
    max_tokens: int = 256,
    temperature: float = 0.0,
    cooldown: int = 0,
    progress_cb: Callable[[int, int, InferenceResult], None] | None = None,
) -> BenchmarkStats:
    """
    Run `n_runs` inference passes per prompt and return aggregated stats.

    warmup_runs are discarded to avoid cold-start / JIT compilation bias.
    """
    if prompts is None:
        prompts = DEFAULT_PROMPTS

    all_ttft: list[float] = []
    all_tps: list[float] = []
    all_latency: list[float] = []
    prompt_token_count = 0

    total = (warmup_runs + n_runs) * len(prompts)
    completed = 0

    for prompt in prompts:
        for i in range(warmup_runs + n_runs):
            result = backend.generate(
                prompt, max_tokens=max_tokens, temperature=temperature
            )
            completed += 1
            is_warmup = i < warmup_runs

            if not is_warmup:
                all_ttft.append(result.ttft)
                all_tps.append(result.tokens_per_second)
                all_latency.append(result.total_time)
                prompt_token_count = result.prompt_tokens

            if progress_cb is not None:
                progress_cb(completed, total, result)

            if cooldown > 0 and completed < total:
                time.sleep(cooldown)

    hw = hardware_summary()
    return BenchmarkStats(
        model=model,
        backend=backend.name,
        n_runs=len(all_tps),
        prompt_tokens=prompt_token_count,
        mean_ttft=sum(all_ttft) / len(all_ttft),
        mean_tps=sum(all_tps) / len(all_tps),
        mean_latency=sum(all_latency) / len(all_latency),
        ttft_ci=bootstrap_ci(all_ttft),
        tps_ci=bootstrap_ci(all_tps),
        latency_ci=bootstrap_ci(all_latency),
        ttft_samples=all_ttft,
        tps_samples=all_tps,
        latency_samples=all_latency,
        hardware=f"{hw['chip']} · {hw['memory_gb']:.0f} GB",
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
    )
