"""Core metric types for LLM inference benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class TokenEvent:
    """Timestamp of a single generated token."""
    token: str
    timestamp: float  # seconds since epoch


@dataclass
class InferenceResult:
    """Raw timing data from a single inference run."""
    prompt: str
    output: str
    prompt_tokens: int
    output_tokens: int
    first_token_time: float   # seconds
    total_time: float         # seconds
    token_timestamps: list[float] = field(default_factory=list)

    @property
    def ttft(self) -> float:
        """Time to first token in seconds."""
        return self.first_token_time

    @property
    def tokens_per_second(self) -> float:
        if self.total_time <= 0 or self.output_tokens <= 1:
            return 0.0
        # Exclude TTFT: measure decode throughput only
        decode_time = self.total_time - self.first_token_time
        if decode_time <= 0:
            return 0.0
        return (self.output_tokens - 1) / decode_time

    @property
    def inter_token_latencies(self) -> list[float]:
        """Time between consecutive tokens (seconds)."""
        if len(self.token_timestamps) < 2:
            return []
        return [
            self.token_timestamps[i] - self.token_timestamps[i - 1]
            for i in range(1, len(self.token_timestamps))
        ]

    @property
    def mean_tbt(self) -> float:
        """Mean time between tokens."""
        itls = self.inter_token_latencies
        return sum(itls) / len(itls) if itls else 0.0


@dataclass
class BenchmarkStats:
    """Aggregated statistics over multiple inference runs."""
    model: str
    backend: str
    n_runs: int
    prompt_tokens: int

    # Point estimates
    mean_ttft: float
    mean_tps: float
    mean_latency: float

    # Bootstrap confidence intervals (95%)
    ttft_ci: tuple[float, float]
    tps_ci: tuple[float, float]
    latency_ci: tuple[float, float]

    # Raw samples for downstream analysis
    ttft_samples: list[float] = field(default_factory=list)
    tps_samples: list[float] = field(default_factory=list)
    latency_samples: list[float] = field(default_factory=list)

    hardware: str = ""
    timestamp: str = ""


class Timer:
    """Context manager for precise wall-clock timing."""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.end: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return self.end - self.start
