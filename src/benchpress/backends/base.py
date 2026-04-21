"""Abstract base class for inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from benchpress.metrics import InferenceResult


class BackendError(Exception):
    pass


class Backend(ABC):
    """Common interface all backends must implement."""

    @abstractmethod
    def load(self, model: str, **kwargs) -> None:
        """Load a model. Called once before any generate() calls."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> InferenceResult:
        """Run a single inference pass and return structured timing data."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def perplexity_of(self, text: str) -> float:
        """Return per-token perplexity for *text*.

        Requires direct logprob access. Backends that cannot provide this
        should leave this default, which raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"Backend '{self.name}' does not support perplexity scoring. "
            "Use --no-perplexity or switch to mlx/transformers."
        )

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *_) -> None:
        self.unload()
