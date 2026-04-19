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

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *_) -> None:
        self.unload()
