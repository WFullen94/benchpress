"""WikiText-2 perplexity scorer."""

from __future__ import annotations

from typing import Callable

from benchpress.backends.base import Backend


def compute_perplexity(
    backend: Backend,
    n_docs: int = 20,
    max_doc_chars: int = 2048,
    progress_cb: Callable[[int, int], None] | None = None,
) -> float:
    """Return mean per-token perplexity over *n_docs* WikiText-2 test documents.

    Requires the backend to implement ``perplexity_of()``.
    Raises ``NotImplementedError`` for backends that don't support logprob access.
    Raises ``ImportError`` if the ``datasets`` package is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "datasets not installed — run: pip install datasets"
        )

    ds = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
        trust_remote_code=False,
    )
    texts = [
        row["text"][:max_doc_chars]
        for row in ds
        if len(row["text"].strip()) > 100
    ][:n_docs]

    if not texts:
        raise ValueError("No suitable WikiText-2 documents found.")

    ppls: list[float] = []
    for i, text in enumerate(texts):
        ppls.append(backend.perplexity_of(text))
        if progress_cb:
            progress_cb(i + 1, len(texts))

    return sum(ppls) / len(ppls)
