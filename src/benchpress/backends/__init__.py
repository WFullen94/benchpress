"""Inference backends for benchpress."""

from benchpress.backends.base import Backend, BackendError

__all__ = ["Backend", "BackendError", "get_backend"]


def get_backend(name: str) -> "Backend":
    name = name.lower()
    if name == "mlx":
        from benchpress.backends.mlx import MLXBackend
        return MLXBackend()
    if name == "ollama":
        from benchpress.backends.ollama import OllamaBackend
        return OllamaBackend()
    if name in ("transformers", "hf"):
        from benchpress.backends.transformers import TransformersBackend
        return TransformersBackend()
    raise ValueError(
        f"Unknown backend '{name}'. Choose: mlx, ollama, transformers"
    )
