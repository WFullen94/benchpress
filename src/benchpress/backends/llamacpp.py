"""llama.cpp backend — Metal-accelerated inference via llama-cpp-python."""

from __future__ import annotations

import time

from benchpress.backends.base import Backend, BackendError
from benchpress.metrics import InferenceResult


class LlamaCppBackend(Backend):
    """Wraps llama_cpp.Llama with token-level timing.

    Expects a local GGUF file path or a HuggingFace repo/filename pair:
        local:  /path/to/model.gguf
        hub:    bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
    """

    name = "llamacpp"

    def __init__(self) -> None:
        self._llm = None
        self._model_id = ""

    def load(self, model: str, **kwargs) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise BackendError(
                "llama-cpp-python not installed.\n"
                "  Apple Silicon: CMAKE_ARGS='-DGGML_METAL=on' "
                "pip install llama-cpp-python --no-cache-dir"
            )

        self._model_id = model

        if ":" in model and not model.startswith("/"):
            # HuggingFace hub: "repo/name:filename_or_quant"
            repo, filename_or_quant = model.rsplit(":", 1)
            # If it looks like a quant tag (Q4_K_M etc.) resolve the filename
            if not filename_or_quant.endswith(".gguf"):
                filename_or_quant = self._resolve_gguf_filename(repo, filename_or_quant)
            self._llm = Llama.from_pretrained(
                repo_id=repo,
                filename=filename_or_quant,
                n_gpu_layers=-1,   # offload all layers to Metal
                verbose=False,
                **kwargs,
            )
        else:
            # Local GGUF file path
            self._llm = Llama(
                model_path=model,
                n_gpu_layers=-1,
                verbose=False,
                **kwargs,
            )

    def _resolve_gguf_filename(self, repo: str, quant_tag: str) -> str:
        """Find the GGUF filename in a HuggingFace repo matching a quant tag."""
        try:
            from huggingface_hub import list_repo_files
            for fname in list_repo_files(repo):
                if fname.endswith(".gguf") and quant_tag.upper() in fname.upper():
                    return fname
        except Exception:
            pass
        # Fall back to the tag itself as a glob pattern
        return f"*{quant_tag}*.gguf"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> InferenceResult:
        if self._llm is None:
            raise BackendError("Call load() before generate()")

        # Count prompt tokens
        prompt_tokens = len(self._llm.tokenize(prompt.encode()))

        token_timestamps: list[float] = []
        output_tokens = 0
        first_token_time = 0.0
        output_text = ""

        t_start = time.perf_counter()

        for chunk in self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        ):
            now = time.perf_counter()
            token_timestamps.append(now)
            output_tokens += 1
            if output_tokens == 1:
                first_token_time = now - t_start
            output_text += chunk["choices"][0]["text"]

        total_time = time.perf_counter() - t_start

        return InferenceResult(
            prompt=prompt,
            output=output_text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            first_token_time=first_token_time,
            total_time=total_time,
            token_timestamps=token_timestamps,
        )

    def perplexity_of(self, text: str) -> float:
        """Perplexity via llama.cpp's built-in log-likelihood scoring."""
        if self._llm is None:
            raise BackendError("Call load() before perplexity_of()")

        import math
        import numpy as np

        tokens = self._llm.tokenize(text.encode())
        if len(tokens) < 2:
            raise ValueError("Text too short for perplexity (need ≥2 tokens).")

        # Evaluate the model to get logits, then compute cross-entropy
        self._llm.reset()
        self._llm.eval(tokens)

        # llama_cpp exposes scores as (n_tokens, vocab_size) after eval
        scores = self._llm.scores  # numpy array
        if scores is None or len(scores) == 0:
            raise NotImplementedError(
                "llama-cpp-python build does not expose logits. "
                "Recompile with LLAMA_LOGITS_ALL=1 or use --no-perplexity."
            )

        log_probs = []
        for i, next_tok in enumerate(tokens[1:]):
            logits = scores[i]
            # Numerically stable log-softmax
            logits = logits - logits.max()
            log_softmax = logits - np.log(np.exp(logits).sum())
            log_probs.append(log_softmax[next_tok])

        return float(math.exp(-sum(log_probs) / len(log_probs)))

    def unload(self) -> None:
        self._llm = None
