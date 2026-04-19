"""Ollama backend — calls the local Ollama REST API with streaming."""

from __future__ import annotations

import time

from benchpress.backends.base import Backend, BackendError
from benchpress.metrics import InferenceResult


class OllamaBackend(Backend):
    """Streams tokens from a running Ollama instance."""

    name = "ollama"

    def __init__(self, host: str = "http://localhost:11434") -> None:
        self._host = host
        self._model_id = ""

    def load(self, model: str, **kwargs) -> None:
        try:
            import ollama as _ollama
        except ImportError:
            raise BackendError(
                "ollama Python client not installed. Run: pip install ollama"
            )
        self._ollama = _ollama
        self._model_id = model
        # Warm-up pull check — ollama pull is idempotent
        try:
            self._ollama.chat(
                model=model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
            )
        except Exception as e:
            raise BackendError(
                f"Cannot reach Ollama or model '{model}' not pulled: {e}"
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> InferenceResult:
        token_timestamps: list[float] = []
        output_tokens = 0
        first_token_time = 0.0
        output_text = ""
        prompt_tokens = 0

        t_start = time.perf_counter()
        stream = self._ollama.chat(
            model=self._model_id,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"num_predict": max_tokens, "temperature": temperature},
        )
        for chunk in stream:
            now = time.perf_counter()
            msg = chunk.get("message", {})
            token_text = msg.get("content", "")
            if token_text:
                output_tokens += 1
                token_timestamps.append(now)
                output_text += token_text
                if output_tokens == 1:
                    first_token_time = now - t_start
            # Final chunk has eval_count
            if chunk.get("done"):
                prompt_tokens = chunk.get("prompt_eval_count", 0)

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

    def unload(self) -> None:
        pass
