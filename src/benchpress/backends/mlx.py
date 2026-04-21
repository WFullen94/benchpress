"""MLX-LM backend — native Apple Silicon inference via mlx-lm."""

from __future__ import annotations

import time

from benchpress.backends.base import Backend, BackendError
from benchpress.metrics import InferenceResult


class MLXBackend(Backend):
    """Wraps mlx_lm.generate() with token-level timing."""

    name = "mlx"

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_id = ""

    def load(self, model: str, **kwargs) -> None:
        try:
            import mlx_lm
        except ImportError:
            raise BackendError(
                "mlx-lm not installed. Run: pip install mlx-lm"
            )
        self._model_id = model
        self._model, self._tokenizer = mlx_lm.load(model)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> InferenceResult:
        if self._model is None:
            raise BackendError("Call load() before generate()")

        import mlx_lm
        import mlx.core as mx

        messages = [{"role": "user", "content": prompt}]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = len(self._tokenizer.encode(formatted))

        token_timestamps: list[float] = []
        output_tokens = 0
        first_token_time = 0.0
        t_start = time.perf_counter()

        # mlx_lm.stream_generate yields (token_str, logprobs) or token_str
        # depending on version — handle both
        output_text = ""
        for chunk in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=max_tokens,
            temp=temperature,
        ):
            now = time.perf_counter()
            token_timestamps.append(now)
            output_tokens += 1
            if output_tokens == 1:
                first_token_time = now - t_start
            # chunk may be a string or (text, metadata) tuple
            if isinstance(chunk, tuple):
                output_text += chunk[0]
            else:
                output_text += chunk

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
        if self._model is None:
            raise BackendError("Call load() before perplexity_of()")

        import mlx.core as mx
        import mlx.nn as nn

        tokens = self._tokenizer.encode(text)
        if len(tokens) < 2:
            raise ValueError("Text too short for perplexity calculation (need ≥2 tokens).")

        tokens_mx = mx.array(tokens)
        logits = self._model(tokens_mx[None, :-1])   # (1, seq-1, vocab)
        loss = nn.losses.cross_entropy(logits[0], tokens_mx[1:], reduction="mean")
        mx.eval(loss)
        return float(mx.exp(loss).item())

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
