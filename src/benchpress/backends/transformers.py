"""HuggingFace Transformers backend with MPS (Apple Silicon) support."""

from __future__ import annotations

import time

from benchpress.backends.base import Backend, BackendError
from benchpress.metrics import InferenceResult


class TransformersBackend(Backend):
    """Uses HuggingFace pipeline with MPS device when available."""

    name = "transformers"

    def __init__(self) -> None:
        self._pipeline = None
        self._tokenizer = None
        self._model_id = ""
        self._device = "cpu"

    def load(self, model: str, **kwargs) -> None:
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
        except ImportError:
            raise BackendError(
                "torch + transformers not installed. Run: pip install torch transformers"
            )

        self._model_id = model
        if torch.backends.mps.is_available():
            self._device = "mps"
        elif torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        dtype = torch.float16 if self._device != "cpu" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=dtype, device_map=self._device
        )
        self._model.eval()
        self._TextIteratorStreamer = TextIteratorStreamer

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> InferenceResult:
        if self._model is None:
            raise BackendError("Call load() before generate()")

        import torch
        import threading

        messages = [{"role": "user", "content": prompt}]
        formatted = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(formatted, return_tensors="pt").to(self._device)
        prompt_tokens = inputs["input_ids"].shape[1]

        streamer = self._TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )

        token_timestamps: list[float] = []
        output_tokens = 0
        first_token_time = 0.0
        output_text = ""

        t_start = time.perf_counter()
        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        for token_str in streamer:
            now = time.perf_counter()
            if token_str:
                output_tokens += 1
                token_timestamps.append(now)
                output_text += token_str
                if output_tokens == 1:
                    first_token_time = now - t_start

        thread.join()
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

        import torch

        inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs, labels=inputs["input_ids"])
        return float(torch.exp(outputs.loss).item())

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        try:
            import torch
            if self._device == "mps":
                torch.mps.empty_cache()
        except Exception:
            pass
