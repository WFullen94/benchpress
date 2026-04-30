# benchpress

LLM inference benchmark for Apple Silicon — combines **speed** and **quality** measurement with statistical rigor.

Most benchmarking tools measure one thing: MLPerf targets datacenters, `llm-benchmark` measures speed only, `lm-eval` measures quality only. benchpress does both simultaneously, on consumer M-series hardware.

---

## Features

- **Speed metrics** — tokens/sec, time-to-first-token, end-to-end latency, all with bootstrap 95% confidence intervals
- **Quality metrics** — perplexity on WikiText-2, task accuracy on MMLU / HellaSwag / TruthfulQA, composite quality score
- **Statistical rigour** — paired Wilcoxon / Mann-Whitney U significance testing, Holm-Bonferroni correction for multiple comparisons, Cohen's d effect size, thermal throttling detection (Mann-Kendall trend test)
- **Multiple backends** — MLX (recommended for M-series), Ollama, HuggingFace Transformers + MPS
- **Rich terminal output** — tables, progress bars, optional JSON and Markdown export

---

## Installation

```bash
git clone https://github.com/WFullen94/benchpress
cd benchpress
pip install -e .
```

Install at least one inference backend:

```bash
pip install mlx-lm          # recommended for Apple Silicon (M1/M2/M3)
# or: install Ollama.app and run: pip install ollama
# or: pip install torch transformers
# or: CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python --no-cache-dir
```

Phase 2 quality evaluation also needs:

```bash
pip install datasets        # HuggingFace datasets (WikiText-2, MMLU, HellaSwag, TruthfulQA)
```

---

## Leaderboard

Community results on Apple Silicon hardware — live at **[wfullen94.github.io/benchpress](https://wfullen94.github.io/benchpress)**. [Add yours](#contributing-results).

<!-- LEADERBOARD_START -->
| Model | Backend | Hardware | tok/s | TTFT (s) | Perplexity ↓ | Quality ↑ |
|-------|---------|----------|------:|--------:|-------------:|----------:|
| _Your model here_ | — | — | — | — | — | — |
<!-- LEADERBOARD_END -->

_tok/s = mean tokens/sec with 95% bootstrap CI · 5 runs · 256 max tokens · temperature 0.0_

---

## Usage

### Speed benchmark

```bash
benchpress run mlx-community/Llama-3.2-3B-Instruct-4bit --backend mlx
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `mlx` | `mlx`, `ollama`, `transformers`, `llamacpp` |
| `--runs` | `5` | Benchmark iterations per prompt |
| `--warmup` | `1` | Discarded warmup iterations |
| `--max-tokens` | `256` | Max tokens to generate |
| `--output` | — | Save results as JSON |
| `--markdown` | — | Print Markdown table |
| `--cooldown` | `0` | Seconds to wait between runs (reduces thermal throttling) |

### Quality evaluation

```bash
benchpress quality mlx-community/Llama-3.2-3B-Instruct-4bit --backend mlx
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--tasks` | all | `mmlu`, `hellaswag`, `truthfulqa` (repeat flag to select subset) |
| `--n-per-task` | `50` | Examples per task |
| `--perplexity/--no-perplexity` | on | WikiText-2 perplexity (needs logprob access) |
| `--n-docs` | `20` | WikiText-2 documents for perplexity |
| `--output` | — | Save results as JSON |

### Compare two models (or two backends)

```bash
# Same model, same backend
benchpress compare model-a model-b --backend mlx

# Same model, different backends — apples-to-apples
benchpress compare bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M \
    mlx-community/Llama-3.2-3B-Instruct-4bit \
    --backend llamacpp --backend-b mlx
```

Reports side-by-side speed metrics with Holm-Bonferroni adjusted p-values, significance stars, and Cohen's d. Uses paired Wilcoxon when run counts match, Mann-Whitney otherwise.

### Quantization sweep

```bash
benchpress sweep bartowski/Llama-3.2-3B-Instruct-GGUF
```

Downloads and benchmarks Q4\_K\_M, Q5\_K\_M, Q6\_K, Q8\_0 by default. Plots a Pareto frontier of tokens/sec vs perplexity so you can pick the best quality/speed tradeoff for your hardware.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--quants` | `Q4_K_M,Q5_K_M,Q6_K,Q8_0` | Comma-separated quant tags |
| `--runs` | `3` | Speed runs per quant |
| `--ppl-docs` | `10` | WikiText-2 docs for perplexity |
| `--output` | — | Save results as JSON |
| `--plot` | — | Save Pareto frontier as PNG |
| `--no-perplexity` | — | Speed-only sweep |

```bash
# Custom quants + save everything
benchpress sweep bartowski/Llama-3.2-3B-Instruct-GGUF \
    --quants Q2_K,Q4_K_M,Q8_0 \
    --output sweep.json \
    --plot pareto.png
```

### Submit to leaderboard

```bash
benchpress run my-model --output speed.json
benchpress quality my-model --output quality.json   # optional
benchpress submit speed.json --quality quality.json
```

Saves a validated entry to `results/`. Commit it and open a PR to share.

### View local leaderboard

```bash
benchpress leaderboard
benchpress leaderboard --markdown   # for README/CI output
```

### List available backends

```bash
benchpress backends
```

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ done | Speed foundation — tokens/sec, TTFT, bootstrap CI |
| 2 | ✅ done | Quality foundation — perplexity + task accuracy |
| 3 | ✅ done | Statistical rigor — Wilcoxon/Holm-Bonferroni, thermal throttling detection |
| 4 | ✅ done | Leaderboard — `benchpress submit`, JSON schema, community results |
| 5 | ✅ done | Multi-backend — llama.cpp (Metal), apples-to-apples comparison |
| 6 | ✅ done | Quantization sweep — `benchpress sweep`, Pareto frontier plot |
| 7 | planned | GitHub Pages — auto-rendered leaderboard from `results/` |
| 8 | planned | Distribution — PyPI, Homebrew |

---

## Hardware

Developed and tested on **M3 Max**. Code paths exist for M1/M2 and falls back gracefully to CPU when MPS is unavailable.

---

## License

MIT
