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
```

Phase 2 quality evaluation also needs:

```bash
pip install datasets        # HuggingFace datasets (WikiText-2, MMLU, HellaSwag, TruthfulQA)
```

---

## Usage

### Speed benchmark

```bash
benchpress run mlx-community/Llama-3.2-3B-Instruct-4bit --backend mlx
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `mlx` | `mlx`, `ollama`, or `transformers` |
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

### Compare two models

```bash
benchpress compare model-a model-b --backend mlx
```

Reports side-by-side speed metrics with Holm-Bonferroni adjusted p-values, significance stars, and Cohen's d. Uses paired Wilcoxon when run counts match, Mann-Whitney otherwise.

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
| 4 | planned | Multi-backend — llama.cpp, MLX, transformers, apples-to-apples |
| 5 | planned | Quantization sweep — Q2–Q8 Pareto frontier |
| 6 | planned | Model coverage — Llama, Mistral, Gemma, Phi, Qwen |
| 7 | planned | Reporting — Markdown/JSON reports, GitHub Actions |
| 8 | planned | Distribution — PyPI, GitHub Pages leaderboard |

---

## Hardware

Developed and tested on **M3 Max**. Code paths exist for M1/M2 and falls back gracefully to CPU when MPS is unavailable.

---

## License

MIT
