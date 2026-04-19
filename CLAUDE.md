# benchpress

LLM inference benchmark for Apple Silicon — combines speed + quality with statistical rigor.

**GitHub:** `github.com/wfullen/benchpress`
**Part of:** Wayne's tooling portfolio. Full backlog at `~/computer-vision-notebooks/BACKLOG.md`.

## What this project does

Benchmarks LLMs running locally on Apple Silicon (M-series) across two dimensions simultaneously:
- **Speed:** tokens/sec, time-to-first-token, end-to-end latency — with bootstrap confidence intervals
- **Quality:** perplexity, task accuracy (MMLU subset, HellaSwag, TruthfulQA)

The gap this fills: MLPerf targets datacenters, llm-benchmark measures speed only, LM-Eval measures quality only. Nobody combines both with statistical rigor on consumer hardware.

## Phased Roadmap

- **Phase 1 — Speed Foundation:** ollama backend, tokens/sec + TTFT, bootstrap CI, CLI (`benchpress run`)
- **Phase 2 — Quality Foundation:** perplexity on WikiText-2, task accuracy probes, combined quality score
- **Phase 3 — Statistical Rigor:** pairwise significance testing, thermal throttling detection
- **Phase 4 — Multi-Backend:** ollama, llama.cpp, MLX, transformers+MPS — apples-to-apples
- **Phase 5 — Quantization Sweep:** Q8/Q6/Q4/Q3/Q2 Pareto frontier (merges with `lossy`)
- **Phase 6 — Model Coverage:** Llama, Mistral, Gemma, Phi, Qwen families
- **Phase 7 — Reporting:** Markdown + JSON output, `benchpress compare`, GitHub Actions
- **Phase 8 — Distribution:** PyPI, GitHub Pages leaderboard

## Conventions

- Conventional commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`
- Tag each phase completion: `v0.1.0`, `v0.2.0`, etc.
- Each commit should be a working state
- MPS device support: `torch.backends.mps.is_available()`
- Target hardware: M3 Max (also test paths for M1/M2)

## Current Status

Phase 1 complete. Core structure in place:
- `src/benchpress/` package installed as CLI (`benchpress run`, `benchpress compare`, `benchpress backends`)
- Backends: mlx, ollama, transformers (all lazy-imported; fail gracefully if not installed)
- Metrics: TTFT, TBT, tokens/sec with bootstrap 95% CIs
- Stats: bootstrap CI (percentile), Mann-Whitney U significance test, Cohen's d
- Report: rich terminal table, JSON output (`--output`), Markdown (`--markdown`)
- 14 unit tests passing

To run a benchmark, first install a backend:
```
pip install mlx-lm          # recommended for M-series
# or: install Ollama.app + pip install ollama
```
Then:
```
benchpress run mlx-community/Llama-3.2-3B-Instruct-4bit --backend mlx
benchpress compare model-a model-b --backend mlx
```

Next: Phase 2 — quality evaluation (perplexity on WikiText-2 + task accuracy probes).
