# benchpress — Backlog

> Speed + quality benchmarking for LLMs on Apple Silicon.
> GitHub: `github.com/wfullen/benchpress`

---

## Completed

| Phase | Description |
|-------|-------------|
| 1 | Speed foundation — tokens/sec, TTFT, bootstrap CI |
| 2 | Quality foundation — perplexity + task accuracy (MMLU, HellaSwag, TruthfulQA) |
| 3 | Statistical rigor — Wilcoxon/Holm-Bonferroni, thermal throttling detection (Mann-Kendall) |
| 4 | Leaderboard — `benchpress submit`, JSON schema v1, `results/` directory, `benchpress leaderboard` command |
| 5 | Multi-backend apples-to-apples — `benchpress compare-backends`, normalized settings, pairwise significance, per-backend caveats |
| 6 | Quantization Pareto frontier — `benchpress sweep`, speed vs perplexity plot, GGUF auto-download |
| 7 | GitHub Actions CI + GitHub Pages leaderboard |

---

## Phase 5 — Quantization Pareto frontier ✅ (renumbered → Phase 6)

Sweep Q2–Q8 GGUF quantizations of the same base model, plot speed vs quality tradeoff curve. Answers: "what's the best quality I can get at N tokens/sec?"

- [ ] Auto-download GGUF variants from HuggingFace if available
- [ ] Run speed + perplexity at each quantization level
- [ ] Output Pareto frontier plot (speed vs perplexity) as PNG + JSON
- [ ] `benchpress sweep model-name --quants q4,q5,q6,q8`

---

## Phase 6 — Model coverage leaderboard

Standardized results across popular models (Llama 3, Mistral, Gemma, Phi, Qwen) so the community can compare without running everything themselves.

- [ ] Define canonical benchmark config (prompts, n_runs, max_tokens) — freeze as `v1`
- [ ] JSON schema for leaderboard entries: model, backend, hardware, scores
- [ ] `benchpress submit` command that formats results for leaderboard PR
- [ ] GitHub Pages table rendered from `results/` JSON files

---

## Phase 7 — Reporting & CI

- [ ] `--output report.md` generates a full Markdown report with tables and plots
- [ ] GitHub Actions workflow: run benchmark on push, comment results on PR
- [ ] `benchpress diff baseline.json current.json` — shows regression/improvement with significance test

---

## Phase 8 — Distribution

- [ ] PyPI package (`pip install benchpress-llm` or similar)
- [ ] Homebrew formula
- [ ] `benchpress leaderboard` command that fetches latest results from GitHub Pages

---

## Parked

- **Energy/watt efficiency** — tokens per joule using powermetrics; good idea, complex to get right on Apple Silicon
- **Batch inference mode** — throughput at batch_size > 1; less relevant for local use cases
- **Windows/CUDA path** — significant work, dilutes the Apple Silicon focus
