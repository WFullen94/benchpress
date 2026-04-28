# benchpress results

Community benchmark results. Each `.json` file is one model run on one machine.

## How to add your results

```bash
# 1. Run the speed benchmark
benchpress run mlx-community/Llama-3.2-3B-Instruct-4bit --backend mlx --output speed.json

# 2. Optionally run quality eval
benchpress quality mlx-community/Llama-3.2-3B-Instruct-4bit --backend mlx --output quality.json

# 3. Package into a leaderboard entry
benchpress submit speed.json --quality quality.json

# 4. Commit the new file in results/ and open a PR
```

## Schema (v1)

```json
{
  "schema_version": "1",
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "backend": "mlx",
  "hardware": "M3 Max · 128 GB",
  "submitted_at": "2026-04-28T12:00:00",
  "speed": {
    "tokens_per_second_mean": 142.3,
    "tokens_per_second_ci": [138.1, 146.5],
    "ttft_mean": 0.124,
    "latency_mean": 1.82
  },
  "quality": {
    "perplexity": 8.4,
    "tasks": [
      {"name": "mmlu", "accuracy": 0.62, "n_correct": 31, "n_total": 50}
    ],
    "quality_score": 0.681
  }
}
```

## Canonical config

Results are only comparable when run with the same config:

| Parameter | Value |
|-----------|-------|
| `--runs` | 5 |
| `--warmup` | 1 |
| `--max-tokens` | 256 |
| `--temperature` | 0.0 |
| `--n-per-task` | 50 |
| `--n-docs` | 20 |
