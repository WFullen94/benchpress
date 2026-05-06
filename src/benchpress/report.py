"""Output formatting: rich terminal tables, JSON, Markdown."""

from __future__ import annotations

import json

from benchpress.metrics import BenchmarkStats
from benchpress.stats import mannwhitney_p, wilcoxon_p, effect_size_cohen_d, sig_stars, thermal_trend, holm_bonferroni


def _fmt_ci(mean: float, ci: tuple[float, float], unit: str = "") -> str:
    return f"{mean:.2f}{unit}  [{ci[0]:.2f}, {ci[1]:.2f}]"


def print_result(stats: BenchmarkStats) -> None:
    """Rich terminal output for a single benchmark result."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print()
    console.print(f"[bold cyan]benchpress[/] · {stats.model}", highlight=False)
    console.print(
        f"  backend: [yellow]{stats.backend}[/]  "
        f"hardware: [green]{stats.hardware}[/]  "
        f"runs: {stats.n_runs}",
        highlight=False,
    )
    console.print()

    table = Table(box=box.SIMPLE_HEAD, show_header=True, header_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("95% CI", justify="right")

    table.add_row(
        "Tokens / sec",
        f"{stats.mean_tps:.1f}",
        f"[{stats.tps_ci[0]:.1f}, {stats.tps_ci[1]:.1f}]",
    )
    table.add_row(
        "TTFT (s)",
        f"{stats.mean_ttft:.3f}",
        f"[{stats.ttft_ci[0]:.3f}, {stats.ttft_ci[1]:.3f}]",
    )
    table.add_row(
        "Latency (s)",
        f"{stats.mean_latency:.2f}",
        f"[{stats.latency_ci[0]:.2f}, {stats.latency_ci[1]:.2f}]",
    )

    console.print(table)
    console.print(f"  [dim]{stats.timestamp}[/]")

    trend = thermal_trend(stats.tps_samples)
    if trend["throttling"]:
        console.print(
            f"  [yellow]⚠  Thermal throttling detected[/] "
            f"(τ = {trend['tau']:.2f}, p = {trend['p_value']:.3f})\n"
            f"  [dim]Tokens/sec declined across runs. "
            f"Consider --cooldown or fewer --runs.[/]"
        )
    console.print()


def print_comparison(a: BenchmarkStats, b: BenchmarkStats) -> None:
    """Side-by-side comparison with significance test."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print()
    console.print(
        f"[bold cyan]benchpress compare[/]  "
        f"[yellow]{a.model}[/] vs [yellow]{b.model}[/]",
        highlight=False,
    )
    console.print()

    # Use paired Wilcoxon when same prompt count (same prompts, same order)
    paired = len(a.tps_samples) == len(b.tps_samples)
    test_fn = wilcoxon_p if paired else mannwhitney_p
    test_name = "Wilcoxon signed-rank" if paired else "Mann-Whitney U"

    raw_p_tps = test_fn(a.tps_samples, b.tps_samples)
    raw_p_ttft = test_fn(a.ttft_samples, b.ttft_samples)
    p_tps, p_ttft = holm_bonferroni([raw_p_tps, raw_p_ttft])
    d_tps = effect_size_cohen_d(a.tps_samples, b.tps_samples)

    table = Table(box=box.SIMPLE_HEAD, header_style="bold")
    table.add_column("Metric")
    table.add_column(a.model, justify="right")
    table.add_column(b.model, justify="right")
    table.add_column("p (adj.)", justify="right")
    table.add_column("sig", justify="right")
    table.add_column("Cohen d", justify="right")

    table.add_row(
        "Tokens / sec",
        f"{a.mean_tps:.1f}",
        f"{b.mean_tps:.1f}",
        f"{p_tps:.3f}",
        sig_stars(p_tps),
        f"{d_tps:.2f}",
    )
    table.add_row(
        "TTFT (s)",
        f"{a.mean_ttft:.3f}",
        f"{b.mean_ttft:.3f}",
        f"{p_ttft:.3f}",
        sig_stars(p_ttft),
        "—",
    )
    table.add_row(
        "Latency (s)",
        f"{a.mean_latency:.2f}",
        f"{b.mean_latency:.2f}",
        "—",
        "—",
        "—",
    )
    console.print(table)
    console.print(
        f"  [dim]p-values: {test_name}, two-sided, Holm-Bonferroni adjusted. "
        f"Cohen d: |d|≥0.2 small, ≥0.5 medium, ≥0.8 large.[/]"
    )
    console.print()


_BACKEND_CAVEATS = {
    "mlx":          "Apple Silicon only · MLX format models · logprob access (Tier 1)",
    "ollama":       "Any platform · auto-download · no logprob access (Tier 3)",
    "llamacpp":     "GGUF format · Metal-accelerated on Apple Silicon",
    "transformers": "Any platform · HuggingFace IDs · logprob access (Tier 1) · slowest",
}


def print_backend_comparison(results: list[BenchmarkStats]) -> None:
    """Multi-column backend comparison table with pairwise significance tests."""
    from itertools import combinations
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    console.print()
    backends = [s.backend for s in results]
    console.print(
        f"[bold cyan]benchpress compare-backends[/]  "
        f"[yellow]{results[0].model}[/]  ·  "
        f"backends: [cyan]{', '.join(backends)}[/]",
        highlight=False,
    )
    console.print(
        "  [dim]Normalized: temperature=0.0, same prompts across all backends.[/]\n"
    )

    # --- Main metrics table ---
    table = Table(box=box.SIMPLE_HEAD, header_style="bold")
    table.add_column("Metric", style="bold")
    for s in results:
        table.add_column(s.backend, justify="right")

    tps_row = ["Tokens / sec"] + [f"{s.mean_tps:.1f}  [{s.tps_ci[0]:.1f}, {s.tps_ci[1]:.1f}]" for s in results]
    ttft_row = ["TTFT (s)"] + [f"{s.mean_ttft:.3f}  [{s.ttft_ci[0]:.3f}, {s.ttft_ci[1]:.3f}]" for s in results]
    lat_row = ["Latency (s)"] + [f"{s.mean_latency:.2f}" for s in results]
    table.add_row(*tps_row)
    table.add_row(*ttft_row)
    table.add_row(*lat_row)
    console.print(table)

    # --- Pairwise significance table ---
    pairs = list(combinations(range(len(results)), 2))
    if pairs:
        all_raw_p = []
        for i, j in pairs:
            raw_p = mannwhitney_p(results[i].tps_samples, results[j].tps_samples)
            all_raw_p.append(raw_p)
        adj_p = holm_bonferroni(all_raw_p)

        sig_table = Table(box=box.SIMPLE_HEAD, header_style="bold", title="Pairwise significance (tokens/sec)")
        sig_table.add_column("Pair")
        sig_table.add_column("Faster", justify="right")
        sig_table.add_column("Δ tps", justify="right")
        sig_table.add_column("p (adj.)", justify="right")
        sig_table.add_column("sig", justify="right")
        sig_table.add_column("Cohen d", justify="right")

        for (i, j), p in zip(pairs, adj_p):
            a, b = results[i], results[j]
            faster = a.backend if a.mean_tps > b.mean_tps else b.backend
            delta = abs(a.mean_tps - b.mean_tps)
            d = effect_size_cohen_d(a.tps_samples, b.tps_samples)
            sig_table.add_row(
                f"{a.backend} vs {b.backend}",
                faster,
                f"+{delta:.1f}",
                f"{p:.3f}",
                sig_stars(p),
                f"{d:.2f}",
            )
        console.print(sig_table)
        console.print("  [dim]Mann-Whitney U, two-sided, Holm-Bonferroni adjusted.[/]")

    # --- Caveats ---
    console.print("\n  [bold]Backend notes:[/]")
    for s in results:
        caveat = _BACKEND_CAVEATS.get(s.backend, "")
        if caveat:
            console.print(f"  [cyan]{s.backend:<14}[/] {caveat}")

    # --- Throttling warnings ---
    for s in results:
        trend = thermal_trend(s.tps_samples)
        if trend["throttling"]:
            console.print(
                f"\n  [yellow]⚠  {s.backend}: thermal throttling detected[/] "
                f"(τ={trend['tau']:.2f}, p={trend['p_value']:.3f})"
            )
    console.print()


def to_json(stats: BenchmarkStats) -> str:
    return json.dumps(
        {
            "model": stats.model,
            "backend": stats.backend,
            "hardware": stats.hardware,
            "timestamp": stats.timestamp,
            "n_runs": stats.n_runs,
            "tokens_per_second": {
                "mean": round(stats.mean_tps, 2),
                "ci_lower": round(stats.tps_ci[0], 2),
                "ci_upper": round(stats.tps_ci[1], 2),
                "samples": [round(v, 2) for v in stats.tps_samples],
            },
            "ttft_seconds": {
                "mean": round(stats.mean_ttft, 4),
                "ci_lower": round(stats.ttft_ci[0], 4),
                "ci_upper": round(stats.ttft_ci[1], 4),
                "samples": [round(v, 4) for v in stats.ttft_samples],
            },
            "latency_seconds": {
                "mean": round(stats.mean_latency, 3),
                "ci_lower": round(stats.latency_ci[0], 3),
                "ci_upper": round(stats.latency_ci[1], 3),
                "samples": [round(v, 3) for v in stats.latency_samples],
            },
        },
        indent=2,
    )


def to_markdown(stats: BenchmarkStats) -> str:
    lines = [
        f"## {stats.model} — {stats.backend}",
        f"",
        f"**Hardware:** {stats.hardware}  ",
        f"**Runs:** {stats.n_runs}  ",
        f"**Timestamp:** {stats.timestamp}",
        f"",
        f"| Metric | Mean | 95% CI |",
        f"|--------|-----:|-------:|",
        f"| Tokens / sec | {stats.mean_tps:.1f} | [{stats.tps_ci[0]:.1f}, {stats.tps_ci[1]:.1f}] |",
        f"| TTFT (s) | {stats.mean_ttft:.3f} | [{stats.ttft_ci[0]:.3f}, {stats.ttft_ci[1]:.3f}] |",
        f"| Latency (s) | {stats.mean_latency:.2f} | [{stats.latency_ci[0]:.2f}, {stats.latency_ci[1]:.2f}] |",
    ]
    return "\n".join(lines)
