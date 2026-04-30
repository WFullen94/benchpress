"""Quantization sweep: run speed + perplexity across quant levels, plot Pareto frontier."""

from __future__ import annotations

import json
import datetime
from dataclasses import dataclass, field
from typing import Callable

from benchpress.backends.base import Backend


STANDARD_QUANTS = ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]


@dataclass
class QuantResult:
    quant: str
    tokens_per_second: float
    tps_ci: tuple[float, float]
    perplexity: float | None
    model_size_gb: float | None = None
    error: str | None = None


@dataclass
class SweepResult:
    base_model: str
    backend: str
    hardware: str
    timestamp: str
    results: list[QuantResult] = field(default_factory=list)

    def pareto_points(self) -> list[QuantResult]:
        """Return results on the Pareto frontier (higher tps AND lower perplexity)."""
        valid = [r for r in self.results if r.error is None and r.perplexity is not None]
        if not valid:
            return []
        pareto = []
        for r in sorted(valid, key=lambda x: x.tokens_per_second, reverse=True):
            if not pareto or r.perplexity < pareto[-1].perplexity:
                pareto.append(r)
        return pareto

    def to_dict(self) -> dict:
        return {
            "base_model": self.base_model,
            "backend": self.backend,
            "hardware": self.hardware,
            "timestamp": self.timestamp,
            "results": [
                {
                    "quant": r.quant,
                    "tokens_per_second": round(r.tokens_per_second, 2),
                    "tps_ci": [round(r.tps_ci[0], 2), round(r.tps_ci[1], 2)],
                    "perplexity": round(r.perplexity, 3) if r.perplexity else None,
                    "model_size_gb": round(r.model_size_gb, 2) if r.model_size_gb else None,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


def run_sweep(
    repo: str,
    quants: list[str],
    n_runs: int = 3,
    warmup_runs: int = 1,
    max_tokens: int = 256,
    n_ppl_docs: int = 10,
    progress_cb: Callable[[str, str], None] | None = None,
) -> SweepResult:
    """Run speed + perplexity benchmarks across quant levels for a HuggingFace GGUF repo.

    Args:
        repo: HuggingFace repo ID (e.g. 'bartowski/Llama-3.2-3B-Instruct-GGUF')
        quants: list of quant tags to sweep (e.g. ['Q4_K_M', 'Q6_K', 'Q8_0'])
        n_runs: benchmark runs per quant (fewer than normal — sweep is long)
        n_ppl_docs: WikiText-2 docs for perplexity (fewer = faster)
    """
    from benchpress.backends.llamacpp import LlamaCppBackend
    from benchpress.runner import run_benchmark, DEFAULT_PROMPTS
    from benchpress.quality.perplexity import compute_perplexity
    from benchpress.hardware import hardware_summary

    hw = hardware_summary()
    sweep = SweepResult(
        base_model=repo,
        backend="llamacpp",
        hardware=f"{hw['chip']} · {hw['memory_gb']:.0f} GB",
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
    )

    for quant in quants:
        model_id = f"{repo}:{quant}"
        if progress_cb:
            progress_cb(quant, "loading")

        backend = LlamaCppBackend()
        try:
            backend.load(model_id)
        except Exception as e:
            sweep.results.append(QuantResult(
                quant=quant, tokens_per_second=0.0,
                tps_ci=(0.0, 0.0), perplexity=None, error=str(e),
            ))
            continue

        # Speed
        if progress_cb:
            progress_cb(quant, "benchmarking speed")
        try:
            stats = run_benchmark(
                backend=backend,
                model=model_id,
                n_runs=n_runs,
                warmup_runs=warmup_runs,
                max_tokens=max_tokens,
            )
            tps = stats.mean_tps
            tps_ci = stats.tps_ci
        except Exception as e:
            backend.unload()
            sweep.results.append(QuantResult(
                quant=quant, tokens_per_second=0.0,
                tps_ci=(0.0, 0.0), perplexity=None, error=f"speed error: {e}",
            ))
            continue

        # Perplexity
        if progress_cb:
            progress_cb(quant, "computing perplexity")
        ppl: float | None = None
        try:
            ppl = compute_perplexity(backend, n_docs=n_ppl_docs)
        except (NotImplementedError, ImportError, Exception):
            pass

        backend.unload()
        sweep.results.append(QuantResult(
            quant=quant,
            tokens_per_second=tps,
            tps_ci=tps_ci,
            perplexity=ppl,
        ))

    return sweep


def plot_pareto(sweep: SweepResult, output_path: str) -> None:
    """Save a Pareto frontier plot (tokens/sec vs perplexity) as a PNG."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    valid = [r for r in sweep.results if r.error is None and r.perplexity is not None]
    pareto = sweep.pareto_points()
    pareto_quants = {r.quant for r in pareto}

    if not valid:
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    for r in valid:
        on_frontier = r.quant in pareto_quants
        color = "#2563eb" if on_frontier else "#94a3b8"
        marker = "D" if on_frontier else "o"
        size = 100 if on_frontier else 60
        ax.scatter(r.tokens_per_second, r.perplexity, c=color, s=size,
                   marker=marker, zorder=3)
        ax.annotate(
            r.quant,
            (r.tokens_per_second, r.perplexity),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
            color="#1e293b",
        )
        # CI whiskers on x-axis
        ax.errorbar(
            r.tokens_per_second, r.perplexity,
            xerr=[[r.tokens_per_second - r.tps_ci[0]],
                  [r.tps_ci[1] - r.tokens_per_second]],
            fmt="none", ecolor=color, alpha=0.4, capsize=3,
        )

    # Connect Pareto points
    if len(pareto) > 1:
        px = [r.tokens_per_second for r in pareto]
        py = [r.perplexity for r in pareto]
        ax.plot(px, py, "--", color="#2563eb", alpha=0.5, linewidth=1.5, zorder=2)

    ax.set_xlabel("Tokens / sec (higher → faster)", fontsize=11)
    ax.set_ylabel("Perplexity on WikiText-2 (lower → better quality)", fontsize=11)
    ax.set_title(
        f"Quantization Pareto Frontier\n{sweep.base_model}  ·  {sweep.hardware}",
        fontsize=12,
    )
    ax.grid(True, alpha=0.25, linestyle="--")

    frontier_patch = mpatches.Patch(color="#2563eb", label="Pareto frontier")
    dominated_patch = mpatches.Patch(color="#94a3b8", label="Dominated")
    ax.legend(handles=[frontier_patch, dominated_patch], fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def render_sweep_table(sweep: SweepResult) -> None:
    """Print a rich terminal table of sweep results."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()
    pareto_quants = {r.quant for r in sweep.pareto_points()}

    table = Table(box=box.SIMPLE_HEAD, header_style="bold",
                  title=f"Quantization sweep · {sweep.base_model}")
    table.add_column("Quant")
    table.add_column("tok/s", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Perplexity ↓", justify="right")
    table.add_column("Pareto", justify="center")
    table.add_column("Note")

    for r in sweep.results:
        if r.error:
            table.add_row(r.quant, "—", "—", "—", "—", f"[red]{r.error[:40]}[/]")
            continue
        on_frontier = r.quant in pareto_quants
        table.add_row(
            f"[bold]{r.quant}[/]" if on_frontier else r.quant,
            f"{r.tokens_per_second:.1f}",
            f"[{r.tps_ci[0]:.1f}, {r.tps_ci[1]:.1f}]",
            f"{r.perplexity:.2f}" if r.perplexity else "—",
            "[green]✓[/]" if on_frontier else "",
            "",
        )

    console.print()
    console.print(table)
    console.print(
        f"  [dim]Pareto frontier: highest speed without sacrificing quality. "
        f"{len(sweep.results)} quants tested · {sweep.hardware}[/]\n"
    )
