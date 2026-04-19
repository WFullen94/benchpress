"""benchpress CLI entry point."""

from __future__ import annotations

import sys
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from benchpress.metrics import InferenceResult

console = Console()


@click.group()
@click.version_option()
def main():
    """benchpress — LLM inference benchmark for Apple Silicon."""


@main.command()
@click.argument("model")
@click.option(
    "--backend", "-b",
    default="mlx",
    type=click.Choice(["mlx", "ollama", "transformers"], case_sensitive=False),
    show_default=True,
    help="Inference backend to use.",
)
@click.option("--runs", "-n", default=5, show_default=True, help="Benchmark runs per prompt.")
@click.option("--warmup", default=1, show_default=True, help="Warmup runs (discarded).")
@click.option("--max-tokens", default=256, show_default=True, help="Max tokens to generate.")
@click.option("--temperature", default=0.0, show_default=True, help="Sampling temperature.")
@click.option(
    "--prompts", "prompts_file",
    type=click.Path(exists=True),
    default=None,
    help="Path to a file with one prompt per line.",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Save JSON results to file.")
@click.option("--markdown", is_flag=True, help="Print Markdown table instead of rich output.")
def run(model, backend, runs, warmup, max_tokens, temperature, prompts_file, output, markdown):
    """Benchmark MODEL and print speed metrics with confidence intervals."""
    from benchpress.backends import get_backend, BackendError
    from benchpress.runner import run_benchmark, DEFAULT_PROMPTS
    from benchpress.report import print_result, to_json, to_markdown

    prompts = None
    if prompts_file:
        with open(prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

    try:
        b = get_backend(backend)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        sys.exit(1)

    console.print(f"\n[bold]Loading[/] [yellow]{model}[/] via [cyan]{backend}[/]…")
    try:
        b.load(model)
    except BackendError as e:
        console.print(f"[red]Load failed:[/] {e}")
        sys.exit(1)

    total_iters = (warmup + runs) * len(prompts or DEFAULT_PROMPTS)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking…", total=total_iters)
        completed_runs = [0]

        def on_progress(done: int, total: int, result: InferenceResult) -> None:
            progress.update(task, completed=done)

        stats = run_benchmark(
            backend=b,
            model=model,
            prompts=prompts,
            n_runs=runs,
            warmup_runs=warmup,
            max_tokens=max_tokens,
            temperature=temperature,
            progress_cb=on_progress,
        )

    b.unload()

    if markdown:
        click.echo(to_markdown(stats))
    else:
        print_result(stats)

    if output:
        with open(output, "w") as f:
            f.write(to_json(stats))
        console.print(f"[dim]Results saved to {output}[/]")


@main.command()
@click.argument("model_a")
@click.argument("model_b")
@click.option(
    "--backend", "-b",
    default="mlx",
    type=click.Choice(["mlx", "ollama", "transformers"], case_sensitive=False),
    show_default=True,
)
@click.option("--runs", "-n", default=5, show_default=True)
@click.option("--warmup", default=1, show_default=True)
@click.option("--max-tokens", default=256, show_default=True)
@click.option(
    "--backend-b", default=None,
    type=click.Choice(["mlx", "ollama", "transformers"], case_sensitive=False),
    help="Override backend for MODEL_B (default: same as --backend).",
)
def compare(model_a, model_b, backend, runs, warmup, max_tokens, backend_b):
    """Benchmark MODEL_A and MODEL_B and show a side-by-side significance test."""
    from benchpress.backends import get_backend, BackendError
    from benchpress.runner import run_benchmark
    from benchpress.report import print_result, print_comparison

    results = []
    pairs = [
        (model_a, backend),
        (model_b, backend_b or backend),
    ]

    for model, bname in pairs:
        console.print(f"\n[bold]Loading[/] [yellow]{model}[/] via [cyan]{bname}[/]…")
        b = get_backend(bname)
        try:
            b.load(model)
        except BackendError as e:
            console.print(f"[red]Load failed:[/] {e}")
            sys.exit(1)

        from benchpress.runner import DEFAULT_PROMPTS
        total = (warmup + runs) * len(DEFAULT_PROMPTS)

        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
            task = progress.add_task(f"Benchmarking {model}…", total=total)

            stats = run_benchmark(
                backend=b,
                model=model,
                n_runs=runs,
                warmup_runs=warmup,
                max_tokens=max_tokens,
                progress_cb=lambda done, total, r: progress.update(task, completed=done),
            )
        b.unload()
        results.append(stats)
        print_result(stats)

    print_comparison(results[0], results[1])


@main.command()
def backends():
    """List available backends and their installation status."""
    from rich.table import Table
    from rich import box

    checks = {
        "mlx": ("mlx_lm", "pip install mlx-lm"),
        "ollama": ("ollama", "pip install ollama  (also install Ollama.app)"),
        "transformers": ("transformers", "pip install torch transformers"),
    }
    table = Table(box=box.SIMPLE_HEAD, header_style="bold")
    table.add_column("Backend")
    table.add_column("Status")
    table.add_column("Install")

    for name, (pkg, install) in checks.items():
        try:
            __import__(pkg)
            status = "[green]available[/]"
        except ImportError:
            status = "[red]not installed[/]"
        table.add_row(name, status, install)

    console.print()
    console.print(table)
