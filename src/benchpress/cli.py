"""benchpress CLI entry point."""

from __future__ import annotations

import sys
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from benchpress.metrics import InferenceResult
from benchpress.backends import BACKENDS

console = Console()

_BACKEND_CHOICE = click.Choice(BACKENDS, case_sensitive=False)


@click.group()
@click.version_option()
def main():
    """benchpress — LLM inference benchmark for Apple Silicon."""


@main.command()
@click.argument("model")
@click.option(
    "--backend", "-b",
    default="mlx",
    type=_BACKEND_CHOICE,
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
@click.option("--cooldown", default=0, show_default=True, help="Seconds to wait between runs (reduces thermal throttling).")
def run(model, backend, runs, warmup, max_tokens, temperature, prompts_file, output, markdown, cooldown):
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
            cooldown=cooldown,
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
    type=_BACKEND_CHOICE,
    show_default=True,
)
@click.option("--runs", "-n", default=5, show_default=True)
@click.option("--warmup", default=1, show_default=True)
@click.option("--max-tokens", default=256, show_default=True)
@click.option(
    "--backend-b", default=None,
    type=_BACKEND_CHOICE,
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
@click.argument("model")
@click.option(
    "--backend", "-b",
    default="mlx",
    type=_BACKEND_CHOICE,
    show_default=True,
    help="Inference backend to use.",
)
@click.option(
    "--tasks", "-t",
    multiple=True,
    default=["mmlu", "hellaswag", "truthfulqa"],
    show_default=True,
    help="Task probes to run. Repeat flag to select a subset.",
)
@click.option("--n-per-task", default=50, show_default=True, help="Examples per task.")
@click.option("--perplexity/--no-perplexity", default=True, show_default=True,
              help="Compute WikiText-2 perplexity (requires logprob access).")
@click.option("--n-docs", default=20, show_default=True, help="WikiText-2 docs for perplexity.")
@click.option("--output", "-o", type=click.Path(), default=None, help="Save JSON results to file.")
def quality(model, backend, tasks, n_per_task, perplexity, n_docs, output):
    """Evaluate MODEL quality: perplexity on WikiText-2 + task accuracy probes."""
    import datetime
    import json

    from benchpress.backends import get_backend, BackendError
    from benchpress.hardware import hardware_summary
    from benchpress.quality import compute_perplexity, run_task_probes
    from benchpress.quality.scorer import QualityResult
    from rich.table import Table
    from rich import box

    try:
        b = get_backend(backend)
    except ValueError as e:
        console.print(f"[red]Error:[/] {e}")
        import sys; sys.exit(1)

    console.print(f"\n[bold]Loading[/] [yellow]{model}[/] via [cyan]{backend}[/]…")
    try:
        b.load(model)
    except BackendError as e:
        console.print(f"[red]Load failed:[/] {e}")
        import sys; sys.exit(1)

    hw = hardware_summary()
    ppl: float | None = None

    if perplexity:
        console.print("\n[bold]Perplexity[/] — WikiText-2 test set…")
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                      TaskProgressColumn(), console=console) as prog:
            ptask = prog.add_task("Computing perplexity…", total=n_docs)

            def ppl_cb(done: int, total: int) -> None:
                prog.update(ptask, completed=done)

            try:
                ppl = compute_perplexity(b, n_docs=n_docs, progress_cb=ppl_cb)
            except NotImplementedError as e:
                console.print(f"  [yellow]Skipped:[/] {e}")
            except ImportError as e:
                console.print(f"  [yellow]Skipped:[/] {e}")

    console.print("\n[bold]Task accuracy probes[/]…")
    task_results = []
    task_list = list(tasks)
    total_examples = len(task_list) * n_per_task

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  TaskProgressColumn(), console=console) as prog:
        ttask = prog.add_task("Running tasks…", total=total_examples)
        completed = [0]

        def task_cb(done: int, total: int, name: str) -> None:
            completed[0] += 1
            prog.update(ttask, completed=completed[0], description=f"[cyan]{name}[/]…")

        try:
            task_results = run_task_probes(b, tasks=task_list, n_per_task=n_per_task, progress_cb=task_cb)
        except ImportError as e:
            console.print(f"  [yellow]Skipped:[/] {e}")

    b.unload()

    result = QualityResult(
        model=model,
        backend=backend,
        hardware=f"{hw['chip']} · {hw['memory_gb']:.0f} GB",
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        perplexity=ppl,
        tasks=task_results,
    )

    # --- Rich output ---
    console.print()
    console.print(f"[bold cyan]benchpress quality[/] · {model}", highlight=False)
    console.print(
        f"  backend: [yellow]{backend}[/]  hardware: [green]{result.hardware}[/]",
        highlight=False,
    )
    console.print()

    table = Table(box=box.SIMPLE_HEAD, header_style="bold")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    if ppl is not None:
        table.add_row("Perplexity (WikiText-2 ↓)", f"{ppl:.2f}")
    for tr in task_results:
        table.add_row(
            f"{tr.name} accuracy (↑)",
            f"{tr.accuracy:.1%}  ({tr.n_correct}/{tr.n_total})",
        )
    if result.quality_score is not None:
        table.add_row("Quality score (↑)", f"{result.quality_score:.3f}")

    console.print(table)

    if output:
        data = {
            "model": result.model,
            "backend": result.backend,
            "hardware": result.hardware,
            "timestamp": result.timestamp,
            "perplexity": result.perplexity,
            "tasks": [
                {"name": t.name, "accuracy": round(t.accuracy, 4),
                 "n_correct": t.n_correct, "n_total": t.n_total}
                for t in result.tasks
            ],
            "quality_score": result.quality_score,
        }
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[dim]Results saved to {output}[/]")


@main.command()
@click.argument("speed_file", type=click.Path(exists=True))
@click.option("--quality", "quality_file", type=click.Path(exists=True), default=None,
              help="Quality JSON from `benchpress quality --output`.")
@click.option("--results-dir", default="results", show_default=True,
              help="Directory to save the leaderboard entry.")
@click.option("--dry-run", is_flag=True, help="Print the entry without saving.")
def submit(speed_file, quality_file, results_dir, dry_run):
    """Format and save a leaderboard entry from benchmark output files.

    SPEED_FILE is the JSON output from `benchpress run --output`.
    Optionally pair with --quality from `benchpress quality --output`.

    Example:
        benchpress run my-model --output speed.json
        benchpress quality my-model --output quality.json
        benchpress submit speed.json --quality quality.json
    """
    import json
    from pathlib import Path
    from benchpress.leaderboard import make_entry, validate_entry, save_entry, render_rich

    with open(speed_file) as f:
        speed_json = json.load(f)

    quality_json = None
    if quality_file:
        with open(quality_file) as f:
            quality_json = json.load(f)

    entry = make_entry(speed_json, quality_json)
    errors = validate_entry(entry)
    if errors:
        for err in errors:
            console.print(f"[red]Validation error:[/] {err}")
        sys.exit(1)

    if dry_run:
        console.print_json(json.dumps(entry, indent=2))
        return

    path = save_entry(entry, Path(results_dir))
    console.print(f"\n[green]Saved[/] leaderboard entry → [bold]{path}[/]")
    console.print(
        f"\n  [dim]To share: commit [bold]{path}[/] and open a PR to "
        f"github.com/wfullen/benchpress[/]\n"
    )
    render_rich([entry])


@main.command("leaderboard")
@click.option("--results-dir", default="results", show_default=True,
              help="Directory containing leaderboard JSON files.")
@click.option("--markdown", is_flag=True, help="Print as Markdown table.")
def leaderboard_cmd(results_dir, markdown):
    """Display the local leaderboard from saved results."""
    from pathlib import Path
    from benchpress.leaderboard import load_all, render_rich, render_markdown

    entries = load_all(Path(results_dir))
    if markdown:
        click.echo(render_markdown(entries))
    else:
        render_rich(entries)


@main.command()
def backends():
    """List available backends and their installation status."""
    from rich.table import Table
    from rich import box

    checks = {
        "mlx": ("mlx_lm", "pip install mlx-lm"),
        "ollama": ("ollama", "pip install ollama  (also install Ollama.app)"),
        "transformers": ("transformers", "pip install torch transformers"),
        "llamacpp": ("llama_cpp", "CMAKE_ARGS='-DGGML_METAL=on' pip install llama-cpp-python --no-cache-dir"),
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
