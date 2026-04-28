"""Leaderboard: schema, validation, and rendering for benchpress results."""

from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1"

# Canonical benchmark config — freeze this so leaderboard entries are comparable
CANONICAL_CONFIG = {
    "n_runs": 5,
    "warmup_runs": 1,
    "max_tokens": 256,
    "temperature": 0.0,
    "n_per_task": 50,
    "n_docs": 20,
}


def make_entry(
    speed_json: dict[str, Any],
    quality_json: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge speed + quality JSON outputs into a single leaderboard entry."""
    entry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": speed_json["model"],
        "backend": speed_json["backend"],
        "hardware": speed_json["hardware"],
        "submitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "speed": {
            "tokens_per_second_mean": speed_json["tokens_per_second"]["mean"],
            "tokens_per_second_ci": [
                speed_json["tokens_per_second"]["ci_lower"],
                speed_json["tokens_per_second"]["ci_upper"],
            ],
            "ttft_mean": speed_json["ttft_seconds"]["mean"],
            "latency_mean": speed_json["latency_seconds"]["mean"],
        },
    }

    if quality_json:
        entry["quality"] = {
            "perplexity": quality_json.get("perplexity"),
            "tasks": quality_json.get("tasks", []),
            "quality_score": quality_json.get("quality_score"),
        }
    else:
        entry["quality"] = None

    return entry


def validate_entry(entry: dict[str, Any]) -> list[str]:
    """Return a list of validation errors; empty list means valid."""
    errors = []
    required = ["schema_version", "model", "backend", "hardware", "speed"]
    for field in required:
        if field not in entry:
            errors.append(f"missing required field: {field}")
    speed = entry.get("speed", {})
    for sf in ["tokens_per_second_mean", "tokens_per_second_ci", "ttft_mean"]:
        if sf not in speed:
            errors.append(f"missing speed field: {sf}")
    return errors


def save_entry(entry: dict[str, Any], results_dir: Path) -> Path:
    """Write entry to results_dir as <model_slug>_<timestamp>.json."""
    results_dir.mkdir(parents=True, exist_ok=True)
    slug = entry["model"].replace("/", "_").replace(":", "_")
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = results_dir / f"{slug}_{ts}.json"
    path.write_text(json.dumps(entry, indent=2))
    return path


def load_all(results_dir: Path) -> list[dict[str, Any]]:
    """Load all valid .json entries from results_dir, newest first."""
    if not results_dir.exists():
        return []
    entries = []
    for p in sorted(results_dir.glob("*.json"), reverse=True):
        try:
            e = json.loads(p.read_text())
            if not validate_entry(e):
                entries.append(e)
        except (json.JSONDecodeError, OSError):
            pass
    return entries


def render_markdown(entries: list[dict[str, Any]]) -> str:
    """Render leaderboard entries as a Markdown table, sorted by tokens/sec."""
    if not entries:
        return "_No results yet. Run `benchpress submit` to add yours._\n"

    sorted_entries = sorted(
        entries,
        key=lambda e: e["speed"]["tokens_per_second_mean"],
        reverse=True,
    )

    lines = [
        "| Model | Backend | Hardware | tok/s | TTFT (s) | Perplexity ↓ | Quality ↑ |",
        "|-------|---------|----------|------:|--------:|-------------:|----------:|",
    ]

    for e in sorted_entries:
        s = e["speed"]
        q = e.get("quality") or {}
        tps = s["tokens_per_second_mean"]
        ci = s["tokens_per_second_ci"]
        ttft = s["ttft_mean"]
        ppl = q.get("perplexity")
        qs = q.get("quality_score")

        lines.append(
            f"| {e['model']} "
            f"| {e['backend']} "
            f"| {e['hardware']} "
            f"| {tps:.1f} [{ci[0]:.1f}, {ci[1]:.1f}] "
            f"| {ttft:.3f} "
            f"| {f'{ppl:.1f}' if ppl else '—'} "
            f"| {f'{qs:.3f}' if qs else '—'} |"
        )

    lines += [
        "",
        f"_tok/s = mean tokens/sec with 95% bootstrap CI · "
        f"Canonical config: {CANONICAL_CONFIG['n_runs']} runs, "
        f"{CANONICAL_CONFIG['max_tokens']} max tokens, "
        f"temperature {CANONICAL_CONFIG['temperature']}_",
    ]
    return "\n".join(lines) + "\n"


def render_rich(entries: list[dict[str, Any]]) -> None:
    """Print a rich terminal table of leaderboard entries."""
    from rich.console import Console
    from rich.table import Table
    from rich import box

    console = Console()

    if not entries:
        console.print("[dim]No results yet. Run [bold]benchpress submit[/] to add yours.[/]")
        return

    sorted_entries = sorted(
        entries,
        key=lambda e: e["speed"]["tokens_per_second_mean"],
        reverse=True,
    )

    table = Table(box=box.SIMPLE_HEAD, header_style="bold", title="benchpress leaderboard")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model")
    table.add_column("Backend")
    table.add_column("Hardware")
    table.add_column("tok/s", justify="right")
    table.add_column("TTFT (s)", justify="right")
    table.add_column("Perplexity ↓", justify="right")
    table.add_column("Quality ↑", justify="right")

    for i, e in enumerate(sorted_entries, 1):
        s = e["speed"]
        q = e.get("quality") or {}
        tps = s["tokens_per_second_mean"]
        ci = s["tokens_per_second_ci"]
        ppl = q.get("perplexity")
        qs = q.get("quality_score")

        table.add_row(
            str(i),
            e["model"],
            e["backend"],
            e["hardware"],
            f"{tps:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]",
            f"{s['ttft_mean']:.3f}",
            f"{ppl:.1f}" if ppl else "—",
            f"{qs:.3f}" if qs else "—",
        )

    console.print()
    console.print(table)
    console.print(
        f"  [dim]Sorted by tokens/sec · "
        f"95% bootstrap CI · "
        f"{len(sorted_entries)} result(s)[/]\n"
    )
