"""Build the GitHub Pages leaderboard site from results/ JSON files."""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results"
SITE_DIR = ROOT / "_site"

sys.path.insert(0, str(ROOT / "src"))
from benchpress.leaderboard import load_all, render_markdown

SITE_DIR.mkdir(exist_ok=True)

entries = load_all(RESULTS_DIR)
leaderboard_md = render_markdown(entries)

# Count stats for the header
n_models = len({e["model"] for e in entries})
n_results = len(entries)
hardware_set = sorted({e["hardware"] for e in entries})

def _empty_state() -> str:
    return """
  <div class="empty">
    <p>No results yet.</p>
    <p style="margin-top:1rem">
      Run <code>benchpress run your-model --output speed.json</code>
      then <code>benchpress submit speed.json</code> and open a PR.
    </p>
  </div>"""


def _build_table(entries: list) -> str:
    sorted_entries = sorted(
        entries, key=lambda e: e["speed"]["tokens_per_second_mean"], reverse=True
    )
    rows = []
    for i, e in enumerate(sorted_entries, 1):
        s = e["speed"]
        q = e.get("quality") or {}
        tps = s["tokens_per_second_mean"]
        ci = s["tokens_per_second_ci"]
        ppl = q.get("perplexity")
        qs = q.get("quality_score")
        rank_class = f"rank-{i}" if i <= 3 else "rank"
        rows.append(f"""
    <tr>
      <td class="{rank_class}">{i}</td>
      <td class="model">{e['model']}</td>
      <td><span class="chip">{e['backend']}</span></td>
      <td>{e['hardware']}</td>
      <td class="right">{tps:.1f} <span style="color:var(--muted);font-size:0.8rem">[{ci[0]:.1f}, {ci[1]:.1f}]</span></td>
      <td class="right">{s['ttft_mean']:.3f}</td>
      <td class="right">{"—" if not ppl else f"{ppl:.1f}"}</td>
      <td class="right">{"—" if not qs else f"{qs:.3f}"}</td>
    </tr>""")

    return f"""
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Model</th>
        <th>Backend</th>
        <th>Hardware</th>
        <th class="right">tok/s</th>
        <th class="right">TTFT (s)</th>
        <th class="right">Perplexity ↓</th>
        <th class="right">Quality ↑</th>
      </tr>
    </thead>
    <tbody>{"".join(rows)}
    </tbody>
  </table>"""


html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>benchpress leaderboard</title>
  <style>
    :root {{
      --bg: #0f172a;
      --surface: #1e293b;
      --border: #334155;
      --text: #e2e8f0;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --green: #4ade80;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 2rem 1rem;
    }}
    .container {{ max-width: 960px; margin: 0 auto; }}
    h1 {{ font-size: 1.75rem; color: var(--accent); margin-bottom: 0.25rem; }}
    .subtitle {{ color: var(--muted); margin-bottom: 2rem; font-size: 0.95rem; }}
    .stats {{
      display: flex; gap: 2rem; margin-bottom: 2rem;
      padding: 1rem 1.5rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    .stat {{ text-align: center; }}
    .stat-value {{ font-size: 1.5rem; font-weight: 700; color: var(--accent); }}
    .stat-label {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th {{
      background: var(--surface);
      color: var(--muted);
      font-weight: 600;
      text-align: left;
      padding: 0.75rem 1rem;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }}
    th.right, td.right {{ text-align: right; }}
    td {{
      padding: 0.65rem 1rem;
      border-bottom: 1px solid var(--border);
    }}
    tr:hover td {{ background: var(--surface); }}
    .model {{ font-weight: 500; }}
    .chip {{
      display: inline-block;
      padding: 0.15rem 0.5rem;
      border-radius: 4px;
      font-size: 0.75rem;
      background: var(--surface);
      border: 1px solid var(--border);
      color: var(--muted);
    }}
    .rank {{ color: var(--muted); font-size: 0.85rem; }}
    .rank-1 {{ color: #fbbf24; }}
    .rank-2 {{ color: #94a3b8; }}
    .rank-3 {{ color: #b45309; }}
    .empty {{
      text-align: center;
      padding: 3rem;
      color: var(--muted);
    }}
    .empty code {{
      background: var(--surface);
      padding: 0.2rem 0.5rem;
      border-radius: 4px;
      font-family: monospace;
    }}
    footer {{
      margin-top: 3rem;
      color: var(--muted);
      font-size: 0.8rem;
      text-align: center;
    }}
    footer a {{ color: var(--accent); text-decoration: none; }}
    .note {{ margin-top: 1rem; color: var(--muted); font-size: 0.8rem; }}
  </style>
</head>
<body>
<div class="container">
  <h1>benchpress</h1>
  <p class="subtitle">
    LLM inference benchmark for Apple Silicon — speed + quality with statistical rigor
  </p>

  <div class="stats">
    <div class="stat">
      <div class="stat-value">{n_results}</div>
      <div class="stat-label">results</div>
    </div>
    <div class="stat">
      <div class="stat-value">{n_models}</div>
      <div class="stat-label">models</div>
    </div>
    <div class="stat">
      <div class="stat-value">{len(hardware_set)}</div>
      <div class="stat-label">hardware configs</div>
    </div>
  </div>

  {"_build_table(entries)_" if entries else _empty_state()}

  <p class="note">
    tok/s = mean tokens/sec · 95% bootstrap CI · 5 runs · 256 max tokens · temperature 0.0<br>
    <a href="https://github.com/WFullen94/benchpress#contributing-results">Submit your results</a>
    by running <code>benchpress submit</code> and opening a PR.
  </p>

  <footer>
    <a href="https://github.com/WFullen94/benchpress">github.com/WFullen94/benchpress</a>
    &nbsp;·&nbsp; MIT License
  </footer>
</div>
</body>
</html>"""

# Replace placeholder with real content
if entries:
    html = html.replace('"_build_table(entries)_"', _build_table(entries))
else:
    html = html.replace('"_build_table(entries)_"', _empty_state())

(SITE_DIR / "index.html").write_text(html)
print(f"Built leaderboard: {len(entries)} entries → _site/index.html")
