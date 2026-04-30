#!/usr/bin/env python3
"""Render ometer benchmark JSON into a tiny static HTML page.
Uses Jinja2 for templating – only a few lines of logic.
"""

from __future__ import annotations
import argparse
import json
import pathlib
import sys
import os
from datetime import datetime, timezone, timedelta

from jinja2 import Environment, FileSystemLoader, select_autoescape

# -------------------------------------------------------------------------
# Paths (relative to repo root)
ROOT = pathlib.Path(__file__).resolve().parents[1]   # repo root
RESULTS_JSON = ROOT / "results.json"
SITE_DIR = ROOT / "site"
TEMPLATE_DIR = ROOT / "templates"

# -------------------------------------------------------------------------
def load_results() -> list[dict]:
    """Read the JSON file produced by ometer."""
    if not RESULTS_JSON.is_file():
        sys.exit(f"[error] {RESULTS_JSON} missing – benchmark step failed?")
    with RESULTS_JSON.open(encoding="utf-8") as f:
        data = json.load(f)
    return data


def sparkline_svg(values: list[float], width: int = 80, height: int = 24, color: str = "#4ade80") -> str:
    """Return a tiny inline SVG sparkline."""
    if len(values) < 2:
        return ""
    min_v = min(values)
    max_v = max(values)
    rng = max_v - min_v
    if rng == 0:
        y_coords = [height / 2] * len(values)
    else:
        y_coords = [height - ((v - min_v) / rng) * height for v in values]
    step = width / (len(values) - 1)
    points = " ".join(f"{i * step:.1f},{y:.1f}" for i, y in enumerate(y_coords))
    title = " ".join(f"{v:.2f}" for v in values)
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'class="sparkline" title="{title}">'
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>'
        f'</svg>'
    )


def build_sparklines(history: list[dict], current_results: list[dict]) -> dict[str, dict[str, str]]:
    """Build sparkline SVGs for each model/metric from 24 h history."""
    # Aggregate historical values per model
    model_history: dict[str, dict[str, list[float]]] = {}
    for entry in history:
        for r in entry.get("results", []):
            name = r.get("model")
            if not name:
                continue
            if name not in model_history:
                model_history[name] = {"ttft": [], "tps": []}
            if r.get("ttft") is not None:
                model_history[name]["ttft"].append(r["ttft"])
            if r.get("tps") is not None:
                model_history[name]["tps"].append(r["tps"])

    sparklines: dict[str, dict[str, str]] = {}
    for r in current_results:
        name = r.get("model")
        sparklines[name] = {}
        for metric, color in (("ttft", "#4ade80"), ("tps", "#60a5fa")):
            vals = model_history.get(name, {}).get(metric, [])
            # Need at least two points to draw a line
            if len(vals) >= 2:
                sparklines[name][metric] = sparkline_svg(vals, color=color)
            else:
                sparklines[name][metric] = ""
    return sparklines


def update_history(history_path: pathlib.Path | None, results: list[dict]) -> list[dict]:
    """Load existing history, append current run, trim to last 24 h."""
    history: list[dict] = []
    if history_path and history_path.is_file():
        with history_path.open(encoding="utf-8") as f:
            history = json.load(f)

    now = datetime.now(timezone.utc)
    history.append({
        "timestamp": now.isoformat(),
        "results": results,
    })

    cutoff = now - timedelta(hours=24)
    history = [
        h for h in history
        if datetime.fromisoformat(h["timestamp"]) >= cutoff
    ]
    return history


def render(results: list[dict], history: list[dict]) -> None:
    """Render index.html using a Jinja2 template."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("index.html.j2")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Pull GitHub context from environment variables injected by the workflow
    github_ctx = {
        "repository": os.getenv("GITHUB_REPOSITORY", ""),
        "run_id": os.getenv("GITHUB_RUN_ID", ""),
        "run_number": os.getenv("GITHUB_RUN_NUMBER", ""),
    }

    sparklines = build_sparklines(history, results)

    rendered = tmpl.render(
        results=results,
        generated_at=now,
        github=github_ctx,
        sparklines=sparklines,
    )

    SITE_DIR.mkdir(parents=True, exist_ok=True)

    # Write the HTML page
    (SITE_DIR / "index.html").write_text(rendered, encoding="utf-8")
    # Copy raw JSON for curious visitors (optional)
    (SITE_DIR / "results.json").write_text(json.dumps(results, indent=2))
    # Write rolling history JSON so next runs can accumulate
    (SITE_DIR / "history.json").write_text(json.dumps(history, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=pathlib.Path, default=None, help="Path to existing history.json")
    args = parser.parse_args()

    results = load_results()
    history = update_history(args.history, results)
    render(results, history)
    print(f"[info] static site generated in {SITE_DIR}")

if __name__ == "__main__":
    main()
