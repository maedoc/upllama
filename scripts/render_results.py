#!/usr/bin/env python3
"""Render ometer benchmark JSON into a tiny static HTML page.
Uses Jinja2 for templating – only a few lines of logic.
"""

from __future__ import annotations
import argparse
import json
import math
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


def sparkline_svg(
    values: list[float],
    *,
    width: int = 100,
    height: int = 32,
    color: str = "#4ade80",
    scale: str = "linear",
    refs: list[float] | None = None,
    unit: str = "",
) -> str:
    """Return a tiny inline SVG sparkline with optional log scale & reference grid lines."""
    if len(values) < 2:
        return ""

    plot_left = 2
    plot_right = 72
    plot_top = 6
    plot_bottom = height - 2
    plot_h = plot_bottom - plot_top

    if scale == "log":
        transform = lambda v: math.log10(max(v, 1e-9))
    else:
        transform = lambda v: v

    values_t = [transform(v) for v in values]

    min_v = min(values_t)
    max_v = max(values_t)

    if scale == "log":
        pad = 0.15
    else:
        pad = (max_v - min_v) * 0.1 if min_v != max_v else 0.1

    ymin = min_v - pad
    ymax = max_v + pad
    rng = ymax - ymin
    if rng == 0:
        rng = 1.0

    def y_for(val_t: float) -> float:
        return plot_bottom - ((val_t - ymin) / rng) * plot_h

    step = (plot_right - plot_left) / (len(values) - 1)
    points = " ".join(
        f"{plot_left + i * step:.1f},{y_for(v):.1f}" for i, v in enumerate(values_t)
    )
    title = " ".join(f"{v:.2f}{unit}" for v in values)

    # Reference lines & labels
    ref_svg = ""
    if refs:
        for ref in refs:
            if ref <= 0 and scale == "log":
                continue
            ref_t = transform(ref)
            if ref_t < ymin or ref_t > ymax:
                continue
            ry = y_for(ref_t)
            label = f"{ref:g}{unit}"
            ref_svg += (
                f'<line x1="{plot_left}" y1="{ry:.1f}" x2="{plot_right}" y2="{ry:.1f}" '
                f'stroke="#bbb" stroke-width="0.5" stroke-dasharray="2,2" stroke-opacity="0.5" />'
            )
            ref_svg += (
                f'<text x="{plot_right + 2}" y="{ry:.1f}" font-size="7" fill="#999" '
                f'font-family="Arial,Helvetica,sans-serif" dominant-baseline="middle">{label}</text>'
            )

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'class="sparkline" title="{title}">'
        f'{ref_svg}'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}" />'
        f'</svg>'
    )


def build_sparklines(history: list[dict], current_results: list[dict]) -> dict[str, dict[str, str]]:
    """Build sparkline SVGs for each model/metric from 24 h history."""
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
        for metric, color, refs, unit in (
            ("ttft", "#4ade80", [0.1, 1.0, 10.0], "s"),
            ("tps", "#60a5fa", [1.0, 10.0, 100.0], ""),
        ):
            vals = model_history.get(name, {}).get(metric, [])
            if len(vals) >= 2:
                sparklines[name][metric] = sparkline_svg(
                    vals,
                    color=color,
                    scale="log",
                    refs=refs,
                    unit=unit,
                )
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
