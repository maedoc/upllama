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
    timestamps: list[datetime] | None = None,
    error_timestamps: list[datetime] | None = None,
    width: int = 100,
    height: int = 38,
    color: str = "#4ade80",
    scale: str = "linear",
    refs: list[float] | None = None,
    unit: str = "",
) -> str:
    """Return a tiny inline SVG sparkline with optional log scale, reference grid lines, time labels, and error bars."""
    if len(values) < 2:
        return ""

    plot_left = 2
    plot_right = 72
    plot_top = 6
    plot_bottom = height - 8
    plot_h = plot_bottom - plot_top
    plot_w = plot_right - plot_left

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

    # Always include the requested reference values in the visible domain
    ref_vals_t = [transform(ref) for ref in refs or []]
    if ref_vals_t:
        ymin = min(min_v, *ref_vals_t) - pad
        ymax = max(max_v, *ref_vals_t) + pad
    else:
        ymin = min_v - pad
        ymax = max_v + pad
    rng = ymax - ymin
    if rng == 0:
        rng = 1.0

    def y_for(val_t: float) -> float:
        return plot_bottom - ((val_t - ymin) / rng) * plot_h

    # Compute x positions – linear time
    all_timestamps: list[datetime] = list(timestamps or []) + list(error_timestamps or [])
    ts_oldest = min(all_timestamps) if all_timestamps else None
    ts_newest = max(all_timestamps) if all_timestamps else None
    ts_span = (ts_newest - ts_oldest).total_seconds() if ts_oldest and ts_newest else 0

    if timestamps and len(timestamps) == len(values) and ts_span > 0:
        xs = [plot_left + ((ts - ts_oldest).total_seconds() / ts_span) * plot_w for ts in timestamps]
    else:
        step = plot_w / (len(values) - 1)
        xs = [plot_left + i * step for i in range(len(values))]

    points = " ".join(f"{x:.1f},{y_for(v):.1f}" for x, v in zip(xs, values_t))
    title = " ".join(f"{v:.2f}{unit}" for v in values)

    # Error bars – vertical red lines at error timestamps
    error_svg = ""
    if error_timestamps and ts_span > 0:
        for ets in error_timestamps:
            ratio = (ets - ts_oldest).total_seconds() / ts_span
            ex = plot_left + ratio * plot_w
            error_svg += (
                f'<line x1="{ex:.1f}" y1="{plot_top}" x2="{ex:.1f}" y2="{plot_bottom}" '
                f'stroke="#ef4444" stroke-width="1.5" opacity="0.7" />'
            )

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

    # Time labels at the bottom – linear: -6h, -12h, -24h, -48h
    time_svg = ""
    if ts_span > 0:
        for hours_ago in [6, 12, 24, 48]:
            label_ts = ts_newest - timedelta(hours=hours_ago)
            if label_ts >= ts_oldest:
                ratio = (label_ts - ts_oldest).total_seconds() / ts_span
                tx = plot_left + ratio * plot_w
                label_text = f"-{hours_ago}h"
                time_svg += (
                    f'<line x1="{tx:.1f}" y1="{plot_bottom}" x2="{tx:.1f}" y2="{plot_bottom + 3}" '
                    f'stroke="#999" stroke-width="0.5" />'
                )
                time_svg += (
                    f'<text x="{tx:.1f}" y="{height - 2}" font-size="6" fill="#999" '
                    f'font-family="Arial,Helvetica,sans-serif" text-anchor="middle">'
                    f'{label_text}</text>'
                )

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'class="sparkline" title="{title}">'
        f'{error_svg}'
        f'{ref_svg}'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.5" points="{points}" />'
        f'{time_svg}'
        f'</svg>'
    )


def build_sparklines(history: list[dict], current_results: list[dict]) -> dict[str, dict[str, str]]:
    """Build sparkline SVGs for each model/metric from 72 h history."""
    model_history: dict[str, dict[str, list[tuple[datetime, float]]]] = {}
    model_errors: dict[str, list[datetime]] = {}
    for entry in history:
        ts = datetime.fromisoformat(entry["timestamp"])
        for r in entry.get("results", []):
            name = r.get("model")
            if not name:
                continue
            if name not in model_history:
                model_history[name] = {"ttft": [], "tps": []}
                model_errors[name] = []
            if r.get("error"):
                model_errors[name].append(ts)
            else:
                if r.get("ttft") is not None:
                    model_history[name]["ttft"].append((ts, r["ttft"]))
                if r.get("tps") is not None:
                    model_history[name]["tps"].append((ts, r["tps"]))

    sparklines: dict[str, dict[str, str]] = {}
    for r in current_results:
        name = r.get("model")
        sparklines[name] = {}
        for metric, color, refs, unit in (
            ("ttft", "#4ade80", [1.0, 10.0], "s"),
            ("tps", "#60a5fa", [10.0, 100.0], ""),
        ):
            data = model_history.get(name, {}).get(metric, [])
            vals = [v for _, v in data]
            timestamps = [t for t, _ in data]
            error_timestamps = model_errors.get(name, [])
            if len(vals) >= 2:
                sparklines[name][metric] = sparkline_svg(
                    vals,
                    timestamps=timestamps,
                    error_timestamps=error_timestamps,
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

    cutoff = now - timedelta(hours=48)
    history = [
        h for h in history
        if datetime.fromisoformat(h["timestamp"]) >= cutoff
    ]
    return history


def compute_means(history: list[dict]) -> dict[str, dict[str, float | None]]:
    """Compute geometric mean of non-error values per model/metric."""
    model_vals: dict[str, dict[str, list[float]]] = {}
    for entry in history:
        for r in entry.get("results", []):
            name = r.get("model")
            if not name or r.get("error"):
                continue
            if name not in model_vals:
                model_vals[name] = {"ttft": [], "tps": []}
            if r.get("ttft") is not None:
                model_vals[name]["ttft"].append(r["ttft"])
            if r.get("tps") is not None:
                model_vals[name]["tps"].append(r["tps"])

    means: dict[str, dict[str, float | None]] = {}
    for name, metrics in model_vals.items():
        means[name] = {}
        for metric in ("ttft", "tps"):
            vals = metrics.get(metric, [])
            if vals and all(v > 0 for v in vals):
                means[name][metric] = math.exp(sum(math.log(v) for v in vals) / len(vals))
            else:
                means[name][metric] = None
    return means


def render(results: list[dict], history: list[dict], free_models: dict | None = None) -> None:
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
    means = compute_means(history)

    rendered = tmpl.render(
        results=results,
        generated_at=now,
        github=github_ctx,
        sparklines=sparklines,
        means=means,
        free_models=free_models,
    )

    SITE_DIR.mkdir(parents=True, exist_ok=True)

    # Write the HTML page
    (SITE_DIR / "index.html").write_text(rendered, encoding="utf-8")
    # Copy raw JSON for curious visitors (optional)
    (SITE_DIR / "results.json").write_text(json.dumps(results, indent=2))
    # Write rolling history JSON so next runs can accumulate
    (SITE_DIR / "history.json").write_text(json.dumps(history, indent=2))
    # Favicon – smile-sweat llama emoji in black & white
    (SITE_DIR / "favicon.svg").write_text(FAVICON_SVG, encoding="utf-8")


FAVICON_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <path d="M10 11 L8 4 L13 10 Z" fill="#111"/>
  <path d="M22 11 L24 4 L19 10 Z" fill="#111"/>
  <path d="M16 28 C9 28 7 24 7 20 C7 16 9 13 12 12 C14 11 15 11 16 11 C17 11 18 11 20 12 C23 13 25 16 25 20 C25 24 23 28 16 28 Z" fill="#111"/>
  <circle cx="12" cy="17" r="1.5" fill="#fff"/>
  <circle cx="20" cy="17" r="1.5" fill="#fff"/>
  <path d="M11 22 Q16 25 21 22" fill="none" stroke="#fff" stroke-width="1.5" stroke-linecap="round"/>
  <path d="M24 5 C26.5 3 27.5 6 27.5 8 C27.5 10.5 25.5 11 24 11 C22.5 11 20.5 10.5 20.5 8 C20.5 6 21.5 3 24 5 Z" fill="#111"/>
</svg>
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=pathlib.Path, default=None, help="Path to existing history.json")
    parser.add_argument("--free-models", type=pathlib.Path, default=None, help="Path to free_models.json")
    args = parser.parse_args()

    results = load_results()
    history = update_history(args.history, results)

    free_models: dict | None = None
    if args.free_models and args.free_models.is_file():
        with args.free_models.open(encoding="utf-8") as f:
            free_models = json.load(f)

    render(results, history, free_models)
    print(f"[info] static site generated in {SITE_DIR}")

if __name__ == "__main__":
    main()
