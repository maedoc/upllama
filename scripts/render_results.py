#!/usr/bin/env python3
"""Render ometer benchmark JSON into a tiny static HTML page.
Uses Jinja2 for templating – only a few lines of logic.
"""

from __future__ import annotations
import json
import pathlib
import sys
from datetime import datetime, timezone

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

def render(results: list[dict]) -> None:
    """Render index.html using a Jinja2 template."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("index.html.j2")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rendered = tmpl.render(results=results, generated_at=now)

    SITE_DIR.mkdir(parents=True, exist_ok=True)

    # Write the HTML page
    (SITE_DIR / "index.html").write_text(rendered, encoding="utf-8")
    # Copy raw JSON for curious visitors (optional)
    (SITE_DIR / "results.json").write_text(json.dumps(results, indent=2))

def main() -> None:
    results = load_results()
    render(results)
    print(f"[info] static site generated in {SITE_DIR}")

if __name__ == "__main__":
    main()
