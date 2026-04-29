#!/usr/bin/env python3
"""Generate a tiny dummy results.json for the benchmark site.
Used when the CI environment cannot reach a real Ollama server.
"""
import json
import pathlib

OUTPUT = pathlib.Path(__file__).resolve().parents[1] / "results.json"

# Minimal record structure matching ExportRow fields used by the template
dummy = [
    {
        "model": "dummy-model:latest",
        "size": "1B",
        "context": "2048",
        "quant": "none",
        "capabilities": "chat",
        "ttft": 0.12,
        "tps": 45.6,
        "error": None,
        "runs": [],
        "modified_at": "2024-01-01T00:00:00Z",
    }
]

OUTPUT.write_text(json.dumps(dummy, indent=2))
print(f"[info] dummy results written to {OUTPUT}")
