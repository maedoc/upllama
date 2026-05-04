#!/usr/bin/env python3
"""Quick probe of all cloud models with a free-tier API key.
Outputs free_models.json with availability per model.
"""

from __future__ import annotations
import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
from datetime import datetime, timezone

import httpx

# -------------------------------------------------------------------------
BASE_URL = os.getenv("OLLAMA_CLOUD_BASE_URL", "https://ollama.com")
KEY = os.getenv("UPLLAMA_FREE_KEY", "")

# Fallback: read .upllama-free-key in repo root
if not KEY:
    key_path = pathlib.Path(__file__).resolve().parents[1] / ".upllama-free-key"
    if key_path.is_file():
        KEY = key_path.read_text().strip()

HEADERS = {"Authorization": f"Bearer {KEY}"} if KEY else {}


# -------------------------------------------------------------------------
async def list_models(client: httpx.AsyncClient) -> list[dict]:
    resp = await client.get(f"{BASE_URL}/api/tags", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json().get("models", [])


async def probe_model(client: httpx.AsyncClient, name: str) -> dict:
    """Send a minimal 1-token prompt and return availability."""
    start = time.perf_counter()
    payload = {
        "model": name,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "options": {"num_predict": 1},
    }
    try:
        async with client.stream(
            "POST",
            f"{BASE_URL}/api/chat",
            json=payload,
            headers=HEADERS,
            timeout=30,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if chunk.get("error"):
                    return {
                        "model": name,
                        "available": False,
                        "error": chunk["error"],
                        "ttft_ms": None,
                    }
                msg = chunk.get("message") or {}
                if msg.get("content") or chunk.get("done"):
                    elapsed = (time.perf_counter() - start) * 1000
                    return {
                        "model": name,
                        "available": True,
                        "error": None,
                        "ttft_ms": round(elapsed, 1),
                    }
    except Exception as e:
        return {"model": name, "available": False, "error": str(e), "ttft_ms": None}

    return {"model": name, "available": False, "error": "No response", "ttft_ms": None}


async def main(output: pathlib.Path) -> None:
    if not KEY:
        sys.exit("[error] No UPLLAMA_FREE_KEY found in env or .upllama-free-key")
    async with httpx.AsyncClient() as client:
        models = await list_models(client)
        models.sort(key=lambda m: m.get("name", "").lower())
        print(f"[info] Probing {len(models)} model(s)…")
        tasks = [probe_model(client, m["name"]) for m in models]
        results = await asyncio.gather(*tasks)

    out = {
        "tested_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "count_total": len(results),
        "count_available": sum(1 for r in results if r["available"]),
        "results": results,
    }
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(
        f"[info] Wrote {output}  ({out['count_available']}/{out['count_total']} available)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("free_models.json"))
    args = parser.parse_args()
    asyncio.run(main(args.output))
