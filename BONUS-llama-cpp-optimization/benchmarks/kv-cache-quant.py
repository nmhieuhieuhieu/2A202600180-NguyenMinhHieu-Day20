#!/usr/bin/env python3
"""Bonus C2 — KV-cache quantization sweep using llama-cpp-python.

Compares F16 (default), Q8_0, and Q4_0 KV-cache types.
Measures: decode tokens/sec, latency, and quality consistency.

Usage:
    python BONUS-llama-cpp-optimization/benchmarks/kv-cache-quant.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from llama_cpp import Llama
import llama_cpp

# Test prompts for quality comparison
TEST_PROMPTS = [
    "Explain what PagedAttention is in one sentence.",
    "What is the difference between TTFT and TPOT in LLM serving?",
    "Why is KV-cache important for inference optimization?",
]

# KV cache configurations to sweep
# Use llama_cpp.GGML_TYPE_* constants
KV_CONFIGS = [
    {"label": "F16 (default)", "type_k": llama_cpp.GGML_TYPE_F16,  "type_v": llama_cpp.GGML_TYPE_F16},
    {"label": "Q8_0",          "type_k": llama_cpp.GGML_TYPE_Q8_0, "type_v": llama_cpp.GGML_TYPE_Q8_0},
    {"label": "Q4_0",          "type_k": llama_cpp.GGML_TYPE_Q4_0, "type_v": llama_cpp.GGML_TYPE_Q4_0},
]


def load_active() -> str:
    active_path = Path("models/active.json")
    if not active_path.exists():
        print("ERROR: models/active.json not found.", file=sys.stderr)
        sys.exit(1)
    return json.loads(active_path.read_text())["primary_model"]


def load_hw() -> dict:
    hw_path = Path("hardware.json")
    if not hw_path.exists():
        print("ERROR: hardware.json not found.", file=sys.stderr)
        sys.exit(1)
    return json.loads(hw_path.read_text())


def run_benchmark(model_path: str, n_gpu: int, type_k: int, type_v: int, n_threads: int) -> dict:
    """Run a benchmark with a specific KV cache type."""
    llm = Llama(
        model_path=model_path,
        n_threads=n_threads,
        n_gpu_layers=n_gpu,
        n_ctx=1024,
        type_k=type_k,
        type_v=type_v,
        verbose=False,
    )

    # Warmup
    llm("Hello", max_tokens=4)

    total_tokens = 0
    total_time = 0.0
    answers = []

    for prompt in TEST_PROMPTS:
        t0 = time.perf_counter()
        out = llm(prompt, max_tokens=64)
        elapsed = time.perf_counter() - t0

        tokens = out["usage"]["completion_tokens"]
        total_tokens += tokens
        total_time += elapsed
        answers.append(out["choices"][0]["text"].strip()[:80])

    avg_tps = total_tokens / total_time
    del llm

    return {
        "avg_tps": avg_tps,
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 2),
        "answers": answers,
    }


def main() -> int:
    model_path = load_active()
    hw = load_hw()
    backends = hw.get("gpu", {}).get("backends", {})
    n_gpu = 99 if any(v for k, v in backends.items() if k != "cpu_only") else 0
    n_threads = hw["cpu"].get("cores_physical") or 4

    print(f"==> KV-cache Quantization Sweep (Bonus C2)")
    print(f"    Model  : {Path(model_path).name}")
    print(f"    GPU    : n_gpu_layers={n_gpu}")
    print(f"    Threads: {n_threads}")
    print(f"    Prompts: {len(TEST_PROMPTS)} test prompts × 64 tokens each\n")

    rows = []
    for cfg in KV_CONFIGS:
        print(f"   Testing KV={cfg['label']}...", end="", flush=True)
        result = run_benchmark(model_path, n_gpu, cfg["type_k"], cfg["type_v"], n_threads)
        rows.append({**cfg, **result})
        print(f" {result['avg_tps']:6.1f} tok/s  ({result['total_tokens']} tokens in {result['total_time_s']}s)")

    # --- Print Summary Table ---
    print("\n# Bonus C2 — KV-cache Quantization Results\n")
    print(f"| KV Cache Type | Avg (tok/s) | Speedup vs F16 |")
    print(f"|---------------|-------------|----------------|")

    baseline_tps = rows[0]["avg_tps"]
    for r in rows:
        speedup = r["avg_tps"] / baseline_tps
        marker = " ← baseline" if r["label"] == "F16 (default)" else f" ({speedup:.2f}x)"
        print(f"| {r['label']:13s} | {r['avg_tps']:11.1f} |{marker:16s}|")

    # Quality spot-check
    print("\n## Quality Spot-check (first prompt answer)\n")
    prompt_shown = TEST_PROMPTS[0]
    print(f"Prompt: \"{prompt_shown}\"\n")
    for r in rows:
        print(f"  [{r['label']}]: {r['answers'][0]}")

    # --- Write result to benchmarks/ ---
    best = max(rows, key=lambda x: x["avg_tps"])
    out_dir = Path("benchmarks")
    out_dir.mkdir(exist_ok=True)

    md = "# Bonus C2 — KV-cache Quantization\n\n"
    md += f"**Model:** `{Path(model_path).name}`  ·  **GPU layers:** `{n_gpu}`  ·  **Threads:** `{n_threads}`\n\n"
    md += "## Results\n\n"
    md += "| KV Cache Type | Avg (tok/s) | Speedup |\n|---|--:|--:|\n"
    for r in rows:
        speedup = r["avg_tps"] / baseline_tps
        md += f"| {r['label']} | {r['avg_tps']:.1f} | {speedup:.2f}x |\n"
    md += f"\n**Best**: `{best['label']}` at {best['avg_tps']:.1f} tok/s.\n\n"
    md += "## Analysis\n\n"
    md += (
        "KV-cache quantization reduces the memory footprint of the attention key-value cache.\n"
        "For small models like Qwen-0.5B, the KV cache is tiny and the benefit is marginal.\n"
        "On larger models (7B+) with long contexts, Q8_0 KV-cache typically saves 30-50% RAM\n"
        "with minimal quality degradation — a key technique in production serving.\n"
    )
    md += "\n## Quality Spot-check\n\n"
    md += f"Prompt: *\"{TEST_PROMPTS[0]}\"*\n\n"
    for r in rows:
        md += f"- **{r['label']}**: {r['answers'][0]}\n"

    (out_dir / "bonus-kv-cache-quant.md").write_text(md)
    print(f"\n==> Written to benchmarks/bonus-kv-cache-quant.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
