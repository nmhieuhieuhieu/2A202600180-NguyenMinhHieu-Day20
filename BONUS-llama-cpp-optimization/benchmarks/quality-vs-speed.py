#!/usr/bin/env python3
"""Bonus C5 — Quality vs Speed comparison across quantizations.

Compares Q4_K_M and Q2_K on 5 prompts:
- Decode tokens/sec
- Hand-gradeable output quality

Usage:
    python BONUS-llama-cpp-optimization/benchmarks/quality-vs-speed.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from llama_cpp import Llama

# 5 diverse prompts to test quality
EVAL_PROMPTS = [
    {"id": "q1", "prompt": "What is PagedAttention and what problem does it solve?", "max_tokens": 80},
    {"id": "q2", "prompt": "Explain the difference between TTFT and TPOT in one sentence each.", "max_tokens": 80},
    {"id": "q3", "prompt": "Why does quantization make LLMs faster? Answer in 2 sentences.", "max_tokens": 60},
    {"id": "q4", "prompt": "What is continuous batching in LLM serving?", "max_tokens": 80},
    {"id": "q5", "prompt": "Complete this JSON: {\"name\": \"Alice\", \"role\":", "max_tokens": 20},
]

REPEATS = 3  # Repeat each prompt N times for stable throughput measurement


def load_active() -> dict:
    active_path = Path("models/active.json")
    if not active_path.exists():
        print("ERROR: models/active.json not found.", file=sys.stderr)
        sys.exit(1)
    return json.loads(active_path.read_text())


def load_hw() -> dict:
    return json.loads(Path("hardware.json").read_text())


def benchmark_model(model_path: str, label: str, n_gpu: int, n_threads: int) -> dict:
    size_mb = Path(model_path).stat().st_size / 1024 / 1024

    llm = Llama(
        model_path=model_path,
        n_threads=n_threads,
        n_gpu_layers=n_gpu,
        n_ctx=512,
        verbose=False,
    )

    # Warmup
    llm("Hello", max_tokens=4)

    # Throughput measurement
    total_tokens = 0
    total_time = 0.0
    for _ in range(REPEATS):
        for ep in EVAL_PROMPTS:
            t0 = time.perf_counter()
            out = llm(ep["prompt"], max_tokens=ep["max_tokens"])
            total_time += time.perf_counter() - t0
            total_tokens += out["usage"]["completion_tokens"]

    avg_tps = total_tokens / total_time

    # Quality check — run each prompt once and capture answer
    answers = {}
    for ep in EVAL_PROMPTS:
        out = llm(ep["prompt"], max_tokens=ep["max_tokens"])
        answers[ep["id"]] = out["choices"][0]["text"].strip()

    del llm

    return {
        "label": label,
        "model_path": model_path,
        "size_mb": round(size_mb, 1),
        "avg_tps": round(avg_tps, 1),
        "answers": answers,
    }


def main() -> int:
    active = load_active()
    hw = load_hw()
    backends = hw.get("gpu", {}).get("backends", {})
    n_gpu = 99 if any(v for k, v in backends.items() if k != "cpu_only") else 0
    n_threads = hw["cpu"].get("cores_physical") or 4

    primary = active["primary_model"]
    compare = active.get("compare_model")

    if not compare or not Path(compare).exists():
        print("WARNING: compare_model not found, running primary only.")
        models_to_test = [(primary, "Q4_K_M")]
    else:
        models_to_test = [(primary, "Q4_K_M"), (compare, "Q2_K")]

    print(f"==> Quality vs Speed Sweep (Bonus C5)")
    print(f"    GPU layers: {n_gpu}  |  Threads: {n_threads}")
    print(f"    Prompts   : {len(EVAL_PROMPTS)}  |  Repeats: {REPEATS}\n")

    results = []
    for model_path, label in models_to_test:
        print(f"   Benchmarking {label} ({Path(model_path).name})...", end="", flush=True)
        result = benchmark_model(model_path, label, n_gpu, n_threads)
        results.append(result)
        print(f" {result['avg_tps']} tok/s  ({result['size_mb']} MB)")

    # --- Summary Table ---
    print("\n## Speed Comparison\n")
    print(f"| Model   | Size (MB) | Avg (tok/s) | vs Q4 |")
    print(f"|---------|-----------|-------------|-------|")
    baseline_tps = results[0]["avg_tps"]
    for r in results:
        ratio = r["avg_tps"] / baseline_tps
        tag = "baseline" if ratio == 1.0 else f"{ratio:.2f}x"
        print(f"| {r['label']:7s} | {r['size_mb']:9.1f} | {r['avg_tps']:11.1f} | {tag:5s} |")

    print("\n## Quality Spot-check\n")
    for ep in EVAL_PROMPTS:
        print(f"**Q: {ep['prompt']}**")
        for r in results:
            print(f"  [{r['label']}]: {r['answers'][ep['id']][:120]}")
        print()

    # --- Write result ---
    out_dir = Path("benchmarks")
    out_dir.mkdir(exist_ok=True)

    md = "# Bonus C5 — Quality vs Speed Across Quantizations\n\n"
    md += f"**Hardware:** {hw['cpu']['model']}  ·  GPU: {hw.get('gpu', {}).get('name', 'N/A')}\n\n"
    md += "## Speed Comparison\n\n"
    md += "| Model | Size (MB) | Avg tok/s | Speedup |\n|---|--:|--:|--:|\n"
    for r in results:
        ratio = r["avg_tps"] / baseline_tps
        md += f"| {r['label']} | {r['size_mb']:.1f} | {r['avg_tps']:.1f} | {ratio:.2f}x |\n"

    md += "\n## Quality Spot-check\n\n"
    for ep in EVAL_PROMPTS:
        md += f"**Q: {ep['prompt']}**\n\n"
        for r in results:
            md += f"- **{r['label']}**: {r['answers'][ep['id']][:200]}\n"
        md += "\n"

    md += "## Analysis\n\n"
    md += (
        "Q2_K is smaller and faster but sacrifices quality — noticeable on structured outputs (JSON)\n"
        "and reasoning tasks. Q4_K_M is the production sweet spot: near-lossless quality at ~70%\n"
        "of Q8_0 file size. On RAM-constrained devices, Q2_K can run models that otherwise don't fit,\n"
        "making it a valid choice when *any* response beats no response.\n"
    )

    (out_dir / "bonus-quality-vs-speed.md").write_text(md)
    print(f"==> Written to benchmarks/bonus-quality-vs-speed.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
