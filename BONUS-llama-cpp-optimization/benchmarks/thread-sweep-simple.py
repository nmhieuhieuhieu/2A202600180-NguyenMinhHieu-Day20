#!/usr/bin/env python3
"""A simple thread sweep script using llama-cpp-python library.
Does not require the native llama-bench binary.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from llama_cpp import Llama

def load_active() -> str:
    active_path = Path("models/active.json")
    if not active_path.exists():
        print("ERROR: models/active.json not found. Run download-model.py first.")
        sys.exit(1)
    return json.loads(active_path.read_text())["primary_model"]

def load_hw() -> dict:
    hw_path = Path("hardware.json")
    if not hw_path.exists():
        print("ERROR: hardware.json not found. Run detect-hardware.py first.")
        sys.exit(1)
    return json.loads(hw_path.read_text())

def thread_grid(hw: dict) -> list[int]:
    physical = hw["cpu"].get("cores_physical") or 4
    logical = hw["cpu"]["cores_logical"]
    # Test a few interesting points
    raw = sorted({1, 2, max(physical // 2, 1), physical, logical})
    if logical >= 8:
        raw.append(logical + 2)
    return [t for t in raw if t > 0]

def main():
    model_path = load_active()
    hw = load_hw()
    grid = thread_grid(hw)
    
    # Check for GPU
    backends = hw.get("gpu", {}).get("backends", {})
    n_gpu = 99 if any(v for k, v in backends.items() if k != "cpu_only") else 0

    print(f"==> Simple thread sweep using llama-cpp-python")
    print(f"    Model: {Path(model_path).name}")
    print(f"    Grid : {grid}")
    print()

    results = []
    for t in grid:
        print(f"   Testing t={t:3d}...", end="", flush=True)
        
        # Load model with specific thread count
        llm = Llama(
            model_path=model_path,
            n_threads=t,
            n_gpu_layers=n_gpu,
            n_ctx=512,
            verbose=False
        )
        
        # Warmup
        llm("Hello", max_tokens=1)
        
        # Actual test
        start = time.time()
        output = llm("Once upon a time in a galaxy far, far away,", max_tokens=64)
        end = time.time()
        
        duration = end - start
        tokens = output["usage"]["completion_tokens"]
        tps = tokens / duration
        
        results.append({"threads": t, "tps": tps})
        print(f" {tps:6.1f} tok/s")
        
        # Cleanup
        del llm

    # Save and Print Results Table
    print("\n# Bonus — Thread Sweep Results\n")
    print("| Threads | Tokens/sec |")
    print("|---------|------------|")
    for r in results:
        print(f"| {r['threads']:7d} | {r['tps']:10.1f} |")
    
    best = max(results, key=lambda x: x["tps"])
    print(f"\n**Best Result**: {best['tps']:.1f} tok/s with {best['threads']} threads.")
    
    # Save to file
    out_dir = Path("benchmarks")
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "bonus-thread-sweep.md", "w") as f:
        f.write("# Bonus — Thread Sweep\n\n")
        f.write(f"Model: `{Path(model_path).name}`\n\n")
        f.write("| Threads | Tokens/sec |\n|---|---:|\n")
        for r in results:
            f.write(f"| {r['threads']} | {r['tps']:.1f} |\n")
        f.write(f"\n**Best**: {best['tps']:.1f} tok/s with {best['threads']} threads.\n")

if __name__ == "__main__":
    main()
