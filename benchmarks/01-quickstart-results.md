# 01 — Quickstart Results

Settings: `n_threads=6`, `n_ctx=2048`, `n_batch=512`, `n_gpu_layers=99`.

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|---:|---:|---:|---:|---:|
| qwen2.5-0.5b-instruct-q4_k_m.gguf | 646 | 72 / 84 | 27.3 / 31.8 | 1785 / 2062 / 2106 | 36.7 |
| qwen2.5-0.5b-instruct-q2_k.gguf | 762 | 59 / 109 | 24.7 / 27.8 | 1616 / 1812 / 1843 | 40.5 |

## Observations

- TTFT is the prefill cost. With short prompts this is small; with long prompts it dominates.
- TPOT is per-token decode latency. The decode rate is `1000 / TPOT_p50`.
- The bigger quantization (Q4_K_M) is usually only ~30–60% slower than Q2_K but produces noticeably better text. Q2_K is for *truly* tight RAM.
- `n_threads = physical_cores` is usually best on CPU. Hyperthreading (`logical_cores`) often hurts because the work is bandwidth-bound.

(Edit this file with your own observations before submitting.)
