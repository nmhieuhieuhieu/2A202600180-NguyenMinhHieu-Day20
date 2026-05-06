# Reflection — Lab 20 (Personal Report)


**Name:** NGUYEN MINH HIEU 

**Student ID:** 2A202600180

**Cohort:** A20-K1  

**Submission Date:** 2026-05-06  

---

## 1. Hardware spec (from `00-setup/detect-hardware.py`)

- **OS:** Windows 10 (AMD64)
- **CPU:** AMD Ryzen 5 5600H with Radeon Graphics
- **Cores:** 6 physical / 12 logical
- **CPU extensions:** SSE3, SSSE3, AVX, AVX2, F16C, FMA
- **RAM:** 7.4 GB
- **Accelerator:** NVIDIA GeForce RTX 3050 Laptop GPU (4096 MiB)
- **Selected llama.cpp backend:** CUDA (nvidia_cuda)
- **Recommended model tier:** Qwen2.5-0.5B-Instruct (Custom selected for efficiency)

**Setup story**:
Initial setup faced Windows path length (Long Path) limitations and missing MSVC compiler for `llama-cpp-python` build. Resolved by installing the pre-built binary wheel. Forced `Qwen2.5-0.5B-Instruct` model to ensure smooth execution on laptop resources. Fixed Unicode encoding issues in setup scripts to support Windows PowerShell terminal output.

---

## 2. Track 01 — Quickstart numbers (from `benchmarks/01-quickstart-results.md`)

| Model | Load (ms) | TTFT P50/P95 (ms) | TPOT P50/P95 (ms) | E2E P50/P95/P99 (ms) | Decode rate (tok/s) |
|---|--:|--:|--:|--:|--:|
| qwen2.5-0.5b (Q4_K_M) | 646 | 72 / 84 | 27.3 / 31.8 | 1785 / 2062 / 2106 | 36.7 |
| qwen2.5-0.5b (Q2_K)   | 762 | 59 / 109 | 24.7 / 27.8 | 1616 / 1812 / 1843 | 40.5 |

**Observation** (≤ 50 words):
Q2_K provides a ~10% speedup in decoding but shows much higher variance in P95 TTFT. For a model this small, the Q4_K_M quantization is the sweet spot, as it maintains quality without any noticeable latency penalty on this hardware.

---

## 3. Track 02 — llama-server load test

| Concurrency | Total RPS | TTFB P50 (ms) | E2E P95 (ms) | E2E P99 (ms) | Failures |
|--:|--:|--:|--:|--:|--:|
| 10 | 0.44 | 16000 | 26000 | 26000 | 0 |
| 50 | 0.37 | 26000 | 41000 | 41000 | 0 |

**KV-cache observation** (from `record-metrics.py`):
Metrics endpoint was unavailable on the Python server build, but inferred KV-cache usage was minimal due to the tiny 0.5B model size. However, end-to-end latency scaled significantly with concurrency, indicating that the bottleneck is raw compute/scheduling on the laptop GPU/CPU.

---

## 4. Track 03 — Milestone integration

- **N16 (Cloud/IaC):** stub: docker-compose local stack
- **N17 (Data pipeline):** stub: batch processing script
- **N18 (Lakehouse):** stub: SQLite local storage
- **N19 (Vector + Feature Store):** stub: TOY_DOCS in-memory index
- **embed:** 0.0 ms (mocked)
- **retrieve:** 0.0 ms (in-memory search)
- **llama-server:** 8736.4 ms (peak)

**Reflection** (≤ 60 chữ):
The primary bottleneck is `llama-server` inference latency. While retrieval is instantaneous due to the toy dataset, the generation phase takes ~8s for long responses. In a real production environment, optimizing KV-cache or using a larger GPU would be necessary to reduce E2E time.

---

## 5. Bonus — The single change that mattered most

**Change:** Optimizing thread count via Thread Sweep (hạ `-t` từ 12 xuống 1).

**Before vs after**:

```
t=12: 10.4 tok/s
t= 1: 18.6 tok/s
speedup: ~1.79x
```

**Tại sao nó work**:
For the extremely lightweight Qwen-0.5B model, the computational density is low. When using many threads, the overhead of context switching and inter-thread synchronization on the Ryzen 5600H actually slows down inference. A single thread keeps the data hot in the L1/L2 caches and avoids the "coordination tax," leading to nearly double the performance.

---

## 6. (Optional) Biggest Surprise

It was surprising that the smallest thread count (t=1) outperformed the physical core count (t=6) by such a wide margin. It proves that "more threads" is not always better for small models.

---

## 7. Self-graded checklist

- [x] `hardware.json` committed
- [x] `models/active.json` committed
- [x] `benchmarks/01-quickstart-results.md` committed
- [x] `benchmarks/02-server-results.md` (or CSV from `record-metrics.py`) committed
- [x] `benchmarks/bonus-*.md` committed (at least 1 sweep)
- [x] At least 6 screenshots in `submission/screenshots/` (see `submission/screenshots/README.md`)
- [x] `make verify` (or `verify.py`) exits 0 (run right before pushing)
- [x] Repo on GitHub is set to **public**
- [x] Public repo URL pasted into VinUni LMS

---

