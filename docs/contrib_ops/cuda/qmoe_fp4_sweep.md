# QMoE FP4 GEMV Decode — Config/Autotune/Lever-A Sweep (H200)

**Purpose:** Decide which FP4 QMoE decode-path kernels/techniques are worth keeping in production and
which to remove. This document records the H200 sweep only. **Data from other GPUs (A100, B200/SM100,
consumer SM120) is still needed before finalizing the cleanup decision.**

- **Branch:** `tlwu/20260630/qmoe_fp4_production`
- **HEAD at sweep time:** `7a050d1e0d` ("no sharing buffer for gemm and gemv")
- **GPU:** 1× NVIDIA H200 (sm_90), idle. Clock locking denied on this box; SM clock floats
  345→1980 MHz, so single-run wall-clock is bimodal. Treat differences below ~10% as noise.
- **Build:** `build/cu130_fp4_bench/Release` (`onnxruntime_USE_FP4_QMOE=ON`), CUDA 13.0,
  provider synced into `.venv_cu130` capi and `~/ort_home_cu130_fp4_bench/lib`.
- **Sweep script:** `fp4_cleanup_sweep.py` (MODE=time and MODE=autotune_log) in /scripts/h200_18/ of github.com/tianleiwu/dev
- **Date:** 2026-07-01.

## Kernels / techniques under test

The FP4 QMoE decode path (SM<120 dequant-fallback regime, default-on) fuses:
`prologue → expand → fc1 SwiGLU GEMV → fc2 GEMV → finalize`. The two GEMVs
(`launch_moe_gemv_fp4_symmetric_interleaved_swiglu` for fc1, `launch_moe_gemv_fp4_symmetric` for fc2)
support:

- **Tiling configs** (`MoeGemvConfig`, numerically bit-exact — only CtaN/Threads differ):
  - `kDefault` = CtaN 8, Threads 128
  - `kCtaN16` = CtaN 16, Threads 128
  - `kThreads64` = CtaN 8, Threads 64
- **Per-shape autotune** (`ORT_FP4_GEMV_AUTOTUNE`, default ON): on the first non-captured call, times
  all 3 configs × {fc1,fc2} and caches the fastest per shape bucket. `ORT_FP4_GEMV_AUTOTUNE_LOG` logs it.
- **Lever-A interleaved** (`ORT_FP4_GEMV_INTERLEAVED`, default OFF): `ColumnMajorInterleaved` layout +
  dtype-conditional accumulation (fp16→fp16 accum, bf16→fp32 accum). `ORT_FP4_GEMV_INTERLEAVED_HALFACC`
  is a diagnostic that forces 16-bit accum for both dtypes.

## Shapes

| model | hidden | inter | E | top_k | fc1 (n,k) | fc2 (n,k) |
|-------|-------:|------:|---:|------:|-----------|-----------|
| gpt-oss-20b | 2880 | 2880 | 32 | 4 | 5760, 2880 | 2880, 2880 |
| qwen3-a3b   | 2048 | 512  | 256| 8 | 1024, 2048 | 2048, 512  |
| gemma-a4b   | 2816 | 704  | 128| 8 | 1408, 2816 | 2816, 704  |

## Result 1 — per-config GEMV kernel timings (autotuner's own measurements, µs/call)

Derived from `ORT_FP4_GEMV_AUTOTUNE_LOG=1` (ms/20it ÷ 20 × 1000). Bit-exact configs, so lowest wins.

| shape | fc | `kDefault` | `kCtaN16` | `kThreads64` | winner |
|-------|----|-----------:|----------:|-------------:|--------|
| gpt-oss | fc1 (n5760,k2880) | **74.0** | 91.9 | 81.6 | kDefault |
| qwen3   | fc1 (n1024,k2048) | **20.6** | 24.8 | 21.7 | kDefault |
| gemma   | fc1 (n1408,k2816) | **37.3** | 51.2 | 43.4 | kDefault |
| gpt-oss | fc2 (n2880,k2880) | **37.8** | 49.0 | 41.6 | kDefault |
| qwen3   | fc2 (n2048,k512)  | 13.2 | 18.9 | **11.8** | kThreads64 (+11%) |
| gemma   | fc2 (n2816,k704)  | **21.3** | 27.7 | 26.0 | kDefault |

Notes:
- `kCtaN16` is **strictly dominated** — slowest in all 36 measurements (fp16+bf16), never selected.
- `kThreads64` wins **only** qwen3-fc2 (small k=512), by ~11% on a ~12 µs kernel.
- `kDefault` wins fc1 universally and fc2 in 2 of 3 shapes.

## Result 2 — end-to-end QMoE-node latency (µs/run), autotune vs forced-default vs Lever-A

`auto_us` = autotune ON; `dflt_us` = `ORT_FP4_GEMV_AUTOTUNE=0` (forced kDefault); `ileav_us` =
`ORT_FP4_GEMV_INTERLEAVED=1`. Averaged over 50 iters after 20 warmup.

| model | dt | tok | exp | auto_us | dflt_us | ileav_us | auto/dflt | ileav/dflt |
|-------|----|----:|----:|--------:|--------:|---------:|----------:|-----------:|
| gpt-oss | fp16 | 1 | 4 | 143.7 | 144.7 | 143.4 | 0.993 | 0.991 |
| gpt-oss | fp16 | 2 | 8 | 241.2 | 243.0 | 242.7 | 0.993 | 0.999 |
| gpt-oss | fp16 | 4 | 16 | 402.6 | 402.5 | 435.9 | 1.000 | 1.083 |
| gpt-oss | fp16 | 8 | 32 | 517.0 | 513.6 | 516.8 | 1.007 | 1.006 |
| gpt-oss | fp16 | 16 | 64 | 723.5 | 775.5 | 721.0 | 0.933 | 0.930 |
| gpt-oss | fp16 | 32 | 128 | 778.7 | 763.8 | 766.9 | 1.020 | 1.004 |
| gpt-oss | bf16 | 1 | 4 | 142.1 | 142.1 | 143.8 | 1.000 | 1.012 |
| gpt-oss | bf16 | 2 | 8 | 241.6 | 241.5 | 240.8 | 1.000 | 0.997 |
| gpt-oss | bf16 | 4 | 16 | 435.3 | 404.8 | 466.2 | 1.075 | 1.152 |
| gpt-oss | bf16 | 8 | 32 | 637.7 | 539.3 | 457.0 | 1.182 | 0.847 |
| gpt-oss | bf16 | 16 | 64 | 768.5 | 718.4 | 723.3 | 1.070 | 1.007 |
| gpt-oss | bf16 | 32 | 128 | 765.0 | 772.7 | 774.2 | 0.990 | 1.002 |
| qwen3 | fp16 | 1 | 8 | 63.3 | 64.8 | 65.4 | 0.977 | 1.009 |
| qwen3 | fp16 | 2 | 16 | 124.7 | 122.3 | 123.7 | 1.020 | 1.011 |
| qwen3 | fp16 | 4 | 32 | 139.5 | 138.7 | 140.6 | 1.005 | 1.014 |
| qwen3 | fp16 | 8 | 64 | 224.1 | 225.3 | 226.2 | 0.995 | 1.004 |
| qwen3 | fp16 | 16 | 128 | 395.4 | 391.8 | 394.2 | 1.009 | 1.006 |
| qwen3 | fp16 | 32 | 256 | 562.3 | 559.8 | 504.0 | 1.004 | 0.900 |
| qwen3 | bf16 | 1 | 8 | 65.1 | 65.0 | 66.4 | 1.002 | 1.021 |
| qwen3 | bf16 | 2 | 16 | 125.3 | 121.6 | 127.1 | 1.030 | 1.045 |
| qwen3 | bf16 | 4 | 32 | 139.6 | 138.6 | 138.8 | 1.007 | 1.001 |
| qwen3 | bf16 | 8 | 64 | 219.1 | 219.6 | 223.0 | 0.997 | 1.015 |
| qwen3 | bf16 | 16 | 128 | 378.2 | 377.4 | 378.4 | 1.002 | 1.003 |
| qwen3 | bf16 | 32 | 256 | 562.6 | 505.9 | 505.2 | 1.112 | 0.999 |
| gemma | fp16 | 1 | 8 | 94.6 | 93.5 | 91.9 | 1.011 | 0.983 |
| gemma | fp16 | 2 | 16 | 170.6 | 174.1 | 168.3 | 0.980 | 0.966 |
| gemma | fp16 | 4 | 32 | 250.3 | 247.3 | 249.8 | 1.012 | 1.010 |
| gemma | fp16 | 8 | 64 | 370.5 | 375.0 | 374.6 | 0.988 | 0.999 |
| gemma | fp16 | 16 | 128 | 520.6 | 522.7 | 520.8 | 0.996 | 0.996 |
| gemma | fp16 | 32 | 256 | 659.0 | 657.8 | 665.9 | 1.002 | 1.012 |
| gemma | bf16 | 1 | 8 | 91.3 | 91.4 | 94.2 | 0.999 | 1.030 |
| gemma | bf16 | 2 | 16 | 173.2 | 173.6 | 172.5 | 0.998 | 0.994 |
| gemma | bf16 | 4 | 32 | 256.1 | 253.1 | 250.6 | 1.012 | 0.990 |
| gemma | bf16 | 8 | 64 | 368.8 | 371.0 | 378.3 | 0.994 | 1.020 |
| gemma | bf16 | 16 | 128 | 518.4 | 522.8 | 516.7 | 0.992 | 0.988 |
| gemma | bf16 | 32 | 256 | 659.8 | 660.0 | 663.9 | 1.000 | 1.006 |

## Findings (H200)

1. **Autotune does not help end-to-end.** `auto/dflt` is frequently **> 1.0** (autotune slower than
   forced kDefault; e.g. gpt-oss bf16 tok=8 at 1.18×) and never below the ~10% clock-noise floor. The
   3-way profiling sweep + per-shape cache + 2 env vars buy nothing net-positive on H200.
2. **`kCtaN16` is dead.** Slowest in every measurement; never selected.
3. **`kThreads64` is marginal.** Wins only qwen3-fc2 (small k), ~11% on a ~12 µs kernel; washed out
   end-to-end.
4. **`kDefault` (CtaN 8, Threads 128) is the universal best** single tiling for both fc1 and fc2.
5. **Lever-A interleaved is a wash.** `ileav/dflt` ranges 0.85–1.15 with no consistent direction;
   off by default and never selected.

## Recommendations (pending multi-GPU confirmation)

If A100 / other GPUs corroborate H200:

- **Remove** `kCtaN16` and `kThreads64`; keep only `kDefault` as a single static tiling.
- **Remove** the GEMV autotune machinery: `fp4_gemv_tune_cache_`, the sweep in `ComputeInternal`,
  the `MoeGemvConfig` enum, and env vars `ORT_FP4_GEMV_AUTOTUNE` / `ORT_FP4_GEMV_AUTOTUNE_LOG`.
- **Remove** the Lever-A interleaved path: env vars `ORT_FP4_GEMV_INTERLEAVED` /
  `ORT_FP4_GEMV_INTERLEAVED_HALFACC`, the `ColumnMajorInterleaved` GEMV specializations, and the
  interleaved branch of `PrePackRepackFP4Weights`.
- **Separately evaluate for removal** the native SM90 CUTLASS WFP4A16 path
  (`ORT_ENABLE_FP4_CUTLASS_GEMM` + `ORT_ENABLE_FP4_CUTLASS_UNSAFE`): documented ~50× slower than vLLM
  at prefill, numerically unsafe, two opt-in flags, never a default. Largest code removal (vendored
  CUTLASS header diffs, TMA-WS launcher, native prepack + dual-runner). Risky — do as its own PR.
- **Keep** the production defaults: FP4 GEMV decode, SM80 FP4 grouped-GEMM prefill (default ON), and
  the dequant fallback (safety net).

## Open items / data still needed

- **A100 (sm_80):** re-run `~/fp4_cleanup_sweep.py` (MODE=time and MODE=autotune_log). A100 has locked
  clocks → cleaner signal; confirm kDefault dominance and whether kThreads64 ever wins a non-trivial
  kernel.
- **B200 / SM100 and consumer SM120:** these regimes may take the native TMA path instead of the
  SM80/GEMV fallback; verify the cleanup does not remove a kernel those GPUs rely on.
- Consider `nsys`/`ncu` clock-normalized deltas for the Lever-A fp16 case to confirm the wash before
  deleting the interleaved kernel specializations.

## Reproduction

```bash
export PATH=/home/tianlei/cuda13.0/bin:$PATH
cd ~/onnxruntime/build/cu130_fp4_bench/Release && ninja onnxruntime_providers_cuda
cp libonnxruntime_providers_cuda.so \
   ~/onnxruntime/.venv_cu130/lib/python3.14/site-packages/onnxruntime/capi/
cp libonnxruntime_providers_cuda.so ~/ort_home_cu130_fp4_bench/lib/

cd /tmp
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/home/tianlei/ort_home_cu130_fp4_bench/lib:/home/tianlei/onnxruntime/build/cu130_fp4_bench/Release:/home/tianlei/cuda13.0/lib64:$LD_LIBRARY_PATH
PY=/home/tianlei/onnxruntime/.venv_cu130/bin/python
MODE=time $PY ~/fp4_cleanup_sweep.py                       # end-to-end table
MODE=autotune_log TOKS=1,4,16,32 $PY ~/fp4_cleanup_sweep.py 2>/tmp/autotune_log.txt
grep "autotune candidate" /tmp/autotune_log.txt            # per-config kernel timings
```
