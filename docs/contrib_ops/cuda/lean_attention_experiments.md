# Lean Attention Decode Experiments

Date: 2026-07-05

This note records the analysis and benchmark results for evaluating whether the Lean Attention kernel added by PR
#22352 should be kept as a standalone implementation, integrated into Flash Attention dispatch, or removed after its
useful ideas are incorporated elsewhere.

## Goal

Evaluate decode-phase attention performance on CUDA, especially flash decoding, without materially increasing binary
size or build time.

Questions covered:

- Does Lean Attention beat the existing Flash Attention split-KV decode path for `MultiHeadAttention`?
- Does Lean Attention have value for `GroupQueryAttention`, where the current CUDA implementation already has XQA,
  Flash decode, and cuDNN routes?
- If Lean helps, can it be treated as a smart-dispatch idea for Flash Attention rather than as a permanently separate
  implementation?

## Environment

Build and run environment used for these measurements:

- Machine/GPU: H200 class GPU, compute capability SM90.
- ONNX Runtime wheel: `onnxruntime_gpu-1.28.0-cp314-cp314-linux_x86_64.whl`.
- Build script: `bash .env/cuda_130_lean.sh --build --install`.
- Build result: success, exit code 0.
- Build options of interest: CUDA 13.0, cuDNN 9.19, Flash Attention on, Memory Efficient Attention on, Lean Attention
  on, SM90 only.
- Build used `--skip_tests`, so C++ test binaries were not built.

Runtime setup:

```bash
source /home/tianlei/onnxruntime/.venv_cu130/bin/activate
export CUDA_HOME=/home/tianlei/cuda13.0
export CUDNN_HOME=/home/tianlei/cudnn9.19_cuda13
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$CUDA_HOME/lib64:$CUDNN_HOME/lib64:$CUDNN_HOME/lib
```

Important import detail: run Python from outside the repo root, or from a subdirectory that does not shadow the
installed package. Running from `/home/tianlei/onnxruntime` imports the source tree's `onnxruntime/` package and fails
with `ModuleNotFoundError: No module named 'onnxruntime.capi'`. Running from `/tmp` or
`onnxruntime/test/python/transformers` imports the installed wheel correctly.

Wheel validation:

```text
ort version: 1.28.0
providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Correctness Baseline

The existing Python Lean Attention parity helper was run directly because it is commented out of `test_all`.

```bash
cd /home/tianlei/onnxruntime/onnxruntime/test/python/transformers
python - <<'PY'
import test_mha
t = test_mha.TestMultiHeadAttention()
t.run_lean_attention()
print("LEAN PARITY OK")
PY
```

Result:

```text
LEAN PARITY OK
```

This validates the shipped decode shapes against the current PR #29550 fix. A lower-level C++ gtest for the exact
small-SM/divide-by-zero corner case still requires a test-enabled rebuild because the current build used `--skip_tests`.

## Implementation Notes

### Lean Attention Coverage

Lean Attention is currently wired only through contrib `MultiHeadAttention`.

The important runtime gate is effectively:

- fp16 only;
- decode only: `sequence_length == 1`;
- non-empty past KV cache;
- no attention bias;
- no key padding mask;
- `head_size == v_head_size`;
- Lean support check passes: head size 64 or 128 and `num_heads == num_heads_k`.

That last condition makes the current Lean path MHA-only. It does not cover true GQA/MQA where `kv_num_heads < num_heads`.

### GQA Existing Decode Paths

`GroupQueryAttention` is not wired to Lean Attention. On SM90, the current GQA decode hierarchy is important:

- XQA is enabled by default for eligible decode shapes.
- `ORT_ENABLE_XQA=0` is required to expose Flash or cuDNN GQA decode paths in isolation.
- `ORT_DISABLE_FLASH_DECODE=1` distinguishes Flash's non-split route from Flash decode.
- The `sdpa_kernel` provider option alone is not enough to force Flash/cuDNN if XQA remains enabled, because XQA
  outranks those routes for eligible GQA decode shapes.

## Benchmark Harness

The benchmark driver was added at:

```text
onnxruntime/test/python/transformers/benchmark_lean_decode.py
```

The driver has two sweeps:

- MHA: direct Lean vs Flash vs cuDNN vs Efficient comparison. The benchmark requests kernels through the
  `sdpa_kernel` provider option and keeps only rows where debug routing confirms the requested kernel actually ran.
  A Lean row is therefore real Lean, not a silent fallback.
- GQA: compares GQA's existing decode routes: XQA, Flash decode, Flash non-split, and cuDNN. The harness explicitly
  controls `ORT_ENABLE_XQA` and `ORT_DISABLE_FLASH_DECODE` around session construction and timing.

Timing uses CUDA events around many queued `CudaSession.infer(..., synchronize=False)` calls:

1. warm up the session;
2. record a CUDA start event;
3. enqueue `N` inference calls without per-call synchronization;
4. record a CUDA end event;
5. synchronize once and divide event elapsed time by `N`.

This is necessary because wall-clock timing with per-call synchronization had a roughly 100-130 microsecond launch/sync
floor, which hides decode kernels that differ by tens of microseconds.

Smoke comparison at `batch=1`, `heads=8`, `head_size=128`, `past=8192` shows the difference between wall-clock and
device timing: wall-clock made Flash and Lean look nearly equal, while device timing showed Flash at 105.8 us and Lean
at 142.6 us.

Commands used for the quick sweeps:

```bash
cd /home/tianlei/onnxruntime/onnxruntime/test/python/transformers
python benchmark_lean_decode.py --op mha --quick --repeats 2000 --csv /tmp/lean_mha_quick.csv
python benchmark_lean_decode.py --op gqa --quick --repeats 1000 --csv /tmp/lean_gqa_quick.csv
```

Quick-grid coverage:

- batches: 1, 4, 16;
- MHA heads: 8, 32;
- GQA ratios: 32/32, 32/8, 64/8;
- head sizes: 64, 128;
- past lengths: 512, 2048, 8192, 32768.

## MHA Quick-Sweep Results

CSV rows: 192 kernel measurements, covering 48 MHA shapes and four requested kernels per shape.

Lean vs Flash summary:

```text
Lean faster in 31/48 MHA shapes.
```

Win rate by batch:

| Batch | Lean wins | Shapes |
|---:|---:|---:|
| 1 | 7 | 16 |
| 4 | 9 | 16 |
| 16 | 15 | 16 |

Win rate by head count:

| Heads | Lean wins | Shapes |
|---:|---:|---:|
| 8 | 13 | 24 |
| 32 | 18 | 24 |

Win rate by head size:

| Head size | Lean wins | Shapes |
|---:|---:|---:|
| 64 | 17 | 24 |
| 128 | 14 | 24 |

Representative Lean wins:

| Batch | Heads | Head size | Past | Flash us | Lean us | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 64 | 2048 | 137.3 | 67.4 | 2.037 |
| 1 | 8 | 64 | 512 | 108.3 | 57.4 | 1.889 |
| 16 | 8 | 128 | 512 | 176.2 | 103.8 | 1.698 |
| 4 | 32 | 64 | 2048 | 189.4 | 119.6 | 1.583 |
| 16 | 8 | 64 | 2048 | 168.2 | 118.2 | 1.424 |
| 4 | 32 | 128 | 8192 | 716.7 | 537.6 | 1.333 |
| 16 | 8 | 128 | 32768 | 2281.0 | 1739.8 | 1.311 |
| 16 | 8 | 128 | 8192 | 661.2 | 506.7 | 1.305 |

Representative Lean losses:

| Batch | Heads | Head size | Past | Flash us | Lean us | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 8 | 64 | 2048 | 105.1 | 140.3 | 0.749 |
| 4 | 8 | 64 | 512 | 92.8 | 116.7 | 0.795 |
| 4 | 8 | 128 | 2048 | 98.8 | 121.1 | 0.816 |
| 1 | 32 | 128 | 8192 | 198.8 | 242.1 | 0.821 |
| 16 | 32 | 64 | 512 | 129.8 | 152.0 | 0.854 |
| 1 | 8 | 128 | 8192 | 85.2 | 99.1 | 0.859 |
| 4 | 32 | 128 | 2048 | 209.8 | 238.0 | 0.882 |
| 1 | 32 | 128 | 512 | 58.7 | 65.4 | 0.897 |

Full Lean-vs-Flash table:

| Batch | Heads | Head size | Past | Flash us | Lean us | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 128 | 2048 | 101.5 | 91.0 | 1.115 |
| 1 | 32 | 128 | 32768 | 554.8 | 543.9 | 1.020 |
| 1 | 32 | 128 | 512 | 58.7 | 65.4 | 0.897 |
| 1 | 32 | 128 | 8192 | 198.8 | 242.1 | 0.821 |
| 1 | 32 | 64 | 2048 | 137.3 | 67.4 | 2.037 |
| 1 | 32 | 64 | 32768 | 321.8 | 301.3 | 1.068 |
| 1 | 32 | 64 | 512 | 56.3 | 58.1 | 0.968 |
| 1 | 32 | 64 | 8192 | 139.0 | 120.9 | 1.149 |
| 1 | 8 | 128 | 2048 | 59.5 | 61.1 | 0.973 |
| 1 | 8 | 128 | 32768 | 211.3 | 225.0 | 0.939 |
| 1 | 8 | 128 | 512 | 54.8 | 56.5 | 0.970 |
| 1 | 8 | 128 | 8192 | 85.2 | 99.1 | 0.859 |
| 1 | 8 | 64 | 2048 | 123.1 | 115.6 | 1.065 |
| 1 | 8 | 64 | 32768 | 158.6 | 171.0 | 0.928 |
| 1 | 8 | 64 | 512 | 108.3 | 57.4 | 1.889 |
| 1 | 8 | 64 | 8192 | 144.5 | 148.1 | 0.976 |
| 16 | 32 | 128 | 2048 | 649.6 | 539.9 | 1.203 |
| 16 | 32 | 128 | 32768 | 9119.4 | 7688.6 | 1.186 |
| 16 | 32 | 128 | 512 | 214.7 | 188.5 | 1.139 |
| 16 | 32 | 128 | 8192 | 2309.1 | 1960.4 | 1.178 |
| 16 | 32 | 64 | 2048 | 345.1 | 323.2 | 1.068 |
| 16 | 32 | 64 | 32768 | 4731.5 | 3912.8 | 1.209 |
| 16 | 32 | 64 | 512 | 129.8 | 152.0 | 0.854 |
| 16 | 32 | 64 | 8192 | 1198.7 | 1010.5 | 1.186 |
| 16 | 8 | 128 | 2048 | 192.7 | 165.7 | 1.163 |
| 16 | 8 | 128 | 32768 | 2281.0 | 1739.8 | 1.311 |
| 16 | 8 | 128 | 512 | 176.2 | 103.8 | 1.698 |
| 16 | 8 | 128 | 8192 | 661.2 | 506.7 | 1.305 |
| 16 | 8 | 64 | 2048 | 168.2 | 118.2 | 1.424 |
| 16 | 8 | 64 | 32768 | 1235.7 | 951.3 | 1.299 |
| 16 | 8 | 64 | 512 | 71.3 | 70.4 | 1.012 |
| 16 | 8 | 64 | 8192 | 359.0 | 294.4 | 1.219 |
| 4 | 32 | 128 | 2048 | 209.8 | 238.0 | 0.882 |
| 4 | 32 | 128 | 32768 | 2528.6 | 1973.7 | 1.281 |
| 4 | 32 | 128 | 512 | 95.5 | 101.3 | 0.943 |
| 4 | 32 | 128 | 8192 | 716.7 | 537.6 | 1.333 |
| 4 | 32 | 64 | 2048 | 189.4 | 119.6 | 1.583 |
| 4 | 32 | 64 | 32768 | 1298.4 | 1016.7 | 1.277 |
| 4 | 32 | 64 | 512 | 67.7 | 67.4 | 1.005 |
| 4 | 32 | 64 | 8192 | 362.3 | 334.3 | 1.084 |
| 4 | 8 | 128 | 2048 | 98.8 | 121.1 | 0.816 |
| 4 | 8 | 128 | 32768 | 489.1 | 478.5 | 1.022 |
| 4 | 8 | 128 | 512 | 58.1 | 59.4 | 0.977 |
| 4 | 8 | 128 | 8192 | 202.9 | 167.3 | 1.213 |
| 4 | 8 | 64 | 2048 | 105.1 | 140.3 | 0.749 |
| 4 | 8 | 64 | 32768 | 419.2 | 402.0 | 1.043 |
| 4 | 8 | 64 | 512 | 92.8 | 116.7 | 0.795 |
| 4 | 8 | 64 | 8192 | 128.5 | 134.3 | 0.957 |

## GQA Quick-Sweep Results

CSV rows: 288 kernel measurements, covering 72 GQA shapes and four variants per shape.

Best decode route counts:

| Best route | Shapes |
|---|---:|
| XQA | 47 |
| cuDNN | 24 |
| Flash decode | 1 |
| Flash non-split | 0 |

Best route by GQA ratio:

| Heads / KV heads | XQA wins | cuDNN wins | Flash decode wins |
|---:|---:|---:|---:|
| 32 / 32 | 11 | 12 | 1 |
| 32 / 8 | 18 | 6 | 0 |
| 64 / 8 | 18 | 6 | 0 |

Representative `heads/kv_heads=32/8`, `head_size=128`, `past=8192` rows:

| Batch | XQA us | cuDNN us | Flash decode us | Flash non-split us | Best |
|---:|---:|---:|---:|---:|---|
| 1 | 69.9 | 94.7 | 130.4 | 103.8 | XQA |
| 4 | 118.6 | 121.2 | 234.3 | 445.2 | XQA |
| 16 | 183.8 | 209.4 | 614.8 | 936.0 | XQA |

Earlier smoke at `batch=32`, `heads/kv_heads=32/8`, `head_size=128`, `past=8192` showed the same pattern:

| Route | Latency us |
|---|---:|
| XQA | 301.9 |
| cuDNN | 296.8 |
| Flash decode | 1058.7 |
| Flash non-split | 1815.9 |

The GQA result is clear: on SM90, XQA and cuDNN dominate decode. Flash decode is not the path to optimize for true
GQA shapes in this environment. A Lean-like fork of Flash Attention would need to beat XQA/cuDNN, not just Flash decode.

## Binary Size and Build-Time Considerations

Current CMake default is `onnxruntime_USE_LEAN_ATTENTION=OFF`. When disabled, the Lean `.cu` translation units are
guarded by `#if USE_LEAN_ATTENTION`, so the default binary-size and build-time impact should be close to zero.

When enabled, Lean currently adds separate CUDA sources and a separate dispatch path. The remaining cost measurement is
to build the same SM target with Lean off and on, then compare:

```bash
stat -c '%n %s' build/<lean-off>/Release/libonnxruntime_providers_cuda.so
stat -c '%n %s' build/<lean-on>/Release/libonnxruntime_providers_cuda.so
```

and record clean build wall time for both configurations. That was not completed in this run because the available
build was already Lean-enabled and test-skipping.

## Recommendation

Do not wire Lean Attention into GQA as a new standalone decode path. GQA already routes to XQA or cuDNN for the shapes
where decode matters, and those paths are consistently faster than Flash decode in the quick sweep.

For MHA, Lean has real but shape-dependent value: 31/48 quick-sweep shapes were faster than Flash, with the strongest
wins around larger effective parallelism or some head-size-64 cases. It is not uniformly better: several low-batch or
small-context shapes regress, and cuDNN is often competitive or faster on SM90.

The best next step is not to make Lean a broad default. Instead:

1. Keep Lean behind the existing build option and runtime/provider-option controls while evaluating.
2. Add a smart-dispatch experiment for MHA only, using the quick-sweep boundary as a starting point. Dispatch to Lean
   only when `(batch_size, num_heads, head_size, past_sequence_length)` falls in a measured win region and cuDNN is not
   the better available route.
3. Prefer moving the useful scheduling idea into Flash Attention's decode heuristic or implementation, then delete the
   separate Lean code if the merged Flash path matches Lean's wins without increasing binary size/build time.
4. Do not prioritize Lean-for-GQA unless future non-SM90 or XQA-ineligible measurements show a gap against XQA/cuDNN.

## Prototype: MHA-Only Smart Dispatch

Implemented prototype location:

```text
onnxruntime/contrib_ops/cuda/bert/multihead_attention.cc
```

The prototype keeps the existing explicit Lean behavior intact:

- `sdpa_kernel=LEAN_ATTENTION` still forces Lean for supported MHA decode shapes.
- `ORT_ENABLE_LEAN_ATTENTION=1` still enables Lean for supported MHA decode shapes.
- explicit `sdpa_kernel=FLASH_ATTENTION` or `sdpa_kernel=CUDNN_FLASH_ATTENTION` still overrides the smart auto path.

The new behavior only applies when all of these are true:

- ONNX Runtime was built with `USE_LEAN_ATTENTION`;
- the user did not explicitly select an SDPA kernel through the provider option;
- Flash Attention is not disabled;
- the MHA node is in a Lean-supported decode shape;
- the shape falls in a measured quick-sweep Lean win region.

The heuristic is intentionally conservative and table-derived. It has two tiers:

- when cuDNN SDPA is unavailable or not selected, use Lean for quick-sweep regions where Lean beat Flash;
- when cuDNN SDPA is supported, use Lean only for quick-sweep regions where Lean beat both Flash and cuDNN.

Focused validation was run against the rebuilt CUDA provider in `build/cu130_lean_attention/Release`:

```text
DEFAULT on b1 h8 d64 p512   -> ort:lean
DEFAULT on b1 h8 d128 p8192 -> ort:cudnn
FLASH_ATTENTION on b1 h8 d64 p512 -> ort:flash
CUDNN_FLASH_ATTENTION on b1 h8 d64 p512 -> ort:cudnn
LEAN_ATTENTION on b1 h8 d64 p512 -> ort:lean
```

Build validation:

```text
ninja CMakeFiles/onnxruntime_providers_cuda.dir/home/tianlei/onnxruntime/onnxruntime/contrib_ops/cuda/bert/multihead_attention.cc.o
ninja onnxruntime_providers_cuda -j4
```

Both completed successfully after refreshing the build-tree Python package's provider copy for routing probes.

## Remaining Work

- Run the full shape grid, not just the quick grid, with the same device-timed harness.
- Repeat on at least one Ampere or Ada GPU, because SM90 cuDNN/XQA behavior may not represent all supported CUDA GPUs.
- Add the C++ regression test for the PR #29550 small-SM/divide-by-zero case using a test-enabled build.
- Measure Lean-off vs Lean-on CUDA provider binary size and build time.
- Promote the smart-dispatch prototype from table-derived quick-grid rules to a full-grid heuristic, then add a narrow
  correctness/routing test that verifies Lean is selected only for intended MHA decode shapes.
