# QMoE FP4 CUDA Production Experiments

This note summarizes the FP4 QMoE experiments that justify the production CUDA changes. Experimental kernels that were slower or only useful as one-off probes are intentionally excluded from the production branch.

## Retained Changes

### 1. Fused MXFP4 GEMV Decode

The SM<120 FP4 fallback regime previously dequantized all active experts to dense FP16/BF16 weights for every decode step. The retained decode fast path prepacks MXFP4 weights and scales into a GEMV-consumed layout and routes small decode shapes through a fused pipeline:

1. build expert maps,
2. expand permuted activations,
3. FC1 MXFP4 GEMV with fused SwiGLU,
4. FC2 MXFP4 GEMV,
5. finalize routing.

The path is enabled by default and can be disabled with `ORT_ENABLE_FP4_GEMV=0`. Unsupported shapes fall back to the dense dequant path.

### 2. Opt-In Interleaved GEMV Layout

The retained `ORT_FP4_GEMV_INTERLEAVED=1` path is default-off and targets FP16 decode. It combines:

- `ColumnMajorInterleaved` FP4 weight layout,
- dtype-conditional accumulation (`half` accumulation for FP16, `float` for BF16),
- fixed `CtaN=4` / `Threads=128` so prepack and dispatch agree.

A100 SM80 validation, locked 1410 MHz, `hidden=inter=2880`, `E=32`, `top_k=4`, `tokens=1`:

| dtype | baseline | interleaved | result |
|---|---:|---:|---:|
| FP16 | 172.1 us | 165.3 us | 4.0% faster |
| BF16 | 173.8 us | 176.6 us | 1.6% slower |

Correctness passed for FP16 and BF16 decode SwiGLU cases. The path remains opt-in because BF16 does not benefit: the FP32 accumulator cost erases the layout win.

### 3. Native WFP4A16 Prefill With Dense Fallback

The SM90 native CUTLASS WFP4A16 path is retained for production shapes that are not 256-aligned, including GPT-OSS `hidden=inter=2880`. The retained scale-factor tail-padding change pads only the TMA scale-factor buffer tail; activations and weights are unchanged.

The native path wins at low per-expert token counts and loses when each expert is large enough for dense A16 grouped GEMM to be compute-bound. The production code therefore keeps both native FP4 and dense A16 runners and routes by average tokens per expert using `ORT_FP4_NATIVE_MAX_TOKENS_PER_EXPERT`.

Validation on H200/SM90:

- `test_qmoe_fp4_cuda.py`: 20/20 pass.
- compute-sanitizer memcheck on the large-M dense route: 0 errors.
- GPT-OSS shape with default threshold 128:

| tokens | per-expert tokens | route | result vs dense fallback |
|---:|---:|---|---:|
| 512 | 64 | native | 1.66x faster |
| 2048 | 256 | dense | parity |
| 4096 | 512 | dense | parity |

### 4. SM80 WFP4A16 Grouped GEMM

The largest production win is routing WFP4A16 through the Ampere SM80 fused-dequant grouped GEMM instead of the SM90/TMA path or dense fallback. The retained changes:

- allow e2m1 weights with groupwise scales in the SM80 mixed-input kernel,
- add the e2m1 interleaved converter support needed by the SM80 path,
- prepack SM80 interleaved FP4 weights and activation-dtype group scales,
- select SM80 configs for WFP4A16 when `ORT_FP4_SM80_GEMM` is enabled,
- keep `ORT_FP4_SM80_GEMM=0` as the dense fallback comparison knob.

The SM80 path is default-on in the FP4 fallback regime unless native FP4 CUTLASS is explicitly requested with `ORT_ENABLE_FP4_CUTLASS_GEMM=1`.

#### A100 FP16 Sweep

GPU: A100-SXM4-80GB, SM80, locked 1410 MHz. Shape: GPT-OSS QMoE node (`hidden=inter=2880`, `E=32`, `top_k=4`), `--warmup 5 --iters 20 --reps 3`.

| tokens | SM80 FP4 default-on | dense fallback | speedup |
|---:|---:|---:|---:|
| 512 | 1.69445 ms | 18.20956 ms | 10.75x |
| 1024 | 2.50430 ms | 18.61366 ms | 7.43x |
| 2048 | 4.03634 ms | 19.45979 ms | 4.82x |
| 4096 | 7.35478 ms | 21.35033 ms | 2.90x |

SM80-relevant FP4 CUDA tests passed with `ORT_FP4_SM80_GEMM=1`; the SM90-only native scale-prepack case is expected to skip on A100.

#### A100 BF16 Sweep

The BF16 enablement removes the FP16-only policy gate and dispatches both `half` and `__nv_bfloat16` activations through the same Ampere WFP4A16 grouped GEMM.

Same A100 setup and shape as above:

| tokens | SM80 FP4 default-on | dense fallback | speedup |
|---:|---:|---:|---:|
| 512 | 1.70004 ms | 18.22175 ms | 10.72x |
| 1024 | 2.44945 ms | 18.61720 ms | 7.60x |
| 2048 | 4.04069 ms | 19.46783 ms | 4.82x |
| 4096 | 7.37281 ms | 21.37039 ms | 2.90x |

BF16-specific correctness headlines:

- `test_fp4_bf16_silu_basic`: max diff 0.031250.
- `test_fp4_bf16_swiglu`: max diff 0.062500.
- BF16 decode SwiGLU cases: max diff 0.062500.

## Dropped Experimental Changes

The following experiments were reviewed and intentionally not retained in this production branch:

- FP4 GEMV Split-K (`ORT_FP4_GEMV_SPLITK`): correct but slower. H200 showed about 1.2% end-to-end regression and A100 showed about 4.5% regression because the FP4 grid already saturates SMs and the second reduction pass only adds overhead.
- Triton MXFP4 grouped-GEMM spike files: useful for exploration, not integrated as the production ORT CUDA path.
- FP4 converter two-nibble store experiment: bit-identical probe with no production performance win retained here.
- Standalone profiling shell scripts and broad benchmark-result dumps: omitted from the production branch; validation commands and results are summarized above.

## Validation Commands

Typical validation on A100:

```bash
cd /home/tianlei/git/onnxruntime
source .venv/bin/activate
cmake --build build/cu130_fp4_bench/Release --target onnxruntime_providers_cuda --parallel 16

# Keep GPU idle and lock clocks before timing.
sudo -n nvidia-smi -i 0 -lgc 1410
# Run FP16/BF16 GPT-OSS-shaped sweeps with ORT_FP4_SM80_GEMM=1 and =0.
sudo -n nvidia-smi -i 0 -rgc
```

Run Python tests from `/tmp` or from the test directory rather than the repository root to avoid source-tree shadowing of the installed ONNX Runtime package.
