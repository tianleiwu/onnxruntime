# GPT-OSS-20B CUDA Optimization Notes

This note summarizes the current CUDA decode performance state for `gpt-oss-20b`, based on the local benchmark scripts, the supplied ONNX model variant, recent Olive recipe experiment logs, and a source read of the relevant ORT CUDA kernels. The target model for the first round of work is:

```text
/tianlei/models/gpt-oss-20b/variants/cuda_int4_int4_qmoe_rtn_matmul_only
```

The benchmark entry points are:

```bash
MODEL=/tianlei/models/gpt-oss-20b/variants/cuda_int4_int4_qmoe_rtn_matmul_only \
  bash /tianlei/scripts/h200_18/bench_gpt_oss_ort_decode.sh

MODEL=/tianlei/models/gpt-oss-20b/variants/cuda_int4_int4_qmoe_rtn_matmul_only \
  bash /tianlei/scripts/h200_18/bench_gpt_oss_ort_decode_sweep.sh

bash /tianlei/scripts/run_gpt_oss.sh
```

Important: CUDA `GroupQueryAttention` XQA is not enabled by default for the GPT-OSS non-quantized KV-cache path, so benchmark scripts should still opt in with `ORT_ENABLE_XQA=1` for negative-control testing. However, the supplied GPT-OSS graph also wires `head_sink` into `GroupQueryAttention`, which sets smooth-softmax mode and currently disqualifies XQA. The observed path for this model is FlashAttention/FlashDecode, not XQA. `/tianlei/scripts/run_gpt_oss.sh` now exports `ORT_ENABLE_XQA=1` by default and normalizes generated model configs to `enable_cuda_graph=1` and `enable_skip_layer_norm_strict_mode=0`.

## Current Baseline

Local short run on GPU 0, H200, CUDA graph enabled, batch 1, prompt 512, generation 128, warmup 1, repeat 2:

| Metric | Value |
|---|---:|
| Prompt processing throughput | 24468.6 tok/s |
| Decode latency | 2.953 ms/token |
| Decode throughput | 338.6 tok/s |
| Sampling latency | 0.056 ms/token |

The benchmark script printed that it saved `/tmp/ort_gptoss20b_rtn_matmul_only_short.csv`, but that file was not present after the run, so the numbers above come from the benchmark stdout. The same script restored `genai_config.json` after temporarily enabling CUDA graph.

The longer experiment record in `~/olive-recipes/gpt-oss-20b/gpt_oss_20b_experiments.md` reports these representative batch-1 CUDA graph results:

| Model | MMLU | Size | Prompt 512 decode | Notes |
|---|---:|---:|---:|---|
| `rtn_matmul_only` | 0.7980 | 11.76 GiB | 297.7 tok/s | Fastest ORT decode in recorded sweep |
| `k_quant_mixed` | 0.8085 | 12.14 GiB | 270.7 tok/s | Best recorded accuracy |
| `rtn_mixed_lmh8_bs64` | 0.8010 | 11.58 GiB | 256.4 tok/s | Mixed int8 layers and int8 lm_head |
| `rtn_mixed_lmh8_bs64`, strict mode off | 0.8010 model weights | 11.58 GiB | 275.6 tok/s | Config-only speedup |
| `foundry_cuda_v1` | 0.7929 | 11.04 GiB | 260.0 tok/s | Per-channel MoE scales, CUDA graph on |

Cross-engine notes from the same experiment log:

| Engine/model | Decode behavior |
|---|---|
| ORT `rtn_matmul_only` | Approximately 297 tok/s across 256 to 512 prompt length, gently falling to 276 tok/s at 2048 |
| ORT `k_quant_mixed` | Approximately 269 tok/s at 256 to 512, falling to 251 tok/s at 2048 |
| llama.cpp MXFP4 | Approximately 296 to 297 tok/s across 256 to 2048, flatter with sequence length |
| vLLM bf16 | Approximately 279 tok/s at 256, falling to 263 tok/s at 2048 |

The local short run is faster than the recorded 5-run result, likely due to branch/build differences, run length, GPU state, or benchmark noise. Use the longer sweep for trend comparisons and rerun a standardized sweep before claiming a regression or improvement.

## Model Graph Summary

The supplied model has 276 ONNX nodes:

| Op type | Count | Notes |
|---|---:|---|
| `MatMulNBits` | 73 | All 4-bit, block size 32 |
| `Add` | 72 | Residual adds |
| `SkipSimplifiedLayerNormalization` | 48 | Two per layer |
| `GroupQueryAttention` | 24 | 12 sliding-window, 12 full attention |
| `Reshape` | 24 | GQA shape plumbing |
| `QMoE` | 24 | One per layer, `k=4`, 32 experts |

Key dimensions:

| Component | Shape/attribute |
|---|---|
| Hidden size | 2880 |
| Layers | 24 |
| Query heads / KV heads | 64 / 8 |
| Head size | 64 |
| Attention pattern | 12 layers `local_window_size=128`, 12 layers `local_window_size=-1` |
| QKV projection | 24 x `MatMulNBits(K=2880, N=5120, bits=4, block_size=32)` |
| O projection | 24 x `MatMulNBits(K=4096, N=2880, bits=4, block_size=32)` |
| Router projection | 24 x `MatMulNBits(K=2880, N=32, bits=4, block_size=32)` |
| lm_head | 1 x `MatMulNBits(K=2880, N=201088, bits=4, block_size=32)` |
| QMoE | 24 x int4, block size 32, `activation_type=swiglu`, `swiglu_fusion=1`, `normalize_routing_weights=1` |

## Immediate Findings

### 1. CUDA graph, XQA, and skip-layernorm mode are first-order controls

The decode benchmark explicitly enables CUDA graph before running. Earlier Olive builds left `enable_cuda_graph=0`, causing a large decode throughput loss due to per-token launch overhead. Any build or benchmark intended for decode comparison should normalize:

```json
"enable_cuda_graph": "1",
"enable_skip_layer_norm_strict_mode": "0"
```

and should set:

```bash
export ORT_ENABLE_XQA=1
```

`ORT_ENABLE_XQA=1` is necessary but not sufficient for this model. The current GPT-OSS graph has a non-empty `head_sink` input on `GroupQueryAttention`; ORT converts that to `parameters.use_smooth_softmax=true`, and the XQA eligibility gate requires `!parameters.use_smooth_softmax`. As a result, the current model selects FlashAttention/FlashDecode even when `ORT_ENABLE_XQA=1` is set.

The experiment log attributes about 277 us/token of the `rtn_matmul_only` vs `rtn_mixed_lmh8_bs64` gap to `enable_skip_layer_norm_strict_mode=1`, which disables the fused `SkipLayerNormKernelSmall` and falls back to slower standalone layer norm kernels. Flipping the flag from `1` to `0` recovered about 7 percent decode throughput in the recorded sweep.

Action: make the benchmark scripts and generated `genai_config.json` agree on these settings before comparing kernel changes. `/tianlei/scripts/h200_18/bench_gpt_oss_ort_decode.sh` and `/tianlei/scripts/run_gpt_oss.sh` now default to `XQA=1`, `CUDA_GRAPH=1`, and `SKIP_LAYER_NORM_STRICT_MODE=0`; set `XQA=0` or `CUDA_GRAPH=0` only for negative controls.

### Profiling update: observed GQA path on H200

Profiling was run with CUDA graph disabled to expose per-kernel attribution:

```bash
env MODEL=/tianlei/models/gpt-oss-20b/variants/cuda_int4_int4_qmoe_rtn_matmul_only \
  PROMPT_LEN=512 GEN_LEN=8 REPS=1 WARMUP=0 SYNC_LIB=0 CUDA_GRAPH=0 \
  SKIP_LAYER_NORM_STRICT_MODE=0 XQA=1 \
  /home/tianlei/cuda13.0/bin/nsys profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true \
    -o /tmp/gptoss_gqa_xqa1_cg0 \
    bash /tianlei/scripts/h200_18/bench_gpt_oss_ort_decode.sh
```

The matching `XQA=0` run used the same command with `XQA=0` and output prefix `/tmp/gptoss_gqa_xqa0_cg0`. Nsight Systems summaries are in:

```text
/tmp/gptoss_gqa_xqa1_cg0.nsys-rep
/tmp/gptoss_gqa_xqa1_cg0_stats_cuda_gpu_kern_sum.csv
/tmp/gptoss_gqa_xqa0_cg0.nsys-rep
/tmp/gptoss_gqa_xqa0_cg0_stats_cuda_gpu_kern_sum.csv
```

Short debug runs with `ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO=1`, `PROMPT_LEN=512`, `GEN_LEN=2`, `CUDA_GRAPH=0` printed 48 `GroupQueryAttention` lines for each XQA setting. Every line reported:

```text
Operator=GroupQueryAttention ... DataType=fp16 SdpaKernel=FLASH_ATTENTION
```

So the selected path is:

| Phase | Observed path | Evidence |
|---|---|---|
| Prompt/prefill | FlashAttention | `flash_fwd_kernel` rows and attention debug `SdpaKernel=FLASH_ATTENTION` |
| Decode | FlashDecode / split-K FlashAttention | `flash_fwd_splitkv_kernel` and `flash_fwd_splitkv_combine_kernel`; 168 split-K launches in the `GEN_LEN=8` profile equals 7 decode steps x 24 layers |
| XQA | Not selected | `XQA=1` and `XQA=0` debug logs both report `FLASH_ATTENTION` for every GQA invocation |
| cuDNN SDPA | Not selected | debug header reports `CUDNN_FLASH_ATTENTION=0` |

Do not use `nvjet_hsh_*` kernel names alone as XQA evidence for this workload. They appeared in both `XQA=1` and `XQA=0` nsys summaries, while ORT debug output showed the selected GQA path was FlashAttention in both runs.

The exact XQA blocker is `head_sink`: the ONNX node has input 11 set, for example `model.layers.0.attn.sinks`. In `GroupQueryAttention`, `head_sink != nullptr` sets `parameters.use_smooth_softmax=true`; XQA requires `!parameters.use_smooth_softmax`. The graph attributes otherwise look XQA-friendly for the 12 global layers (`softcap=0`, `head_size=64`, group size 8, shared cache configured), while the 12 sliding-window layers still have `local_window_size=128` and are independently ineligible for today's XQA path.

In the no-CUDA-graph nsys runs, total CUDA kernel time was dominated by QMoE grouped GEMM rather than attention. Representative grouped categories from the kernel summary:

| Category | XQA=1 total | XQA=0 total | Notes |
|---|---:|---:|---|
| QMoE CUTLASS GEMM | 5323.5 ms | 5302.9 ms | dominant in this profiling mode |
| FlashDecode split-K kernels | 3.72 ms | 3.72 ms | decode attention split-K launches |
| FlashDecode combine kernels | 0.54 ms | 0.55 ms | decode attention combine launches |
| FlashAttention main kernels | 0.81 ms | 0.78 ms | prefill/other Flash kernels in the summary |
| `MatMulFloatInt4Kernel` | 4.89 ms | 4.96 ms | non-MoE int4 matmul fallback rows |

No-graph throughput under nsys was about 10 tok/s decode, so these numbers should be used only for path attribution, not end-user throughput. Use CUDA graph enabled for throughput comparisons.

### Implementation update: local-window FlashDecode split planning

A scoped FlashDecode optimization was implemented in `onnxruntime/contrib_ops/cuda/bert/group_query_attention.cc`. For fast decode with `local_window_size > 0`, split-K planning now uses the effective local-window KV length instead of the full total KV sequence length:

```cpp
size_t sequence_length_for_split = static_cast<size_t>(parameters.total_sequence_length);
if (data.use_flash_attention_fast_decode && parameters.local_window_size > 0) {
  sequence_length_for_split = std::min(sequence_length_for_split, static_cast<size_t>(parameters.local_window_size));
}
```

The Flash kernel already applies the sliding-window mask, so this change only avoids over-planning split-K scratch and combine work for local-window decode layers. It does not change global attention layers and does not alter the head-sink/smooth-softmax path.

Validation:

```bash
cd /home/tianlei/onnxruntime/onnxruntime/test/python/transformers
PYTHONPATH=/home/tianlei/onnxruntime/build/cu130/Release:$PYTHONPATH \
  ORT_TEST_CUDA_PLUGIN_EP=1 PIPELINE_MODE=1 \
  python test_gqa.py -k 'TestFlashGQA.test_gqa_past_flash_attention'
```

Result: 4 Flash GQA past/decode tests passed.

Nsight Systems was rerun with the rebuilt provider synced into the benchmark venv:

```bash
env MODEL=/tianlei/models/gpt-oss-20b/variants/cuda_int4_int4_qmoe_rtn_matmul_only \
  PROMPT_LEN=512 GEN_LEN=8 REPS=1 WARMUP=0 SYNC_LIB=1 CUDA_GRAPH=0 \
  SKIP_LAYER_NORM_STRICT_MODE=0 XQA=1 \
  /home/tianlei/cuda13.0/bin/nsys profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=true \
    -o /tmp/gptoss_flashdecode_localwin_opt \
    bash /tianlei/scripts/h200_18/bench_gpt_oss_ort_decode.sh
```

Artifacts:

```text
/tmp/gptoss_flashdecode_localwin_opt.nsys-rep
/tmp/gptoss_flashdecode_localwin_opt_stats_cuda_gpu_kern_sum.csv
/tmp/ort_gptoss20b_flashdecode_localwin_opt.csv
```

Launch-count comparison for the same `PROMPT_LEN=512`, `GEN_LEN=8`, no-CUDA-graph profile:

| Kernel category | Before | After | Notes |
|---|---:|---:|---|
| `flash_fwd_splitkv_kernel` | 168 launches, 3.72 ms | 168 launches, 5.83 ms | FlashDecode kernel name remains used for both global and local-window decode layers |
| `flash_fwd_splitkv_combine_kernel` | 168 launches, 0.54 ms | 84 launches, 0.27 ms | Combine pass now runs for the 12 global layers only: 7 decode steps x 12 layers |
| `flash_fwd_kernel` | 24 launches, 0.81 ms | 24 launches, 0.90 ms | Prompt/prefill kernels unchanged |
| QMoE CUTLASS GEMM | 5323.5 ms | 5297.9 ms | Still dominates this no-CUDA-graph nsys profile |

The measured wall-clock decode latency in the profiled no-CUDA-graph nsys run changed from about 98.0 ms/token to 97.3 ms/token. Treat this as directional only because attention is a tiny fraction of this profiling-mode run and QMoE dominates total kernel time. The robust signal is the split-K combine launch reduction from 168 to 84.

### 2. The fastest current ORT decode model is already competitive with llama.cpp

In the recorded sweep, `rtn_matmul_only` and llama.cpp are both about 297 tok/s for prompt lengths 256 to 512. The bigger remaining gap is sequence-length robustness: llama.cpp decode stays flat through 2048, while ORT loses about 7 percent. That points toward attention/KV-cache work rather than pure weight-only GEMV throughput.

Action: profile decode at prompt 512 and 2048 for `rtn_matmul_only`, llama.cpp, and ORT with attention debug enabled. Attribute the 2048-token ORT loss to `GroupQueryAttention` path selection, KV append, RoPE, sequence-length helper kernels, or memory traffic.

### 3. lm_head and promoted int8 layers are bandwidth costs, not inefficient kernels

The experiment log measured int4 and int8 `MatMulNBits` at roughly the same bandwidth, about 1.56 TB/s. The int8 variants are slower because they read twice the weight bytes, not because the CUDA int8 path is fundamentally worse. For decode, the `lm_head` is a large `2880 x 201088` projection, so promoting it to int8 costs about 187 us/token in the recorded nsys attribution.

Action: prefer all-int4 `lm_head` for pure throughput targets. Keep mixed/int8 variants as accuracy tradeoffs, not as performance optimizations.

### 4. QMoE block size is not the decode bottleneck, but MoE scale layout affects size and prefill

The recorded nsys comparison found no decode cost from QMoE block size 32 vs 64. However, Foundry's smaller model size is mainly due to per-channel MoE scales instead of block-wise scales. Block-wise int4 scales for this model cost about 1.15 GiB, while per-channel scales cost about 25 MiB.

Action: for decode throughput, do not prioritize QMoE block-size kernel tuning until profiling shows QMoE is hot. For size and prefill, investigate a QMoE mode that supports the Foundry-style per-channel scale layout with acceptable accuracy.

## Relevant ORT CUDA Paths

| Area | Files | Current state |
|---|---|---|
| `MatMulNBits` CUDA | `onnxruntime/contrib_ops/cuda/quantization/matmul_nbits.cc`, `matmul_4bits.cu`, `matmul_8bits.cu` | Uses fpA_intB prepacking when available, with fallback dequant plus cuBLAS. Supports 4-bit and 8-bit, optional zero-points, and chunked dequant for large outputs. |
| QMoE CUDA | `onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc`, `qmoe_kernels.cu`, `contrib_ops/cuda/llm/moe_gemm/*` | CUTLASS MoE runner, fused SwiGLU, in-kernel softmax-topk helpers. Integer QMoE uses Ampere-style grouped GEMM layout even on SM90. |
| GQA CUDA | `onnxruntime/contrib_ops/cuda/bert/group_query_attention.cc`, `group_query_attention_impl.cu` | Supports XQA, cuDNN SDPA, Flash Attention, memory-efficient attention, and unfused fallback. XQA is opt-in for the GPT-OSS non-quantized KV path via `ORT_ENABLE_XQA=1`; it rejects softcap, smooth softmax/head sink, and local-window attention. |
| Skip/RMSNorm fusion | `onnxruntime/contrib_ops/cuda/bert/skip_layer_norm_impl.cu`, `onnxruntime/core/optimizer/skip_layer_norm_fusion.*` | CUDA has fused skip layer norm. Strict mode can prevent the fast path. Standalone Add + SimplifiedLayerNorm fusion is not the same as regular LayerNorm fusion. |
| Existing MatMulNBits QKV fusion | `onnxruntime/core/optimizer/matmul_nbits_qkv_fusion.*` | Graph transformer exists, but the fused `MatMulNBitsQkv` contrib op is WebGPU-only today. |
| Existing GQA pre-norm fusion | `onnxruntime/core/optimizer/group_query_attention_pre_norm_fusion.*` | WebGPU-only. CUDA `GroupQueryAttention` rejects q/k norm inputs in slots 14 and 15. |

## Prioritized Optimization Plan

### P0: Standardize measurement

1. Use `/tianlei/scripts/h200_18/bench_gpt_oss_ort_decode_sweep.sh` for prompt lengths 256, 512, 1024, and 2048, with `REPS=5`, `WARMUP=2`, `CUDA_GRAPH=1`, `XQA=1`, `SKIP_LAYER_NORM_STRICT_MODE=0`, `BATCH=1`.
2. Always log the ORT commit, provider library timestamp, model path, GPU id, `genai_config.json` CUDA provider options, and environment variables such as `ORT_ENABLE_XQA`, `ORT_DISABLE_FLASH_DECODE`, `ORT_DISABLE_FLASH_ATTENTION`, and `ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION`.
3. Add a negative control with `CUDA_GRAPH=0` and strict mode toggles so config regressions are obvious.
4. For profiling runs, disable CUDA graph to expose per-kernel attribution, then rerun with CUDA graph on to confirm wall-clock impact.

Expected outcome: avoid confusing config changes with kernel changes. This is the cheapest and highest-confidence step.

### P1: Keep the fused skip/RMSNorm path enabled

The current graph already has 48 `SkipSimplifiedLayerNormalization` nodes. The recorded profiler data shows strict mode can force slower layer norm kernels. This is a config/build hygiene item rather than a new operator.

Work items:

1. Ensure model builder output for CUDA decode defaults to `enable_skip_layer_norm_strict_mode=0` when numerically acceptable.
2. Add a small benchmark or GenAI config test that fails if generated CUDA configs disable CUDA graph or enable strict skip-layernorm mode for this model family.
3. Verify accuracy parity or tolerance for strict mode off on MMLU smoke tests, because it changes optimization eligibility rather than model weights.

Expected gain: up to about 7 percent on variants currently using strict mode. The supplied `rtn_matmul_only` model already uses strict mode off.

### P1: Improve `GroupQueryAttention` decode path selection

GPT-OSS alternates local-window and full attention, and every layer includes `head_sink`. CUDA XQA is a strong decode path, but for this non-quantized KV-cache path it must be explicitly enabled with `ORT_ENABLE_XQA=1` and currently rejects both local-window attention and smooth-softmax/head-sink mode. The profiled model therefore uses FlashAttention/FlashDecode for all 24 layers. The supplied model has 12 local-window layers, so those remain ineligible for today's XQA path even if head-sink support is added.

Work items:

1. Keep attention debug logging in the profiling recipe so the selected path is recorded explicitly. For the current model, the expected debug line is `SdpaKernel=FLASH_ATTENTION` for all GQA invocations.
2. Compare prompt 512 vs 2048 kernel time for full-attention layers and sliding-window layers separately, focusing on FlashDecode split-K behavior and KV-cache memory traffic.
3. Evaluate whether XQA should support GPT-OSS head-sink/smooth-softmax semantics for the 12 global layers. This is a correctness-sensitive kernel design item because the current XQA launch does not take `head_sink`.
4. Prototype or evaluate a sliding-window XQA decode path for `head_size=64`, `num_heads=64`, `kv_num_heads=8`, group size 8, batch 1, and shared KV cache only after the global-layer head-sink question is resolved.
5. If XQA support is too invasive, reduce overhead in the selected FlashDecode path: avoid sequence-length helper launches for single-token decode when possible, avoid unnecessary scratch allocation, and keep RoPE/KV append fused with the selected attention path.

Expected gain: likely the best path to close the ORT-vs-llama.cpp long-context decode gap, because llama.cpp remains flat across prompt length while ORT loses about 7 percent by 2048. The immediate work should target the measured FlashDecode path and only move to XQA after head-sink semantics are designed.

### P2: Design CUDA support for fused MatMulNBits QKV

The graph currently has one packed QKV projection per layer: `MatMulNBits(K=2880, N=5120)`, followed by GQA. That is already better than three separate projections. However, ORT also has a WebGPU-only `MatMulNBitsQkv` fusion that can absorb normalization/residual plumbing and expose separate Q/K/V outputs without materializing the same intermediate pattern.

For CUDA, two options are worth comparing:

1. Extend CUDA `GroupQueryAttention` to accept a fused `MatMulNBits` QKV input and apply the split/reshape/RoPE internally.
2. Implement CUDA kernel support for the existing fused `MatMulNBitsQkv` op or a CUDA-specific equivalent, then let the transformer target CUDA.

Design constraints:

1. The current QKV output size is 5120 = query 4096 + key 512 + value 512.
2. Decode has `M=1`, so weight-only GEMV launch overhead and intermediate writes matter.
3. The fused path must preserve past/present sharing, RoPE, alternating local-window/full attention, and CUDA graph capture.
4. Any fused op should be optional and guarded by exact shape/attribute checks.

Expected gain: uncertain until profiling. This mainly reduces launches and memory traffic around QKV projection and attention setup. It is a larger design item than the config fixes.

### P2: Fuse or specialize router projection plus QMoE gating

Each layer has a tiny router projection `MatMulNBits(K=2880, N=32)` feeding QMoE, and QMoE then runs `SoftmaxTopK` for 32 experts and `k=4`. The current QMoE kernel has optimized softmax-topk paths, but the router matmul is a separate op and materializes logits.

Possible design:

```text
hidden -> router MatMulNBits -> softmax/topk -> QMoE
```

becomes either:

```text
hidden -> QMoEWithRouter
```

or a lighter fusion where QMoE accepts optional router weight/scales and computes logits internally before top-k.

Design constraints:

1. Router output is tiny, so the main benefit is launch reduction and avoiding logits materialization, not GEMV bandwidth.
2. The fused op must preserve quantization formats and support `normalize_routing_weights=1` and `k=4`.
3. The benefit may disappear under CUDA graph if the router kernel time is small. Profile first.

Expected gain: likely modest for batch 1 decode, but could help graph launch count and improve prefill/batched decode. Do only after profiling shows router plus top-k is material.

### P2: Evaluate a per-channel-scale QMoE variant

Foundry is smaller because it uses per-channel MoE scales. ORT's current block-wise scales add about 1.13 GiB compared with per-channel scales for this model. The decode throughput impact is not clearly positive, but smaller weights/scales can improve memory residency and model distribution.

Work items:

1. Add or prototype a QMoE quantization mode with per-channel expert scales for int4 weights.
2. Confirm whether current CUTLASS MoE runner can consume this layout directly or needs a new runner/prepack path.
3. Evaluate MMLU against `rtn_matmul_only`, `k_quant_mixed`, and Foundry.
4. Benchmark prefill separately from decode. Foundry currently wins prefill but not decode.

Expected gain: model size reduction and possibly prefill improvement. Do not expect immediate decode improvement without profiling evidence.

### P3: lm_head specialization

The all-int4 `lm_head` is already the fastest recorded choice for decode. Still, because it is a very large `M=1, K=2880, N=201088` projection, small improvements can matter.

Work items:

1. Profile `lm_head` `MatMulNBits` separately with the fpA_intB GEMV path and verify selected tactics on SM90.
2. Compare int4 symmetric RTN against k-quant asymmetric with zero-points for bandwidth, not just accuracy.
3. Investigate whether top-k sampling can consume partial logits or whether a fused lm_head plus sampling path is feasible. This is invasive because logits are a graph output and sampling lives in GenAI.

Expected gain: limited unless sampling/logits materialization can be fused. Keeping `lm_head` int4 is already the main performance choice.

## Suggested Profiling Matrix

Run each with prompt lengths 512 and 2048, generation 128:

| Model/config | Purpose |
|---|---|
| `rtn_matmul_only`, CUDA graph on | Main throughput baseline |
| `rtn_matmul_only`, CUDA graph off with nsys | Per-kernel attribution |
| `rtn_matmul_only`, `ORT_ENABLE_XQA=1` and `ORT_ENABLE_XQA=0` | GQA path sensitivity |
| `k_quant_mixed`, CUDA graph on/off | Accuracy-oriented ORT comparison |
| `rtn_mixed_lmh8_bs64`, strict mode on/off | Revalidate layernorm config effect |
| llama.cpp MXFP4 | Decode flatness comparison |
| vLLM bf16 | Framework comparison |

Minimum data to capture:

1. Decode tok/s and ms/token.
2. Prefill tok/s.
3. Top kernels by total time and calls/token.
4. GQA path selection per layer.
5. MatMulNBits tactic choice for qkv, o_proj, router, and lm_head.
6. QMoE GEMM tactics and top-k time.

## Recommended First Implementation Sequence

1. Normalize configs and benchmark harness.
2. Add debug/profiling instrumentation for per-layer GQA path selection and MatMulNBits/QMoE tactic logging if existing logs are insufficient.
3. Optimize or add sliding-window decode support in the XQA/fast attention path if profiling confirms the 12 local-window layers are falling back to slower kernels.
4. Prototype CUDA support for a fused QKV/attention path only after the attention path selection data is clear.
5. Prototype router-plus-QMoE fusion only if router/top-k launch time is visible after CUDA graph and attention fixes.
6. Explore per-channel QMoE scales as a size/prefill project, with accuracy gates.

## Open Questions

1. Why did the local short run produce 338.6 tok/s while the longer recorded sweep reported about 297.7 tok/s for the same model class? Rerun a standardized sweep before using either number as the official baseline.
2. How much of the ORT long-context decode drop is KV-cache bandwidth versus FlashDecode split-K behavior, helper kernels, or RoPE/KV append overhead?
3. Can XQA be extended to correctly support GPT-OSS `head_sink`/smooth-softmax for the 12 global layers?
4. Can CUDA safely use the WebGPU-style pre-norm/QKV fusion contracts, or should CUDA get a separate fused operator with stricter shape requirements?
5. Is per-channel QMoE scale accuracy acceptable for the RTN model, or is Foundry using a quantization recipe that needs additional calibration?

## Bottom Line

For immediate throughput, the priority is not a new QMoE kernel. The supplied `rtn_matmul_only` model is already the fastest recorded ORT decode variant and competitive with llama.cpp at short context. The most promising performance work is:

1. eliminate config regressions (`CUDA graph=1`, strict skip-layernorm off),
2. close the long-context decode gap by improving the measured `GroupQueryAttention` FlashDecode path and designing head-sink-aware XQA only if profiling justifies it,
3. only then design CUDA graph fusions around QKV/GQA and router/QMoE to reduce launch count and intermediate memory traffic.
