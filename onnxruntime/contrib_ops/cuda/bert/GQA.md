# GroupQueryAttention (GQA) CUDA Implementation

This document describes the technical details of the GroupQueryAttention operator in ONNX Runtime's CUDA execution provider.

## Overview

GroupQueryAttention (GQA) is an optimized attention mechanism that reduces memory bandwidth by using fewer key-value heads than query heads. This implementation supports:

- **Data Types**: `float16` (MLFloat16), `bfloat16` (BFloat16)
- **Attention Backends**: Flash Attention, Cutlass Memory Efficient Attention
- **Features**: Rotary Positional Embeddings (RoPE), KV Cache, Grouped Query Heads

---

## 1. Flash Attention Conditions

Flash Attention is the preferred backend due to superior performance. It is used when ALL of the following conditions are met:

```cpp
bool use_flash_attention = !disable_flash_attention_ &&
                           onnxruntime::flash::is_supported<CudaT>(device_prop,
                                                                   parameters.head_size,
                                                                   parameters.num_heads,
                                                                   parameters.kv_num_heads);
```

### Specific Requirements:
| Condition | Requirement |
|-----------|-------------|
| **Data Type** | `float16` or `bfloat16` (2 bytes) |
| **GPU Architecture** | SM 80+ (Ampere and later) |
| **Head Size** | Supported sizes (typically 32, 64, 96, 128, 160, 192, 224, 256) |
| **Environment** | `ORT_DISABLE_FLASH_ATTENTION` not set to "1" |

### Disabling Flash Attention:
```bash
export ORT_DISABLE_FLASH_ATTENTION=1
```

---

## 2. Cutlass FMHA (Memory Efficient Attention) Conditions

When Flash Attention is unavailable, the implementation falls back to Cutlass Memory Efficient Attention:

```cpp
bool use_memory_efficient_attention =
    !use_flash_attention &&
    !disable_memory_efficient_attention_ &&
    has_memory_efficient_attention(sm, is_fp16, is_bf16, head_size, head_size);
```

### Specific Requirements:
| Condition | Requirement |
|-----------|-------------|
| **GPU Architecture** | SM 53+ (Maxwell and later) |
| **Data Type** | `float16`, `bfloat16` (SM 80+ for bf16), `float32` |
| **Head Size** | Must be supported by Cutlass kernels |
| **Environment** | `ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION` not set to "1" |

### Buffer Allocations for Memory Efficient:
- **KV Buffer**: Allocated when `num_heads != kv_num_heads` for head expansion
- **Rotary Buffer**: For explicit Q/K rotation scratch space
- **FMHA Buffer**: Workspace for Cutlass kernel (when required)

---

## 3. KV Cache Handling

The GQA implementation supports two KV cache management modes:

### 3.1 Shared Buffer Mode (`kv_share_buffer=true`)

Past and present KV tensors share the same memory. New KV values are appended in-place.

```cpp
if (parameters.kv_share_buffer) {
    // In-place append using LaunchConcatKVInPlace
    ORT_RETURN_IF_ERROR(LaunchConcatKVInPlace(...));
}
```

**Key Kernel**: `ConcatKVInPlace`
- Directly writes new KV to the shared buffer at the correct position
- Uses `seqlens_k` to determine write offset for each batch

### 3.2 Separate Buffer Mode (`kv_share_buffer=false`)

Past KV is copied and new KV is appended to create present KV.

```cpp
else {
    // Copy past + append new using LaunchConcatNewToPastKV
    ORT_RETURN_IF_ERROR(LaunchConcatNewToPastKVHelper<T>(...));
}
```

**Key Kernel**: `ConcatNewToPastKV`
- Copies past KV elements to present buffer
- Appends new KV elements with optional **Fused RoPE** rotation

### KV Cache Layout

| Format | Description | Shape |
|--------|-------------|-------|
| **BSNH** | Batch-Sequence-Heads-HeadDim | `[B, S, N, H]` |
| **BNSH** | Batch-Heads-Sequence-HeadDim | `[B, N, S, H]` |

Present KV output is always **BNSH** format: `[batch_size, kv_num_heads, seqlen_present, head_size]`

---

## 4. Rotary Positional Embeddings (RoPE) Handling

RoPE is applied to Query and Key tensors when `do_rotary=true`.

### 4.1 Rotation Modes

| Mode | Formula |
|------|---------|
| **Non-Interleaved (Half-Split)** | `[x0..x_d/2, x_d/2..x_d]` paired with `[x_d/2..x_d, x0..x_d/2]` |
| **Interleaved** | `[x0, x1, x2, x3...]` paired with `[x1, x0, x3, x2...]` |

### 4.2 Implementation Paths

#### Path A: Explicit RoPE (for `kv_share_buffer=true`)
```cpp
// Rotate Q
ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(stream, q_buffer, query, ...));

// Rotate K (explicit, required for ConcatKVInPlace which has no RoPE support)
if (parameters.kv_share_buffer) {
    ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(stream, k_buffer, key, ...));
}
```

#### Path B: Fused RoPE (for `kv_share_buffer=false`)
```cpp
// Rotate Q explicitly
ORT_RETURN_IF_ERROR(LaunchRotaryEmbeddingKernel<T>(...));

// Rotate K fused with KV append
ORT_RETURN_IF_ERROR(LaunchConcatNewToPastKVHelper<T>(
    ...,
    cos_cache, sin_cache, rotary_dim, position_ids, rotary_interleaved));
```

**Fused RoPE Advantage**: Eliminates one kernel launch and reduces memory bandwidth by rotating K in the same kernel that appends to KV cache.

### Position IDs Generation

Position IDs are generated on-device based on sequence lengths:

```cpp
// For prompts: position_ids[b,s] = s
// For tokens: position_ids[b,s] = past_seqlen + s
ORT_RETURN_IF_ERROR(LaunchSeqlensToPosIds(parameters, seqlens_k, position_ids, ...));
```

---

## 5. Kernel Details

### 5.1 Core Kernels

| Kernel | Purpose | File |
|--------|---------|------|
| `ConcatNewToPastKV` | Copy past + append new KV with optional Fused RoPE | `attention_kv_cache.cu` |
| `ConcatNewToPastKVLarge` | Same as above for large head_size * num_heads | `attention_kv_cache.cu` |
| `ConcatKVInPlace` | In-place KV append for shared buffer | `attention_kv_cache.cu` |
| `RotaryEmbeddingBSNH` | Explicit RoPE for BSNH layout | `rotary_embedding_impl.cu` |
| `UnpackQKV` | Unpack packed QKV input | `group_query_attention_impl.cu` |

### 5.2 Flash Attention Kernels

| Kernel | Purpose |
|--------|---------|
| `mha_fwd_kvcache` | Forward pass with KV cache for incremental decoding |
| `flash_fwd_splitkv_kernel` | Split-KV kernel for long sequences |

### 5.3 Memory Efficient Attention Kernels

| Kernel | Purpose |
|--------|---------|
| `MemoryEfficientAttention` | Cutlass-based FMHA for any SM |

### 5.4 Vectorization Strategy

For performance, kernels use `float2` vectorization (8 bytes = 4 fp16 values):

```cpp
// Each thread processes 4 fp16 elements
int num_elements_per_thread = 8 / sizeof(T);  // = 4 for fp16
const int H = head_size / num_elements_per_thread;

// Thread block dimensions
dim3 block(H, kv_num_heads, 1);  // H threads per head, one warp per head-row
```

### 5.5 RotaryDispatcher (Fused RoPE)

The `RotaryDispatcher` template applies RoPE within the KV append kernel:

```cpp
template<typename T, typename ElementT>
struct RotaryDispatcher {
    static __device__ void apply(T& val, const T* cos_cache, const T* sin_cache,
                                  int rotary_dim, int h_idx, int pos_id,
                                  bool interleaved, const T* new_kv_base, int in_offset);
};
```

Specializations handle:
- `float2` + `half`: Non-interleaved (half-split) and interleaved modes
- `float2` + `BFloat16`: Same patterns for bf16

---

## 6. Environment Variables

| Variable | Effect |
|----------|--------|
| `ORT_DISABLE_FLASH_ATTENTION=1` | Force Memory Efficient Attention backend |
| `ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION=1` | Disable Cutlass FMHA fallback |

---

## 7. Performance Tips

1. **Use Flash Attention**: Prefer SM 80+ GPUs for best performance
2. **Enable Fused RoPE**: Automatically used for non-shared buffer cases
3. **Use BNSH KV Format**: Matches kernel expectations for better memory access
4. **Batch Size**: Larger batches amortize kernel launch overhead
5. **Share Buffer**: Use `kv_share_buffer=true` for in-place updates in production

---

## 8. File References

| File | Description |
|------|-------------|
| [group_query_attention.cc](group_query_attention.cc) | Operator registration and buffer allocation |
| [group_query_attention_impl.cu](group_query_attention_impl.cu) | Attention backend dispatch and RoPE handling |
| [attention_kv_cache.cu](attention_kv_cache.cu) | KV cache kernels and Fused RoPE dispatcher |
| [flash_attention/flash_api.h](flash_attention/flash_api.h) | Flash Attention API |
| [cutlass_fmha/memory_efficient_attention.h](cutlass_fmha/memory_efficient_attention.h) | Cutlass FMHA API |

---

## 9. Testing

### Test File
**Location**: `onnxruntime/test/python/transformers/test_gqa.py`

### Running Tests
```bash
# Run all GQA tests
cd onnxruntime/test/python/transformers
python test_gqa.py

# Run specific test class
python test_gqa.py TestFlashGQA
python test_gqa.py TestMemoryEfficientGQA

# Run a specific test
python -m pytest test_gqa.py -k "test_gqa_prompt_flash_00" -v

# Run with verbose output
DEBUG_GQA=1 python test_gqa.py
```

### Test Classes
| Class | Attention Backend | Description |
|-------|-------------------|-------------|
| `TestFlashGQA` | Flash Attention | Tests for SM 80+ GPUs |
| `TestMemoryEfficientGQA` | Cutlass FMHA | Tests for SM 53+ GPUs |

### Test Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_MODE` | `1` | Reduce test count for CI pipelines |
| `PARAM_COUNT` | `3` | Number of parameter variations (non-pipeline) |
| `QUICK_BUILD` | `0` | Only test hdim=128 for fast iteration |
| `DEBUG_GQA` | `0` | Enable debug output |

---

## 10. Benchmarking

### Benchmark File
**Location**: `onnxruntime/test/python/transformers/benchmark_gqa.py`

### Running Benchmarks
```bash
cd onnxruntime/test/python/transformers

# Run benchmark with default settings
python benchmark_gqa.py

# Benchmark specific configuration
python benchmark_gqa.py --batch_size 4 --num_heads 32 --kv_num_heads 8 --head_size 128

# Test only first config (quick check)
python benchmark_gqa.py --first_only

# Benchmark with bfloat16
python benchmark_gqa.py --dtype bfloat16
```

### Benchmark Modes
| Mode | Description |
|------|-------------|
| **Prompt** | First token generation (prefill) with varying sequence lengths |
| **Token** | Incremental decoding with fixed KV cache |

### Output
Benchmarks generate Triton-style plots showing latency (ms) vs sequence length for different GQA configurations.

---

## 11. Profiling

### Using NVIDIA Nsight Systems
```bash
nsys profile -o gqa_profile python test_gqa.py TestFlashGQA.test_gqa_prompt_flash_00
nsys-ui gqa_profile.nsys-rep
```

### Using NVIDIA Nsight Compute
```bash
# Profile specific kernel
ncu --set detailed -o gqa_kernel python test_gqa.py TestFlashGQA.test_gqa_prompt_flash_00
ncu-ui gqa_kernel.ncu-rep
```

### Using Custom Profiler Script
**Location**: `onnxruntime/test/python/transformers/profile_gqa_custom.py`

```bash
# Profile with nvtx markers
python profile_gqa_custom.py --batch_size 4 --seq_len 1024
```

### Key Metrics to Monitor
| Metric | Target | Description |
|--------|--------|-------------|
| **Kernel Launch Overhead** | < 5% | Time spent launching vs executing |
| **Memory Bandwidth** | > 80% peak | GPU memory utilization |
| **Occupancy** | > 50% | GPU SM utilization |
| **L2 Cache Hit Rate** | > 70% | Cache efficiency for KV access |

---

## 12. Environment Variables Reference

### Attention Backend Selection
| Variable | Values | Description |
|----------|--------|-------------|
| `ORT_DISABLE_FLASH_ATTENTION` | `0`, `1` | Disable Flash Attention |
| `ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION` | `0`, `1` | Disable Cutlass FMHA |

### Test Configuration
| Variable | Values | Description |
|----------|--------|-------------|
| `PIPELINE_MODE` | `0`, `1` | Reduce tests for CI |
| `PARAM_COUNT` | integer | Parameter variations |
| `QUICK_BUILD` | `0`, `1` | Fast build testing (hdim128 only) |
| `DEBUG_GQA` | `0`, `1` | Verbose debug output |

### CUDA Configuration
| Variable | Values | Description |
|----------|--------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0,1,...` | Select GPU device |
| `CUDA_LAUNCH_BLOCKING` | `0`, `1` | Synchronize all launches (debugging) |

### Profiling
| Variable | Values | Description |
|----------|--------|-------------|
| `NSYS_NVTX_PROFILER_REGISTER_ONLY` | `0`, `1` | Enable NVTX markers |

---

## 13. Debugging Tips

### Common Issues

1. **Flash Attention not being used**
   - Check GPU SM version (requires SM 80+)
   - Verify `ORT_DISABLE_FLASH_ATTENTION` is not set
   - Check head_size is supported (32, 64, 96, 128, 160, 192, 224, 256)

2. **Numerical Mismatches**
   - Enable `DEBUG_GQA=1` for tensor dumps
   - Compare KV cache shapes between BNSH and BSNH
   - Check position_ids format matches kernel expectations

3. **OOM Errors**
   - Reduce batch_size or sequence_length
   - Check `seqlen_present_kv_cache` allocation size
   - Use `kv_share_buffer=true` for memory efficiency

### Adding Debug Output
```cpp
// In group_query_attention_impl.cu
#include "contrib_ops/cpu/utils/debug_macros.h"
DUMP_TENSOR_INIT();
DUMP_TENSOR("tensor_name", tensor_ptr, dim0, dim1, dim2, dim3);
```
