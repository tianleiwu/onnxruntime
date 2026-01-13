# Implementation Plan: GQA-Aware Blocking for INT8 Flash Attention

## Goal

Reduce memory traffic by 4× by changing from 32 blocks (one per Q head) to 8 blocks (one per KV head), with each block computing 4 Q heads that share the same K/V data.

## Background

### Current Architecture
- **Grid**: [(num_m_block, splits, params.h)](file:///home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/utils.h#126-131) where `params.h = 32` (Q heads)
- **Each block**: Processes 1 Q head, loads full K/V cache independently
- **Memory traffic**: 32 blocks × 4 MB = 128 MB (with 4× redundant KV loads)

### Target Architecture  
- **Grid**: [(num_m_block, splits, params.h_k)](file:///home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/utils.h#126-131) where `params.h_k = 8` (KV heads)
- **Each block**: 4 warps, each processing 1 Q head in the GQA group
- **Memory traffic**: 8 blocks × 4 MB = 32 MB (no redundancy)

---

## Proposed Changes

### [Launcher] [flash_fwd_launch_template.h](file:///home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h)

#### [MODIFY] [flash_fwd_launch_template.h](file:///home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h)

Change grid configuration in [run_flash_int8_dequant_fwd](file:///home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h#338-391) (line 350):

```diff
-dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
+// GQA-aware blocking: use KV heads (h_k) instead of Q heads (h)
+// Each block will compute h_h_k_ratio Q heads (typically 4)
+dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h_k : params.h_k);
```

Also update the Split-K combine grid to use Q heads for output (unchanged since output is per-Q-head).

---

### [Kernel Call] [flash_fwd_launch_template.h](file:///home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h)

#### [MODIFY] Kernel invocation

Update `flash_fwd_int8_dequant_kernel` to pass correct head indexing:

```diff
// In flash_fwd_int8_dequant_kernel (line 268-281)
-const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
-const int bidh = Split ? blockIdx.z % params.h : blockIdx.z;
+// GQA-aware: bidh is now KV head index (0-7), not Q head index (0-31)
+const int bidb = Split ? blockIdx.z / params.h_k : blockIdx.y;
+const int bidh_kv = Split ? blockIdx.z % params.h_k : blockIdx.z;
```

---

### [Core Kernel] [flash_int8_fwd_kernel.h](file:///home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/flash_int8_fwd_kernel.h)

#### [MODIFY] [flash_int8_fwd_kernel.h](file:///home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/flash_int8_fwd_kernel.h)

Major changes needed in [compute_attn_1rowblock](file:///home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/flash_int8_fwd_kernel.h#238-806):

1. **Function signature**: Pass KV head index instead of Q head index
2. **Warp specialization**: Each of 4 warps handles 1 Q head
3. **Q loading**: Load 4 different Q vectors (one per warp)
4. **Accumulator**: 4 separate softmax states and output accumulators
5. **K/V loading**: Shared across all 4 warps (loaded once)
6. **Output write**: Each warp writes to its Q head's output location

Key code changes:

```cpp
// Line 240: Change bidh to bidh_kv and add warp_id
template <...>
inline __device__ void compute_attn_1rowblock(
    const Params& params, const int bidb, const int bidh_kv, const int m_block,
    const int n_split_idx, const int num_n_splits) {
  
  // Warp identification for GQA
  const int warp_id = threadIdx.x / 32;  // 0-3
  const int lane_id = threadIdx.x % 32;
  
  // Each warp handles a different Q head in this KV head group
  const int bidh = bidh_kv * params.h_h_k_ratio + warp_id;  // 0-31
  
  // Skip if this warp's Q head exceeds total Q heads (edge case)
  if (bidh >= params.h) return;
  
  // KV head index is same for all warps in this block
  const int kv_head_idx = bidh_kv;  // Was: bidh / params.h_h_k_ratio
  
  // ... rest of kernel uses bidh for Q access, bidh_kv for K/V access
```

---

## Risk Assessment

> [!WARNING]
> This modification is invasive and affects the core attention computation loop. Key risks:
> 1. **Register pressure**: 4× more Q vectors in registers
> 2. **Shared memory**: Same K/V usage, but 4× Q space needed
> 3. **Softmax state**: Must track 4 independent softmax states
> 4. **Correctness**: Complex indexing changes

---

## Verification Plan

### Automated Tests

Run the existing GQA tests after build:

```bash
# Quick build and test
./gqa_release.sh --quick_build --install --test
```

This runs `test_gqa.py` which includes:
- `test_gqa_past_flash_attention` (FP16/BF16)
- `test_gqa_quantized_past` (INT8/INT4)
- Various batch sizes, sequence lengths, head configurations

### Performance Profiling

```bash
# After tests pass, profile to verify speedup
./gqa_release.sh --quick_build --install --profile
```

Expected results:
- `flash_fwd_int8_dequant_kernel`: ~20 μs (down from 82 μs)
- Grid blocks: 8 (down from 32)

---

## Alternative Approach (Simpler, Incremental)

If the full GQA-aware kernel is too complex, consider an incremental approach:

1. **Phase 1**: Keep 32 blocks but add shared memory K/V caching across Q heads
2. **Phase 2**: Add warp specialization within existing structure
3. **Phase 3**: Reorganize grid to use KV heads

This reduces risk but may not achieve the full 4× speedup.

---

## User Review Required

> [!IMPORTANT]
> This is a significant kernel modification. Please confirm:
> 1. Is the 4× speedup target worth the implementation complexity?
> 2. Should we try the simpler incremental approach first?
> 3. Any concern about test coverage for the changes?
