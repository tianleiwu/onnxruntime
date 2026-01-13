# Optimized CUDA Kernel Design: Flash Attention with INT8 Quantized KV Cache

## Overview

This document describes an optimized CUDA kernel design for Flash Attention with INT8 quantized KV cache, specifically targeting the **decoding phase** of autoregressive language models.

## Problem Statement

### Scenario Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 1 | Single sequence inference |
| Query Length | 1 | Single token generation (decoding) |
| Past Sequence Length | 1024 | Context length to attend over |
| Number of Query Heads | 32 | Query head count |
| Number of KV Heads | 8 | KV head count (GQA) |
| GQA Ratio | 4:1 | 4 query heads share 1 KV head |
| Head Dimension | 128 | Dimension per head |
| KV Cache Precision | INT8 | Per-tensor FP16 scaling |

### Key Characteristics
- **Memory-bound workload**: Reading 1024×128 K and V vectors dominates execution time
- **Grouped Query Attention (GQA)**: Each KV head is shared by 4 query heads
- **Low arithmetic intensity**: Single query token means minimal compute reuse

---

## Design Philosophy

The core insight for decoding-phase attention is that we are **memory-bandwidth limited**, not compute-limited. Therefore, we should:

1. **Maximize parallelism** to hide memory latency
2. **Minimize register pressure** for high occupancy
3. **Simplify intra-block synchronization** to reduce overhead
4. **Accept small reduction overhead** in exchange for better parallelism

---

## Kernel Architecture: Split-K with Persistent Warps

### Grid & Block Organization

```
Grid:  (num_kv_heads, num_splits, 1) = (8, 4, 1) → 32 blocks total
Block: (128, 1, 1) = 4 warps per block
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_kv_heads` | 8 | One block group per KV head |
| `num_splits` | 4 | Sequence partitioning (1024/4 = 256 timesteps per split) |
| Threads/block | 128 | 4 warps, each handling one Q head |
| Total blocks | 32 | Saturates modern GPU SMs |

### Workload Distribution

```
Block (kv_head=h, split=s):
├── Warp 0 → Q Head (4h + 0), Timesteps [s×256, (s+1)×256)
├── Warp 1 → Q Head (4h + 1), Timesteps [s×256, (s+1)×256)
├── Warp 2 → Q Head (4h + 2), Timesteps [s×256, (s+1)×256)
└── Warp 3 → Q Head (4h + 3), Timesteps [s×256, (s+1)×256)
```

Each warp independently computes attention for its assigned Q head over its timestep range, then performs a warp-level reduction.

---

## Memory Hierarchy

### Register Allocation (Per Warp)

| Data | Size | Type | Purpose |
|------|------|------|---------|
| Q[128] | 256 B | FP16 | Query vector (reused across all timesteps) |
| O_acc[128] | 256 B | FP32 | Output accumulator |
| max_score | 4 B | FP32 | Online softmax: running maximum |
| sum_exp | 4 B | FP32 | Online softmax: running sum of exponentials |
| **Total** | ~520 B | | Per-warp register footprint |

### Shared Memory Layout (Per Block)

```
Shared Memory (16 KB total):
├── K_tile[TILE_K][128]: INT8 →  8 KB (TILE_K = 64)
└── V_tile[TILE_K][128]: INT8 →  8 KB (TILE_K = 64)
```

This fits comfortably within the 48 KB L1 cache configuration on modern NVIDIA GPUs.

### Global Memory Access Pattern

**Recommended Layout: BNSH `[batch, num_kv_heads, seq_len, head_dim]`**

```
K Cache: [batch, num_kv_heads, seq_len, head_dim] INT8
V Cache: [batch, num_kv_heads, seq_len, head_dim] INT8

Access Pattern:
- Each block loads K/V for its assigned kv_head and timestep range
- Contiguous memory access within each KV head's sequence range
- Coalesced 128-byte loads (16 INT8 elements × 8 threads)
- Vector loads using int4 (16 bytes per thread)
```

---

## Memory Layout Analysis: BSNH vs BNSH

### Layout Definitions

| Layout | Shape | Description |
|--------|-------|-------------|
| **BSNH** | `[batch, seq_len, num_kv_heads, head_dim]` | Sequence dimension is outer |
| **BNSH** | `[batch, num_kv_heads, seq_len, head_dim]` | Head dimension is outer |

### Access Pattern Comparison

For our Split-K kernel, each block needs to access:
- **One KV head** (kv_head = `blockIdx.x`)
- **A contiguous range of timesteps** (256 out of 1024)
- **The full head dimension** (128 elements)

#### BSNH Layout

```
Memory Address: addr(b, t, h, d) = b*(S*H*D) + t*(H*D) + h*D + d

For consecutive timesteps at same kv_head:
  K[b, t,   h, :] → base + t*(H*D) + h*D
  K[b, t+1, h, :] → base + (t+1)*(H*D) + h*D

Stride between timesteps = H × D = 8 × 128 = 1024 bytes
```

#### BNSH Layout

```
Memory Address: addr(b, h, t, d) = b*(H*S*D) + h*(S*D) + t*D + d

For consecutive timesteps at same kv_head:
  K[b, h, t,   :] → base + h*(S*D) + t*D
  K[b, h, t+1, :] → base + h*(S*D) + (t+1)*D

Stride between timesteps = D = 128 bytes (contiguous!)
```

### Performance Comparison

| Aspect | BSNH `[B,S,N,H]` | BNSH `[B,N,S,H]` |
|--------|-----------------|-----------------|
| **Stride between timesteps** | 1024 bytes | 128 bytes |
| **Contiguity for one KV head** | Fragmented | Fully contiguous |
| **Cache line utilization** | ~12.5% | 100% |
| **Memory coalescing** | Strided access | Sequential access |
| **Prefetch efficiency** | Low | High |

### Memory Coalescing Analysis

For a warp loading K values for consecutive timesteps:

**BSNH (Strided):**
```
Lane 0: addr = base + 0 × 1024 = base
Lane 1: addr = base + 1 × 1024 = base + 1024
Lane 2: addr = base + 2 × 1024 = base + 2048
...
→ 32 separate 128-byte cache line fetches!
→ Only 12.5% of fetched data is used
```

**BNSH (Contiguous):**
```
Lane 0: addr = base + 0 × 128 = base
Lane 1: addr = base + 1 × 128 = base + 128
Lane 2: addr = base + 2 × 128 = base + 256
...
→ 32 sequential 128-byte accesses = 4 KB contiguous
→ 100% of fetched data is used
```

### Bandwidth Impact

```
For loading 256 timesteps × 128 dims × 1 byte = 32 KB per KV head:

BSNH:
  - Each 128-byte access pulls in 128 bytes but only 12.5% is useful
  - Effective memory traffic: 32 KB / 0.125 = 256 KB (8× amplification!)

BNSH:
  - Contiguous access, 100% utilization
  - Effective memory traffic: 32 KB (no amplification)
```

| Metric | BSNH | BNSH | Improvement |
|--------|------|------|-------------|
| **Memory transactions** | 8× more | Baseline | 8× fewer |
| **Effective bandwidth** | ~250 GB/s | ~2000 GB/s | 8× higher |
| **Expected kernel time** | ~8-10 μs | ~1.2-1.5 μs | 6-8× faster |

### Recommended Stride Constants

```cpp
// BNSH layout (optimal for decoding)
constexpr int K_STRIDE_SEQ = HEAD_DIM;              // 128 bytes between timesteps
constexpr int K_STRIDE_HEAD = SEQ_LEN * HEAD_DIM;   // Contiguous per head

// BSNH layout (suboptimal - 8× bandwidth amplification)
// constexpr int K_STRIDE_SEQ = NUM_KV_HEADS * HEAD_DIM;  // 1024 bytes (strided!)
```

### When BSNH Might Be Acceptable

1. **Cross-head attention**: If accessing multiple KV heads for the same timestep
2. **Legacy compatibility**: Some systems expect BSNH format
3. **Sequence-parallel training**: Better gradient parallelism across sequence dimension

> **Recommendation**: For decoding workloads, **always use BNSH layout** unless constrained by external requirements.

---

## Execution Flow

### Phase 1: Main Kernel (FlashDecodeGQA_SplitK)

```cuda
__global__ void FlashDecodeGQA_SplitK(
    const half* __restrict__ Q,           // [batch, num_q_heads, head_dim]
    const int8_t* __restrict__ K,         // [batch, seq_len, num_kv_heads, head_dim]
    const int8_t* __restrict__ V,         // [batch, seq_len, num_kv_heads, head_dim]
    half* __restrict__ O_partial,         // [num_kv_heads, num_splits, q_per_kv, head_dim]
    float* __restrict__ LSE_partial,      // [num_kv_heads, num_splits, q_per_kv]
    const half k_scale,
    const half v_scale,
    const int seq_len
) {
    // Block/warp identification
    const int kv_head = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int warp_id = threadIdx.x / 32;   // 0-3
    const int lane_id = threadIdx.x % 32;

    const int q_head = kv_head * Q_PER_KV + warp_id;

    // Timestep range for this split
    const int chunk_size = (seq_len + NUM_SPLITS - 1) / NUM_SPLITS;
    const int seq_start = split_idx * chunk_size;
    const int seq_end = min(seq_start + chunk_size, seq_len);

    // ========================================
    // Step 1: Load Q into registers
    // ========================================
    half Q_reg[ELEMENTS_PER_THREAD];  // 128 / 32 = 4 elements per lane
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        Q_reg[i] = Q[q_head * HEAD_DIM + lane_id * ELEMENTS_PER_THREAD + i];
    }

    // ========================================
    // Step 2: Initialize online softmax state
    // ========================================
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float O_acc[ELEMENTS_PER_THREAD] = {0.0f};

    // ========================================
    // Step 3: Shared memory for K/V tiles
    // ========================================
    __shared__ int8_t K_smem[TILE_K][HEAD_DIM];
    __shared__ int8_t V_smem[TILE_K][HEAD_DIM];

    // ========================================
    // Step 4: Main loop over timesteps
    // ========================================
    for (int t_base = seq_start; t_base < seq_end; t_base += TILE_K) {
        const int tile_len = min(TILE_K, seq_end - t_base);

        // ---- Collaborative K/V tile load ----
        // All 128 threads load cooperatively
        // Each thread loads 16 bytes (int4 vectorized)
        load_kv_tiles_vectorized(K, V, kv_head, t_base, tile_len, K_smem, V_smem);
        __syncthreads();

        // ---- Compute attention for timesteps in tile ----
        // Each lane handles different timesteps within the tile
        for (int dt = lane_id; dt < tile_len; dt += WARP_SIZE) {
            // Dot product: Q · K[t_base + dt]
            float score = compute_qk_dot_product_int8(
                Q_reg, K_smem[dt], k_scale
            );

            // Online softmax update
            float new_max = fmaxf(max_score, score);
            float exp_diff = __expf(max_score - new_max);
            float exp_score = __expf(score - new_max);

            // Rescale existing accumulator
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
                O_acc[i] *= exp_diff;
            }

            // Update softmax state
            sum_exp = sum_exp * exp_diff + exp_score;
            max_score = new_max;

            // Accumulate weighted V
            accumulate_v_int8(O_acc, V_smem[dt], v_scale, exp_score);
        }
        __syncthreads();
    }

    // ========================================
    // Step 5: Warp-level reduction
    // ========================================
    // Reduce across lanes (different timesteps) using online softmax
    warp_reduce_attention_output(O_acc, max_score, sum_exp);

    // ========================================
    // Step 6: Write partial results
    // ========================================
    if (lane_id == 0) {
        const int out_idx = kv_head * NUM_SPLITS * Q_PER_KV + split_idx * Q_PER_KV + warp_id;

        // Store log-sum-exp for later reduction
        LSE_partial[out_idx] = max_score + __logf(sum_exp);

        // Store normalized partial output
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            O_partial[out_idx * HEAD_DIM + i] = __float2half(O_acc[i] / sum_exp);
        }
    }
}
```

### Phase 2: Reduction Kernel (FlashDecodeGQA_Reduce)

```cuda
__global__ void FlashDecodeGQA_Reduce(
    const half* __restrict__ O_partial,    // [num_kv_heads, num_splits, q_per_kv, head_dim]
    const float* __restrict__ LSE_partial, // [num_kv_heads, num_splits, q_per_kv]
    half* __restrict__ O_final,            // [batch, num_q_heads, head_dim]
    const int num_splits
) {
    // One block per Q head, or one warp per Q head
    const int q_head = blockIdx.x;
    const int kv_head = q_head / Q_PER_KV;
    const int q_idx_in_group = q_head % Q_PER_KV;
    const int lane_id = threadIdx.x % 32;

    // Load LSE values for all splits
    float lse[MAX_SPLITS];
    float max_lse = -INFINITY;

    #pragma unroll
    for (int s = 0; s < num_splits; s++) {
        const int idx = kv_head * num_splits * Q_PER_KV + s * Q_PER_KV + q_idx_in_group;
        lse[s] = LSE_partial[idx];
        max_lse = fmaxf(max_lse, lse[s]);
    }

    // Compute rescaling factors
    float sum_weights = 0.0f;
    float weights[MAX_SPLITS];

    #pragma unroll
    for (int s = 0; s < num_splits; s++) {
        weights[s] = __expf(lse[s] - max_lse);
        sum_weights += weights[s];
    }

    // Weighted sum of partial outputs
    float O_final_val[ELEMENTS_PER_THREAD] = {0.0f};

    #pragma unroll
    for (int s = 0; s < num_splits; s++) {
        const int base_idx = (kv_head * num_splits * Q_PER_KV + s * Q_PER_KV + q_idx_in_group) * HEAD_DIM;

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            float partial = __half2float(O_partial[base_idx + lane_id * ELEMENTS_PER_THREAD + i]);
            O_final_val[i] += weights[s] * partial;
        }
    }

    // Normalize and store
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        O_final[q_head * HEAD_DIM + lane_id * ELEMENTS_PER_THREAD + i] =
            __float2half(O_final_val[i] / sum_weights);
    }
}
```

---

## Key Optimizations

### 1. Vectorized INT8 Loads

```cuda
__device__ __forceinline__ void load_kv_tiles_vectorized(
    const int8_t* K, const int8_t* V,
    int kv_head, int t_base, int tile_len,
    int8_t K_smem[][HEAD_DIM], int8_t V_smem[][HEAD_DIM]
) {
    const int tid = threadIdx.x;
    const int total_elements = tile_len * HEAD_DIM;
    const int elements_per_thread = 16;  // int4 = 16 bytes

    // Vectorized load using int4 (128-bit)
    for (int i = tid * elements_per_thread; i < total_elements; i += blockDim.x * elements_per_thread) {
        int t = i / HEAD_DIM;
        int d = i % HEAD_DIM;

        if (t < tile_len) {
            int4 k_vec = *reinterpret_cast<const int4*>(&K[(t_base + t) * STRIDE + kv_head * HEAD_DIM + d]);
            int4 v_vec = *reinterpret_cast<const int4*>(&V[(t_base + t) * STRIDE + kv_head * HEAD_DIM + d]);

            *reinterpret_cast<int4*>(&K_smem[t][d]) = k_vec;
            *reinterpret_cast<int4*>(&V_smem[t][d]) = v_vec;
        }
    }
}
```

### 2. Efficient Q·K Dot Product with INT8 Dequantization

```cuda
__device__ __forceinline__ float compute_qk_dot_product_int8(
    const half* Q_reg,        // FP16 query in registers
    const int8_t* K_row,      // INT8 key in shared memory
    half k_scale              // Dequantization scale
) {
    float acc = 0.0f;
    const float scale_f = __half2float(k_scale);

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        float q_val = __half2float(Q_reg[i]);
        float k_val = static_cast<float>(K_row[lane_id * ELEMENTS_PER_THREAD + i]) * scale_f;
        acc += q_val * k_val;
    }

    // Warp-level reduction for complete dot product
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFF, acc, mask);
    }

    return acc;
}
```

### 3. Online Softmax with Warp Reduction

```cuda
__device__ __forceinline__ void warp_reduce_attention_output(
    float* O_acc,           // [ELEMENTS_PER_THREAD]
    float& max_score,
    float& sum_exp
) {
    // Step 1: Find global max across warp
    float global_max = max_score;
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        global_max = fmaxf(global_max, __shfl_xor_sync(0xFFFFFFFF, global_max, mask));
    }

    // Step 2: Rescale local values to global max
    float rescale = __expf(max_score - global_max);
    sum_exp *= rescale;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        O_acc[i] *= rescale;
    }

    // Step 3: Sum across warp
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        sum_exp += __shfl_xor_sync(0xFFFFFFFF, sum_exp, mask);
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            O_acc[i] += __shfl_xor_sync(0xFFFFFFFF, O_acc[i], mask);
        }
    }

    max_score = global_max;
}
```

---

## Performance Analysis

### Comparison: Original vs Split-K Design

| Metric | Original (1 Block/KV Head) | Split-K (4 Splits) |
|--------|---------------------------|-------------------|
| **Total Blocks** | 8 | 32 |
| **SM Utilization** | Low (~7% on A100) | High (~30% on A100) |
| **Registers/Warp** | ~1 KB (4 Q heads) | ~520 B (1 Q head) |
| **Occupancy** | Limited by registers | Higher occupancy |
| **Intra-block Sync** | Complex 4-head merge | Per-warp (simple) |
| **Reduction Overhead** | None | Small 2nd kernel |
| **K/V Reloads** | 1× per KV head | 1× per KV head (same) |

### Memory Bandwidth Analysis

```
K/V Cache Size: 1024 × 8 × 128 × 2 (K+V) × 1 byte = 2 MB

Memory Bandwidth (A100 HBM): 2039 GB/s
Theoretical minimum time: 2 MB / 2039 GB/s ≈ 1 μs

With 32 blocks, achieved bandwidth efficiency should be 70-80%
Expected execution time: ~1.3-1.5 μs
```

### Arithmetic Intensity

```
FLOPs: 32 Q heads × 1024 timesteps × 128 dims × 2 (Q·K + V acc) = 8.4 MFLOP
Bytes: 2 MB (K/V) + 16 KB (Q) + 16 KB (O) ≈ 2 MB

Arithmetic Intensity: 8.4 MFLOP / 2 MB = 4.2 FLOP/byte

A100 Roofline:
- Memory-bound region: < 39 FLOP/byte (FP32)
- This kernel is clearly memory-bound at 4.2 FLOP/byte
```

---

## Configuration Constants

```cpp
// Kernel configuration
constexpr int HEAD_DIM = 128;
constexpr int Q_PER_KV = 4;          // GQA ratio
constexpr int NUM_KV_HEADS = 8;
constexpr int NUM_Q_HEADS = 32;
constexpr int NUM_SPLITS = 4;        // Tunable: 2-8
constexpr int TILE_K = 64;           // Timesteps per tile
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * WARP_SIZE;  // 128
constexpr int ELEMENTS_PER_THREAD = HEAD_DIM / WARP_SIZE;       // 4
```

---

## Extension: Batched Inference

For batched inference with `batch_size > 1`, extend the grid:

```cpp
dim3 grid(num_kv_heads, num_splits, batch_size);
```

Each batch element is processed independently with no cross-batch dependencies.

---

## Extension: Variable Sequence Lengths

For variable sequence lengths across batches, use `seqlen_k` array:

```cpp
__global__ void FlashDecodeGQA_SplitK_VarLen(
    // ... existing parameters ...
    const int* seqlen_k  // [batch_size] actual sequence lengths
) {
    const int batch_id = blockIdx.z;
    const int actual_seq_len = seqlen_k[batch_id];

    // Adjust chunk boundaries based on actual_seq_len
    const int chunk_size = (actual_seq_len + NUM_SPLITS - 1) / NUM_SPLITS;
    // ... rest of kernel logic ...
}
```

---

## Future Optimizations

1. **Tensor Core Utilization**: For BF16 queries, explore using `mma` instructions for the Q·K dot products
2. **Fused Reduction**: For small `NUM_SPLITS`, fuse reduction into the main kernel epilogue using cooperative groups
3. **Adaptive Split Selection**: Dynamically choose `NUM_SPLITS` based on sequence length
4. **Prefetching**: Use async copy (`cp.async`) for overlapping K/V loads with computation

---

## References

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- [FlashDecoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
