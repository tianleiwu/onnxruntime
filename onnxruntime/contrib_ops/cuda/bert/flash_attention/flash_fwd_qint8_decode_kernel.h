// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
/******************************************************************************
 * Specialized Flash Attention Decode Kernel with INT8 Quantized KV Cache
 *
 * This kernel is optimized for decoding (Q_len=1) with GQA awareness.
 * Key optimizations:
 * - GQA-aware: 1 block per KV head, 4 warps handle 4 Q heads sharing K/V
 * - Q=1 specialization: GEMV instead of GEMM, no wasted M-dimension
 * - Online softmax: Streaming computation reduces memory requirements
 * - Vectorized INT8 dequantization for K/V
 *
 * Architecture:
 * - Grid: (num_kv_heads, num_splits, batch)
 * - Block: 128 threads (4 warps)
 * - Each warp: 1 Q head, shares K/V with other warps
 *
 ******************************************************************************/
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "contrib_ops/cuda/bert/flash_attention/flash.h"
#include "contrib_ops/cuda/bert/flash_attention/block_info.h"
#include "contrib_ops/cuda/bert/flash_attention/namespace_config.h"

namespace FLASH_NAMESPACE {
using namespace cute;
namespace decode_int8 {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Warp-level reduction utilities
////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
  }
  return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
  }
  return val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Vectorized Dequantization Helpers
// Optimized for int8 KV cache with fp16/bf16 computation
////////////////////////////////////////////////////////////////////////////////////////////////////

// Dequantize 4 int8 values to 4 floats using vectorized operations
// Input: 4 consecutive int8 values packed in an int (as bytes)
// Output: 4 floats in an array
__device__ __forceinline__ void dequant_int8x4_to_float4(
    const int8_t* src, float* dst, float scale) {
  // Load 4 int8s as a single 32-bit word
  int32_t packed = *reinterpret_cast<const int32_t*>(src);

  // Extract individual bytes using bit operations
  int8_t v0 = static_cast<int8_t>(packed & 0xFF);
  int8_t v1 = static_cast<int8_t>((packed >> 8) & 0xFF);
  int8_t v2 = static_cast<int8_t>((packed >> 16) & 0xFF);
  int8_t v3 = static_cast<int8_t>((packed >> 24) & 0xFF);

  // Convert to float with scale
  dst[0] = static_cast<float>(v0) * scale;
  dst[1] = static_cast<float>(v1) * scale;
  dst[2] = static_cast<float>(v2) * scale;
  dst[3] = static_cast<float>(v3) * scale;
}

// Compute dot product of two float4 vectors
__device__ __forceinline__ float dot4(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

// Accumulate: dst += scale * src (float4)
__device__ __forceinline__ void fma4(float* dst, float scale, const float* src) {
  dst[0] += scale * src[0];
  dst[1] += scale * src[1];
  dst[2] += scale * src[2];
  dst[3] += scale * src[3];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel Traits for Decode (Q=1)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kHeadDim_, int kBlockN_, typename elem_type = cutlass::half_t>
struct Flash_decode_int8_kernel_traits {
  using Element = elem_type;
  using ElementAccum = float;
  using ElementInt8 = int8_t;
  using index_t = int64_t;

  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kBlockN = kBlockN_;        // KV block size per iteration
  static constexpr int kNWarps = 4;               // 4 warps for 4 Q heads in GQA group
  static constexpr int kNThreads = kNWarps * 32;  // 128 threads

  // Shared memory sizes
  // Note: kSmemSizeQ is reused for sO (intermediate float accumulation),
  // so it must be large enough to hold floats (4 bytes) even if Element is half (2 bytes).
  static constexpr int kSmemRowStrideInt8 = kHeadDim + 16;  // Padded for bank conflicts
  static constexpr int kSmemSizeQ = kNWarps * kHeadDim * sizeof(float);
  static constexpr int kSmemSizeK = kBlockN * kSmemRowStrideInt8;
  static constexpr int kSmemSizeV = kBlockN * kSmemRowStrideInt8;
  static constexpr int kSmemSize = kSmemSizeQ + kSmemSizeK + kSmemSizeV;

  struct SharedStorage {
    float sQ[kNWarps][kHeadDim];
    ElementInt8 sK[kBlockN][kSmemRowStrideInt8];
    ElementInt8 sV[kBlockN][kSmemRowStrideInt8];
  };
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main Decode Kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_causal, bool Split, typename Params>
inline __device__ void compute_attn_decode(const Params& params) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using ElementInt8 = typename Kernel_traits::ElementInt8;
  using index_t = typename Kernel_traits::index_t;

  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kBlockN = Kernel_traits::kBlockN;

  constexpr int kSmemRowStride = Kernel_traits::kSmemRowStrideInt8;

  extern __shared__ char smem_[];

  const int tidx = threadIdx.x;
  const int warp_id = tidx / 32;
  const int lane_id = tidx % 32;

  // Grid mapping: GQA-aware
  const int kv_head_idx = blockIdx.x;  // 0..h_k-1
  const int n_split_idx = Split ? blockIdx.y : 0;
  const int num_n_splits = Split ? gridDim.y : 1;
  const int bidb = blockIdx.z;  // Grid is (h_k, num_splits, b)

  // Each warp handles 1 Q head in the GQA group
  const int q_head_idx = kv_head_idx * params.h_h_k_ratio + warp_id;

  // Early exit if this warp's Q head doesn't exist
  if (warp_id >= params.h_h_k_ratio) return;

  // BlockInfo for sequence lengths
  const BlockInfo<true> binfo(params, bidb);
  if (binfo.actual_seqlen_q == 0 || binfo.actual_seqlen_k == 0) {
    if constexpr (Split) {
      if (lane_id == 0) {
        const index_t lse_offset = ((n_split_idx * params.b + bidb) * params.h + q_head_idx) * params.seqlen_q;
        ElementAccum* lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + lse_offset;
        lse_ptr[0] = -INFINITY;
      }
      const index_t o_offset = ((n_split_idx * params.b + bidb) * params.h + q_head_idx) * params.seqlen_q * params.d_rounded;
      ElementAccum* o_ptr = reinterpret_cast<ElementAccum*>(params.oaccum_ptr) + o_offset;
      for (int d = lane_id; d < kHeadDim; d += 32) {
        if (d < params.d) o_ptr[d] = 0.0f;
      }
    } else {
      // For Q_len=1, we can just write 0 to O
      const index_t o_offset = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) + q_head_idx * params.o_head_stride;
      Element* o_ptr = reinterpret_cast<Element*>(params.o_ptr) + o_offset;
      for (int d = lane_id; d < kHeadDim; d += 32) {
        if (d < params.d) o_ptr[d] = Element(0);
      }
    }
    return;
  }

  // Shared memory layout
  float* sQ = reinterpret_cast<float*>(smem_);  // [kNWarps][kHeadDim]
  ElementInt8* sK = reinterpret_cast<ElementInt8*>(smem_ + Kernel_traits::kSmemSizeQ);
  ElementInt8* sV = sK + kBlockN * kSmemRowStride;

  // Get scales
  const int scale_idx = (params.k_quant_type == 2) ? kv_head_idx : 0;
  const float k_scale = params.k_scale_ptr
                            ? static_cast<float>(reinterpret_cast<const Element*>(params.k_scale_ptr)[scale_idx])
                            : 1.0f;
  const float v_scale = params.v_scale_ptr
                            ? static_cast<float>(reinterpret_cast<const Element*>(params.v_scale_ptr)[scale_idx])
                            : 1.0f;

  // ============================================================================
  // Load Q (each warp loads its own Q vector)
  // ============================================================================
  const Element* q_ptr = reinterpret_cast<const Element*>(params.q_ptr) +
                         binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) +
                         q_head_idx * params.q_head_stride;

  // Vectorized Q Load: Each thread loads 4 halves/floats and stores to shared as floats
  float* my_sQ = sQ + warp_id * kHeadDim;
  for (int d = lane_id * 4; d < kHeadDim; d += 128) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (d + i < kHeadDim) {
        my_sQ[d + i] = static_cast<float>(q_ptr[d + i]);
      }
    }
  }

  // ============================================================================
  // K/V pointers (shared across all warps)
  // ============================================================================
  const index_t k_base_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                kv_head_idx * params.k_head_stride;
  const index_t v_base_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                kv_head_idx * params.v_head_stride;

  const ElementInt8* k_ptr = reinterpret_cast<const ElementInt8*>(params.k_ptr) + k_base_offset;
  const ElementInt8* v_ptr = reinterpret_cast<const ElementInt8*>(params.v_ptr) + v_base_offset;

  // ============================================================================
  // Split-K range
  // ============================================================================
  const int n_blocks_per_split = (binfo.actual_seqlen_k + kBlockN - 1) / kBlockN;
  const int blocks_per_split = (n_blocks_per_split + num_n_splits - 1) / num_n_splits;
  const int n_block_min = n_split_idx * blocks_per_split;
  const int n_block_max = min((n_split_idx + 1) * blocks_per_split, n_blocks_per_split);

  if (n_block_min >= n_block_max) {
    // Early exit for empty splits - write zeros
    if constexpr (Split) {
      if (lane_id == 0) {
        const index_t lse_offset = ((n_split_idx * params.b + bidb) * params.h + q_head_idx) * params.seqlen_q;
        ElementAccum* lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + lse_offset;
        lse_ptr[0] = -INFINITY;
      }
      const index_t o_offset = ((n_split_idx * params.b + bidb) * params.h + q_head_idx) * params.seqlen_q * params.d_rounded;
      ElementAccum* o_ptr = reinterpret_cast<ElementAccum*>(params.oaccum_ptr) + o_offset;
      for (int d = lane_id; d < kHeadDim; d += 32) {
        if (d < params.d) o_ptr[d] = 0.0f;
      }
    }
    return;
  }

  // ============================================================================
  // Online Softmax State (per warp)
  // ============================================================================
  float m_i = -INFINITY;  // Running max
  float l_i = 0.0f;       // Running sum of exp

  // Output accumulator: each lane holds kElemsPerLane consecutive elements
  // For HeadDim=128: lane 0 holds [0,1,2,3], lane 1 holds [4,5,6,7], etc.
  constexpr int kElemsPerLane = kHeadDim / 32;  // 4 for HeadDim=128
  float acc_o[kElemsPerLane];
#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    acc_o[i] = 0.0f;
  }

  __syncthreads();  // Ensure Q is loaded

  // ============================================================================
  // Main Loop: Process KV blocks
  // ============================================================================
  for (int n_block = n_block_min; n_block < n_block_max; ++n_block) {
    const int k_start = n_block * kBlockN;
    const int k_end = min(k_start + kBlockN, binfo.actual_seqlen_k);
    const int k_len = k_end - k_start;

    // --------------------------------------------------------------------
    // --------------------------------------------------------------------
    // Load K/V blocks (vectorized int4 = 16 bytes = 16 int8s)
    // --------------------------------------------------------------------
    const ::int4* k_ptr_vec = reinterpret_cast<const ::int4*>(k_ptr);
    const ::int4* v_ptr_vec = reinterpret_cast<const ::int4*>(v_ptr);
    ::int4* sK_vec = reinterpret_cast<::int4*>(sK);
    ::int4* sV_vec = reinterpret_cast<::int4*>(sV);

    // Total vectors to load: (kBlockN * kHeadDim) / 16
    constexpr int kVecSize = 16;
    constexpr int kVecsPerBlock = (kBlockN * kHeadDim) / kVecSize;

    constexpr int kGlobalStrideVecs = kHeadDim / kVecSize;      // 128/16 = 8
    constexpr int kSmemStrideVecs = kSmemRowStride / kVecSize;  // 144/16 = 9

    for (int i = tidx; i < kVecsPerBlock; i += Kernel_traits::kNThreads) {
      int row = i / kGlobalStrideVecs;
      int col_vec = i % kGlobalStrideVecs;

      if (row < k_len) {
        // Global load: params.k_row_stride is in bytes. Assume multiple of 16.
        // int4 pointer arithmetic uses stride in 16-byte chunks.
        const int row_stride_vecs = params.k_row_stride / kVecSize;

        // If stride is not aligned, this fails. Assuming D=128 and default strides.
        int global_offset = (k_start + row) * row_stride_vecs + col_vec;
        int smem_offset = row * kSmemStrideVecs + col_vec;

        sK_vec[smem_offset] = k_ptr_vec[global_offset];

        const int v_row_stride_vecs = params.v_row_stride / kVecSize;
        int v_global_offset = (k_start + row) * v_row_stride_vecs + col_vec;
        sV_vec[smem_offset] = v_ptr_vec[v_global_offset];
      } else {
        // Zero-initialize tail to avoid NaN propagation (0 * NaN = NaN)
        int smem_offset = row * kSmemStrideVecs + col_vec;
        // int4 zero is {0,0,0,0} -> integer 0
        ::int4 zero;
        zero.x = 0;
        zero.y = 0;
        zero.z = 0;
        zero.w = 0;
        sK_vec[smem_offset] = zero;
        sV_vec[smem_offset] = zero;
      }
    }

    __syncthreads();

    // --------------------------------------------------------------------
    // Compute S = Q @ K^T (warp-level GEMV with vectorized dequant)
    // Each warp uses its own Q vector, shared K
    // OPTIMIZATION: Each lane handles 4 consecutive head dims per iteration
    // HeadDim=128, 32 lanes → each lane covers 4 elements → 1 full pass
    // Using vectorized int8x4 loads for K data
    // --------------------------------------------------------------------
    float scores[kBlockN];

    // Pre-load Q values for this lane (kElemsPerLane elements per lane)
    float q_local[kElemsPerLane];
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      q_local[i] = my_sQ[lane_id * kElemsPerLane + i];
    }

    // Process each K token
    for (int n = 0; n < kBlockN; ++n) {
      float score = 0.0f;
      const ElementInt8* k_row = sK + n * kSmemRowStride;

      // Vectorized dot product: process 4 elements at a time
#pragma unroll
      for (int i = 0; i < kElemsPerLane; ++i) {
        int d = lane_id * kElemsPerLane + i;
        float k_val = static_cast<float>(k_row[d]) * k_scale;
        score += q_local[i] * k_val;
      }
      scores[n] = warp_reduce_sum(score);
    }

    // Apply scaling
    for (int n = 0; n < kBlockN; ++n) {
      scores[n] *= params.scale_softmax;
    }

    // Causal masking (for decode, typically not needed since Q_row=0)
    if constexpr (Is_causal) {
      for (int n = 0; n < kBlockN; ++n) {
        if (k_start + n > binfo.actual_seqlen_k - 1) {
          scores[n] = -INFINITY;
        }
      }
    }

    // Out-of-bounds masking
    for (int n = 0; n < kBlockN; ++n) {
      if (n >= k_len) {
        scores[n] = -INFINITY;
      }
    }

    // --------------------------------------------------------------------
    // Online Softmax Update
    // --------------------------------------------------------------------
    // Find max in this block
    float m_block = -INFINITY;
    for (int n = 0; n < kBlockN; ++n) {
      m_block = fmaxf(m_block, scores[n]);
    }

    // Update running max and rescale previous accumulator
    float m_new = fmaxf(m_i, m_block);
    float rescale = expf(m_i - m_new);

// Rescale existing O accumulator
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) {
      acc_o[i] *= rescale;
    }
    l_i *= rescale;

    // Compute exp(s - m_new) and update sum
    float p[kBlockN];
    for (int n = 0; n < kBlockN; ++n) {
      p[n] = expf(scores[n] - m_new);
      l_i += p[n];
    }
    m_i = m_new;

    // --------------------------------------------------------------------
    // Accumulate O += P @ V (warp-level with vectorized access)
    // OPTIMIZATION: Each lane handles 4 consecutive head dims (matches Q×K)
    // acc_o[i] accumulates element (lane_id * 4 + i) of the output
    // --------------------------------------------------------------------
    for (int n = 0; n < kBlockN; ++n) {
      float p_val = p[n];
      const ElementInt8* v_row = sV + n * kSmemRowStride;

#pragma unroll
      for (int i = 0; i < kElemsPerLane; ++i) {
        int d = lane_id * kElemsPerLane + i;
        float v_val = static_cast<float>(v_row[d]) * v_scale;
        acc_o[i] += p_val * v_val;
      }
    }

    __syncthreads();  // Prepare for next K/V load
  }

  // ============================================================================
  // Finalize: Normalize O and Write Output
  // ============================================================================
  // Normalize by softmax sum
  float l_inv = 1.0f / l_i;

  // Write O to global memory
  using ElementO = std::conditional_t<!Split, Element, ElementAccum>;
  const index_t o_offset = Split
                               ? ((n_split_idx * params.b + bidb) * params.h + q_head_idx) * params.seqlen_q * params.d_rounded
                               : binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) + q_head_idx * params.o_head_stride;

  ElementO* o_ptr = reinterpret_cast<ElementO*>(Split ? params.oaccum_ptr : params.o_ptr) + o_offset;

#pragma unroll
  for (int i = 0; i < kElemsPerLane; ++i) {
    int d = lane_id * kElemsPerLane + i;
    float val = acc_o[i] * l_inv;
    if (d < params.d) {
      o_ptr[d] = static_cast<ElementO>(val);
    } else if (d < params.d_rounded) {
      o_ptr[d] = ElementO(0);  // Zero out tails for Split-K determinism
    }
  }

  // Write LSE for Split-K
  if constexpr (Split) {
    if (lane_id == 0) {
      const index_t lse_offset = ((n_split_idx * params.b + bidb) * params.h + q_head_idx) * params.seqlen_q;
      ElementAccum* lse_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + lse_offset;
      lse_ptr[0] = m_i + logf(l_i);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace decode_int8
}  // namespace FLASH_NAMESPACE
