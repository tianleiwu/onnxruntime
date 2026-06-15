// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemv.h"

#include <cuda_fp16.h>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "core/common/common.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/dispatcher.h"

namespace onnxruntime::llm {
namespace kernels {
namespace fpA_intB_gemv {

// MoE batched GEMV kernel. One thread-block (CtaM = 1 row) processes a single
// expanded row; the row's expert determines the weight/scale/bias base pointers.
// Mirrors the dense fpA_intB_gemv `kernel<>` body for GroupSize=0 (per-channel),
// EnableActScale=false, EnableZero=false, ApplyAlphaInAdvance=false.
template <typename Details, int CtaN, int Threads, int GroupSize, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA>
__global__ void moe_gemv_kernel(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                                int64_t const* expert_first_token_offset, int const* permuted_row_to_expert,
                                int num_experts,
                                int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr bool Mandatory = true;
  static constexpr int CtaM = 1;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);
  if constexpr (GroupSize != 0) {
    static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
  }

  int const row = blockIdx.x;

  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
  // Fallback path for prologues that have not materialized the row-to-expert map.
#pragma unroll 1
  for (int e = 0; e < num_experts && permuted_row_to_expert == nullptr; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  weight += expert * weight_expert_stride;
  scales += static_cast<int64_t>(expert) * scale_expert_stride;
  if constexpr (EnableBias) {
    bias += static_cast<int64_t>(expert) * n;
  }

  int const origin_k = k, interleaved_k = k * Details::kInterleave;

  int const tile_id_m = row, tile_id_n = blockIdx.y, tid = threadIdx.x;
  int const offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
  int const real_offset_n = interleaved_offset_n * Details::kInterleave +
                            ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  int const real_offset_k =
      (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize +
      ((tid * StepK) % Details::LayoutDetails::kTileSize);

  GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
      act, offset_m * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
  GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
      weight, (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW,
      CtaK / Details::kElemsPerByteW, interleaved_k / Details::kElemsPerByteW);
  GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(
      scales,
      (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
      (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  out += offset_m * n + tile_id_n * CtaN * Details::kInterleave;
  if constexpr (EnableBias) {
    bias += tile_id_n * CtaN * Details::kInterleave;
  }

  AccT tile_acc[CtaM * CtaN];
  fill<CtaM * CtaN>(tile_acc, static_cast<AccT>(0.f));

  TypeA vec_scale[CtaN];
  if constexpr (GroupSize == 0) {
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      scales_iterator.load(vec_scale + i, 0, i);
    }
  }

  for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
    TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
    uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
    if constexpr (GroupSize != 0) {
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        scales_iterator.load(vec_scale + i, iter, i);
      }
    }
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized, iter, i);
      dequantize<Details, 1, StepK, false, false>(tile_w, tile_w_quantized, vec_scale + i, nullptr, 1.0f);
      pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
      act_iterator.load(tile_a, iter, i);
      mma<Details, 1, CtaN, StepK, AccT>(tile_acc + i * CtaN, tile_w_pack2, tile_a);
    }
  }
  epilogue<Details, CtaM, CtaN, Threads, EnableBias, false, AccT>(out, n, tile_acc, bias, 1.0f);
#endif
}

template <typename Details, int CtaM, int CtaN, int Threads, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA>
__device__ __forceinline__ void swiglu_epilogue(void* out, void* tile_acc, void* bias,
                                                cutlass_kernels::ActivationParams activation_params) {
  static constexpr int Interleave = Details::kInterleave;
  static constexpr int ThreadsPerInterleavedTile = Details::kThreadsPerInterleavedTile;
  static constexpr int WarpSize = Details::kWarpSize;
  static constexpr int WarpNum = Threads / WarpSize;
  static constexpr int RawCols = CtaN * Interleave;
  static_assert(CtaM == 1);
  static_assert(RawCols % 2 == 0);
  static_assert(Threads % WarpSize == 0);

  __shared__ float shmem[CtaM * CtaN * Interleave * WarpNum];
  int tid = threadIdx.x;
  int warp_id = tid / WarpSize, lane_id = tid % WarpSize;
#pragma unroll
  for (int n = 0; n < CtaN; ++n) {
    float v = static_cast<float>(reinterpret_cast<AccT*>(tile_acc)[n]);
    v = warp_reduce_sum<Interleave, ThreadsPerInterleavedTile>(v);
    if (lane_id < Interleave * ThreadsPerInterleavedTile && lane_id % ThreadsPerInterleavedTile == 0) {
      shmem[warp_id * RawCols + n * Interleave + lane_id / ThreadsPerInterleavedTile] = v;
    }
  }
  __syncthreads();

#pragma unroll
  for (int pair = tid; pair < RawCols / 2; pair += Threads) {
    int const gate_idx = pair * 2;
    int const linear_idx = gate_idx + 1;
    float gate = 0.f;
    float linear = 0.f;
#pragma unroll
    for (int warp = 0; warp < WarpNum; ++warp) {
      gate += shmem[warp * RawCols + gate_idx];
      linear += shmem[warp * RawCols + linear_idx];
    }
    if constexpr (EnableBias) {
      gate += static_cast<float>(reinterpret_cast<TypeA*>(bias)[gate_idx]);
      linear += static_cast<float>(reinterpret_cast<TypeA*>(bias)[linear_idx]);
    }
    if (isfinite(activation_params.limit)) {
      gate = fminf(gate, activation_params.limit);
      linear = fminf(fmaxf(linear, -activation_params.limit), activation_params.limit);
    }
    linear += activation_params.beta;
    float const sigmoid = 1.0f / (1.0f + expf(-activation_params.alpha * gate));
    reinterpret_cast<TypeA*>(out)[pair] = static_cast<TypeA>(gate * sigmoid * linear);
  }
}

template <typename Details, int CtaM, int CtaN, int Threads,
          typename TypeA = typename Details::TypeDetailsA::Type>
__device__ __forceinline__ void atomic_epilogue(void* out, int stride, void* tile_acc) {
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
  static constexpr int Interleave = Details::kInterleave;
  static constexpr int ThreadsPerInterleavedTile = Details::kThreadsPerInterleavedTile;
  static constexpr int WarpSize = Details::kWarpSize;
  static constexpr int WarpNum = Threads / WarpSize;
  static_assert(CtaM == 1);
  static_assert(Threads % WarpSize == 0);

  __shared__ float shmem[CtaM * CtaN * Interleave * WarpNum];
  int tid = threadIdx.x;
  int warp_id = tid / WarpSize, lane_id = tid % WarpSize;
#pragma unroll
  for (int n = 0; n < CtaN; ++n) {
    float v = static_cast<float>(reinterpret_cast<Type*>(tile_acc)[n]);
    v = warp_reduce_sum<Interleave, ThreadsPerInterleavedTile>(v);
    if (lane_id < Interleave * ThreadsPerInterleavedTile && lane_id % ThreadsPerInterleavedTile == 0) {
      shmem[warp_id * CtaN * Interleave + n * Interleave + lane_id / ThreadsPerInterleavedTile] = v;
    }
  }
  __syncthreads();

#pragma unroll
  for (int ii = tid; ii < CtaN * Interleave; ii += Threads) {
    float val = 0.f;
#pragma unroll
    for (int warp = 0; warp < WarpNum; ++warp) {
      val += shmem[warp * CtaN * Interleave + ii];
    }
    atomicAdd(reinterpret_cast<Type*>(out) + ii, static_cast<Type>(val));
  }
  (void)stride;
}

template <typename Details, int CtaN, int Threads, int GroupSize, int SplitK,
          typename TypeA = typename Details::TypeDetailsA::Type>
__global__ void moe_gemv_splitk_atomic_kernel(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr bool Mandatory = true;
  static constexpr int CtaM = 1;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);
  if constexpr (GroupSize != 0) {
    static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
  }

  int const row = blockIdx.x;
  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
#pragma unroll 1
  for (int e = 0; e < num_experts && permuted_row_to_expert == nullptr; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  weight += expert * weight_expert_stride;
  scales += static_cast<int64_t>(expert) * scale_expert_stride;

  int const origin_k = k, interleaved_k = k * Details::kInterleave;
  int const tile_id_m = row, tile_id_n = blockIdx.y, split_id = blockIdx.z, tid = threadIdx.x;
  int const offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
  int const real_offset_n = interleaved_offset_n * Details::kInterleave +
                            ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  int const real_offset_k =
      (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize +
      ((tid * StepK) % Details::LayoutDetails::kTileSize);

  GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
      act, offset_m * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
  GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
      weight, (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW,
      CtaK / Details::kElemsPerByteW, interleaved_k / Details::kElemsPerByteW);
  GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(
      scales,
      (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
      (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  out += offset_m * n + tile_id_n * CtaN * Details::kInterleave;

  TypeA tile_acc[CtaM * CtaN];
  fill<CtaM * CtaN>(tile_acc, static_cast<TypeA>(0.f));

  TypeA vec_scale[CtaN];
  if constexpr (GroupSize == 0) {
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      scales_iterator.load(vec_scale + i, 0, i);
    }
  }

  int const num_iters = (interleaved_k + CtaK - 1) / CtaK;
  for (int iter = split_id; iter < num_iters; iter += SplitK) {
    int const idx_k = iter * CtaK + tid * StepK;
    if (idx_k >= interleaved_k) {
      continue;
    }
    TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
    uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
    if constexpr (GroupSize != 0) {
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        scales_iterator.load(vec_scale + i, iter, i);
      }
    }
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized, iter, i);
      dequantize<Details, 1, StepK, false, false>(tile_w, tile_w_quantized, vec_scale + i, nullptr, 1.0f);
      pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
    }
    act_iterator.load(tile_a, iter, 0);
    mma<Details, 1, CtaN, StepK>(tile_acc, tile_w_pack2, tile_a);
  }
  atomic_epilogue<Details, CtaM, CtaN, Threads>(out, n, tile_acc);
#endif
}

// Reduces the per-thread CtaN dot-product accumulators to CtaN*Interleave raw output
// columns (same warp + shared-memory reduction as `epilogue`/`atomic_epilogue`) and writes
// them to a per-split fp32 partial buffer in natural [0, n) column order. No bias, no
// activation, no atomics — this is the matmul half of the two-pass split-K design.
template <typename Details, int CtaM, int CtaN, int Threads,
          typename TypeA = typename Details::TypeDetailsA::Type>
__device__ __forceinline__ void partial_epilogue(float* partial_out, void* tile_acc) {
  using Type = typename MathWrapper<typename Details::TypeDetailsA>::Type;
  static constexpr int Interleave = Details::kInterleave;
  static constexpr int ThreadsPerInterleavedTile = Details::kThreadsPerInterleavedTile;
  static constexpr int WarpSize = Details::kWarpSize;
  static constexpr int WarpNum = Threads / WarpSize;
  static_assert(CtaM == 1);
  static_assert(Threads % WarpSize == 0);

  __shared__ float shmem[CtaM * CtaN * Interleave * WarpNum];
  int tid = threadIdx.x;
  int warp_id = tid / WarpSize, lane_id = tid % WarpSize;
#pragma unroll
  for (int n = 0; n < CtaN; ++n) {
    float v = static_cast<float>(reinterpret_cast<Type*>(tile_acc)[n]);
    v = warp_reduce_sum<Interleave, ThreadsPerInterleavedTile>(v);
    if (lane_id < Interleave * ThreadsPerInterleavedTile && lane_id % ThreadsPerInterleavedTile == 0) {
      shmem[warp_id * CtaN * Interleave + n * Interleave + lane_id / ThreadsPerInterleavedTile] = v;
    }
  }
  __syncthreads();

#pragma unroll
  for (int ii = tid; ii < CtaN * Interleave; ii += Threads) {
    float val = 0.f;
#pragma unroll
    for (int warp = 0; warp < WarpNum; ++warp) {
      val += shmem[warp * CtaN * Interleave + ii];
    }
    partial_out[ii] = val;
  }
}

// Two-pass split-K, pass 1: each (row, n-tile, split) block computes the partial dot
// products over its strided-K subset and writes fp32 partials to
// ``partials[split_id, row, n]`` (n = inter_size*2 for the fused-SwiGLU FC1 case).
// Biasless and activation-free; the reduction + activation happen in pass 2.
template <typename Details, int CtaN, int Threads, int GroupSize, int SplitK,
          typename TypeA = typename Details::TypeDetailsA::Type>
__global__ void moe_gemv_splitk_partials_kernel(
    TypeA* act, uint8_t* weight, TypeA* scales, float* partials,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int n, int k, int64_t expanded_num_rows) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr bool Mandatory = true;
  static constexpr int CtaM = 1;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);
  if constexpr (GroupSize != 0) {
    static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
  }

  int const row = blockIdx.x;
  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
#pragma unroll 1
  for (int e = 0; e < num_experts && permuted_row_to_expert == nullptr; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  weight += expert * weight_expert_stride;
  scales += static_cast<int64_t>(expert) * scale_expert_stride;

  int const origin_k = k, interleaved_k = k * Details::kInterleave;
  int const tile_id_m = row, tile_id_n = blockIdx.y, split_id = blockIdx.z, tid = threadIdx.x;
  int const offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
  int const real_offset_n = interleaved_offset_n * Details::kInterleave +
                            ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  int const real_offset_k =
      (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize +
      ((tid * StepK) % Details::LayoutDetails::kTileSize);

  GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
      act, offset_m * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
  GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
      weight, (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW,
      CtaK / Details::kElemsPerByteW, interleaved_k / Details::kElemsPerByteW);
  GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(
      scales,
      (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
      (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  // partials[split_id, row, n]: stride between splits is expanded_num_rows * n.
  partials += (static_cast<int64_t>(split_id) * expanded_num_rows + offset_m) * n +
              tile_id_n * CtaN * Details::kInterleave;

  TypeA tile_acc[CtaM * CtaN];
  fill<CtaM * CtaN>(tile_acc, static_cast<TypeA>(0.f));

  TypeA vec_scale[CtaN];
  if constexpr (GroupSize == 0) {
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      scales_iterator.load(vec_scale + i, 0, i);
    }
  }

  int const num_iters = (interleaved_k + CtaK - 1) / CtaK;
  for (int iter = split_id; iter < num_iters; iter += SplitK) {
    int const idx_k = iter * CtaK + tid * StepK;
    if (idx_k >= interleaved_k) {
      continue;
    }
    TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
    uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
    if constexpr (GroupSize != 0) {
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        scales_iterator.load(vec_scale + i, iter, i);
      }
    }
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized, iter, i);
      dequantize<Details, 1, StepK, false, false>(tile_w, tile_w_quantized, vec_scale + i, nullptr, 1.0f);
      pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
    }
    act_iterator.load(tile_a, iter, 0);
    mma<Details, 1, CtaN, StepK>(tile_acc, tile_w_pack2, tile_a);
  }
  partial_epilogue<Details, CtaM, CtaN, Threads>(partials, tile_acc);
#endif
}

// Two-pass split-K, pass 2: reduces the SplitK fp32 partials for each (row, output column)
// in fp32, applies the per-expert SwiGLU bias + gated activation (gate = column 2j, linear =
// column 2j+1, matching ``swiglu_epilogue``), and writes the fp16/bf16 result. One thread per
// output element; grid = (expanded_rows, ceil(inter_size / Threads)).
template <typename Details, int Threads, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type>
__global__ void moe_gemv_splitk_reduce_swiglu_kernel(
    float const* partials, TypeA* bias, TypeA* out,
    int const* permuted_row_to_expert, int num_experts, int64_t const* expert_first_token_offset,
    int inter_size, int split_k, int64_t expanded_num_rows,
    cutlass_kernels::ActivationParams activation_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  int const row = blockIdx.x;
  int const j = blockIdx.y * Threads + threadIdx.x;
  if (j >= inter_size) {
    return;
  }

  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
#pragma unroll 1
  for (int e = 0; e < num_experts && permuted_row_to_expert == nullptr; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  float const* alpha = activation_params.swiglu_alpha;
  float const* beta = activation_params.swiglu_beta;
  float const* limit = activation_params.swiglu_limit;
  float const act_alpha = alpha ? alpha[expert] : activation_params.alpha;
  float const act_beta = beta ? beta[expert] : activation_params.beta;
  float const act_limit = limit ? limit[expert] : activation_params.limit;

  int const n = inter_size * 2;
  int const gate_idx = j * 2;
  int const linear_idx = gate_idx + 1;

  // partials[split, row, col]: stride between splits is expanded_num_rows * n.
  int64_t const row_base = static_cast<int64_t>(row) * n;
  int64_t const split_stride = expanded_num_rows * n;
  float gate = 0.f;
  float linear = 0.f;
  for (int s = 0; s < split_k; ++s) {
    int64_t const base = static_cast<int64_t>(s) * split_stride + row_base;
    gate += partials[base + gate_idx];
    linear += partials[base + linear_idx];
  }

  if constexpr (EnableBias) {
    bias += static_cast<int64_t>(expert) * n;
    gate += static_cast<float>(bias[gate_idx]);
    linear += static_cast<float>(bias[linear_idx]);
  }
  if (isfinite(act_limit)) {
    gate = fminf(gate, act_limit);
    linear = fminf(fmaxf(linear, -act_limit), act_limit);
  }
  linear += act_beta;
  float const sigmoid = 1.0f / (1.0f + expf(-act_alpha * gate));
  out[static_cast<int64_t>(row) * inter_size + j] = static_cast<TypeA>(gate * sigmoid * linear);
#endif
}

template <typename Details, int CtaN, int Threads, int GroupSize, bool EnableBias,
          typename TypeA = typename Details::TypeDetailsA::Type, typename AccT = TypeA>
__global__ void moe_gemv_interleaved_swiglu_kernel(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t weight_expert_stride, int64_t scale_expert_stride, int inter_size, int k,
    cutlass_kernels::ActivationParams activation_params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
  using AccessTypeA = typename Details::AccessTypeA;
  using AccessTypeW = typename Details::AccessTypeW;

  static constexpr bool Mandatory = true;
  static constexpr int CtaM = 1;
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  static_assert(CtaN % 2 == 0);
  if constexpr (GroupSize != 0) {
    static_assert((CtaK / Details::kInterleave) % GroupSize == 0);
  }

  int const row = blockIdx.x;

  int expert = permuted_row_to_expert != nullptr ? permuted_row_to_expert[row] : 0;
#pragma unroll 1
  for (int e = 0; e < num_experts && permuted_row_to_expert == nullptr; ++e) {
    if (row >= static_cast<int>(expert_first_token_offset[e + 1])) {
      expert = e + 1;
      continue;
    }
    break;
  }
  if (expert < 0 || expert >= num_experts) {
    return;
  }

  float const* alpha = activation_params.swiglu_alpha;
  float const* beta = activation_params.swiglu_beta;
  float const* limit = activation_params.swiglu_limit;
  activation_params.alpha = alpha ? alpha[expert] : activation_params.alpha;
  activation_params.beta = beta ? beta[expert] : activation_params.beta;
  activation_params.limit = limit ? limit[expert] : activation_params.limit;

  int const n = inter_size * 2;
  weight += expert * weight_expert_stride;
  scales += static_cast<int64_t>(expert) * scale_expert_stride;
  if constexpr (EnableBias) {
    bias += static_cast<int64_t>(expert) * n;
  }

  int const origin_k = k, interleaved_k = k * Details::kInterleave;

  int const tile_id_m = row, tile_id_n = blockIdx.y, tid = threadIdx.x;
  int const offset_m = tile_id_m * CtaM, interleaved_offset_n = tile_id_n * CtaN;
  int const real_offset_n = interleaved_offset_n * Details::kInterleave +
                            ((tid * StepK / Details::LayoutDetails::kTileSize) % Details::kInterleave);
  int const real_offset_k =
      (tid * StepK / (Details::kInterleave * Details::LayoutDetails::kTileSize)) * Details::LayoutDetails::kTileSize +
      ((tid * StepK) % Details::LayoutDetails::kTileSize);

  GMemIterator<Mandatory, AccessTypeA, CtaM, Details::kAccessNumA, TypeA> act_iterator(
      act, offset_m * origin_k + real_offset_k, CtaK / Details::kInterleave, origin_k);
  GMemIterator<Mandatory, AccessTypeW, CtaN, Details::kAccessNumW, uint8_t> weight_iterator(
      weight, (interleaved_offset_n * interleaved_k + tid * StepK) / Details::kElemsPerByteW,
      CtaK / Details::kElemsPerByteW, interleaved_k / Details::kElemsPerByteW);
  GMemIterator<Mandatory, TypeA, CtaN, 1, TypeA> scales_iterator(
      scales,
      (GroupSize != 0 ? real_offset_k / GroupSize * n : 0) + real_offset_n,
      (GroupSize != 0 ? CtaK / Details::kInterleave / GroupSize * n : 0), Details::kInterleave);

  out += offset_m * inter_size + tile_id_n * CtaN * Details::kInterleave / 2;
  if constexpr (EnableBias) {
    bias += tile_id_n * CtaN * Details::kInterleave;
  }

  AccT tile_acc[CtaM * CtaN];
  fill<CtaM * CtaN>(tile_acc, static_cast<AccT>(0.f));

  TypeA vec_scale[CtaN];
  if constexpr (GroupSize == 0) {
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      scales_iterator.load(vec_scale + i, 0, i);
    }
  }

  for (int idx_k = tid * StepK, iter = 0; idx_k < interleaved_k; idx_k += CtaK, ++iter) {
    TypeA tile_a[StepK], tile_w[StepK], tile_w_pack2[CtaN * StepK];
    uint8_t tile_w_quantized[StepK / Details::kElemsPerByteW];
    if constexpr (GroupSize != 0) {
#pragma unroll
      for (int i = 0; i < CtaN; ++i) {
        scales_iterator.load(vec_scale + i, iter, i);
      }
    }
#pragma unroll
    for (int i = 0; i < CtaN; ++i) {
      weight_iterator.load(tile_w_quantized, iter, i);
      dequantize<Details, 1, StepK, false, false>(tile_w, tile_w_quantized, vec_scale + i, nullptr, 1.0f);
      pack_to_vec2<Details, StepK>(tile_w_pack2, tile_w, i);
    }
#pragma unroll
    for (int i = 0; i < CtaM; ++i) {
      act_iterator.load(tile_a, iter, i);
      mma<Details, 1, CtaN, StepK, AccT>(tile_acc + i * CtaN, tile_w_pack2, tile_a);
    }
  }
  swiglu_epilogue<Details, CtaM, CtaN, Threads, EnableBias, TypeA, AccT>(out, tile_acc, bias, activation_params);
#endif
}

template <typename Details, int CtaN, int Threads, int GroupSize, typename TypeA, typename AccT = TypeA>
static void launch_moe_gemv(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                            int64_t const* expert_first_token_offset, int const* permuted_row_to_expert,
                            int num_experts, int64_t expanded_num_rows, int64_t n, int64_t k,
                            cudaStream_t stream) {
  int64_t const weight_expert_stride = n * k / Details::kElemsPerByteW;
  int64_t const scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(n / (CtaN * Details::kInterleave)));
  dim3 block(Threads);
  if (bias != nullptr) {
    moe_gemv_kernel<Details, CtaN, Threads, GroupSize, true, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k));
  } else {
    moe_gemv_kernel<Details, CtaN, Threads, GroupSize, false, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k));
  }
}

template <typename Details, int CtaN, int Threads, int GroupSize, int SplitK, typename TypeA>
static void launch_moe_gemv_splitk_atomic(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* out,
                                          int64_t const* expert_first_token_offset,
                                          int const* permuted_row_to_expert, int num_experts,
                                          int64_t expanded_num_rows, int64_t n, int64_t k,
                                          cudaStream_t stream) {
  int64_t const weight_expert_stride = n * k / Details::kElemsPerByteW;
  int64_t const scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  CUDA_CALL_THROW(cudaMemsetAsync(out, 0, sizeof(TypeA) * expanded_num_rows * n, stream));
  dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(n / (CtaN * Details::kInterleave)), SplitK);
  dim3 block(Threads);
  moe_gemv_splitk_atomic_kernel<Details, CtaN, Threads, GroupSize, SplitK><<<grid, block, 0, stream>>>(
      act, weight, scales, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
      weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k));
}

template <typename Details, int CtaN, int Threads, int SplitK, typename TypeA>
static void dispatch_moe_gemv_splitk_group_size(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* out,
                                                int64_t const* expert_first_token_offset,
                                                int const* permuted_row_to_expert, int num_experts,
                                                int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                                                cudaStream_t stream) {
  if (group_size <= 0) {
    launch_moe_gemv_splitk_atomic<Details, CtaN, Threads, 0, SplitK, TypeA>(
        act, weight, scales, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, n, k, stream);
  } else if (group_size == 32) {
    launch_moe_gemv_splitk_atomic<Details, CtaN, Threads, 32, SplitK, TypeA>(
        act, weight, scales, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, n, k, stream);
  } else if (group_size == 64) {
    launch_moe_gemv_splitk_atomic<Details, CtaN, Threads, 64, SplitK, TypeA>(
        act, weight, scales, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, n, k, stream);
  } else if (group_size == 128) {
    launch_moe_gemv_splitk_atomic<Details, CtaN, Threads, 128, SplitK, TypeA>(
        act, weight, scales, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, n, k, stream);
  } else {
    ORT_THROW("unsupported MoE GEMV split-K group_size: ", group_size);
  }
}

template <typename Details, int CtaN, int Threads, int GroupSize, typename TypeA, typename AccT = TypeA>
static void launch_moe_gemv_interleaved_swiglu(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k,
    cutlass_kernels::ActivationParams activation_params, cudaStream_t stream) {
  int64_t const n = inter_size * 2;
  int64_t const weight_expert_stride = n * k / Details::kElemsPerByteW;
  int64_t const scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;
  dim3 grid(static_cast<unsigned>(expanded_num_rows), static_cast<unsigned>(n / (CtaN * Details::kInterleave)));
  dim3 block(Threads);
  if (bias != nullptr) {
    moe_gemv_interleaved_swiglu_kernel<Details, CtaN, Threads, GroupSize, true, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(inter_size), static_cast<int>(k), activation_params);
  } else {
    moe_gemv_interleaved_swiglu_kernel<Details, CtaN, Threads, GroupSize, false, TypeA, AccT><<<grid, block, 0, stream>>>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        weight_expert_stride, scale_expert_stride, static_cast<int>(inter_size), static_cast<int>(k), activation_params);
  }
}

template <typename Details, int CtaN, int Threads, typename TypeA, typename AccT = TypeA>
static void dispatch_moe_gemv_group_size(TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
                                         int64_t const* expert_first_token_offset,
                                         int const* permuted_row_to_expert, int num_experts,
                                         int64_t expanded_num_rows, int64_t n, int64_t k,
                                         int group_size, cudaStream_t stream) {
  if (group_size <= 0) {
    launch_moe_gemv<Details, CtaN, Threads, 0, TypeA, AccT>(act, weight, scales, bias, out, expert_first_token_offset,
                                                            permuted_row_to_expert, num_experts, expanded_num_rows, n, k, stream);
  } else if (group_size == 32) {
    launch_moe_gemv<Details, CtaN, Threads, 32, TypeA, AccT>(act, weight, scales, bias, out, expert_first_token_offset,
                                                             permuted_row_to_expert, num_experts, expanded_num_rows, n, k, stream);
  } else if (group_size == 64) {
    launch_moe_gemv<Details, CtaN, Threads, 64, TypeA, AccT>(act, weight, scales, bias, out, expert_first_token_offset,
                                                             permuted_row_to_expert, num_experts, expanded_num_rows, n, k, stream);
  } else if (group_size == 128) {
    launch_moe_gemv<Details, CtaN, Threads, 128, TypeA, AccT>(act, weight, scales, bias, out, expert_first_token_offset,
                                                              permuted_row_to_expert, num_experts, expanded_num_rows, n, k, stream);
  } else {
    ORT_THROW("unsupported MoE GEMV group_size: ", group_size);
  }
}

template <typename Details, int CtaN, int Threads, typename TypeA, typename AccT = TypeA>
static void dispatch_moe_gemv_interleaved_swiglu_group_size(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size,
    cutlass_kernels::ActivationParams activation_params, cudaStream_t stream) {
  if (group_size <= 0) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 0, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else if (group_size == 32) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 32, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else if (group_size == 64) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 64, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else if (group_size == 128) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, 128, TypeA, AccT>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else {
    ORT_THROW("unsupported MoE GEMV group_size: ", group_size);
  }
}

// Two-pass split-K for the fused-SwiGLU FC1 GEMV. Pass 1 writes fp32 partials for each
// K-split; pass 2 sums them in fp32 and applies the per-expert SwiGLU activation. This
// exposes more CTAs along K (filling the GPU at batch-1) while keeping fp32 accumulation
// and the fused activation, unlike the FP16-atomic prototype. The fp32 partial buffer is
// stream-ordered scratch (pool-backed), so it adds no host sync in steady state.
template <typename Details, int CtaN, int Threads, int GroupSize, int SplitK, typename TypeA>
static void launch_moe_gemv_splitk_twopass_swiglu(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k,
    cutlass_kernels::ActivationParams activation_params, cudaStream_t stream) {
  static constexpr int StepK = Details::kStepK;
  static constexpr int CtaK = StepK * Threads;
  int64_t const n = inter_size * 2;
  int64_t const interleaved_k = k * Details::kInterleave;
  int const num_iters = static_cast<int>((interleaved_k + CtaK - 1) / CtaK);

  // Adaptive degrade: when the K loop is a single tile there is no parallelism for split-K
  // to expose and the extra splits would be pure overhead, so fall back to the fused kernel.
  if (num_iters < 2) {
    launch_moe_gemv_interleaved_swiglu<Details, CtaN, Threads, GroupSize, TypeA>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
    return;
  }

  int64_t const weight_expert_stride = n * k / Details::kElemsPerByteW;
  int64_t const scale_expert_stride = GroupSize == 0 ? n : ((k + GroupSize - 1) / GroupSize) * n;

  float* partials = nullptr;
  size_t const partials_bytes = sizeof(float) * static_cast<size_t>(SplitK) *
                                static_cast<size_t>(expanded_num_rows) * static_cast<size_t>(n);
  CUDA_CALL_THROW(cudaMallocAsync(&partials, partials_bytes, stream));

  dim3 grid1(static_cast<unsigned>(expanded_num_rows),
             static_cast<unsigned>(n / (CtaN * Details::kInterleave)), SplitK);
  dim3 block1(Threads);
  moe_gemv_splitk_partials_kernel<Details, CtaN, Threads, GroupSize, SplitK><<<grid1, block1, 0, stream>>>(
      act, weight, scales, partials, expert_first_token_offset, permuted_row_to_expert, num_experts,
      weight_expert_stride, scale_expert_stride, static_cast<int>(n), static_cast<int>(k), expanded_num_rows);

  static constexpr int kReduceThreads = 256;
  dim3 grid2(static_cast<unsigned>(expanded_num_rows),
             static_cast<unsigned>((inter_size + kReduceThreads - 1) / kReduceThreads));
  dim3 block2(kReduceThreads);
  if (bias != nullptr) {
    moe_gemv_splitk_reduce_swiglu_kernel<Details, kReduceThreads, true><<<grid2, block2, 0, stream>>>(
        partials, bias, out, permuted_row_to_expert, num_experts, expert_first_token_offset,
        static_cast<int>(inter_size), SplitK, expanded_num_rows, activation_params);
  } else {
    moe_gemv_splitk_reduce_swiglu_kernel<Details, kReduceThreads, false><<<grid2, block2, 0, stream>>>(
        partials, bias, out, permuted_row_to_expert, num_experts, expert_first_token_offset,
        static_cast<int>(inter_size), SplitK, expanded_num_rows, activation_params);
  }
  CUDA_CALL_THROW(cudaFreeAsync(partials, stream));
}

template <typename Details, int CtaN, int Threads, int SplitK, typename TypeA>
static void dispatch_moe_gemv_splitk_twopass_swiglu_group_size(
    TypeA* act, uint8_t* weight, TypeA* scales, TypeA* bias, TypeA* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size,
    cutlass_kernels::ActivationParams activation_params, cudaStream_t stream) {
  if (group_size <= 0) {
    launch_moe_gemv_splitk_twopass_swiglu<Details, CtaN, Threads, 0, SplitK, TypeA>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else if (group_size == 32) {
    launch_moe_gemv_splitk_twopass_swiglu<Details, CtaN, Threads, 32, SplitK, TypeA>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else if (group_size == 64) {
    launch_moe_gemv_splitk_twopass_swiglu<Details, CtaN, Threads, 64, SplitK, TypeA>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else if (group_size == 128) {
    launch_moe_gemv_splitk_twopass_swiglu<Details, CtaN, Threads, 128, SplitK, TypeA>(
        act, weight, scales, bias, out, expert_first_token_offset, permuted_row_to_expert, num_experts,
        expanded_num_rows, inter_size, k, activation_params, stream);
  } else {
    ORT_THROW("unsupported MoE GEMV split-K group_size: ", group_size);
  }
}

}  // namespace fpA_intB_gemv

namespace moe_gemv {

namespace fiv = onnxruntime::llm::kernels::fpA_intB_gemv;

// Maps a runtime accumulator-precision choice to a compile-time type tag so the
// host launcher can select the fp32- or 16-bit-accumulation kernel instantiation.
template <typename T>
struct TypeTag {
  using type = T;
};

// Opt-in: accumulate the GEMV inner product in 16-bit (fp16) instead of the default
// fp32. Honored only for fp16 activations (bf16 always accumulates in fp32). Set
// ORT_MOE_GEMV_FP16_ACCUM=1 to measure the perf/accuracy tradeoff of 16-bit accumulation.
inline bool MoeGemvUseFp16Accum() {
  static bool const enabled = []() {
    char const* v = std::getenv("ORT_MOE_GEMV_FP16_ACCUM");
    return v != nullptr && v[0] == '1';
  }();
  return enabled;
}

// CtaN/Threads match the dense per-channel (EnableZero=false) m-tile config.
static constexpr int kDefaultCtaN = 8;
static constexpr int kDefaultThreads = 128;
static constexpr int kCtaN16 = 16;
static constexpr int kThreads64 = 64;
// int4 ColumnMajorInterleave (Sm80) tile width along N.
static constexpr int kTileSizeK = 64;
static constexpr int kInt4Interleave = 128 * 8 / (kTileSizeK * 4);  // = 4
static constexpr int kInt8Interleave = 128 * 8 / (kTileSizeK * 8);  // = 2

int CtaNForConfig(MoeGemvConfig config) {
  return config == MoeGemvConfig::kCtaN16 ? kCtaN16 : kDefaultCtaN;
}

bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k,
                           int weight_bits, int group_size) {
  return is_moe_gemv_supported(sm, expanded_num_rows, n, k, weight_bits, group_size, MoeGemvConfig::kDefault);
}

bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k,
                           int weight_bits, int group_size, MoeGemvConfig config) {
  if (sm < 80) {
    return false;
  }
  if (weight_bits != 4 && weight_bits != 8) {
    return false;
  }
  // group_size <= 0 selects the per-column (per-channel) path; block-wise scales must be 32, 64, or 128.
  if (group_size > 0 && group_size != 32 && group_size != 64 && group_size != 128) {
    return false;
  }
  // Keep the first block-wise GEMV implementation on complete K blocks.
  if (group_size > 0 && k % group_size != 0) {
    return false;
  }
  if (expanded_num_rows <= 0 || expanded_num_rows > std::numeric_limits<int>::max() ||
      n <= 0 || n > std::numeric_limits<int>::max() ||
      k <= 0 || k > std::numeric_limits<int>::max()) {
    return false;
  }
  // n must tile evenly; k must tile evenly into StepK along interleaved-K.
  int const interleave = weight_bits == 4 ? kInt4Interleave : kInt8Interleave;
  if (n % (CtaNForConfig(config) * interleave) != 0) {
    return false;
  }
  int64_t const interleaved_k = k * interleave;
  int const step_k = 128 / weight_bits;
  if (interleaved_k % step_k != 0) {
    return false;
  }
  return true;
}

bool is_moe_gemv_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k) {
  return is_moe_gemv_supported(sm, expanded_num_rows, n, k, 4, 0);
}

template <typename T, typename WeightType>
struct DetailsForTAndWeight;

template <>
struct DetailsForTAndWeight<half, cutlass::uint4b_t> {
  using Details = fiv::KernelDetails<fiv::FP16DetailsA, fiv::Int4DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = half;
  static constexpr int kWeightBits = 4;
};

template <>
struct DetailsForTAndWeight<half, uint8_t> {
  using Details = fiv::KernelDetails<fiv::FP16DetailsA, fiv::Int8DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = half;
  static constexpr int kWeightBits = 8;
};

#ifdef ENABLE_BF16
template <>
struct DetailsForTAndWeight<__nv_bfloat16, cutlass::uint4b_t> {
  using Details = fiv::KernelDetails<fiv::BF16DetailsA, fiv::Int4DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = __nv_bfloat16;
  static constexpr int kWeightBits = 4;
};

template <>
struct DetailsForTAndWeight<__nv_bfloat16, uint8_t> {
  using Details = fiv::KernelDetails<fiv::BF16DetailsA, fiv::Int8DetailsW, fiv::ColumnMajorInterleaved, true, kTileSizeK>;
  using TypeA = __nv_bfloat16;
  static constexpr int kWeightBits = 8;
};
#endif

template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric(T const* act, WeightType const* weight, T const* scales, T const* bias, T* out,
                                   int64_t const* expert_first_token_offset, int const* permuted_row_to_expert,
                                   int num_experts,
                                   int64_t expanded_num_rows, int64_t n, int64_t k, int group_size, int sm,
                                   MoeGemvConfig config, cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = typename DetailsForTAndWeight<T, WeightType>::Details;
  using TypeA = typename DetailsForTAndWeight<T, WeightType>::TypeA;
  if (config == MoeGemvConfig::kSplitK2) {
    if constexpr (std::is_same_v<T, half>) {
      fiv::dispatch_moe_gemv_splitk_group_size<Details, kDefaultCtaN, kDefaultThreads, 2, TypeA>(
          const_cast<TypeA*>(reinterpret_cast<TypeA const*>(act)),
          const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(weight)),
          const_cast<TypeA*>(reinterpret_cast<TypeA const*>(scales)),
          reinterpret_cast<TypeA*>(out),
          expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size, stream);
      return;
    } else {
      ORT_THROW("MoE GEMV split-K prototype supports FP16 activations only.");
    }
  }
  // Accumulate in fp32 by default. fp16 activations may opt back into 16-bit accumulation
  // via ORT_MOE_GEMV_FP16_ACCUM=1; bf16 always accumulates in fp32 (16-bit bf16 accumulation
  // is too lossy). use_fp32_accum selects the kernel's AccT at runtime.
  bool const use_fp32_accum = !std::is_same_v<T, half> || !MoeGemvUseFp16Accum();
  auto launch = [&](auto cta_n, auto threads, auto acc_tag) {
    using AccT = typename decltype(acc_tag)::type;
    fiv::dispatch_moe_gemv_group_size<Details, cta_n(), threads(), TypeA, AccT>(
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(act)),
        const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(weight)),
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(scales)),
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(bias)),
        reinterpret_cast<TypeA*>(out),
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size, stream);
  };
  auto launch_cfg = [&](auto cta_n, auto threads) {
    if (use_fp32_accum) {
      launch(cta_n, threads, TypeTag<float>{});
    } else {
      launch(cta_n, threads, TypeTag<TypeA>{});
    }
  };
  if (config == MoeGemvConfig::kCtaN16) {
    launch_cfg([] { return kCtaN16; }, [] { return kDefaultThreads; });
  } else if (config == MoeGemvConfig::kThreads64) {
    launch_cfg([] { return kDefaultCtaN; }, [] { return kThreads64; });
  } else {
    launch_cfg([] { return kDefaultCtaN; }, [] { return kDefaultThreads; });
  }
}

template <typename T, typename WeightType>
void launch_moe_gemv_int_symmetric_interleaved_swiglu(
    T const* act, WeightType const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size, int sm,
    cutlass_kernels::ActivationParams activation_params, MoeGemvConfig config, cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = typename DetailsForTAndWeight<T, WeightType>::Details;
  using TypeA = typename DetailsForTAndWeight<T, WeightType>::TypeA;
  if (config == MoeGemvConfig::kSplitK2) {
    if constexpr (std::is_same_v<T, half>) {
      // Two-pass fp32 split-K keeps the SwiGLU fusion (pass 2 applies the activation while
      // reducing the fp32 partials), unlike the plain-path FP16-atomic split-K prototype.
      fiv::dispatch_moe_gemv_splitk_twopass_swiglu_group_size<Details, kDefaultCtaN, kDefaultThreads, 2, TypeA>(
          const_cast<TypeA*>(reinterpret_cast<TypeA const*>(act)),
          const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(weight)),
          const_cast<TypeA*>(reinterpret_cast<TypeA const*>(scales)),
          const_cast<TypeA*>(reinterpret_cast<TypeA const*>(bias)),
          reinterpret_cast<TypeA*>(out),
          expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
          activation_params, stream);
      return;
    } else {
      ORT_THROW("MoE GEMV split-K prototype supports FP16 activations only.");
    }
  }
  // Accumulate in fp32 by default (see launch_moe_gemv_int_symmetric for the policy).
  bool const use_fp32_accum = !std::is_same_v<T, half> || !MoeGemvUseFp16Accum();
  auto launch = [&](auto cta_n, auto threads, auto acc_tag) {
    using AccT = typename decltype(acc_tag)::type;
    fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<Details, cta_n(), threads(), TypeA, AccT>(
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(act)),
        const_cast<uint8_t*>(reinterpret_cast<uint8_t const*>(weight)),
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(scales)),
        const_cast<TypeA*>(reinterpret_cast<TypeA const*>(bias)),
        reinterpret_cast<TypeA*>(out),
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
        activation_params, stream);
  };
  auto launch_cfg = [&](auto cta_n, auto threads) {
    if (use_fp32_accum) {
      launch(cta_n, threads, TypeTag<float>{});
    } else {
      launch(cta_n, threads, TypeTag<TypeA>{});
    }
  };
  if (config == MoeGemvConfig::kCtaN16) {
    launch_cfg([] { return kCtaN16; }, [] { return kDefaultThreads; });
  } else if (config == MoeGemvConfig::kThreads64) {
    launch_cfg([] { return kDefaultCtaN; }, [] { return kThreads64; });
  } else {
    launch_cfg([] { return kDefaultCtaN; }, [] { return kDefaultThreads; });
  }
}

template <typename T>
void launch_moe_gemv_int4_per_channel(T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
                                      int64_t const* expert_first_token_offset, int const* permuted_row_to_expert,
                                      int num_experts,
                                      int64_t expanded_num_rows, int64_t n, int64_t k, int sm, MoeGemvConfig config,
                                      cudaStream_t stream) {
  launch_moe_gemv_int_symmetric<T, cutlass::uint4b_t>(
      act, reinterpret_cast<cutlass::uint4b_t const*>(weight), scales, bias, out, expert_first_token_offset,
      permuted_row_to_expert, num_experts, expanded_num_rows, n, k, 0, sm, config, stream);
}

template <typename T>
void launch_moe_gemv_int4_per_channel_interleaved_swiglu(
    T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int sm,
    cutlass_kernels::ActivationParams activation_params, MoeGemvConfig config, cudaStream_t stream) {
  launch_moe_gemv_int_symmetric_interleaved_swiglu<T, cutlass::uint4b_t>(
      act, reinterpret_cast<cutlass::uint4b_t const*>(weight), scales, bias, out, expert_first_token_offset,
      permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, 0, sm, activation_params, config, stream);
}

// --- MXFP4 (e2m1) GEMV: non-interleaved ColumnMajor layout (kInterleave = 1) ---
// Weights are the QMoERepackFP4ColToRow output ([experts, n, k/2] row-major, two e2m1
// codes per byte, even-K in the low nibble). Block scales are the
// LaunchQMoECombineFp4ScalesForGemv output ([experts, k/32, n] in TypeA, already folded with
// the per-expert global scale). The kernel decodes e2m1 in-register via Fp4I2FConverter.
template <typename T>
struct Fp4ADetails;
template <>
struct Fp4ADetails<half> {
  using Type = fiv::FP16DetailsA;
};
#ifdef ENABLE_BF16
template <>
struct Fp4ADetails<__nv_bfloat16> {
  using Type = fiv::BF16DetailsA;
};
#endif

// TileSizeK is unused by the ColumnMajor (kInterleave = 1) indexing/reduction beyond the
// shmem-write lane gating (only lane 0 of each warp writes), so 64 matches the INT convention.
static constexpr int kTileSizeKFp4 = 64;
template <typename T>
using Fp4KernelDetails =
    fiv::KernelDetails<typename Fp4ADetails<T>::Type, fiv::Fp4DetailsW, fiv::ColumnMajor, false, kTileSizeKFp4>;

// MXFP4 GEMV shape support. Mirrors is_moe_gemv_supported but for the non-interleaved
// ColumnMajor layout: kInterleave = 1, so n need only be divisible by the CtaN tile width
// selected by `config`, and the per-thread step is StepK = 128 / activation_bits = 8
// (not 128 / weight_bits).
bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                               MoeGemvConfig config) {
  if (sm < 80) {
    return false;
  }
  if (group_size != 32) {  // MXFP4 block size
    return false;
  }
  if (k % group_size != 0) {
    return false;
  }
  if (expanded_num_rows <= 0 || expanded_num_rows > kMaxProfiledExpandedRows) {
    return false;
  }
  if (n < kMinProfiledProblemDim || k < kMinProfiledProblemDim) {
    return false;
  }
  if (expanded_num_rows > kMaxProfiledExpandedRowsForSmallProblemDim &&
      (n < kMinProfiledProblemDimForExpandedRowsAbove4 || k < kMinProfiledProblemDimForExpandedRowsAbove4)) {
    return false;
  }
  if (n % CtaNForConfig(config) != 0) {  // kInterleave = 1
    return false;
  }
  // StepK = 128 / activation_bits = 8; k is a multiple of 32, so k % 8 == 0 always holds.
  if (k % (128 / 16) != 0) {
    return false;
  }
  return true;
}

bool is_moe_gemv_fp4_supported(int sm, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size) {
  return is_moe_gemv_fp4_supported(sm, expanded_num_rows, n, k, group_size, MoeGemvConfig::kDefault);
}

template <typename T>
void launch_moe_gemv_fp4_symmetric(T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
                                   int64_t const* expert_first_token_offset, int const* permuted_row_to_expert,
                                   int num_experts, int64_t expanded_num_rows, int64_t n, int64_t k, int group_size,
                                   int sm, MoeGemvConfig config, cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = Fp4KernelDetails<T>;
  // CtaN/Threads are pure parallelization/tiling knobs: the reduction and 16-bit accumulation
  // are identical for every config, so this sweep is numerically bit-exact (no accuracy gate).
  auto launch = [&](auto cta_n, auto threads) {
    fiv::dispatch_moe_gemv_group_size<Details, cta_n(), threads(), T>(
        const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, n, k, group_size, stream);
  };
  if (config == MoeGemvConfig::kCtaN16) {
    launch([] { return kCtaN16; }, [] { return kDefaultThreads; });
  } else if (config == MoeGemvConfig::kThreads64) {
    launch([] { return kDefaultCtaN; }, [] { return kThreads64; });
  } else {
    launch([] { return kDefaultCtaN; }, [] { return kDefaultThreads; });
  }
}

template <typename T>
void launch_moe_gemv_fp4_symmetric_interleaved_swiglu(
    T const* act, uint8_t const* weight, T const* scales, T const* bias, T* out,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_expert, int num_experts,
    int64_t expanded_num_rows, int64_t inter_size, int64_t k, int group_size, int sm,
    cutlass_kernels::ActivationParams activation_params, MoeGemvConfig config, cudaStream_t stream) {
  ORT_UNUSED_PARAMETER(sm);
  using Details = Fp4KernelDetails<T>;
  // CtaN/Threads are numerically bit-exact across configs (see launch_moe_gemv_fp4_symmetric).
  auto launch = [&](auto cta_n, auto threads) {
    fiv::dispatch_moe_gemv_interleaved_swiglu_group_size<Details, cta_n(), threads(), T>(
        const_cast<T*>(act), const_cast<uint8_t*>(weight), const_cast<T*>(scales), const_cast<T*>(bias), out,
        expert_first_token_offset, permuted_row_to_expert, num_experts, expanded_num_rows, inter_size, k, group_size,
        activation_params, stream);
  };
  if (config == MoeGemvConfig::kCtaN16) {
    launch([] { return kCtaN16; }, [] { return kDefaultThreads; });
  } else if (config == MoeGemvConfig::kThreads64) {
    launch([] { return kDefaultCtaN; }, [] { return kThreads64; });
  } else {
    launch([] { return kDefaultCtaN; }, [] { return kDefaultThreads; });
  }
}

template void launch_moe_gemv_fp4_symmetric<half>(
    half const*, uint8_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_fp4_symmetric_interleaved_swiglu<half>(
    half const*, uint8_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, MoeGemvConfig, cudaStream_t);
#ifdef ENABLE_BF16
template void launch_moe_gemv_fp4_symmetric<__nv_bfloat16>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_fp4_symmetric_interleaved_swiglu<__nv_bfloat16>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    MoeGemvConfig, cudaStream_t);
#endif

template void launch_moe_gemv_int_symmetric<half, cutlass::uint4b_t>(
    half const*, cutlass::uint4b_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_int_symmetric<half, uint8_t>(
    half const*, uint8_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<half, cutlass::uint4b_t>(
    half const*, cutlass::uint4b_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<half, uint8_t>(
    half const*, uint8_t const*, half const*, half const*, half*, int64_t const*, int const*, int,
    int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams, MoeGemvConfig, cudaStream_t);

template void launch_moe_gemv_int4_per_channel<half>(half const*, uint8_t const*, half const*, half const*, half*,
                                                     int64_t const*, int const*, int, int64_t, int64_t, int64_t,
                                                     int, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_int4_per_channel_interleaved_swiglu<half>(
    half const*, uint8_t const*, half const*, half const*, half*, int64_t const*, int const*, int, int64_t,
    int64_t, int64_t, int, cutlass_kernels::ActivationParams, MoeGemvConfig, cudaStream_t);

#ifdef ENABLE_BF16
template void launch_moe_gemv_int_symmetric<__nv_bfloat16, cutlass::uint4b_t>(
    __nv_bfloat16 const*, cutlass::uint4b_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_int_symmetric<__nv_bfloat16, uint8_t>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<__nv_bfloat16, cutlass::uint4b_t>(
    __nv_bfloat16 const*, cutlass::uint4b_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_int_symmetric_interleaved_swiglu<__nv_bfloat16, uint8_t>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, int, cutlass_kernels::ActivationParams,
    MoeGemvConfig, cudaStream_t);

template void launch_moe_gemv_int4_per_channel<__nv_bfloat16>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, MoeGemvConfig, cudaStream_t);
template void launch_moe_gemv_int4_per_channel_interleaved_swiglu<__nv_bfloat16>(
    __nv_bfloat16 const*, uint8_t const*, __nv_bfloat16 const*, __nv_bfloat16 const*, __nv_bfloat16*,
    int64_t const*, int const*, int, int64_t, int64_t, int64_t, int, cutlass_kernels::ActivationParams,
    MoeGemvConfig, cudaStream_t);
#endif
}  // namespace moe_gemv
}  // namespace kernels
}  // namespace onnxruntime::llm
