// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
/******************************************************************************
 * Flash Attention Kernel with Int8 MMA and Int8 KV Cache
 *
 * This kernel implements Flash Attention using Int8 Tensor Cores (MMA).
 * - Q is quantized on-the-fly from FP16/BF16 to Int8.
 * - K and V are loaded as Int8.
 * - Computation uses SM80 Int8 MMA (16x8x32), accumulating into Int32.
 * - Output is converted back to FP16/BF16.
 *
 * Key Differences from Dequantization Kernel:
 * - Uses Int8 MMA instructions not FP16 MMA.
 * - Quantizes Q online instead of dequantizing K/V.
 * - Accumulates in Int32.
 *
 ******************************************************************************/
#pragma once

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4100)
#pragma warning(disable : 4101)
#pragma warning(disable : 4189)

#endif

#include <cute/tensor.hpp>

#include <cute/algorithm/copy.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "contrib_ops/cuda/bert/flash_attention/block_info.h"
#include "contrib_ops/cuda/bert/flash_attention/kernel_traits.h"
#include "contrib_ops/cuda/bert/flash_attention/utils.h"
#include "contrib_ops/cuda/bert/flash_attention/softmax.h"
#include "contrib_ops/cuda/bert/flash_attention/mask.h"

// Set to 1 to enable debug prints for this kernel
#define FLASH_INT8_QUANT_DEBUG 0

namespace onnxruntime {
namespace flash {
using namespace cute;
namespace int8_mma {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel Traits for Int8 MMA
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = cutlass::half_t>
struct Flash_int8_quant_kernel_traits {
  using Element = elem_type;
  using ElementAccum = float;
  using ElementInt8 = int8_t;
  using ElementInt32 = int32_t;
  using index_t = int64_t;

  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * 32;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;

  static_assert(kHeadDim % 32 == 0, "kHeadDim must be multiple of 32");

  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

  // Int8 MMA Atom
  // Uses SM80_16x8x32_S32S8S8S32_TN
  // A: s8 (M, K), B: s8 (N, K), C: s32 (M, N)
  // for Q@K^T: Q (M,K), K^T (K,N).
  // Cute/Cutlass TN convention: A is RowMajor (M, K), B is ColMajor (K, N).
  // If we load B (K) as (N, K) RowMajor, then B^T is (K, N) ColMajor, which matches TN requirement.
  using MMA_Atom_Arch = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;

  using TiledMma = TiledMMA<
      MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _8, _32>>;  // K=32 for Int8 MMA atom

  // For P@V, we use FP16 MMA.
  // P (FP16/BF16), V (Int8 -> FP16 dummy load).
  // We employ standard Mixed Precision MMA for P@V.
  using MMA_Atom_PV = std::conditional_t<
      std::is_same_v<elem_type, cutlass::half_t>,
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using TiledMma_PV = TiledMMA<
      MMA_Atom_PV,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _8, _16>>;

  // Shared memory layouts
  using SmemLayoutAtom = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                                              Layout<Shape<_8, Int<kBlockKSmem>>,
                                                     Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));

  using SmemLayoutK = decltype(tile_to_shape(
      SmemLayoutAtom{},
      Shape<Int<kBlockN>, Int<kHeadDim>>{}));

  using SmemLayoutV = SmemLayoutK;

  // Int8 Layout for K/V (Shared Memory)
  static constexpr int kSmemRowPaddingInt8 = 16;
  static constexpr int kSmemRowStrideInt8 = kHeadDim + kSmemRowPaddingInt8;

  using SmemLayoutKInt8 = decltype(make_layout(
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      Stride<Int<kSmemRowStrideInt8>, _1>{}));

  // Gmem Copy Atoms
  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;

  using GmemLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;

  using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                                                    GmemLayoutAtom{},
                                                    Layout<Shape<_1, _8>>{}));

  using GmemTiledCopyKInt8 = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementInt8>{},
                                                      GmemLayoutAtom{},
                                                      Layout<Shape<_1, _16>>{}));

  // Output Smem Layout
  using SmemLayoutO = decltype(make_layout(Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                           Stride<Int<kHeadDim>, _1>{}));

  using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
  using SmemCopyAtom = Copy_Atom<DefaultCopy, Element>;
  using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, cute::uint128_t>{},
                                                  GmemLayoutAtom{},
                                                  Layout<Shape<_1, _8>>{}));
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Q Quantization Helper
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TensorSrc, typename TensorDst>
__forceinline__ __device__ float quantize_q_block(TensorSrc const& src, TensorDst& dst) {
  // src: FP16/BF16 fragment
  // dst: Int8 fragment
  // Return scale (absmax / 127)

  float absmax = 1e-6f;
  for (int i = 0; i < size(src); ++i) {
    absmax = fmaxf(absmax, fabsf(static_cast<float>(src(i))));
  }

// Warp-wide reduction to find a common max for all threads in the warp.
// This is necessary because Tensor Core MMA operations spread work across threads
// that must share a consistent scale for correct accumulation.
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    absmax = fmaxf(absmax, __shfl_xor_sync(0xFFFFFFFF, absmax, offset));
  }

  float scale = absmax / 127.0f;
  float scale_inv = 127.0f / absmax;

  for (int i = 0; i < size(src); ++i) {
    float val = static_cast<float>(src(i)) * scale_inv;
    val = fmaxf(-127.0f, fminf(127.0f, roundf(val)));
    dst(i) = static_cast<int8_t>(val);
  }

  return scale;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main Kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
          bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_1rowblock(
    const Params& params, const int bidb, const int bidh, const int m_block,
    const int n_split_idx, const int num_n_splits) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using ElementInt8 = typename Kernel_traits::ElementInt8;
  using ElementInt32 = typename Kernel_traits::ElementInt32;
  using index_t = typename Kernel_traits::index_t;

  // Use ElementO for output
  using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

  extern __shared__ char smem_[];

  const int tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  const BlockInfo<!Is_even_MN> binfo(params, bidb);

  if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

  // Split-K logic
  const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
  const int n_block_min = !Is_local
                              ? n_split_idx * n_blocks_per_split
                              : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
  int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx == num_n_splits - 1) ? cute::ceil_div(binfo.actual_seqlen_k, kBlockN) : (n_split_idx + 1) * n_blocks_per_split);
  if (Is_causal || Is_local) {
    int w_right = params.window_size_right;
    if (w_right < 0) w_right = 0;
    n_block_max = std::min(n_block_max,
                           cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + w_right, kBlockN));
  }

  if (n_block_min >= n_block_max) {
    if constexpr (Split) {
      const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_rounded;
      const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
      Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                     Shape<Int<kBlockM>>{}, Stride<_1>{});
      // writes
      for (int i = tidx; i < kBlockM; i += Kernel_traits::kNThreads) {
        if (i < binfo.actual_seqlen_q - m_block * kBlockM) {
          gLSEaccum(i) = -INFINITY;
        }
      }
      ElementAccum* gOaccum_ptr = reinterpret_cast<ElementAccum*>(params.oaccum_ptr) + row_offset_oaccum;
      int total_elems = kBlockM * kHeadDim;
      for (int i = tidx; i < total_elems; i += Kernel_traits::kNThreads) {
        if (i / kHeadDim < binfo.actual_seqlen_q - m_block * kBlockM) gOaccum_ptr[i] = 0.0f;
      }
    }
    return;
  }

  // Shared Memory Tensors
  constexpr int kSmemSizeQ = kBlockM * kHeadDim * sizeof(Element);
  constexpr int kSmemSizeKInt8 = kBlockN * Kernel_traits::kSmemRowStrideInt8 * sizeof(ElementInt8);

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(smem_ + kSmemSizeQ)),
                          typename Kernel_traits::SmemLayoutKInt8{});
  Tensor sV = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(smem_ + kSmemSizeQ + kSmemSizeKInt8)),
                          typename Kernel_traits::SmemLayoutKInt8{});
  Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutO{});

  // Global Memory Pointers
  const int kv_head_idx = bidh / params.h_h_k_ratio;
  const index_t k_base_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                kv_head_idx * params.k_head_stride;
  const index_t v_base_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                kv_head_idx * params.v_head_stride;

  Tensor mK_int8 = make_tensor(make_gmem_ptr(reinterpret_cast<const ElementInt8*>(params.k_ptr) + k_base_offset),
                               make_shape(binfo.actual_seqlen_k, params.d),
                               make_stride(params.k_row_stride, _1{}));
  Tensor mV_int8 = make_tensor(make_gmem_ptr(reinterpret_cast<const ElementInt8*>(params.v_ptr) + v_base_offset),
                               make_shape(binfo.actual_seqlen_k, params.d),
                               make_stride(params.v_row_stride, _1{}));

  Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(params.q_ptr) +
                                        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                          make_shape(binfo.actual_seqlen_q, params.h, params.d),
                          make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

  Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(Split ? params.oaccum_ptr : params.o_ptr) +
                                        (Split
                                             ? (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_rounded
                                             : binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb))),
                          make_shape(binfo.actual_seqlen_q, params.h, params.d),
                          make_stride(Split ? kHeadDim : params.o_row_stride, Split ? _0{} : params.o_head_stride, _1{}));
  Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

  // Scales - use 0 for per-tensor (k_quant_type==1), kv_head_idx for per-channel (k_quant_type==2)
  const int scale_idx = (params.k_quant_type == 2) ? kv_head_idx : 0;
  const float k_scale_global = params.k_scale_ptr ? static_cast<float>(reinterpret_cast<const Element*>(params.k_scale_ptr)[scale_idx]) : 1.0f;
  const float v_scale_global = params.v_scale_ptr ? static_cast<float>(reinterpret_cast<const Element*>(params.v_scale_ptr)[scale_idx]) : 1.0f;

  // 1. Load Q (FP16)
  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

  // Masks
  Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
  }

  copy<Is_even_MN, Is_even_K, !Is_even_MN, true>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                                 binfo.actual_seqlen_q - m_block * kBlockM);
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  // 2. Prepare for Q@K Loop
  typename Kernel_traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);

  // Q Loading from Smem (FP16)
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);  // FP16

  // Storage for Q fragments
  // tSrQ_int8 is used for MMA
  Tensor tSrQ_int8 = thr_mma.partition_fragment_A(sQ);
  // tSrQ_fp16_storage handles loading
  auto tSrQ_fp16_storage = make_fragment_like<Element>(tSrQ_int8);

  // K/V Int8 Copier
  typename Kernel_traits::GmemTiledCopyKInt8 gmem_tiled_copy_KInt8;
  auto gmem_thr_copy_KInt8 = gmem_tiled_copy_KInt8.get_thread_slice(tidx);

  using SmemCopyAtomInt8Default = Copy_Atom<DefaultCopy, ElementInt8>;
  auto smem_tiled_copy_KInt8 = make_tiled_copy_B(SmemCopyAtomInt8Default{}, tiled_mma);
  auto smem_thr_copy_KInt8 = smem_tiled_copy_KInt8.get_thread_slice(tidx);

  // Setup Accumulator (Int32)
  Tensor acc_s_int32 = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
  clear(acc_s_int32);

  // Define acc_o and softmax outside loop
  // acc_o accumulates Result of P@V. Layout should match P@V result (M, HeadDim).
  typename Kernel_traits::TiledMma_PV tiled_mma_pv_def;
  Tensor acc_o = partition_fragment_C(tiled_mma_pv_def, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(acc_o);

  constexpr int kNRows = 2 * decltype(size<1>(acc_o))::value;
  Softmax<kNRows> softmax;

  // Load Q tile (FP16) and Quantize
  cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_fp16_storage);
  float q_scale_frf = quantize_q_block(tSrQ_fp16_storage, tSrQ_int8);

  // 3. Loop
  bool is_first_block = true;
  for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
    // Clear accumulator for this block (otherwise it accumulates across all blocks)
    clear(acc_s_int32);

    // Load K/V Int8 from Gmem -> Smem
    Tensor gK_int8 = local_tile(mK_int8, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));
    Tensor gV_int8 = local_tile(mV_int8, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));

    Tensor tKgK_int8 = gmem_thr_copy_KInt8.partition_S(gK_int8);
    Tensor tKsK_int8 = gmem_thr_copy_KInt8.partition_D(sK);
    Tensor tVgV_int8 = gmem_thr_copy_KInt8.partition_S(gV_int8);
    Tensor tVsV_int8 = gmem_thr_copy_KInt8.partition_D(sV);

    // Preds
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKcK = gmem_thr_copy_KInt8.partition_D(cK);
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK_int8)));
    if (!Is_even_K) {
#pragma unroll
      for (int k = 0; k < size(tKpK); ++k) tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d;
    }

    // Copy K/V Int8
    copy<Is_even_MN, Is_even_K, !Is_even_MN, true>(gmem_tiled_copy_KInt8, tKgK_int8, tKsK_int8, tKcK, tKpK,
                                                   binfo.actual_seqlen_k - n_block * kBlockN);
    copy<Is_even_MN, Is_even_K, !Is_even_MN, true>(gmem_tiled_copy_KInt8, tVgV_int8, tVsV_int8, tKcK, tKpK,
                                                   binfo.actual_seqlen_k - n_block * kBlockN);

    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // Load K (Int8) from Smem -> Regs
    Tensor tSsK = smem_thr_copy_KInt8.partition_S(sK);
    Tensor tSrK_int8 = thr_mma.partition_fragment_B(sK);
    cute::copy(smem_tiled_copy_KInt8, tSsK, tSrK_int8);

    // Execute Int8 MMA
    cute::gemm(tiled_mma, acc_s_int32, tSrQ_int8, tSrK_int8, acc_s_int32);

    // Process Accumulators (Int32 -> FP16) for Softmax
    float combined_scale = q_scale_frf * k_scale_global;

    auto acc_s_fp = make_fragment_like<ElementAccum>(acc_s_int32);
    for (int i = 0; i < size(acc_s_int32); ++i) {
      acc_s_fp(i) = static_cast<float>(acc_s_int32(i)) * combined_scale;
    }

#if FLASH_INT8_QUANT_DEBUG
    // DEBUG: Check for NaN after Q@K scaling
    {
      bool has_nan = false;
      for (int i = 0; i < size(acc_s_fp); ++i) {
        if (isnan(acc_s_fp(i)) || isinf(acc_s_fp(i))) {
          has_nan = true;
          break;
        }
      }
      if (has_nan && tidx == 0) {
        printf("DEBUG Q@K NaN FOUND: bidb=%d bidh=%d n_block=%d q_scale=%.6f k_scale=%.6f\n",
               bidb, bidh, n_block, q_scale_frf, k_scale_global);
      }
      // Sanity check print
      if (bidb == 0 && bidh == 0 && m_block == 0 && n_block == n_block_max - 1 && tidx == 0) {
        printf("DEBUG Q@K SANITY: q_scale=%.6f k_scale=%.6f\n", q_scale_frf, k_scale_global);
      }
    }
#endif

    // Masking
    constexpr int kNWarps = Kernel_traits::kNWarps;
    const int col_idx_offset = n_block * kBlockN;
    const int row_idx_offset = m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4;
    const int warp_row_stride = kNWarps * 16;
    Mask<Is_causal, Is_local, /*Has_alibi=*/false> mask(
        binfo.actual_seqlen_k, binfo.actual_seqlen_q,
        params.window_size_left, params.window_size_right);
    mask.template apply_mask<Is_causal, Is_even_MN>(
        acc_s_fp, col_idx_offset, row_idx_offset, warp_row_stride);

    if constexpr (Is_softcap) {
      apply_softcap(acc_s_fp, params.softcap);
    }

    // Softmax
    // Rescale output accumulation
    if (is_first_block) {
      softmax.template softmax_rescale_o</*Is_first=*/true>(acc_s_fp, acc_o, params.scale_softmax_log2);
      is_first_block = false;
    } else {
      softmax.template softmax_rescale_o</*Is_first=*/false>(acc_s_fp, acc_o, params.scale_softmax_log2);
    }

#if FLASH_INT8_QUANT_DEBUG
    // DEBUG: Check for NaN after Softmax
    {
      bool has_nan = false;
      for (int i = 0; i < size(acc_s_fp); ++i) {
        if (isnan(acc_s_fp(i)) || isinf(acc_s_fp(i))) {
          has_nan = true;
          break;
        }
      }
      if (has_nan && tidx == 0) {
        printf("DEBUG Softmax NaN FOUND: bidh=%d n_block=%d\n", bidh, n_block);
      }
    }
#endif

    // Define PV MMA traits
    typename Kernel_traits::TiledMma_PV tiled_mma_pv;
    auto thr_mma_pv = tiled_mma_pv.get_thread_slice(tidx);

    // P @ V (Mixed Precision: FP16/BF16 MMA)
    // We need to reshuffle P (acc_s_fp) from Q@K layout to P@V A-operand layout via Smem (sP)

    // 1. Store P (acc_s_fp) to sP
    constexpr int kSmemOffsetP = kSmemSizeQ + kSmemSizeKInt8 + kSmemSizeKInt8;
    Tensor sP = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + kSmemOffsetP)),
                            Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>{});  // RowMajor P

    using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
    auto smem_tiled_copy_P_store = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_P_store = smem_tiled_copy_P_store.get_thread_slice(tidx);
    Tensor tSsP = smem_thr_copy_P_store.partition_D(sP);
    Tensor rP_half = convert_type<Element>(acc_s_fp);
    cute::copy(smem_tiled_copy_P_store, rP_half, tSsP);

    __syncthreads();

    // 2. Load P from sP as A-operand for P@V
    auto smem_tiled_copy_P_load = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_pv);
    auto smem_thr_copy_P_load = smem_tiled_copy_P_load.get_thread_slice(tidx);
    Tensor tLsP = smem_thr_copy_P_load.partition_S(sP);
    Tensor tOrP = thr_mma_pv.partition_fragment_A(sP);
    cute::copy(smem_tiled_copy_P_load, tLsP, tOrP);

    // 3. Load V (Int8) from sV, transpose to ColMajor for MMA, Dequantize
    auto sV_transposed = make_tensor(sV.data(), make_layout(make_shape(size<1>(sV), size<0>(sV)), make_stride(_1{}, get<0>(sV.stride()))));

    // Use DefaultCopy for safe fallback (keep gemm_quant_manual)
    using SmemCopyAtomInt8PV = Copy_Atom<DefaultCopy, ElementInt8>;
    auto smem_tiled_copy_V_quant = make_tiled_copy_B(SmemCopyAtomInt8PV{}, tiled_mma_pv);
    auto smem_thr_copy_V_quant = smem_tiled_copy_V_quant.get_thread_slice(tidx);

    Tensor tSsVt_quant = smem_thr_copy_V_quant.partition_S(sV_transposed);
    // Allocate buffer for Dequantized V (FP16)
    auto tOrVt_fp16 = make_fragment_like<Element>(thr_mma_pv.partition_fragment_B(sV_transposed));

    // 4. MMA P@V (with on-the-fly vectorized dequantization)
    flash::gemm_quant_manual<false>(acc_o, tOrP, tOrVt_fp16, tSsVt_quant, tiled_mma_pv, smem_tiled_copy_V_quant, smem_thr_copy_V_quant, v_scale_global);

    __syncthreads();
  }

  // Finalize
  // Normalize acc_o
  float sink = (params.head_sink_ptr != nullptr)
                   ? static_cast<float>(reinterpret_cast<const Element*>(params.head_sink_ptr)[bidh])
                   : (params.smooth_softmax ? 0.0f : -kInfinity);
  Tensor lse = softmax.template normalize_softmax_lse<Split>(acc_o, params.scale_softmax, sink);

  // Store Reg -> Smem
  Tensor rO = convert_type<Element>(acc_o);
  using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
  auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  // Warning: `tiled_mma` is Int8 (M, N) = (16, 32). `acc_o` is from PV (M, N) = (16, 16)?
  // Yes, standard Flash uses different atom for P@V vs Q@K sometimes.
  // Here, `tiled_mma` (Q@K) has N=32. `tiled_mma_pv` (P@V) has N=16.
  // acc_o depends on `tiled_mma_pv`.
  // So we must use `tiled_mma_pv` for creating Output copier!
  typename Kernel_traits::TiledMma_PV tiled_mma_pv_final;  // Needed again (optimize: hoist)

  auto smem_tiled_copy_O_pv = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma_pv_final);
  auto smem_thr_copy_O = smem_tiled_copy_O_pv.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
  auto taccOsO = smem_thr_copy_O.partition_D(sO);
  cute::copy(smem_tiled_copy_O_pv, taccOrO, taccOsO);

  __syncthreads();

  // Copy Smem -> Global
  if constexpr (!Split) {
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOsO)));
    copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOsO, tOgO,
        tOcO, tOpO,
        binfo.actual_seqlen_q - m_block * kBlockM);
  } else {
    // Split case
    const int total_elems = kBlockM * kHeadDim;
#pragma unroll
    for (int i = tidx; i < total_elems; i += Kernel_traits::kNThreads) {
      int r = i / kHeadDim;
      int c = i % kHeadDim;
      if (r < binfo.actual_seqlen_q - m_block * kBlockM) {
        gO(r, c) = static_cast<ElementAccum>(sO(r, c));
      }
    }
    const int lane = tidx % 32;
    const int warp = tidx / 32;
    const int local_row_0 = warp * 16 + (lane / 4);
    const int local_row_1 = local_row_0 + 8;
    const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                   Shape<Int<kBlockM>>{}, Stride<_1>{});
    if ((lane % 4) == 0) {
      if (local_row_0 < binfo.actual_seqlen_q - m_block * kBlockM) gLSEaccum(local_row_0) = lse(0);
      if (local_row_1 < binfo.actual_seqlen_q - m_block * kBlockM) gLSEaccum(local_row_1) = lse(1);
    }
  }
}

}  // namespace int8_mma

template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
__global__ void flash_fwd_int8_quant_kernel(const Params params) {
  const int m_block = blockIdx.x;
  int bidb, bidh, n_split_idx;
  if (Split) {
    n_split_idx = blockIdx.y;
    bidb = blockIdx.z / params.h;
    bidh = blockIdx.z % params.h;
  } else {
    n_split_idx = 0;
    bidb = blockIdx.y;
    bidh = blockIdx.z;
  }
  int8_mma::compute_attn_1rowblock<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, std::max(1, params.num_splits));
}
}  // namespace flash
}  // namespace onnxruntime

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
