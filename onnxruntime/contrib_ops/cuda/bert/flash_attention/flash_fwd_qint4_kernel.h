// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
/******************************************************************************
 * Flash Attention Kernel with INT4 Quantized KV Cache
 *
 * This kernel implements Flash Attention with on-the-fly dequantization of
 * INT4 quantized Key and Value tensors. INT4 provides 2x compression compared
 * to INT8, further reducing memory bandwidth requirements.
 *
 * Architecture Overview:
 * - Uses FP16/BF16 MMA (Matrix Multiply-Accumulate) Tensor Core operations
 * - INT4 K/V data packed as 2 nibbles per byte (hi/lo 4-bit values)
 * - Dequantization unpacks nibbles and converts to FP16/BF16 before MMA
 * - Supports both per-tensor and per-channel quantization scales
 *
 * Key Differences from INT8 Kernel:
 * - Physical storage: HeadDim/2 bytes per row (vs HeadDim for INT8)
 * - Uses "duplicate layout" (HeadDim/2, 2) with inner stride 0 for broadcast
 * - Scalar copy from global memory due to non-contiguous physical layout
 * - Optimized Unpacking in gemm_quant_manual:
 *   - Uses 'bfe.s32' inline assembly for single-cycle bit-extraction and sign-extension
 *   - Directly vectorizes to __half2/__nv_bfloat162 for high throughput
 * - INT4 values are 2's complement: range [-8, 7]
 *
 * Memory Layout:
 * - K/V stored as packed INT4 (2 elements per byte)
 * - Each byte contains: [hi_nibble:4][lo_nibble:4]
 * - Logical view (duplicate layout) maps physical bytes to 2 logical elements
 * - Shared memory: [Q (FP16)] [K (INT8/INT4, padded)] [V (INT8/INT4, padded)] ... [sP (FP16, overlap with K)]
 * - Padded layouts (16-byte padding) ensure 128-bit alignment and avoid bank conflicts.
 *
 * Performance Note:
 * - Scalar copy is slower than INT8's vectorized cp.async
 * - Trade-off: 2x memory compression vs lower copy bandwidth
 * - Net benefit depends on memory-bound vs compute-bound regime
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
#include "contrib_ops/cuda/bert/flash_attention/namespace_config.h"

namespace FLASH_NAMESPACE {
using namespace cute;
namespace int4 {

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_, typename elem_type = cutlass::half_t>
struct Flash_dq_kernel_traits {
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

  // FP16/BF16 MMA for all GEMM operations (Q×K^T and P×V)
  // INT4 K/V data is dequantized in registers before MMA, not using native INT8 Tensor Cores.
  // NOTE: We use N=8 tiling (matching MMA Atom's native N dimension) rather than N=16.
  // This is critical for the dequantization pipeline because:
  // 1. Int4 data is loaded into FP16-shaped register fragments using dummy tensors
  // 2. With N=8, each tile contains exactly one MMA atom, simplifying the mapping
  // 3. N=16 creates a 2-atom-per-tile structure that causes cute::gemm shape mismatches
  // The trade-off: gemm_quant_manual uses a manual loop to bypass cute::gemm's assertions.
  using MMA_Atom_Arch = std::conditional_t<
      std::is_same_v<elem_type, cutlass::half_t>,
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using TiledMma_PV = TiledMMA<
      MMA_Atom_Arch,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _8, _16>>;  // N=8 matches MMA Atom N dimension

  // Shared memory layouts
  using SmemLayoutAtom = decltype(composition(Swizzle<kSwizzle, 3, 3>{},
                                              Layout<Shape<_8, Int<kBlockKSmem>>,
                                                     Stride<Int<kBlockKSmem>, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{},
      Shape<Int<kBlockM>, Int<kHeadDim>>{}));

  // Padded layout for K/V to avoid bank conflicts and ensure alignment
  // For Int4, we store duplicated bytes, so we need kHeadDim bytes per row (same as Int8).
  static constexpr int kSmemRowPaddingInt4 = 16;
  static constexpr int kSmemRowStrideInt4 = kHeadDim + kSmemRowPaddingInt4;

  using SmemLayoutKInt8 = decltype(make_layout(
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      Stride<Int<kSmemRowStrideInt4>, _1>{}));

  using SmemLayoutV = SmemLayoutKInt8;

  // V transposed layout for P×V GEMM (swapped strides)
  using SmemLayoutVtransposed = decltype(make_layout(
      Shape<Int<kHeadDim>, Int<kBlockN>>{},
      Stride<_1, Int<kSmemRowStrideInt4>>{}));

  // Logical FP16 layout matching the physical shape (for dummy tensors)
  using SmemLayoutK = decltype(make_layout(
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      Stride<Int<kSmemRowStrideInt4>, _1>{}));

  // Copy atoms
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;  // For V transposition
  using SmemCopyAtomInt8 = Copy_Atom<SM75_U32x4_LDSM_N, ElementInt8>;

  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;

  using GmemLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;

  using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                                                    GmemLayoutAtom{},
                                                    Layout<Shape<_1, _8>>{}));

  // For Int8 global-to-shared memory copy:
  // - cp.async (SM80_CP_ASYNC_CACHEGLOBAL) requires 128-bit aligned contiguous memory
  // - Swizzle<3> breaks 128-bit contiguity, causing cp.async to fail validation
  // - Solution: Use UniversalCopy (standard load/store) with 64-bit vectorization
  // - This is slightly slower than cp.async but works correctly with swizzled layouts
  // - 64-bit (8 bytes) = 8 int8 elements, which is safe with Swizzle<3>
  using GmemTiledCopyKInt8 = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<cute::uint64_t>, ElementInt8>{},
                                                      GmemLayoutAtom{},
                                                      Layout<Shape<_1, _8>>{}));  // 8 int8s = 8 bytes

  // For Int4 duplicate layout (non-contiguous physical access), we use scalar copy for safety
  // Slow but correct. Optimization: Load packed then unpacking in registers.
  using GmemTiledCopyKInt4 = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, ElementInt8>{},
                                                      GmemLayoutAtom{},
                                                      Layout<Shape<_1, _1>>{}));

  // Output Smem Layout (Row-Major for Vectorized Global Copy)
  using SmemLayoutO = decltype(make_layout(Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                           Stride<Int<kHeadDim>, _1>{}));

  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
  using SmemTiledCopyO = decltype(make_tiled_copy_C(SmemCopyAtomO{}, TiledMma_PV{}));

  using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, cute::uint128_t>{},
                                                  GmemLayoutAtom{},
                                                  Layout<Shape<_1, _8>>{}));
};

template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
          bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV,
          int QUANT_TYPE, typename Params>
inline __device__ void compute_attn_1rowblock(
    const Params& params, const int bidb, const int bidh, const int m_block,
    const int n_split_idx, const int num_n_splits) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using ElementInt8 = typename Kernel_traits::ElementInt8;
  using ElementInt32 = typename Kernel_traits::ElementInt32;
  using index_t = typename Kernel_traits::index_t;

  const int tidx = threadIdx.x;
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  // Use ElementO for output: accumulates if Split is true
  using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

  extern __shared__ char smem_[];

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  // Int8/Int4 Shared Memory Buffers
  // Note: Padding calc logic from Int8
  constexpr int kSmemSizeQ = kBlockM * kHeadDim * sizeof(Element);
  constexpr int kSmemSizeKInt4 = kBlockN * Kernel_traits::kSmemRowStrideInt4 * sizeof(ElementInt8);

  Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(smem_ + kSmemSizeQ)),
                          typename Kernel_traits::SmemLayoutKInt8{});
  Tensor sV = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(smem_ + kSmemSizeQ + kSmemSizeKInt4)),
                          typename Kernel_traits::SmemLayoutV{});

  // Zero-initialize sK and sV physical memory once to prevent garbage in padding
  const int total_kv_bytes = 2 * kBlockN * Kernel_traits::kSmemRowStrideInt4;
  for (int i = tidx; i < total_kv_bytes; i += Kernel_traits::kNThreads) {
    reinterpret_cast<ElementInt8*>(smem_ + kSmemSizeQ)[i] = 0;
  }
  __syncthreads();

  const BlockInfo<!Is_even_MN> binfo(params, bidb);

  if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

  // Compute n_block range for Split-K
  const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
  const int n_block_min = !Is_local
                              ? n_split_idx * n_blocks_per_split
                              : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
  int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
  if (Is_causal || Is_local) {
    n_block_max = std::min(n_block_max,
                           cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
  }

  // Early Exit / Initialization for Empty Split Blocks
  if (n_block_min >= n_block_max) {
    // We exit early and write 0 to gOaccum and -inf to gLSEaccum (if Split).
    if constexpr (Split) {
      const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_rounded;
      const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;

      // Write -inf to LSEaccum
      ElementAccum* gLSEaccum_ptr = reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lseaccum;
      for (int i = tidx; i < kBlockM; i += Kernel_traits::kNThreads) {
        if (i < binfo.actual_seqlen_q - m_block * kBlockM) {
          gLSEaccum_ptr[i] = -INFINITY;
        }
      }

      // Write 0 to Oaccum
      ElementAccum* gOaccum_ptr = reinterpret_cast<ElementAccum*>(params.oaccum_ptr) + row_offset_oaccum;
      int total_elems = kBlockM * kHeadDim;
      for (int i = tidx; i < total_elems; i += Kernel_traits::kNThreads) {
        int r = i / kHeadDim;
        if (r < binfo.actual_seqlen_q - m_block * kBlockM) {
          gOaccum_ptr[i] = 0.0f;
        }
      }
    }
    return;
  }

  // Duplicate Layout Logic for Int4:
  // We want to load 128 logical columns (HeadDim) which physically map to 64 bytes (HeadDim/2).
  // A "duplicate layout" (HeadDim/2, 2) with inner stride 0 allows reading the same byte twice for pairs of columns.
  // mK/mV moved to after offsets calculation

  // Scales in Smem (for Per-Channel quantization)
  // Use float type to preserve precision (avoid fp16/bf16 round-trip loss)
  // Scale Tensors in Shared Memory
  // Place sKScale AFTER sV_int8 with PADDING to account for Swizzled Layout gaps
  // Offset = sQ + sK + sV + Padding
  using ElementScale = float;
  Tensor sKScale = make_tensor(make_smem_ptr(reinterpret_cast<ElementScale*>(smem_ + kSmemSizeQ + 2 * kSmemSizeKInt4 + 1024)),
                               make_layout(Shape<Int<kHeadDim>>{}, Stride<_1>{}));

  // Place sVScale after sKScale (accounting for float size)
  Tensor sVScale = make_tensor(make_smem_ptr(reinterpret_cast<ElementScale*>(smem_ + kSmemSizeQ + 2 * kSmemSizeKInt4 + 1024 + kHeadDim * sizeof(ElementScale))),
                               make_layout(Shape<Int<kHeadDim>>{}, Stride<_1>{}));

  // Define sO early (reusing Q memory) for Output Staging (FP16)
  Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutO{});

  // Define sP buffer (Reuse sK memory)
  // sK size is kBlockN * 144 (approx 64 * 144 = 9216 bytes).
  // sP needs kBlockM * kBlockN * 2 (approx 64 * 64 * 2 = 8192 bytes).
  // Fits safely.
  auto sP = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + kSmemSizeQ)),
                        make_layout(Shape<Int<kBlockM>, Int<kBlockN>>{}, Stride<Int<kBlockN>, _1>{}));

  // ============================================================================
  // Global Memory Indices & Offsets using BlockInfo
  // ============================================================================
  const int kv_head_idx = bidh / params.h_h_k_ratio;

  // K/V base offsets (elements)
  const index_t k_base_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                kv_head_idx * params.k_head_stride;
  const index_t v_base_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                kv_head_idx * params.v_head_stride;

  // Get scalar scales (for Per-Tensor or Per-Channel as scalar)
  const int scale_idx = (QUANT_TYPE == 2) ? kv_head_idx : 0;
  const float k_scale = params.k_scale_ptr ? reinterpret_cast<const float*>(params.k_scale_ptr)[scale_idx] : 1.0f;
  const float v_scale = params.v_scale_ptr ? reinterpret_cast<const float*>(params.v_scale_ptr)[scale_idx] : 1.0f;

  // Global Scale Tensors (for Per-Channel)
  auto gKScale = make_tensor(make_gmem_ptr(reinterpret_cast<const float*>(params.k_scale_ptr) + kv_head_idx * params.d),
                             make_layout(Shape<Int<kHeadDim>>{}, Stride<_1>{}));
  auto gVScale = make_tensor(make_gmem_ptr(reinterpret_cast<const float*>(params.v_scale_ptr) + kv_head_idx * params.d),
                             make_layout(Shape<Int<kHeadDim>>{}, Stride<_1>{}));

  // ============================================================================
  // Global K/V Tensors (Unified Int8/Int4)
  // ============================================================================
  // These are defined as (SeqLen, Dim) for the specific head.
  // For Int4, Dim is hierarchical (Dim/2, 2).
  // Shape: (SeqLen, (HeadDim/2, 2))
  // Stride: (RowStride, (1, 0))
  // Pointer: base + offset
  auto layout_dup = make_layout(
      make_shape(binfo.actual_seqlen_k, make_shape(Int<kHeadDim / 2>{}, Int<2>{})),
      make_stride(params.k_row_stride, make_stride(_1{}, _0{})));
  Tensor mK_int8 = make_tensor(make_gmem_ptr(reinterpret_cast<const ElementInt8*>(params.k_ptr) + k_base_offset),
                               layout_dup);

  auto v_layout_dup = make_layout(
      make_shape(binfo.actual_seqlen_k, make_shape(Int<kHeadDim / 2>{}, Int<2>{})),
      make_stride(params.v_row_stride, make_stride(_1{}, _0{})));
  Tensor mV_int8 = make_tensor(make_gmem_ptr(reinterpret_cast<const ElementInt8*>(params.v_ptr) + v_base_offset),
                               v_layout_dup);

  // ============================================================================
  // Global Memory Tensors
  // ============================================================================
  Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(params.q_ptr) +
                                        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                          make_shape(binfo.actual_seqlen_q, params.h, params.d),
                          make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

  // Output Tensors (Split or Standard)
  Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(Split ? params.oaccum_ptr : params.o_ptr) +
                                        (Split
                                             ? (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM) * params.d_rounded
                                             : binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb))),
                          make_shape(binfo.actual_seqlen_q, params.h, params.d),
                          make_stride(Split ? kHeadDim : params.o_row_stride, Split ? _0{} : params.o_head_stride, _1{}));
  Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

  // ============================================================================
  // Copy Setup
  // ============================================================================
  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

  Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
  }

  // Tiled Copy for K/V INT8 loading
  typename Kernel_traits::GmemTiledCopyKInt8 gmem_tiled_copy_KInt8;
  auto gmem_thr_copy_KInt8 = gmem_tiled_copy_KInt8.get_thread_slice(tidx);

  // ============================================================================
  // MMA Setup
  // ============================================================================
  typename Kernel_traits::TiledMma_PV tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);

  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);

  // Register fragments for K and V (FP16 structure for MMA)
  // Register fragments for K and V (FP16 structure for MMA)
  auto sK_dummy = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + kSmemSizeQ)),
                              typename Kernel_traits::SmemLayoutK{});
  Tensor tSrK = thr_mma.partition_fragment_B(sK_dummy);

  // Scale Partitioning (Broadcast along N)
  // sKScale is (HeadDim). Treat as (BlockN, HeadDim) with stride (0, 1).
  auto sKScale_broadcast = make_tensor(sKScale.data(), make_layout(Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<_0, _1>{}));
  auto sVScale_broadcast = make_tensor(sVScale.data(), make_layout(Shape<Int<kBlockN>, Int<kHeadDim>>{}, Stride<_0, _1>{}));

  auto tSrKScale = thr_mma.partition_fragment_B(sKScale_broadcast);

  // For V, P@V. V matches K layout in structure (BlockN, HeadDim) via sV.
  // For V, P@V. V matches K layout in structure (BlockN, HeadDim) via sV.
  auto sV_dummy = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + kSmemSizeQ + kSmemSizeKInt4)),
                              make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}));
  auto tOrVt = thr_mma.partition_fragment_B(sV_dummy);

  // We also partition V-Scale using same pattern
  auto tSrVScale = thr_mma.partition_fragment_B(sVScale_broadcast);

  // Smem-to-Register copy for Int8 K/V data
  using SmemCopyAtomInt8Default = Copy_Atom<DefaultCopy, typename Kernel_traits::ElementInt8>;
  auto smem_tiled_copy_KInt8 = make_tiled_copy_B(SmemCopyAtomInt8Default{}, tiled_mma);
  auto smem_thr_copy_KInt8 = smem_tiled_copy_KInt8.get_thread_slice(tidx);

  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(acc_o);

  constexpr int kNRows = 2 * decltype(size<1>(acc_o))::value;
  flash::Softmax<kNRows> softmax;

  // ============================================================================
  // Load Q
  // ============================================================================
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  // ============================================================================
  // Load Scales (if Per-Channel)
  // ============================================================================
  if constexpr (QUANT_TYPE == 2) {
    // Simple copy using threads
    int num_elems = kHeadDim;
    for (int i = tidx; i < num_elems; i += Kernel_traits::kNThreads) {
      if (i < params.d) {
        sKScale(i) = gKScale(i);  // Store as float directly, no precision loss
        sVScale(i) = gVScale(i);
      }
    }
    __syncthreads();

    // Load scales from smem broadcast to register fragments
    cute::copy(sKScale_broadcast, tSrKScale);
    cute::copy(sVScale_broadcast, tSrVScale);
  }

  // ============================================================================
  // Smem TiledCopiers
  // ============================================================================
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  // K/V Int8 Copier
  // We use the physical Int8 layout so that partitioning correctly accounts for padding.
  auto sK_quant = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKInt8{});
  // Partition the physical Smem tensor for loading into registers
  // Explicitly match logical shape (HeadDim, BlockN) for B-operand logic if needed?
  // No, K is (BlockN, HeadDim).
  auto tSsK = smem_thr_copy_KInt8.partition_S(sK_quant);

  // V physical for partitioning (must be transposed shape (HeadDim, BlockN) for B operand)
  // Use explicitly transposed layout
  auto sV_quant_t = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  // Partition the physical Smem tensor for loading into registers
  auto tSsV = smem_thr_copy_KInt8.partition_S(sV_quant_t);

  // ============================================================================
  // Main Attention Loop
  // ============================================================================
  bool is_first_block = true;
  for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
    // Int4 Load with Duplicate Layout using Scalar Copy
    // We use GmemTiledCopyKInt4 (Scalar) to handle the non-contiguous nature of duplicate layout.
    typename Kernel_traits::GmemTiledCopyKInt4 gmem_tiled_copy_KInt4;
    auto gmem_thr_copy_KInt4 = gmem_tiled_copy_KInt4.get_thread_slice(tidx);

    // Slice Global Tensor (Hierarchical)
    // mK_int8 is (SeqLen, H, (Dim/2, 2)), but here we view as (SeqLen, (Dim/2, 2)) [head handling implicit in ptr]
    // Actually mK_int8 defined above is (SeqLen, (Dim/2, 2)).
    // local_tile slices first dim.
    Tensor gK_int4 = local_tile(mK_int8, Shape<Int<kBlockN>, Shape<Int<kHeadDim / 2>, Int<2>>>{}, make_coord(n_block, _0{}));
    Tensor gV_int4 = local_tile(mV_int8, Shape<Int<kBlockN>, Shape<Int<kHeadDim / 2>, Int<2>>>{}, make_coord(n_block, _0{}));

    Tensor tKgK_int4 = gmem_thr_copy_KInt4.partition_S(gK_int4);
    Tensor tKsK_int4 = gmem_thr_copy_KInt4.partition_D(sK);
    Tensor tVgV_int4 = gmem_thr_copy_KInt4.partition_S(gV_int4);
    Tensor tVsV_int4 = gmem_thr_copy_KInt4.partition_D(sV);

    // Predicates
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKcK = gmem_thr_copy_KInt4.partition_D(cK);

    // For scalar copy, tKpK might be rank 1 or 2 depending on partitioning.
    // We construct it safely.
    auto tKpK = [&] {
      if constexpr (!Is_even_K) {
        constexpr int R = cute::rank_v<decltype(tKsK_int4)>;
        if constexpr (R >= 3) {
          auto t = make_tensor<bool>(make_shape(size<2>(tKsK_int4)));
#pragma unroll
          for (int k = 0; k < size(t); ++k) t(k) = get<1>(tKcK(0, 0, k)) < params.d;
          return t;
        } else {
          auto t = make_tensor<bool>(make_shape(_1{}));
          t(0) = get<1>(tKcK(0, 0)) < params.d;
          return t;
        }
      } else {
        return make_tensor<bool>(make_shape(_1{}));
      }
    }();

    // Use flash::copy_w_min_idx which is more generic or standard copy
    // Given the custom layout, we might just use cute::copy if Is_even_K is true
    // But checking predicates safely:
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KInt4, tKgK_int4, tKsK_int4, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KInt4, tVgV_int4, tVsV_int4, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN);

    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);

    if constexpr (QUANT_TYPE == 2) {
      ::onnxruntime::flash::gemm_quant_manual<true>(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_KInt8, smem_thr_copy_KInt8, tSrKScale);
    } else {
      ::onnxruntime::flash::gemm_quant_manual<true>(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_KInt8, smem_thr_copy_KInt8, k_scale);
    }

    // Masking
    constexpr int kNWarps = Kernel_traits::kNWarps;
    const int col_idx_offset = n_block * kBlockN;
    const int row_idx_offset = m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4;
    const int warp_row_stride = kNWarps * 16;
    flash::Mask<Is_causal, Is_local, /*Has_alibi=*/false> mask(
        binfo.actual_seqlen_k, binfo.actual_seqlen_q,
        params.window_size_left, params.window_size_right);
    mask.template apply_mask<Is_causal, Is_even_MN>(
        acc_s, col_idx_offset, row_idx_offset, warp_row_stride);

    // Softcap
    if constexpr (Is_softcap) {
      flash::apply_softcap(acc_s, params.softcap);
    }

    // Softmax Rescale
    if (is_first_block) {
      softmax.template softmax_rescale_o</*Is_first=*/true>(acc_s, acc_o, params.scale_softmax_log2);
      is_first_block = false;
    } else {
      softmax.template softmax_rescale_o</*Is_first=*/false>(acc_s, acc_o, params.scale_softmax_log2);
    }

    // 3. P @ V (Using sP shared memory to avoid register layout issues)
    // Convert acc_s (Float) to sP (Half/Element)
    // Map acc_s fragments to sP indices
    // Each thread in quad handles 2 rows and 2*MMA_N columns
    auto acc_s_rowcol = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
    const int lane_id = tidx % 32;
    const int warp_id = tidx / 32;

#pragma unroll
    for (int mi = 0; mi < size<0, 1>(acc_s_rowcol); ++mi) {
      int r_base = warp_id * 16 + mi * 0 + (lane_id / 4);  // warp covers 16 rows, MMA_M=1
#pragma unroll
      for (int i = 0; i < size<0, 0>(acc_s_rowcol); ++i) {
        int r = r_base + i * 8;
#pragma unroll
        for (int nj = 0; nj < size<1, 1>(acc_s_rowcol); ++nj) {
          int c_base = (lane_id % 4) * 2 + nj * 8;
#pragma unroll
          for (int j = 0; j < size<1, 0>(acc_s_rowcol); ++j) {
            int c = c_base + j;
            if (r < kBlockM && c < kBlockN) {
              sP(r, c) = static_cast<Element>(acc_s_rowcol(make_coord(i, mi), make_coord(j, nj)));
            }
          }
        }
      }
    }
    __syncthreads();

    // Reload P from Smem as operand A for PV GEMM
    auto smem_tiled_copy_P = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(tidx);
    auto tOrP = tiled_mma.get_thread_slice(tidx).partition_fragment_A(sP);
    auto tSrP = smem_thr_copy_P.retile_D(tOrP);
    auto tSsP = smem_thr_copy_P.partition_S(sP);
    cute::copy(smem_tiled_copy_P, tSsP, tSrP);

    if constexpr (QUANT_TYPE == 2) {
      ::onnxruntime::flash::gemm_quant_manual<true>(acc_o, tSrP, tOrVt, tSsV, tiled_mma, smem_tiled_copy_KInt8, smem_thr_copy_KInt8, tSrVScale);
    } else {
      ::onnxruntime::flash::gemm_quant_manual<true>(acc_o, tSrP, tOrVt, tSsV, tiled_mma, smem_tiled_copy_KInt8, smem_thr_copy_KInt8, v_scale);
    }

    __syncthreads();
  }  // End n_block loop

  // ============================================================================
  // Finalize
  // ============================================================================
  // Normalize acc_o (full N=128)
  float sink = (params.head_sink_ptr != nullptr)
                   ? static_cast<float>(reinterpret_cast<const Element*>(params.head_sink_ptr)[bidh])
                   : (params.smooth_softmax ? 0.0f : -flash::kInfinity);
  Tensor lse = softmax.template normalize_softmax_lse<Split>(acc_o, params.scale_softmax, sink);

  // Store Reg -> Smem (use same approach as old kernel with retile_S)
  Tensor rO = flash::convert_type<Element>(acc_o);

  using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
  auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);   // ((Atom,AtomNum), MMA_M, MMA_N)
  auto taccOsO = smem_thr_copy_O.partition_D(sO);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
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

    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOsO, tOgO,
        tOcO, tOpO,
        binfo.actual_seqlen_q - m_block * kBlockM);
  } else {
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
      if (local_row_0 < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSEaccum(local_row_0) = lse(0);
      }
      if (local_row_1 < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSEaccum(local_row_1) = lse(1);
      }
    }
  }
}

}  // namespace int4
}  // namespace FLASH_NAMESPACE

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
