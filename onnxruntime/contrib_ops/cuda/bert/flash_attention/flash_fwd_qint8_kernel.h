// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
/******************************************************************************
 * Flash Attention Kernel with INT8 Quantized KV Cache
 *
 * This kernel implements Flash Attention with on-the-fly dequantization of
 * INT8 quantized Key and Value tensors. It is optimized for inference with
 * quantized KV-cache to reduce memory bandwidth and storage requirements.
 *
 * Architecture Overview:
 * - Uses FP16/BF16 MMA (Matrix Multiply-Accumulate) Tensor Core operations
 * - INT8 K/V data is loaded from global memory using cp.async (128-bit)
 * - Dequantization from INT8 to FP16/BF16 happens in registers before MMA
 *   - Uses vectorized __half2/__nv_bfloat162 instructions for high throughput
 * - Supports both per-tensor and per-channel quantization scales
 *
 * Key Features:
 * - 128-bit asynchronous global memory loads (cp.async) for K/V
 *   - Padded shared memory layout for bank conflict avoidance (16-byte padding)
 *   - Explicit sP shared memory buffer to avoid register layout issues in 2nd GEMM
 *   - Compatible with SM80+ (Ampere and newer GPUs)
 *   - Supports causal masking, local attention, and softcap
 *
 * Memory Layout:
 * - K/V stored as INT8 (1 byte per element)
 * - Row-major with padding for 128-bit alignment and bank conflict avoidance
 * - Shared memory: [Q (FP16)] [K (INT8, padded)] [V (INT8, padded)] ... [sP (FP16, overlap with K)]
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

#include "contrib_ops/cuda/bert/flash_attention/flash.h"
#include "contrib_ops/cuda/bert/flash_attention/block_info.h"
#include "contrib_ops/cuda/bert/flash_attention/kernel_traits.h"
#include "contrib_ops/cuda/bert/flash_attention/utils.h"
#include "contrib_ops/cuda/bert/flash_attention/softmax.h"
#include "contrib_ops/cuda/bert/flash_attention/mask.h"
#include "contrib_ops/cuda/bert/flash_attention/namespace_config.h"

namespace FLASH_NAMESPACE {

namespace int8 {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel Traits for INT8 Quantized KV Cache with FP16/BF16 MMA Dequantization
////////////////////////////////////////////////////////////////////////////////////////////////////

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

  // Double-buffering (pipelining) configuration
  // kNumStages=2: Use two K/V buffers to overlap memory loads with compute
  static constexpr int kNumStages = 2;

  static_assert(kHeadDim % 32 == 0, "kHeadDim must be multiple of 32");

  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

  // FP16/BF16 MMA for all GEMM operations (Q×K^T and P×V)
  // INT8 K/V data is dequantized in registers before MMA, not using native INT8 Tensor Cores.
  // NOTE: We use N=8 tiling (matching MMA Atom's native N dimension) rather than N=16.
  // This is critical for the dequantization pipeline because:
  // 1. Int8 data is loaded into FP16-shaped register fragments using dummy tensors
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

  static constexpr int kNThreadsPerRow = 4;  // Each row of 16x8x16 atom is handled by 4 threads (for N=8 tile)

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

  // V transposed layout for P×V GEMM (same as old kernel)
  using SmemLayoutVtransposed = decltype(composition(SmemLayoutV{}, make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
  using SmemLayoutVtransposedNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtransposed{}));

  // INT8 layout for K/V: Padded row-major (no swizzle)
  // =====================================================
  // Bank conflict avoidance strategy:
  // - Row-major with kHeadDim=128 has all rows starting at bank 0 (128 % 128 = 0)
  // - This causes 32-way bank conflicts when threads access the same column
  // - Solution: Add 16-byte padding per row -> stride = 144
  // - Bank(row r, col c) = (r * 144 + c) / 4 % 32 = (r * 36 + c/4) % 32
  // - Now consecutive rows map to different banks: row 0->bank 0, row 1->bank 4, etc.
  // - 16-byte padding ensures each row is 128-bit aligned for cp.async!
  static constexpr int kSmemRowPaddingInt8 = 16;                             // 16 bytes for 128-bit alignment
  static constexpr int kSmemRowStrideInt8 = kHeadDim + kSmemRowPaddingInt8;  // 144 for HeadDim=128

  using SmemLayoutKInt8 = decltype(make_layout(
      Shape<Int<kBlockN>, Int<kHeadDim>>{},
      Stride<Int<kSmemRowStrideInt8>, _1>{}));  // Padded row-major, 16-byte aligned rows

  // V transposed layout for P×V GEMM
  using SmemLayoutVInt8transposed = decltype(make_layout(
      Shape<Int<kHeadDim>, Int<kBlockN>>{},
      Stride<_1, Int<kSmemRowStrideInt8>>{}));

  // Copy atoms
  using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, elem_type>;
  using SmemCopyAtomTransposed = Copy_Atom<SM75_U16x8_LDSM_T, elem_type>;  // For V transposition
  using SmemCopyAtomInt8 = Copy_Atom<SM75_U32x4_LDSM_N, ElementInt8>;
  using SmemCopyAtomInt8Default = Copy_Atom<cute::DefaultCopy, ElementInt8>;

  static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
  static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;

  using GmemLayoutAtom = Layout<Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                Stride<Int<kGmemThreadsPerRow>, _1>>;

  using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
  using GmemTiledCopyQKV = decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                                                    GmemLayoutAtom{},
                                                    Layout<Shape<_1, _8>>{}));

  // For Int8 global-to-shared memory copy:
  // - Use 128-bit cp.async (16 bytes = 16 int8 elements per instruction)
  // - The padded row-major layout preserves 128-bit column contiguity
  // - cp.async provides latency hiding by overlapping memory access with compute
  using GmemTiledCopyKInt8 = decltype(make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, ElementInt8>{},
                                                      GmemLayoutAtom{},
                                                      Layout<Shape<_1, _16>>{}));  // 16 int8s = 16 bytes

  // Output Smem Layout (Row-Major for Vectorized Global Copy)
  using SmemLayoutO = decltype(make_layout(Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                           Stride<Int<kHeadDim>, _1>{}));

  using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
  using SmemTiledCopyO = decltype(make_tiled_copy_C(SmemCopyAtomO{}, TiledMma_PV{}));

  using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, cute::uint128_t>{},
                                                  GmemLayoutAtom{},
                                                  Layout<Shape<_1, _8>>{}));  // For Half

  using GmemTiledCopyOaccum = decltype(make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, cute::uint128_t>{},
                                                       GmemLayoutAtom{},
                                                       Layout<Shape<_1, _4>>{}));  // For Float
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Online Q Quantization Helper
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Tensor>
__forceinline__ __device__ float compute_absmax(Tensor const& tensor) {
  float absmax = 0.0f;
#pragma unroll
  for (int i = 0; i < size(tensor); ++i) {
    float val = static_cast<float>(tensor(i));
    absmax = fmaxf(absmax, fabsf(val));
  }
  // Reduce across warp
  // Note: In full implementation, need warp shuffle reduction
  return absmax;
}

template <typename TensorSrc, typename TensorDst>
__forceinline__ __device__ void quantize_fp16_to_int8(
    TensorSrc const& src, TensorDst& dst, float scale_inv) {
#pragma unroll
  for (int i = 0; i < size(src); ++i) {
    float val = static_cast<float>(src(i)) * scale_inv;
    // Clamp to INT8 range
    val = fmaxf(-127.0f, fminf(127.0f, roundf(val)));
    dst(i) = static_cast<int8_t>(val);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Unused Legacy Helpers (kept for potential future native INT8 MMA implementation)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename TensorInt32, typename TensorFP>
__forceinline__ __device__ void dequant_scores(
    TensorInt32 const& src_int32, TensorFP& dst_fp,
    float q_scale, float k_scale) {
  const float combined_scale = q_scale * k_scale;
#pragma unroll
  for (int i = 0; i < size(src_int32); ++i) {
    dst_fp(i) = static_cast<typename TensorFP::value_type>(
        static_cast<float>(src_int32(i)) * combined_scale);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main Kernel: Flash Attention with INT8 Quantized KV Cache
// Uses FP16/BF16 MMA with on-the-fly dequantization (not native INT8 Tensor Cores)
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

  // Use ElementO for output: accumulates if Split is true
  using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

  extern __shared__ char smem_[];

  const int tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

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
      const index_t row_offset_oaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q * params.d_rounded;
      const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q;

      Tensor gOaccum_base = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(params.oaccum_ptr) + row_offset_oaccum),
                                        make_shape(params.seqlen_q, Int<kHeadDim>{}),
                                        make_stride(kHeadDim, _1{}));
      Tensor gOaccum = local_tile(gOaccum_base, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

      Tensor gLSEaccum_base = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                          make_shape(params.seqlen_q), Stride<_1>{});
      Tensor gLSEaccum = local_tile(gLSEaccum_base, Shape<Int<kBlockM>>{}, make_coord(m_block));

      // Use GmemTiledCopyOaccum if available, otherwise reuse GmemTiledCopyO but adapt for ElementO type?
      // Actually, we can just use simple writes since this is initialization.
      // Or reuse Kernel_traits definitions if suitable.
      // Let's use simple per-thread loop for initialization to avoid complexity with TiledCopy types for now.
      // Actually best is to use tiled copy.

      // Output Smem Layout (Row-Major for Vectorized Global Copy)
      // using SmemLayoutO = decltype(make_layout(Shape<Int<kBlockM>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{}));
      // We can reuse that for gOaccum initialization.

      // For efficiency, just use a basic loop.
      int total_elems = kBlockM * kHeadDim;
      for (int i = tidx; i < total_elems; i += Kernel_traits::kNThreads) {
        int row = i / kHeadDim;
        int col = i % kHeadDim;
        if (row < binfo.actual_seqlen_q - m_block * kBlockM) {
          // Bounds check
          if (col < params.d) {
            gOaccum(row, col) = 0.0f;
          }
        }
      }

      // Init LSE
      for (int i = tidx; i < kBlockM; i += Kernel_traits::kNThreads) {
        if (i < binfo.actual_seqlen_q - m_block * kBlockM) {
          gLSEaccum(i) = -std::numeric_limits<ElementAccum>::infinity();
        }
      }
    }
    return;
  }

  // Layout: [sQ (FP16)] [sK[0] (Int8, padded)] [sV[0] (Int8, padded)] [sK[1] (Int8, padded)] [sV[1] (Int8, padded)]
  // ============================================================================
  // Double-buffering: kNumStages=2 for overlapping memory loads with compute
  // Note: SmemLayoutKInt8 uses padded stride (kSmemRowStrideInt8 = 144 for HeadDim=128)
  // size(layout) returns logical size (64 × 128 = 8192), not physical size (64 × 144 = 9216)
  // We must use physical size for tensor placement to avoid overlap.
  constexpr int kSmemSizeQ = kBlockM * kHeadDim * sizeof(Element);
  constexpr int kSmemSizeKInt8 = kBlockN * Kernel_traits::kSmemRowStrideInt8 * sizeof(ElementInt8);
  constexpr int kSmemSizeVInt8 = kBlockN * Kernel_traits::kSmemRowStrideInt8 * sizeof(ElementInt8);
  constexpr int kSmemSizeKVInt8 = kSmemSizeKInt8 + kSmemSizeVInt8;  // One K+V pair
  constexpr int kNumStages = Kernel_traits::kNumStages;

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});

  // Double-buffered K/V tensors: explicit declarations for each stage
  // Stage 0: smem_ + kSmemSizeQ
  // Stage 1: smem_ + kSmemSizeQ + kSmemSizeKVInt8
  char* stage0_base = smem_ + kSmemSizeQ;
  char* stage1_base = smem_ + kSmemSizeQ + kSmemSizeKVInt8;

  auto sK_0 = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(stage0_base)),
                          typename Kernel_traits::SmemLayoutKInt8{});
  auto sV_0 = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(stage0_base + kSmemSizeKInt8)),
                          typename Kernel_traits::SmemLayoutKInt8{});
  auto sK_1 = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(stage1_base)),
                          typename Kernel_traits::SmemLayoutKInt8{});
  auto sV_1 = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(stage1_base + kSmemSizeKInt8)),
                          typename Kernel_traits::SmemLayoutKInt8{});

  // Zero-initialize all K/V buffers to prevent garbage in padding
  // (16-byte padding per row is used for 128-bit aligned cp.async and bank conflict avoidance)
  const int total_kv_bytes = kNumStages * kSmemSizeKVInt8;
  for (int i = tidx; i < total_kv_bytes / 16; i += Kernel_traits::kNThreads) {
    reinterpret_cast<uint4*>(smem_ + kSmemSizeQ)[i] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  // Output Smem Layout (Row-Major for Vectorized Global Copy)
  Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(smem_)),
                          typename Kernel_traits::SmemLayoutO{});

  // Compute KV head index (for GQA)
  const int kv_head_idx = bidh / params.h_h_k_ratio;

  // Get scales
  // For PER_TENSOR (1): use index 0. For PER_CHANNEL (2): use head scale.
  // Note: GQA typically uses one scale per KV head.
  const int scale_idx = (params.k_quant_type == 2) ? kv_head_idx : 0;
  const float k_scale = params.k_scale_ptr ? static_cast<float>(reinterpret_cast<const Element*>(params.k_scale_ptr)[scale_idx]) : 1.0f;
  const float v_scale = params.v_scale_ptr ? static_cast<float>(reinterpret_cast<const Element*>(params.v_scale_ptr)[scale_idx]) : 1.0f;

  // Global Memory Tensors
  Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(params.q_ptr) +
                                        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                          make_shape(binfo.actual_seqlen_q, params.h, params.d),
                          make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

  // O and LSE accumulators
  const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb);
  const index_t row_offset_oaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q * params.d_rounded;
  const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q;

  Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
                          make_shape(binfo.actual_seqlen_q, Split ? Int<1>{} : params.h, params.d),
                          make_stride(Split ? Int<kHeadDim>{} : params.o_row_stride, Split ? Int<0>{} : params.o_head_stride, _1{}));
  Tensor gO = local_tile(mO(_, Split ? _0{} : bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

  Tensor gLSEaccum_base = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
                                      make_shape(binfo.actual_seqlen_q), Stride<_1>{});
  Tensor gLSEaccum = local_tile(gLSEaccum_base, Shape<Int<kBlockM>>{}, make_coord(m_block));

  // K is INT8 in global memory
  const index_t k_base_offset = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) +
                                kv_head_idx * params.k_head_stride;
  Tensor mK_int8 = make_tensor(make_gmem_ptr(reinterpret_cast<const ElementInt8*>(params.k_ptr) + k_base_offset),
                               make_shape(binfo.actual_seqlen_k, params.d),
                               make_stride(params.k_row_stride, _1{}));

  // V is INT8 in global memory
  const index_t v_base_offset = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) +
                                kv_head_idx * params.v_head_stride;
  Tensor mV_int8 = make_tensor(make_gmem_ptr(reinterpret_cast<const ElementInt8*>(params.v_ptr) + v_base_offset),
                               make_shape(binfo.actual_seqlen_k, params.d),
                               make_stride(params.v_row_stride, _1{}));

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
  // tSrK must match the logical B-operand shape for QDQ: (HeadDim, BlockN) i.e. K^T
  // We use dummy layout matching SmemLayoutK (BlockN, HeadDim) but pointing to the REAL data.
  // For double-buffering, we create the dummy tensor pointing to stage 0's K buffer.
  auto sK_dummy = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(sK_0.data().get())),
                              typename Kernel_traits::SmemLayoutK{});
  auto tSrK = thr_mma.partition_fragment_B(sK_dummy);

  // Smem-to-Register copy for Int8 K/V data:
  // - We use DefaultCopy (scalar/vector) instead of LDSM (SM75_U32x4_LDSM_N)
  // - LDSM loads 4 elements per instruction, but FP16 MMA B-fragment expects different sizing
  // - DefaultCopy is more flexible and handles the Int8->FP16 layout mapping correctly
  // - The data is dequantized in registers after loading (in gemm_quant_manual)
  using SmemCopyAtomInt8Default = typename Kernel_traits::SmemCopyAtomInt8Default;
  auto smem_tiled_copy_KInt8 = make_tiled_copy_B(SmemCopyAtomInt8Default{}, tiled_mma);
  auto smem_thr_copy_KInt8 = smem_tiled_copy_KInt8.get_thread_slice(tidx);

  // V dummy for partitioning (must be transposed shape (HeadDim, BlockN) for B operand)
  // For double-buffering, we point to stage 0's V buffer.
  auto sV_dummy = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(sV_0.data().get())),
                              make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}));
  auto tOrVt = thr_mma.partition_fragment_B(sV_dummy);

  auto acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(acc_o);

  constexpr int kNRows = 2 * decltype(size<1>(acc_o))::value;
  flash::Softmax<kNRows, Kernel_traits::kNThreadsPerRow> softmax;

  // ============================================================================
  // Load Q
  // ============================================================================
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  // ============================================================================
  // Smem TiledCopiers
  // ============================================================================
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  // K/V Int8 Copier
  // Note: we redefined smem_tiled_copy_KInt8 above using DefaultCopy
  // For P@V: P(64, 128) @ V(128, 64)? No.
  // P is (BlockM, BlockN) (64, 64)? No.
  // P comes from acc_s.
  // P dot V.
  // Dimensions match?
  // P (M, K_gemm). V (K_gemm, N_gemm).
  // M=64.
  // K_gemm = BlockN (64).
  // N_gemm = HeadDim (128).
  // So V is (64, 128).
  // sV is (64, 128).
  // So sV is correct as is.
  // tSsV partitions sV. Correct.

  // ============================================================================
  // Main Attention Loop (Double-Buffered for Pipelining)
  // ============================================================================
  // Double-buffering strategy:
  // - Use kNumStages=2 K/V buffers to overlap memory loads with compute
  // - Pre-load first tile before entering main loop
  // - In each iteration: issue async load for next tile, then compute on current tile
  // - Use cp_async_wait<1> instead of cp_async_wait<0> to keep one load in flight

  // Prepare predicates for K/V copy (shared across all iterations)
  Tensor cK = make_identity_tensor(make_shape(size<0>(sK_0), size<1>(sK_0)));
  Tensor tKcK = gmem_thr_copy_KInt8.partition_D(cK);
  auto tKsK_int8_0 = gmem_thr_copy_KInt8.partition_D(sK_0);
  Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK_int8_0)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tKpK); ++k) tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d;
  }

  // Helper lambda to load K/V tile into specified stage buffer (stage 0 or 1)
  auto load_kv_tile_stage0 = [&](int n_block) {
    if (n_block < n_block_min) return;
    Tensor gK_int8 = local_tile(mK_int8, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));
    Tensor gV_int8 = local_tile(mV_int8, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));
    Tensor tKgK_int8 = gmem_thr_copy_KInt8.partition_S(gK_int8);
    Tensor tKsK_int8 = gmem_thr_copy_KInt8.partition_D(sK_0);
    Tensor tVgV_int8 = gmem_thr_copy_KInt8.partition_S(gV_int8);
    Tensor tVsV_int8 = gmem_thr_copy_KInt8.partition_D(sV_0);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KInt8, tKgK_int8, tKsK_int8, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KInt8, tVgV_int8, tVsV_int8, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN);
  };

  auto load_kv_tile_stage1 = [&](int n_block) {
    if (n_block < n_block_min) return;
    Tensor gK_int8 = local_tile(mK_int8, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));
    Tensor gV_int8 = local_tile(mV_int8, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));
    Tensor tKgK_int8 = gmem_thr_copy_KInt8.partition_S(gK_int8);
    Tensor tKsK_int8 = gmem_thr_copy_KInt8.partition_D(sK_1);
    Tensor tVgV_int8 = gmem_thr_copy_KInt8.partition_S(gV_int8);
    Tensor tVsV_int8 = gmem_thr_copy_KInt8.partition_D(sV_1);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KInt8, tKgK_int8, tKsK_int8, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KInt8, tVgV_int8, tVsV_int8, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN);
  };

  // Pre-load first tile (stage 0)
  int first_n_block = n_block_max - 1;
  if (first_n_block >= n_block_min) {
    load_kv_tile_stage0(first_n_block);
    cute::cp_async_fence();
  }

  bool is_first_block = true;
  for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
    // Determine current and next stage indices
    int cur_stage = (n_block_max - 1 - n_block) % kNumStages;
    int next_stage = (cur_stage + 1) % kNumStages;
    int next_n_block = n_block - 1;

    // Issue async load for NEXT tile into NEXT stage buffer (if there is a next tile)
    if (next_n_block >= n_block_min) {
      if (next_stage == 0) {
        load_kv_tile_stage0(next_n_block);
      } else {
        load_kv_tile_stage1(next_n_block);
      }
      cute::cp_async_fence();
    }

    // Wait for CURRENT tile to be ready (allow 1 async op in flight if there's a next tile)
    if (next_n_block >= n_block_min) {
      cute::cp_async_wait<1>();  // Wait for current tile, keep next tile loading
    } else {
      cute::cp_async_wait<0>();  // Last iteration, wait for all
    }
    __syncthreads();

    // Select current stage's K/V buffers
    auto& sK_cur = (cur_stage == 0) ? sK_0 : sK_1;
    auto& sV_cur = (cur_stage == 0) ? sV_0 : sV_1;

    // Create tSsK for current stage (K transposed for B operand)
    auto tSsK = smem_thr_copy_KInt8.partition_S(
        make_tensor(sK_cur.data(), make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{},
                                               Stride<_1, Int<Kernel_traits::kSmemRowStrideInt8>>{})));

    // Create tSsV for current stage (V transposed for B operand)
    auto tSsV = smem_thr_copy_KInt8.partition_S(
        make_tensor(sV_cur.data(), typename Kernel_traits::SmemLayoutVInt8transposed{}));

    // 2. Q @ K^T (Gemm Quant)
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);

    // Q @ K^T using gemm_quant_manual:
    // - Uses manual loop to bypass cute::gemm shape assertions
    // - Int8 K data is loaded into registers, dequantized to FP16, then used in MMA
    // - k_scale converts Int8 values back to original FP16 range
    flash::gemm_quant_manual<false>(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_KInt8, smem_thr_copy_KInt8, k_scale);

    // Masking
    constexpr int kNWarps_ = Kernel_traits::kNWarps;
    const int col_idx_offset = n_block * kBlockN;
    const int row_idx_offset = m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4;
    const int warp_row_stride = kNWarps_ * 16;
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
    // - sP reuses the current stage's K buffer memory (sK[cur_stage])
    // - This is safe because we finished reading K in the Q×K^T GEMM above
    // - sK is (BlockN, HeadDim + Pad) = (64, 144) -> ~9KB.
    // - sP needs (BlockM, BlockN) = (64, 64) elements (FP16) -> 8KB.
    // - Fits safely within sK's allocation.
    auto sP = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(sK_cur.data().get())),
                          make_layout(Shape<Int<kBlockM>, Int<kBlockN>>{}, Stride<Int<kBlockN>, _1>{}));

    // Convert acc_s (Float) to sP (Half/Element)
    auto acc_s_rowcol = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
    const int lane_id = tidx % 32;
    const int warp_id = tidx / 32;
    // Map acc_s fragments to sP indices
    // Each thread in quad handles 2 rows and 2*MMA_N columns
#pragma unroll
    for (int mi = 0; mi < size<0, 1>(acc_s_rowcol); ++mi) {
      int r_base = warp_id * 16 + (lane_id / 4);
#pragma unroll
      for (int i = 0; i < size<0, 0>(acc_s_rowcol); ++i) {
        int r = r_base + i * 8;
#pragma unroll
        for (int nj = 0; nj < size<1, 1>(acc_s_rowcol); ++nj) {
          int c_base = (lane_id % 4) * 2 + nj * 8;
          if (r < kBlockM && c_base < kBlockN) {
            __half2 vals_h2 = __halves2half2(
                reinterpret_cast<half const&>(acc_s_rowcol(make_coord(i, mi), make_coord(0, nj))),
                reinterpret_cast<half const&>(acc_s_rowcol(make_coord(i, mi), make_coord(1, nj))));
            reinterpret_cast<__half2&>(sP(r, c_base)) = vals_h2;
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

    // P @ V using gemm_quant_manual:
    flash::gemm_quant_manual<false>(acc_o, tSrP, tOrVt, tSsV, tiled_mma, smem_tiled_copy_KInt8, smem_thr_copy_KInt8, v_scale);

    __syncthreads();
  }  // End n_block loop

  // ============================================================================
  // Finalize
  // ============================================================================
  // Normalize acc_o (full N=128)
  // Handle sink parameter correctly for smooth softmax (same as old kernel)
  float sink = (params.head_sink_ptr != nullptr)
                   ? static_cast<float>(reinterpret_cast<const Element*>(params.head_sink_ptr)[bidh])
                   : (params.smooth_softmax ? 0.0f : -flash::kInfinity);
  Tensor lse = softmax.template normalize_softmax_lse<Split>(acc_o, params.scale_softmax, sink);

  // Store Reg -> Smem (use same approach as old kernel with retile_S)
  Tensor rO = flash::convert_type<ElementO>(acc_o);

  using SmemCopyAtomO = Copy_Atom<DefaultCopy, ElementO>;
  auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
  Tensor taccOrO = smem_thr_copy_O.retile_S(rO);   // ((Atom,AtomNum), MMA_M, MMA_N)
  auto taccOsO = smem_thr_copy_O.partition_D(sO);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

  cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
  __syncthreads();

  // Copy Smem -> Global
  if constexpr (!Split) {
    using GmemTiledCopyO = typename Kernel_traits::GmemTiledCopyO;
    GmemTiledCopyO gmem_tiled_copy_O;
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
    // Split case: Write sO (Element/Half) to gO (ElementAccum/Float).
    // We also need to write LSE.

    // 1. Write O accum (sO -> gO)
    // This is a manual copy from shared memory to global memory with type conversion.
    // sO is [kBlockM, kHeadDim] in shared memory.
    // gO is [kBlockM, kHeadDim] tile in global memory.
    // We iterate through all elements this thread group is responsible for,
    // but since it's just a raw copy loop, we can distribute work simply.

    const int total_elems = kBlockM * kHeadDim;
#pragma unroll
    for (int i = tidx; i < total_elems; i += Kernel_traits::kNThreads) {
      int r = i / kHeadDim;
      int c = i % kHeadDim;
      if (r < binfo.actual_seqlen_q - m_block * kBlockM) {
        gO(r, c) = static_cast<ElementAccum>(sO(r, c));
      }
    }

    // 2. Write LSE accum
    // Map LSE values (registers) to Global Rows manually for SM80_16x8x16 layout.
    // acc_o structure for 16x8x16:
    // Each warp handles 16 rows (MMA_M=1 for BlockM=64/4warps=16 rows per warp).
    // Lanes 0-3 handle Row 0 and Row 8. (Cols 0-7).
    // We rely on consistent layout for SM80.

    const int lane = tidx % 32;
    const int warp = tidx / 32;
    // Local row index within the block (0..kBlockM-1)
    const int local_row_0 = warp * 16 + (lane / 4);
    const int local_row_1 = local_row_0 + 8;

    // Only the first thread in the quad (Col 0) writes LSE
    if ((lane % 4) == 0) {
      // LSE[0] corresponds to local_row_0
      if (local_row_0 < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSEaccum(local_row_0) = lse(0);
      }
      // LSE[1] corresponds to local_row_1
      if (local_row_1 < binfo.actual_seqlen_q - m_block * kBlockM) {
        gLSEaccum(local_row_1) = lse(1);
      }
    }
  }
}

// =================================================================================================
// GQA-Aware Optimized Kernel (Q=1 specialization)
// - One Block per KV Head (Grid = b * h_k)
// - Collaborative K/V Load (Shared by all warps)
// - Multiple Q-heads packed into sQ (One-Shot GEMM)
// =================================================================================================
template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
          bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
inline __device__ void compute_attn_1rowblock_gqa(
    const Params& params, const int bidb, const int kv_head_idx, const int m_block,
    const int n_split_idx, const int num_n_splits) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using ElementInt8 = typename Kernel_traits::ElementInt8;
  using ElementInt32 = typename Kernel_traits::ElementInt32;
  using index_t = typename Kernel_traits::index_t;
  using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

  extern __shared__ char smem_[];
  const int tidx = threadIdx.x;
  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;
  constexpr int kNWarps = Kernel_traits::kNWarps;

  const BlockInfo<!Is_even_MN> binfo(params, bidb);

  // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) printf("GQA Kernel Running! M=%d N=%d Warps=%d Splits=%d\n", kBlockM, kBlockN, kNWarps, num_n_splits);

  // Group Info
  const int num_q_heads_per_kv = params.h_h_k_ratio;
  const int q_head_start = kv_head_idx * num_q_heads_per_kv;

  // Split-K Ranges
  const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
  const int n_block_min = n_split_idx * n_blocks_per_split;
  const int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);

  // Early Exit / Empty Handling
  if (n_block_min >= n_block_max) {
    if constexpr (Split) {
      for (int i = 0; i < num_q_heads_per_kv; ++i) {
        int bidh = q_head_start + i;
        const index_t row_offset_oaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q * params.d_rounded;
        const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q;
        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(params.oaccum_ptr) + row_offset_oaccum),
                                     make_shape(Int<kBlockM>{}, Int<kHeadDim>{}), make_stride(kHeadDim, _1{}));
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                       make_shape(Int<kBlockM>{}), Stride<_1>{});
        if (tidx < kHeadDim) gOaccum(0, tidx) = 0.0f;  // Only Row 0 matters for Q=1
        if (tidx == 0) gLSEaccum(0) = -std::numeric_limits<ElementAccum>::infinity();
      }
    }
    return;
  }

  // Smem Setup
  // Smem Setup
  constexpr int kSmemQRows = (kNWarps * 16 > kBlockM) ? (kNWarps * 16) : kBlockM;
  constexpr int kSmemSizeQ = kSmemQRows * kHeadDim * sizeof(Element);
  constexpr int kSmemSizeKInt8 = kBlockN * Kernel_traits::kSmemRowStrideInt8 * sizeof(ElementInt8);

  using SmemLayoutQ_GQA = decltype(tile_to_shape(typename Kernel_traits::SmemLayoutAtom{}, Shape<Int<kSmemQRows>, Int<kHeadDim>>{}));
  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)), SmemLayoutQ_GQA{});
  Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(smem_ + kSmemSizeQ)), typename Kernel_traits::SmemLayoutKInt8{});
  Tensor sV = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(smem_ + kSmemSizeQ + kSmemSizeKInt8)), typename Kernel_traits::SmemLayoutKInt8{});

  // Zero-init Int8 padding
  const int total_kv_bytes = 2 * kBlockN * Kernel_traits::kSmemRowStrideInt8;
  for (int i = tidx; i < total_kv_bytes / 16; i += Kernel_traits::kNThreads) {
    reinterpret_cast<uint4*>(smem_ + kSmemSizeQ)[i] = make_uint4(0, 0, 0, 0);
  }

  // Zero-init Q (safety against uninitialized memory causing non-determinism)
  // Especially for rows 4-15 which are not loaded but used in GEMM/Softmax.
  for (int i = tidx; i < kSmemSizeQ / 16; i += Kernel_traits::kNThreads) {
    reinterpret_cast<uint4*>(smem_)[i] = make_uint4(0, 0, 0, 0);
  }

  // Load ALL Q-heads in group to sQ (packed)
  for (int i = tidx; i < num_q_heads_per_kv * kHeadDim; i += Kernel_traits::kNThreads) {
    int h_in_grp = i / kHeadDim;
    int col = i % kHeadDim;
    int bidh = q_head_start + h_in_grp;
    // Map head to row: Warp i handles Row i*16. Head h -> Warp (h%4). Row = (h%4)*16 + (h/4).
    int target_row = (h_in_grp % kNWarps) * 16 + (h_in_grp / kNWarps);

    if (col < params.d) {
      const Element* q_ptr = reinterpret_cast<const Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) + bidh * params.q_head_stride;
      sQ(target_row, col) = q_ptr[col];
    } else {
      sQ(target_row, col) = Element(0);
    }
  }

  cute::cp_async_fence();
  __syncthreads();

  // MMA Setup
  typename Kernel_traits::TiledMma_PV tiled_mma;
  auto acc_o = partition_fragment_C(tiled_mma, Shape<Int<kSmemQRows>, Int<kHeadDim>>{});
  clear(acc_o);

  constexpr int kNRows = 2 * decltype(size<1>(acc_o))::value;
  flash::Softmax<kNRows, Kernel_traits::kNThreadsPerRow> softmax;

  typename Kernel_traits::GmemTiledCopyKInt8 gmem_tiled_copy_KInt8;
  auto gmem_thr_copy_KInt8 = gmem_tiled_copy_KInt8.get_thread_slice(tidx);

  // Scales
  const float k_scale = params.k_scale_ptr ? static_cast<float>(reinterpret_cast<const Element*>(params.k_scale_ptr)[(params.k_quant_type == 2) ? kv_head_idx : 0]) : 1.0f;
  const float v_scale = params.v_scale_ptr ? static_cast<float>(reinterpret_cast<const Element*>(params.v_scale_ptr)[(params.k_quant_type == 2) ? kv_head_idx : 0]) : 1.0f;

  bool is_first_block = true;
  for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
    // Load K/V
    const index_t k_base = binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) + kv_head_idx * params.k_head_stride;
    const index_t v_base = binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) + kv_head_idx * params.v_head_stride;

    Tensor gK = local_tile(make_tensor(make_gmem_ptr(reinterpret_cast<const ElementInt8*>(params.k_ptr) + k_base), make_shape(binfo.actual_seqlen_k, params.d), make_stride(params.k_row_stride, _1{})), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));
    Tensor gV = local_tile(make_tensor(make_gmem_ptr(reinterpret_cast<const ElementInt8*>(params.v_ptr) + v_base), make_shape(binfo.actual_seqlen_k, params.d), make_stride(params.v_row_stride, _1{})), Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));

    Tensor tKsK = gmem_thr_copy_KInt8.partition_D(sK);
    Tensor tVsV = gmem_thr_copy_KInt8.partition_D(sV);

    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KInt8, gmem_thr_copy_KInt8.partition_S(gK), tKsK, gmem_thr_copy_KInt8.partition_D(make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)))), make_tensor<bool>(make_shape(size<2>(tKsK))), binfo.actual_seqlen_k - n_block * kBlockN);
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KInt8, gmem_thr_copy_KInt8.partition_S(gV), tVsV, gmem_thr_copy_KInt8.partition_D(make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)))), make_tensor<bool>(make_shape(size<2>(tKsK))), binfo.actual_seqlen_k - n_block * kBlockN);

    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // Gemm Quant Q*K^T
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    Tensor tSsQ = smem_tiled_copy_Q.get_thread_slice(tidx).partition_S(sQ);
    auto tSrQ = tiled_mma.get_thread_slice(tidx).partition_fragment_A(sQ);
    cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ);

    // Manual GEMM
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomInt8Default{}, tiled_mma);
    auto sK_quant = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKInt8{});
    auto tSsK = smem_tiled_copy_K.get_thread_slice(tidx).partition_S(make_tensor(sK_quant.data(), make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, Stride<_1, Int<Kernel_traits::kSmemRowStrideInt8>>{})));
    auto tSrK = tiled_mma.get_thread_slice(tidx).partition_fragment_B(make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + kSmemSizeQ)), typename Kernel_traits::SmemLayoutK{}));
    flash::gemm_quant_manual<false>(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_K, smem_tiled_copy_K.get_thread_slice(tidx), k_scale);

    // Masking Handled by implicit bounds check (or simple clip)
    // Actually we need to mask out tokens beyond seqlen_k
    if (n_block * kBlockN + kBlockN > binfo.actual_seqlen_k) {
      flash::Mask<false, false, false> mask(binfo.actual_seqlen_k, 0, 0, 0);
      mask.template apply_mask<false, false>(acc_s, n_block * kBlockN, 0, 0);
    }

    if constexpr (Is_softcap) flash::apply_softcap(acc_s, params.softcap);

    if (is_first_block)
      softmax.template softmax_rescale_o<true>(acc_s, acc_o, params.scale_softmax_log2);
    else
      softmax.template softmax_rescale_o<false>(acc_s, acc_o, params.scale_softmax_log2);

    if (tidx == 0) {
      // P@V
      // sP reuses sK shared memory. We must ensure all warps finished reading sK (in gemm_quant_manual)
      // before any warp starts writing sP.
      __syncthreads();

      auto sP = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + kSmemSizeQ)), make_layout(Shape<Int<kSmemQRows>, Int<kBlockN>>{}, Stride<Int<kBlockN>, _1>{}));
      // Acc_s to sP
      auto acc_s_rowcol = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
      const int lane_id = tidx % 32;
      const int warp_id = tidx / 32;
      for (int mi = 0; mi < size<0, 1>(acc_s_rowcol); ++mi) {
        int r_base = warp_id * 16 + (lane_id / 4);
        for (int i = 0; i < size<0, 0>(acc_s_rowcol); ++i) {
          int r = r_base + i * 8;
          for (int nj = 0; nj < size<1, 1>(acc_s_rowcol); ++nj) {
            int c_base = (lane_id % 4) * 2 + nj * 8;
            if (r < kBlockM && c_base < kBlockN) {
              reinterpret_cast<__half2&>(sP(r, c_base)) = __halves2half2(static_cast<half>(acc_s_rowcol(make_coord(i, mi), make_coord(0, nj))), static_cast<half>(acc_s_rowcol(make_coord(i, mi), make_coord(1, nj))));
            }
          }
        }
      }
      __syncthreads();

      // P@V GEMM
      auto smem_tiled_copy_P = make_tiled_copy_A(Copy_Atom<DefaultCopy, Element>{}, tiled_mma);
      auto tOrP = tiled_mma.get_thread_slice(tidx).partition_fragment_A(sP);
      auto tSsP = smem_tiled_copy_P.get_thread_slice(tidx).partition_S(sP);
      cute::copy(smem_tiled_copy_P, tSsP, smem_tiled_copy_P.get_thread_slice(tidx).retile_D(tOrP));  // Retile?

      auto sV_dummy = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + kSmemSizeQ + kSmemSizeKInt8)), make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}));
      auto tOrVt = tiled_mma.get_thread_slice(tidx).partition_fragment_B(sV_dummy);
      auto sV_quant_t = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVInt8transposed{});
      auto tSsV = smem_tiled_copy_K.get_thread_slice(tidx).partition_S(sV_quant_t);

      flash::gemm_quant_manual<false>(acc_o, tOrP, tOrVt, tSsV, tiled_mma, smem_tiled_copy_K, smem_tiled_copy_K.get_thread_slice(tidx), v_scale);

      __syncthreads();
      is_first_block = false;
    }

    // Epilogue: Output
    float sink = (params.head_sink_ptr != nullptr) ? static_cast<float>(reinterpret_cast<const Element*>(params.head_sink_ptr)[kv_head_idx]) : (params.smooth_softmax ? 0.0f : -flash::kInfinity);
    Tensor lse = softmax.template normalize_softmax_lse<Split>(acc_o, params.scale_softmax, sink);

    Tensor rO = flash::convert_type<ElementO>(acc_o);
    Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(smem_)), make_layout(Shape<Int<kSmemQRows>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{}));
    using SmemCopyAtomO = Copy_Atom<DefaultCopy, ElementO>;
    auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);
    cute::copy(smem_tiled_copy_O, smem_tiled_copy_O.get_thread_slice(tidx).retile_S(rO), smem_tiled_copy_O.get_thread_slice(tidx).partition_D(sO));
    __syncthreads();

    // Write Out Each Head
    for (int i = 0; i < num_q_heads_per_kv; ++i) {
      int row = (i % kNWarps) * 16 + (i / kNWarps);
      int bidh = q_head_start + i;

      // Only Row 'row' matters
      // Manual write to Global
      const int total_elems = kHeadDim;
      for (int j = tidx; j < total_elems; j += Kernel_traits::kNThreads) {
        ElementO val = sO(row, j);
        // Global Pointer
        // gOaccum or gO
        if constexpr (Split) {
          const index_t row_offset_oaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q * params.d_rounded;
          ElementO* gOaccum_ptr = reinterpret_cast<ElementO*>(params.oaccum_ptr) + row_offset_oaccum;
          gOaccum_ptr[j] = val;
        } else {
          const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) + bidh * params.o_head_stride;
          ElementO* gO_ptr = reinterpret_cast<ElementO*>(params.o_ptr) + row_offset_o;
          gO_ptr[j] = val;
        }
      }

      // Write LSE
      const int lane = tidx % 32;
      const int warp = tidx / 32;
      // LSE is distributed. lse(0) -> row (warp*16 + lane/4).
      // Match row.
      if ((lane % 4) == 0) {
        int local_row = warp * 16 + (lane / 4);
        if (local_row == row) {
          index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q;
          ElementAccum* lse_ptr = reinterpret_cast<ElementAccum*>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum;
          lse_ptr[0] = lse(0);
        }
        local_row += 8;
        if (local_row == row) {
          index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q;
          ElementAccum* lse_ptr = reinterpret_cast<ElementAccum*>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum;
          lse_ptr[0] = lse(1);
        }
      }
    }
  }
}

}  // namespace int8
}  // namespace FLASH_NAMESPACE

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
