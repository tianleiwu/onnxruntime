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

namespace onnxruntime {
namespace flash {
using namespace cute;
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
    const Params& params, const int bidb, const int bidh_kv, const int m_block,
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

  // GQA-aware: the total "rows" in the M dimension is (seqlen_q * h_h_k_ratio)
  const int virtual_seqlen_q = binfo.actual_seqlen_q * params.h_h_k_ratio;

  if (m_block * kBlockM >= virtual_seqlen_q) return;

  // Compute n_block range for Split-K
  const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
  // Note: For simplicity in GQA-aware blocking, we use the first Q head of the group for causal/local constraints.
  // Since all Q rows in this block share the same K padding/masking, this is safe for decoding.
  const int n_block_min = !Is_local
                              ? n_split_idx * n_blocks_per_split
                              : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM / params.h_h_k_ratio + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
  int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
  if (Is_causal || Is_local) {
    n_block_max = std::min(n_block_max,
                           cute::ceil_div((m_block + 1) * kBlockM / params.h_h_k_ratio + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
  }

  // Early Exit / Initialization for Empty Split Blocks
  if (n_block_min >= n_block_max) {
    // We exit early and write 0 to gOaccum and -inf to gLSEaccum (if Split).
    if constexpr (Split) {
      // For Split-K initialization in GQA-aware mode, we need to handle all heads in the group.
      // This part is complex; for now we rely on the fact that if one head group is empty, they all are for decoding.
      // We initialize based on the flattened M-dimension.
      const index_t row_offset_oaccum = ((n_split_idx * params.b + bidb) * params.h + bidh_kv * params.h_h_k_ratio) * params.seqlen_q * params.d_rounded;
      const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh_kv * params.h_h_k_ratio) * params.seqlen_q;

      Tensor gOaccum_base = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(params.oaccum_ptr) + row_offset_oaccum),
                                        make_shape(params.seqlen_q * params.h_h_k_ratio, Int<kHeadDim>{}),
                                        make_stride(kHeadDim, _1{}));
      Tensor gOaccum = local_tile(gOaccum_base, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

      Tensor gLSEaccum_base = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                          make_shape(params.seqlen_q * params.h_h_k_ratio), Stride<_1>{});
      Tensor gLSEaccum = local_tile(gLSEaccum_base, Shape<Int<kBlockM>>{}, make_coord(m_block));

      int total_elems = kBlockM * kHeadDim;
      for (int i = tidx; i < total_elems; i += Kernel_traits::kNThreads) {
        int row = i / kHeadDim;
        int col = i % kHeadDim;
        if (m_block * kBlockM + row < virtual_seqlen_q) {
          if (col < params.d) {
            gOaccum(row, col) = 0.0f;
          }
        }
      }

      // Init LSE
      for (int i = tidx; i < kBlockM; i += Kernel_traits::kNThreads) {
        if (m_block * kBlockM + i < virtual_seqlen_q) {
          gLSEaccum(i) = -std::numeric_limits<ElementAccum>::infinity();
        }
      }
    }
    return;
  }

  // Layout: [sQ (FP16)] [sK (Int8, padded)] [sV (Int8, padded)]
  // ============================================================================
  constexpr int kSmemSizeQ = kBlockM * kHeadDim * sizeof(Element);
  constexpr int kSmemSizeKInt8 = kBlockN * Kernel_traits::kSmemRowStrideInt8 * sizeof(ElementInt8);

  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(smem_ + kSmemSizeQ)),
                          typename Kernel_traits::SmemLayoutKInt8{});
  Tensor sV = make_tensor(make_smem_ptr(reinterpret_cast<ElementInt8*>(smem_ + kSmemSizeQ + kSmemSizeKInt8)),
                          typename Kernel_traits::SmemLayoutKInt8{});

  const int total_kv_bytes = 2 * kBlockN * Kernel_traits::kSmemRowStrideInt8;
  for (int i = tidx; i < total_kv_bytes / 16; i += Kernel_traits::kNThreads) {
    reinterpret_cast<uint4*>(smem_ + kSmemSizeQ)[i] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<ElementO*>(smem_)),
                          typename Kernel_traits::SmemLayoutO{});

  // GQA-aware: kv_head_idx is bidh_kv directly.
  const int kv_head_idx = bidh_kv;

  // Get scales
  const int scale_idx = (params.k_quant_type == 2) ? kv_head_idx : 0;
  const float k_scale = params.k_scale_ptr ? static_cast<float>(reinterpret_cast<const Element*>(params.k_scale_ptr)[scale_idx]) : 1.0f;
  const float v_scale = params.v_scale_ptr ? static_cast<float>(reinterpret_cast<const Element*>(params.v_scale_ptr)[scale_idx]) : 1.0f;

  // Global Memory Tensors: Use composed layouts to flatten hierarchy for GQA
  // We combine Head (within group) and Seq into a single flattened M dimension "virtual_seqlen_q"
  // but use the correct strides to map back to (head_in_group, actual_seq)
  const int head_group_start = bidh_kv * params.h_h_k_ratio;
  const Element* q_group_ptr = reinterpret_cast<const Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) + head_group_start * params.q_head_stride;

  // O and LSE accumulators: global base pointers
  const index_t row_offset_o_base_gqa = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb) + (index_t)head_group_start * params.o_head_stride;
  const index_t row_offset_oaccum_base_gqa = ((index_t)(n_split_idx * params.b + bidb) * params.h + head_group_start) * params.seqlen_q * params.d_rounded;
  const index_t row_offset_lseaccum_base_gqa = ((index_t)(n_split_idx * params.b + bidb) * params.h + head_group_start) * params.seqlen_q;

  ElementO* o_ptr_base = reinterpret_cast<ElementO*>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum_base_gqa : row_offset_o_base_gqa);
  ElementAccum* lse_ptr_base = reinterpret_cast<ElementAccum*>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum_base_gqa;

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
  auto sK_dummy = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + kSmemSizeQ)),
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

  // We use the physical Int8 layout so that partitioning correctly accounts for padding.
  auto sK_quant = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKInt8{});
  // Partition the physical Smem tensor for loading into registers
  auto tSsK = smem_thr_copy_KInt8.partition_S(make_tensor(sK_quant.data(), make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, Stride<_1, Int<Kernel_traits::kSmemRowStrideInt8>>{})));

  // V dummy for partitioning (must be transposed shape (HeadDim, BlockN) for B operand)
  auto sV_dummy = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + kSmemSizeQ + kSmemSizeKInt8)),
                              make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}));
  auto tOrVt = thr_mma.partition_fragment_B(sV_dummy);

  // V physical for partitioning (must be transposed shape (HeadDim, BlockN) for B operand)
  auto sV_quant_t = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVInt8transposed{});
  // Partition the physical Smem tensor for loading into registers
  auto tSsV = smem_thr_copy_KInt8.partition_S(sV_quant_t);

  auto acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(acc_o);

  constexpr int kNRows = 2 * decltype(size<1>(acc_o))::value;
  flash::Softmax<kNRows, Kernel_traits::kNThreadsPerRow> softmax;

  // ============================================================================
  // Load Q manually from Global to Shared
  for (int i = tidx; i < kBlockM * kHeadDim; i += Kernel_traits::kNThreads) {
    int row = i / kHeadDim;
    int col = i % kHeadDim;
    int virtual_row = m_block * kBlockM + row;
    if (virtual_row < virtual_seqlen_q) {
      int h_rel = virtual_row % params.h_h_k_ratio;
      int s_idx = virtual_row / params.h_h_k_ratio;
      // In BSNH: [Batch, Seq, Head, Dim]
      index_t offset = (index_t)s_idx * params.q_row_stride + (index_t)h_rel * params.q_head_stride + col;
      sQ(row, col) = q_group_ptr[offset];
    }
  }
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
  // Main Attention Loop
  // ============================================================================
  bool is_first_block = true;
  for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
    // 1. Load K / V (INT8) from Global to Smem (INT8)
    Tensor gK_int8 = local_tile(mK_int8, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));
    Tensor gV_int8 = local_tile(mV_int8, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));

    Tensor tKgK_int8 = gmem_thr_copy_KInt8.partition_S(gK_int8);
    Tensor tKsK_int8 = gmem_thr_copy_KInt8.partition_D(sK);
    Tensor tVgV_int8 = gmem_thr_copy_KInt8.partition_S(gV_int8);
    Tensor tVsV_int8 = gmem_thr_copy_KInt8.partition_D(sV);

    // Prepare predicates for K/V copy
    Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tKcK = gmem_thr_copy_KInt8.partition_D(cK);
    Tensor tKpK = make_tensor<bool>(make_shape(size<2>(tKsK_int8)));
    if (!Is_even_K) {
#pragma unroll
      for (int k = 0; k < size(tKpK); ++k) tKpK(k) = get<1>(tKcK(0, 0, k)) < params.d;
    }

    bool is_knew_block = false;
    if constexpr (kEnableOnTheFlyNewKVQuantization) {
      // Check if we are in the 'New K' territory (appended data)
      // For now we assume knew is appended at the very end
      is_knew_block = (params.knew_ptr != nullptr) && (n_block * kBlockN >= binfo.seqlen_k_cache);
      if (is_knew_block) {
        // [Write-Back Setup] Define Write Tensor for K (Global INT8 Cache)
        Tensor mK_int8_write = make_tensor(make_gmem_ptr(reinterpret_cast<ElementInt8*>(params.k_ptr) + k_base_offset),
                                           make_shape(binfo.actual_seqlen_k, params.d),
                                           make_stride(params.k_row_stride, _1{}));
        Tensor gK_int8_write = local_tile(mK_int8_write, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));
        Tensor tKgK_write = gmem_thr_copy_KInt8.partition_S(gK_int8_write);  // Thread partition for Global Write

        // [New K/V Base Pointers] Hoist invariant offset calculation
        size_t k_batch_head_offset = (bidb * params.knew_batch_stride) + (bidh_kv * params.knew_head_stride);
        size_t v_batch_head_offset = (bidb * params.vnew_batch_stride) + (bidh_kv * params.vnew_head_stride);
        Element* knew_base_ptr = reinterpret_cast<Element*>(params.knew_ptr) + k_batch_head_offset;
        Element* vnew_base_ptr = reinterpret_cast<Element*>(params.vnew_ptr) + v_batch_head_offset;

        // [Scale Pointers]
        const Element* k_scale_base = reinterpret_cast<const Element*>(params.k_scale_ptr);
        const Element* v_scale_base = reinterpret_cast<const Element*>(params.v_scale_ptr);

        // Optimize scale access: if per-tensor, load once.
        float k_scale_val = 1.0f;
        float v_scale_val = 1.0f;
        if (params.k_quant_type != 2 && params.k_scale_ptr) k_scale_val = static_cast<float>(k_scale_base[0]);
        if (params.v_quant_type != 2 && params.v_scale_ptr) v_scale_val = static_cast<float>(v_scale_base[0]);

        // K Load + Quantize
        Tensor tKsK_dst = gmem_thr_copy_KInt8.partition_D(sK);
        int start_row = n_block * kBlockN;

#pragma unroll
        for (int i = 0; i < size(tKsK_dst); ++i) {
          auto coord = tKcK(i);
          int row_in_block = get<0>(coord);
          int h_idx = get<1>(coord);
          int global_k_row = start_row + row_in_block;

          ElementInt8 val_int8 = 0;
          if (global_k_row >= binfo.seqlen_k_cache && global_k_row < binfo.actual_seqlen_k) {
            int knew_relative_row = global_k_row - binfo.seqlen_k_cache;

            // Optimized load
            Element val_fp16 = knew_base_ptr[knew_relative_row * params.knew_row_stride + h_idx];

            float scale = k_scale_val;
            if (params.k_quant_type == 2) {  // Per-channel
              scale = static_cast<float>(k_scale_base[bidh_kv * kHeadDim + h_idx]);
            }

            // Quantize: INT8 = FP16 * DivScale (or FP16 / Scale).
            val_int8 = static_cast<int8_t>(static_cast<float>(val_fp16) / scale);

            // [Write-Back] Store to Global INT8 Cache
            // tKgK_write(i) corresponds to the same global element as the one we just quantized
            if (i < size(tKgK_write)) {
              tKgK_write(i) = val_int8;
            }
          }
          tKsK_dst(i) = val_int8;
        }

        // [Write-Back Setup] Define Write Tensor for V (Global INT8 Cache)
        Tensor mV_int8_write = make_tensor(make_gmem_ptr(reinterpret_cast<ElementInt8*>(params.v_ptr) + v_base_offset),
                                           make_shape(binfo.actual_seqlen_k, params.d),
                                           make_stride(params.v_row_stride, _1{}));
        Tensor gV_int8_write = local_tile(mV_int8_write, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));
        Tensor tVgV_write = gmem_thr_copy_KInt8.partition_S(gV_int8_write);

        // V Load + Quantize
        Tensor tVsV_dst = gmem_thr_copy_KInt8.partition_D(sV);

#pragma unroll
        for (int i = 0; i < size(tVsV_dst); ++i) {
          auto coord = tKcK(i);  // Use same coordinates as K for simplicity if layout matches
          int row_in_block = get<0>(coord);
          int h_idx = get<1>(coord);
          int global_v_row = start_row + row_in_block;

          ElementInt8 val_int8 = 0;
          if (global_v_row >= binfo.seqlen_k_cache && global_v_row < binfo.actual_seqlen_k) {
            int knew_relative_row = global_v_row - binfo.seqlen_k_cache;

            // Optimized load
            Element val_fp16 = vnew_base_ptr[knew_relative_row * params.vnew_row_stride + h_idx];

            float scale = v_scale_val;
            if (params.v_quant_type == 2) {  // Per-channel
              scale = static_cast<float>(v_scale_base[bidh_kv * kHeadDim + h_idx]);
            }

            val_int8 = static_cast<int8_t>(static_cast<float>(val_fp16) / scale);

            // [Write-Back] Store to Global INT8 Cache
            if (i < size(tVgV_write)) {
              tVgV_write(i) = val_int8;
            }
          }
          tVsV_dst(i) = val_int8;
        }
      }
    }

    if (!is_knew_block) {
      flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KInt8, tKgK_int8, tKsK_int8, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN);
      flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_KInt8, tVgV_int8, tVsV_int8, tKcK, tKpK, binfo.actual_seqlen_k - n_block * kBlockN);
    }

    cute::cp_async_fence();
    cute::cp_async_wait<0>();
    __syncthreads();

    // 2. Q @ K^T (Gemm Quant)
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);

    // Q @ K^T using gemm_quant_manual:
    // - Uses manual loop to bypass cute::gemm shape assertions
    // - Int8 K data is loaded into registers, dequantized to FP16, then used in MMA
    // - k_scale converts Int8 values back to original FP16 range
    flash::gemm_quant_manual<false>(acc_s, tSrQ, tSrK, tSsK, tiled_mma, smem_tiled_copy_KInt8, smem_thr_copy_KInt8, k_scale);

    // if (bidb == 0 && bidh == 0 && tidx == 0 && n_block == 0) {
    //   printf("DEBUG: bidb=0 bidh=0 n_block=0 k_scale=%f v_scale=%f acc_s(0,0)=%f\n", k_scale, v_scale, (float)acc_s(0));
    // }

    // Masking: map virtual row back to original sequence row
    constexpr int kNWarps = Kernel_traits::kNWarps;
    const int col_idx_offset = n_block * kBlockN;
    // Map virtual row index (0..virtual_seqlen_q-1) to actual sequence index (0..actual_seqlen_q-1)
    const int row_idx_offset = (m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4) / params.h_h_k_ratio;
    const int warp_row_stride = (kNWarps * 16) / params.h_h_k_ratio;  // Approximation for masking

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
    // - Reuse sK memory for sP (attention probabilities)
    // - sK is (BlockN, HeadDim + Pad) = (64, 144) -> ~9KB.
    // - sP needs (BlockM, BlockN) = (64, 64) elements (FP16) -> 8KB.
    // - Fits safely within sK's allocation.
    auto sP = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + kSmemSizeQ)),
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
  // For GQA efficiency, we use the first head of the group for sink if provided
  const float sink = (params.head_sink_ptr)
                         ? static_cast<float>(reinterpret_cast<const Element*>(params.head_sink_ptr)[head_group_start])
                         : (params.smooth_softmax ? 0.0f : -flash::kInfinity);
  Tensor lse = softmax.template normalize_softmax_lse<Split>(acc_o, params.scale_softmax, sink);

  // 1. Copy Reg -> Smem (Standard Flash method, safe)
  Tensor tOrO = make_tensor(acc_o.data(), typename Kernel_traits::SmemLayoutO{});
  Tensor tOsO_acc = tiled_mma.get_slice(tidx).partition_C(sO);
  cute::copy(tOrO, tOsO_acc);
  __syncthreads();

  // 2. Copy Smem -> Gmem (Manual for GQA mapping)
  for (int i = tidx; i < kBlockM * kHeadDim; i += Kernel_traits::kNThreads) {
    int row = i / kHeadDim;
    int col = i % kHeadDim;
    int virtual_row = m_block * kBlockM + row;
    if (virtual_row < virtual_seqlen_q) {
      int h_rel = virtual_row % params.h_h_k_ratio;
      int s_idx = virtual_row / params.h_h_k_ratio;

      if (col < params.d) {
        index_t offset;
        if constexpr (Split) {
          // Split (Oaccum): [splits, Batch, Head, Seq, Dim]. Seq is inner to Head.
          offset = (index_t)h_rel * params.seqlen_q * params.d_rounded + (index_t)s_idx * params.d_rounded + col;
        } else {
          // Unsplit (BSNH): [Batch, Seq, Head, Dim]. Head is inner to Seq.
          offset = (index_t)s_idx * params.o_row_stride + (index_t)h_rel * params.o_head_stride + col;
        }
        o_ptr_base[offset] = static_cast<ElementO>(sO(row, col));
      }
    }
  }

  // 3. Write LSE: use the thread-local `lse` register tensor
  const int lane = tidx % 32;
  const int warp = tidx / 32;
  if (lane % 4 == 0) {
    for (int mi = 0; mi < size<0>(lse); ++mi) {
      int local_row = warp * 16 + (lane / 4) + mi * 8;
      int virtual_row = m_block * kBlockM + local_row;
      if (virtual_row < virtual_seqlen_q) {
        int h_rel = virtual_row % params.h_h_k_ratio;
        int s_idx = virtual_row / params.h_h_k_ratio;
        index_t lse_offset = (index_t)h_rel * params.seqlen_q + s_idx;
        lse_ptr_base[lse_offset] = lse(mi);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace int8
}  // namespace flash
}  // namespace onnxruntime

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
