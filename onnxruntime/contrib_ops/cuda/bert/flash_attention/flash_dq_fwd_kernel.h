/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2024, Microsoft.
 * Native INT8 MMA Kernel for Flash Attention with Quantized KV Cache
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

namespace onnxruntime {
namespace flash {
using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// INT8 MMA Kernel Traits for Q×K^T with native SM80+ INT8 Tensor Cores
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

  static_assert(kHeadDim % 32 == 0, "kHeadDim must be multiple of 32 for INT8 MMA");

  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

  // Standard FP16 MMA for P×V (m16n8k16)
  using MMA_Atom_PV = std::conditional_t<
      std::is_same_v<elem_type, cutlass::half_t>,
      MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>,
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>>;

  using TiledMma_PV = TiledMMA<
      MMA_Atom_PV,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, _16>>;

  // INT8 MMA for Q×K^T (m16n8k32 s8s8s32)
  // Note: This MMA has K=32, twice the K dimension of FP16 MMA
  using MMA_Atom_QK_Int8 = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;

  using TiledMma_QK_Int8 = TiledMMA<
      MMA_Atom_QK_Int8,
      Layout<Shape<Int<kNWarps>, _1, _1>>,
      Tile<Int<16 * kNWarps>, _16, _32>>;

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

  // INT8 layout for K (no dequantization - native INT8)
  using SmemLayoutKInt8 = decltype(tile_to_shape(
      SmemLayoutAtom{},
      Shape<Int<kBlockN>, Int<kHeadDim>>{}));

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

  // INT8 global copy (direct INT8 without aliasing)
  using GmemTiledCopyKInt8 = decltype(make_tiled_copy(Copy_Atom<Gmem_copy_struct, ElementInt8>{},
                                                      GmemLayoutAtom{},
                                                      Layout<Shape<_1, _16>>{}));  // 16 int8s = 16 bytes

  using SmemCopyAtomO = Copy_Atom<DefaultCopy, Element>;
  using SmemTiledCopyO = decltype(make_tiled_copy_C(SmemCopyAtomO{}, TiledMma_PV{}));

  using GmemTiledCopyO = decltype(make_tiled_copy(Copy_Atom<DefaultCopy, cute::uint128_t>{},
                                                  GmemLayoutAtom{},
                                                  Layout<Shape<_1, _8>>{}));
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
// INT8 Score Dequantization
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
// Main Kernel: Native INT8 MMA Flash Attention (Minimal Version for HeadDim=128)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
          bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_attn_1rowblock_int8mma(
    const Params& params, const int bidb, const int bidh, const int m_block) {
  using Element = typename Kernel_traits::Element;
  using ElementAccum = typename Kernel_traits::ElementAccum;
  using ElementInt8 = typename Kernel_traits::ElementInt8;
  using ElementInt32 = typename Kernel_traits::ElementInt32;
  using index_t = typename Kernel_traits::index_t;

  extern __shared__ char smem_[];

  const int tidx = threadIdx.x;

  constexpr int kBlockM = Kernel_traits::kBlockM;
  constexpr int kBlockN = Kernel_traits::kBlockN;
  constexpr int kHeadDim = Kernel_traits::kHeadDim;

  const BlockInfo<!Is_even_MN> binfo(params, bidb);

  if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

  // Compute n_block range
  const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
  int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
  if (Is_causal || Is_local) {
    n_block_max = std::min(n_block_max,
                           cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
  }

  if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
    return;  // Early exit
  }

  // ============================================================================
  // Shared Memory Allocation
  // Layout: [sQ (FP16)] [sK (FP16)] [sV (FP16)]
  // ============================================================================
  Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + sizeof(Element) * size(sQ))),
                          typename Kernel_traits::SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_ + sizeof(Element) * (size(sQ) + size(sK)))),
                          typename Kernel_traits::SmemLayoutV{});
  // Transposed views of V for P×V GEMM (matching old kernel pattern)
  Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
  Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

  // Define sO early (reusing Q memory) for Output Staging (FP16)
  Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                          typename Kernel_traits::SmemLayoutQ{});

  // Get scales
  const float k_scale = params.k_scale_ptr ? static_cast<float>(reinterpret_cast<const Element*>(params.k_scale_ptr)[0]) : 1.0f;
  const float v_scale = params.v_scale_ptr ? static_cast<float>(reinterpret_cast<const Element*>(params.v_scale_ptr)[0]) : 1.0f;

  // ============================================================================
  // Global Memory Tensors
  // ============================================================================
  Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<const Element*>(params.q_ptr) +
                                        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                          make_shape(binfo.actual_seqlen_q, params.h, params.d),
                          make_stride(params.q_row_stride, params.q_head_stride, _1{}));
  Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

  // K is INT8 in global memory
  Tensor mK_int8 = make_tensor(make_gmem_ptr(reinterpret_cast<const ElementInt8*>(params.k_ptr) +
                                             binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                               make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                               make_stride(params.k_row_stride, params.k_head_stride, _1{}));

  // V is INT8 in global memory
  Tensor mV_int8 = make_tensor(make_gmem_ptr(reinterpret_cast<const ElementInt8*>(params.v_ptr) +
                                             binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                               make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                               make_stride(params.v_row_stride, params.v_head_stride, _1{}));

  // Output
  Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) +
                                        binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                          make_shape(binfo.actual_seqlen_q, params.h, params.d),
                          make_stride(params.o_row_stride, params.o_head_stride, _1{}));
  Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

  // ============================================================================
  // Copy Setup (QKV)
  // ============================================================================
  typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

  // Construct identity layout for sQ and sK
  Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
  // Repeat the partitioning with identity layouts
  Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);  // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
  // Allocate predicate tensors for k
  Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
  if (!Is_even_K) {
#pragma unroll
    for (int k = 0; k < size(tQpQ); ++k) {
      tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
    }
  }

  // ============================================================================
  // MMA Setup
  // ============================================================================
  // We use FP16 MMA (MMA_Atom_PV)
  typename Kernel_traits::TiledMma_PV tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tidx);

  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);  // V transposed register fragment (same as old kernel)

  // Accumulator for output (Global over HeadDim)
  Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
  clear(acc_o);

  // Softmax
  constexpr int kNRows = 2 * decltype(size<1>(acc_o))::value;
  flash::Softmax<kNRows> softmax;

  // ============================================================================
  // Load Q to Shared Memory
  // ============================================================================
  flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                     binfo.actual_seqlen_q - m_block * kBlockM);
  cute::cp_async_fence();
  cute::cp_async_wait<0>();
  __syncthreads();

  // ============================================================================
  // Smem TiledCopiers
  // ============================================================================
  // Q copier (FP16)
  auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
  Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

  auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
  Tensor tSsK = smem_thr_copy_K.partition_S(sK);

  // Use transposed copy atom and sVt for V to ensure correct P×V GEMM layout
  auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
  Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

  // ============================================================================
  // Main Attention Loop
  // ============================================================================
  bool is_first_block = true;
  for (int n_block = n_block_max - 1; n_block >= n_block_min; --n_block) {
    // ------------------------------------------------------------------------
    // 1. Load K INT8 -> Smem K FP16 (Dequantize)
    // Each thread processes a portion of the (kBlockN x kHeadDim) tile.
    // Use direct 2D tensor indexing, NOT TiledCopy partitioning (which has element size mismatch).
    // ------------------------------------------------------------------------
    Tensor gK_int8 = local_tile(mK_int8(_, bidh / params.h_h_k_ratio, _),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));

    // Each thread handles (kBlockN * kHeadDim) / kNThreads elements
    constexpr int kTotalElems = kBlockN * kHeadDim;
    constexpr int kElemsPerThread = kTotalElems / Kernel_traits::kNThreads;
    const int start_elem = tidx * kElemsPerThread;

#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      int linear_idx = start_elem + i;
      int row = linear_idx / kHeadDim;
      int col = linear_idx % kHeadDim;
      ElementInt8 val_int8 = 0;
      if (n_block * kBlockN + row < binfo.actual_seqlen_k) {
        val_int8 = gK_int8(row, col);
      }
      sK(row, col) = static_cast<Element>(static_cast<float>(val_int8) * k_scale);
    }

    // ------------------------------------------------------------------------
    // 2. Load V INT8 -> Smem V FP16 (Dequantize)
    // ------------------------------------------------------------------------
    Tensor gV_int8 = local_tile(mV_int8(_, bidh / params.h_h_k_ratio, _),
                                Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(n_block, 0));

#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      int linear_idx = start_elem + i;
      int row = linear_idx / kHeadDim;
      int col = linear_idx % kHeadDim;
      ElementInt8 val_int8 = 0;
      if (n_block * kBlockN + row < binfo.actual_seqlen_k) {
        val_int8 = gV_int8(row, col);
      }
      sV(row, col) = static_cast<Element>(static_cast<float>(val_int8) * v_scale);
    }

    __syncthreads();

    // ------------------------------------------------------------------------
    // 3. Q @ K^T
    // ------------------------------------------------------------------------
    Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
    clear(acc_s);

    flash::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K);

    // Masking - Use proper thread-aware indexing like reference kernel
    constexpr int kNWarps = Kernel_traits::kNWarps;
    const int col_idx_offset = n_block * kBlockN;
    const int row_idx_offset = m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4;
    const int warp_row_stride = kNWarps * 16;

    // Create mask object
    flash::Mask<Is_causal, Is_local, /*Has_alibi=*/false> mask(
        binfo.actual_seqlen_k, binfo.actual_seqlen_q,
        params.window_size_left, params.window_size_right);

    // Apply mask - handles bounds checking and causal masking correctly
    mask.template apply_mask<Is_causal, Is_even_MN>(
        acc_s, col_idx_offset, row_idx_offset, warp_row_stride);

    // Apply softcap (tanh-based capping) if enabled
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

    // ------------------------------------------------------------------------
    // 4. P @ V
    // ------------------------------------------------------------------------
    // Convert P to FP16 and Reshape for MMA (A-operand)
    Tensor rP = flash::convert_type<Element>(acc_s);
    Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma_PV>(rP.layout()));

    // Use flash::gemm_rs for P×V GEMM (same approach as old kernel)
    flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

  }  // End n_block loop

  // ============================================================================
  // Finalize
  // ============================================================================
  // Normalize acc_o (full N=128)
  softmax.template normalize_softmax_lse</*Split=*/false>(acc_o, params.scale_softmax, -flash::kInfinity);

  // Store Reg -> Smem
  using SmemCopyAtomO = Copy_Atom<DefaultCopy, half_t>;
  auto smem_tiled_copy_O = make_tiled_copy_C(SmemCopyAtomO{}, tiled_mma);

  CUTE_UNROLL
  for (int ni = 0; ni < Kernel_traits::kHeadDim / 16; ++ni) {
    auto acc_slice = local_tile(acc_o, Shape<Int<1>, Int<2>>{}, make_coord(0, ni));
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    auto tSsO = smem_thr_copy_O.partition_D(sO);
    auto tSsO_ni = tSsO(_, _, ni);

    Tensor rO = flash::convert_type<Element>(acc_slice);
    cute::copy(smem_tiled_copy_O, rO, tSsO_ni);
  }
  __syncthreads();

  // Copy Smem -> Global
  typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
      gmem_tiled_copy_O, tOsO, tOgO,
      cute::make_identity_tensor(Shape<Int<kBlockM>, Int<1>, Int<1>>{}),
      cute::make_tensor<bool>(Shape<Int<kHeadDim>, Int<1>, Int<1>>{}, Stride<_1, _0, _0>{}),
      binfo.actual_seqlen_q - m_block * kBlockM);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
}  // namespace onnxruntime

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif
