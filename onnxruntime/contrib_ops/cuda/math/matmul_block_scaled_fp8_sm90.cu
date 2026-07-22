// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CUTLASS tensor-core implementation of the blockwise-scaled FP8 (E4M3) GEMM used by the
// MatMulBlockScaledFp8 contrib operator, targeting NVIDIA Hopper (SM90).
//
// This translation unit is compiled at exactly 90a-real inside a dedicated CUDA OBJECT
// library (see cmake/onnxruntime_providers_cuda.cmake). The dispatcher in
// matmul_block_scaled_fp8.cc only references the symbols defined here when the SM90 object
// library is built (guarded by ORT_ENABLE_BLOCKQUANT_SM90 on the parent target).

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"

#if !defined(DISABLE_FLOAT8_TYPES)

#include "core/providers/cuda/cuda_common.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-parameter"
// CUTLASS blockwise-scale layout helpers use structured bindings that are unused in some
// specializations; keep this suppressed for the whole TU because the diagnostic is emitted at
// template-instantiation sites inside this file's launcher, not at include time.
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#endif

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/numeric_types.h"
#include "cute/tensor.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace onnxruntime::contrib::cuda {

namespace {

using namespace cute;

// GEMM operand configuration.
//   A: [M, K] row-major fp8 e4m3
//   B: [N, K] row-major fp8 e4m3 (K-major; CUTLASS ColumnMajor B == TN layout)
//   D: [M, N] row-major bf16
using ElementA = cutlass::float_e4m3_t;
using LayoutA = cutlass::layout::RowMajor;
constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

using ElementB = cutlass::float_e4m3_t;
using LayoutB = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

using ElementC = cutlass::bfloat16_t;
using LayoutC = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementD = ElementC;
using LayoutD = LayoutC;
constexpr int AlignmentD = AlignmentC;

using ElementAccumulator = float;
using ElementCompute = float;

// Small-M Hopper kernel adapted from the DeepGEMM dataflow: one 64x64 output tile per CTA,
// TMA loads of 64x128 FP8 tiles, and a single warpgroup issuing m64n64k32 WGMMA operations.
// Unlike DeepGEMM's blockwise B scaling, ORT has one B scale per output column, so scale
// promotion uses the coordinate of every accumulator fragment element.
constexpr int kSmallMMin = 9;
constexpr int kSmallMMax = 64;
constexpr int kSmallTileM = 64;
constexpr int kSmallTileN = 64;
constexpr int kSmallTileK = 128;

using SmallTileShape = Shape<Int<kSmallTileM>, Int<kSmallTileN>, Int<kSmallTileK>>;
using SmallSmemLayoutA = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<ElementA>{}, make_shape(Int<kSmallTileM>{}, Int<kSmallTileK>{})));
using SmallSmemLayoutB = decltype(tile_to_shape(
    GMMA::Layout_K_SW128_Atom<ElementB>{}, make_shape(Int<kSmallTileN>{}, Int<kSmallTileK>{})));
using SmallTiledMma = decltype(make_tiled_mma(
    SM90_64x64x32_F32E4M3E4M3_SS_TN<GMMA::ScaleIn::One, GMMA::ScaleIn::One>{}));

template <typename TmaA, typename TmaB>
struct alignas(128) SmallMSharedStorage {
  cute::ArrayEngine<ElementA, cosize_v<SmallSmemLayoutA>> a;
  cute::ArrayEngine<ElementB, cosize_v<SmallSmemLayoutB>> b;
  uint64_t tma_barrier;
};

template <typename TmaA, typename TmaB>
__global__ void __launch_bounds__(128)
    SmallMBlockScaledFp8Kernel(int m,
                               int n,
                               int k,
                               const ElementA* a,
                               CUTLASS_GRID_CONSTANT TmaA const tma_a,
                               const ElementB* b,
                               CUTLASS_GRID_CONSTANT TmaB const tma_b,
                               const float* scale_a,
                               const float* scale_b,
                               ElementD* output) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900)
  extern __shared__ char shared_memory[];
  using SharedStorage = SmallMSharedStorage<TmaA, TmaB>;
  SharedStorage& storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  cute::Tensor s_a = make_tensor(make_smem_ptr(storage.a.begin()), SmallSmemLayoutA{});
  cute::Tensor s_b = make_tensor(make_smem_ptr(storage.b.begin()), SmallSmemLayoutB{});
  cute::Tensor m_a = tma_a.get_tma_tensor(make_shape(m, k));
  cute::Tensor m_b = tma_b.get_tma_tensor(make_shape(n, k));
  cute::Tensor g_a = local_tile(m_a, SmallTileShape{}, make_coord(0, _, _), Step<_1, X, _1>{});
  cute::Tensor g_b = local_tile(m_b, SmallTileShape{}, make_coord(_, blockIdx.x, _), Step<X, _1, _1>{});

  auto [tAgA, tAsA] = tma_partition(
      tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(s_a), group_modes<0, 2>(g_a));
  auto [tBgB, tBsB] = tma_partition(
      tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(s_b), group_modes<0, 2>(g_b));

  using TmaBarrier = cutlass::arch::ClusterTransactionBarrier;
  if (threadIdx.x == 0) {
    TmaBarrier::init(&storage.tma_barrier, 1);
  }
  __syncthreads();

  SmallTiledMma tiled_mma;
  auto thread_mma = tiled_mma.get_thread_slice(threadIdx.x);
  cute::Tensor tCsA = thread_mma.partition_A(s_a);
  cute::Tensor tCsB = thread_mma.partition_B(s_b);
  cute::Tensor tCrA = thread_mma.make_fragment_A(tCsA);
  cute::Tensor tCrB = thread_mma.make_fragment_B(tCsB);

  cute::Tensor c_output = make_identity_tensor(make_shape(Int<kSmallTileM>{}, Int<kSmallTileN>{}));
  cute::Tensor tCcOutput = thread_mma.partition_C(c_output);
  cute::Tensor tCrOutput = thread_mma.make_fragment_C(tCcOutput);
  clear(tCrOutput);

  const int k_blocks = k / kSmallTileK;
  for (int k_block = 0; k_block < k_blocks; ++k_block) {
    if (threadIdx.x == 0) {
      TmaBarrier::arrive_and_expect_tx(
          &storage.tma_barrier, sizeof(storage.a) + sizeof(storage.b));
      copy(tma_a.with(storage.tma_barrier), tAgA(_, k_block), tAsA);
      copy(tma_b.with(storage.tma_barrier), tBgB(_, k_block), tBsB);
    }
    TmaBarrier::wait(&storage.tma_barrier, k_block & 1);

    cute::Tensor tCrBlock = thread_mma.make_fragment_C(tCcOutput);
    clear(tCrBlock);
    warpgroup_arrive();
    gemm(tiled_mma, tCrA, tCrB, tCrBlock);
    warpgroup_commit_batch();
    warpgroup_wait<0>();

#pragma unroll
    for (int i = 0; i < size(tCrOutput); ++i) {
      const int row = get<0>(tCcOutput(i));
      const int column = static_cast<int>(blockIdx.x) * kSmallTileN + get<1>(tCcOutput(i));
      if (row < m && column < n) {
        const float combined_scale = scale_a[row * k_blocks + k_block] *
                                     scale_b[column * k_blocks + k_block];
        tCrOutput(i) += tCrBlock(i) * combined_scale;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < size(tCrOutput); ++i) {
    const int row = get<0>(tCcOutput(i));
    const int column = static_cast<int>(blockIdx.x) * kSmallTileN + get<1>(tCcOutput(i));
    if (row < m && column < n) {
      output[row * n + column] = static_cast<ElementD>(tCrOutput(i));
    }
  }
#endif
}

bool CanUseSmallMKernel(int m, int n, int k) {
  return m >= kSmallMMin && m <= kSmallMMax && n > 0 && k > 0 && k % kSmallTileK == 0;
}

Status LaunchSmallMKernel(const void* a_fp8,
                          const void* b_fp8,
                          const float* scale_a,
                          const float* scale_b,
                          void* output_bf16,
                          int m,
                          int n,
                          int k,
                          cudaStream_t stream) {
  auto problem_shape = make_shape(m, n, k);
  auto stride_a = make_stride(k, Int<1>{});
  auto stride_b = make_stride(k, Int<1>{});
  cute::Tensor tensor_a = make_tensor(reinterpret_cast<const ElementA*>(a_fp8),
                                      make_shape(m, k), stride_a);
  cute::Tensor tensor_b = make_tensor(reinterpret_cast<const ElementB*>(b_fp8),
                                      make_shape(n, k), stride_b);
  auto tma_a = make_tma_atom(SM90_TMA_LOAD{}, tensor_a, SmallSmemLayoutA{},
                             make_shape(Int<kSmallTileM>{}, Int<kSmallTileK>{}));
  auto tma_b = make_tma_atom(SM90_TMA_LOAD{}, tensor_b, SmallSmemLayoutB{},
                             make_shape(Int<kSmallTileN>{}, Int<kSmallTileK>{}));

  using TmaA = decltype(tma_a);
  using TmaB = decltype(tma_b);
  using SharedStorage = SmallMSharedStorage<TmaA, TmaB>;
  auto kernel = &SmallMBlockScaledFp8Kernel<TmaA, TmaB>;
  CUDA_RETURN_IF_ERROR(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(SharedStorage)));
  kernel<<<dim3((n + kSmallTileN - 1) / kSmallTileN), 128, sizeof(SharedStorage), stream>>>(
      get<0>(problem_shape), get<1>(problem_shape), get<2>(problem_shape),
      reinterpret_cast<const ElementA*>(a_fp8), tma_a,
      reinterpret_cast<const ElementB*>(b_fp8), tma_b,
      scale_a, scale_b, reinterpret_cast<ElementD*>(output_bf16));
  return CUDA_CALL(cudaGetLastError());
}

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _2, _1>;

// scale_a: [M, K/128] fp32, K-major (per token, one scale per 128-element K block).
// scale_b: [N, K/128] fp32, K-major (per column, one scale per 128-element K block).
constexpr int kScaleGranularityM = 1;
constexpr int kScaleGranularityN = 1;
constexpr int kScaleGranularityK = 128;
using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
    kScaleGranularityM, kScaleGranularityN, kScaleGranularityK,
    cute::GMMA::Major::K, cute::GMMA::Major::K>;

using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8Blockwise;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    KernelSchedule>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
    cutlass::gemm::PersistentScheduler>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

typename Gemm::Arguments MakeArguments(const void* a_fp8,
                                       const void* b_fp8,
                                       const float* scale_a,
                                       const float* scale_b,
                                       void* output_bf16,
                                       int m, int n, int k) {
  const int l = 1;
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));

  LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, l));
  LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, l));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {
          reinterpret_cast<const ElementA*>(a_fp8),
          stride_a,
          reinterpret_cast<const ElementB*>(b_fp8),
          stride_b,
          scale_a,
          layout_sfa,
          scale_b,
          layout_sfb,
      },
      {
          {},  // epilogue.thread
          reinterpret_cast<ElementC*>(output_bf16),
          stride_c,
          reinterpret_cast<ElementD*>(output_bf16),
          stride_d,
      }};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  return arguments;
}

}  // namespace

size_t GetBlockQuantizedFp8GemmSm90WorkspaceSize(int m, int n, int k) {
  if (CanUseSmallMKernel(m, n, k)) {
    return 0;
  }
  return Gemm::get_workspace_size(MakeArguments(nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k));
}

Status LaunchBlockQuantizedFp8GemmSm90(const void* a_fp8,
                                       const void* b_fp8,
                                       const float* scale_a,
                                       const float* scale_b,
                                       void* output_bf16,
                                       int m,
                                       int n,
                                       int k,
                                       int block_size,
                                       void* workspace,
                                       size_t workspace_size,
                                       cudaStream_t stream) {
  ORT_RETURN_IF_NOT(block_size == kScaleGranularityK,
                    "SM90 blockwise FP8 GEMM only supports block_size == ", kScaleGranularityK);

  if (CanUseSmallMKernel(m, n, k)) {
    return LaunchSmallMKernel(a_fp8, b_fp8, scale_a, scale_b, output_bf16, m, n, k, stream);
  }

  auto arguments = MakeArguments(a_fp8, b_fp8, scale_a, scale_b, output_bf16, m, n, k);
  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess,
                    "SM90 blockwise FP8 GEMM cannot implement the given problem: ",
                    cutlassGetStatusString(status));
  status = gemm.initialize(arguments, workspace, stream);
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess,
                    "SM90 blockwise FP8 GEMM initialize failed: ", cutlassGetStatusString(status));
  status = gemm.run(stream);
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess,
                    "SM90 blockwise FP8 GEMM run failed: ", cutlassGetStatusString(status));
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#endif  // !defined(DISABLE_FLOAT8_TYPES)
