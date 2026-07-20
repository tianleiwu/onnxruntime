// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// CUTLASS tensor-core implementation of the blockwise-scaled FP8 (E4M3) GEMM used by the
// MatMulBlockQuantized contrib operator, targeting NVIDIA Hopper (SM90).
//
// This translation unit is compiled at exactly 90a-real inside a dedicated CUDA OBJECT
// library (see cmake/onnxruntime_providers_cuda.cmake). The dispatcher in
// matmul_block_quantized.cc only references the symbols defined here when the SM90 object
// library is built (guarded by ORT_ENABLE_BLOCKQUANT_SM90 on the parent target).

#include "contrib_ops/cuda/math/matmul_block_quantized.h"

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
//   B: [K, N] row-major fp8 e4m3  == [N, K] column-major (CUTLASS TN layout)
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

using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using TileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _2, _1>;

// scale_a: [M, K/128] fp32, K-major (per token, one scale per 128-element K block).
// scale_b: [K/128, N] fp32, MN-major (per column, one scale per 128-element K block).
constexpr int kScaleGranularityM = 1;
constexpr int kScaleGranularityN = 1;
constexpr int kScaleGranularityK = 128;
using ScaleConfig = cutlass::detail::Sm90BlockwiseScaleConfig<
    kScaleGranularityM, kScaleGranularityN, kScaleGranularityK,
    cute::GMMA::Major::K, cute::GMMA::Major::MN>;

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
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;

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
  auto arguments = MakeArguments(nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k);
  return Gemm::get_workspace_size(arguments);
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
