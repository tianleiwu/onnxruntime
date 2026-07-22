// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"

#include "core/providers/cuda/cuda_common.h"

#if !defined(DISABLE_FLOAT8_TYPES)

#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/float8.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"

namespace onnxruntime::contrib::cuda {
namespace {

using namespace cute;

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
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using MmaTileShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

constexpr int kScaleGranularityM = 1;
constexpr int kScaleGranularityN = 1;
constexpr int kScaleGranularityK = 128;
using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
    kScaleGranularityM, kScaleGranularityN, kScaleGranularityK,
    cute::UMMA::Major::K, cute::UMMA::Major::K>;
using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
        sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelScheduleSm120Blockwise>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
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
  constexpr int l = 1;
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, l));
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, l));
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, l));
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, l));

  LayoutSFA layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, l));
  LayoutSFB layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, l));

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, l},
      {reinterpret_cast<const ElementA*>(a_fp8), stride_a,
       reinterpret_cast<const ElementB*>(b_fp8), stride_b,
       scale_a, layout_sfa, scale_b, layout_sfb},
      {{}, reinterpret_cast<ElementC*>(output_bf16), stride_c, reinterpret_cast<ElementD*>(output_bf16), stride_d}};
  arguments.epilogue.thread.alpha = 1.0f;
  arguments.epilogue.thread.beta = 0.0f;
  return arguments;
}

}  // namespace

size_t GetBlockQuantizedFp8GemmSm120WorkspaceSize(int m, int n, int k) {
  return Gemm::get_workspace_size(MakeArguments(nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k));
}

Status LaunchBlockQuantizedFp8GemmSm120(const void* a_fp8,
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
                    "SM120 blockwise FP8 GEMM only supports block_size == ", kScaleGranularityK);

  auto arguments = MakeArguments(a_fp8, b_fp8, scale_a, scale_b, output_bf16, m, n, k);
  Gemm gemm;
  cutlass::Status status = gemm.can_implement(arguments);
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess,
                    "SM120 blockwise FP8 GEMM cannot implement the given problem: ",
                    cutlassGetStatusString(status));
  status = gemm.initialize(arguments, workspace, stream);
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess,
                    "SM120 blockwise FP8 GEMM initialize failed: ", cutlassGetStatusString(status));
  status = gemm.run(stream);
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess,
                    "SM120 blockwise FP8 GEMM run failed: ", cutlassGetStatusString(status));
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda

#endif  // !defined(DISABLE_FLOAT8_TYPES)