// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "contrib_ops/cuda/moe/moe_quantization.h"
#include <type_traits>
#include "cutlass/numeric_types.h"
#include "core/common/safeint.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"

#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"
#include "contrib_ops/cpu/utils/debug_macros.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      QMoE,                                                             \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .MayInplace(0, 0)                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),      \
      QMoE);

REGISTER_KERNEL_TYPED(MLFloat16)

#define QUICK_BUILD 1
#if QUICK_BUILD == 0
REGISTER_KERNEL_TYPED(BFloat16)
#endif

QMoE::QMoE(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info), MoEBase(op_kernel_info, GetDeviceProp()) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
              "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);

  this->block_size_ = op_kernel_info.GetAttrOrDefault<int64_t>("block_size", -1);

  using namespace onnxruntime::llm::kernels::cutlass_kernels;

  constexpr int kInputIndexFc3Weight = 8;
  has_fc3_ = op_kernel_info.GetInputCount() > kInputIndexFc3Weight && op_kernel_info.node().InputDefs()[kInputIndexFc3Weight]->Exists();

  int32_t input_type = op_kernel_info.node().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  bool is_fp16 = input_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16;
  is_fp16_ = is_fp16;

#if QUICK_BUILD
  if (is_fp16) {
    if (expert_weight_bits_ == 4) {
      m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, cutlass::uint4b_t, half>>(
          sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
    } else {  // expert_weight_bits_ == 8
      m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, uint8_t, half>>(
          sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
    }
  }
#else
  if (is_fp16) {
    if (expert_weight_bits_ == 4) {
      m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, cutlass::uint4b_t, half>>(
          sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
    } else {  // expert_weight_bits_ == 8
      m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, uint8_t, half>>(
          sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
    }
  } else {  // BFloat16
    if (expert_weight_bits_ == 4) {
      m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16>>(
          sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
    } else {  // expert_weight_bits_ == 8
      m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, uint8_t, __nv_bfloat16>>(
          sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
    }
  }
#endif
}

Status QMoE::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_scales = packed_fc1_scales_ ? nullptr : context->Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(4);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(5);
  const Tensor* fc2_scales = packed_fc2_scales_ ? nullptr : context->Input<Tensor>(6);
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(7);
  const Tensor* fc3_experts_weights_optional = context->Input<Tensor>(8);
  const Tensor* fc3_scales_optional = context->Input<Tensor>(9);
  const Tensor* fc3_experts_bias_optional = context->Input<Tensor>(10);

  const Tensor* fc1_zeros = packed_fc1_bias_ ? nullptr : context->Input<Tensor>(11);
  const Tensor* fc2_zeros = packed_fc2_bias_ ? nullptr : context->Input<Tensor>(12);
  const Tensor* fc3_zeros = context->Input<Tensor>(13);

  int64_t pack_size = expert_weight_bits_ == 4 ? 2 : 1;
  bool is_fused_swiglu = activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu;
  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(onnxruntime::contrib::moe_helper::CheckInputs(
      moe_params, input, router_probs, fc1_experts_weights,
      fc1_experts_bias_optional, fc1_scales, fc1_zeros,
      fc2_experts_weights, fc2_experts_bias_optional, fc2_scales, fc2_zeros,
      fc3_experts_weights_optional, fc3_experts_bias_optional, fc3_scales_optional, fc3_zeros,
      pack_size, is_fused_swiglu, block_size_));

  constexpr bool use_lora = false;
  constexpr bool use_deepseek_fp8_block_scale = false;
  constexpr bool min_latency_mode = false;
  bool use_awq = (fc1_zeros != nullptr) || (packed_fc1_bias_ != nullptr);
  onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig parallelism_config{};

  // Use profiler with proper weight type for quantized weights
  if (onnxruntime::llm::common::getEnvForceDeterministicMOE()) {
    auto tactics = m_moe_runner->getTactics();
    if (!tactics.empty()) {
      m_moe_runner->setTactic(tactics[0], tactics[0]);
    }
  } else {
    mGemmProfiler.setAllocator(this->Info().GetAllocator(OrtMemType::OrtMemTypeDefault));
    mGemmProfiler.setProfilerParams(static_cast<int>(moe_params.num_experts), static_cast<int>(k_),
                                    static_cast<int64_t>(moe_params.hidden_size), static_cast<int64_t>(moe_params.inter_size),
                                    static_cast<int64_t>(block_size_), activation_type_,
                                    false, false, false, true, parallelism_config, false, sm_);

    onnxruntime::llm::nvinfer::DataType dtype = is_fp16_ ? onnxruntime::llm::nvinfer::DataType::kHALF : onnxruntime::llm::nvinfer::DataType::kBF16;
    // Weight type: INT4 for 4-bit, INT8 for 8-bit quantization
    onnxruntime::llm::nvinfer::DataType wtype = (expert_weight_bits_ == 4)
                                                    ? onnxruntime::llm::nvinfer::DataType::kINT4
                                                    : onnxruntime::llm::nvinfer::DataType::kINT8;

    using onnxruntime::llm::kernels::cutlass_kernels::MoeGemmId;
    using onnxruntime::llm::kernels::weight_only::GemmDims;

    // For gated activations (SwiGLU), fc1_out_size is doubled
    int64_t fc1_out_size = static_cast<int64_t>(moe_params.inter_size);
    if (is_fused_swiglu) {
      fc1_out_size = static_cast<int64_t>(moe_params.inter_size) * 2;
    }

    // GEMM 1: N=fc1_out_size (doubled for gated), K=hidden_size
    MoeGemmId id1(static_cast<int>(fc1_out_size), static_cast<int>(moe_params.hidden_size), dtype, wtype, MoeGemmId::GemmType::Gemm1);
    if (mGemmId1 != id1) {
      mGemmId1 = id1;
      GemmDims dims(static_cast<int64_t>(moe_params.num_rows), static_cast<int64_t>(moe_params.num_rows),
                    fc1_out_size, static_cast<int64_t>(moe_params.hidden_size));
      mGemmProfiler.profileTactics(m_moe_runner.get(), dtype, dims, id1);
    }
    auto config1 = mGemmProfiler.getBestConfig(static_cast<int>(moe_params.num_rows), mGemmId1);

    // GEMM 2
    MoeGemmId id2(static_cast<int>(moe_params.hidden_size), static_cast<int>(moe_params.inter_size), dtype, wtype, MoeGemmId::GemmType::Gemm2);
    if (mGemmId2 != id2) {
      mGemmId2 = id2;
      GemmDims dims(static_cast<int64_t>(moe_params.num_rows), static_cast<int64_t>(moe_params.num_rows),
                    static_cast<int64_t>(moe_params.hidden_size), static_cast<int64_t>(moe_params.inter_size));
      mGemmProfiler.profileTactics(m_moe_runner.get(), dtype, dims, id2);
    }
    auto config2 = mGemmProfiler.getBestConfig(static_cast<int>(moe_params.num_rows), mGemmId2);

    m_moe_runner->setTactic(config1, config2);
  }

  size_t workspace_size = m_moe_runner->getWorkspaceSize(
      moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, k_,
      activation_type_, parallelism_config, use_lora, use_deepseek_fp8_block_scale, min_latency_mode, use_awq);

  // Scratch buffer for workspace + expert_scales + expert_indices
  // expert_scales: num_rows * k * sizeof(float)
  // expert_indices: num_rows * k * sizeof(int)
  size_t scales_bytes = moe_params.num_rows * k_ * sizeof(float);
  size_t indices_bytes = moe_params.num_rows * k_ * sizeof(int);
  size_t permutation_bytes = moe_params.num_rows * k_ * sizeof(int);
  size_t total_scratch_bytes = workspace_size + scales_bytes + indices_bytes + permutation_bytes;

  auto work_space = GetScratchBuffer<void>(total_scratch_bytes, context->GetComputeStream());
  char* workspace_ptr = reinterpret_cast<char*>(work_space.get());
  float* expert_scales = reinterpret_cast<float*>(workspace_ptr + workspace_size);
  int* expert_indices = reinterpret_cast<int*>(workspace_ptr + workspace_size + scales_bytes);
  int* unpermuted_row_to_permuted_row = reinterpret_cast<int*>(workspace_ptr + workspace_size + scales_bytes + indices_bytes);

  cudaStream_t stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());

  // Perform Softmax + TopK
  // Input router_probs is (num_rows, num_experts)
  bool is_fp16 = input->IsDataType<MLFloat16>();
  if (is_fp16) {
    LaunchSoftmaxTopK(
        reinterpret_cast<const half*>(router_probs->DataRaw()),
        expert_scales,
        expert_indices,
        static_cast<int>(moe_params.num_rows),
        static_cast<int>(moe_params.num_experts),
        static_cast<int>(k_),
        normalize_routing_weights_,
        stream);
  } else {
    // Fallback for float
    LaunchSoftmaxTopK(
        reinterpret_cast<const float*>(router_probs->DataRaw()),
        expert_scales,
        expert_indices,
        static_cast<int>(moe_params.num_rows),
        static_cast<int>(moe_params.num_experts),
        static_cast<int>(k_),
        normalize_routing_weights_,
        stream);
  }

  // Holders for packed tensors (if packing is needed for SwiGLU)
  IAllocatorUniquePtr<void> packed_fc1_scales_holder;
  IAllocatorUniquePtr<void> packed_fc1_zp_holder;
  IAllocatorUniquePtr<void> transposed_fc1_scales_holder;
  IAllocatorUniquePtr<void> transposed_fc2_scales_holder;
  IAllocatorUniquePtr<void> transposed_fc1_zp_holder;
  IAllocatorUniquePtr<void> transposed_fc2_zp_holder;

  // Determine effective pointers for scales and zero points
  const void* p_fc1_scales = nullptr;
  const void* p_fc1_zp = nullptr;
  const void* p_fc2_scales = nullptr;
  const void* p_fc2_zp = nullptr;

  // Use pre-packed buffers if available, otherwise use input tensors (and potentially compute bias on the fly)
  IAllocatorUniquePtr<void> transient_fc1_bias;
  IAllocatorUniquePtr<void> transient_fc2_bias;

  auto prepare_scale_zp = [&](const Tensor* scales, const Tensor* zeros,
                              const IAllocatorUniquePtr<void>& packed_scale, const IAllocatorUniquePtr<void>& packed_bias,
                              IAllocatorUniquePtr<void>& transient_bias,
                              const void*& eff_scale, const void*& eff_zp) {
    if (packed_scale) {
      eff_scale = packed_scale.get();
    } else if (scales) {
      eff_scale = scales->DataRaw();
    }

    if (packed_bias) {
      eff_zp = packed_bias.get();
    } else if (zeros) {
      if (expert_weight_bits_ == 4 || (expert_weight_bits_ == 8 && block_size_ > 0)) {
        // Compute bias on the fly: bias = -zp * scale
        // We need 'eff_scale' to be available.
        if (eff_scale && block_size_ > 0) {
          size_t num_elements = zeros->Shape().Size();
          // Determine type size based on scale type
          bool is_fp16 = scales->IsDataType<MLFloat16>();
          size_t bytes = num_elements * (is_fp16 ? 2 : 4);

          transient_bias = GetScratchBuffer<void>(bytes, context->GetComputeStream());
          eff_zp = transient_bias.get();

          const uint8_t* p_zp = static_cast<const uint8_t*>(zeros->DataRaw());

          // Transpose ZP if needed (for 3D ZP)
          auto shape = zeros->Shape();
          IAllocatorUniquePtr<void> temp_zp_transposed;
          if (shape.NumDimensions() == 3 && shape[2] > 1) {
            size_t rows = shape[1];   // N
            size_t cols = shape[2];   // Blocks
            size_t batch = shape[0];  // Experts
            size_t zp_bytes = zeros->SizeInBytes();
            temp_zp_transposed = GetScratchBuffer<void>(zp_bytes, context->GetComputeStream());
            LaunchQMoETranspose2D(p_zp, static_cast<uint8_t*>(temp_zp_transposed.get()), batch, rows, cols, stream);
            p_zp = static_cast<const uint8_t*>(temp_zp_transposed.get());
          }

          if (is_fp16) {
            if (expert_weight_bits_ == 8) {
              LaunchQMoEPrePackOffsetBias(
                  p_zp,
                  static_cast<const half*>(eff_scale),
                  static_cast<half*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  128.0f,
                  stream);
            } else {
              // 4-bit: Check if ZP is packed
              size_t scale_el = scales->Shape().Size();
              if (scale_el > num_elements * 3 / 2) {
                // Packed ZP (num_elements is ZP bytes, roughly scale_el / 2)
                int N_stride = static_cast<int>(zeros->Shape()[1]);
                LaunchQMoEPrePackPacked4BitZPKernel(
                    p_zp,
                    static_cast<const half*>(eff_scale),
                    static_cast<half*>(transient_bias.get()),
                    static_cast<int>(scale_el),
                    N_stride,
                    stream);
              } else {
                LaunchQMoEPrePackZP(
                    p_zp,
                    static_cast<const half*>(eff_scale),
                    static_cast<half*>(transient_bias.get()),
                    static_cast<int>(num_elements),
                    stream);
              }
            }
          } else {
            if (expert_weight_bits_ == 8) {
              LaunchQMoEPrePackOffsetBias(
                  p_zp,
                  static_cast<const float*>(eff_scale),
                  static_cast<float*>(transient_bias.get()),
                  static_cast<int>(num_elements),
                  128.0f,
                  stream);
            } else {
              // 4-bit
              size_t scale_el = scales->Shape().Size();
              if (scale_el > num_elements * 3 / 2) {
                int N_stride = static_cast<int>(zeros->Shape()[1]);
                LaunchQMoEPrePackPacked4BitZPKernel(
                    p_zp,
                    static_cast<const float*>(eff_scale),
                    static_cast<float*>(transient_bias.get()),
                    static_cast<int>(scale_el),
                    N_stride,
                    stream);
              } else {
                LaunchQMoEPrePackZP(
                    p_zp,
                    static_cast<const float*>(eff_scale),
                    static_cast<float*>(transient_bias.get()),
                    static_cast<int>(num_elements),
                    stream);
              }
            }
          }
        }
      } else {
        // For 8-bit, ZP is used as is (or transposed).
        // Since we are not packing, we use the raw pointer unless transpose is needed.
        // Transpose on the fly is tricky without allocation. BUT, ComputeInternal is usually called
        // with pre-packed weights/scales if coming from unit tests or offline tools.
        // If not pre-packed (e.g. dynamic graph), we might need to transpose if 3D.
        // For now, assuming standard path or 1D ZP for 2D weights.
        // If 3D, we must transpose.
        auto shape = zeros->Shape();
        if (shape.NumDimensions() == 3 && shape[2] > 1) {
          // Need temporary buffer for transpose
          size_t bytes = zeros->SizeInBytes();
          transient_bias = GetScratchBuffer<void>(bytes, context->GetComputeStream());
          eff_zp = transient_bias.get();

          size_t rows = shape[1];   // N
          size_t cols = shape[2];   // Blocks
          size_t batch = shape[0];  // Experts
          LaunchQMoETranspose2D(static_cast<const uint8_t*>(zeros->DataRaw()), static_cast<uint8_t*>(transient_bias.get()), batch, rows, cols, stream);
        } else {
          eff_zp = zeros->DataRaw();
        }
      }
    }
  };

  prepare_scale_zp(fc1_scales, fc1_zeros, packed_fc1_scales_, packed_fc1_bias_, transient_fc1_bias, p_fc1_scales, p_fc1_zp);
  prepare_scale_zp(fc2_scales, fc2_zeros, packed_fc2_scales_, packed_fc2_bias_, transient_fc2_bias, p_fc2_scales, p_fc2_zp);

  // DEBUG PRINTS (Preserved and fixed)
  // if (block_size_ > 0) {
  //   printf("QMoE Debug: block_size=%ld, expert_bits=%ld\n", block_size_, expert_weight_bits_);
  //   printf("QMoE Debug: FC1 Scales=%p, ZP=%p (PackedScales=%p, PackedBias=%p)\n", p_fc1_scales, p_fc1_zp, packed_fc1_scales_.get(), packed_fc1_bias_.get());
  // }

  onnxruntime::llm::kernels::cutlass_kernels::QuantParams quant_params;
  if (block_size_ > 0) {
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::GroupWise(
        block_size_,
        p_fc1_scales,
        p_fc2_scales,
        nullptr,
        nullptr,
        p_fc1_zp,
        p_fc2_zp);
  } else {
    // Per-column quantization
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::Int(
        p_fc1_scales,
        p_fc2_scales);
  }

  Tensor* output = context->Output(0, input->Shape());

  onnxruntime::llm::kernels::LoraParams lora_params;
  onnxruntime::llm::kernels::cutlass_kernels::MoeMinLatencyParams min_latency_params;

  m_moe_runner->runMoe(
      input->DataRaw(),
      nullptr,
      expert_indices,
      expert_scales,
      fc1_experts_weights->DataRaw(),
      fc1_experts_bias_optional ? fc1_experts_bias_optional->DataRaw() : nullptr,
      activation_type_,
      fc2_experts_weights->DataRaw(),
      fc2_experts_bias_optional ? fc2_experts_bias_optional->DataRaw() : nullptr,
      quant_params,
      moe_params.num_rows,
      moe_params.hidden_size,
      moe_params.inter_size,
      moe_params.num_experts,
      k_,
      workspace_ptr,
      output->MutableDataRaw(),
      unpermuted_row_to_permuted_row,
      parallelism_config,
      false,  // enable_alltoall
      use_lora,
      lora_params,
      use_deepseek_fp8_block_scale,
      min_latency_mode,
      min_latency_params,
      [&]() {
        onnxruntime::llm::kernels::cutlass_kernels::ActivationParams params(activation_type_);
        params.alpha = activation_alpha_;
        params.beta = activation_beta_;
        params.swiglu_fusion = swiglu_fusion_;
        params.limit = swiglu_limit_;
        return params;
      }(),
      stream);

  return Status::OK();
}

Status QMoE::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                     bool& is_packed, PrePackedWeights* prepacked_weights) {
  is_packed = false;

  cudaStream_t stream = 0;  // Use default stream for PrePack operations

  // Scale/Bias layout is [Experts, Blocks, N] in cutlass kernel
  // But passed from Python as [Experts, N, Blocks] for block-wise (3D)
  // For per-column (2D), it is [Experts, N], which is effectively [Experts, 1, N] (compatible with [Experts, Blocks, N] where Blocks=1)
  // So we only transpose if 3D.

  auto TransposeAndPack = [&](IAllocatorUniquePtr<void>& packed_buf) {
    auto shape = tensor.Shape();
    size_t bytes = tensor.SizeInBytes();
    packed_buf = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

    const void* p_src = tensor.DataRaw();
    IAllocatorUniquePtr<void> temp_src_gpu;
    if (tensor.Location().device.Type() == OrtDevice::CPU) {
      temp_src_gpu = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);
      cudaMemcpyAsync(temp_src_gpu.get(), p_src, bytes, cudaMemcpyDefault, stream);
      p_src = temp_src_gpu.get();
    }

    if (shape.NumDimensions() == 3 && shape[2] > 1) {
      size_t rows = shape[1];   // N
      size_t cols = shape[2];   // Blocks
      size_t batch = shape[0];  // Experts
      auto type = tensor.DataType();
      if (type == DataTypeImpl::GetType<MLFloat16>()) {
        LaunchQMoETranspose2D(static_cast<const half*>(p_src), static_cast<half*>(packed_buf.get()), batch, rows, cols, stream);
      } else if (type == DataTypeImpl::GetType<float>()) {
        LaunchQMoETranspose2D(static_cast<const float*>(p_src), static_cast<float*>(packed_buf.get()), batch, rows, cols, stream);
      } else if (type == DataTypeImpl::GetType<uint8_t>()) {
        LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src), static_cast<uint8_t*>(packed_buf.get()), batch, rows, cols, stream);
      } else {
        ORT_THROW("Unsupported data type for scale transposition");
      }
    } else {
      // 2D case or others: Direct Copy
      cudaMemcpyAsync(packed_buf.get(), p_src, bytes, cudaMemcpyDefault, stream);
    }

    cudaStreamSynchronize(stream);
    is_packed = true;
  };

  auto compute_bias = [&](const IAllocatorUniquePtr<void>& packed_scale, IAllocatorUniquePtr<void>& packed_bias) {
    // If not computing bias (e.g. 8-bit ZP), we might not need scales at all, but we check anyway.
    if ((expert_weight_bits_ == 4) && !packed_scale) return;

    size_t num_elements = tensor.Shape().Size();
    auto shape = tensor.Shape();

    // For 8-bit: packed_bias holds the ZP (uint8) for column-wise, OR pre-computed bias (float/half) for block-wise.
    // If block_size > 0, we need to compute bias = -ZP * Scale, similar to 4-bit case.

    if (expert_weight_bits_ == 8) {
      // For 8-bit: packed_bias holds the ZP (uint8) for column-wise, OR pre-computed bias (float/half) for block-wise.
      // If block_size > 0, we need to compute bias = -ZP * Scale, similar to 4-bit case.

      if (block_size_ > 0) {
        // Block-wise: Compute bias = -ZP * Scale
        auto type = Info().node().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
        bool is_fp16 = (type == TensorProto_DataType_FLOAT16);
        size_t bytes = num_elements * (is_fp16 ? 2 : 4);
        packed_bias = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

        const void* p_src_zp = tensor.DataRaw();
        IAllocatorUniquePtr<void> temp_zp_gpu;
        if (tensor.Location().device.Type() == OrtDevice::CPU) {
          temp_zp_gpu = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
          cudaMemcpyAsync(temp_zp_gpu.get(), p_src_zp, tensor.SizeInBytes(), cudaMemcpyDefault, stream);
          p_src_zp = temp_zp_gpu.get();
        }

        const void* p_zp_for_calc = p_src_zp;
        IAllocatorUniquePtr<void> temp_zp_transposed;

        if (shape.NumDimensions() == 3 && shape[2] > 1) {
          size_t rows = shape[1];   // N
          size_t cols = shape[2];   // Blocks
          size_t batch = shape[0];  // Experts

          // Transpose ZP to match Scale layout [Experts, Blocks, N]
          temp_zp_transposed = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
          LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src_zp), static_cast<uint8_t*>(temp_zp_transposed.get()), batch, rows, cols, stream);
          p_zp_for_calc = temp_zp_transposed.get();
        }

        if (is_fp16) {
          LaunchQMoEPrePackOffsetBias(static_cast<const uint8_t*>(p_zp_for_calc), static_cast<const half*>(packed_scale.get()), static_cast<half*>(packed_bias.get()), num_elements, 128.0f, stream);
        } else {
          LaunchQMoEPrePackOffsetBias(static_cast<const uint8_t*>(p_zp_for_calc), static_cast<const float*>(packed_scale.get()), static_cast<float*>(packed_bias.get()), num_elements, 128.0f, stream);
        }
      } else {
        // For 8-bit per-column: packed_bias holds the ZP (uint8), possibly transposed.
        // Current QuantParams::Int takes scales and ignores ZP for per-column usually,
        // but let's keep it consistent with previous logic just in case.
        size_t bytes = num_elements * sizeof(uint8_t);
        packed_bias = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

        const void* p_src_zp = tensor.DataRaw();
        IAllocatorUniquePtr<void> temp_zp_gpu;
        if (tensor.Location().device.Type() == OrtDevice::CPU) {
          temp_zp_gpu = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
          cudaMemcpyAsync(temp_zp_gpu.get(), p_src_zp, tensor.SizeInBytes(), cudaMemcpyDefault, stream);
          p_src_zp = temp_zp_gpu.get();
        }

        if (shape.NumDimensions() == 3 && shape[2] > 1) {
          size_t rows = shape[1];   // N
          size_t cols = shape[2];   // Blocks
          size_t batch = shape[0];  // Experts
          LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src_zp), static_cast<uint8_t*>(packed_bias.get()), batch, rows, cols, stream);
        } else {
          cudaMemcpyAsync(packed_bias.get(), p_src_zp, bytes, cudaMemcpyDefault, stream);
        }
      }
    } else {
      // For 4-bit: packed_bias holds floating point bias = -ZP * Scale
      auto type = Info().node().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
      bool is_fp16 = (type == TensorProto_DataType_FLOAT16);

      // Assume packed ZP (2 elements per byte) for 4-bit PrePack scenarios
      size_t zp_bytes = num_elements;
      size_t output_count = zp_bytes * 2;
      size_t bytes = output_count * (is_fp16 ? 2 : 4);
      packed_bias = IAllocator::MakeUniquePtr<void>(alloc, bytes, true);

      const void* p_src_zp = tensor.DataRaw();
      IAllocatorUniquePtr<void> temp_zp_gpu;
      if (tensor.Location().device.Type() == OrtDevice::CPU) {
        temp_zp_gpu = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
        cudaMemcpyAsync(temp_zp_gpu.get(), p_src_zp, tensor.SizeInBytes(), cudaMemcpyDefault, stream);
        p_src_zp = temp_zp_gpu.get();
      }

      const void* p_zp_for_calc = p_src_zp;
      IAllocatorUniquePtr<void> temp_zp_transposed;

      if (shape.NumDimensions() == 3 && shape[2] > 1) {
        size_t rows = shape[1];   // N
        size_t cols = shape[2];   // Blocks
        size_t batch = shape[0];  // Experts

        // Transpose ZP to match layout [Experts, Blocks, N]
        // Note: For packed ZP, 'cols' is packed columns. Transpose handles bytes.
        temp_zp_transposed = IAllocator::MakeUniquePtr<void>(alloc, tensor.SizeInBytes(), true);
        LaunchQMoETranspose2D(static_cast<const uint8_t*>(p_src_zp), static_cast<uint8_t*>(temp_zp_transposed.get()), batch, rows, cols, stream);
        p_zp_for_calc = temp_zp_transposed.get();
      }

      int N_stride = static_cast<int>(shape[1]);  // N from ZP input shape (rows)

      if (is_fp16) {
        LaunchQMoEPrePackPacked4BitZPKernel(static_cast<const uint8_t*>(p_zp_for_calc), static_cast<const half*>(packed_scale.get()), static_cast<half*>(packed_bias.get()), static_cast<int>(output_count), N_stride, stream);
      } else {
        LaunchQMoEPrePackPacked4BitZPKernel(static_cast<const uint8_t*>(p_zp_for_calc), static_cast<const float*>(packed_scale.get()), static_cast<float*>(packed_bias.get()), static_cast<int>(output_count), N_stride, stream);
      }
    }
    cudaStreamSynchronize(stream);
    is_packed = true;
  };

  DUMP_TENSOR_INIT();

#if DUMP_TENSOR_LEVEL >= 1
  auto dump_tensor = [&](const char* name, const IAllocatorUniquePtr<void>& packed_scales, const Tensor& scales) {
    auto shape = scales.Shape();
    if (shape.NumDimensions() == 3 && is_fp16_) {
      size_t rows = shape[1];   // N
      size_t cols = shape[2];   // Blocks
      size_t batch = shape[0];  // Experts
      DUMP_TENSOR(name, static_cast<const half*>(packed_scales.get()), int(batch), int(cols), int(rows));
    }
  };
#define DUMP_PACK_TENSOR(name, packed_scales, scales) dump_tensor(name, packed_scales, scales)
#else
#define DUMP_PACK_TENSOR(name, packed_scales, scales)
#endif

  if (input_idx == 3) {  // fc1_scales
    DUMP_TENSOR("fc1_scales", tensor);
    TransposeAndPack(packed_fc1_scales_);
    DUMP_PACK_TENSOR("packed_fc1_scales", packed_fc1_scales_, tensor);
  } else if (input_idx == 6) {  // fc2_scales
    DUMP_TENSOR("fc2_scales", tensor);
    TransposeAndPack(packed_fc2_scales_);
    DUMP_PACK_TENSOR("packed_fc2_scales", packed_fc2_scales_, tensor);
  } else if (input_idx == 9 && has_fc3_) {  // fc3_scales
    DUMP_TENSOR("fc3_scales", tensor);
    TransposeAndPack(packed_fc3_scales_);
    DUMP_PACK_TENSOR("packed_fc3_scales", packed_fc3_scales_, tensor);
  } else if (input_idx == 11) {  // fc1_zeros
    // DUMP_TENSOR("fc1_zeros", tensor);
    compute_bias(packed_fc1_scales_, packed_fc1_bias_);
    // DUMP_PACK_TENSOR("packed_fc1_bias", packed_fc1_bias_, tensor);
  } else if (input_idx == 12) {  // fc2_zeros
    // DUMP_TENSOR("fc2_zeros", tensor);
    compute_bias(packed_fc2_scales_, packed_fc2_bias_);
    // DUMP_PACK_TENSOR("packed_fc2_bias", packed_fc2_bias_, tensor);
  }
  // TODO: fc3_zeros (13) not handled for now as it's optional and rarely used?
  // Code structure allows adding it easily.

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
