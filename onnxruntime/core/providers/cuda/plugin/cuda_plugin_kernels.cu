// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_plugin_kernels.h"
#include "cuda_stream_plugin.h"

#include <cstring>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
namespace cuda_plugin {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

CudaSyncStream* GetCudaSyncStream(const Ort::KernelContext& ctx) {
  void* stream = ctx.GetGPUComputeStream();
  if (!stream) return nullptr;
  return CudaSyncStream::FromCudaStream(static_cast<cudaStream_t>(stream));
}

}  // namespace

// ---------------------------------------------------------------------------
// Relu Kernel Implementation
// ---------------------------------------------------------------------------

struct ReluKernelImpl : public OrtKernelImpl {
  ReluKernelImpl() : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr,
                                             OrtKernelContext* context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<ReluKernelImpl*>(this_ptr);
  }
};

// Simple CUDA Relu kernel
__global__ void ReluKernelCuda(const float* input, float* output, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    output[idx] = input[idx] > 0.0f ? input[idx] : 0.0f;
  }
}

/*static*/
OrtStatus* ORT_API_CALL ReluKernelImpl::ComputeImpl(
    OrtKernelImpl* /*this_ptr*/, OrtKernelContext* context) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  Ort::KernelContext ctx{context};
  Ort::ConstValue input = ctx.GetInput(0);
  auto shape_info = input.GetTensorTypeAndShapeInfo();
  auto shape = shape_info.GetShape();
  size_t count = shape_info.GetElementCount();

  Ort::UnownedValue output = ctx.GetOutput(0, shape);

  const float* input_data = input.GetTensorData<float>();
  float* output_data = output.GetTensorMutableData<float>();

  if (count > 0) {
    // Get CUDA stream from kernel context
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.GetGPUComputeStream());

    const int block_size = 256;
    const int grid_size = static_cast<int>((count + block_size - 1) / block_size);
    ReluKernelCuda<<<grid_size, block_size, 0, stream>>>(input_data, output_data, count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("CUDA Relu kernel launch failed: ") + cudaGetErrorString(err)).c_str());
    }
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

// ---------------------------------------------------------------------------
// Add Kernel Implementation
// ---------------------------------------------------------------------------

struct AddKernelImpl : public OrtKernelImpl {
  AddKernelImpl() : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr,
                                             OrtKernelContext* context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<AddKernelImpl*>(this_ptr);
  }
};

__global__ void AddKernelCuda(const float* a, const float* b, float* c, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    c[idx] = a[idx] + b[idx];
  }
}

/*static*/
OrtStatus* ORT_API_CALL AddKernelImpl::ComputeImpl(
    OrtKernelImpl* /*this_ptr*/, OrtKernelContext* context) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  Ort::KernelContext ctx{context};
  Ort::ConstValue input_a = ctx.GetInput(0);
  Ort::ConstValue input_b = ctx.GetInput(1);

  auto shape_info = input_a.GetTensorTypeAndShapeInfo();
  auto shape = shape_info.GetShape();
  size_t count = shape_info.GetElementCount();

  Ort::UnownedValue output = ctx.GetOutput(0, shape);

  const float* a_data = input_a.GetTensorData<float>();
  const float* b_data = input_b.GetTensorData<float>();
  float* c_data = output.GetTensorMutableData<float>();

  if (count > 0) {
    cudaStream_t stream = static_cast<cudaStream_t>(ctx.GetGPUComputeStream());

    const int block_size = 256;
    const int grid_size = static_cast<int>((count + block_size - 1) / block_size);
    AddKernelCuda<<<grid_size, block_size, 0, stream>>>(a_data, b_data, c_data, count);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, cudaGetErrorString(err));
    }
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

// ---------------------------------------------------------------------------
// MatMul Kernel Implementation
// ---------------------------------------------------------------------------

struct MatMulKernelImpl : public OrtKernelImpl {
  MatMulKernelImpl() : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr,
                                             OrtKernelContext* context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<MatMulKernelImpl*>(this_ptr);
  }
};

/*static*/
OrtStatus* ORT_API_CALL MatMulKernelImpl::ComputeImpl(
    OrtKernelImpl* /*this_ptr*/, OrtKernelContext* context) noexcept {
  EXCEPTION_TO_STATUS_BEGIN

  Ort::KernelContext ctx{context};
  Ort::ConstValue input_a = ctx.GetInput(0);
  Ort::ConstValue input_b = ctx.GetInput(1);

  auto shape_info_a = input_a.GetTensorTypeAndShapeInfo();
  auto shape_info_b = input_b.GetTensorTypeAndShapeInfo();
  auto shape_a = shape_info_a.GetShape();
  auto shape_b = shape_info_b.GetShape();

  // MatMul: [M, K] x [K, N] -> [M, N]
  int M = static_cast<int>(shape_a[0]);
  int K = static_cast<int>(shape_a[1]);
  int N = static_cast<int>(shape_b[1]);

  std::vector<int64_t> output_shape = {M, N};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);

  const float* a_data = input_a.GetTensorData<float>();
  const float* b_data = input_b.GetTensorData<float>();
  float* y_data = output.GetTensorMutableData<float>();

  if (M > 0 && N > 0 && K > 0) {
    CudaSyncStream* stream_impl = GetCudaSyncStream(ctx);
    if (!stream_impl) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "Failed to get CUDA stream");
    }

    cublasHandle_t cublas_handle = stream_impl->GetCublasHandle();

    float alpha = 1.0f;
    float beta = 0.0f;

    CUBLAS_RETURN_IF_ERROR(cublasSgemm(cublas_handle,
                                       CUBLAS_OP_N, CUBLAS_OP_N,
                                       N, M, K,
                                       &alpha,
                                       b_data, N,
                                       a_data, K,
                                       &beta,
                                       y_data, N));
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

// ---------------------------------------------------------------------------
// Gemm Kernel Implementation
// ---------------------------------------------------------------------------

struct GemmKernelImpl : public OrtKernelImpl {
  GemmKernelImpl(const OrtKernelInfo* info) : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;

    OrtStatus* status = Ort::GetApi().KernelInfoGetAttribute_float(info, "alpha", &alpha_);
    if (status != nullptr) {
      alpha_ = 1.0f;
      Ort::GetApi().ReleaseStatus(status);
    }
    status = Ort::GetApi().KernelInfoGetAttribute_float(info, "beta", &beta_);
    if (status != nullptr) {
      beta_ = 1.0f;
      Ort::GetApi().ReleaseStatus(status);
    }
    int64_t tA = 0;
    status = Ort::GetApi().KernelInfoGetAttribute_int64(info, "transA", &tA);
    if (status != nullptr) {
      trans_a_ = 0;
      Ort::GetApi().ReleaseStatus(status);
    } else {
      trans_a_ = static_cast<int>(tA);
    }
    int64_t tB = 0;
    status = Ort::GetApi().KernelInfoGetAttribute_int64(info, "transB", &tB);
    if (status != nullptr) {
      trans_b_ = 0;
      Ort::GetApi().ReleaseStatus(status);
    } else {
      trans_b_ = static_cast<int>(tB);
    }
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr,
                                             OrtKernelContext* context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<GemmKernelImpl*>(this_ptr);
  }

 private:
  float alpha_;
  float beta_;
  int trans_a_;
  int trans_b_;
};

/*static*/
OrtStatus* ORT_API_CALL GemmKernelImpl::ComputeImpl(
    OrtKernelImpl* this_ptr, OrtKernelContext* context) noexcept {
  auto* self = static_cast<GemmKernelImpl*>(this_ptr);
  EXCEPTION_TO_STATUS_BEGIN
  Ort::KernelContext ctx{context};
  Ort::ConstValue input_a = ctx.GetInput(0);
  Ort::ConstValue input_b = ctx.GetInput(1);

  auto shape_a = input_a.GetTensorTypeAndShapeInfo().GetShape();
  auto shape_b = input_b.GetTensorTypeAndShapeInfo().GetShape();

  int M = static_cast<int>(self->trans_a_ == 0 ? shape_a[0] : shape_a[1]);
  int K = static_cast<int>(self->trans_a_ == 0 ? shape_a[1] : shape_a[0]);
  int N = static_cast<int>(self->trans_b_ == 0 ? shape_b[1] : shape_b[0]);

  std::vector<int64_t> output_shape = {M, N};
  Ort::UnownedValue output = ctx.GetOutput(0, output_shape);

  const float* a_data = input_a.GetTensorData<float>();
  const float* b_data = input_b.GetTensorData<float>();
  float* y_data = output.GetTensorMutableData<float>();

  if (M > 0 && N > 0 && K > 0) {
    CudaSyncStream* stream_impl = GetCudaSyncStream(ctx);
    if (!stream_impl) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "Failed to get CUDA stream");
    }

    cublasHandle_t cublas_handle = stream_impl->GetCublasHandle();

    // Handle optional bias C
    if (ctx.GetInputCount() > 2) {
      Ort::ConstValue input_c = ctx.GetInput(2);
      const float* c_data = input_c.GetTensorData<float>();
      // Copy C to output initially if beta != 0
      if (self->beta_ != 0.0f) {
        // Gemm spec says C can be scalar, [N], [M, 1] or [M, N].
        // For now we assume [M, N] or [N] (broadcast row).
        // To simplify, we'll just support [M, N] or broadcast manually if needed.
        // cuBLAS sgemm does Y = alpha*op(A)*op(B) + beta*C.
        // If we want to use the output as C, we must initialize it with C data.
        auto shape_c = input_c.GetTensorTypeAndShapeInfo().GetShape();
        if (shape_c.size() == 2 && shape_c[0] == M && shape_c[1] == N) {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(y_data, c_data, M * N * sizeof(float), cudaMemcpyDeviceToDevice, stream_impl->GetCudaStream()));
        } else if (shape_c.size() == 1 && shape_c[0] == N) {
          // Broadcast [N] to [M, N]
          for (int i = 0; i < M; ++i) {
            CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(y_data + i * N, c_data, N * sizeof(float), cudaMemcpyDeviceToDevice, stream_impl->GetCudaStream()));
          }
        } else {
          // Fallback - just zero if unsupported broadcast
          CUDA_RETURN_IF_ERROR(cudaMemsetAsync(y_data, 0, M * N * sizeof(float), stream_impl->GetCudaStream()));
        }
      } else {
        CUDA_RETURN_IF_ERROR(cudaMemsetAsync(y_data, 0, M * N * sizeof(float), stream_impl->GetCudaStream()));
      }
    } else {
      CUDA_RETURN_IF_ERROR(cudaMemsetAsync(y_data, 0, M * N * sizeof(float), stream_impl->GetCudaStream()));
    }

    cublasOperation_t transA = self->trans_a_ == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t transB = self->trans_b_ == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;

    // Row-major A[M,K], B[K,N] -> C[M,N]
    // cuBLAS (col-major): C = alpha * op(B) * op(A) + beta * C
    // op(B) is [N, K] in col-major (if transB=0) or [K, N] (if transB=1)
    // op(A) is [K, M] in col-major (if transA=0) or [M, K] (if transA=1)

    int lda = (self->trans_a_ == 0) ? K : M;
    int ldb = (self->trans_b_ == 0) ? N : K;
    int ldc = N;

    CUBLAS_RETURN_IF_ERROR(cublasSgemm(cublas_handle,
                                       transB, transA,
                                       N, M, K,
                                       &self->alpha_,
                                       b_data, ldb,
                                       a_data, lda,
                                       &self->beta_,
                                       y_data, ldc));
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

// ---------------------------------------------------------------------------
// Conv Kernel Implementation
// ---------------------------------------------------------------------------

struct ConvKernelImpl : public OrtKernelImpl {
  ConvKernelImpl(const OrtKernelInfo* info) : OrtKernelImpl{} {
    ort_version_supported = ORT_API_VERSION;
    flags = 0;
    Compute = ComputeImpl;
    Release = ReleaseImpl;
    PrePackWeight = nullptr;
    SetSharedPrePackedWeight = nullptr;

    Ort::ConstKernelInfo k_info{info};
    try {
      pads_ = k_info.GetAttributes<int64_t>("pads");
    } catch (...) {
      pads_ = {0, 0, 0, 0};
    }
    try {
      strides_ = k_info.GetAttributes<int64_t>("strides");
    } catch (...) {
      strides_ = {1, 1};
    }
    try {
      dilations_ = k_info.GetAttributes<int64_t>("dilations");
    } catch (...) {
      dilations_ = {1, 1};
    }
    try {
      group_ = k_info.GetAttribute<int64_t>("group");
    } catch (...) {
      group_ = 1;
    }

    cudnnCreateTensorDescriptor(&x_desc_);
    cudnnCreateTensorDescriptor(&y_desc_);
    cudnnCreateFilterDescriptor(&w_desc_);
    cudnnCreateConvolutionDescriptor(&conv_desc_);
  }

  ~ConvKernelImpl() {
    cudnnDestroyTensorDescriptor(x_desc_);
    cudnnDestroyTensorDescriptor(y_desc_);
    cudnnDestroyFilterDescriptor(w_desc_);
    cudnnDestroyConvolutionDescriptor(conv_desc_);
  }

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr,
                                             OrtKernelContext* context) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
    delete static_cast<ConvKernelImpl*>(this_ptr);
  }

 private:
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> dilations_;
  int64_t group_;

  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
};

/*static*/
OrtStatus* ORT_API_CALL ConvKernelImpl::ComputeImpl(
    OrtKernelImpl* this_ptr, OrtKernelContext* context) noexcept {
  auto* self = static_cast<ConvKernelImpl*>(this_ptr);
  EXCEPTION_TO_STATUS_BEGIN

  Ort::KernelContext ctx{context};
  Ort::ConstValue input_x = ctx.GetInput(0);
  Ort::ConstValue input_w = ctx.GetInput(1);

  auto shape_x = input_x.GetTensorTypeAndShapeInfo().GetShape();
  auto shape_w = input_w.GetTensorTypeAndShapeInfo().GetShape();

  int n = static_cast<int>(shape_x[0]);
  int c = static_cast<int>(shape_x[1]);
  int h = static_cast<int>(shape_x.size() > 2 ? shape_x[2] : 1);
  int w = static_cast<int>(shape_x.size() > 3 ? shape_x[3] : 1);

  int m = static_cast<int>(shape_w[0]);
  int wc = static_cast<int>(shape_w[1]);
  int kh = static_cast<int>(shape_w.size() > 2 ? shape_w[2] : 1);
  int kw = static_cast<int>(shape_w.size() > 3 ? shape_w[3] : 1);

  int out_h = static_cast<int>((h + self->pads_[0] + self->pads_[2] - self->dilations_[0] * (kh - 1) - 1) / self->strides_[0] + 1);
  int out_w = static_cast<int>((w + self->pads_[1] + self->pads_[3] - self->dilations_[1] * (kw - 1) - 1) / self->strides_[1] + 1);

  std::vector<int64_t> output_shape = {n, m, out_h, out_w};
  Ort::UnownedValue output_y = ctx.GetOutput(0, output_shape);

  CudaSyncStream* stream_impl = GetCudaSyncStream(ctx);
  if (!stream_impl) {
    return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "Failed to get CUDA stream");
  }

  cudnnHandle_t cudnn = stream_impl->GetCudnnHandle();

  cudnnSetTensor4dDescriptor(self->x_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w);
  cudnnSetTensor4dDescriptor(self->y_desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, m, out_h, out_w);
  cudnnSetFilter4dDescriptor(self->w_desc_, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, m, wc, kh, kw);

  cudnnSetConvolution2dDescriptor(self->conv_desc_,
                                  static_cast<int>(self->pads_[0]), static_cast<int>(self->pads_[1]),
                                  static_cast<int>(self->strides_[0]), static_cast<int>(self->strides_[1]),
                                  static_cast<int>(self->dilations_[0]), static_cast<int>(self->dilations_[1]),
                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  cudnnSetConvolutionGroupCount(self->conv_desc_, static_cast<int>(self->group_));

  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

  size_t workspace_size = 0;
  cudnnGetConvolutionForwardWorkspaceSize(cudnn, self->x_desc_, self->w_desc_, self->conv_desc_, self->y_desc_, algo, &workspace_size);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    cudaMallocAsync(&workspace, workspace_size, stream_impl->GetCudaStream());
  }

  const float alpha = 1.0f, beta = 0.0f;
  const float* x_data = input_x.GetTensorData<float>();
  const float* w_data = input_w.GetTensorData<float>();
  float* y_data = output_y.GetTensorMutableData<float>();

  cudnnConvolutionForward(cudnn, &alpha, self->x_desc_, x_data,
                          self->w_desc_, w_data, self->conv_desc_, algo,
                          workspace, workspace_size, &beta, self->y_desc_, y_data);

  if (workspace_size > 0) {
    cudaFreeAsync(workspace, stream_impl->GetCudaStream());
  }

  if (ctx.GetInputCount() > 2) {
    Ort::ConstValue input_b = ctx.GetInput(2);
    const float* b_data = input_b.GetTensorData<float>();
    cudnnTensorDescriptor_t b_desc;
    cudnnCreateTensorDescriptor(&b_desc);
    cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, m, 1, 1);
    const float alpha_b = 1.0f, beta_b = 1.0f;
    cudnnAddTensor(cudnn, &alpha_b, b_desc, b_data, &beta_b, self->y_desc_, y_data);
    cudnnDestroyTensorDescriptor(b_desc);
  }

  return nullptr;

  EXCEPTION_TO_STATUS_END
}

namespace {

struct GeneratedKernelRegistration {
  const char* op_type;
  int since_version_start;
  int since_version_end;
  const char* domain;
  int registration_id;
  const char* constraint_name;
  ONNXTensorElementDataType type_constraint;
};

using PluginKernelCreateFn = OrtStatus*(ORT_API_CALL*)(void*, const OrtKernelInfo*, OrtKernelImpl**) noexcept;

OrtStatus* ORT_API_CALL CreateReluKernel(void* /*state*/,
                                         const OrtKernelInfo* /*info*/,
                                         OrtKernelImpl** kernel_out) noexcept {
  *kernel_out = new ReluKernelImpl();
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateAddKernel(void* /*state*/,
                                        const OrtKernelInfo* /*info*/,
                                        OrtKernelImpl** kernel_out) noexcept {
  *kernel_out = new AddKernelImpl();
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateMatMulKernel(void* /*state*/,
                                           const OrtKernelInfo* /*info*/,
                                           OrtKernelImpl** kernel_out) noexcept {
  *kernel_out = new MatMulKernelImpl();
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateGemmKernel(void* /*state*/,
                                         const OrtKernelInfo* info,
                                         OrtKernelImpl** kernel_out) noexcept {
  *kernel_out = new GemmKernelImpl(info);
  return nullptr;
}

OrtStatus* ORT_API_CALL CreateConvKernel(void* /*state*/,
                                         const OrtKernelInfo* info,
                                         OrtKernelImpl** kernel_out) noexcept {
  *kernel_out = new ConvKernelImpl(info);
  return nullptr;
}

PluginKernelCreateFn GetCreateFnForOp(std::string_view op_type) {
  if (op_type == "Relu") return CreateReluKernel;
  if (op_type == "Add") return CreateAddKernel;
  if (op_type == "MatMul") return CreateMatMulKernel;
  if (op_type == "Gemm") return CreateGemmKernel;
  if (op_type == "Conv") return CreateConvKernel;
  return nullptr;
}

constexpr GeneratedKernelRegistration kGeneratedKernelRegistrations[] = {
#include "core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc"
#include "core/providers/cuda/plugin/cuda_plugin_generated_contrib_registrations.inc"
};

}  // namespace

OrtStatus* CreateCudaKernelRegistry(const OrtEpApi& ep_api,
                                    const char* ep_name,
                                    void* /*create_kernel_state*/,
                                    OrtKernelRegistry** out_registry) {
  *out_registry = nullptr;

  EXCEPTION_TO_STATUS_BEGIN

  Ort::KernelRegistry registry;

  std::vector<const OrtDataType*> type_constraint;
  for (size_t i = 0; i < std::size(kGeneratedKernelRegistrations);) {
    const auto& reg = kGeneratedKernelRegistrations[i];
    PluginKernelCreateFn create_fn = GetCreateFnForOp(reg.op_type);
    if (!create_fn) {
      // Skip all rows in this registration group.
      const auto group_id = reg.registration_id;
      const auto* group_op = reg.op_type;
      const auto* group_domain = reg.domain;
      const auto group_start = reg.since_version_start;
      const auto group_end = reg.since_version_end;
      while (i < std::size(kGeneratedKernelRegistrations)) {
        const auto& row = kGeneratedKernelRegistrations[i];
        if (row.registration_id != group_id ||
            row.since_version_start != group_start ||
            row.since_version_end != group_end ||
            std::strcmp(row.op_type, group_op) != 0 ||
            std::strcmp(row.domain, group_domain) != 0) {
          break;
        }
        ++i;
      }
      continue;
    }

    Ort::KernelDefBuilder builder;
    const auto* domain = reg.domain;
    builder.SetOperatorType(reg.op_type)
        .SetDomain(domain == nullptr ? "" : domain)
        .SetSinceVersion(reg.since_version_start, reg.since_version_end)
        .SetExecutionProvider(ep_name);

    const auto group_id = reg.registration_id;
    const auto* group_op = reg.op_type;
    const auto* group_domain = reg.domain;
    const auto group_start = reg.since_version_start;
    const auto group_end = reg.since_version_end;

    std::unordered_map<std::string, std::vector<const OrtDataType*>> constraints_by_name;
    while (i < std::size(kGeneratedKernelRegistrations)) {
      const auto& row = kGeneratedKernelRegistrations[i];
      if (row.registration_id != group_id ||
          row.since_version_start != group_start ||
          row.since_version_end != group_end ||
          std::strcmp(row.op_type, group_op) != 0 ||
          std::strcmp(row.domain, group_domain) != 0) {
        break;
      }

      if (row.constraint_name != nullptr &&
          row.constraint_name[0] != '\0' &&
          row.type_constraint != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        const OrtDataType* data_type = nullptr;
        RETURN_IF_ERROR(ep_api.GetTensorDataType(row.type_constraint, &data_type));
        constraints_by_name[row.constraint_name].push_back(data_type);
      }

      ++i;
    }

    for (auto& kv : constraints_by_name) {
      auto& constraint_name = kv.first;
      auto& constraint_types = kv.second;
      if (constraint_types.empty()) {
        continue;
      }
      const OrtDataType* data_type = nullptr;
      type_constraint.clear();
      for (const auto* t : constraint_types) {
        data_type = t;
        type_constraint.push_back(data_type);
      }
      builder.AddTypeConstraint(constraint_name.c_str(), type_constraint);
    }

    Ort::KernelDef kernel_def = builder.Build();
    RETURN_IF_ERROR(registry.AddKernel(kernel_def.release(), create_fn, nullptr));
  }

  *out_registry = registry.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
