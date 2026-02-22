// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_plugin_kernels.h"
#include "cuda_stream_plugin.h"

#include <cstring>
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
  printf("AddKernelImpl::ComputeImpl start\n");
  fflush(stdout);
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
    printf("MatMulKernelImpl::ReleaseImpl called\n");
    fflush(stdout);
    delete static_cast<MatMulKernelImpl*>(this_ptr);
  }
};

/*static*/
OrtStatus* ORT_API_CALL MatMulKernelImpl::ComputeImpl(
    OrtKernelImpl* /*this_ptr*/, OrtKernelContext* context) noexcept {
  printf("MatMulKernelImpl::ComputeImpl start\n");
  fflush(stdout);
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

  printf("MatMul: M=%d, N=%d, K=%d, a=%p, b=%p, y=%p\n", M, N, K, (void*)a_data, (void*)b_data, (void*)y_data);
  fflush(stdout);

  if (M > 0 && N > 0 && K > 0) {
    CudaSyncStream* stream_impl = GetCudaSyncStream(ctx);
    if (!stream_impl) {
      return Ort::GetApi().CreateStatus(ORT_EP_FAIL, "Failed to get CUDA stream");
    }

    cublasHandle_t cublas_handle = stream_impl->GetCublasHandle();
    printf("MatMul: cublas_handle=%p\n", cublas_handle);
    fflush(stdout);

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
    printf("MatMul: cublasSgemm completed\n");
    fflush(stdout);
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
  printf("GemmKernelImpl::ComputeImpl start: alpha=%f, beta=%f, trans_a=%d, trans_b=%d\n",
         self->alpha_, self->beta_, self->trans_a_, self->trans_b_);
  fflush(stdout);
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
OrtStatus* CreateCudaKernelRegistry(const OrtEpApi& ep_api,
                                    const char* ep_name,
                                    void* /*create_kernel_state*/,
                                    OrtKernelRegistry** out_registry) {
  *out_registry = nullptr;

  EXCEPTION_TO_STATUS_BEGIN

  Ort::KernelRegistry registry;

  // Get float tensor data type for type constraints
  const OrtDataType* float_type = nullptr;
  RETURN_IF_ERROR(ep_api.GetTensorDataType(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &float_type));
  std::vector<const OrtDataType*> float_types = {float_type};

  // --- Register Relu (ONNX opset 14+) ---
  {
    Ort::KernelDef relu_def = Ort::KernelDefBuilder()
                                  .SetOperatorType("Relu")
                                  .SetDomain("")
                                  .SetSinceVersion(1, 21)
                                  .SetExecutionProvider(ep_name)
                                  .AddTypeConstraint("T", float_types)
                                  .SetInputMemType(0, OrtMemTypeDefault)
                                  .SetOutputMemType(0, OrtMemTypeDefault)
                                  .Build();

    auto relu_create_fn = [](void* /*state*/,
                             const OrtKernelInfo* /*info*/,
                             OrtKernelImpl** kernel_out) noexcept -> OrtStatus* {
      *kernel_out = new ReluKernelImpl();
      return nullptr;
    };

    RETURN_IF_ERROR(registry.AddKernel(relu_def.release(), relu_create_fn, nullptr));
  }

  // --- Register Add (ONNX opset 14+) ---
  {
    Ort::KernelDef add_def = Ort::KernelDefBuilder()
                                 .SetOperatorType("Add")
                                 .SetDomain("")
                                 .SetSinceVersion(1, 21)
                                 .SetExecutionProvider(ep_name)
                                 .AddTypeConstraint("T", float_types)
                                 .SetInputMemType(0, OrtMemTypeDefault)
                                 .SetInputMemType(1, OrtMemTypeDefault)
                                 .SetOutputMemType(0, OrtMemTypeDefault)
                                 .Build();

    auto add_create_fn = [](void* /*state*/,
                            const OrtKernelInfo* /*info*/,
                            OrtKernelImpl** kernel_out) noexcept -> OrtStatus* {
      *kernel_out = new AddKernelImpl();
      return nullptr;
    };

    RETURN_IF_ERROR(registry.AddKernel(add_def.release(), add_create_fn, nullptr));
  }

  // --- Register MatMul (ONNX opset 14+) ---
  {
    Ort::KernelDef matmul_def = Ort::KernelDefBuilder()
                                    .SetOperatorType("MatMul")
                                    .SetDomain("")
                                    .SetSinceVersion(1, 21)
                                    .SetExecutionProvider(ep_name)
                                    .AddTypeConstraint("T", float_types)
                                    .SetInputMemType(0, OrtMemTypeDefault)
                                    .SetInputMemType(1, OrtMemTypeDefault)
                                    .SetOutputMemType(0, OrtMemTypeDefault)
                                    .Build();

    auto matmul_create_fn = [](void* /*state*/,
                               const OrtKernelInfo* /*info*/,
                               OrtKernelImpl** kernel_out) noexcept -> OrtStatus* {
      auto* kernel = new MatMulKernelImpl();
      printf("matmul_create_fn: kernel=%p, Compute=%p, Release=%p\n",
             (void*)kernel, (void*)kernel->Compute, (void*)kernel->Release);
      fflush(stdout);
      *kernel_out = kernel;
      return nullptr;
    };

    RETURN_IF_ERROR(registry.AddKernel(matmul_def.release(), matmul_create_fn, nullptr));
  }

  // --- Register Gemm (ONNX opset 14+) ---
  {
    Ort::KernelDef gemm_def = Ort::KernelDefBuilder()
                                  .SetOperatorType("Gemm")
                                  .SetDomain("")
                                  .SetSinceVersion(1, 21)
                                  .SetExecutionProvider(ep_name)
                                  .AddTypeConstraint("T", float_types)
                                  .SetInputMemType(0, OrtMemTypeDefault)
                                  .SetInputMemType(1, OrtMemTypeDefault)
                                  .SetInputMemType(2, OrtMemTypeDefault)
                                  .SetOutputMemType(0, OrtMemTypeDefault)
                                  .Build();

    auto gemm_create_fn = [](void* /*state*/,
                             const OrtKernelInfo* info,
                             OrtKernelImpl** kernel_out) noexcept -> OrtStatus* {
      auto* kernel = new GemmKernelImpl(info);
      printf("gemm_create_fn: kernel=%p, Compute=%p, Release=%p\n",
             (void*)kernel, (void*)kernel->Compute, (void*)kernel->Release);
      fflush(stdout);
      *kernel_out = kernel;
      return nullptr;
    };

    RETURN_IF_ERROR(registry.AddKernel(gemm_def.release(), gemm_create_fn, nullptr));
  }

  *out_registry = registry.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
