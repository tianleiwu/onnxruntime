// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_plugin_kernels.h"
#include "cuda_stream_plugin.h"

#include <cstring>
#include <vector>

namespace onnxruntime {
namespace cuda_plugin {

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
// Kernel Registry Creation
// ---------------------------------------------------------------------------

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
                                  .SetSinceVersion(14, 14)
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

    RETURN_IF_ERROR(registry.AddKernel(std::move(relu_def), relu_create_fn, nullptr));
  }

  // --- Register Add (ONNX opset 14+) ---
  {
    Ort::KernelDef add_def = Ort::KernelDefBuilder()
                                 .SetOperatorType("Add")
                                 .SetDomain("")
                                 .SetSinceVersion(14, 14)
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

    RETURN_IF_ERROR(registry.AddKernel(std::move(add_def), add_create_fn, nullptr));
  }

  *out_registry = registry.release();
  return nullptr;

  EXCEPTION_TO_STATUS_END
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
