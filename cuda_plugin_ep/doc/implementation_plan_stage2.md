# Stage 2: CUDA Plugin EP Operator Expansion & Handle Integration

Expand the CUDA Plugin EP to support high-impact numerical operators using cuBLAS and cuDNN, and refine handle management for optimized kernel execution.

## Proposed Changes

### 1. Robust Handle Management
To support cuBLAS/cuDNN in kernels, we need to retrieve the [CudaSyncStream](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.h#20-22) instance from the opaque [OrtSyncStream](file:///home/tlwu/onnxruntime/onnxruntime/core/framework/plugin_ep_stream.h#12-13) provided by the kernel context.

#### [MODIFY] [cuda_plugin_kernels.cu](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_kernels.cu)
- Use `OrtEpApi::SyncStream_GetImpl` to cast `OrtSyncStream*` to `CudaSyncStream*`.
- Extract `cublasHandle_t` and `cudnnHandle_t` for use in compute functions.

---

### 2. High-Impact Operators

#### [NEW] [MatMul]
- **Implementation**: Wrap `cublasSgemm` (for float).
- **Registration**: Register as `MatMul` for domain "" and opset 14+.

#### [NEW] [Gemm]
- **Implementation**: Wrap `cublasSgemm` with support for alpha, beta, and optional bias (C).
- **Registration**: Register as `Gemm` for domain "" and opset 14+.

#### [NEW] [Conv]
- **Implementation**: Wrap `cudnnConvolutionForward`.
- **Registration**: Register as [Conv](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.h#91-92) for domain "" and opset 14+.
- **Note**: For Stage 2, we will focus on simple configurations (no dilation/groups) and static layout.

---

### 3. Verification Plan

#### Automated Tests
Add the following test cases to [test_cuda_plugin_ep.py](file:///home/tlwu/onnxruntime/build/cuda/Release/test_cuda_plugin_ep.py):
- **`test_cuda_plugin_gemm`**: Verify `Gemm` output against NumPy (`alpha*A*B + beta*C`).
- **`test_cuda_plugin_matmul`**: Verify `MatMul` output against NumPy (`A @ B`).
- **`test_cuda_plugin_conv`**: Verify [Conv](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.h#91-92) output against a reference implementation or simply check valid execution/output shape.

#### Manual Verification
- Re-run `bash cuda.sh --build --test` to ensure new kernels are correctly registered and utilized by the ORT session.
