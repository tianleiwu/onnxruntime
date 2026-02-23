# CUDA Plugin EP Migration — Task Tracking

## Stage 1: Plugin Shell & Infrastructure ✅
- [x] DLL entry points (`CreateEpFactories`, `ReleaseEpFactory`)
- [x] `CudaEpFactory` (device enumeration, EP creation)
- [x] [CudaEp](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_ep.cc#11-39) (GetCapability, kernel registry)
- [x] [CudaAllocator](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.cc#138-205) (device + pinned)
- [x] `CudaDataTransfer` (CPU↔GPU, GPU↔GPU)
- [x] [CudaSyncStream](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.h#22-24) + [CudaSyncNotification](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.h#62-63) (handles, events)
- [x] [CudaKernel](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#20-25) base adapter
- [x] CMake integration (`BUILD_CUDA_EP_AS_PLUGIN`)
- [x] Validation: Relu, Add pass through plugin EP

## Stage 2: Operator Expansion (MatMul/Gemm/Conv) ✅
- [x] Handle management (cuBLAS/cuDNN via [CudaSyncStream](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.h#22-24))
- [x] MatMul (cublasSgemm)
- [x] Gemm (alpha, beta, bias)
- [x] Conv (cudnnConvolutionForward)
- [x] Python test validation

## Stage 3: Kernel Adapter Completion & Registration Migration ✅
- [x] [migrate_cuda_registrations.py](file:///home/tlwu/onnxruntime/tools/python/migrate_cuda_registrations.py) — multi-type, contrib domain
- [x] 1,049 standard + 272 contrib [.inc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc) registrations
- [x] CMake-driven [.inc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc) regeneration
- [x] Memory type annotations (37 ops)
- [x] [MLTypeCallDispatcher](file:///home/tlwu/onnxruntime/include/onnxruntime/core/framework/data_types_internal.h#713-864) workaround for [IsInf](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops.cc#97-102)/[IsNan](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops_impl.h#165-167)
- [x] Adapter-compiled activations ([activations.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/activation/activations.cc), [unary_elementwise_ops.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops.cc))
- [x] `cpuinfo` linkage fix
- [x] Code documentation/comments

## Stage 4: Expand Operator Coverage & Harden Adapter
- [x] **4.1** Port tensor manipulation kernels (Reshape, Squeeze, Unsqueeze, Flatten, Transpose, Concat, Split, Gather — remaining)
- [x] **4.2** Port math & binary element-wise kernels (Sub, Mul, Div, Pow, Cast, Where, etc.)
- [x] **4.3** Port reduction kernels (ReduceMean, Softmax, ArgMax, etc.)
- [x] **4.4** Harden adapter (vector attributes [GetAttrs](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h#155-166), string attrs, multi-output)
- [x] **4.5** Contrib op support (GroupQueryAttention, RotaryEmbedding)

## Stage 5: Remove Private Bridge & Polish (Future)
- [ ] Audit remaining internal includes
- [ ] Port remaining dependencies
- [ ] CUDA Graph implementation
- [ ] Performance regression testing
