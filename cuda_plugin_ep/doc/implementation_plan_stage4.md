# Stage 4: Expand Operator Coverage & Harden Adapter

**Objective**: Activate the majority of standard ONNX operators in the plugin EP by porting their [.cc](file:///home/tlwu/onnxruntime/build_b/Release/tml.pb.cc)/[.cu](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/clip_impl.cu) source files and wiring them into `GetCreateFnForOp`. Complete the adapter layer to handle patterns required by a broader kernel population.

---

## Current State (End of Stage 3)

| Item | Count |
|------|-------|
| Active operators (via `GetCreateFnForOp`) | 15 (activations + Add + MatMul + Gemm + Conv) |
| Generated [.inc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc) registrations (standard) | 1,049 |
| Generated [.inc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc) registrations (contrib) | 272 |
| Source files in CMake | 8 [.cc](file:///home/tlwu/onnxruntime/build_b/Release/tml.pb.cc) + 3 [.cu](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/clip_impl.cu) |
| [CudaKernel](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h#205-221) adapter methods | Complete ([GetScratchBuffer](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h#253-287), [GetConstOnes](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h#337-377), [AllocateBufferOnCPUPinned](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h#304-329), handles, etc.) |
| Memory type annotations | ✅ Implemented for 37 ops |

---

## Proposed Changes

### Task 4.1 — Port Tensor Manipulation Kernels

These operators have CPU-input memory types already annotated and are among the most commonly used.

**Operators**: [Reshape](file:///home/tlwu/onnxruntime/include/onnxruntime/core/framework/tensor.h#256-267), [Transpose](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.cc#2799-2861), `Squeeze`, `Unsqueeze`, `Flatten`, `Gather`, `Concat`, `Split`, `Expand`, `Tile`

**Approach**: These ops use [provider_api.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/shared_library/provider_api.h)-dependent types ([Tensor](file:///home/tlwu/onnxruntime/include/onnxruntime/core/framework/tensor.h#77-84), [OpKernelContext](file:///home/tlwu/onnxruntime/include/onnxruntime/core/framework/op_kernel_context.h#15-18), `TensorShape`) which conflict in the plugin context. Two strategies:

1. **Thin wrapper kernels** (preferred for ops like [Reshape](file:///home/tlwu/onnxruntime/include/onnxruntime/core/framework/tensor.h#256-267), `Squeeze`, `Unsqueeze`, `Flatten`): Implement directly in [cuda_plugin_kernels.cu](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_kernels.cu) using the [OrtKernelContext](file:///home/tlwu/onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_api.h#2853-2854) API. These are simple memory copies or no-ops with shape manipulation.
2. **Adapter-compiled kernels** (for [Transpose](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.cc#2799-2861), `Gather`, `Concat`, `Split`, `Expand`, `Tile`): Add the CUDA provider [.cc](file:///home/tlwu/onnxruntime/build_b/Release/tml.pb.cc)/[.cu](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/clip_impl.cu) files to CMake and use the `AdapterKernelImpl` / `DEFINE_ADAPTER_CREATE_FN_TYPED` pattern.

#### Files to Modify
- [MODIFY] [cuda_plugin_kernels.cu](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_kernels.cu) — Add creation functions and `GetCreateFnForOp` entries
- [MODIFY] [onnxruntime_providers_cuda_plugin.cmake](file:///home/tlwu/onnxruntime/cmake/onnxruntime_providers_cuda_plugin.cmake) — Add source files for adapter-compiled ops

---

### Task 4.2 — Port Math & Binary Element-wise Kernels

**Operators**: [Sub](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/binary_elementwise_ops.h#148-149), [Mul](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/binary_elementwise_ops.h#155-156), [Div](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/binary_elementwise_ops.h#159-165), [Pow](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/binary_elementwise_ops.h#174-179), [Abs](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops.h#31-37), [Neg](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops.h#38-44), [Floor](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops.h#45-51), [Ceil](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops.h#55-56), [Sqrt](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops.h#66-72), [Exp](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops.h#83-84), [Log](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops.h#73-79), `Clip`, `Where`, [Equal](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/binary_elementwise_ops.h#254-255), [Greater](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/binary_elementwise_ops.h#246-247), [Less](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/binary_elementwise_ops.h#259-266), [Cast](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops_impl.h#121-125)

**Approach**: Most of these already have their CUDA implementations in `math/` directory files. Some are already partially covered by [unary_elementwise_ops.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops.cc)/[.cu](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/clip_impl.cu) (already in CMake).

#### Files to Add to CMake
- [core/providers/cuda/math/binary_elementwise_ops.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/binary_elementwise_ops.cc) + [binary_elementwise_ops_impl.cu](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/binary_elementwise_ops_impl.cu)
- Additional math [.cc](file:///home/tlwu/onnxruntime/build_b/Release/tml.pb.cc) files as needed per operator

---

### Task 4.3 — Port Reduction Kernels

**Operators**: `ReduceMean`, `ReduceSum`, `ReduceMax`, `ReduceMin`, [ArgMax](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.cc#2893-2918), [ArgMin](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.cc#2893-2918), `Softmax`, `LogSoftmax`

**Approach**: These use [cub](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_common.h#257-258) device functions and require [GetScratchBuffer](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h#253-287) (already implemented). Add source files and creation functions.

---

### Task 4.4 — Harden Adapter for Broader Kernel Patterns

Several kernel patterns not yet encountered in Stage 3 will appear:

| Pattern | Needed By | Implementation |
|---------|-----------|----------------|
| **Optional inputs** ([Input(i)](file:///home/tlwu/onnxruntime/include/onnxruntime/core/framework/op_kernel_context.h#35-41) returns `nullptr`) | [Gemm](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_common.h#195-256) (bias), [Conv](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.cc#2862-2874) (bias), `Clip` | Already handled via `OpKernelContext::Input<Tensor>` null check |
| **Multi-output** | `Split`, `TopK`, `Dropout` | Add `OpKernelContext::Output(i, shape)` for arbitrary indices |
| **String attributes** | [Cast](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops_impl.h#121-125), [Pad](file:///home/tlwu/onnxruntime/onnxruntime/test/python/transformers/test_gqa.py#2192-2201) | Add `OpKernelInfo::GetAttr<std::string>` specialization |
| **Vector attributes** (e.g., `axes`, `perm`) | [Transpose](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.cc#2799-2861), `Reduce*`, `Squeeze` | Add `OpKernelInfo::GetAttrs<int64_t>` for vector attributes |

> [!IMPORTANT]
> [GetAttrs<int64_t>](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/shared_library/provider_wrappedtypes.h#1428-1430) (vector attribute retrieval) is the most critical gap. Many tensor manipulation and reduction ops require reading `axes`, `perm`, `starts`, `ends` etc. as vector attributes from `OrtKernelInfo`.

#### Files to Modify
- [MODIFY] [cuda_kernel_adapter.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h) — Add [GetAttrs](file:///home/tlwu/onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_api.h#2867-2868), string attribute support

---

### Task 4.5 — Implement Contrib Op Support (GQA Focus)

**Operators**: `GroupQueryAttention`, [RotaryEmbedding](file:///home/tlwu/onnxruntime/onnxruntime/test/python/transformers/test_gqa.py#109-161) (contrib domain)

**Approach**: These are already in `cuda_plugin_generated_contrib_registrations.inc`. Add their source files and wire into `GetCreateFnForOp` with a domain check.

> [!NOTE]
> The [test_gqa.py](file:///home/tlwu/onnxruntime/onnxruntime/test/python/transformers/test_gqa.py) test already has `ORT_TEST_GQA_USE_CUDA_PLUGIN_EP=1` support in [cuda.sh](file:///home/tlwu/onnxruntime/cuda.sh). This is the validation target for contrib op support.

---

## Verification Plan

### Automated Tests
```bash
# Build and verify
make -j8 onnxruntime_providers_cuda_plugin

# Run plugin-specific tests
./cuda.sh --build --install --test_plugin

# GQA with plugin EP
ORT_TEST_GQA_USE_CUDA_PLUGIN_EP=1 python test_gqa.py
```

### Build Validation
- Zero compilation errors for all newly added source files
- `nm -u libonnxruntime_providers_cuda_plugin.so` should show no unexpected undefined symbols

### Operator Coverage Metric
- Target: **50+ active operators** (up from 15)
- Track via: count of `if (op_type ==` lines in `GetCreateFnForOp`
