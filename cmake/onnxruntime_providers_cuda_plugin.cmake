# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Build the CUDA Execution Provider as a plugin shared library.
# This file is included from the main CMakeLists.txt when onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON.

message(STATUS "Building CUDA EP as plugin shared library")



set(CUDA_PLUGIN_EP_DIR "${ONNXRUNTIME_ROOT}/core/providers/cuda/plugin")

file(GLOB_RECURSE CUDA_EP_CC_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cc"
)

file(GLOB_RECURSE CUDA_EP_CU_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cu"
)

file(GLOB_RECURSE CUDA_EP_CU_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cu"
)



file(GLOB_RECURSE CUDA_CONTRIB_OPS_CC_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cc"
)

file(GLOB_RECURSE CUDA_CONTRIB_OPS_CU_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cu"
)

list(APPEND CUDA_PLUGIN_EP_CC_SRCS
     ${CUDA_EP_CC_SRCS}
     ${CUDA_CONTRIB_OPS_CC_SRCS}
)

list(APPEND CUDA_PLUGIN_EP_CU_SRCS
     ${CUDA_EP_CU_SRCS}
     ${CUDA_CONTRIB_OPS_CU_SRCS}
)

list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/aten_ops/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/collective/.*")

list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/aten_ops/.*")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/collective/.*")

# Exclude files that include cuda_execution_provider.h (directly or transitively),
# which conflicts with the adapter shim CUDAExecutionProvider class.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_execution_provider\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_provider_factory\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_provider_interface\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/einsum\\.cc$")

# Exclude the entire controlflow/ subdirectory — these inherit from CPU base
# classes (If, Loop, Scan) which are not available in the plugin build.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/controlflow/.*")

# Exclude the entire tunable/ subdirectory — it depends on the real CudaTuningContext
# and CUDAExecutionProvider which are not available in the plugin build.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tunable/.*")

# Exclude real EP infrastructure files (replaced by plugin/ equivalents).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_stream_handle\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_execution_provider_info\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_graph\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_mempool_arena\\.cc$")

# Exclude cuda_common.cc — its HalfGemmOptions definitions conflict with the
# adapter's inline shim. Utility functions are replaced or not needed.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_common\\.cc$")

# Exclude NHWC kernel registry — uses KernelRegistry directly (EP infrastructure).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_nhwc_kernels\\.cc$")

# Exclude files that use CudaStream (incomplete type in plugin build — requires
# cuda_stream_handle.h which depends on EP infrastructure).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/integer_gemm\\.cc$")

# Exclude entire rnn/ subdirectory — CudnnRnnBase and its subclasses (RNN, GRU, LSTM)
# all depend on CudaStream (dynamic_cast in cudnn_rnn_base.cc).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/rnn/.*")

list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/rnn/.*")

# Exclude files that use TensorSeq (incomplete type in plugin build).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/identity_op\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/sequence_op\\.cc$")

# Exclude size.cc — registers onnxruntime::Size (CPU op) whose Compute() body
# lives in the CPU provider and is not linked into the plugin.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/size\\.cc$")

# Exclude scatter_nd.cc — calls ScatterND::ValidateShapes whose implementation
# lives in the CPU provider's scatter_nd.cc (not linked into the plugin).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/scatter_nd\\.cc$")

# Exclude llm/ — attention.cc calls QkvToContext which dereferences
# onnxruntime::Stream* (not available in plugin build's adapter OpKernelContext).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/llm/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/llm/.*")

# Exclude constant_of_shape — inherits from ConstantOfShapeBase (CPU provider)
# which is not linked into the plugin.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/generator/constant_of_shape\\.cc$")

# Exclude matmul_integer.cc — uses GetComputeStream() with GemmInt8 which expects
# onnxruntime::Stream* (not available in adapter OpKernelContext).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/matmul_integer\\.cc$")

# Exclude matmul.cc — uses GetComputeStream() in FuncCallAdapter (needs onnxruntime::Stream*).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/matmul\\.cc$")

# Exclude variadic_elementwise_ops.cc — uses InputArgCount/RequiredInput/RequiredOutput
# which are not in the adapter Node/OpKernelContext.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/variadic_elementwise_ops\\.cc$")

# Exclude slice — inherits from SliceBase (CPU provider) not linked into the plugin.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/slice\\.cc$")

# Exclude space_depth_ops — inherits from SpaceDepthBase (CPU provider).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/space_depth_ops\\.cc$")

# Exclude concat.cc — uses InputArgCount and OpKernelContext::GetComputeStream() with CopyToGpu.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/concat\\.cc$")

# Exclude gather.cc — passes adapter OpKernelContext* to framework PrepareForCompute
# which expects onnxruntime::OpKernelContext*.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/gather\\.cc$")

# Exclude gather_nd.cc — uses GetComputeStream() with PrepareCompute framework function.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/gather_nd\\.cc$")

# Exclude pad.cc — passes adapter OpKernelContext to framework PadBase::HandleDimension.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/pad\\.cc$")

# Exclude reshape.cc/reshape.h — uses GetComputeStream() and CopyTensor with framework types.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/reshape\\.cc$")

# Exclude split.cc — uses GetComputeStream() with CopyToGpu.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/split\\.cc$")

# Exclude object_detection/ — NonMaxSuppression and RoiAlign inherit from CPU
# base classes (NonMaxSuppressionBase, RoiAlignBase) not linked into the plugin.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/object_detection/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/object_detection/.*")

# Exclude upsample.cc — UpsampleBase uses InputDefs() and
# OpKernelInfo::GetAllocator() not available in adapter.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/upsample\\.cc$")

# Exclude resize.cc — Resize inherits from Upsample (excluded above).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/resize\\.cc$")

# Exclude einsum — einsum_auxiliary_ops.cc calls ReductionOps::ReduceCompute
# which is framework-only (guarded by #ifndef BUILD_CUDA_EP_AS_PLUGIN).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/einsum_utils/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/math/einsum_utils/.*")

# Exclude unsqueeze.cc — passes adapter OpKernelContext* to framework
# FlattenHelper/CopyTensor which expects onnxruntime::OpKernelContext*.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/unsqueeze\\.cc$")

# Exclude shape_op.cc — Shape inherits from onnxruntime::OpKernel (framework)
# which cannot convert to ep::adapter::OpKernel in the plugin build.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/shape_op\\.cc$")

# Exclude cumsum.cc — cumsum_op::GetAxis is defined in framework CPU provider.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/cumsum\\.cc$")

# Exclude tile.cc — TileOp::IsTileMemcpy is defined in framework CPU provider.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/tile\\.cc$")

# Create shared library target using the ORT helper function for plugins
onnxruntime_add_shared_library_module(onnxruntime_providers_cuda_plugin
    ${CUDA_PLUGIN_EP_CC_SRCS}
    ${CUDA_PLUGIN_EP_CU_SRCS}
)
target_sources(onnxruntime_providers_cuda_plugin PRIVATE
    ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_adapter_registry.cc
)

# Set CUDA standard and flags
set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Suppress -Werror=maybe-uninitialized for local variables written by
# adapter OpKernelInfo::GetAttr<> (GCC falsely warns about variables that are
# initialised inside GetAttr’s output parameter path).
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-maybe-uninitialized>
)
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
    # Flash-attention, XQA, MoE, and other pure CUDA kernel .cu files must NOT
    # receive the ORT-framework force-include (it conflicts with cute::Tensor etc.).
    # cuda_plugin_kernels.cu already #include "cuda_kernel_adapter.h" directly.
    # Op-registration .cc files do not include it directly, so they need it here.
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcudafe;--diag_suppress=550>"
    "$<$<COMPILE_LANGUAGE:CXX>:-include;${REPO_ROOT}/include/onnxruntime/ep/adapters.h>"
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ${CUDA_PLUGIN_EP_DIR}/cuda_kernel_adapter.h>"
)

# --- EP Adapter Framework ---
# Define ORT_CUDA_PLUGIN_USE_ADAPTER for the plugin target.
# This skips SHARED_PROVIDER in provider_api.h, allowing kernel files
# to use real framework types from core/framework/op_kernel.h directly.
target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE ORT_CUDA_PLUGIN_USE_ADAPTER=1)



include(cudnn_frontend)
include(cutlass)

# --- Find cuDNN (may be at a custom path via onnxruntime_CUDNN_HOME) ---
set(_CUDNN_SEARCH_PATHS "")
if(onnxruntime_CUDNN_HOME)
  list(APPEND _CUDNN_SEARCH_PATHS "${onnxruntime_CUDNN_HOME}")
endif()
if(DEFINED ENV{CUDNN_HOME})
  list(APPEND _CUDNN_SEARCH_PATHS "$ENV{CUDNN_HOME}")
endif()

set(CUDA_PLUGIN_CUDNN_INCLUDE_DIR ${CUDNN_INCLUDE_DIR})
set(CUDA_PLUGIN_CUDNN_LIBRARY ${cudnn_LIBRARY})

if(NOT CUDA_PLUGIN_CUDNN_INCLUDE_DIR OR NOT CUDA_PLUGIN_CUDNN_LIBRARY)
  message(FATAL_ERROR "cuDNN not found (from main ORT search) for CUDA Plugin EP.")
endif()

message(STATUS "CUDA Plugin EP: cuDNN include: ${CUDA_PLUGIN_CUDNN_INCLUDE_DIR}")
message(STATUS "CUDA Plugin EP: cuDNN library: ${CUDA_PLUGIN_CUDNN_LIBRARY}")

# Include directories — only public ORT headers + CUDA toolkit + cuDNN + internal headers for adapter
target_include_directories(onnxruntime_providers_cuda_plugin PRIVATE
    ${REPO_ROOT}/include
    ${REPO_ROOT}/include/onnxruntime/core/session
    ${ONNXRUNTIME_ROOT}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDA_PLUGIN_CUDNN_INCLUDE_DIR}
    ${Eigen3_SOURCE_DIR}
    ${cutlass_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/examples
    ${cutlass_SOURCE_DIR}/tools/util/include
)

onnxruntime_add_include_to_target(
    onnxruntime_providers_cuda_plugin
    onnxruntime_common
    onnx
    onnx_proto
    ${PROTOBUF_LIB}
    flatbuffers::flatbuffers
)

# Link libraries
target_link_libraries(onnxruntime_providers_cuda_plugin PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDNN::cudnn_all
    cudnn_frontend
    ${CUDA_PLUGIN_CUDNN_LIBRARY}
    Boost::mp11
    safeint_interface
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_mlas
    onnxruntime_flatbuffers
    onnxruntime_common
    cpuinfo::cpuinfo
    onnx
    onnx_proto
    ${PROTOBUF_LIB}
)

# Symbol visibility — only export CreateEpFactories and ReleaseEpFactory
target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE ORT_API_MANUAL_INIT BUILD_CUDA_EP_AS_PLUGIN ONNX_ML=1 ONNX_NAMESPACE=onnx ONNX_USE_LITE_PROTO=1)

if(WIN32)
  # Windows: use .def file for symbol exports
  set(CUDA_PLUGIN_DEF_FILE ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_ep_symbols.def)
  if(EXISTS ${CUDA_PLUGIN_DEF_FILE})
    target_sources(onnxruntime_providers_cuda_plugin PRIVATE ${CUDA_PLUGIN_DEF_FILE})
  endif()
else()
  # Linux/macOS: hide all symbols by default, explicitly export via __attribute__((visibility("default")))
  set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
      C_VISIBILITY_PRESET hidden
      CXX_VISIBILITY_PRESET hidden
  )
endif()



# Set output name
set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
    OUTPUT_NAME "onnxruntime_providers_cuda_plugin"
)

# Install
install(TARGETS onnxruntime_providers_cuda_plugin
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
