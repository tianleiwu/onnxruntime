# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Build the CUDA Execution Provider as a plugin shared library.
# This file is included from the main CMakeLists.txt when onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON.

message(STATUS "Building CUDA EP as plugin shared library")

set(CUDA_PLUGIN_EP_DIR "${ONNXRUNTIME_ROOT}/core/providers/cuda/plugin")
set(CUDA_PLUGIN_REGISTRATION_SCRIPT "${REPO_ROOT}/tools/python/migrate_cuda_registrations.py")
set(CUDA_PLUGIN_REGISTRATION_INPUT "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_execution_provider.cc")
set(CUDA_PLUGIN_CONTRIB_REGISTRATION_INPUT "${REPO_ROOT}/onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc")
set(CUDA_PLUGIN_REGISTRATION_OUTPUT "${CUDA_PLUGIN_EP_DIR}/cuda_plugin_generated_registrations.inc")
set(CUDA_PLUGIN_CONTRIB_REGISTRATION_OUTPUT "${CUDA_PLUGIN_EP_DIR}/cuda_plugin_generated_contrib_registrations.inc")

# Source files (C++ and CUDA)
set(CUDA_PLUGIN_EP_CC_SRCS
    ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_ep.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_ep_factory.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_ep.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_allocator_plugin.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_data_transfer_plugin.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_stream_plugin.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/activation/activations.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/binary_elementwise_ops.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/clip.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/unary_elementwise_ops.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/cast_op.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/concat.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/where.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/split.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/gather.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/transpose.cc
)

set(CUDA_PLUGIN_EP_CU_SRCS
    ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_kernels.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/activation/activations_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/binary_elementwise_ops_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/clip_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/unary_elementwise_ops_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/cast_op.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/concat_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/where_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/split_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/gather_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/transpose_impl.cu
)

# Create shared library target using the ORT helper function for plugins
onnxruntime_add_shared_library_module(onnxruntime_providers_cuda_plugin
    ${CUDA_PLUGIN_EP_CC_SRCS}
    ${CUDA_PLUGIN_EP_CU_SRCS}
)

add_custom_command(
    OUTPUT ${CUDA_PLUGIN_REGISTRATION_OUTPUT} ${CUDA_PLUGIN_CONTRIB_REGISTRATION_OUTPUT}
    COMMAND ${Python_EXECUTABLE} ${CUDA_PLUGIN_REGISTRATION_SCRIPT}
            --input ${CUDA_PLUGIN_REGISTRATION_INPUT}
            --output ${CUDA_PLUGIN_REGISTRATION_OUTPUT}
    COMMAND ${Python_EXECUTABLE} ${CUDA_PLUGIN_REGISTRATION_SCRIPT}
            --contrib
            --check-critical-contrib
            --input ${CUDA_PLUGIN_CONTRIB_REGISTRATION_INPUT}
            --output ${CUDA_PLUGIN_CONTRIB_REGISTRATION_OUTPUT}
    DEPENDS
        ${CUDA_PLUGIN_REGISTRATION_SCRIPT}
        ${CUDA_PLUGIN_REGISTRATION_INPUT}
        ${CUDA_PLUGIN_CONTRIB_REGISTRATION_INPUT}
    COMMENT "Generating CUDA plugin kernel registrations"
    VERBATIM
)

add_custom_target(onnxruntime_cuda_plugin_generate_registrations
    DEPENDS ${CUDA_PLUGIN_REGISTRATION_OUTPUT} ${CUDA_PLUGIN_CONTRIB_REGISTRATION_OUTPUT}
)
add_dependencies(onnxruntime_providers_cuda_plugin onnxruntime_cuda_plugin_generate_registrations)

# Set CUDA standard and flags
set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcudafe;--diag_suppress=550>")

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
    ${REPO_ROOT}/onnxruntime
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDA_PLUGIN_CUDNN_INCLUDE_DIR}
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
    ${CUDA_PLUGIN_CUDNN_LIBRARY}
    Boost::mp11
    safeint_interface
    onnxruntime_common
    onnxruntime_framework
    cpuinfo::cpuinfo
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

# Keep the ported kernel list for clarity/reuse in plugin build.
# Do not use global source COMPILE_FLAGS here because those file properties
# leak to other targets (e.g. onnxruntime_providers_cuda) and cause
# redefinition errors when cuda_kernel_adapter.h is force-included there.
set(PORTED_KERNEL_SRCS
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/activation/activations.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/binary_elementwise_ops.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/clip.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/unary_elementwise_ops.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/cast_op.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/concat.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/where.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/split.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/gather.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/transpose.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/activation/activations_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/binary_elementwise_ops_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/clip_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/unary_elementwise_ops_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/cast_op.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/concat_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/where_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/split_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/gather_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/transpose_impl.cu
)

# Set output name
set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
    OUTPUT_NAME "onnxruntime_providers_cuda_plugin"
)

# Install
install(TARGETS onnxruntime_providers_cuda_plugin
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
