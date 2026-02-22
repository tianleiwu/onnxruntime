# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Build the CUDA Execution Provider as a plugin shared library.
# This file is included from the main CMakeLists.txt when onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON.

message(STATUS "Building CUDA EP as plugin shared library")

set(CUDA_PLUGIN_EP_DIR "${ONNXRUNTIME_ROOT}/core/providers/cuda/plugin")

# Source files (C++ and CUDA)
set(CUDA_PLUGIN_EP_CC_SRCS
    ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_ep.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_ep_factory.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_ep.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_allocator_plugin.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_data_transfer_plugin.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_stream_plugin.cc
)

set(CUDA_PLUGIN_EP_CU_SRCS
    ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_kernels.cu
)

# Create shared library target using the ORT helper function for plugins
onnxruntime_add_shared_library_module(onnxruntime_providers_cuda_plugin
    ${CUDA_PLUGIN_EP_CC_SRCS}
    ${CUDA_PLUGIN_EP_CU_SRCS}
)

# Set CUDA standard
set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# --- Find cuDNN (may be at a custom path via onnxruntime_CUDNN_HOME) ---
set(_CUDNN_SEARCH_PATHS "")
if(onnxruntime_CUDNN_HOME)
  list(APPEND _CUDNN_SEARCH_PATHS "${onnxruntime_CUDNN_HOME}")
endif()
if(DEFINED ENV{CUDNN_HOME})
  list(APPEND _CUDNN_SEARCH_PATHS "$ENV{CUDNN_HOME}")
endif()

find_path(CUDA_PLUGIN_CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS ${_CUDNN_SEARCH_PATHS}
    PATH_SUFFIXES include
)

find_library(CUDA_PLUGIN_CUDNN_LIBRARY
    NAMES cudnn
    HINTS ${_CUDNN_SEARCH_PATHS}
    PATH_SUFFIXES lib64 lib
)

if(NOT CUDA_PLUGIN_CUDNN_INCLUDE_DIR OR NOT CUDA_PLUGIN_CUDNN_LIBRARY)
  message(FATAL_ERROR "cuDNN not found for CUDA Plugin EP. Set onnxruntime_CUDNN_HOME or CUDNN_HOME.")
endif()

message(STATUS "CUDA Plugin EP: cuDNN include: ${CUDA_PLUGIN_CUDNN_INCLUDE_DIR}")
message(STATUS "CUDA Plugin EP: cuDNN library: ${CUDA_PLUGIN_CUDNN_LIBRARY}")

# Include directories — only public ORT headers + CUDA toolkit + cuDNN
target_include_directories(onnxruntime_providers_cuda_plugin PRIVATE
    ${REPO_ROOT}/include
    ${REPO_ROOT}/include/onnxruntime/core/session
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDA_PLUGIN_CUDNN_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(onnxruntime_providers_cuda_plugin PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    ${CUDA_PLUGIN_CUDNN_LIBRARY}
)

# Symbol visibility — only export CreateEpFactories and ReleaseEpFactory
target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE ORT_API_MANUAL_INIT)

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
