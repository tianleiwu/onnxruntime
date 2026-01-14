#!/bin/bash
# Usage:
#   ./run.sh [OPTIONS]
#
# Options:
#   --build           Run build.sh
#   --clean           Clean build directory (rm -rf build)
#   --clean_gqa       Clean GQA build artifacts
#   --clean_flash     Clean flash attention build artifacts
#   --install         Install the built wheel
#   --test            Run test_gqa.py
#   --benchmark       Run benchmark_gqa.py
#   --profile         Run profiling with nsys
#

source $(conda info --base)/etc/profile.d/conda.sh

conda activate py312

pip install cmake ninja packaging numpy nvtx

LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:/home/tlwu/anaconda3/envs/py312/lib:/home/tlwu/cudnn9.8/lib64:/home/tlwu/cudnn9.8/lib

# Parse arguments
RUN_BUILD=false
RUN_INSTALL=false
RUN_TEST=false
RUN_BENCHMARK=false
RUN_PROFILE=false
ENABLE_DUMP="OFF"
RUN_TEST_CASE=false
TEST_CASE=""
BUILD_TYPE="Debug"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --build)
            RUN_BUILD=true
            echo "==== 🚀 Build run enabled ===="
            ;;
        --clean)
            echo "==== 🧹 Cleaning all build artifacts... ===="
            rm -rf build/cuda/$BUILD_TYPE
            ;;
        --clean_gqa)
            echo "==== 🧹 Cleaning GQA build artifacts... ===="
            rm -f build/cuda/$BUILD_TYPE/CMakeFiles/onnxruntime_providers_cuda.dir/home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/group_query_attention*
            ;;
        --clean_flash)
            echo "==== 🧹 Cleaning flash attention build artifacts... ===="
            rm -rf build/cuda/$BUILD_TYPE/CMakeFiles/onnxruntime_providers_cuda.dir/home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention
            ;;
        --install)
            echo "==== 📦 Install built wheel enabled ===="
            RUN_INSTALL=true
            ;;
        --test)
            RUN_TEST=true
            echo "==== 🚫 Test run enabled ===="
            ;;
        --test_case)
            RUN_TEST_CASE=true
            TEST_CASE="$2"
            echo "==== 🚫 Test case run enabled: $TEST_CASE ===="
            shift
            ;;
        --benchmark)
            RUN_BENCHMARK=true
            echo "==== 📊 Benchmark run enabled ===="
            ;;
        --dump)
            ENABLE_DUMP="ON"
            echo "==== 📊 Enable dump ===="
            ;;
        --profile)
            RUN_PROFILE=true
            echo "==== 📊 Profiling enabled ===="
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done


# Format code with clang-format.
lintrunner -a

set -e
set -o pipefail


if [ "$RUN_BUILD" = true ]; then
    if [ "$USE_QUICK_BUILD" = true ]; then
        rm -f build/cuda/$BUILD_TYPE/CMakeFiles/onnxruntime_providers_cuda.dir/home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/flash_int*_fwd_*
    fi
    start_build=$(date +%s)

    sh build.sh --config $BUILD_TYPE  --build_dir build/cuda --parallel  --use_cuda \
            --cuda_version 12.8 --cuda_home  /home/tlwu/cuda12.8/  \
            --cudnn_home /home/tlwu/cudnn9.8/ \
            --build_wheel --skip_tests \
            --cmake_generator Ninja \
            --enable_cuda_nhwc_ops \
            --use_binskim_compliant_compile_flags \
            --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=ON \
            --cmake_extra_defines onnxruntime_DUMP_TENSOR=ON \
            --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=native \
            --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
            --cmake_extra_defines onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS=OFF \
            --cmake_extra_defines onnxruntime_USE_FPA_INTB_GEMM=OFF \
            --cmake_extra_defines onnxruntime_FLASH_BUILD=ON \

    if [ $? -ne 0 ]; then
        echo "==== ❌ Build failed! Exiting. ===="
        exit 1
    fi

    end_build=$(date +%s)

    total_time=$((end_build - start_build))
    echo "==== ✅ Build completed in $total_time seconds ($(echo "scale=2; $total_time/60" | bc) minutes) ===="
fi

if [ "$RUN_INSTALL" = true ]; then
    pip uninstall onnxruntime-gpu onnxruntime -y
    pip install build/cuda/$BUILD_TYPE/dist/onnxruntime_gpu-1.24.0-cp312-cp312-linux_x86_64.whl --force-reinstall --no-cache-dir
fi

case "${ENABLE_DUMP}" in
  ON)
    export ORT_DEBUG_NODE_IO_DUMP_SHAPE_DATA=1
    export ORT_DEBUG_NODE_IO_DUMP_NODE_PLACEMENT=1
    export ORT_DEBUG_NODE_IO_DUMP_INPUT_DATA=1
    export ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=1
    export ORT_DEBUG_NODE_IO_SNIPPET_THRESHOLD=1000
    export ORT_DEBUG_NODE_IO_SNIPPET_EDGE_ITEMS=64
    export ORT_DEBUG_NODE_IO_DUMP_STATISTICS_DATA=0
    export ORT_ENABLE_GPU_DUMP=1
    export ORT_TENSOR_SNIPPET_THRESHOLD=1000
    export ORT_TENSOR_SNIPPET_EDGE_ITEMS=64
    ;;
  *)
    export ORT_DEBUG_NODE_IO_DUMP_SHAPE_DATA=0
    export ORT_DEBUG_NODE_IO_DUMP_NODE_PLACEMENT=0
    export ORT_DEBUG_NODE_IO_DUMP_INPUT_DATA=0
    export ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=0
    export ORT_DEBUG_NODE_IO_SNIPPET_THRESHOLD=200
    export ORT_DEBUG_NODE_IO_SNIPPET_EDGE_ITEMS=3
    export ORT_DEBUG_NODE_IO_DUMP_STATISTICS_DATA=0
    export ORT_ENABLE_GPU_DUMP=0
    export ORT_TENSOR_SNIPPET_THRESHOLD=200
    export ORT_TENSOR_SNIPPET_EDGE_ITEMS=3
esac

# Test/Benchmark/profile scripts are in the following directory
cd onnxruntime/test/python/transformers

if [ "$RUN_TEST" = true ]; then
    env $QUICK_BUILD_ENV python test_gqa.py
    test_exit_code=$?
    if [ $test_exit_code -ne 0 ]; then
        echo "==== ❌ test_gqa.py failed with exit code $test_exit_code! Exiting. ===="
        exit $test_exit_code
    fi
    echo "==== ✅ test_gqa.py passed! ===="
fi

if [ "$RUN_TEST_CASE" = true ]; then
    python test_gqa.py -k "$TEST_CASE"
    test_exit_code=$?
    if [ $test_exit_code -ne 0 ]; then
        echo "==== ❌ test_gqa.py -k $TEST_CASE failed with exit code $test_exit_code! Exiting. ===="
        exit $test_exit_code
    fi
    echo "==== ✅ test_gqa.py -k $TEST_CASE passed! ===="
fi

if [ "$RUN_BENCHMARK" = true ]; then
    echo "==== 📊 Running benchmark_gqa.py ... ===="
    python benchmark_gqa.py
fi

if [ "$RUN_PROFILE" = true ]; then
    echo "==== 🚀 Running profile_gqa.sh... ===="
    # bash profile_gqa.sh --fp16
    # bash profile_gqa.sh --fp16 --qkv
    # bash profile_gqa.sh --int8 --int4 --fp16 --int8_quant --qkv
    bash profile_gqa.sh --fp16 --qkv -b 1 -s 2048 -p 0 --qkv
    bash profile_gqa.sh --int8 --qkv -b 1 -s 2048 -p 0 --qkv

fi
