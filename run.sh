#!/bin/bash
# Usage:
#   ./run.sh [OPTIONS]
#
# Options:
#   --build           Run build.sh
#   --clean           Clean build directory (rm -rf build)
#   --install         Install the built wheel
#   --dump            Enable dump node inputs/outputs
# Options for MoE or QMoE:
#   --clean_moe       Clean moe build artifacts
#   --test_moe        Run test_moe_cuda.py
#   --test_qmoe       Run test_qmoe_cuda.py
#   --test_qmoe_case  Run specific test case in test_qmoe_cuda.py (e.g. TestSwigluQMoE.test_swiglu_qmoe_blockwise_parity_3)
# Options for Flash Attention only (Not Used for MoE or QMoE):
#   --quick           Configure quick build/test (flash attention hdim128 only and exclude bf16, sets onnxruntime_QUICK_BUILD=ON)
#   --quick_build     A combination of --quick and --build
#   --benchmark       Run benchmark_gqa.py
#   --profile         Run profiling GQA with nsys

source $(conda info --base)/etc/profile.d/conda.sh

conda activate py312

pip install cmake ninja packaging numpy==2.4.0 nvtx

export CUDA_HOME=/home/tlwu/cuda12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:/home/tlwu/anaconda3/envs/py312/lib:/home/tlwu/cuda12.8/lib64:/home/tlwu/cudnn9.8/lib64:/home/tlwu/cudnn9.8/lib

# Parse arguments
QUICK_BUILD_FLAG=""
QUICK_BUILD_ENV=""
RUN_BUILD=false
USE_QUICK_BUILD=false
RUN_INSTALL=false
RUN_TEST_MOE=false
RUN_TEST_QMOE=false
RUN_TEST_QMOE_CASE=false
TEST_QMOE_CASE=""
RUN_TEST_MOE_CASE=false
TEST_MOE_CASE=""
RUN_BENCHMARK=false
RUN_PROFILE=false
ENABLE_DUMP="ON"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --build)
            RUN_BUILD=true
            echo "==== 🚀 Build run enabled ===="
            ;;
        --quick)
            USE_QUICK_BUILD=true
            QUICK_BUILD_FLAG="--cmake_extra_defines onnxruntime_QUICK_BUILD=ON"
            QUICK_BUILD_ENV="QUICK_BUILD=1"
            echo "==== 🚀 Quick build mode enabled (flash attention hdim128 only and exclude bf16) ===="
            ;;
        --quick_build)
            RUN_BUILD=true
            USE_QUICK_BUILD=true
            QUICK_BUILD_FLAG="--cmake_extra_defines onnxruntime_QUICK_BUILD=ON"
            QUICK_BUILD_ENV="QUICK_BUILD=1"
            echo "==== 🚀 Quick build mode enabled (flash attention hdim128 only and exclude bf16) ===="
            ;;
        --clean)
            echo "==== 🧹 Cleaning all build artifacts... ===="
            rm -rf build
            ;;
        --clean_moe)
            echo "==== 🧹 Cleaning moe build artifacts... ===="
            rm -rf build/cuda/Release/CMakeFiles/onnxruntime_providers_cuda.dir/home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/llm
            rm -rf build/cuda/Release/CMakeFiles/onnxruntime_providers_cuda.dir/home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/moe
            ;;
        --install)
            echo "==== 📦 Install built wheel enabled ===="
            RUN_INSTALL=true
            ;;
        --test_moe)
            RUN_TEST_MOE=true
            echo "==== 🚫 Test run enabled ===="
            ;;
        --test_qmoe)
            RUN_TEST_QMOE=true
            echo "==== 🚫 Test QMoE run enabled ===="
            ;;
        --test_moe_case)
            RUN_TEST_MOE_CASE=true
            TEST_MOE_CASE="$2"
            echo "==== 🚫 Test MoE case run enabled: $TEST_MOE_CASE ===="
            shift
            ;;
        --test_qmoe_case)
            RUN_TEST_QMOE_CASE=true
            TEST_QMOE_CASE="$2"
            echo "==== 🚫 Test QMoE case run enabled: $TEST_QMOE_CASE ===="
            shift
            ;;
        --benchmark)
            RUN_BENCHMARK=true
            echo "==== 📊 Benchmark run enabled ===="
            ;;
        --profile)
            RUN_PROFILE=true
            echo "==== 📊 Profiling enabled ===="
            ;;
        --dump)
            ENABLE_DUMP="ON"
            echo "==== 📊 Enable dump ===="
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
        rm -f build/cuda/Release/CMakeFiles/onnxruntime_providers_cuda.dir/home/tlwu/onnxruntime/onnxruntime/contrib_ops/cuda/bert/flash_attention/flash_int*_fwd_*
    fi
    start_build=$(date +%s)

    sh build.sh --config Release  --build_dir build/cuda --parallel  --use_cuda \
            --cuda_version 12.8 --cuda_home  /home/tlwu/cuda12.8/  \
            --cudnn_home /home/tlwu/cudnn9.8/ \
            --build_wheel --skip_tests \
            --cmake_generator Ninja \
            --enable_cuda_nhwc_ops \
            --use_binskim_compliant_compile_flags \
            --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=$ENABLE_DUMP \
            --cmake_extra_defines onnxruntime_DUMP_TENSOR=$ENABLE_DUMP \
            --cmake_extra_defines onnxruntime_LLM_VERBOSE=$ENABLE_DUMP \
            --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="90" \
            --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
            --cmake_extra_defines onnxruntime_USE_FPA_INTB_GEMM=OFF \
            --cmake_extra_defines onnxruntime_USE_FLASH_ATTENTION=OFF \
            --cmake_extra_defines onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION=OFF \
            $QUICK_BUILD_FLAG

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
    pip install build/cuda/Release/dist/onnxruntime_gpu-1.25.0-cp312-cp312-linux_x86_64.whl --force-reinstall
fi

if [ "$ENABLE_DUMP" = "ON" ]; then
    export ORT_DEBUG_NODE_IO_DUMP_SHAPE_DATA=1
    export ORT_DEBUG_NODE_IO_DUMP_INPUT_DATA=1
    export ORT_DEBUG_NODE_IO_DUMP_OUTPUT_DATA=1
    export ORT_DEBUG_NODE_IO_SNIPPET_THRESHOLD=200
    export ORT_DEBUG_NODE_IO_SNIPPET_EDGE_ITEMS=3
    export ORT_DEBUG_NODE_IO_DUMP_STATISTICS_DATA=0
fi

# Test/Benchmark/profile scripts are in the following directory
cd onnxruntime/test/python/transformers

if [ "$RUN_TEST_MOE" = true ]; then
    python test_moe_cuda.py
    test_exit_code=$?
    if [ $test_exit_code -ne 0 ]; then
        echo "==== ❌ test_moe_cuda.py failed with exit code $test_exit_code! Exiting. ===="
        exit $test_exit_code
    fi
    echo "==== ✅ test_moe_cuda.py passed! ===="
fi

if [ "$RUN_TEST_QMOE" = true ]; then
    python test_qmoe_cuda.py
    test_exit_code=$?
    if [ $test_exit_code -ne 0 ]; then
        echo "==== ❌ test_qmoe_cuda.py failed with exit code $test_exit_code! Exiting. ===="
        exit $test_exit_code
    fi
    echo "==== ✅ test_qmoe_cuda.py passed! ===="
fi

if [ "$RUN_TEST_QMOE_CASE" = true ]; then
    python test_qmoe_cuda.py -k "$TEST_QMOE_CASE"
    test_exit_code=$?
    if [ $test_exit_code -ne 0 ]; then
        echo "==== ❌ test_qmoe_cuda.py -k $TEST_QMOE_CASE failed with exit code $test_exit_code! Exiting. ===="
        exit $test_exit_code
    fi
    echo "==== ✅ test_qmoe_cuda.py -k $TEST_QMOE_CASE passed! ===="
fi

if [ "$RUN_TEST_MOE_CASE" = true ]; then
    python test_moe_cuda.py -k "$TEST_MOE_CASE"
    test_exit_code=$?
    if [ $test_exit_code -ne 0 ]; then
        echo "==== ❌ test_moe_cuda.py -k $TEST_MOE_CASE failed with exit code $test_exit_code! Exiting. ===="
        exit $test_exit_code
    fi
    echo "==== ✅ test_moe_cuda.py -k $TEST_MOE_CASE passed! ===="
fi

if [ "$RUN_BENCHMARK" = true ]; then
    echo "==== 📊 Running benchmark_gqa.py ... ===="
    python benchmark_gqa.py
fi

if [ "$RUN_PROFILE" = true ]; then
    echo "==== 🚀 Running profile_gqa.sh... ===="
    if [ "$RUN_QUICK_BUILD" = false ]; then
        bash profile_gqa.sh --all
    else
        bash profile_gqa.sh --int8 --int4 --fp16 --int8_quant
    fi
fi
