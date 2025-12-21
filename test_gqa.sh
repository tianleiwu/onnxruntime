#!/bin/bash

set -e
set -o pipefail

source $(conda info --base)/etc/profile.d/conda.sh

conda activate py312

pip install cmake ninja packaging numpy nvtx

pip uninstall onnxruntime-gpu onnxruntime -y

LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:/home/tlwu/anaconda3/envs/py312/lib:/home/tlwu/cudnn9.8/lib64:/home/tlwu/cudnn9.8/lib

rm -rf build

start_build=$(date +%s)

sh build.sh --config Release  --build_dir build/cuda --parallel  --use_cuda \
            --cuda_version 12.8 --cuda_home  /home/tlwu/cuda12.8/  \
            --cudnn_home /home/tlwu/cudnn9.8/ \
            --build_wheel --skip_tests \
            --cmake_generator Ninja \
            --enable_cuda_nhwc_ops \
            --use_binskim_compliant_compile_flags \
            --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=0 \
            --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES=90 \
            --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS=OFF \
            --cmake_extra_defines onnxruntime_USE_FPA_INTB_GEMM=OFF

if [ $? -ne 0 ]; then
    echo "==== ❌ Build failed! Exiting. ===="
    exit 1
fi

end_build=$(date +%s)
total_time=$((end_build - start_build))
echo "==== ✅ Build completed in $total_time seconds ($(echo "scale=2; $total_time/60" | bc) minutes) ===="

pip install build/cuda/Release/dist/onnxruntime_gpu-1.24.0-cp312-cp312-linux_x86_64.whl --force-reinstall

cd onnxruntime/test/python/transformers
python test_gqa.py
test_exit_code=$?
if [ $test_exit_code -ne 0 ]; then
    echo "==== ❌ test_gqa.py failed with exit code $test_exit_code! Exiting. ===="
    exit $test_exit_code
fi
echo "==== ✅ test_gqa.py passed! ===="

rm -f gqa_fp16.nsys-rep
rm -f gqa_fp16.sqlite
nsys profile -o gqa_fp16 --export=sqlite python profile_gqa.py --mode fp16 --warmup 5 --repeat 100
python parse_nsys.py gqa_fp16.sqlite --skip-first 5 --tag Fp16

rm -f gqa_int8.nsys-rep
rm -f gqa_int8.sqlite
nsys profile -o gqa_int8 --export=sqlite python profile_gqa.py --mode int8 --warmup 5 --repeat 100
python parse_nsys.py gqa_int8.sqlite --skip-first 5 --tag Int8

rm -f gqa_int4.nsys-rep
rm -f gqa_int4.sqlite
nsys profile -o gqa_int4 --export=sqlite python profile_gqa.py --mode int4 --warmup 5 --repeat 100
python parse_nsys.py gqa_int4.sqlite --skip-first 5 --tag Int4

cd -
