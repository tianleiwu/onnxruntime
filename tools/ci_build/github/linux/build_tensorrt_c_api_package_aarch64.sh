#!/bin/bash
set -e -x

CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual"

# Detect TensorRT home: tar-based install goes to /opt/tensorrt, RPM to /usr
if [ -f /opt/tensorrt/include/NvInfer.h ]; then
    TENSORRT_HOME=/opt/tensorrt
else
    TENSORRT_HOME=/usr
fi

mkdir -p $HOME/.onnx
docker run -e SYSTEM_COLLECTIONURI --rm --volume /data/onnx:/data/onnx:ro --volume $BUILD_SOURCESDIRECTORY:/onnxruntime_src \
  --volume $BUILD_BINARIESDIRECTORY:/build --volume /data/models:/build/models:ro \
  --volume $HOME/.onnx:/home/onnxruntimedev/.onnx -e NIGHTLY_BUILD onnxruntimecuda${CUDA_VERSION_MAJOR}xtrt86buildaarch64 \
  /bin/bash -c "/usr/bin/python3 /onnxruntime_src/tools/ci_build/build.py --build_dir /build --config Release --skip_tests \
  --skip_submodule_sync --parallel --use_binskim_compliant_compile_flags --build_shared_lib --build_java --build_nodejs \
  --use_tensorrt --cuda_version=$CUDA_VERSION --cuda_home=/usr/local/cuda-$CUDA_VERSION --cudnn_home=/usr --tensorrt_home=/opt/tensorrt \
  --cmake_extra_defines 'CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHS}' 'onnxruntime_USE_FPA_INTB_GEMM=OFF' \
  --use_vcpkg --use_vcpkg_ms_internal_asset_cache && cd /build/Release && make install DESTDIR=/build/installed"
