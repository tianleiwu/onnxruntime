# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file is used by Zip-Nuget Packaging NoContribOps Pipeline,Zip-Nuget-Java Packaging Pipeline
ARG BASEIMAGE=onnxruntimebuildcache.azurecr.io/internal/azureml/onnxruntime/build/cuda13_aarch64_almalinux8_gcc14_dotnet:20260329.1
FROM $BASEIMAGE
ARG TRT_VERSION
# For aarch64 tar-based TensorRT install
ARG TRT_DOWNLOAD_URL=""
ARG TENSORRT_ROOT=/opt/tensorrt

# Install TensorRT: use tar download for aarch64 since RPMs are not available
RUN set -eux; \
    if [ -z "${TRT_VERSION}" ]; then \
        echo "TRT_VERSION is empty; skipping TensorRT installation"; \
    elif [ -n "${TRT_DOWNLOAD_URL}" ]; then \
        echo "Installing TensorRT ${TRT_VERSION} from tar"; \
        mkdir -p /tmp/trt "${TENSORRT_ROOT}"; \
        curl -fsSL "${TRT_DOWNLOAD_URL}" -o /tmp/trt/tensorrt.tar.gz; \
        tar -xzf /tmp/trt/tensorrt.tar.gz -C /tmp/trt; \
        extracted_dir="$(find /tmp/trt -mindepth 1 -maxdepth 1 -type d | head -n 1)"; \
        cp -a "${extracted_dir}/." "${TENSORRT_ROOT}/"; \
        rm -rf /tmp/trt; \
        if [ -d "${TENSORRT_ROOT}/targets/sbsa-linux-gnu/lib" ] && [ ! -e "${TENSORRT_ROOT}/lib" ]; then \
            ln -s "${TENSORRT_ROOT}/targets/sbsa-linux-gnu/lib" "${TENSORRT_ROOT}/lib"; \
        fi; \
        if [ -d "${TENSORRT_ROOT}/targets/sbsa-linux-gnu/include" ] && [ ! -e "${TENSORRT_ROOT}/include" ]; then \
            ln -s "${TENSORRT_ROOT}/targets/sbsa-linux-gnu/include" "${TENSORRT_ROOT}/include"; \
        fi; \
    else \
        echo "TRT_VERSION is ${TRT_VERSION} but no TRT_DOWNLOAD_URL provided; skipping"; \
    fi

ENV TENSORRT_ROOT=${TENSORRT_ROOT}
ENV PATH=${TENSORRT_ROOT}/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=${TENSORRT_ROOT}/lib:${LD_LIBRARY_PATH}
ENV CPATH=${TENSORRT_ROOT}/include:${CPATH}

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

ENV CUDAHOSTCXX=/opt/rh/gcc-toolset-14/root/usr/bin/g++
ENV PATH=/usr/lib/jvm/msopenjdk-17/bin:$PATH
ENV JAVA_HOME=/usr/lib/jvm/msopenjdk-17
ARG BUILD_UID=1001
ARG BUILD_USER=onnxruntimedev
RUN adduser --uid $BUILD_UID $BUILD_USER
WORKDIR /home/$BUILD_USER
USER $BUILD_USER
