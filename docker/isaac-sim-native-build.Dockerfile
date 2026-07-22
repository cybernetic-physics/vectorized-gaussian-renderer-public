ARG CUDA_DEVEL_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43
ARG ISAAC_SIM_IMAGE=nvcr.io/nvidia/isaac-sim:6.0.1@sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9

FROM ${CUDA_DEVEL_IMAGE} AS cuda-devel
FROM ${ISAAC_SIM_IMAGE} AS isaac-native-build

USER root

COPY --from=cuda-devel /usr/local/cuda-12.8 /usr/local/cuda-12.8

RUN ln -sfn /usr/local/cuda-12.8 /usr/local/cuda \
    && apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

ENV CUDA_HOME=/usr/local/cuda-12.8 \
    CUDACXX=/usr/local/cuda-12.8/bin/nvcc \
    CC=/usr/bin/gcc \
    CXX=/usr/bin/g++ \
    PATH=/usr/local/cuda-12.8/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

LABEL org.opencontainers.image.description="Isaac Sim 6.0.1 with the CUDA 12.8 native-extension toolchain"

USER 1234:1234
