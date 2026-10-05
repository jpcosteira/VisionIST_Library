ARG WORKSPACE=/workspace

# =============================================================================
# Stage 1: Builder - compile protobuf
# =============================================================================
FROM python:3.10-slim AS builder

ARG WORKSPACE=/workspace

WORKDIR ${WORKSPACE}

RUN pip install --upgrade pip && pip install grpcio grpcio-tools protobuf

COPY protos ${WORKSPACE}/
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pipeline.proto

# =============================================================================
# Stage 2: Runtime - CUDA base with all dependencies
# =============================================================================
FROM nvidia/cuda:12.2.2-base-ubuntu22.04

ARG WORKSPACE=/workspace
ARG USER=runner

# Create non-root user and workspace
RUN addgroup --system runner-group && \
    adduser --system --no-create-home --ingroup runner-group runner

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        git \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /home/runner ${WORKSPACE} ${WORKSPACE}/.cache/huggingface ${WORKSPACE}/.cache/triton && \
    chown -R runner:runner-group /home/runner ${WORKSPACE}

COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu122 \
        -r requirements.txt

# Copy compiled protobuf and service
COPY --from=builder /workspace/*.py /workspace/
COPY --from=builder /workspace/*.proto /workspace/protos/
COPY src/moge_service.py /workspace/service.py
COPY protos/pipeline.proto /workspace/

# Set environment
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=${WORKSPACE}

# HuggingFace cache directory (needs write access)
ENV HF_HOME=${WORKSPACE}/.cache/huggingface
ENV TRANSFORMERS_CACHE=${WORKSPACE}/.cache/transformers
ENV HF_HOME=/workspace/.cache/huggingface

# Triton kernel cache: MoGe-3 (flex_gemm) compiles Triton launchers at runtime.
# The container's $HOME (/home/runner) does not exist, so point the cache at
# the writable workspace. (gcc/g++ above are required for the same step.)
ENV TRITON_CACHE_DIR=${WORKSPACE}/.cache/triton

# Expose port
EXPOSE 8061

# Run as non-root
USER runner
WORKDIR ${WORKSPACE}

# Start service
CMD ["python3", "service.py"]
