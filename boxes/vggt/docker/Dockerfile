# Define workdir folder for all stages
# Must be renewed in the beggining of each stage
ARG WORKSPACE=/workspace
# This argument names the server entry point:
# the file copied in is ${SERVICE_NAME}_service.py -> service.py
ARG SERVICE_NAME=vggt


# --------------------------------------
# Builder stage to generate .proto files
# --------------------------------------

FROM python:slim-buster@sha256:c46b0ae5728c2247b99903098ade3176a58e274d9c7d2efeaaab3e0621a53935 AS builder
#FROM python:latest as builder
## Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

# Path for the protos folder to copy
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools==1.74.0 protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
WORKDIR ${WORKSPACE}

# Compile proto file and remove it
RUN python -m grpc_tools.protoc -I. --python_out=.  --grpc_python_out=. pipeline.proto


# --------------------------------------
# Weights stage — download the VGGT 1B checkpoint once at build time
# (the service also downloads on startup if the file is absent)
# --------------------------------------

FROM python:slim-bookworm AS weights
ARG VGGT_WEIGHTS_REPO=facebook/VGGT-1B
ARG VGGT_WEIGHTS_FILENAME=model.pt
ENV VGGT_WEIGHTS_REPO=${VGGT_WEIGHTS_REPO} VGGT_WEIGHTS_FILENAME=${VGGT_WEIGHTS_FILENAME}

RUN pip install --no-cache-dir huggingface_hub && \
    python -c "import shutil; from huggingface_hub import hf_hub_download; \
    p = hf_hub_download(repo_id='$VGGT_WEIGHTS_REPO', filename='$VGGT_WEIGHTS_FILENAME', local_dir='/weights'); \
    shutil.move(p, '/weights/vggt-1b.pt')"   # runtime name used by the service


# --------------------------------------
# Vendor stage — fetch the VGGT codebase (not committed to this repo)
# --------------------------------------

FROM python:slim-bookworm AS vendor
ARG VGGT_CODE_REPO=https://github.com/facebookresearch/vggt
ARG VGGT_CODE_REV=main

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends git ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    (git clone --depth 1 --branch ${VGGT_CODE_REV} ${VGGT_CODE_REPO} /vggt 2>/dev/null \
     || git clone --depth 1 ${VGGT_CODE_REPO} /vggt) && \
    rm -rf /vggt/.git


# -----------------------------
# Stage to generate final image
# -----------------------------

FROM nvidia/cuda:12.2.2-base-ubuntu22.04@sha256:1a8a738e81d4adbef0c709241f5238cec5bb77186dcb5b2103db293315ed42d1
# Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

ARG USER=runner
ARG GROUP=runner-group
ARG SRC=src


# Create non-privileged user and workspace
RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir ${WORKSPACE} && \
    chown -R ${USER}:${GROUP} ${WORKSPACE}

# Install requirements
COPY requirements.txt .
RUN apt update -y && \
    apt install -y pip ffmpeg libsm6 libxext6 &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && \
    # Install headless version of opencv-python for server usage
    # Does not install graphical modules
    # See https://github.com/opencv/opencv-python#installation-and-usage
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# COPY .proto file to root to meet ai4eu specifications
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /

# Copy generated .py files to workspace
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

# Copy the service file and the utils to workspace
# (rename service file to only service.py for generic usage)
COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

# Change to non-privileged user
USER ${USER}

# Expose port 8061 according to ai4eu specifications
EXPOSE 8061

WORKDIR ${WORKSPACE}

# -----------------------------
# Specific steps for vggt service
# -----------------------------

# VGGT codebase (from the vendor stage above)
COPY --from=vendor --chown=${USER}:${GROUP} /vggt ${WORKSPACE}/vggt
RUN pip install -r ${WORKSPACE}/vggt/requirements.txt

# Baked-in VGGT weights (from the weights stage above)
COPY --from=weights --chown=${USER}:${GROUP} /weights/vggt-1b.pt ${WORKSPACE}/vggt-1b.pt

# set $TORCH_HOME for torchhub cache
ENV TORCH_HOME=${WORKSPACE}/.cache
ENV HF_HOME=${WORKSPACE}/.cache
ENV HF_HUB_CACHE=${WORKSPACE}/.cache

CMD ["python3", "service.py"]