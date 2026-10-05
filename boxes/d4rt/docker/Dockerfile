# d4rt box - OpenD4RT 4D reconstruction and tracking.
#
# Two differences from the fleet's standard template, both forced by the model:
#
#   * the base image is PyTorch's CUDA runtime rather than python:slim, because
#     the upstream repo pins torch 2.6.0+cu124 and the model is ~1.16B
#     parameters - CPU inference runs but is not practical;
#   * the checkpoint (~4.6 GB fp32) is baked in at build time, so the image is
#     self-contained and a fleet never waits on a Hugging Face download. That
#     makes this the heaviest image in the registry. Pass
#     --build-arg DOWNLOAD_WEIGHTS=false to build a thin image and mount the
#     weights at /weights instead.
#
# Build:
#   cd boxes/d4rt
#   docker build --tag sipgisr/visionist-d4rt --build-arg SERVICE_NAME=d4rt -f docker/Dockerfile .
#
# Upstream is pinned by commit: this box imports helpers from the repo's
# scripts (_infer_tracks, _infer_point_cloud_ref0, _predict_camera_branches),
# which are private functions with no stability promise. Moving the pin is a
# deliberate act - re-test the box when you do.

ARG WORKSPACE=/workspace
ARG SERVICE_NAME=d4rt

# --------------------------------------
# Builder stage: generate the proto stubs
# --------------------------------------
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2 AS builder
ARG WORKSPACE
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools==1.74.0 protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
WORKDIR ${WORKSPACE}
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pipeline.proto


# --------------------------------------
# Weights stage: fetch the checkpoint once, in a layer of its own
# --------------------------------------
FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2 AS weights
ARG D4RT_VARIANT=OpenD4RT_48CLIP_9Mix_NoCropAUG
ARG HF_REPO=Lijiaxin0111/OpenD4RT
ARG DOWNLOAD_WEIGHTS=true

ENV HF_VARIANT=${D4RT_VARIANT} \
    HF_REPO_ID=${HF_REPO} \
    WANT_WEIGHTS=${DOWNLOAD_WEIGHTS}

RUN pip install --no-cache-dir huggingface_hub
COPY docker/fetch_weights.py /tmp/fetch_weights.py
RUN mkdir -p "/weights/${D4RT_VARIANT}" && python /tmp/fetch_weights.py


# --------------------------------------
# Runtime
# --------------------------------------
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
ARG WORKSPACE
ARG SERVICE_NAME
ARG D4RT_VARIANT=OpenD4RT_48CLIP_9Mix_NoCropAUG
# Pinned upstream commit - see the note at the top of this file.
ARG D4RT_COMMIT=403290a6e7ea6262a1f20f8c02d5461cd7b6c9b3
ARG D4RT_REPO_URL=https://github.com/Lijiaxin0111/Open-d4rt.git

ARG USER=runner
ARG GROUP=runner-group
ARG SRC=src

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    D4RT_REPO=/opt/opend4rt \
    D4RT_CKPT_DIR=/weights \
    D4RT_VARIANT=${D4RT_VARIANT} \
    HF_HOME=/workspace/.hf \
    TORCH_HOME=/workspace/.torch_cache

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# The upstream checkout. Its own requirements pin torch 2.6.0+cu124, which is
# what this base image already provides, so only the rest is installed.
RUN git clone --filter=blob:none ${D4RT_REPO_URL} /opt/opend4rt && \
    cd /opt/opend4rt && git checkout --quiet ${D4RT_COMMIT} && \
    rm -rf /opt/opend4rt/.git

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

COPY --from=weights /weights /weights

# Non-privileged user, after the big layers so they stay cacheable.
RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir -p ${WORKSPACE} ${WORKSPACE}/.hf ${WORKSPACE}/.torch_cache && \
    chown -R ${USER}:${GROUP} ${WORKSPACE} && \
    chown -R ${USER}:${GROUP} /weights

# The .proto at the root meets the AI4EU spec.
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/
COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

USER ${USER}
EXPOSE 8061
WORKDIR ${WORKSPACE}

CMD ["python3", "service.py"]
