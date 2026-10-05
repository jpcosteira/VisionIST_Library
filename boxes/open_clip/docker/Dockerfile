# Define workdir folder for all stages
# Must be renewed in the beggining of each stage
ARG WORKSPACE=/workspace
# SERVICE_NAME identifies the python file with the server
# The python file should have the name ${SERVICE_NAME}_service.py
# All boxes share the same proto file: pipeline.proto
ARG SERVICE_NAME=open_clip


# --------------------------------------
# Builder stage to generate .proto files
# --------------------------------------

FROM python:3.10-slim@sha256:fd76ade0c607f27677bc04be3c60749f400eedc941d9e72967e19a4cedff80c2 AS builder
## Renew build args
ARG WORKSPACE
ARG SERVICE_NAME

# Path for the protos folder to copy
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools==1.74.0 protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
#COPY image_generic.proto ${WORKSPACE}/
WORKDIR ${WORKSPACE}

# Compile the shared proto file
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pipeline.proto


# -----------------------------
# Stage to generate final image
# -----------------------------

#FROM python:slim-buster
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

# Install requirements. torch's PyPI wheel bundles the CUDA runtime, so the same
# image runs on a GPU or, much more slowly, on CPU.
COPY requirements.txt .
RUN apt update -y && \
    apt install -y --no-install-recommends python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# COPY the shared .proto file to root to meet ai4eu specifications
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /

# Copy generated .py files to workspace
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

# Copy the service file and the utils to workspace
# (rename service file to only service.py for generic usage)
COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

# -----------------------------
# open_clip specifics
# -----------------------------

# Hugging Face / torch caches live in the workspace so the unprivileged user
# can write to them when a request asks for a model that is not baked in.
ENV TORCH_HOME=${WORKSPACE}/.cache
ENV HF_HOME=${WORKSPACE}/.cache
ENV HF_HUB_CACHE=${WORKSPACE}/.cache

# The model a request gets when it names none (also read by the service).
# Its weights are baked into the image so the first call does not download
# them; build with --build-arg PRELOAD_WEIGHTS=false for a thin image and let
# the box fetch on demand (or mount a cache at /workspace/.cache).
ARG OPEN_CLIP_MODEL=ViT-B-32
ARG OPEN_CLIP_PRETRAINED=laion2b_s34b_b79k
ARG PRELOAD_WEIGHTS=true
ENV OPEN_CLIP_MODEL=${OPEN_CLIP_MODEL}
ENV OPEN_CLIP_PRETRAINED=${OPEN_CLIP_PRETRAINED}

RUN mkdir -p ${HF_HOME} && \
    if [ "${PRELOAD_WEIGHTS}" = "true" ]; then \
        python3 -c "import open_clip; open_clip.create_model_and_transforms('${OPEN_CLIP_MODEL}', pretrained='${OPEN_CLIP_PRETRAINED}')"; \
    fi && \
    chown -R ${USER}:${GROUP} ${HF_HOME}

# Change to non-privileged user
USER ${USER}

# Expose port 8061 according to ai4eu specifications
EXPOSE 8061

WORKDIR ${WORKSPACE}

CMD ["python3", "service.py"]
