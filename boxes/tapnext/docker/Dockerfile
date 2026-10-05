ARG WORKSPACE=/workspace
ARG SERVICE_NAME=tapnext

FROM python:slim-buster@sha256:c46b0ae5728c2247b99903098ade3176a58e274d9c7d2efeaaab3e0621a53935 AS builder
ARG WORKSPACE
ARG SERVICE_NAME
ARG PROTOS_FOLDER_DIR=protos

RUN pip install --upgrade pip && \
    pip install grpcio grpcio-tools protobuf

COPY ${PROTOS_FOLDER_DIR} ${WORKSPACE}/
WORKDIR ${WORKSPACE}

RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. pipeline.proto

FROM nvidia/cuda:12.2.2-base-ubuntu22.04@sha256:1a8a738e81d4adbef0c709241f5238cec5bb77186dcb5b2103db293315ed42d1
ARG WORKSPACE
ARG SERVICE_NAME=tapnext
ARG USER=runner
ARG GROUP=runner-group
ARG SRC=src

RUN addgroup --system ${GROUP} && \
    adduser --system --no-create-home --ingroup ${GROUP} ${USER} && \
    mkdir ${WORKSPACE} && \
    chown -R ${USER}:${GROUP} ${WORKSPACE}

COPY requirements.txt .
RUN apt update -y && \
    apt install -y curl pip ffmpeg libsm6 libxext6 git &&\
    apt-get clean && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm requirements.txt

# Clone TAPNet repository
RUN git clone https://github.com/google-deepmind/tapnet.git ${WORKSPACE}/tapnet
RUN pip install --no-cache-dir ${WORKSPACE}/tapnet/

COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/pipeline.proto /
COPY --from=builder --chown=${USER}:${GROUP} ${WORKSPACE}/*.py ${WORKSPACE}/

COPY --chown=${USER}:${GROUP} ${SRC} ${WORKSPACE}
COPY --chown=${USER}:${GROUP} ${SRC}/${SERVICE_NAME}_service.py ${WORKSPACE}/service.py

# Download TAPNext checkpoint (740MB)
RUN curl -s -S -o /workspace/bootstapnext_ckpt.npz https://storage.googleapis.com/dm-tapnet/tapnext/bootstapnext_ckpt.npz && echo "Checkpoint downloaded successfully" || (echo "Checkpoint download failed: $?" && exit 1)

USER ${USER}

EXPOSE 8061

WORKDIR ${WORKSPACE}

ENV TORCH_HOME=${WORKSPACE}/.cache
ENV HF_HOME=${WORKSPACE}/.cache
ENV HF_HUB_CACHE=${WORKSPACE}/.cache
ENV PYTHONPATH=${WORKSPACE}:${WORKSPACE}/tapnet:${PYTHONPATH}

RUN echo "Torch hub cache path: $TORCH_HOME"
RUN echo "Python path: $PYTHONPATH"

CMD ["python3", "service.py"]

