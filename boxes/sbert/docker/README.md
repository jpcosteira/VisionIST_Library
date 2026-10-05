# SBERT gRPC Docker integration


## Overview

This directory provides Dockerfiles to be able to run the gRPC service with Docker.
The docker image already has all the project dependencies installed.
Also, it is already built with the necessary sources for the gRPC service to run.


## Usage

In order to use the images, execute the following command:

```shell
$ docker run --rm --gpus all -p 8061:8061 -e PORT=8061 sipgisr/textembedding
```

NOTE: The `<path to optional host directory>` must be the absolute path to some directory needed to run the service (it is optional).


## Building the image

All boxes share the same proto file (`pipeline.proto`); only the service file
(`${SERVICE_NAME}_service.py`) differs.

In order to build the image, execute the respective command *(from the box root
directory, i.e. this folder's parent)*:

```shell
$ docker build --tag sipgisr/textembedding --build-arg SERVICE_NAME=sbert -f docker/Dockerfile .
```
