
ARG DEBIAN_FRONTEND=noninteractive
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 AS fastchat-base

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3-pip git curl vim iputils-ping \
  && rm -rf /var/lib/apt/lists/*

ARG GH_TOKEN
ARG GH_TOKEN_PUB

WORKDIR /app

COPY ./requirements.txt requirements.txt
RUN pip3 install --upgrade setuptools pip
RUN python3.9 -m pip install -r requirements.txt
# RUN python3.9 -m pip install -r fs-rina-requirements.txt

COPY . /app


# # Install vLLM with CUDA 11.8.
FROM fastchat-base AS image-build
# export VLLM_VERSION=0.2.4
# export PYTHON_VERSION=39
RUN python3.9 -m pip install https://github.com/vllm-project/vllm/releases/download/v0.2.4/vllm-0.2.4+cu118-cp39-cp39-manylinux1_x86_64.whl

RUN python3.9 -m pip uninstall torch -y
RUN python3.9 -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

RUN python3.9 -m pip uninstall xformers -y
RUN python3.9 -m pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

RUN python3.9 -m pip freeze > freeze-requirements.txt