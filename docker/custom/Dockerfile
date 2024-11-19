
ARG DEBIAN_FRONTEND=noninteractive
FROM nvidia/cuda:12.5.0-devel-ubuntu20.04

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3-pip git curl vim iputils-ping \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

ARG GH_TOKEN
ARG GH_TOKEN_PUB

RUN pip3 install --upgrade setuptools pip
RUN pip3 install torch==2.1.0 torchvision torchaudio sentence_transformers
RUN pip3 install xformers

# RUN python3.9 -m pip install --no-cache-dir -r requirements.txt
RUN python3.9 -m pip install -r requirements.txt

# For Dolomite Engine
RUN python3.9 -m pip install flash-attn --no-build-isolation
RUN python3.9 -m pip install tokenizers==0.19.1

