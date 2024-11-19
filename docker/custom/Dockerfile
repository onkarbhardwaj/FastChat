
ARG DEBIAN_FRONTEND=noninteractive
FROM nvidia/cuda:12.5.0-devel-ubuntu20.04

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3.9-distutils curl python3-pip git vim iputils-ping \
  && rm -rf /var/lib/apt/lists/*

ARG GH_TOKEN
ARG GH_TOKEN_PUB


RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

RUN pip3 install git+https://${GH_TOKEN_PUB}@github.com/onkarbhardwaj/FastChat.git@dolomite#egg=fschat[fschat]
RUN pip3 install git+https://${GH_TOKEN_PUB}@github.com/onkarbhardwaj/FastChat.git@dolomite#egg=fschat[model_worker]


