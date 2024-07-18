ARG DEBIAN_FRONTEND=noninteractive
FROM nvidia/cuda:12.4.0-runtime-ubuntu20.04

RUN DEBIAN_FRONTEND=noninteractive \
  apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3-pip git curl vim iputils-ping \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

ARG GH_TOKEN
ARG GH_TOKEN_PUB
RUN pip3 install --upgrade setuptools pip
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt