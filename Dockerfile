FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG VENV=/venv
ARG PYTHON=$VENV/bin/python3

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    python3.10-venv \
    python3-pip \
    make \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /segmentation

COPY requirements requirements

RUN python3 -m venv $VENV
RUN $PYTHON -m pip install --timeout 3600 --no-cache-dir -r requirements/requirements-gpu.txt
RUN $PYTHON -m pip install --timeout 1200 --no-cache-dir -r requirements/requirements-base.txt

COPY . /segmentation/

CMD make run_training
