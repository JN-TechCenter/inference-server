# YOLO智能推理服务器 - GPU优先，CPU降级
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as gpu-base

FROM ubuntu:20.04 as cpu-base

FROM ${BASE_IMAGE:-cpu-base} as final

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    wget \
    curl \
    git \
    libglib2.0-0 \
    libgomp1 \
    libjpeg-dev \
    libpng-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# 设置pip默认镜像源并升级
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN python3 -m pip install --upgrade pip

WORKDIR /app

# 设置应用环境变量
ENV PYTHONUNBUFFERED=1 \
    AUTO_DEVICE_SELECTION=true \
    DEVICE_DETECTION_MODE=smart

# 安装基础Python依赖
COPY requirements.txt .
RUN echo "正在安装Python依赖..." && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# 只复制必需的应用文件
COPY app/ ./app/
COPY weights/ ./weights/

