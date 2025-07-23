# YOLO智能推理服务器 - CUDA 11.6 GPU版本
FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 配置清华镜像源以提高下载速度
RUN sed -i 's@http://archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list \
 && sed -i 's@http://security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list

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
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 设置pip默认镜像源并升级
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/ \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

RUN python3 -m pip install --upgrade pip

WORKDIR /app

# 设置应用环境变量 - CUDA 11.6 GPU模式
ENV PYTHONUNBUFFERED=1 \
    DEVICE_TYPE=gpu \
    CUDA_VISIBLE_DEVICES=0 \
    MINDSPORE_DEVICE_TARGET=GPU \
    CUDA_VERSION=11.6

# 安装基础Python依赖
COPY requirements.txt .
RUN echo "正在安装Python依赖..." && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# 安装MindSpore GPU Nightly版本
RUN echo "正在安装MindSpore GPU Nightly版本..." && \
    python3 -m pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装MindYOLO（基于GPU版MindSpore）
RUN echo "正在安装MindYOLO..." && \
    python3 -m pip install --no-cache-dir mindyolo -i https://pypi.tuna.tsinghua.edu.cn/simple

# 只复制必需的应用文件
COPY app/ ./app/
COPY weights/ ./weights/
COPY docker/scripts/docker-entrypoint-smart.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint-smart.sh

# 创建必要的目录
RUN mkdir -p /app/runs_infer /app/logs /app/outputs

EXPOSE 8000

# GPU健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["docker-entrypoint-smart.sh"]
CMD ["python3", "app/src/core/predict.py"]

