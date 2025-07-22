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

RUN python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    AUTO_DEVICE_SELECTION=true \
    DEVICE_DETECTION_MODE=smart

COPY docker/scripts/install_dependencies.sh .
RUN chmod +x install_dependencies.sh && ./install_dependencies.sh

COPY . .

RUN mkdir -p /app/runs_infer /app/logs /app/outputs

COPY docker/scripts/docker-entrypoint-smart.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint-smart.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import mindspore; print('OK')" || exit 1

ENTRYPOINT ["docker-entrypoint-smart.sh"]
CMD ["python3", "app/src/core/predict.py"]
