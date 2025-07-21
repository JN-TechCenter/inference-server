# YOLO推理服务器 - 高并发架构设计

## 架构概述

这是一个基于FastAPI的高并发YOLO推理服务器，采用**高内聚低耦合**的设计原则，支持多用户并发推理请求。

### 🏗️ 架构分层

```
┌─────────────────────────────────────────┐
│              API层 (FastAPI)             │
│  - RESTful API                          │
│  - WebSocket支持                        │
│  - 中间件 (CORS, 监控)                   │
├─────────────────────────────────────────┤
│              业务逻辑层                   │
│  ┌──────────┬──────────┬──────────────┐ │
│  │ 请求管理器 │ 任务调度器 │   用户会话   │ │
│  │RequestMgr│TaskSched │SessionMgr    │ │
│  └──────────┴──────────┴──────────────┘ │
├─────────────────────────────────────────┤
│                服务层                    │
│  ┌──────────┬──────────┬──────────────┐ │
│  │ 推理服务  │ 模型管理器 │   资源池     │ │
│  │InferSvc  │ModelMgr  │ResourcePool  │ │
│  └──────────┴──────────┴──────────────┘ │
├─────────────────────────────────────────┤
│                核心层                    │
│  ┌──────────┬──────────┬──────────────┐ │
│  │ 模型引擎  │ 预处理器  │   后处理器   │ │
│  │ModelEng  │PreProc   │PostProc      │ │
│  └──────────┴──────────┴──────────────┘ │
└─────────────────────────────────────────┘
```

## 🎯 设计特点

### 高内聚
- **服务模块化**: 每个服务职责单一，功能内聚
- **接口标准化**: 统一的抽象接口定义
- **配置集中化**: 统一的配置管理

### 低耦合
- **依赖注入**: 服务间通过接口交互
- **事件驱动**: 观察者模式实现松耦合
- **服务注册**: 动态服务发现和管理

### 高并发
- **异步处理**: 基于asyncio的异步架构
- **资源池化**: 模型实例复用
- **队列管理**: 请求队列和工作器池
- **会话管理**: 用户会话和配额控制

## 📁 项目结构

```
yolo_inference_server/
├── app/
│   ├── main.py                 # FastAPI应用入口
│   ├── config_manager.py       # 配置管理
│   ├── performance_monitor.py  # 性能监控
│   ├── mindspore_compat.py    # MindSpore兼容层
│   ├── services/              # 服务层
│   │   ├── __init__.py
│   │   ├── base.py           # 基础服务接口
│   │   ├── model_manager.py  # 模型管理服务
│   │   ├── inference_service.py # 推理服务
│   │   └── session_manager.py   # 会话管理服务
│   └── core/                 # 核心组件
│       ├── interfaces.py     # 核心接口定义
│       ├── models.py         # 模型工厂
│       ├── preprocessors.py  # 预处理器
│       ├── postprocessors.py # 后处理器
│       └── visualizers.py    # 可视化组件
├── _configs/                 # 配置文件
├── models/                   # 模型配置
├── weights/                  # 模型权重
├── run_server.py            # 服务器启动脚本
├── test_client.py           # 客户端测试脚本
└── requirements.txt         # 依赖包
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务器

```bash
# 方法1: 使用启动脚本
python run_server.py

# 方法2: 直接运行
python -m app.main

# 方法3: 使用uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 验证服务

访问 `http://localhost:8000/docs` 查看API文档

### 4. 并发测试

```bash
# 创建测试图像并运行并发测试
python test_client.py --create-image --requests 100 --users 10
```

## 📊 核心服务详解

### 1. 模型管理服务 (ModelManagerService)

**职责**: 管理模型的生命周期、实例池化、资源分配

**特性**:
- 模型实例池化 (最大实例数可配置)
- 自动空闲实例清理
- 并发安全的实例获取/释放
- 动态模型重载

**配置示例**:
```python
ModelConfig(
    model_name="yolov5s",
    model_path="./weights/yolov5/yolov5s.ckpt",
    config_path="./models/yolov5s.yaml",
    max_instances=3,        # 最大实例数
    idle_timeout=300        # 空闲超时(秒)
)
```

### 2. 推理服务 (InferenceService)

**职责**: 处理推理请求、管理工作器池、任务调度

**特性**:
- 异步请求队列 (支持优先级)
- 工作器池 (可配置工作器数量)
- 请求超时处理
- 结果缓存和分发

**工作流程**:
1. 接收推理请求 → 入队
2. 工作器获取请求 → 获取模型实例
3. 执行推理 → 释放模型实例
4. 返回结果 → 发送给用户

### 3. 会话管理服务 (UserSessionManager)

**职责**: 用户会话管理、配额控制、并发限制

**特性**:
- 用户会话生命周期管理
- 每日/每小时请求配额
- 并发请求数限制
- 自动过期会话清理

**配额配置**:
```python
UserQuota(
    daily_limit=1000,       # 每日请求限制
    hourly_limit=100,       # 每小时请求限制
    concurrent_limit=5      # 并发请求限制
)
```

### 4. 性能监控 (PerformanceMonitor)

**职责**: 实时监控系统性能和应用指标

**监控指标**:
- 系统指标: CPU、内存、磁盘IO、网络IO
- 应用指标: 请求数、响应时间、错误率、队列大小
- 历史数据: 可配置保留时长的历史数据

## 🔧 配置管理

### 统一配置管理 (ConfigManager)

**特性**:
- 分层配置 (服务器、推理、会话、模型)
- YAML配置文件支持
- 运行时配置验证
- 动态配置保存

**配置文件示例** (`_configs/inference_config.yaml`):
```yaml
server:
  host: '0.0.0.0'
  port: 8000
  debug: true

inference:
  max_workers: 8
  queue_size: 1000
  timeout: 60.0

session:
  session_timeout: 3600
  daily_limit: 1000
  hourly_limit: 100

models:
  yolov5s:
    model_path: './weights/yolov5/yolov5s.ckpt'
    config_path: './models/yolov5s.yaml'
    max_instances: 3
```

## 🛡️ 并发安全设计

### 1. 线程安全
- 所有共享资源使用锁保护
- 异步锁 (asyncio.Lock) 用于异步操作
- 线程锁 (threading.Lock) 用于同步操作

### 2. 资源管理
- 模型实例池化避免重复加载
- 自动资源清理和回收
- 内存泄漏检测和预防

### 3. 错误处理
- 分层异常处理
- 优雅降级策略
- 详细错误日志

## 📈 性能优化策略

### 1. 模型层面
- 模型实例复用
- 批处理优化
- 混合精度推理

### 2. 服务层面
- 异步处理
- 请求队列
- 连接池

### 3. 系统层面
- 资源监控
- 自动扩缩容
- 负载均衡

## 🔌 API接口

### 核心推理接口

**目标检测**:
```http
POST /inference/detect
Content-Type: multipart/form-data

file: 图像文件
model_name: 模型名称 (默认: yolov5s)
confidence_threshold: 置信度阈值 (默认: 0.25)
iou_threshold: IOU阈值 (默认: 0.65)
return_image: 是否返回可视化图像 (默认: false)
```

**实例分割**:
```http
POST /inference/segment
# 参数同上
```

### 管理接口

**健康检查**:
```http
GET /health
```

**模型管理**:
```http
GET /models                    # 获取模型列表
POST /models/{model_name}/reload # 重载模型
```

**性能监控**:
```http
GET /metrics/performance      # 获取性能指标
POST /metrics/export         # 导出指标数据
```

**配置管理**:
```http
GET /config                  # 获取配置
POST /config/save           # 保存配置
GET /config/validate        # 验证配置
```

## 🧪 测试和部署

### 并发测试

使用提供的测试客户端进行压力测试:

```bash
# 基础测试
python test_client.py --requests 50 --users 5

# 高负载测试
python test_client.py --requests 500 --users 20

# 自定义测试
python test_client.py \
    --server http://localhost:8000 \
    --image your_image.jpg \
    --requests 100 \
    --users 10 \
    --model yolov5s
```

### 生产部署

**Docker部署**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "run_server.py"]
```

**Nginx负载均衡**:
```nginx
upstream yolo_backend {
    server yolo-inference-1:8000;
    server yolo-inference-2:8000;
    server yolo-inference-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://yolo_backend;
    }
}
```

## 🎯 扩展建议

### 1. 功能扩展
- 支持更多YOLO版本
- 视频流推理
- 批量图像处理
- 模型A/B测试

### 2. 性能扩展
- GPU支持
- 模型量化
- 动态批处理
- 缓存策略

### 3. 运维扩展
- 指标告警
- 日志聚合
- 分布式追踪
- 自动扩缩容

这个架构设计充分考虑了高并发、高可用、易扩展的需求，是一个生产就绪的YOLO推理服务解决方案。
