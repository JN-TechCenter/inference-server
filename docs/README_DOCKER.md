# 智能Docker推理服务器解决方案

## 概述
本解决方案为推理服务器提供智能GPU检测和CPU回退机制，支持多种部署模式。

## 核心特性

### 🔧 智能设备检测
- **自动GPU检测**: 运行时自动检测NVIDIA GPU可用性
- **CPU回退机制**: 当GPU不可用时自动切换到CPU模式  
- **动态MindSpore配置**: 根据硬件自动选择最优设备

### 🐳 多阶段Docker构建
- **GPU基础镜像**: NVIDIA CUDA 11.8 + Ubuntu 20.04
- **CPU基础镜像**: Ubuntu 20.04 (轻量级)
- **智能镜像选择**: 构建时自动选择合适的基础镜像

### 🌐 TUNA镜像加速
- **APT包管理**: 使用清华大学TUNA镜像源
- **PyPI包管理**: 加速Python包下载
- **国内网络优化**: 显著提升构建速度

## 部署模式

### 1. 智能模式 (推荐)
```bash
# 启动智能检测模式 - 端口8000
docker-compose up -d inference-server
```
- 自动检测GPU可用性
- GPU优先，CPU回退
- 运行时动态设备选择

### 2. CPU强制模式
```bash
# 强制使用CPU - 端口8002  
docker-compose --profile cpu-only up -d
```
- 强制使用CPU计算
- 适用于无GPU环境
- 确保稳定性

### 3. GPU显式模式
```bash
# 显式使用GPU - 端口8001
docker-compose --profile gpu-explicit up -d
```
- 强制使用GPU
- 适用于GPU环境
- 最大性能

## 快速启动

### 前提条件
1. **Docker Desktop**: 确保Docker Desktop正在运行
2. **GPU驱动** (可选): 如需GPU支持，安装NVIDIA驱动

### 启动步骤
1. **启动Docker Desktop**
   ```powershell
   # 确保Docker Desktop正在运行
   docker info
   ```

2. **使用便捷脚本启动**
   ```bash
   ./start-docker-intelligent.sh
   ```
   或手动选择模式：
   
3. **验证部署**
   ```bash
   # 检查容器状态
   docker-compose ps
   
   # 查看日志
   docker-compose logs -f inference-server
   ```

## 配置文件说明

### Dockerfile
- 多阶段构建支持GPU/CPU
- TUNA镜像源加速
- 智能依赖安装

### docker-compose.yml  
- 三种服务配置
- Profile系统支持
- 端口映射: 8000/8001/8002

### 脚本文件
- `install_dependencies.sh`: 智能依赖安装
- `docker-entrypoint-smart.sh`: 运行时设备检测
- `start-docker-intelligent.sh`: 交互式启动

## 测试验证

### 配置测试
```bash
# 完整配置验证
python test_docker_config.py

# 简化验证(无需Docker运行)
python test_docker_simple.py
```

### 服务测试
```bash
# 健康检查
curl http://localhost:8000/health

# 设备信息
curl http://localhost:8000/device-info
```

## 故障排除

### Docker未运行
```bash
# 启动Docker Desktop
# Windows: 从开始菜单启动Docker Desktop
# 验证: docker info
```

### GPU检测失败
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查Docker GPU支持
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### 端口冲突
```bash
# 检查端口占用
netstat -ano | findstr ":8000"

# 修改docker-compose.yml中的端口映射
```

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    智能Docker解决方案                        │
├─────────────────────────────────────────────────────────────┤
│ 多阶段构建                                                   │
│ ├─ GPU镜像: nvidia/cuda:11.8-devel-ubuntu20.04            │
│ └─ CPU镜像: ubuntu:20.04                                   │
├─────────────────────────────────────────────────────────────┤
│ 运行时检测                                                   │
│ ├─ GPU检测: nvidia-smi, CUDA可用性                        │
│ ├─ 回退机制: GPU失败 → CPU模式                            │
│ └─ 环境设置: 设备目标, MindSpore配置                       │
├─────────────────────────────────────────────────────────────┤
│ 部署模式                                                     │
│ ├─ 智能模式: 端口8000, 自动检测                           │
│ ├─ CPU模式:  端口8002, 强制CPU                            │
│ └─ GPU模式:  端口8001, 强制GPU                            │
└─────────────────────────────────────────────────────────────┘
```

## 下一步
1. 启动Docker Desktop
2. 运行配置测试验证
3. 选择适合的部署模式
4. 验证服务健康状态

---
*智能GPU检测 + CPU回退 + 多模式部署 = 完美的Docker解决方案* 🚀
