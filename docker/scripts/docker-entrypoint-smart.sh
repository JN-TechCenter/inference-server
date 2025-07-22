#!/bin/bash
# 智能启动脚本 - 运行时设备检测
set -e

echo "🚀 启动YOLO推理服务 (智能模式)..."

# 运行时设备检测
detect_runtime_device() {
    # 检查环境变量
    if [ "${DEVICE_TARGET:-}" = "GPU" ]; then
        echo "🎯 环境变量指定GPU模式"
        return 0
    fi
    
    if [ "${DEVICE_TARGET:-}" = "CPU" ]; then
        echo "🎯 环境变量指定CPU模式"
        return 1
    fi
    
    # 动态检测GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null 2>&1; then
            GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
            echo "✅ 检测到GPU: $GPU_INFO"
            export DEVICE_TARGET=GPU
            return 0
        fi
    fi
    
    echo "ℹ️  运行时未检测到GPU，使用CPU模式"
    export DEVICE_TARGET=CPU
    return 1
}

# 设置MindSpore环境
setup_mindspore_env() {
    echo "🔧 配置MindSpore环境..."
    
    if detect_runtime_device; then
        # GPU模式配置
        export MS_DEV_DEVICE_TARGET=GPU
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
        echo "🚀 MindSpore GPU模式已启用"
        echo "  GPU设备: $CUDA_VISIBLE_DEVICES"
    else
        # CPU模式配置
        export MS_DEV_DEVICE_TARGET=CPU
        export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$(nproc)}
        echo "🔄 MindSpore CPU模式已启用"
        echo "  CPU线程数: $OMP_NUM_THREADS"
    fi
    
    # 通用环境设置
    export PYTHONUNBUFFERED=1
    export QT_QPA_PLATFORM=offscreen
}

# 系统信息检查
check_system_info() {
    echo "🔍 检测运行环境..."
    echo "✅ Python版本: $(python3 --version)"
    echo "✅ 工作目录: $(pwd)"
    echo "✅ 系统架构: $(uname -m)"
    
    # 内存信息
    if command -v free &> /dev/null; then
        MEMORY_INFO=$(free -h | grep "Mem:" | awk '{print $3"/"$2}')
        echo "✅ 内存使用: $MEMORY_INFO"
    fi
}

# MindSpore功能测试
test_mindspore() {
    echo "🧪 测试MindSpore CPU支持..."
    python3 -c "
import mindspore as ms
import numpy as np
import sys

# 基本张量运算测试
try:
    a = ms.Tensor([1, 2, 3, 4], ms.float32)
    b = ms.Tensor([2, 3, 4, 5], ms.float32)
    result = a + b
    print('✅ MindSpore CPU测试成功')
    sys.exit(0)
except Exception as e:
    print(f'❌ MindSpore测试失败: {e}')
    sys.exit(1)
" || {
    echo "❌ MindSpore测试失败，退出"
    exit 1
}
}

# 权重文件检查
check_weights() {
    if [ -d "/app/weights" ]; then
        WEIGHT_COUNT=$(find /app/weights -name "*.ckpt" | wc -l)
        echo "✅ 权重目录存在: $WEIGHT_COUNT 个文件"
    else
        echo "⚠️  权重目录不存在，请挂载权重文件"
    fi
    
    # 检查测试图片
    if [ -f "/app/test_image.jpg" ] || [ -d "/app/test_images" ]; then
        echo "✅ 测试图片存在"
    else
        echo "ℹ️  无测试图片"
    fi
}

# 显示运行配置
show_config() {
    echo "📋 运行配置:"
    echo "  设备类型: ${DEVICE_TARGET:-AUTO}"
    echo "  模式: ${DEVICE_DETECTION_MODE:-smart}"
    
    if [ "${DEVICE_TARGET:-}" = "GPU" ]; then
        echo "  GPU设备: ${CUDA_VISIBLE_DEVICES:-0}"
    else
        echo "  CPU线程: ${OMP_NUM_THREADS:-$(nproc)}"
        echo "  内存使用: $(free -h | grep "Mem:" | awk '{print $3"/"$2}' 2>/dev/null || echo 'Unknown')"
    fi
}

# 启动推理服务
start_inference_service() {
    echo "🚀 启动${DEVICE_TARGET:-智能}推理服务..."
    
    # 执行传入的命令
    if [ $# -eq 0 ]; then
        echo "▶️  执行默认推理测试: python3 app/src/core/predict.py --image_path test_image.jpg"
        exec python3 app/src/core/predict.py --image_path test_image.jpg
    else
        echo "▶️  执行命令: $*"
        exec "$@"
    fi
}

# 主函数
main() {
    # 系统检查
    check_system_info
    
    # MindSpore环境设置
    setup_mindspore_env
    
    # 功能测试
    test_mindspore
    
    # 资源检查
    check_weights
    
    # 显示配置
    show_config
    
    # 启动服务
    start_inference_service "$@"
}

# 执行主函数
main "$@"
