#!/bin/bash
# 智能依赖安装脚本 - GPU优先，CPU降级
set -e

echo "🔍 开始智能依赖检测与安装..."

# 检测CUDA是否可用
detect_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            echo "✅ 检测到NVIDIA GPU，启用GPU模式"
            return 0
        fi
    fi
    
    if [ -d "/usr/local/cuda" ] || [ -d "/opt/cuda" ]; then
        echo "✅ 检测到CUDA环境，启用GPU模式"
        return 0
    fi
    
    echo "ℹ️  未检测到GPU，使用CPU模式"
    return 1
}

# 安装基础Python包
install_base_packages() {
    echo "📦 安装基础Python包..."
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
        numpy>=1.21.0,\<1.25.0 \
        opencv-python-headless>=4.8.0 \
        PyYAML>=6.0 \
        tqdm>=4.60.0 \
        matplotlib>=3.5.0 \
        Pillow>=8.0.0 \
        psutil>=5.8.0 \
        fastapi>=0.68.0 \
        uvicorn[standard]>=0.15.0 \
        ipython>=7.25.0
}

# 安装GPU版本MindSpore
install_gpu_mindspore() {
    echo "🚀 安装GPU版本MindSpore..."
    
    # 尝试安装GPU版本
    if pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ mindspore-gpu>=2.5.0; then
        echo "✅ GPU版本MindSpore安装成功"
        
        # 安装MindYOLO
        pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ mindyolo>=0.5.0
        
        # 设置GPU环境变量
        export DEVICE_TARGET=GPU
        export CUDA_VISIBLE_DEVICES=0
        echo "export DEVICE_TARGET=GPU" >> /etc/environment
        echo "export CUDA_VISIBLE_DEVICES=0" >> /etc/environment
        
        return 0
    else
        echo "❌ GPU版本MindSpore安装失败，降级到CPU版本"
        return 1
    fi
}

# 安装CPU版本MindSpore
install_cpu_mindspore() {
    echo "🔄 安装CPU版本MindSpore..."
    
    pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ \
        mindspore>=2.5.0 \
        mindyolo>=0.5.0
    
    # 设置CPU环境变量
    export DEVICE_TARGET=CPU
    echo "export DEVICE_TARGET=CPU" >> /etc/environment
    echo "✅ CPU版本MindSpore安装成功"
}

# 验证安装
verify_installation() {
    echo "🧪 验证安装..."
    
    python3 -c "
import mindspore as ms
import mindyolo
import cv2
import numpy as np

print(f'✅ MindSpore版本: {ms.__version__}')
print(f'✅ MindYOLO版本: {mindyolo.__version__}')
print(f'✅ OpenCV版本: {cv2.__version__}')
print(f'✅ NumPy版本: {np.__version__}')

# 测试设备
try:
    context = ms.get_context()
    device_target = context.get('device_target', 'Unknown')
    print(f'✅ 设备目标: {device_target}')
except:
    print('ℹ️  设备信息获取失败，使用默认设置')
"
}

# 主安装流程
main() {
    echo "🚀 YOLO推理服务器智能依赖安装"
    echo "=================================="
    
    # 安装基础包
    install_base_packages
    
    # 智能GPU/CPU检测与安装
    if detect_cuda; then
        if ! install_gpu_mindspore; then
            echo "🔄 GPU安装失败，回退到CPU模式..."
            install_cpu_mindspore
        fi
    else
        install_cpu_mindspore
    fi
    
    # 验证安装
    verify_installation
    
    echo "🎉 依赖安装完成！"
}

# 执行主函数
main
