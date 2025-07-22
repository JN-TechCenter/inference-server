#!/bin/bash
# 智能Docker启动脚本 - GPU优先，CPU降级
echo "🚀 YOLO推理服务器智能启动脚本"
echo "=================================="

# 检查Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker未安装，请先安装Docker"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "❌ Docker服务未启动，请启动Docker Desktop"
        exit 1
    fi
    
    echo "✅ Docker环境正常"
}

# 检测GPU支持
detect_gpu() {
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        echo "✅ 检测到NVIDIA GPU: $GPU_INFO"
        return 0
    fi
    
    echo "ℹ️  未检测到NVIDIA GPU"
    return 1
}

# 检测Docker Compose
check_compose() {
    if command -v docker-compose &> /dev/null; then
        echo "✅ 使用 docker-compose"
        COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        echo "✅ 使用 docker compose"
        COMPOSE_CMD="docker compose"
    else
        echo "❌ Docker Compose未安装"
        exit 1
    fi
}

# 创建必要目录
prepare_directories() {
    echo "📁 准备目录结构..."
    mkdir -p outputs test_images logs weights configs
    echo "✅ 目录准备完成"
}

# 显示启动选项
show_options() {
    echo ""
    echo "🚀 启动模式选择:"
    echo "1) 智能模式 (推荐) - 自动检测GPU/CPU"
    echo "2) CPU专用模式 - 强制使用CPU"
    echo "3) GPU强制模式 - 强制使用GPU (需要NVIDIA GPU)"
    echo "4) 构建并启动智能模式"
    echo "5) 查看服务状态"
    echo "6) 停止所有服务"
    echo ""
}

# 启动智能模式
start_smart_mode() {
    echo "🧠 启动智能模式..."
    $COMPOSE_CMD up -d inference-server
    echo "✅ 智能推理服务已启动"
    echo "📍 访问地址: http://localhost:8000"
}

# 启动CPU模式
start_cpu_mode() {
    echo "🔄 启动CPU专用模式..."
    $COMPOSE_CMD --profile cpu-only up -d inference-server-cpu
    echo "✅ CPU推理服务已启动"
    echo "📍 访问地址: http://localhost:8002"
}

# 启动GPU模式
start_gpu_mode() {
    if detect_gpu; then
        echo "🚀 启动GPU强制模式..."
        $COMPOSE_CMD --profile gpu-explicit up -d inference-server-gpu
        echo "✅ GPU推理服务已启动"
        echo "📍 访问地址: http://localhost:8001"
    else
        echo "❌ 未检测到GPU，无法启动GPU模式"
        echo "🔄 建议使用智能模式或CPU模式"
        return 1
    fi
}

# 构建并启动
build_and_start() {
    echo "🔨 构建并启动智能模式..."
    $COMPOSE_CMD build --no-cache inference-server
    $COMPOSE_CMD up -d inference-server
    echo "✅ 构建完成，智能推理服务已启动"
    echo "📍 访问地址: http://localhost:8000"
}

# 查看状态
show_status() {
    echo "📊 服务状态:"
    $COMPOSE_CMD ps
    echo ""
    echo "📋 容器日志 (最近20行):"
    $COMPOSE_CMD logs --tail=20
}

# 停止服务
stop_services() {
    echo "🛑 停止所有推理服务..."
    $COMPOSE_CMD --profile gpu-explicit --profile cpu-only down
    echo "✅ 所有服务已停止"
}

# 主函数
main() {
    check_docker
    check_compose
    prepare_directories
    
    if [ $# -eq 0 ]; then
        # 交互模式
        show_options
        read -p "请选择 (1-6): " choice
    else
        # 命令行参数模式
        choice=$1
    fi
    
    case $choice in
        1|smart)
            start_smart_mode
            ;;
        2|cpu)
            start_cpu_mode
            ;;
        3|gpu)
            start_gpu_mode
            ;;
        4|build)
            build_and_start
            ;;
        5|status)
            show_status
            ;;
        6|stop)
            stop_services
            ;;
        *)
            echo "❌ 无效选择，默认启动智能模式"
            start_smart_mode
            ;;
    esac
    
    echo ""
    echo "📋 管理命令:"
    echo "  查看日志: $COMPOSE_CMD logs -f"
    echo "  停止服务: $COMPOSE_CMD down"
    echo "  重启服务: $COMPOSE_CMD restart"
    echo "  查看状态: $COMPOSE_CMD ps"
}

# 执行主函数
main "$@"
