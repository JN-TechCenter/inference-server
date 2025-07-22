#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能Docker解决方案 - 最终验证测试
检查所有配置文件并提供启动指导
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_docker_status():
    """检查Docker状态"""
    print("🐳 Docker环境检查:")
    try:
        # 检查Docker版本
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  ✅ Docker版本: {result.stdout.strip()}")
        else:
            print(f"  ❌ Docker版本检查失败: {result.stderr}")
            return False
            
        # 检查Docker Compose版本
        result = subprocess.run(['docker', 'compose', 'version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"  ✅ Docker Compose版本: {result.stdout.strip()}")
        else:
            print(f"  ❌ Docker Compose版本检查失败: {result.stderr}")
            
        # 检查Docker服务状态
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✅ Docker服务正在运行")
            return True
        else:
            print("  ⚠️  Docker服务未运行，请启动Docker Desktop")
            print("     启动Docker Desktop后再次运行此测试")
            return False
            
    except subprocess.TimeoutExpired:
        print("  ❌ Docker命令超时")
        return False
    except FileNotFoundError:
        print("  ❌ Docker未安装或不在PATH中")
        return False
    except Exception as e:
        print(f"  ❌ Docker检查异常: {e}")
        return False

def check_nvidia_gpu():
    """检查NVIDIA GPU可用性"""
    print("\n🎮 GPU环境检查:")
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✅ NVIDIA GPU已检测到")
            # 提取GPU信息
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line and 'Driver Version' in line:
                    print(f"     {line.strip()}")
            return True
        else:
            print("  ⚠️  未检测到NVIDIA GPU或驱动")
            return False
    except FileNotFoundError:
        print("  ⚠️  nvidia-smi未找到，无GPU支持")
        return False
    except Exception as e:
        print(f"  ⚠️  GPU检查异常: {e}")
        return False

def validate_all_files():
    """验证所有配置文件"""
    print("\n📁 配置文件完整性检查:")
    
    required_files = {
        'Dockerfile': '主Docker镜像配置',
        'docker-compose.yml': 'Docker Compose服务配置', 
        'install_dependencies.sh': '智能依赖安装脚本',
        'docker-entrypoint-smart.sh': '智能启动脚本',
        'start-docker-intelligent.sh': '便捷启动脚本',
        'test_docker_simple.py': '配置验证脚本',
        'README_DOCKER.md': 'Docker解决方案文档'
    }
    
    all_present = True
    for filename, description in required_files.items():
        filepath = Path(filename)
        if filepath.exists():
            size = filepath.stat().st_size
            print(f"  ✅ {filename} - {description} ({size} bytes)")
        else:
            print(f"  ❌ {filename} - {description} (缺失)")
            all_present = False
            
    return all_present

def show_deployment_guide(docker_running, has_gpu):
    """显示部署指导"""
    print("\n🚀 部署指导:")
    
    if not docker_running:
        print("  1. 首先启动Docker Desktop")
        print("     - Windows: 从开始菜单启动Docker Desktop")
        print("     - 等待Docker完全启动")
        print("     - 验证: docker info")
        print("\n  2. 启动Docker Desktop后，重新运行此脚本验证")
        return
    
    print("  Docker已就绪，可以开始部署！")
    print("\n  推荐部署模式:")
    
    if has_gpu:
        print("  🎮 检测到GPU，推荐使用智能模式:")
        print("     docker-compose up -d inference-server")
        print("     # 端口: http://localhost:8000")
        print("     # 自动GPU检测，CPU回退")
        
        print("\n  🔧 也可以显式使用GPU模式:")
        print("     docker-compose --profile gpu-explicit up -d")
        print("     # 端口: http://localhost:8001")
        
    else:
        print("  💻 未检测到GPU，推荐使用CPU模式:")
        print("     docker-compose --profile cpu-only up -d")
        print("     # 端口: http://localhost:8002")
        
        print("\n  🤖 也可以使用智能模式(会自动回退到CPU):")
        print("     docker-compose up -d inference-server")
        print("     # 端口: http://localhost:8000")
    
    print("\n  📊 服务验证:")
    print("     docker-compose ps          # 检查容器状态")
    print("     docker-compose logs -f     # 查看实时日志")
    print("     curl http://localhost:8000/health  # 健康检查")

def show_usage_examples():
    """显示使用示例"""
    print("\n📖 使用示例:")
    print("  # 便捷启动(交互式)")
    print("  ./start-docker-intelligent.sh")
    print("")
    print("  # 手动启动各模式")
    print("  docker-compose up -d inference-server           # 智能模式")
    print("  docker-compose --profile cpu-only up -d         # CPU强制")
    print("  docker-compose --profile gpu-explicit up -d     # GPU显式")
    print("")
    print("  # 停止服务")
    print("  docker-compose down")
    print("")
    print("  # 查看状态")
    print("  docker-compose ps")
    print("  docker-compose logs inference-server")

def main():
    """主函数"""
    print("🏗️  智能Docker解决方案 - 最终验证")
    print("=" * 50)
    
    # 1. 检查配置文件
    files_ok = validate_all_files()
    if not files_ok:
        print("\n❌ 配置文件不完整，请检查缺失的文件")
        return 1
    
    # 2. 检查Docker状态
    docker_running = check_docker_status()
    
    # 3. 检查GPU状态
    has_gpu = check_nvidia_gpu()
    
    # 4. 显示部署指导
    show_deployment_guide(docker_running, has_gpu)
    
    # 5. 显示使用示例
    if docker_running:
        show_usage_examples()
    
    print("\n" + "=" * 50)
    if docker_running:
        print("🎉 系统就绪！可以开始部署Docker服务")
    else:
        print("⏳ 请启动Docker Desktop后重新运行此脚本")
    
    return 0 if docker_running else 1

if __name__ == "__main__":
    sys.exit(main())
