#!/usr/bin/env python3
"""
Docker环境生态测试3 - 综合验证脚本
"""
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def print_header():
    """打印测试头部"""
    print("=" * 60)
    print("🐳 Docker环境生态测试3")
    print("=" * 60)
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 测试目录: {os.getcwd()}")
    print()

def test_docker_files():
    """测试Docker相关文件"""
    print("📋 配置文件检查:")
    
    files = {
        'Dockerfile': 'Docker镜像构建文件',
        'docker-compose.yml': 'Docker Compose配置', 
        'scripts/install_dependencies.sh': '依赖安装脚本',
        'scripts/docker-entrypoint-smart.sh': '智能启动脚本',
        'start-docker-intelligent.sh': '便捷启动脚本'
    }
    
    results = {}
    
    for filepath, description in files.items():
        path = Path(filepath)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {filepath} - {description} ({size:,} bytes)")
            results[filepath] = True
        else:
            print(f"  ❌ {filepath} - {description} (缺失)")
            results[filepath] = False
    
    return results

def test_docker_syntax():
    """测试Docker文件语法"""
    print("\n🔍 Docker语法检查:")
    
    # 检查Dockerfile语法
    if Path('Dockerfile').exists():
        try:
            with open('Dockerfile', 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 基本语法检查
            required_instructions = ['FROM', 'WORKDIR', 'COPY', 'RUN']
            found_instructions = []
            
            for instruction in required_instructions:
                if instruction in content:
                    found_instructions.append(instruction)
            
            print(f"  ✅ Dockerfile 基本指令: {', '.join(found_instructions)}")
            
            # 检查多阶段构建
            if 'as gpu-base' in content and 'as cpu-base' in content:
                print("  ✅ 多阶段构建配置正确")
            
            # 检查智能特性
            if 'AUTO_DEVICE_SELECTION' in content:
                print("  ✅ 智能设备选择配置")
            
        except Exception as e:
            print(f"  ❌ Dockerfile 语法检查失败: {e}")
    
    # 检查docker-compose.yml语法
    if Path('docker-compose.yml').exists():
        try:
            import yaml
            with open('docker-compose.yml', 'r', encoding='utf-8') as f:
                compose_config = yaml.safe_load(f)
            
            services = compose_config.get('services', {})
            print(f"  ✅ docker-compose.yml 包含 {len(services)} 个服务")
            
            for service_name in services.keys():
                print(f"    📦 服务: {service_name}")
                
        except ImportError:
            print("  ⚠️  PyYAML未安装，跳过yaml语法检查")
        except Exception as e:
            print(f"  ❌ docker-compose.yml 语法检查失败: {e}")

def test_docker_environment():
    """测试Docker环境"""
    print("\n🐳 Docker环境检查:")
    
    try:
        # 检查Docker命令
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✅ Docker: {version}")
        else:
            print(f"  ❌ Docker不可用: {result.stderr}")
            return False
    except FileNotFoundError:
        print("  ❌ Docker未安装")
        return False
    except Exception as e:
        print(f"  ❌ Docker检查失败: {e}")
        return False
    
    # 检查Docker服务状态
    try:
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✅ Docker服务运行中")
        else:
            print("  ❌ Docker服务未运行")
            print("  💡 请启动Docker Desktop后重试")
            return False
    except Exception as e:
        print(f"  ❌ Docker服务检查失败: {e}")
        return False
    
    try:
        # 检查Docker Compose
        result = subprocess.run(['docker', 'compose', 'version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"  ✅ Docker Compose: {version}")
        else:
            print(f"  ⚠️  Docker Compose: {result.stderr}")
    except Exception as e:
        print(f"  ⚠️  Docker Compose检查失败: {e}")
    
    return True

def test_gpu_support():
    """测试GPU支持"""
    print("\n🎮 GPU支持检查:")
    
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✅ NVIDIA GPU已检测")
            # 简化GPU信息显示
            lines = result.stdout.split('\n')
            for line in lines[:10]:  # 只显示前10行
                if 'GPU' in line or 'Driver' in line:
                    print(f"    {line.strip()}")
            return True
        else:
            print("  ⚠️  未检测到NVIDIA GPU")
            return False
    except FileNotFoundError:
        print("  ⚠️  nvidia-smi未找到")
        return False
    except Exception as e:
        print(f"  ⚠️  GPU检查异常: {e}")
        return False

def test_project_structure():
    """测试项目结构"""
    print("\n📂 项目结构检查:")
    
    required_dirs = ['../app', '../weights', '../docs']
    required_files = ['../requirements.txt', '../app/src/core/predict.py']
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ 目录: {dir_path}")
        else:
            print(f"  ❌ 目录缺失: {dir_path}")
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✅ 文件: {file_path}")
        else:
            print(f"  ❌ 文件缺失: {file_path}")

def generate_summary(file_results, docker_available, gpu_available):
    """生成测试摘要"""
    print("\n" + "=" * 60)
    print("📊 测试摘要")
    print("=" * 60)
    
    # 文件完整性
    total_files = len(file_results)
    present_files = sum(file_results.values())
    print(f"📁 配置文件: {present_files}/{total_files} 存在")
    
    # 环境可用性
    print(f"🐳 Docker环境: {'✅ 可用' if docker_available else '❌ 不可用'}")
    print(f"🎮 GPU支持: {'✅ 可用' if gpu_available else '⚠️  不可用'}")
    
    # 建议
    print("\n💡 建议:")
    if present_files < total_files:
        print("  🔧 修复缺失的配置文件")
    if not docker_available:
        print("  🐳 启动Docker Desktop或安装Docker")
    if not gpu_available:
        print("  🎮 GPU不可用，将使用CPU模式")
    
    if present_files == total_files and docker_available:
        print("  🎉 环境配置完整，可以开始Docker构建！")
        print("  🚀 推荐命令: docker compose up --build")
        return True  # 返回可以构建的状态
    else:
        return False  # 返回不可构建的状态

def main():
    """主测试函数"""
    print_header()
    
    # 运行各项测试
    file_results = test_docker_files()
    test_docker_syntax()
    docker_available = test_docker_environment()
    gpu_available = test_gpu_support()
    test_project_structure()
    
    # 生成摘要并检查是否可以构建
    can_build = generate_summary(file_results, docker_available, gpu_available)
    
    return can_build

def auto_build_option():
    """提供自动构建选项"""
    print("\n🤖 自动构建选项:")
    user_input = input("是否要自动开始Docker构建? (y/n): ").strip().lower()
    
    if user_input in ['y', 'yes', '是', '1']:
        print("🚀 开始自动构建...")
        try:
            # 切换到项目根目录
            project_root = Path('../').resolve()
            print(f"📁 切换到项目目录: {project_root}")
            
            # 执行Docker构建
            print("🔨 执行命令: docker compose up --build")
            result = subprocess.run(
                ['docker', 'compose', '-f', 'docker/docker-compose.yml', 'up', '--build'],
                cwd=project_root,
                text=True,
                capture_output=False  # 实时显示输出
            )
            
            if result.returncode == 0:
                print("✅ Docker构建成功!")
            else:
                print(f"❌ Docker构建失败，退出码: {result.returncode}")
                
        except KeyboardInterrupt:
            print("\n⏹️  用户中断构建")
        except Exception as e:
            print(f"❌ 构建过程出错: {e}")
    else:
        print("⏭️  跳过自动构建")
        print("💡 手动构建命令:")
        print("   cd ..")
        print("   docker compose -f docker/docker-compose.yml up --build")

if __name__ == "__main__":
    can_build = main()
    
    # 只有在环境完整时才询问是否自动构建
    if can_build:
        auto_build_option()
    else:
        print("\n⚠️  环境未就绪，请先修复问题后再尝试构建")