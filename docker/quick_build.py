#!/usr/bin/env python3
"""
Docker快速构建脚本
自动检查环境并提供构建选项
"""
import os
import subprocess
import sys
from pathlib import Path

def check_docker_ready():
    """快速检查Docker是否就绪"""
    print("🔍 快速检查Docker环境...")
    
    try:
        # 检查Docker命令
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("❌ Docker不可用")
            return False
        
        # 检查Docker服务
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("❌ Docker服务未运行，请启动Docker Desktop")
            return False
        
        print("✅ Docker环境就绪")
        return True
        
    except FileNotFoundError:
        print("❌ Docker未安装")
        return False
    except Exception as e:
        print(f"❌ Docker检查失败: {e}")
        return False

def check_files():
    """检查必要文件"""
    required_files = ['Dockerfile', 'docker-compose.yml']
    
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ 缺失文件: {file}")
            return False
    
    print("✅ 配置文件完整")
    return True

def auto_build():
    """自动构建Docker镜像"""
    print("🚀 开始Docker构建...")
    
    try:
        # 切换到项目根目录
        project_root = Path('../').resolve()
        
        # 执行构建命令
        cmd = ['docker', 'compose', '-f', 'docker/docker-compose.yml', 'up', '--build', '-d']
        print(f"📋 执行命令: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            print("✅ Docker构建成功!")
            print("🌐 服务已启动:")
            print("   - 智能模式: http://localhost:8000")
            print("   - GPU模式: http://localhost:8001 (需要--profile gpu-explicit)")
            print("   - CPU模式: http://localhost:8002 (需要--profile cpu-only)")
            return True
        else:
            print(f"❌ Docker构建失败，退出码: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ 构建过程出错: {e}")
        return False

def show_manual_commands():
    """显示手动命令"""
    print("\n💡 手动操作命令:")
    print("# 1. 切换到项目根目录")
    print("cd ..")
    print()
    print("# 2. 智能模式构建 (推荐)")
    print("docker compose -f docker/docker-compose.yml up --build")
    print()
    print("# 3. 后台运行")
    print("docker compose -f docker/docker-compose.yml up --build -d")
    print()
    print("# 4. GPU强制模式")
    print("docker compose -f docker/docker-compose.yml --profile gpu-explicit up --build")
    print()
    print("# 5. CPU专用模式")
    print("docker compose -f docker/docker-compose.yml --profile cpu-only up --build")

def main():
    """主函数"""
    print("🐳 Docker快速构建工具")
    print("=" * 40)
    
    # 检查环境
    if not check_files():
        print("⚠️  配置文件不完整，请检查项目结构")
        return False
    
    if not check_docker_ready():
        print("⚠️  Docker环境未就绪")
        show_manual_commands()
        return False
    
    # 询问构建
    print("\n🤖 是否立即开始构建?")
    choice = input("选择 (y=是, n=否, h=显示命令): ").strip().lower()
    
    if choice in ['y', 'yes', '是', '1']:
        return auto_build()
    elif choice in ['h', 'help', '帮助']:
        show_manual_commands()
    else:
        print("💡 构建已取消")
        show_manual_commands()
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  用户中断")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
