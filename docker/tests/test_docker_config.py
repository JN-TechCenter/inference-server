#!/usr/bin/env python3
"""
Docker配置验证脚本
验证所有Docker相关文件的完整性和语法正确性
"""
import os
import sys
import yaml
import subprocess
from pathlib import Path

def check_file_exists(filename):
    """检查文件是否存在"""
    path = Path(filename)
    if path.exists():
        # 强制刷新文件状态
        path.stat()
        size = path.stat().st_size
        print(f"✅ {filename} 存在 ({size} bytes)")
        return True
    else:
        print(f"❌ {filename} 缺失")
        return False

def validate_dockerfile():
    """验证Dockerfile语法"""
    try:
        with open('Dockerfile', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 基本语法检查
        lines = content.split('\n')
        dockerfile_commands = ['FROM', 'RUN', 'COPY', 'ENV', 'WORKDIR', 'EXPOSE', 'CMD', 'ENTRYPOINT']
        
        has_from = any(line.strip().startswith('FROM') for line in lines)
        if not has_from:
            print("❌ Dockerfile缺少FROM指令")
            return False
            
        print("✅ Dockerfile基本语法检查通过")
        
        # 检查关键特性
        features = {
            'CUDA支持': 'nvidia/cuda' in content,
            '多阶段构建': content.count('FROM') >= 2,
            '智能脚本': 'install_dependencies.sh' in content,
            'TUNA镜像源': 'tuna.tsinghua.edu.cn' in content,
            '健康检查': 'HEALTHCHECK' in content
        }
        
        print("🔍 Dockerfile特性检查:")
        for feature, exists in features.items():
            status = "✅" if exists else "⚠️"
            print(f"  {status} {feature}")
            
        return True
        
    except Exception as e:
        print(f"❌ Dockerfile验证失败: {e}")
        return False

def validate_docker_compose():
    """验证docker-compose.yml语法"""
    try:
        with open('docker-compose.yml', 'r', encoding='utf-8') as f:
            compose_data = yaml.safe_load(f)
        
        # 检查基本结构
        required_keys = ['version', 'services']
        for key in required_keys:
            if key not in compose_data:
                print(f"❌ docker-compose.yml缺少 {key} 键")
                return False
        
        # 检查服务
        services = compose_data.get('services', {})
        expected_services = ['inference-server', 'inference-server-gpu', 'inference-server-cpu']
        
        print("🔍 Docker Compose服务检查:")
        for service in expected_services:
            if service in services:
                print(f"  ✅ {service} 服务存在")
            else:
                print(f"  ❌ {service} 服务缺失")
        
        # 检查端口映射
        ports_check = {
            'inference-server': '8000:8000',
            'inference-server-gpu': '8001:8000', 
            'inference-server-cpu': '8002:8000'
        }
        
        print("🔍 端口映射检查:")
        for service, expected_port in ports_check.items():
            if service in services:
                ports = services[service].get('ports', [])
                if expected_port in ports:
                    print(f"  ✅ {service}: {expected_port}")
                else:
                    print(f"  ⚠️  {service}: 端口映射可能不正确")
        
        print("✅ docker-compose.yml语法检查通过")
        return True
        
    except yaml.YAMLError as e:
        print(f"❌ docker-compose.yml YAML语法错误: {e}")
        return False
    except Exception as e:
        print(f"❌ docker-compose.yml验证失败: {e}")
        return False

def check_scripts():
    """检查脚本文件"""
    scripts = [
        'install_dependencies.sh',
        'docker-entrypoint-smart.sh',
        'start-docker-intelligent.sh',
        'validate-docker-config.sh'
    ]
    
    print("🔍 脚本文件检查:")
    all_good = True
    
    for script in scripts:
        if check_file_exists(script):
            try:
                with open(script, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.startswith('#!/bin/bash') or content.startswith('#!/usr/bin/env'):
                    print(f"  ✅ {script} 有正确的shebang")
                else:
                    print(f"  ⚠️  {script} 缺少shebang")
                
                if len(content.strip()) > 100:
                    print(f"  ✅ {script} 内容充足")
                else:
                    print(f"  ⚠️  {script} 内容较少")
                    
            except Exception as e:
                print(f"  ❌ {script} 读取失败: {e}")
                all_good = False
        else:
            all_good = False
    
    return all_good

def check_environment():
    """检查运行环境"""
    print("🔍 环境检查:")
    
    # Docker检查
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  ✅ Docker: {result.stdout.strip()}")
        else:
            print("  ❌ Docker不可用")
    except Exception:
        print("  ❌ Docker未安装或不可用")
    
    # Docker Compose检查
    compose_available = False
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"  ✅ docker-compose: {result.stdout.strip()}")
            compose_available = True
    except Exception:
        pass
    
    if not compose_available:
        try:
            result = subprocess.run(['docker', 'compose', 'version'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"  ✅ docker compose: {result.stdout.strip()}")
                compose_available = True
        except Exception:
            pass
    
    if not compose_available:
        print("  ❌ Docker Compose不可用")
    
    # GPU检查
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            gpu_line = next((line for line in lines if 'GeForce' in line or 'RTX' in line or 'GTX' in line), None)
            if gpu_line:
                print(f"  ✅ NVIDIA GPU检测到")
            else:
                print("  ✅ NVIDIA驱动可用")
        else:
            print("  ℹ️  未检测到NVIDIA GPU")
    except Exception:
        print("  ℹ️  NVIDIA驱动不可用")

def main():
    """主函数"""
    print("🔧 Docker配置验证脚本")
    print("=" * 50)
    
    # 检查必要文件
    print("📋 检查必要文件:")
    required_files = [
        'Dockerfile',
        'docker-compose.yml'
    ]
    
    files_ok = all(check_file_exists(f) for f in required_files)
    
    if not files_ok:
        print("❌ 缺少必要文件，请先创建")
        return False
    
    print()
    
    # 验证配置文件
    print("📝 验证配置文件:")
    dockerfile_ok = validate_dockerfile()
    compose_ok = validate_docker_compose()
    
    print()
    
    # 检查脚本
    scripts_ok = check_scripts()
    
    print()
    
    # 环境检查
    check_environment()
    
    print()
    print("=" * 50)
    
    if dockerfile_ok and compose_ok and scripts_ok:
        print("✅ 所有检查通过！Docker配置就绪")
        print()
        print("🚀 可用启动命令:")
        print("  智能模式: docker-compose up -d inference-server")
        print("  CPU模式:  docker-compose --profile cpu-only up -d")
        print("  GPU模式:  docker-compose --profile gpu-explicit up -d")
        print()
        print("🔧 便捷脚本:")
        print("  ./start-docker-intelligent.sh")
        return True
    else:
        print("❌ 存在配置问题，请检查并修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
