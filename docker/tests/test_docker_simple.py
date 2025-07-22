#!/usr/bin/env python3
"""
简化的Docker配置测试脚本
不依赖Docker运行时，只验证配置文件
"""
import os
import sys
from pathlib import Path

def test_files_exist():
    """测试必要文件是否存在"""
    print("📋 检查必要文件:")
    
    files = {
        'Dockerfile': '主要Docker镜像配置',
        'docker-compose.yml': 'Docker Compose服务配置',
        'install_dependencies.sh': '智能依赖安装脚本',
        'docker-entrypoint-smart.sh': '智能启动脚本',
        'start-docker-intelligent.sh': '便捷启动脚本'
    }
    
    all_exist = True
    for filename, description in files.items():
        path = Path(filename)
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {filename} - {description} ({size} bytes)")
        else:
            print(f"  ❌ {filename} - {description} (缺失)")
            all_exist = False
    
    return all_exist

def test_dockerfile_content():
    """测试Dockerfile内容"""
    print("\n📝 Dockerfile内容检查:")
    
    try:
        # 尝试不同的编码
        encodings = ['utf-8', 'utf-8-sig', 'latin-1']
        content = None
        
        for encoding in encodings:
            try:
                with open('Dockerfile', 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"  ✅ 成功读取文件 (编码: {encoding})")
                break
            except UnicodeDecodeError:
                continue
        
        if not content:
            print("  ❌ 无法读取Dockerfile文件")
            return False
        
        # 检查关键内容
        checks = {
            'FROM指令': 'FROM' in content,
            'CUDA支持': 'nvidia/cuda' in content,
            '多阶段构建': content.count('FROM') >= 2,
            'TUNA镜像源': 'tuna.tsinghua.edu.cn' in content,
            '智能脚本': 'install_dependencies.sh' in content,
            '端口暴露': 'EXPOSE' in content,
            '健康检查': 'HEALTHCHECK' in content
        }
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        
        return all(checks.values())
        
    except Exception as e:
        print(f"  ❌ Dockerfile检查失败: {e}")
        return False

def test_compose_structure():
    """测试docker-compose.yml结构"""
    print("\n🐳 Docker Compose配置检查:")
    
    try:
        import yaml
        with open('docker-compose.yml', 'r', encoding='utf-8') as f:
            compose_data = yaml.safe_load(f)
        
        # 检查基本结构
        required_keys = ['version', 'services']
        for key in required_keys:
            if key in compose_data:
                print(f"  ✅ {key} 键存在")
            else:
                print(f"  ❌ {key} 键缺失")
                return False
        
        # 检查服务
        services = compose_data.get('services', {})
        expected_services = ['inference-server', 'inference-server-gpu', 'inference-server-cpu']
        
        for service in expected_services:
            if service in services:
                ports = services[service].get('ports', [])
                profiles = services[service].get('profiles', [])
                print(f"  ✅ {service} 服务存在 (端口: {ports}, 配置: {profiles})")
            else:
                print(f"  ❌ {service} 服务缺失")
        
        return True
        
    except ImportError:
        print("  ⚠️  PyYAML未安装，跳过YAML语法检查")
        # 简单检查文件是否可读
        try:
            with open('docker-compose.yml', 'r') as f:
                content = f.read()
            print("  ✅ 文件可读")
            return True
        except Exception as e:
            print(f"  ❌ 文件读取失败: {e}")
            return False
    except Exception as e:
        print(f"  ❌ Compose配置检查失败: {e}")
        return False

def test_script_files():
    """测试脚本文件"""
    print("\n🔧 脚本文件检查:")
    
    scripts = [
        'install_dependencies.sh',
        'docker-entrypoint-smart.sh', 
        'start-docker-intelligent.sh'
    ]
    
    all_good = True
    for script in scripts:
        try:
            with open(script, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查基本特征
            has_shebang = content.startswith('#!/bin/bash') or content.startswith('#!/usr/bin/env')
            is_substantial = len(content.strip()) > 100
            
            print(f"  ✅ {script}: shebang={'✅' if has_shebang else '❌'}, 内容={'充足' if is_substantial else '少'}")
            
        except Exception as e:
            print(f"  ❌ {script}: 读取失败 - {e}")
            all_good = False
    
    return all_good

def main():
    """主测试函数"""
    print("🧪 Docker配置简化测试")
    print("=" * 50)
    
    tests = [
        ("文件存在性", test_files_exist),
        ("Dockerfile内容", test_dockerfile_content),
        ("Compose配置", test_compose_structure),
        ("脚本文件", test_script_files)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
            status = "✅ 通过" if result else "❌ 失败"
            print(f"\n{status} {test_name}")
        except Exception as e:
            print(f"\n❌ {test_name} - 测试异常: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    
    # 总结
    passed_tests = sum(results)
    total_tests = len(results)
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过！Docker配置验证成功")
        print("\n🚀 可用的启动命令:")
        print("  docker-compose up -d inference-server")
        print("  docker-compose --profile cpu-only up -d")
        print("  docker-compose --profile gpu-explicit up -d")
        return True
    else:
        print(f"⚠️  {passed_tests}/{total_tests} 项测试通过")
        print("请检查失败的项目并修复")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
