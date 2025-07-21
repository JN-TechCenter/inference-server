"""
客户端测试脚本
测试推理服务器的并发性能
"""
import asyncio
import aiohttp
import time
import json
import numpy as np
import cv2
from typing import List, Dict, Any
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading

class InferenceClient:
    """推理客户端"""
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session: aiohttp.ClientSession = None
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_time": 0,
            "response_times": [],
            "errors": []
        }
        self.lock = threading.Lock()
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            async with self.session.get(f"{self.server_url}/health") as response:
                return await response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def get_models(self) -> Dict[str, Any]:
        """获取可用模型"""
        try:
            async with self.session.get(f"{self.server_url}/models") as response:
                return await response.json()
        except Exception as e:
            return {"error": str(e)}
    
    async def detect_objects(self, 
                           image_path: str, 
                           model_name: str = "yolov5s",
                           confidence_threshold: float = 0.25,
                           return_image: bool = False,
                           user_token: str = None) -> Dict[str, Any]:
        """目标检测"""
        start_time = time.time()
        
        try:
            # 读取图像
            if isinstance(image_path, str):
                with open(image_path, 'rb') as f:
                    image_data = f.read()
            else:
                # 假设是numpy数组
                _, buffer = cv2.imencode('.jpg', image_path)
                image_data = buffer.tobytes()
            
            # 准备请求数据
            data = aiohttp.FormData()
            data.add_field('file', image_data, filename='test.jpg', content_type='image/jpeg')
            data.add_field('model_name', model_name)
            data.add_field('confidence_threshold', str(confidence_threshold))
            data.add_field('return_image', str(return_image).lower())
            
            # 准备headers
            headers = {}
            if user_token:
                headers['Authorization'] = f'Bearer {user_token}'
            
            # 发送请求
            async with self.session.post(
                f"{self.server_url}/inference/detect",
                data=data,
                headers=headers
            ) as response:
                response_time = time.time() - start_time
                
                with self.lock:
                    self.stats["total_requests"] += 1
                    self.stats["total_time"] += response_time
                    self.stats["response_times"].append(response_time)
                
                if response.status == 200:
                    with self.lock:
                        self.stats["successful_requests"] += 1
                    result = await response.json()
                    result["response_time"] = response_time
                    return result
                else:
                    with self.lock:
                        self.stats["failed_requests"] += 1
                        self.stats["errors"].append({
                            "status": response.status,
                            "message": await response.text()
                        })
                    return {
                        "error": f"HTTP {response.status}",
                        "message": await response.text(),
                        "response_time": response_time
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            with self.lock:
                self.stats["total_requests"] += 1
                self.stats["failed_requests"] += 1
                self.stats["total_time"] += response_time
                self.stats["response_times"].append(response_time)
                self.stats["errors"].append({"exception": str(e)})
            
            return {
                "error": "Exception",
                "message": str(e),
                "response_time": response_time
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            if self.stats["response_times"]:
                avg_response_time = sum(self.stats["response_times"]) / len(self.stats["response_times"])
                max_response_time = max(self.stats["response_times"])
                min_response_time = min(self.stats["response_times"])
            else:
                avg_response_time = max_response_time = min_response_time = 0
            
            success_rate = (
                self.stats["successful_requests"] / max(self.stats["total_requests"], 1) * 100
            )
            
            return {
                "total_requests": self.stats["total_requests"],
                "successful_requests": self.stats["successful_requests"],
                "failed_requests": self.stats["failed_requests"],
                "success_rate": f"{success_rate:.2f}%",
                "total_time": f"{self.stats['total_time']:.2f}s",
                "avg_response_time": f"{avg_response_time:.3f}s",
                "max_response_time": f"{max_response_time:.3f}s",
                "min_response_time": f"{min_response_time:.3f}s",
                "requests_per_second": self.stats["total_requests"] / max(self.stats["total_time"], 0.001),
                "error_count": len(self.stats["errors"])
            }

async def concurrent_test(server_url: str, 
                         image_path: str, 
                         num_requests: int = 100,
                         concurrent_users: int = 10,
                         model_name: str = "yolov5s"):
    """并发测试"""
    print(f"🧪 开始并发测试:")
    print(f"   服务器: {server_url}")
    print(f"   图像: {image_path}")
    print(f"   请求数: {num_requests}")
    print(f"   并发用户: {concurrent_users}")
    print(f"   模型: {model_name}")
    print("-" * 50)
    
    async with InferenceClient(server_url) as client:
        # 健康检查
        health = await client.health_check()
        if "error" in health:
            print(f"❌ 服务器不可用: {health['error']}")
            return
        
        print("✅ 服务器健康检查通过")
        
        # 获取模型列表
        models = await client.get_models()
        if "error" in models:
            print(f"⚠️ 获取模型列表失败: {models['error']}")
        else:
            print(f"📋 可用模型: {models.get('models', [])}")
        
        # 创建任务
        tasks = []
        for i in range(num_requests):
            user_token = f"test_user_{i % concurrent_users}"
            task = client.detect_objects(
                image_path=image_path,
                model_name=model_name,
                user_token=user_token
            )
            tasks.append(task)
        
        # 执行并发请求
        print(f"\n🚀 执行 {num_requests} 个并发请求...")
        start_time = time.time()
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # 统计结果
        stats = client.get_statistics()
        
        print(f"\n📊 测试结果:")
        print(f"   总时间: {total_time:.2f}s")
        print(f"   总请求数: {stats['total_requests']}")
        print(f"   成功请求: {stats['successful_requests']}")
        print(f"   失败请求: {stats['failed_requests']}")
        print(f"   成功率: {stats['success_rate']}")
        print(f"   平均响应时间: {stats['avg_response_time']}")
        print(f"   最大响应时间: {stats['max_response_time']}")
        print(f"   最小响应时间: {stats['min_response_time']}")
        print(f"   每秒请求数: {stats['requests_per_second']:.2f}")
        
        if stats['error_count'] > 0:
            print(f"\n❌ 错误详情 (前5个):")
            for i, error in enumerate(client.stats['errors'][:5]):
                print(f"   {i+1}. {error}")

def create_test_image(filename: str = "test_image.jpg"):
    """创建测试图像"""
    # 创建一个简单的测试图像
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 添加一些几何形状
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.circle(image, (400, 300), 50, (0, 255, 0), -1)
    cv2.rectangle(image, (300, 50), (500, 150), (0, 0, 255), -1)
    
    cv2.imwrite(filename, image)
    print(f"✅ 创建测试图像: {filename}")
    return filename

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLO推理服务器客户端测试")
    parser.add_argument("--server", default="http://localhost:8000", help="服务器URL")
    parser.add_argument("--image", default="test_image.jpg", help="测试图像路径")
    parser.add_argument("--requests", type=int, default=50, help="请求数量")
    parser.add_argument("--users", type=int, default=5, help="并发用户数")
    parser.add_argument("--model", default="yolov5s", help="模型名称")
    parser.add_argument("--create-image", action="store_true", help="创建测试图像")
    
    args = parser.parse_args()
    
    if args.create_image:
        create_test_image(args.image)
    
    # 检查图像文件是否存在
    if not os.path.exists(args.image):
        print(f"❌ 图像文件不存在: {args.image}")
        print("   使用 --create-image 创建测试图像")
        return
    
    await concurrent_test(
        server_url=args.server,
        image_path=args.image,
        num_requests=args.requests,
        concurrent_users=args.users,
        model_name=args.model
    )

if __name__ == "__main__":
    import os
    asyncio.run(main())
