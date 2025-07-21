"""
模型管理服务
实现模型的生命周期管理、资源池化和并发控制
"""
import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import weakref

from .base import BaseService, ServiceStatus, ServiceMetrics
from ..core.interfaces import ModelInterface, InferenceResult
from ..core.models import ModelFactory

@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str
    model_path: str
    config_path: str
    device: str = "CPU"
    max_instances: int = 3
    warm_up_requests: int = 5
    idle_timeout: int = 300  # 空闲超时时间（秒）

@dataclass
class ModelInstance:
    """模型实例"""
    model: ModelInterface
    instance_id: str
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    is_busy: bool = False
    total_requests: int = 0
    
    def mark_used(self):
        """标记使用"""
        self.last_used = time.time()
        self.total_requests += 1

class ModelPool:
    """模型实例池"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.instances: List[ModelInstance] = []
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(config.max_instances)
        self._creation_lock = asyncio.Lock()
    
    async def get_instance(self) -> Optional[ModelInstance]:
        """获取可用的模型实例"""
        async with self.lock:
            # 查找空闲实例
            for instance in self.instances:
                if not instance.is_busy:
                    instance.is_busy = True
                    instance.mark_used()
                    return instance
        
        # 如果没有空闲实例且未达到最大数量，创建新实例
        if len(self.instances) < self.config.max_instances:
            return await self._create_instance()
        
        return None
    
    async def _create_instance(self) -> Optional[ModelInstance]:
        """创建新的模型实例"""
        async with self._creation_lock:
            try:
                # 再次检查是否超出限制
                if len(self.instances) >= self.config.max_instances:
                    return None
                
                model = await self._load_model()
                if model is None:
                    return None
                
                instance = ModelInstance(
                    model=model,
                    instance_id=f"{self.config.model_name}_{len(self.instances)}"
                )
                instance.is_busy = True
                instance.mark_used()
                
                self.instances.append(instance)
                return instance
                
            except Exception as e:
                print(f"创建模型实例失败: {e}")
                return None
    
    async def _load_model(self) -> Optional[ModelInterface]:
        """加载模型"""
        try:
            # 这里需要根据实际情况调用模型工厂
            # 为了演示，我们使用简化的加载逻辑
            model = ModelFactory.create_model(
                model_name=self.config.model_name,
                model_config={"config_path": self.config.config_path},
                checkpoint_path=self.config.model_path
            )
            return model
        except Exception as e:
            print(f"模型加载失败: {e}")
            return None
    
    async def release_instance(self, instance: ModelInstance):
        """释放模型实例"""
        async with self.lock:
            if instance in self.instances:
                instance.is_busy = False
                instance.mark_used()
    
    async def cleanup_idle_instances(self):
        """清理空闲实例"""
        current_time = time.time()
        async with self.lock:
            instances_to_remove = []
            for instance in self.instances:
                if (not instance.is_busy and 
                    current_time - instance.last_used > self.config.idle_timeout):
                    instances_to_remove.append(instance)
            
            for instance in instances_to_remove:
                self.instances.remove(instance)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取池统计信息"""
        total_instances = len(self.instances)
        busy_instances = sum(1 for i in self.instances if i.is_busy)
        total_requests = sum(i.total_requests for i in self.instances)
        
        return {
            "total_instances": total_instances,
            "busy_instances": busy_instances,
            "idle_instances": total_instances - busy_instances,
            "total_requests": total_requests,
            "max_instances": self.config.max_instances
        }

class ModelManagerService(BaseService):
    """模型管理服务"""
    
    def __init__(self, service_id: Optional[str] = None):
        super().__init__(service_id)
        self.model_pools: Dict[str, ModelPool] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
    
    async def initialize(self) -> bool:
        """初始化模型管理服务"""
        try:
            self.status = ServiceStatus.READY
            # 启动清理任务
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            return True
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.metrics.last_error = str(e)
            return False
    
    async def register_model(self, config: ModelConfig) -> bool:
        """注册模型配置"""
        try:
            if config.model_name in self.model_pools:
                return False
            
            self.model_configs[config.model_name] = config
            self.model_pools[config.model_name] = ModelPool(config)
            return True
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            return False
    
    async def get_model_instance(self, model_name: str) -> Optional[ModelInstance]:
        """获取模型实例"""
        if model_name not in self.model_pools:
            return None
        
        try:
            pool = self.model_pools[model_name]
            instance = await pool.get_instance()
            
            if instance:
                self.metrics.active_requests += 1
            
            return instance
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            return None
    
    async def release_model_instance(self, model_name: str, instance: ModelInstance):
        """释放模型实例"""
        if model_name in self.model_pools:
            await self.model_pools[model_name].release_instance(instance)
            self.metrics.active_requests = max(0, self.metrics.active_requests - 1)
    
    async def process(self, request: Any) -> Any:
        """处理请求（通用接口实现）"""
        # 这里可以实现一些通用的模型管理操作
        return {"status": "ok", "message": "Model manager is ready"}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        model_stats = {}
        for model_name, pool in self.model_pools.items():
            model_stats[model_name] = pool.get_stats()
        
        return {
            "status": self.status.value,
            "service_id": self.service_id,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "active_requests": self.metrics.active_requests,
                "error_count": self.metrics.error_count
            },
            "models": model_stats
        }
    
    async def _cleanup_loop(self):
        """定期清理空闲实例"""
        while self.status != ServiceStatus.STOPPED:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                for pool in self.model_pools.values():
                    await pool.cleanup_idle_instances()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"清理任务异常: {e}")
    
    async def shutdown(self):
        """关闭服务"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await super().shutdown()
    
    def get_model_list(self) -> List[str]:
        """获取已注册的模型列表"""
        return list(self.model_configs.keys())
    
    async def reload_model(self, model_name: str) -> bool:
        """重新加载模型"""
        if model_name not in self.model_configs:
            return False
        
        try:
            # 清空现有实例
            if model_name in self.model_pools:
                old_pool = self.model_pools[model_name]
                async with old_pool.lock:
                    old_pool.instances.clear()
            
            # 创建新的池
            config = self.model_configs[model_name]
            self.model_pools[model_name] = ModelPool(config)
            return True
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            return False
