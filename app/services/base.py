"""
服务层基础接口和抽象类
定义高内聚的服务接口
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncContextManager
from dataclasses import dataclass
import asyncio
import uuid
from enum import Enum

class ServiceStatus(Enum):
    """服务状态枚举"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPED = "stopped"

@dataclass
class ServiceMetrics:
    """服务性能指标"""
    total_requests: int = 0
    active_requests: int = 0
    average_response_time: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None

class BaseService(ABC):
    """基础服务抽象类"""
    
    def __init__(self, service_id: Optional[str] = None):
        self.service_id = service_id or str(uuid.uuid4())
        self.status = ServiceStatus.INITIALIZING
        self.metrics = ServiceMetrics()
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化服务"""
        pass
    
    @abstractmethod
    async def process(self, request: Any) -> Any:
        """处理请求"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass
    
    async def shutdown(self):
        """关闭服务"""
        self.status = ServiceStatus.STOPPED
    
    def get_metrics(self) -> ServiceMetrics:
        """获取服务指标"""
        return self.metrics

class ServiceRegistry:
    """服务注册中心 - 单例模式"""
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._services: Dict[str, BaseService] = {}
            self._initialized = True
    
    async def register(self, service: BaseService) -> bool:
        """注册服务"""
        async with self._lock:
            if service.service_id in self._services:
                return False
            self._services[service.service_id] = service
            return True
    
    async def unregister(self, service_id: str) -> bool:
        """注销服务"""
        async with self._lock:
            if service_id in self._services:
                await self._services[service_id].shutdown()
                del self._services[service_id]
                return True
            return False
    
    async def get_service(self, service_id: str) -> Optional[BaseService]:
        """获取服务"""
        return self._services.get(service_id)
    
    async def get_all_services(self) -> Dict[str, BaseService]:
        """获取所有服务"""
        return self._services.copy()
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """检查所有服务健康状态"""
        results = {}
        for service_id, service in self._services.items():
            try:
                results[service_id] = await service.health_check()
            except Exception as e:
                results[service_id] = {
                    "status": "error",
                    "error": str(e)
                }
        return results
