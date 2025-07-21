"""
推理服务
高并发推理请求处理
"""
import asyncio
import time
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import numpy as np

from .base import BaseService, ServiceStatus
from .model_manager import ModelManagerService, ModelInstance
from ..core.interfaces import InferenceResult
from ..core.preprocessors import PreProcessorChain
from ..core.postprocessors import PostProcessorFactory
from ..predict import infer as sync_infer

@dataclass
class InferenceRequest:
    """推理请求"""
    request_id: str
    user_id: Optional[str]
    model_name: str
    image_data: Union[np.ndarray, bytes]
    task_type: str = "detect"  # detect, segment
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.65
    return_visualization: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class InferenceResponse:
    """推理响应"""
    request_id: str
    success: bool
    result: Optional[InferenceResult] = None
    visualization: Optional[np.ndarray] = None
    error_message: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class RequestQueue:
    """请求队列 - 支持优先级"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queue = asyncio.PriorityQueue(maxsize=max_size)
        self.pending_requests: Dict[str, InferenceRequest] = {}
        self.lock = asyncio.Lock()
    
    async def put(self, request: InferenceRequest, priority: int = 0) -> bool:
        """添加请求到队列"""
        try:
            if self.queue.full():
                return False
            
            await self.queue.put((priority, time.time(), request))
            async with self.lock:
                self.pending_requests[request.request_id] = request
            return True
        except Exception:
            return False
    
    async def get(self) -> Optional[InferenceRequest]:
        """从队列获取请求"""
        try:
            priority, timestamp, request = await self.queue.get()
            async with self.lock:
                self.pending_requests.pop(request.request_id, None)
            return request
        except Exception:
            return None
    
    def qsize(self) -> int:
        """获取队列大小"""
        return self.queue.qsize()
    
    async def cancel_request(self, request_id: str) -> bool:
        """取消请求"""
        async with self.lock:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
                return True
        return False

class InferenceWorker:
    """推理工作器"""
    
    def __init__(self, worker_id: str, model_manager: ModelManagerService):
        self.worker_id = worker_id
        self.model_manager = model_manager
        self.is_running = False
        self.current_request: Optional[InferenceRequest] = None
        self.processed_count = 0
        
    async def start(self, request_queue: RequestQueue, response_handler):
        """启动工作器"""
        self.is_running = True
        while self.is_running:
            try:
                # 获取请求
                request = await request_queue.get()
                if request is None:
                    await asyncio.sleep(0.1)
                    continue
                
                self.current_request = request
                
                # 处理请求
                response = await self._process_request(request)
                
                # 发送响应
                await response_handler(response)
                
                self.processed_count += 1
                self.current_request = None
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.current_request:
                    error_response = InferenceResponse(
                        request_id=self.current_request.request_id,
                        success=False,
                        error_message=str(e)
                    )
                    await response_handler(error_response)
                    self.current_request = None
    
    async def _process_request(self, request: InferenceRequest) -> InferenceResponse:
        """处理单个推理请求"""
        start_time = time.time()
        
        try:
            # 获取模型实例
            model_instance = await self.model_manager.get_model_instance(request.model_name)
            if model_instance is None:
                return InferenceResponse(
                    request_id=request.request_id,
                    success=False,
                    error_message=f"Model {request.model_name} not available"
                )
            
            try:
                # 执行推理
                result = await self._run_inference(model_instance, request)
                
                processing_time = time.time() - start_time
                
                return InferenceResponse(
                    request_id=request.request_id,
                    success=True,
                    result=result,
                    processing_time=processing_time
                )
                
            finally:
                # 释放模型实例
                await self.model_manager.release_model_instance(
                    request.model_name, model_instance
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            return InferenceResponse(
                request_id=request.request_id,
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    async def _run_inference(self, model_instance: ModelInstance, request: InferenceRequest) -> InferenceResult:
        """执行推理"""
        # 这里需要调用实际的推理逻辑
        # 为了保持兼容性，我们可以包装现有的推理函数
        
        # 模拟推理过程
        loop = asyncio.get_event_loop()
        
        # 准备推理参数（这里需要根据实际情况构造args对象）
        args = type('Args', (), {
            'task': request.task_type,
            'image_path': request.image_data,
            'conf_thres': request.confidence_threshold,
            'iou_thres': request.iou_threshold,
            'network': model_instance.model,
            'data': type('Data', (), {'nc': 80, 'names': []})()
        })()
        
        # 在线程池中执行推理
        result = await loop.run_in_executor(None, self._sync_inference, args, request.image_data)
        
        return result
    
    def _sync_inference(self, args, image_data) -> InferenceResult:
        """同步推理包装器"""
        try:
            # 这里需要调用实际的推理函数
            # 暂时返回模拟结果
            return InferenceResult(
                category_ids=[1, 2],
                bboxes=[[100, 100, 200, 200], [150, 150, 250, 250]],
                scores=[0.8, 0.9]
            )
        except Exception as e:
            raise e
    
    def stop(self):
        """停止工作器"""
        self.is_running = False
    
    def get_status(self) -> Dict[str, Any]:
        """获取工作器状态"""
        return {
            "worker_id": self.worker_id,
            "is_running": self.is_running,
            "processed_count": self.processed_count,
            "current_request": self.current_request.request_id if self.current_request else None
        }

class InferenceService(BaseService):
    """推理服务"""
    
    def __init__(self, model_manager: ModelManagerService, max_workers: int = 4):
        super().__init__()
        self.model_manager = model_manager
        self.max_workers = max_workers
        self.request_queue = RequestQueue()
        self.workers: List[InferenceWorker] = []
        self.worker_tasks: List[asyncio.Task] = []
        self.response_handlers: Dict[str, asyncio.Queue] = {}
        self.pending_responses: Dict[str, InferenceResponse] = {}
    
    async def initialize(self) -> bool:
        """初始化推理服务"""
        try:
            # 创建工作器
            for i in range(self.max_workers):
                worker = InferenceWorker(f"worker_{i}", self.model_manager)
                self.workers.append(worker)
            
            # 启动工作器
            for worker in self.workers:
                task = asyncio.create_task(
                    worker.start(self.request_queue, self._handle_response)
                )
                self.worker_tasks.append(task)
            
            self.status = ServiceStatus.READY
            return True
        except Exception as e:
            self.status = ServiceStatus.ERROR
            self.metrics.last_error = str(e)
            return False
    
    async def submit_request(self, request: InferenceRequest) -> str:
        """提交推理请求"""
        try:
            # 为用户创建响应队列
            if request.user_id and request.user_id not in self.response_handlers:
                self.response_handlers[request.user_id] = asyncio.Queue()
            
            # 添加到队列
            success = await self.request_queue.put(request)
            if success:
                self.metrics.total_requests += 1
                return request.request_id
            else:
                raise Exception("Request queue is full")
                
        except Exception as e:
            self.metrics.error_count += 1
            self.metrics.last_error = str(e)
            raise e
    
    async def get_response(self, user_id: str, timeout: float = 30.0) -> Optional[InferenceResponse]:
        """获取用户的推理响应"""
        if user_id not in self.response_handlers:
            return None
        
        try:
            response = await asyncio.wait_for(
                self.response_handlers[user_id].get(),
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            return None
    
    async def _handle_response(self, response: InferenceResponse):
        """处理推理响应"""
        # 存储响应
        self.pending_responses[response.request_id] = response
        
        # 如果有用户队列，发送到用户队列
        request = None
        for user_id, queue in self.response_handlers.items():
            # 这里需要根据request_id找到对应的user_id
            # 简化处理，发送到所有队列（实际应该建立request_id到user_id的映射）
            try:
                await queue.put(response)
            except Exception:
                pass
    
    async def process(self, request: Any) -> Any:
        """处理请求（基类接口实现）"""
        if isinstance(request, InferenceRequest):
            return await self.submit_request(request)
        return {"error": "Invalid request type"}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        worker_stats = [worker.get_status() for worker in self.workers]
        
        return {
            "status": self.status.value,
            "service_id": self.service_id,
            "queue_size": self.request_queue.qsize(),
            "workers": worker_stats,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "active_requests": self.metrics.active_requests,
                "error_count": self.metrics.error_count
            }
        }
    
    async def shutdown(self):
        """关闭服务"""
        # 停止工作器
        for worker in self.workers:
            worker.stop()
        
        # 取消任务
        for task in self.worker_tasks:
            task.cancel()
        
        # 等待任务完成
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        await super().shutdown()
