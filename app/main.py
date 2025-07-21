"""
FastAPI应用主文件
高并发YOLO推理服务器
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import asyncio
import io
import json
import time
import uuid
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import numpy as np
import cv2
from PIL import Image

from .services import (
    ServiceRegistry, 
    ModelManagerService, 
    InferenceService,
    UserSessionManager,
    ModelConfig
)
from .services.inference_service import InferenceRequest, InferenceResponse
from .config_manager import config_manager
from .performance_monitor import performance_monitor

# 全局服务实例
service_registry = ServiceRegistry()
model_manager: Optional[ModelManagerService] = None
inference_service: Optional[InferenceService] = None  
session_manager: Optional[UserSessionManager] = None

security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global model_manager, inference_service, session_manager
    
    print("🚀 启动YOLO推理服务器...")
    
    try:
        # 验证配置
        validation = config_manager.validate_config()
        if not validation["valid"]:
            print("⚠️ 配置验证警告:")
            for issue in validation["issues"]:
                print(f"  - {issue}")
        
        # 启动性能监控
        await performance_monitor.start()
        
        # 初始化服务
        await initialize_services()
        print("✅ 所有服务初始化完成")
        
        yield
        
    finally:
        print("🔄 关闭服务...")
        await shutdown_services()
        await performance_monitor.stop()
        print("✅ 服务已关闭")

async def initialize_services():
    """初始化所有服务"""
    global model_manager, inference_service, session_manager
    
    # 1. 初始化模型管理服务
    model_manager = ModelManagerService()
    await model_manager.initialize()
    await service_registry.register(model_manager)
    
    # 2. 注册配置的模型
    for model_name in config_manager.get_available_models():
        model_config = config_manager.get_model_config(model_name)
        if model_config:
            await model_manager.register_model(model_config)
            print(f"📝 注册模型: {model_name}")
    
    # 3. 初始化推理服务
    inference_service = InferenceService(
        model_manager, 
        max_workers=config_manager.inference_config.max_workers
    )
    await inference_service.initialize()
    await service_registry.register(inference_service)
    
    # 4. 初始化会话管理服务
    session_manager = UserSessionManager()
    session_manager.session_timeout = config_manager.session_config.session_timeout
    session_manager.cleanup_interval = config_manager.session_config.cleanup_interval
    await session_manager.initialize()
    await service_registry.register(session_manager)

async def shutdown_services():
    """关闭所有服务"""
    services = await service_registry.get_all_services()
    for service in services.values():
        await service.shutdown()

# 创建FastAPI应用
app = FastAPI(
    title="YOLO推理服务器",
    description="高并发多用户YOLO目标检测和分割服务",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=config_manager.server_config.allow_origins,
    allow_credentials=config_manager.server_config.allow_credentials,
    allow_methods=config_manager.server_config.allow_methods,
    allow_headers=config_manager.server_config.allow_headers,
)

# 请求监控中间件
@app.middleware("http")
async def monitor_requests(request, call_next):
    """监控请求的中间件"""
    start_time = time.time()
    performance_monitor.record_request_start()
    
    try:
        response = await call_next(request)
        processing_time = time.time() - start_time
        performance_monitor.record_request_end(processing_time, True)
        return response
    except Exception as e:
        processing_time = time.time() - start_time
        performance_monitor.record_request_end(processing_time, False)
        raise e

# 依赖注入函数
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """获取当前用户ID"""
    if credentials and credentials.credentials:
        # 这里可以实现实际的JWT验证
        return credentials.credentials
    # 为演示目的，使用随机用户ID
    return f"user_{uuid.uuid4().hex[:8]}"

async def get_or_create_session(user_id: str = Depends(get_current_user)):
    """获取或创建用户会话"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    # 获取用户现有会话
    sessions = await session_manager.get_user_sessions(user_id)
    if sessions:
        return sessions[0]  # 返回第一个活跃会话
    
    # 创建新会话
    return await session_manager.create_session(user_id)

# API路由
@app.get("/")
async def root():
    """根路径"""
    return {"message": "YOLO推理服务器运行中", "docs": "/docs"}

@app.get("/health")
async def health_check():
    """健康检查"""
    if not service_registry:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Service registry not available"}
        )
    
    health_status = await service_registry.health_check_all()
    
    all_healthy = all(
        service.get("status") == "ready" 
        for service in health_status.values()
    )
    
    status_code = 200 if all_healthy else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all_healthy else "unhealthy",
            "services": health_status,
            "timestamp": time.time()
        }
    )

@app.get("/models")
async def list_models():
    """获取可用模型列表"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    models = model_manager.get_model_list()
    health = await model_manager.health_check()
    
    return {
        "models": models,
        "model_stats": health.get("models", {}),
        "total": len(models)
    }

@app.post("/models/{model_name}/reload")
async def reload_model(model_name: str, user_id: str = Depends(get_current_user)):
    """重新加载模型"""
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not available")
    
    success = await model_manager.reload_model(model_name)
    if success:
        return {"message": f"Model {model_name} reloaded successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to reload model {model_name}")

@app.get("/session/info")
async def get_session_info(session = Depends(get_or_create_session)):
    """获取会话信息"""
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not available")
    
    quota_info = await session_manager.get_quota_info(session.user_id)
    
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "status": session.status.value,
        "total_requests": session.total_requests,
        "active_requests": session.active_requests,
        "quota": quota_info
    }

@app.post("/inference/detect")
async def detect_objects(
    file: UploadFile = File(...),
    model_name: str = "yolov5s",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    return_image: bool = False,
    session = Depends(get_or_create_session)
):
    """目标检测接口"""
    return await _process_inference_request(
        file=file,
        task_type="detect",
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        return_image=return_image,
        session=session
    )

@app.post("/inference/segment")
async def segment_objects(
    file: UploadFile = File(...),
    model_name: str = "yolov5s",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.65,
    return_image: bool = False,
    session = Depends(get_or_create_session)
):
    """实例分割接口"""
    return await _process_inference_request(
        file=file,
        task_type="segment",
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        return_image=return_image,
        session=session
    )

async def _process_inference_request(
    file: UploadFile,
    task_type: str,
    model_name: str,
    confidence_threshold: float,
    iou_threshold: float,
    return_image: bool,
    session
) -> Dict[str, Any]:
    """处理推理请求的通用函数"""
    if not inference_service or not session_manager:
        raise HTTPException(status_code=503, detail="Services not available")
    
    # 检查配额
    can_proceed = await session_manager.check_quota(session.user_id)
    if not can_proceed:
        raise HTTPException(status_code=429, detail="Request quota exceeded")
    
    # 检查并发限制
    can_accept = await session_manager.increment_active_requests(session.session_id)
    if not can_accept:
        raise HTTPException(status_code=429, detail="Too many concurrent requests")
    
    try:
        # 读取图像数据
        image_data = await file.read()
        
        # 转换为numpy数组
        image_array = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # 创建推理请求
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            user_id=session.user_id,
            model_name=model_name,
            image_data=image,
            task_type=task_type,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            return_visualization=return_image
        )
        
        # 提交请求
        request_id = await inference_service.submit_request(request)
        
        # 消费配额
        await session_manager.consume_quota(session.user_id)
        
        # 等待响应
        response = await inference_service.get_response(session.user_id, timeout=60.0)
        
        if response is None:
            raise HTTPException(status_code=408, detail="Request timeout")
        
        if not response.success:
            raise HTTPException(status_code=500, detail=response.error_message)
        
        # 构造响应
        result = {
            "request_id": response.request_id,
            "success": True,
            "processing_time": response.processing_time,
            "detections": {
                "category_ids": response.result.category_ids,
                "bboxes": response.result.bboxes,
                "scores": response.result.scores,
                "count": len(response.result.category_ids)
            }
        }
        
        if task_type == "segment" and response.result.masks:
            result["masks_available"] = True
        
        if return_image and response.visualization is not None:
            # 编码可视化图像为base64
            _, buffer = cv2.imencode('.jpg', response.visualization)
            import base64
            result["visualization"] = base64.b64encode(buffer).decode('utf-8')
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 减少活跃请求计数
        await session_manager.decrement_active_requests(session.session_id)

@app.get("/statistics")
async def get_statistics(user_id: str = Depends(get_current_user)):
    """获取服务统计信息"""
    if not all([model_manager, inference_service, session_manager]):
        raise HTTPException(status_code=503, detail="Services not available")
    
    model_health = await model_manager.health_check()
    inference_health = await inference_service.health_check() 
    session_health = await session_manager.health_check()
    session_stats = session_manager.get_statistics()
    performance_stats = performance_monitor.get_statistics()
    
    return {
        "model_service": model_health,
        "inference_service": inference_health,
        "session_service": session_health,
        "user_statistics": session_stats,
        "performance": performance_stats,
        "timestamp": time.time()
    }

@app.get("/metrics/performance")
async def get_performance_metrics(duration: int = 300):
    """获取性能指标"""
    history = performance_monitor.get_metrics_history(duration)
    current = performance_monitor.get_current_metrics()
    
    return {
        "current": current.to_dict() if current else None,
        "history": [metric.to_dict() for metric in history],
        "duration_seconds": duration,
        "sample_count": len(history)
    }

@app.post("/metrics/export")
async def export_metrics(filename: Optional[str] = None):
    """导出性能指标"""
    try:
        exported_file = performance_monitor.export_metrics(filename)
        return {
            "success": True,
            "filename": exported_file,
            "message": "指标导出成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config")
async def get_config():
    """获取当前配置"""
    return {
        "server": {
            "host": config_manager.server_config.host,
            "port": config_manager.server_config.port,
            "debug": config_manager.server_config.debug,
            "workers": config_manager.server_config.workers
        },
        "inference": {
            "max_workers": config_manager.inference_config.max_workers,
            "queue_size": config_manager.inference_config.queue_size,
            "timeout": config_manager.inference_config.timeout
        },
        "session": {
            "session_timeout": config_manager.session_config.session_timeout,
            "daily_limit": config_manager.session_config.daily_limit,
            "hourly_limit": config_manager.session_config.hourly_limit,
            "concurrent_limit": config_manager.session_config.concurrent_limit
        },
        "available_models": config_manager.get_available_models()
    }

@app.post("/config/save")
async def save_config():
    """保存当前配置"""
    success = config_manager.save_config()
    if success:
        return {"success": True, "message": "配置保存成功"}
    else:
        raise HTTPException(status_code=500, detail="配置保存失败")

@app.get("/config/validate")
async def validate_config():
    """验证配置"""
    return config_manager.validate_config()

@app.websocket("/ws/inference/{user_id}")
async def websocket_inference(websocket, user_id: str):
    """WebSocket推理接口（用于实时推理）"""
    await websocket.accept()
    
    if not session_manager:
        await websocket.send_text(json.dumps({
            "error": "Session manager not available"
        }))
        await websocket.close()
        return
    
    # 创建会话
    session = await session_manager.create_session(user_id)
    
    try:
        while True:
            # 接收数据
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 处理不同类型的消息
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "inference":
                # 处理推理请求
                # 这里需要实现WebSocket推理逻辑
                await websocket.send_text(json.dumps({
                    "type": "result",
                    "data": {"status": "processing"}
                }))
            
    except Exception as e:
        print(f"WebSocket连接异常: {e}")
    finally:
        await session_manager.remove_session(session.session_id)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=config_manager.server_config.host,
        port=config_manager.server_config.port,
        reload=config_manager.server_config.debug,
        workers=1  # 在生产环境中可以增加worker数量
    )
