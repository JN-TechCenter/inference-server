"""
YOLO推理服务器配置文件
包含所有服务配置参数
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
APP_ROOT = PROJECT_ROOT / "app"
CONFIGS_ROOT = PROJECT_ROOT / "_configs"
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"

# 服务配置
class ServerConfig:
    HOST = "0.0.0.0"
    PORT = 8000
    DEBUG = True
    WORKERS = 1
    MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
    
    # API配置
    API_PREFIX = "/api/v1"
    DOCS_URL = "/docs"
    REDOC_URL = "/redoc"
    
    # CORS配置
    ALLOW_ORIGINS = ["*"]
    ALLOW_CREDENTIALS = True
    ALLOW_METHODS = ["*"]
    ALLOW_HEADERS = ["*"]

# 模型配置
class ModelConfig:
    # 默认模型
    DEFAULT_MODEL = "yolov5s"
    
    # 模型文件路径
    WEIGHTS_DIR = PROJECT_ROOT / "weights"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # 支持的模型列表
    SUPPORTED_MODELS = [
        "yolov5s", "yolov5m", "yolov5l", "yolov5x",
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
        "yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x"
    ]
    
    # 推理配置
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    MAX_DETECTIONS = 1000
    
    # 图像预处理
    INPUT_SIZE = (640, 640)
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]

# 视频处理配置
class VideoConfig:
    # 输出目录
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    VIDEO_OUTPUT_DIR = OUTPUT_DIR / "videos"
    LIVE_OUTPUT_DIR = OUTPUT_DIR / "live"
    
    # 视频编解码
    VIDEO_CODEC = "mp4v"
    VIDEO_EXTENSION = ".mp4"
    
    # 处理参数
    MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_VIDEO_DURATION = 600  # 10分钟
    FRAME_SKIP = 1  # 处理每n帧
    
    # 直播流配置
    RTMP_TIMEOUT = 30
    MAX_STREAM_DURATION = 3600  # 1小时
    STREAM_RECONNECT_ATTEMPTS = 3

# 日志配置
class LogConfig:
    LOG_DIR = PROJECT_ROOT / "logs"
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_FILE_BACKUP_COUNT = 5
    
    # 日志文件
    ACCESS_LOG = LOG_DIR / "access.log"
    ERROR_LOG = LOG_DIR / "error.log"
    APP_LOG = LOG_DIR / "app.log"

# 缓存配置
class CacheConfig:
    ENABLE_CACHE = True
    CACHE_TYPE = "memory"  # memory, redis
    CACHE_TTL = 3600  # 1小时
    CACHE_MAX_SIZE = 1000
    
    # Redis配置（如果使用）
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_PASSWORD = None

# 安全配置
class SecurityConfig:
    # API密钥（生产环境必须设置）
    API_KEY = os.getenv("YOLO_API_KEY", None)
    SECRET_KEY = os.getenv("YOLO_SECRET_KEY", "dev-secret-key")
    
    # 文件上传安全
    ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
    
    # 限流配置
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_REQUESTS = 100
    RATE_LIMIT_WINDOW = 3600  # 1小时

# 监控配置
class MonitorConfig:
    ENABLE_METRICS = True
    METRICS_PORT = 9090
    
    # 健康检查
    HEALTH_CHECK_INTERVAL = 30
    HEALTH_CHECK_TIMEOUT = 10
    
    # 性能监控
    TRACK_INFERENCE_TIME = True
    TRACK_MEMORY_USAGE = True
    TRACK_GPU_USAGE = True

# 开发配置
class DevelopmentConfig:
    DEBUG = True
    RELOAD = True
    LOG_LEVEL = "DEBUG"

# 生产配置
class ProductionConfig:
    DEBUG = False
    RELOAD = False
    LOG_LEVEL = "INFO"
    WORKERS = 4

# 根据环境变量选择配置
ENV = os.getenv("YOLO_ENV", "development")

if ENV == "production":
    Config = ProductionConfig
else:
    Config = DevelopmentConfig

# 导出所有配置类
__all__ = [
    "ServerConfig",
    "ModelConfig", 
    "VideoConfig",
    "LogConfig",
    "CacheConfig",
    "SecurityConfig",
    "MonitorConfig",
    "Config",
    "PROJECT_ROOT",
    "APP_ROOT",
    "CONFIGS_ROOT",
    "SCRIPTS_ROOT"
]
