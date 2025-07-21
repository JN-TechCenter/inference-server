"""
配置管理模块
统一管理所有配置项
"""
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIGS_ROOT = PROJECT_ROOT / "_configs"
WEIGHTS_ROOT = PROJECT_ROOT / "weights" 
MODELS_ROOT = PROJECT_ROOT / "models"

@dataclass
class ServerConfig:
    """服务器配置"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    workers: int = 1
    max_upload_size: int = 100 * 1024 * 1024  # 100MB
    
    # API配置
    api_prefix: str = "/api/v1"
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    
    # CORS配置
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])

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
    
    # 推理参数
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.65
    input_size: int = 640
    
    # 性能参数
    batch_size: int = 1
    amp_level: str = "O0"  # 混合精度等级

@dataclass 
class InferenceConfig:
    """推理配置"""
    max_workers: int = 8
    queue_size: int = 1000
    timeout: float = 60.0
    
    # 并发控制
    max_concurrent_per_user: int = 5
    global_concurrent_limit: int = 50

@dataclass
class SessionConfig:
    """会话配置"""
    session_timeout: int = 3600  # 1小时
    cleanup_interval: int = 300  # 5分钟
    
    # 用户配额
    daily_limit: int = 1000
    hourly_limit: int = 100
    concurrent_limit: int = 5

class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self.server_config = ServerConfig()
        self.inference_config = InferenceConfig()
        self.session_config = SessionConfig()
        self.model_configs: Dict[str, ModelConfig] = {}
        
        # 加载配置
        self._load_configs()
        self._register_default_models()
    
    def _load_configs(self):
        """加载配置文件"""
        try:
            config_file = CONFIGS_ROOT / "inference_config.yaml"
            if config_file.exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                    self._update_configs_from_dict(config_data)
        except Exception as e:
            print(f"⚠️ 配置文件加载失败，使用默认配置: {e}")
    
    def _update_configs_from_dict(self, config_data: Dict[str, Any]):
        """从字典更新配置"""
        if 'server' in config_data:
            server_data = config_data['server']
            self.server_config.host = server_data.get('host', self.server_config.host)
            self.server_config.port = server_data.get('port', self.server_config.port)
        
        if 'postprocess' in config_data:
            postprocess = config_data['postprocess']
            self.inference_config.timeout = postprocess.get('timeout', self.inference_config.timeout)
    
    def _register_default_models(self):
        """注册默认模型配置"""
        default_models = [
            {
                "name": "yolov5s",
                "config": "yolov5s.yaml",
                "weight": "yolov5s.ckpt"
            },
            {
                "name": "yolov8s", 
                "config": "yolov8-base.yaml",
                "weight": "yolov8s.ckpt"
            },
            {
                "name": "yolov11s",
                "config": "yolov11-base.yaml", 
                "weight": "yolov11s.ckpt"
            }
        ]
        
        for model_info in default_models:
            config_path = MODELS_ROOT / model_info["config"]
            weight_path = WEIGHTS_ROOT / model_info["name"] / model_info["weight"]
            
            model_config = ModelConfig(
                model_name=model_info["name"],
                model_path=str(weight_path),
                config_path=str(config_path)
            )
            
            self.model_configs[model_info["name"]] = model_config
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """获取模型配置"""
        return self.model_configs.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return list(self.model_configs.keys())
    
    def add_model_config(self, model_config: ModelConfig):
        """添加模型配置"""
        self.model_configs[model_config.model_name] = model_config
    
    def remove_model_config(self, model_name: str) -> bool:
        """移除模型配置"""
        if model_name in self.model_configs:
            del self.model_configs[model_name]
            return True
        return False
    
    def save_config(self, config_file: Optional[Path] = None):
        """保存配置到文件"""
        if config_file is None:
            config_file = CONFIGS_ROOT / "runtime_config.yaml"
        
        config_data = {
            "server": {
                "host": self.server_config.host,
                "port": self.server_config.port,
                "debug": self.server_config.debug,
                "workers": self.server_config.workers
            },
            "inference": {
                "max_workers": self.inference_config.max_workers,
                "queue_size": self.inference_config.queue_size,
                "timeout": self.inference_config.timeout
            },
            "session": {
                "session_timeout": self.session_config.session_timeout,
                "cleanup_interval": self.session_config.cleanup_interval,
                "daily_limit": self.session_config.daily_limit,
                "hourly_limit": self.session_config.hourly_limit
            },
            "models": {}
        }
        
        for name, model_config in self.model_configs.items():
            config_data["models"][name] = {
                "model_path": model_config.model_path,
                "config_path": model_config.config_path,
                "device": model_config.device,
                "max_instances": model_config.max_instances,
                "confidence_threshold": model_config.confidence_threshold,
                "iou_threshold": model_config.iou_threshold
            }
        
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            return True
        except Exception as e:
            print(f"❌ 配置保存失败: {e}")
            return False
    
    def validate_config(self) -> Dict[str, Any]:
        """验证配置有效性"""
        issues = []
        
        # 检查模型文件是否存在
        for name, config in self.model_configs.items():
            if not Path(config.config_path).exists():
                issues.append(f"模型配置文件不存在: {config.config_path}")
            
            if not Path(config.model_path).exists():
                issues.append(f"模型权重文件不存在: {config.model_path}")
        
        # 检查端口是否可用
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((self.server_config.host, self.server_config.port))
        except OSError:
            issues.append(f"端口 {self.server_config.port} 已被占用")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "total_models": len(self.model_configs),
            "available_models": self.get_available_models()
        }

# 全局配置管理器实例
config_manager = ConfigManager()
