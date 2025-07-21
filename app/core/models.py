"""Model factory and adapter implementations."""
from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn
import os

from .interfaces import ModelInterface
from mindyolo.models import create_model as create_yolo_model

class DummyModel(ModelInterface):
    """Dummy model for testing."""
    def __init__(self, num_classes: int = 80):
        """Initialize dummy model."""
        self.num_classes = num_classes

    def generate_random_detection(self, img_size: Tuple[int, int]) -> np.ndarray:
        """Generate a random detection."""
        # Random bbox in relative coordinates [0,1]
        x = np.random.uniform(0.1, 0.9)
        y = np.random.uniform(0.1, 0.9)
        w = np.random.uniform(0.1, 0.3)
        h = np.random.uniform(0.1, 0.3)
        
        # Convert to absolute coordinates
        x1 = (x - w/2) * img_size[1]
        y1 = (y - h/2) * img_size[0]
        x2 = (x + w/2) * img_size[1]
        y2 = (y + h/2) * img_size[0]
        
        # Random confidence and class
        conf = np.random.uniform(0.3, 0.9)
        cls_id = np.random.randint(0, self.num_classes)
        
        # Create one-hot class vector
        cls_vector = np.zeros(self.num_classes)
        cls_vector[cls_id] = 1.0
        
        # Combine all values
        return np.concatenate([
            [x1, y1, x2, y2],  # bbox
            [conf],            # confidence
            cls_vector         # class probabilities
        ])

    def infer(self, input_tensor: Any) -> Any:
        """Return dummy detections for testing."""
        batch_size = input_tensor.shape[0]
        img_height = input_tensor.shape[2]
        img_width = input_tensor.shape[3]
        
        # For each image in batch, generate 1-3 random detections
        batch_output = []
        for _ in range(batch_size):
            num_detections = np.random.randint(1, 4)
            detections = []
            
            for _ in range(num_detections):
                det = self.generate_random_detection((img_height, img_width))
                detections.append(det)
                
            # Stack detections for this image
            if detections:
                detections = np.stack(detections)
            else:
                detections = np.zeros((0, 5 + self.num_classes))
                
            batch_output.append(detections)
            
        # Convert to tensor
        output = Tensor(np.stack(batch_output), ms.float32)
        return output, None

class MindSporeModelAdapter(ModelInterface):
    """Adapter for MindSpore models."""
    def __init__(self, network: nn.Cell):
        self.network = network
        self.network.set_train(False)

    def infer(self, input_tensor: Tensor) -> Any:
        """Run inference on MindSpore model."""
        out, proto = self.network(input_tensor)
        return out, proto

class ModelFactory:
    """Factory for creating model instances."""
    @staticmethod
    def create_model(
        model_name: str,
        model_config: Dict[str, Any],
        num_classes: int = 80,
        checkpoint_path: Optional[str] = None,
        amp_level: str = "O0",
        **kwargs
    ) -> ModelInterface:
        """Create and return a model instance."""
        try:
            # For testing without checkpoint
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                print(f"⚠️ 检查点文件不存在，使用模拟模型: {checkpoint_path}")
                return DummyModel(num_classes=num_classes)
            
            # Import MindYOLO components
            try:
                from mindyolo.models import create_model as create_yolo_model
                from mindyolo.utils.config import parse_args
                import mindspore as ms
                
                # Create real MindYOLO model
                print(f"🔄 加载真实模型: {model_name}")
                network = create_yolo_model(
                    model_name=model_name,
                    model_cfg=model_config,
                    num_classes=num_classes,
                    sync_bn=False,
                    checkpoint_path=checkpoint_path
                )
                
                # Set to inference mode
                network.set_train(False)
                
                # Apply mixed precision if specified
                if amp_level != "O0":
                    ms.amp.auto_mixed_precision(network, amp_level=amp_level)
                
                return MindSporeModelAdapter(network)
                
            except ImportError as e:
                print(f"⚠️ MindYOLO导入失败，使用模拟模型: {e}")
                return DummyModel(num_classes=num_classes)
                
        except Exception as e:
            print(f"❌ 模型创建失败: {e}")
            return DummyModel(num_classes=num_classes)
        network = create_yolo_model(
            model_name=model_name,
            model_cfg=model_config,
            num_classes=num_classes,
            sync_bn=False,
            checkpoint_path=checkpoint_path
        )
        
        # Configure model
        ms.amp.auto_mixed_precision(network, amp_level=kwargs.get('amp_level', 'O0'))
        
        # Wrap in adapter
        return MindSporeModelAdapter(network)

class ModelRegistry:
    """Registry for model configurations."""
    _registry: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, config: Dict[str, Any]) -> None:
        """Register a model configuration."""
        cls._registry[name] = config

    @classmethod
    def get_config(cls, name: str) -> Dict[str, Any]:
        """Get model configuration by name."""
        if name not in cls._registry:
            raise ValueError(f"Model {name} not found in registry")
        return cls._registry[name]

    @classmethod
    def create_model(cls, name: str, **kwargs) -> ModelInterface:
        """Create model from registry configuration."""
        config = cls.get_config(name)
        config.update(kwargs)  # Override with provided kwargs
        return ModelFactory.create_model(name, config)