"""Core interfaces for the inference system."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

@dataclass
class InferenceResult:
    """Inference result data class."""
    category_ids: List[int]
    bboxes: List[List[float]]
    scores: List[float]
    masks: Optional[List[np.ndarray]] = None
    raw_output: Optional[Any] = None

class PreProcessor(ABC):
    """Abstract base class for preprocessing steps."""
    @abstractmethod
    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Process the input image."""
        pass

class PostProcessor(ABC):
    """Abstract base class for postprocessing steps."""
    @abstractmethod
    def process(self, model_output: Any, original_shape: Tuple[int, int], **kwargs) -> InferenceResult:
        """Process model output."""
        pass

class Visualizer(ABC):
    """Abstract base class for visualization."""
    @abstractmethod
    def draw(self, image: np.ndarray, result: InferenceResult, class_names: List[str], **kwargs) -> np.ndarray:
        """Draw inference results on image."""
        pass

class ModelInterface(ABC):
    """Abstract base class for models."""
    @abstractmethod
    def infer(self, input_tensor: Any) -> Any:
        """Run model inference."""
        pass

class Observer(ABC):
    """Observer interface for progress tracking."""
    @abstractmethod
    def update(self, event_type: str, data: Dict[str, Any]) -> None:
        """Handle update event."""
        pass

class InputSource(ABC):
    """Abstract base class for input sources."""
    @abstractmethod
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get next frame."""
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release resources."""
        pass