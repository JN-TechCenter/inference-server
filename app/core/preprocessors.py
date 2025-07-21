"""Preprocessing implementations using chain of responsibility pattern."""
import cv2
import numpy as np
from typing import Tuple, Optional, Any

from .interfaces import PreProcessor

class ResizePreProcessor(PreProcessor):
    """Resize images to target size."""
    def __init__(self, target_size: int, keep_ratio: bool = True, interpolation: str = 'default'):
        self.target_size = target_size
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation
        self._next: Optional[PreProcessor] = None

    def set_next(self, processor: PreProcessor) -> PreProcessor:
        """Set next processor in chain."""
        self._next = processor
        return processor

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Resize image while optionally keeping aspect ratio."""
        h_ori, w_ori = image.shape[:2]
        
        if self.keep_ratio:
            r = self.target_size / max(h_ori, w_ori)
            if r != 1:
                if self.interpolation == 'nearest':
                    interp = cv2.INTER_NEAREST
            else:
                interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
                image = cv2.resize(image, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
        else:
            image = cv2.resize(image, (self.target_size, self.target_size))
        
        return self._next.process(image, **kwargs) if self._next else image

class PadPreProcessor(PreProcessor):
    """Pad images to match stride requirements."""
    def __init__(self, stride: int = 32, pad_value: int = 114):
        self.stride = stride
        self.pad_value = pad_value
        self._next: Optional[PreProcessor] = None

    def set_next(self, processor: PreProcessor) -> PreProcessor:
        """Set next processor in chain."""
        self._next = processor
        return processor

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Pad image to match stride."""
        h, w = image.shape[:2]
        new_h = int(np.ceil(h / self.stride) * self.stride)
        new_w = int(np.ceil(w / self.stride) * self.stride)
        
        if h == new_h and w == new_w:
            padded = image
        else:
            padded = np.full((new_h, new_w, 3), self.pad_value, dtype=np.uint8)
            padded[:h, :w] = image
            
        return self._next.process(padded, **kwargs) if self._next else padded

class ColorTransformPreProcessor(PreProcessor):
    """Transform image color space."""
    def __init__(self):
        self._next: Optional[PreProcessor] = None

    def set_next(self, processor: PreProcessor) -> PreProcessor:
        """Set next processor in chain."""
        self._next = processor
        return processor

    def process(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Convert BGR to RGB and normalize."""
        image = image[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        image = image / 255.0  # 0 - 255 to 0.0 - 1.0
        return self._next.process(image, **kwargs) if self._next else image

class PreProcessorChain:
    """Chain of preprocessors for image processing."""
    def __init__(self, target_size: int = 640, stride: int = 32, interpolation: str = 'default'):
        # Create processors
        self.resize = ResizePreProcessor(target_size, interpolation=interpolation)
        self.pad = PadPreProcessor(stride)
        self.color = ColorTransformPreProcessor()
        
        # Build chain
        self.resize.set_next(self.pad)
        self.pad.set_next(self.color)
        
        # Store head of chain
        self.head = self.resize
        
        # Store metadata
        self.target_size = target_size
        self.stride = stride
        
    def process(self, image: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
        """Process image through the chain."""
        if image is None:
            raise ValueError("Input image is None")
        
        # Store original shape for reference
        original_shape = image.shape[:2]
        
        # Process image through chain
        processed = self.head.process(image.copy(), **kwargs)
        
        # Return processed image and metadata
        metadata = {
            "original_shape": original_shape,
            "processed_shape": processed.shape,
            "target_size": self.target_size,
            "stride": self.stride,
        }
        
        return processed, metadata