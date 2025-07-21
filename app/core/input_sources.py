"""Input source implementations."""
import cv2
import numpy as np
from typing import Optional, Tuple

from .interfaces import InputSource

class WebcamSource(InputSource):
    """Webcam input source."""
    def __init__(self, camera_index: int = 0, frame_size: Tuple[int, int] = (640, 480)):
        """Initialize webcam source."""
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open camera {camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
        
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get next frame from webcam."""
        return self.cap.read()
        
    def release(self) -> None:
        """Release webcam resources."""
        if self.cap:
            self.cap.release()

class VideoSource(InputSource):
    """Video file input source."""
    def __init__(self, video_path: str):
        """Initialize video source."""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
            
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get next frame from video."""
        return self.cap.read()
        
    def release(self) -> None:
        """Release video resources."""
        if self.cap:
            self.cap.release()

class ImageSource(InputSource):
    """Single image input source."""
    def __init__(self, image_path: str):
        """Initialize image source."""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        self.returned = False
            
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the image (only once)."""
        if self.returned:
            return False, None
        self.returned = True
        return True, self.image.copy()
        
    def release(self) -> None:
        """Release image resources."""
        self.image = None