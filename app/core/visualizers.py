"""Visualization implementations."""
import cv2
import numpy as np
import sys
from typing import List, Dict, Any

from .interfaces import Visualizer, InferenceResult

class DetectionVisualizer(Visualizer):
    """Visualizer for object detection results."""
    def __init__(self, class_names: List[str], confidence_threshold: float = 0.25):
        """Initialize visualizer."""
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.colors = self._generate_colors(len(class_names))
        
    def _generate_colors(self, n: int) -> List[tuple]:
        """Generate random colors for visualization."""
        np.random.seed(42)  # For consistent colors
        colors = []
        for _ in range(n):
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            colors.append(color)
        return colors
        
    def draw(self, image: np.ndarray, result: InferenceResult, class_names: List[str], **kwargs) -> np.ndarray:
        """Draw detection results on image."""
        if result is None:
            print("Warning: No detection results to visualize.", file=sys.stderr, flush=True)
            return image.copy()
        img = image.copy()
        
        for bbox, score, class_id in zip(result.bboxes, result.scores, result.category_ids):
            if score < self.confidence_threshold:
                continue
                
            x1, y1, w, h = map(int, bbox)
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), color, 2)
            
            # Draw label
            label = f"{class_names[class_id]} {score:.2f}"
            font_scale = 0.6
            font_thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            cv2.rectangle(img, (x1, y1 - label_h - baseline), (x1 + label_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
        return img

class SegmentationVisualizer(Visualizer):
    """Visualizer for instance segmentation results."""
    def __init__(self, class_names: List[str], confidence_threshold: float = 0.25, alpha: float = 0.5):
        """Initialize visualizer."""
        self.class_names = class_names
        self.confidence_threshold = confidence_threshold
        self.colors = self._generate_colors(len(class_names))
        self.alpha = alpha
        
    def _generate_colors(self, n: int) -> List[tuple]:
        """Generate random colors for visualization."""
        np.random.seed(42)  # For consistent colors
        colors = []
        for _ in range(n):
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            colors.append(color)
        return colors
        
    def draw(self, image: np.ndarray, result: InferenceResult, class_names: List[str], **kwargs) -> np.ndarray:
        """Draw segmentation results on image."""
        if result is None:
            print("Warning: No detection results to visualize.", file=sys.stderr, flush=True)
            return image.copy()
        img = image.copy()
        
        if result.masks is None:
            return img
            
        # Create overlay for masks
        overlay = img.copy()
        for mask, score, class_id in zip(result.masks, result.scores, result.category_ids):
            if score < self.confidence_threshold:
                continue
                
            color = self.colors[class_id % len(self.colors)]
            colored_mask = np.zeros_like(img)
            colored_mask[mask] = color
            
            # Blend mask with image
            cv2.addWeighted(overlay, 1 - self.alpha, colored_mask, self.alpha, 0, overlay)
            
        # Draw bounding boxes and labels on top
        for bbox, score, class_id in zip(result.bboxes, result.scores, result.category_ids):
            if score < self.confidence_threshold:
                continue
                
            x1, y1, w, h = map(int, bbox)
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x1 + w, y1 + h), color, 2)
            
            # Draw label
            label = f"{class_names[class_id]} {score:.2f}"
            font_scale = 0.6
            font_thickness = 1
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            cv2.rectangle(overlay, (x1, y1 - label_h - baseline), (x1 + label_w, y1), color, -1)
            cv2.putText(overlay, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
            
        return overlay

def create_visualizer(task: str, **kwargs) -> Visualizer:
    """Factory function to create visualizer based on task."""
    if task == "detect":
        return DetectionVisualizer(**kwargs)
    elif task == "segment":
        return SegmentationVisualizer(**kwargs)
    else:
        raise ValueError(f"Unsupported task: {task}")