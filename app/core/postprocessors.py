"""Post-processing implementations using strategy pattern."""
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from .interfaces import PostProcessor, InferenceResult
from mindyolo.utils.metrics import (
    non_max_suppression,
    scale_coords,
    xyxy2xywh,
    process_mask_upsample
)
import sys

class DetectionPostProcessor(PostProcessor):
    """Post-processor for object detection."""
    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.65,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.max_det = max_det

    def process(self, model_output: Any, metadata: Dict[str, Any], **kwargs) -> InferenceResult:
        """Process detection model output."""
        original_shape = metadata["original_shape"]
        processed_shape = metadata["processed_shape"][1:]  # CHW -> HW
        
        # Handle output format
        pred = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        
        # Apply NMS
        pred = pred.asnumpy()
        nms_out = non_max_suppression(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes
        )

        # Initialize result containers
        result = InferenceResult(
            category_ids=[],
            bboxes=[],
            scores=[],
            masks=None,
            raw_output=nms_out
        )

        # Process each detection
        for si, det in enumerate(nms_out):
            if len(det) == 0:
                continue
                
            # Scale coordinates to original image
            det_scaled = det.copy()
            scale_coords(processed_shape, det_scaled[:, :4], original_shape)
            
            # Convert to xywh format and adjust coordinates
            boxes = xyxy2xywh(det_scaled[:, :4])
            boxes[:, :2] -= boxes[:, 2:] / 2  # xy center to top-left corner
            
            # Store results
            for di, (d, box) in enumerate(zip(det.tolist(), boxes.tolist())):
                result.category_ids.append(int(d[5]))
                result.bboxes.append([round(x, 3) for x in box])
                result.scores.append(round(d[4], 5))

        return result

class SegmentationPostProcessor(DetectionPostProcessor):
    """Post-processor for instance segmentation."""
    def process(self, model_output: Any, metadata: Dict[str, Any], **kwargs) -> InferenceResult:
        """Process segmentation model output."""
        original_shape = metadata["original_shape"]
        processed_shape = metadata["processed_shape"][1:]  # CHW -> HW
        
        # Split detection and mask outputs
        print("Entering postprocessor process method", file=sys.stderr, flush=True)
        if model_output is None:
            print("Error: model_output is None. Check model inference and input data.", file=sys.stderr, flush=True)
            return None
        print(f"model_output type before unpacking: {type(model_output)}, value: {model_output}", file=sys.stderr, flush=True)
        # Handle case where mask proto is missing
        if len(model_output) < 2 or model_output[1] is None:
            print("Warning: Mask proto output is missing. Running in detection-only mode.", file=sys.stderr, flush=True)
            return None
        try:
            output, (_, _, proto) = model_output
        except TypeError as e:
            print(f"Error unpacking model_output: {e}")
            print(f"model_output type: {type(model_output)}, value: {model_output}")
            return None
        det_out = output.asnumpy()
        proto = proto.asnumpy().squeeze(0)  # 移除批量维度

        # Get number of classes
        nc = kwargs.get("num_classes", 80)
        
        # Apply NMS
        bboxes, mask_coeffs = det_out[..., :nc+5], det_out[..., nc+5:]
        nms_out = non_max_suppression(
                bboxes,
                mask_coeffs,
                conf_thres=self.conf_thres,
                iou_thres=self.iou_thres,
                classes=self.classes
            )

        result = InferenceResult(
            category_ids=[],
            bboxes=[],
            scores=[],
            masks=[],
            raw_output=(nms_out, proto)
        )

        # Process each instance
        for si, (det, inst_proto) in enumerate(zip(nms_out, proto)):
            if len(det) == 0:
                continue
                
            # Generate masks
            pred_masks = process_mask_upsample(
                inst_proto,
                det[:, 6:],
                det[:, :4],
                shape=processed_shape
            )
            pred_masks = pred_masks.astype(np.float32)
            
            # Scale boxes
            det_scaled = det.copy()
            scale_coords(processed_shape, det_scaled[:, :4], original_shape)
            
            # Convert to xywh format
            boxes = xyxy2xywh(det_scaled[:, :4])
            boxes[:, :2] -= boxes[:, 2:] / 2
            
            # Scale masks
            h, w = original_shape
            masks = cv2.resize(pred_masks.transpose(1, 2, 0), (w, h))
            masks = masks > 0.5  # Convert to binary masks
            
            # Store results
            for di, (d, box, mask) in enumerate(zip(det.tolist(), boxes.tolist(), masks.transpose(2, 0, 1))):
                result.category_ids.append(int(d[5]))
                result.bboxes.append([round(x, 3) for x in box])
                result.scores.append(round(d[4], 5))
                result.masks.append(mask)

        return result

class PostProcessorFactory:
    """Factory for creating post-processors."""
    @staticmethod
    def create_processor(task: str, **kwargs) -> PostProcessor:
        """Create post-processor based on task."""
        if task == "detect":
            return DetectionPostProcessor(**kwargs)
        elif task == "segment":
            return SegmentationPostProcessor(**kwargs)
        else:
            raise ValueError(f"Unsupported task: {task}")