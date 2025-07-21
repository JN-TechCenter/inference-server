"""Real-time inference module using the refactored pattern-based architecture."""
import os
import cv2
import time
import argparse
from typing import Optional, Tuple, Dict, Any
import numpy as np
import mindspore as ms

from mindyolo.utils import logger, set_seed
from mindyolo.utils.config import parse_args
from demo.predict import get_parser_infer

from core.interfaces import InputSource, Observer, ModelInterface
from core.models import ModelFactory
from core.preprocessors import PreProcessorChain
from core.postprocessors import PostProcessorFactory
from core.visualizers import create_visualizer
from core.input_sources import WebcamSource, VideoSource, ImageSource

class FPSMonitor(Observer):
    """FPS monitoring and display."""
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
        
    def update(self, event_type: str, data: Dict[str, Any]) -> None:
        """Update FPS calculation."""
        if event_type == "frame_processed":
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.current_fps = self.frame_count / elapsed
                
    def draw(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS on frame."""
        cv2.putText(
            frame,
            f'FPS: {self.current_fps:.2f}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        return frame

class InferenceEngine:
    """Main inference engine using the pattern-based architecture."""
    def __init__(self, args):
        self.args = args
        self.model = self._create_model()
        self.preprocessor = PreProcessorChain(640, 32, interpolation=args.interpolation)
        self.postprocessor = PostProcessorFactory.create_processor(
            args.task,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres
        )
        self.visualizer = create_visualizer(
            args.task,
            class_names=args.data.names,
            confidence_threshold=args.conf_thres
        )
        self.fps_monitor = FPSMonitor()
        
    def _create_model(self) -> ModelInterface:
        """Create and configure model."""
        return ModelFactory.create_model(
            model_name=self.args.network.model_name,
            model_config=self.args.network,
            num_classes=self.args.data.nc,
            checkpoint_path=self.args.weight,
            amp_level=self.args.ms_amp_level
        )
        
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Any]:
        """Process a single frame."""
        # Preprocess
        processed_frame, metadata = self.preprocessor.process(frame)
        if processed_frame is None:
            raise ValueError("Preprocessing returned None for frame")
        if not isinstance(processed_frame, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(processed_frame)} instead")
        if processed_frame.size == 0:
            raise ValueError("Preprocessing returned empty array")
        input_tensor = processed_frame.astype(np.float32)
        input_tensor = ms.Tensor(input_tensor[None], ms.float32)
        
        # Inference
        model_output = self.model.infer(input_tensor)
        
        # Postprocess
        result = self.postprocessor.process(
                model_output,
                metadata,
                original_shape=frame.shape[:2],
                input_shape=input_tensor.shape[2:],
                num_classes=self.args.data.nc
            )
        
        # Visualize
        processed_frame = self.visualizer.draw(
            frame,
            result,
            self.args.data.names
        )
        
        # Update FPS
        self.fps_monitor.update("frame_processed", {})
        if self.args.show_fps:
            processed_frame = self.fps_monitor.draw(processed_frame)
            
        return processed_frame, result

def create_input_source(args) -> InputSource:
    """Create appropriate input source based on arguments."""
    if args.mode == "video":
        if not os.path.exists(args.video_path):
            raise ValueError(f"Video file not found: {args.video_path}")
        return VideoSource(args.video_path)
    elif args.mode == "live":
        return WebcamSource(
            camera_index=int(args.camera_index) if args.camera_index.isdigit() else args.camera_index,
            frame_size=tuple(args.frame_size)
        )
    elif args.mode == "image":
        if not args.image_path:
            raise ValueError("Image path required for image mode")
        return ImageSource(args.image_path)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

def create_video_writer(args, frame_size: Tuple[int, int], fps: float) -> Optional[cv2.VideoWriter]:
    """Create video writer if output is requested."""
    if not args.output_video:
        return None
        
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    
    # Try different codecs
    codecs = {
        '.mp4': ['mp4v', 'avc1', 'H264'],
        '.avi': ['XVID'],
    }
    
    ext = os.path.splitext(args.output_video)[1].lower()
    codec_list = codecs.get(ext, ['mp4v', 'avc1', 'XVID'])
    
    for codec in codec_list:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                args.output_video,
                fourcc,
                fps,  # 使用传入的FPS参数
                frame_size
            )
            if writer.isOpened():
                logger.info(f"Created video writer with codec: {codec}")
                return writer
        except Exception as e:
            logger.warning(f"Failed to create writer with codec {codec}: {e}")
            
    raise RuntimeError(f"Failed to create video writer with any codec: {codec_list}")

def realtime_infer(args):
    """Run real-time inference with the new architecture."""
    try:
        # Initialize components
        set_seed(args.seed)
        engine = InferenceEngine(args)
        source = create_input_source(args)
        writer = None
        
        while True:
            # Get frame
            ret, frame = source.get_frame()
            frame_count = frame_count + 1 if 'frame_count' in locals() else 1
            logger.info(f"Read frame {frame_count}, ret={ret}, frame shape={frame.shape if ret else 'None'}")
            if not ret:
                if args.mode == "image":
                    if args.output_video:
                        cv2.imwrite(args.output_video, frame)
                    else:
                        cv2.imshow('Result', frame)
                        cv2.waitKey(0)
                    break
                logger.info("End of input stream")
                break
                
            # Initialize video writer if needed
            if writer is None and args.output_video:
                # 创建输出目录（如果不存在）
                os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
                # 获取视频FPS
                fps = source.get(cv2.CAP_PROP_FPS) if hasattr(source, 'get') else 30.0
                fps = fps if fps > 0 else 30.0
                # 交换宽度和高度以匹配VideoWriter要求的格式 (width, height)
                writer = create_video_writer(args, (frame.shape[1], frame.shape[0]), fps)
                
            # Process frame first to get dimensions
            processed_frame, _ = engine.process_frame(frame)
            logger.info(f"Processed frame {frame_count}, shape={processed_frame.shape}, dtype={processed_frame.dtype}")
            
            # Initialize video writer if needed (now using processed frame size)
            if writer is None and args.output_video:
                # 创建输出目录（如果不存在）
                os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
                # 使用处理后的帧尺寸初始化视频写入器
                # 获取输入视频FPS（带错误处理）
                fps = source.get(cv2.CAP_PROP_FPS) if hasattr(source, 'get') else 30.0
                fps = fps if fps > 0 else 30.0
                writer = create_video_writer(args, (processed_frame.shape[1], processed_frame.shape[0]), fps)
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {args.output_video}")
                logger.info(f"Video writer initialized with size: {processed_frame.shape[1]}x{processed_frame.shape[0]}, FPS: {source.fps}")
            
            # Display results
            cv2.imshow('Detection Result', processed_frame)
            
            # Save output
            if writer is not None:
                # 确保帧格式为BGR（OpenCV默认格式）
                if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
                    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
                processed_frame = processed_frame.astype(np.uint8)
                write_success = writer.write(processed_frame)
                logger.info(f"Wrote frame {frame_count} to video, success={write_success}")
                
            # Handle exit
            if cv2.waitKey(1) == ord(args.exit_key):
                break
                
        # Release resources and log
        if writer is not None:
            writer.release()
            logger.info(f"Video saved to {args.output_video}")
        logger.info("Inference completed successfully")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}", exc_info=True)
        
    finally:
        # Cleanup
        source.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

def get_parser_realtime(parents=None):
    """Get argument parser for real-time inference."""
    base_parser = get_parser_infer()
    parser = argparse.ArgumentParser(
        description='Real-time inference parameters',
        parents=[base_parser] if base_parser else [],
        add_help=False
    )
    
    parser.add_argument('--device_id', type=int, default=0,
                        help='Inference device ID')
    parser.set_defaults(image_path=None, save_result=False)
    parser.add_argument('--show_fps', type=bool, default=True,
                        help='Show FPS counter')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[640, 480],
                        help='Input frame size [width, height]')
    parser.add_argument('--exit_key', type=str, default='q',
                        help='Key to exit program')
    parser.add_argument('--mode', type=str, default='live',
                        choices=['video', 'image', 'live'],
                        help='Inference mode: video, image or live camera')
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to input video file')
    parser.add_argument('--output_video', type=str, default=None,
                        help='Path to output video file')
    parser.add_argument('--camera_index', type=str, default='0',
                        help='Camera index or video stream URL')
    
    return parser

if __name__ == "__main__":
    parser = get_parser_realtime()
    from mindyolo.utils.config import parse_args
    args = parse_args(parser)
    realtime_infer(args)
