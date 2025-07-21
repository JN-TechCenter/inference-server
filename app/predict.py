import argparse
import ast
import math
import os
import sys
import time
import cv2
import numpy as np
import yaml
from datetime import datetime

import mindspore as ms
from mindspore import Tensor, nn

from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.models import create_model
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh, process_mask_upsample, scale_image
from mindyolo.utils.utils import draw_result, set_seed
def get_parser_infer(parents=None):
    parser = argparse.ArgumentParser(description="推理任务参数", parents=[parents] if parents else [])
    
    # 在现有参数基础上添加data参数
    parser.add_argument('--data', type=str, default='configs/yolov5/yolov5s.yaml', 
                       help='数据集配置文件路径')
    
    parser.add_argument("--task", type=str, default="detect", choices=["detect", "segment"],
                        help="推理任务类型：detect=检测，segment=分割")

    parser.add_argument("--device_target", type=str, default="Ascend",
                        help="设备选择：Ascend、GPU 或 CPU")

    parser.add_argument("--ms_mode", type=int, default=0,
                        help="MindSpore执行模式：0=图模式，1=动态图（pynative）")

    parser.add_argument("--ms_amp_level", type=str, default="O0",
                        help="自动混合精度等级：O0=不使用，O1/O2=混合精度等级")

    parser.add_argument("--ms_enable_graph_kernel", type=ast.literal_eval, default=False,
                        help="是否启用图算融合（Graph Kernel）")

    parser.add_argument("--precision_mode", type=str, default=None,
                        help="精度模式设置（用于Ascend设备）")

    parser.add_argument("--weight", type=str, default="yolov7_300.ckpt",
                        help="权重文件路径（model.ckpt）")

    parser.add_argument("--img_size", type=int, default=640,
                        help="输入图片尺寸（正方形边长，单位像素）")

    parser.add_argument("--interpolation", type=str, default='default', choices=['default', 'nearest'],
                        help="插值方法: 'default' (默认) 或 'nearest' (最近邻)")

    parser.add_argument("--single_cls", type=ast.literal_eval, default=False,
                        help="是否将多类别数据视为单类别")

    parser.add_argument("--exec_nms", type=ast.literal_eval, default=True,
                        help="是否执行非极大值抑制（NMS）")

    parser.add_argument("--nms_time_limit", type=float, default=60.0,
                        help="NMS的时间限制（秒）")

    parser.add_argument("--conf_thres", type=float, default=0.25,
                        help="置信度阈值（低于该值的框会被过滤）")

    parser.add_argument("--iou_thres", type=float, default=0.65,
                        help="NMS的IOU阈值")

    parser.add_argument("--conf_free", type=ast.literal_eval, default=False,
                        help="是否从预测结果中独立剥离置信度值")

    parser.add_argument("--seed", type=int, default=2,
                        help="随机种子（用于结果复现）")

    parser.add_argument("--log_level", type=str, default="INFO",
                        help="日志等级，如INFO、DEBUG等")

    parser.add_argument("--save_dir", type=str, default="./runs_infer",
                        help="推理结果保存目录")

    parser.add_argument("--image_path", type=str,
                        help="输入图片路径")

    parser.add_argument("--save_result", type=ast.literal_eval, default=False,
                        help="是否保存推理结果")

    parser.add_argument("--log_file", type=str, default="infer.log",
                        help="日志文件保存路径")

    return parser



# 在set_default_infer函数中禁用日志文件
def set_default_infer(args):
    # 设置 MindSpore 执行上下文模式
    ms.set_context(mode=args.ms_mode)
    
    # 设置最大递归深度
    ms.set_recursion_limit(2000)

    # 如果设置了精度模式，则配置精度（用于 Ascend）
    if args.precision_mode is not None:
        ms.device_context.ascend.op_precision.precision_mode(args.precision_mode)

    # 如果使用图模式，则设置 JIT 编译等级为 O2
    if args.ms_mode == 0:
        ms.set_context(jit_config={"jit_level": "O2"})

    # 如果目标设备是 Ascend，则将设备设置为 CPU（实际可根据需要改为Ascend/GPU）
    if args.device_target == "Ascend":
        ms.set_device("CPU", int(os.getenv("DEVICE_ID", 0)))

    # 设置 rank 和 rank_size（用于多卡训练，这里默认单卡）xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxpppppppppppppppppppppppppppppppppp
    args.rank, args.rank_size = 0, 1

    # 根据是否为单类别任务，设置类别数和类别名称
    args.data.nc = 1 if args.single_cls else int(args.data.nc)  # 类别数
    args.data.names = ["item"] if args.single_cls and len(args.names) != 1 else args.data.names  # 类别名称

    # 校验类别名称和类别数量是否一致
    assert len(args.data.names) == args.data.nc, "%g names found for nc=%g dataset in %s" % (
        len(args.data.names),
        args.data.nc,
        args.config,
    )

    # 设置保存目录名，格式为：./runs_infer/YYYY.MM.DD-HH:MM:SS
    platform = sys.platform
    if platform == "win32":
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    else:
        args.save_dir = os.path.join(args.save_dir, datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 如果是主进程，保存当前参数配置为 cfg.yaml 文件
    if args.rank % args.rank_size == 0:
        with open(os.path.join(args.save_dir, "cfg.yaml"), "w") as f:
            yaml.dump(vars(args), f, sort_keys=False)

    # 设置日志系统：初始化基本日志记录器
    logger.setup_logging(logger_name="MindYOLO", log_level="INFO", rank_id=args.rank, device_per_servers=args.rank_size)

    # 设置日志文件保存路径
    # 注释掉日志文件设置
    # logger.setup_logging_file(log_dir=os.path.join(args.save_dir, "logs"))
def detect(
    network: nn.Cell,               # MindSpore模型
    img: np.ndarray,                # 输入图像（BGR格式）
    conf_thres: float = 0.25,       # 置信度阈值
    iou_thres: float = 0.65,        # NMS的IoU阈值
    conf_free: bool = False,        # 是否不包含置信度（用于特殊模型）
    exec_nms: bool = True,          # 是否执行NMS
    nms_time_limit: float = 60.0,   # NMS时间限制
    img_size: int = 640,            # 推理图像尺寸
    stride: int = 32,               # 模型下采样步长
    num_class: int = 80,            # 类别数
    is_coco_dataset: bool = True,   # 是否使用COCO数据集（用于标签映射）
):
    # 图像缩放处理
    h_ori, w_ori = img.shape[:2]  # 原图尺寸
    r = img_size / max(h_ori, w_ori)  # 缩放比例
    if r != 1:  # 若需缩放图像
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    h, w = img.shape[:2]
    
    # 如果尺寸仍不足，填充边界到符合步长
    if h < img_size or w < img_size:
        new_h, new_w = math.ceil(h / stride) * stride, math.ceil(w / stride) * stride
        dh, dw = (new_h - h) / 2, (new_w - w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 填充灰色

    # 图像格式转换并归一化
    logger.debug(f'[预处理] 转换前帧信息 形状：{img.shape} 通道顺序：BGR')
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0  # BGR->RGB并转为CHW，归一化
    logger.debug(f'[预处理] 转换后帧形状：{img.shape} 数值范围：[{img.min():.3f}, {img.max():.3f}]')
    imgs_tensor = Tensor(img[None], ms.float32)  # 增加batch维度
    logger.debug(f'[张量转换] 输入张量形状：{imgs_tensor.shape} 数据类型：{imgs_tensor.dtype}')

    # 推理阶段
    _t = time.time()
    out, _ = network(imgs_tensor)
    out = out[-1] if isinstance(out, (tuple, list)) else out  # 取最终输出
    infer_times = time.time() - _t

    # 执行NMS去重
    t = time.time()
    out = out.asnumpy()
    out = non_max_suppression(
        out, conf_thres=conf_thres, iou_thres=iou_thres, conf_free=conf_free,
        multi_label=True, time_limit=nms_time_limit, need_nms=exec_nms
    )
    nms_times = time.time() - t

    # 处理推理结果
    logger.info(f'[推理输出] 原始检测框数量：{sum(len(p) for p in out)} 有效帧：{len([p for p in out if len(p)>0])}/{len(out)}')
    result_dict = {"category_id": [], "bbox": [], "score": []}
    total_category_ids, total_bboxes, total_scores = [], [], []
    for si, pred in enumerate(out):
        logger.debug(f'帧{si+1} 初始检测框：{len(pred)}个')
        if len(pred) == 0:
            logger.debug(f'帧{si+1} 无有效检测结果')
            continue
        predn = np.copy(pred)
        scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))  # 将坐标缩放回原图尺寸

        box = xyxy2xywh(predn[:, :4])  # 转为xywh格式
        box[:, :2] -= box[:, 2:] / 2  # 中心坐标转左上角

        # 提取结果信息
        category_ids, bboxes, scores = [], [], []
        for p, b in zip(pred.tolist(), box.tolist()):
            category_ids.append(COCO80_TO_COCO91_CLASS[int(p[5])] if is_coco_dataset else int(p[5]))
            bboxes.append([round(x, 3) for x in b])
            scores.append(round(p[4], 5))

        total_category_ids.extend(category_ids)
        total_bboxes.extend(bboxes)
        total_scores.extend(scores)

    # 写入结果字典
    result_dict["category_id"].extend(total_category_ids)
    result_dict["bbox"].extend(total_bboxes)
    result_dict["score"].extend(total_scores)

    # 打印日志信息
    t = tuple(x * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size, 1)
    logger.info(f"Predict result is: {result_dict}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
    logger.info(f"Detect a image success.")

    # 在返回结果前添加校验
    if len(total_category_ids) == 0:
        logger.warning("未检测到有效目标，请检查：")
        logger.warning(f"- 置信度阈值（当前值：{conf_thres}）")
        logger.warning("- 模型输入尺寸是否合适")
        logger.warning("- 权重文件是否与模型匹配")
    else:
        logger.info(f"检测到{len(total_category_ids)}个目标，类别分布：{np.bincount(total_category_ids)}")
    
    return result_dict


def segment(
    network: nn.Cell,
    img: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.65,
    conf_free: bool = False,
    nms_time_limit: float = 60.0,
    img_size: int = 640,
    stride: int = 32,
    num_class: int = 80,
    is_coco_dataset: bool = True,
):
    # 图像缩放
    h_ori, w_ori = img.shape[:2]
    r = img_size / max(h_ori, w_ori)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    h, w = img.shape[:2]
    if h < img_size or w < img_size:
        new_h, new_w = math.ceil(h / stride) * stride, math.ceil(w / stride) * stride
        dh, dw = (new_h - h) / 2, (new_w - w) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # 格式转换与归一化
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    imgs_tensor = Tensor(img[None], ms.float32)

    # 推理阶段
    _t = time.time()
    out, (_, _, prototypes) = network(imgs_tensor)  # 同时输出mask原型
    infer_times = time.time() - _t

    # NMS处理
    t = time.time()
    _c = num_class + 4 if conf_free else num_class + 5
    out = out.asnumpy()
    bboxes, mask_coefficient = out[:, :, :_c], out[:, :, _c:]
    out = non_max_suppression(
        bboxes, mask_coefficient,
        conf_thres=conf_thres, iou_thres=iou_thres,
        conf_free=conf_free, multi_label=True,
        time_limit=nms_time_limit
    )
    nms_times = time.time() - t

    prototypes = prototypes.asnumpy()

    result_dict = {"category_id": [], "bbox": [], "score": [], "segmentation": []}
    total_category_ids, total_bboxes, total_scores, total_seg = [], [], [], []

    # 遍历每个图像的结果
    for si, (pred, proto) in enumerate(zip(out, prototypes)):
        if len(pred) == 0:
            continue

        # 生成掩码
        pred_masks = process_mask_upsample(proto, pred[:, 6:], pred[:, :4], shape=imgs_tensor[si].shape[1:])
        pred_masks = pred_masks.astype(np.float32)
        pred_masks = scale_image((pred_masks.transpose(1, 2, 0)), (h_ori, w_ori))

        # 坐标缩放回原图
        predn = np.copy(pred)
        scale_coords(img.shape[1:], predn[:, :4], (h_ori, w_ori))
        box = xyxy2xywh(predn[:, :4])
        box[:, :2] -= box[:, 2:] / 2

        category_ids, bboxes, scores, segs = [], [], [], []
        for ii, (p, b) in enumerate(zip(pred.tolist(), box.tolist())):
            category_ids.append(COCO80_TO_COCO91_CLASS[int(p[5])] if is_coco_dataset else int(p[5]))
            bboxes.append([round(x, 3) for x in b])
            scores.append(round(p[4], 5))
            segs.append(pred_masks[:, :, ii])  # 掩码添加

        total_category_ids.extend(category_ids)
        total_bboxes.extend(bboxes)
        total_scores.extend(scores)
        total_seg.extend(segs)

    # 写入结果字典
    result_dict["category_id"].extend(total_category_ids)
    result_dict["bbox"].extend(total_bboxes)
    result_dict["score"].extend(total_scores)
    result_dict["segmentation"].extend(total_seg)

    # 输出日志
    t = tuple(x * 1e3 for x in (infer_times, nms_times, infer_times + nms_times)) + (img_size, img_size, 1)
    logger.info(f"Predict result is:")
    for k, v in result_dict.items():
        if k == "segmentation":
            logger.info(f"{k} shape: {v[0].shape}")
        else:
            logger.info(f"{k}: {v}")
    logger.info(f"Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;" % t)
    logger.info(f"Detect a image success.")

    return result_dict
def infer(args, frame=None):
    # 增强型输入验证（添加图像格式检查）
    logger.debug(f"[输入验证] 原始帧类型：{type(args.image_path)} 形状：{args.image_path.shape if hasattr(args.image_path, 'shape') else 'N/A'}")
    if frame is not None:
        img = frame.copy()
        logger.debug(f'[输入验证] 内存帧信息 类型：{img.dtype} 形状：{img.shape} 均值：{np.mean(img):.2f}')
    elif isinstance(args.image_path, np.ndarray):
        if args.image_path.dtype != np.uint8 or args.image_path.ndim != 3:
            raise ValueError("视频帧格式必须为uint8类型的3维numpy数组（HWC-BGR格式）")
        img = args.image_path.copy()  # 保持原始帧不变
        logger.debug(f'[输入验证] 拷贝后帧信息 类型：{img.dtype} 形状：{img.shape} 均值：{np.mean(img):.2f}')
    elif isinstance(args.image_path, str) and os.path.isfile(args.image_path):
        img = cv2.imread(args.image_path)
        if img is None:
            raise ValueError(f"无法读取图像文件：{args.image_path}")
    else:
        raise ValueError(f"不支持的输入类型：{type(args.image_path)}，支持类型：numpy数组或图片路径")

    # 移除冗余的图片加载代码块
    # if isinstance(args.image_path, str) and os.path.isfile(args.image_path):
    #     import cv2
    #     img = cv2.imread(args.image_path)  # 读取图像
    # else:
    #     raise ValueError("Detect: input image file not available.") 

    # 添加输入格式日志记录
    logger.debug(f"输入图像类型：{type(img)}，形状：{img.shape}，数据类型：{img.dtype}")

    # 初始化随机种子
    set_seed(args.seed)
    # 设置推理默认参数
    set_default_infer(args)

    # 创建网络模型
    network = create_model(
        model_name=args.network.model_name,      # 模型名称
        model_cfg=args.network,                  # 模型配置
        num_classes=args.data.nc,                # 类别数量
        sync_bn=False,                           # 是否同步 BatchNorm（推理中通常设为 False）
        checkpoint_path=args.weight,             # 模型权重文件路径
    )
    # 设置模型为推理模式（非训练模式）
    network.set_train(False)
    # 启用自动混合精度（如果设置了 AMP 级别）
    ms.amp.auto_mixed_precision(network, amp_level=args.ms_amp_level)

    # 增强型输入处理（支持内存帧和文件路径）
    if frame is not None:
        img = frame.copy()
        logger.debug(f'[输入验证] 内存帧信息 类型：{img.dtype} 形状：{img.shape} 均值：{np.mean(img):.2f}')
    elif isinstance(args.image_path, np.ndarray):
        if args.image_path.dtype != np.uint8 or args.image_path.ndim != 3:
            raise ValueError("视频帧格式必须为uint8类型的3维numpy数组（HWC-BGR格式）")
        img = args.image_path.copy()
    else:
        if not os.path.exists(args.image_path):
            raise ValueError(f"Detect: input image file not available.")  # 图像路径无效时抛出异常
        img = cv2.imread(args.image_path)

    # 判断是否为 COCO 数据集
    is_coco_dataset = "coco" in args.data.dataset_name

    # 根据任务类型进行推理（修改draw_result调用方式）
    if args.task == "detect":
        result_dict = detect(
            network=network,
            img=img,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            conf_free=args.conf_free,
            exec_nms=args.exec_nms,
            nms_time_limit=args.nms_time_limit,
            img_size=args.img_size,
            stride=max(max(args.network.stride), 32),
            num_class=args.data.nc,
            is_coco_dataset=is_coco_dataset,
        )
        # 修改draw_result调用方式
        plot_img = draw_result(
            img,  # 直接使用图像数据而不是路径
            result_dict, 
            args.data.names,
            is_coco_dataset=is_coco_dataset
        )
        # 在detect任务的结果保存部分添加目录创建
        if args.save_result:
            save_dir = os.path.join(args.save_dir, "detect_results")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"result_{int(time.time())}.jpg")
            
            if plot_img is not None:
                cv2.imwrite(save_path, plot_img)
            else:
                logger.error("Failed to save result: empty output image")

    elif args.task == "segment":
        result_dict = segment(
            network=network,
            img=img,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            conf_free=args.conf_free,
            nms_time_limit=args.nms_time_limit,
            img_size=args.img_size,
            stride=max(max(args.network.stride), 32),
            num_class=args.data.nc,
            is_coco_dataset=is_coco_dataset,
        )

        # 修复分割任务绘图参数
        plot_img = draw_result(


            img,  # 改为直接使用图像数据
            result_dict,
            args.data.names,
            is_coco_dataset=is_coco_dataset
        )
        if args.save_result:
            save_path = os.path.join(args.save_dir, "segment_results", "result.jpg")
            cv2.imwrite(save_path, plot_img)
            logger.info(f'分割结果已保存至：{save_path}')

    # 返回处理后的图像（保持BGR格式）
    return plot_img



if __name__ == "__main__":
    # 获取推理参数解析器
    parser = get_parser_infer()
    # 解析命令行参数
    args = parse_args(parser)
    # 执行推理
    infer(args)