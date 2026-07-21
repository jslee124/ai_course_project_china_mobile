# -*- coding: utf-8 -*-
"""
高速车流量计数器 —— YOLOv5 + DeepSORT
检测、追踪并分别统计视频中的人员和车辆
"""

import os
import sys
import copy
import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

# 添加 yolov5 子目录到 Python 路径
FILE = Path(__file__).resolve()
ROOT = FILE.parent
YOLO_ROOT = ROOT / "yolov5"
if str(YOLO_ROOT) not in sys.path:
    sys.path.insert(0, str(YOLO_ROOT))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (
    LOGGER, check_img_size, non_max_suppression, scale_coords,
    check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr
)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box

from tracker.utils.parser import get_config
from tracker.deep_sort import DeepSort

# COCO 类别映射
COCO_CLASSES = {
    0: "person",
    1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant",
    21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella",
    26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis",
    31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}

# 人 和 车辆的类别 ID
PERSON_CLASSES = {0}  # person
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
TARGET_CLASSES = PERSON_CLASSES | VEHICLE_CLASSES  # 所有需要检测的类别


def count_obj(box, w, h, track_id, cls_id, counted_ids, line_y, left_zone, right_zone):
    """基于过线检测的计数逻辑，分别统计人员和车辆"""
    # 对象中心点
    center_x = int(box[0] + (box[2] - box[0]) / 2)
    center_y = int(box[1] + (box[3] - box[1]) / 2)

    result = {"person_left": 0, "person_right": 0, "vehicle_left": 0, "vehicle_right": 0}

    # 判断是否穿过计数线
    if center_y > line_y and track_id not in counted_ids:
        is_person = cls_id in PERSON_CLASSES
        if center_x < int(w / 2):
            # 左侧车道
            if is_person:
                result["person_left"] = 1
            else:
                result["vehicle_left"] = 1
        else:
            # 右侧车道
            if is_person:
                result["person_right"] = 1
            else:
                result["vehicle_right"] = 1

    return result


class VehicleCounter:
    """车流量计数器 —— 封装检测、追踪与计数逻辑"""

    def __init__(self, opt):
        self.opt = opt
        self.device = select_device(opt.device)
        self.half = opt.half and self.device.type != "cpu"

        # 加载 YOLO 模型
        self.model = DetectMultiBackend(opt.yolo_model, device=self.device, dnn=opt.dnn)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(opt.imgsz, s=self.stride)

        if self.pt and self.half:
            self.model.model.half()
        else:
            self.model.model.float()

        # 加载 DeepSORT 配置
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)

        # 创建 DeepSORT 实例
        self.deepsort = DeepSort(
            opt.deep_sort_model,
            self.device,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
        )

        # 计数器
        self.person_count = 0
        self.vehicle_count = 0
        self.counted_ids = set()

        # 计数线位置（距底部 350 像素）
        self.counting_line_y = None

    def process_frame(self, im, im0, path, vid_cap, frame_idx):
        """处理单帧图像，返回标注后的图像"""
        h, w = im0.shape[:2]

        # 动态设置计数线
        if self.counting_line_y is None:
            self.counting_line_y = h - 350

        im_tensor = torch.from_numpy(im).to(self.device)
        im_tensor = im_tensor.half() if self.half else im_tensor.float()
        im_tensor /= 255.0
        if len(im_tensor.shape) == 3:
            im_tensor = im_tensor[None]

        # YOLO 检测
        pred = self.model(im_tensor, augment=self.opt.augment, visualize=False)

        # NMS 非极大值抑制
        pred = non_max_suppression(
            pred, self.opt.conf_thres, self.opt.iou_thres,
            self.opt.classes, self.opt.agnostic_nms, max_det=self.opt.max_det
        )

        im0_annotated = im0.copy()
        annotator = Annotator(im0_annotated, line_width=2, pil=not ascii)

        det_results = []

        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(im_tensor.shape[2:], det[:, :4], im0.shape).round()

                # 过滤：只保留 人员 和 车辆
                keep_mask = torch.zeros(len(det), dtype=torch.bool)
                for i, c in enumerate(det[:, -1]):
                    if int(c.item()) in TARGET_CLASSES:
                        keep_mask[i] = True
                det = det[keep_mask]

                if len(det) == 0:
                    continue

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # DeepSORT 追踪
                outputs = self.deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)

                if len(outputs) > 0:
                    for output in outputs:
                        bboxes = output[0:4]
                        track_id = int(output[4])
                        cls_id = int(output[5])

                        # 过线计数
                        result = count_obj(
                            bboxes, w, h, track_id, cls_id,
                            self.counted_ids, self.counting_line_y, w, h
                        )
                        self.person_count += result["person_left"] + result["person_right"]
                        self.vehicle_count += result["vehicle_left"] + result["vehicle_right"]
                        if track_id not in self.counted_ids:
                            if (result["person_left"] + result["person_right"] +
                                    result["vehicle_left"] + result["vehicle_right"]) > 0:
                                self.counted_ids.add(track_id)

                        # 类别名
                        cls_name = COCO_CLASSES.get(cls_id, f"cls{cls_id}")
                        category = "person" if cls_id in PERSON_CLASSES else "vehicle"

                        # 根据类别使用不同颜色
                        if cls_id in PERSON_CLASSES:
                            color = (0, 255, 0)  # 绿色：人
                        else:
                            color = (255, 0, 0)  # 蓝色：车

                        label = f"ID:{track_id} {cls_name} {category}"
                        annotator.box_label(bboxes, label, color=color)

                        det_results.append({
                            "track_id": track_id,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "category": category,
                            "bbox": bboxes.tolist(),
                        })

        # 画计数线
        line_y = self.counting_line_y
        cv2.line(im0_annotated, (0, line_y), (int(w / 2) - 50, line_y), (0, 255, 0), thickness=2)
        cv2.line(im0_annotated, (int(w / 2) + 50, line_y), (w, line_y), (255, 0, 0), thickness=2)

        # 画中间分隔线
        mid_x = int(w / 2)
        cv2.line(im0_annotated, (mid_x, 0), (mid_x, h), (255, 255, 255), thickness=1)

        # 在画面上显示计数结果
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 2

        # 左侧计数（绿色 = 人, 蓝色 = 车）
        cv2.putText(im0_annotated, f"Person: {self.person_count}",
                    (10, 40), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
        cv2.putText(im0_annotated, f"Vehicle: {self.vehicle_count}",
                    (10, 80), font, font_scale, (255, 0, 0), thickness, cv2.LINE_AA)

        return im0_annotated, det_results


def detect(opt, gr_callback=None):
    """主检测函数 —— 可被命令行和 Gradio 前端调用"""
    source = opt.source
    webcam = source == "0" or source.startswith("rtsp") or source.startswith("http") or source.endswith(".txt")
    out = opt.output

    # 输出目录
    if not opt.evaluate:
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out, exist_ok=True)

    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 初始化计数器
    counter = VehicleCounter(opt)
    counter.model.warmup(imgsz=(1 if counter.pt else 1, 3, *counter.imgsz))

    # 数据加载器
    if webcam:
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=counter.imgsz, stride=counter.stride, auto=counter.pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=counter.imgsz, stride=counter.stride, auto=counter.pt)
        nr_sources = 1

    vid_path, vid_writer = [None] * nr_sources, [None] * nr_sources

    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()

        if webcam:
            p, im0 = path[0], im0s[0].copy()
        else:
            p, im0 = path, im0s.copy()

        # 处理帧
        t2 = time_sync()
        dt[0] += t2 - t1

        im0_annotated, dets = counter.process_frame(im, im0, p, vid_cap, frame_idx)

        t3 = time_sync()
        dt[1] += t3 - t2

        seen += 1

        # Gradio 回调：返回原始帧和处理后帧
        if gr_callback is not None:
            gr_callback(im0, im0_annotated)

        # 保存视频
        if opt.save_vid:
            p_path = Path(p)
            if source.endswith(VID_FORMATS):
                save_path = str(save_dir / p_path.name)
            else:
                save_path = str(save_dir / p_path.parent.name)

            save_path = str(Path(save_path).with_suffix(".mp4"))

            if vid_path[0] != save_path:
                vid_path[0] = save_path
                if vid_writer[0] is not None:
                    vid_writer[0].release()
                if vid_cap:
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    fw = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    fh = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:
                    fps, fw, fh = 30, im0.shape[1], im0.shape[0]
                vid_writer[0] = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh)
                )
            vid_writer[0].write(im0_annotated)

        # 显示结果（自动缩放适配屏幕）
        if opt.show_vid:
            display = im0_annotated.copy()
            max_display_w, max_display_h = 1600, 900
            dh, dw = display.shape[:2]
            if dw > max_display_w or dh > max_display_h:
                scale = min(max_display_w / dw, max_display_h / dh)
                display = cv2.resize(display, (int(dw * scale), int(dh * scale)))
            cv2.namedWindow("Vehicle Counter", cv2.WINDOW_NORMAL)
            cv2.imshow("Vehicle Counter", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # 清理
    if opt.save_vid:
        out_path = str(save_dir / "output.mp4")
        if vid_writer[0] is not None:
            vid_writer[0].release()
            print(f"\n视频已保存到: {save_dir}")

    if opt.show_vid:
        cv2.destroyAllWindows()

    # 打印统计
    t = tuple(x / seen * 1E3 for x in dt)
    print(f"\n{'='*50}")
    print(f"  统计结果:")
    print(f"    人员计数: {counter.person_count}")
    print(f"    车辆计数: {counter.vehicle_count}")
    print(f"    总计数:   {counter.person_count + counter.vehicle_count}")
    print(f"  速度: {t[0]:.1f}ms 预处理, {t[1]:.1f}ms 推理")
    print(f"{'='*50}")

    return counter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="高速车流量计数器")

    # 模型参数
    parser.add_argument("--yolo_model", type=str, default="models/yolov5m.pt",
                        help="YOLOv5 模型路径 (使用 COCO 预训练模型)")
    parser.add_argument("--deep_sort_model", type=str, default="osnet_ibn_x1_0_MSMT17",
                        help="DeepSORT ReID 模型")

    # 输入输出
    parser.add_argument("--source", type=str, default="example.mp4",
                        help="输入视频路径")
    parser.add_argument("--output", type=str, default="inference/output",
                        help="输出目录")

    # 检测参数
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int,
                        default=[640], help="推理图像尺寸")
    parser.add_argument("--conf-thres", type=float, default=0.4,
                        help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45,
                        help="NMS IOU 阈值")
    parser.add_argument("--max-det", type=int, default=1000,
                        help="每帧最大检测数")

    # 类别过滤（默认检测所有类别，在 process_frame 中再过滤）
    parser.add_argument("--classes", nargs="+", type=int,
                        help="按 COCO 类别 ID 过滤，如 --classes 0 2")

    # 设备和性能
    parser.add_argument("--device", default="", help="cuda 设备，如 0 或 cpu")
    parser.add_argument("--half", action="store_true",
                        help="使用 FP16 半精度推理 (仅 GPU)")
    parser.add_argument("--agnostic-nms", action="store_true",
                        help="类别无关 NMS")
    parser.add_argument("--augment", action="store_true",
                        help="增强推理")

    # 显示和保存
    parser.add_argument("--show-vid", action="store_true", default=True,
                        help="实时显示推理结果")
    parser.add_argument("--save-vid", action="store_true", default=True,
                        help="保存输出视频")
    parser.add_argument("--save-txt", action="store_true",
                        help="保存 MOT 格式追踪结果")

    # 其他
    parser.add_argument("--config_deepsort", type=str,
                        default="tracker/configs/deep_sort.yaml")
    parser.add_argument("--dnn", action="store_true",
                        help="使用 OpenCV DNN 进行 ONNX 推理")
    parser.add_argument("--evaluate", action="store_true",
                        help="评估模式")
    parser.add_argument("--project", default="runs/track",
                        help="保存结果到 project/name")
    parser.add_argument("--name", default="exp",
                        help="保存结果到 project/name")
    parser.add_argument("--exist-ok", action="store_true",
                        help="允许覆盖已存在的结果目录")
    parser.add_argument("--update", action="store_true",
                        help="更新所有模型")

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1

    with torch.no_grad():
        detect(opt)
