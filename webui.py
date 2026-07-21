# -*- coding: utf-8 -*-
"""
高速车流量计数器 —— Gradio 前端
支持上传视频，实时显示检测、追踪与计数结果
"""

import os
import sys
import argparse
import copy
from pathlib import Path
import tempfile

import numpy as np
import cv2
import torch
import gradio as gr

# 从 main.py 导入核心模块
from main import VehicleCounter, COCO_CLASSES, TARGET_CLASSES, PERSON_CLASSES


class GradioVehicleCounter:
    """封装 Gradio 前端的车流量计数器"""

    def __init__(self, yolo_model="yolov5/yolov5m.pt", deep_sort_model="osnet_ibn_x1_0_MSMT17",
                 device="", imgsz=640, conf_thres=0.4, iou_thres=0.45):
        # 构建参数
        parser = argparse.ArgumentParser()
        self.opt = parser.parse_args([])
        self.opt.yolo_model = yolo_model
        self.opt.deep_sort_model = deep_sort_model
        self.opt.device = device
        self.opt.imgsz = [imgsz]
        self.opt.conf_thres = conf_thres
        self.opt.iou_thres = iou_thres
        self.opt.classes = None
        self.opt.agnostic_nms = False
        self.opt.augment = False
        self.opt.max_det = 1000
        self.opt.half = False
        self.opt.dnn = False
        self.opt.config_deepsort = "tracker/configs/deep_sort.yaml"

        self.counter = VehicleCounter(self.opt)

    def process_video(self, video_path, progress=gr.Progress()):
        """处理上传的视频，返回输出视频路径"""
        if video_path is None:
            return None, "请先上传视频文件"

        print(f"\n开始处理视频: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "无法打开视频文件"

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 如果视频太大，做尺寸限制
        max_dim = 1280
        if max(frame_width, frame_height) > max_dim:
            scale = max_dim / max(frame_width, frame_height)
            frame_width = int(frame_width * scale)
            frame_height = int(frame_height * scale)
            if frame_width % 2 != 0:
                frame_width -= 1
            if frame_height % 2 != 0:
                frame_height -= 1

        # 输出视频
        output_path = os.path.join(tempfile.gettempdir(), "output_count.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # 重置 DeepSORT (新视频新计数)
        from tracker.utils.parser import get_config
        from tracker.deep_sort import DeepSort
        cfg = get_config()
        cfg.merge_from_file(self.opt.config_deepsort)
        self.counter.deepsort = DeepSort(
            self.opt.deep_sort_model, self.counter.device,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE,
            n_init=cfg.DEEPSORT.N_INIT,
            nn_budget=cfg.DEEPSORT.NN_BUDGET,
        )
        self.counter.person_count = 0
        self.counter.vehicle_count = 0
        self.counter.counted_ids = set()
        self.counter.track_frames = {}
        self.counter.counting_line_y = None

        frame_idx = 0

        with torch.no_grad():
            for frame_idx in progress.tqdm(range(total_frames), desc="处理中..."):
                ret, frame = cap.read()
                if not ret:
                    break

                # 预处理
                h, w = frame.shape[:2]
                if self.counter.counting_line_y is None:
                    self.counter.counting_line_y = h - 350

                # 缩放
                if (w, h) != (frame_width, frame_height):
                    frame = cv2.resize(frame, (frame_width, frame_height))

                # 预处理 (匹配 YOLOv5 输入)
                im = cv2.resize(frame, (self.opt.imgsz[0], self.opt.imgsz[0]))
                im = im.transpose(2, 0, 1)  # HWC → CHW
                im = np.ascontiguousarray(im)

                # 处理帧
                annotated, _ = self.counter.process_frame(im, frame, "", None, frame_idx)

                out.write(annotated)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        summary = f"""
        **处理完成!**
        - 总帧数: {frame_idx + 1}
        - 人员计数: **{self.counter.person_count}**
        - 车辆计数: **{self.counter.vehicle_count}**
        - 总计: **{self.counter.person_count + self.counter.vehicle_count}**
        """

        return output_path, summary

    def process_frame_single(self, image):
        """处理单张图片"""
        if image is None:
            return None, "请上传图片"

        with torch.no_grad():
            h, w = image.shape[:2]
            self.counter.counting_line_y = h - 350

            im = cv2.resize(image, (self.opt.imgsz[0], self.opt.imgsz[0]))
            im = im.transpose(2, 0, 1)
            im = np.ascontiguousarray(im)

            annotated, dets = self.counter.process_frame(im, image, "", None, 0)

        info = f"检测到 {len(dets)} 个目标 | 人员: {self.counter.person_count} | 车辆: {self.counter.vehicle_count}"
        return annotated, info


def create_interface():
    """创建 Gradio 界面"""
    # 添加 yolov5 到 path
    yolo_root = Path(__file__).resolve().parent / "yolov5"
    if str(yolo_root) not in sys.path:
        sys.path.insert(0, str(yolo_root))

    processor = GradioVehicleCounter(
        yolo_model="models/yolov5m.pt",
        deep_sort_model="osnet_ibn_x1_0_MSMT17",
    )

    with gr.Blocks(title="高速车流量计数器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚗 高速车流量计数器
        ### YOLOv5 + DeepSORT —— 检测、追踪并分别统计人员与车辆

        上传视频或图片，系统将自动检测并追踪画面中的**人员**和**车辆**（小汽车、摩托车、公交车、卡车），
        并基于过线检测进行实时计数。
        """)

        with gr.Tabs():
            # Tab 1: 视频处理
            with gr.TabItem("🎬 视频处理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="上传视频", sources=["upload"])
                        process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        video_output = gr.Video(label="处理结果")
                        result_text = gr.Markdown("等待处理...")

                process_btn.click(
                    fn=processor.process_video,
                    inputs=[video_input],
                    outputs=[video_output, result_text],
                )

                gr.Markdown("""
                ---
                ### 使用说明
                1. 上传一个 `.mp4` 视频文件
                2. 点击 **开始处理** 按钮
                3. 等待处理完成，查看输出视频和统计结果
                4. 可下载处理后的视频

                ### 计数规则
                - 🟢 **绿色框** = 人员
                - 🔵 **蓝色框** = 车辆（小汽车、摩托车、公交车、卡车）
                - 目标穿过画面下方的计数线时触发计数
                """)

            # Tab 2: 图片检测
            with gr.TabItem("📷 图片检测"):
                with gr.Row():
                    with gr.Column():
                        img_input = gr.Image(label="上传图片", type="numpy")
                        img_btn = gr.Button("🔍 检测", variant="primary")

                    with gr.Column():
                        img_output = gr.Image(label="检测结果", type="numpy")
                        img_info = gr.Markdown("等待检测...")

                img_btn.click(
                    fn=processor.process_frame_single,
                    inputs=[img_input],
                    outputs=[img_output, img_info],
                )

            # Tab 3: 项目信息
            with gr.TabItem("ℹ️ 关于"):
                gr.Markdown("""
                ## 技术方案

                | 模块 | 技术 |
                |------|------|
                | 目标检测 | **YOLOv5** (COCO pre-trained) |
                | 目标追踪 | **DeepSORT** (OSNet ReID) |
                | 前端框架 | **Gradio** |
                | 计数方式 | 过线检测 (crossing-line) |

                ## 支持的类别

                **人员 (Person):** COCO class 0
                **车辆 (Vehicle):** car(2), motorcycle(3), bus(5), truck(7)

                ## 项目结构
                ```
                ├── main.py          # 主检测追踪程序
                ├── webui.py         # Gradio 前端
                ├── label_tool.py    # 数据标注工具
                ├── voc2yolo.py      # XML → TXT 格式转换
                ├── split_dataset.py # 数据集划分
                ├── yolov5/          # YOLOv5 代码
                ├── tracker/         # DeepSORT 追踪器
                └── models/          # 预训练权重
                ```
                """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
