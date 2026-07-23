# -*- coding: utf-8 -*-
"""
数据标注工具 —— 使用 OpenCV 进行目标检测框标注
支持类别：person(人), car(车), motorcycle(摩托车), bus(公交车), truck(卡车)
输出格式：Pascal VOC XML
"""

import cv2
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse
from pathlib import Path

# 类别定义
CLASSES = ["person", "car", "motorcycle", "bus", "truck"]
CLASS_COLORS = {
    "person": (0, 255, 0),       # 绿色
    "car": (255, 0, 0),          # 蓝色
    "motorcycle": (0, 255, 255), # 黄色
    "bus": (255, 0, 255),        # 紫色
    "truck": (0, 0, 255),        # 红色
}

class LabelTool:
    def __init__(self, image_dir, xml_dir="xml"):
        self.image_dir = Path(image_dir)
        self.xml_dir = Path(xml_dir)
        self.xml_dir.mkdir(parents=True, exist_ok=True)

        self.image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(self.image_exts)
        ])
        if not self.image_files:
            print(f"错误: 在 {self.image_dir} 下未找到图片文件")
            sys.exit(1)

        self.idx = 0
        self.current_class = 0
        self.drawing = False
        self.start_x = self.start_y = -1
        self.end_x = self.end_y = -1
        self.boxes = []  # [(xmin, ymin, xmax, ymax, class_name)]
        self.undo_stack = []
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.load_current_image()
        cv2.namedWindow("Label Tool", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Label Tool", 1280, 720)
        cv2.setMouseCallback("Label Tool", self.mouse_callback)

    def load_current_image(self):
        path = os.path.join(self.image_dir, self.image_files[self.idx])
        self.raw_img = cv2.imread(path)
        if self.raw_img is None:
            print(f"无法加载图片: {path}")
            return
        self.img_h, self.img_w = self.raw_img.shape[:2]
        self.load_annotations()
        self.display_img = self.raw_img.copy()

    def load_annotations(self):
        self.boxes = []
        name = Path(self.image_files[self.idx]).stem
        xml_path = self.xml_dir / f"{name}.xml"
        if xml_path.exists():
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            for obj in root.findall("object"):
                cls = obj.find("name").text
                bndbox = obj.find("bndbox")
                xmin = int(float(bndbox.find("xmin").text))
                ymin = int(float(bndbox.find("ymin").text))
                xmax = int(float(bndbox.find("xmax").text))
                ymax = int(float(bndbox.find("ymax").text))
                self.boxes.append((xmin, ymin, xmax, ymax, cls))

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.end_x, self.end_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end_x, self.end_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_x, self.end_y = x, y
            xmin = min(self.start_x, self.end_x)
            ymin = min(self.start_y, self.end_y)
            xmax = max(self.start_x, self.end_x)
            ymax = max(self.start_y, self.end_y)
            w, h = xmax - xmin, ymax - ymin
            if w > 5 and h > 5:
                cls = CLASSES[self.current_class]
                self.boxes.append((xmin, ymin, xmax, ymax, cls))

    def draw(self):
        img = self.raw_img.copy()
        for (xmin, ymin, xmax, ymax, cls) in self.boxes:
            color = CLASS_COLORS.get(cls, (128, 128, 128))
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            label = f"{cls}"
            cv2.putText(img, label, (xmin, max(ymin - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if self.drawing:
            color = CLASS_COLORS.get(CLASSES[self.current_class], (128, 128, 128))
            cv2.rectangle(img, (self.start_x, self.start_y),
                          (self.end_x, self.end_y), color, 1)

        info_lines = [
            f"Image: {self.idx + 1}/{len(self.image_files)}  {self.image_files[self.idx]}",
            f"Class: [{self.current_class}] {CLASSES[self.current_class]}",
            f"Boxes: {len(self.boxes)}",
            "Keys: 0-4=switch class | S=save | A/D=prev/next",
            "      U=undo | C=clear all | Q=quit",
        ]
        y0 = img.shape[0] - 20 * len(info_lines) - 10
        overlay = img.copy()
        cv2.rectangle(overlay, (0, y0 - 5), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        for i, line in enumerate(info_lines):
            cv2.putText(img, line, (10, y0 + 15 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.display_img = img

    def save(self):
        name = Path(self.image_files[self.idx]).stem
        xml_path = self.xml_dir / f"{name}.xml"

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = self.image_dir.name
        ET.SubElement(annotation, "filename").text = self.image_files[self.idx]
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(self.img_w)
        ET.SubElement(size, "height").text = str(self.img_h)
        ET.SubElement(size, "depth").text = "3"

        for (xmin, ymin, xmax, ymax, cls) in self.boxes:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = cls
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(max(xmin, 0))
            ET.SubElement(bndbox, "ymin").text = str(max(ymin, 0))
            ET.SubElement(bndbox, "xmax").text = str(min(xmax, self.img_w))
            ET.SubElement(bndbox, "ymax").text = str(min(ymax, self.img_h))

        xml_str = minidom.parseString(ET.tostring(annotation, "utf-8")).toprettyxml(indent="  ")
        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_str)
        print(f"已保存: {xml_path} ({len(self.boxes)} 个标注框)")

    def navigate(self, delta):
        new_idx = self.idx + delta
        if 0 <= new_idx < len(self.image_files):
            self.idx = new_idx
            self.load_current_image()
            print(f"切换到: {self.image_files[self.idx]}")

    def run(self):
        print("=" * 60)
        print("数据标注工具已启动")
        print(f"图片目录: {self.image_dir}")
        print(f"XML 标注输出: {self.xml_dir}")
        print(f"共找到 {len(self.image_files)} 张图片")
        print(f"类别: {CLASSES}")
        print("=" * 60)
        print("操作说明:")
        print("  0-4  切换标注类别")
        print("  鼠标拖拽  绘制标注框")
        print("  S    保存当前标注")
        print("  A/D  上一张/下一张")
        print("  U    撤销上一个框")
        print("  C    清除所有框")
        print("  Q    退出")
        print("=" * 60)

        while True:
            self.draw()
            cv2.imshow("Label Tool", self.display_img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save()
            elif key == ord('a'):
                self.navigate(-1)
            elif key == ord('d'):
                self.navigate(1)
            elif key == ord('u'):
                if self.boxes:
                    self.boxes.pop()
                    print("已撤销上一个框")
            elif key == ord('c'):
                self.boxes.clear()
                print("已清除所有框")
            elif key >= ord('0') and key <= ord('4'):
                self.current_class = key - ord('0')
                print(f"当前类别: {CLASSES[self.current_class]}")
            elif key == 27:  # ESC
                break

        cv2.destroyAllWindows()
        print("标注工具已退出")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="目标检测数据标注工具")
    parser.add_argument("--image_dir", type=str, default="data/mydata/images",
                        help="图片目录")
    parser.add_argument("--xml_dir", type=str, default="data/mydata/xml",
                        help="XML 标签输出目录")
    args = parser.parse_args()

    tool = LabelTool(image_dir=args.image_dir, xml_dir=args.xml_dir)
    tool.run()
