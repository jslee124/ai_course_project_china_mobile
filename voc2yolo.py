# -*- coding: utf-8 -*-
"""
XML (Pascal VOC) → TXT (YOLO) 标注格式转换工具
将 label_tool.py 或任何 VOC 格式标注转换为 YOLOv5 训练所需的 TXT 格式
"""

import xml.etree.ElementTree as ET
import os
from pathlib import Path
import argparse


def convert(size, box):
    """将 VOC 框 (xmin, xmax, ymin, ymax) → YOLO 归一化框 (x_center, y_center, w, h)"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(xml_path, labels_dir, classes, image_id):
    """把单个 XML 标注文件转换成 YOLO TXT 格式"""
    in_file = open(xml_path, encoding="utf-8")
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    out_path = os.path.join(labels_dir, "%s.txt" % image_id)
    out_file = open(out_path, "w", encoding="utf-8")

    for obj in root.iter("object"):
        difficult = obj.find("difficult")
        if difficult is not None:
            difficult = difficult.text
        else:
            difficult = "0"
        cls_name = obj.find("name").text
        if cls_name not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls_name)
        xmlbox = obj.find("bndbox")
        xmin = float(xmlbox.find("xmin").text)
        xmax = float(xmlbox.find("xmax").text)
        ymin = float(xmlbox.find("ymin").text)
        ymax = float(xmlbox.find("ymax").text)

        # 边界修正
        xmax = min(xmax, w)
        ymax = min(ymax, h)
        xmin = max(xmin, 0)
        ymin = max(ymin, 0)

        b = (xmin, xmax, ymin, ymax)
        bb = convert((w, h), b)
        out_file.write(f"{cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")

    out_file.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="XML -> YOLO TXT 标注转换")
    parser.add_argument("--xml_dir", type=str, default="data/mydata/xml",
                        help="XML 标注文件目录")
    parser.add_argument("--labels_dir", type=str, default="data/mydata/labels",
                        help="YOLO TXT 输出目录")
    parser.add_argument("--classes", nargs="+", type=str,
                        default=["person", "car", "motorcycle", "bus", "truck"],
                        help="类别列表")
    args = parser.parse_args()

    xml_dir = Path(args.xml_dir)
    labels_dir = Path(args.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if not xml_dir.exists():
        print(f"错误: XML 目录不存在: {xml_dir}")
        return

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"错误: 在 {xml_dir} 下未找到 XML 文件")
        return

    print(f"类别: {args.classes}")
    print(f"XML 目录: {xml_dir}")
    print(f"TXT 输出: {labels_dir}")
    print(f"找到 {len(xml_files)} 个 XML 文件")

    count = 0
    for xml_path in xml_files:
        image_id = xml_path.stem
        try:
            convert_annotation(str(xml_path), str(labels_dir), args.classes, image_id)
            count += 1
            print(f"  ✓ {xml_path.name} → {image_id}.txt")
        except Exception as e:
            print(f"  ✗ {xml_path.name} 转换失败: {e}")

    print(f"\n转换完成! {count}/{len(xml_files)} 个文件成功转换")
    print(f"输出目录: {labels_dir}")


if __name__ == "__main__":
    main()
