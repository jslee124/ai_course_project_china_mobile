# -*- coding: utf-8 -*-
"""
数据集划分工具 —— 将图片按比例划分为 train / val / test
生成 YOLOv5 训练所需的 train.txt 和 val.txt 路径文件
"""

import os
import random
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="数据集划分")
    parser.add_argument("--image_dir", type=str, default="data/mydata/images",
                        help="图片目录")
    parser.add_argument("--labels_dir", type=str, default="data/mydata/labels",
                        help="YOLO 标签目录")
    parser.add_argument("--output_dir", type=str, default="data/mydata",
                        help="输出目录")
    parser.add_argument("--trainval_ratio", type=float, default=0.9,
                        help="训练+验证集占比")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集在 trainval 中的占比")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    image_files = sorted([
        f.stem for f in image_dir.glob("*")
        if f.suffix.lower() in image_exts
    ])
    if not image_files:
        print(f"错误: 在 {image_dir} 下未找到图片文件")
        return

    print(f"图片目录: {image_dir}")
    print(f"标签目录: {labels_dir}")
    print(f"找到 {len(image_files)} 张图片")

    # 检查哪些图片有对应的标注文件
    annotated = []
    unannotated = []
    for name in image_files:
        label_path = labels_dir / f"{name}.txt"
        if label_path.exists():
            annotated.append(name)
        else:
            unannotated.append(name)

    print(f"已标注: {len(annotated)} 张, 未标注: {len(unannotated)} 张")

    if len(annotated) == 0:
        print("错误: 没有找到任何已标注的图片，请先运行 voc2yolo.py 转换标注")
        return

    # 使用有标注的数据进行划分
    random.seed(args.seed)
    random.shuffle(annotated)
    num = len(annotated)
    tv = int(num * args.trainval_ratio)
    tr = int(tv * args.train_ratio)

    train = annotated[:tr]
    val = annotated[tr:tv]
    test = annotated[tv:]

    print(f"\n划分结果:")
    print(f"  train:       {len(train)} 张")
    print(f"  val:         {len(val)} 张")
    print(f"  test:        {len(test)} 张")
    print(f"  未标注(跳过): {len(unannotated)} 张")

    abs_image_dir = image_dir.resolve()

    # 写入 train.txt (包含图片绝对路径)
    for split_name, subset in [("train", train), ("val", val), ("test", test), ("trainval", train + val)]:
        output_path = output_dir / f"{split_name}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            for name in subset:
                img_path = abs_image_dir / f"{name}.jpg"
                if not img_path.exists():
                    # 尝试其他扩展名
                    for ext in image_exts:
                        alt = abs_image_dir / f"{name}{ext}"
                        if alt.exists():
                            img_path = alt
                            break
                f.write(f"{img_path}\n")
        print(f"  已写入: {output_path}")

    print(f"\n数据集划分完成!")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
