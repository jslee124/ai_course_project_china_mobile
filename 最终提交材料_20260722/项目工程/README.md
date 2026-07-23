# 高速车流量计数器

本项目使用 YOLOv5 + DeepSORT 检测、追踪并分别统计人员和车辆，提供命令行与 Gradio 前端。

## 环境

- Python 3.8+
- PyTorch（GPU 可选）
- 安装依赖：`pip install -r requirements.txt`

数据类别顺序必须保持为：`person, car, motorcycle, bus, truck`。

## 命令行推理

```powershell
python main.py --source example.mp4 --yolo_model models/yolov5m.pt --project runs/track --name result --exist-ok
```

结果视频保存在 `runs/track/result/`。目标由 DeepSORT 确认并穿越黄色计数线时计数，同一 Track ID 只计数一次。

## Gradio 前端

```powershell
python webui.py
```

浏览器访问 `http://127.0.0.1:7860`，上传视频后点击“开始处理”。

## 数据标注与转换

规范数据目录为 `data/mydata/`：

- `images/`：训练图片
- `xml/`：Pascal VOC 标注
- `labels/`：YOLO TXT 标签

```powershell
python label_tool.py
python voc2yolo.py
python split_dataset.py
```

## 模型训练

默认参数可直接完成两轮 CPU 训练：

```powershell
python yolov5/train.py
```

GPU 示例：

```powershell
python yolov5/train.py --img 320 --batch-size 2 --epochs 3 --data yolov5/data/mydata.yaml --cfg yolov5/models/yolov5n.yaml --weights models/yolov5n.pt --project runs/train --name my_train --device 0
```

## 注意事项

- 不要调整类别顺序，否则 XML、TXT 和 YAML 会产生类别错位。
- `data/mydata/` 是规范数据目录；`yolov5/data/mydata/` 仅作为历史兼容副本。
- `runs/track/exp6` 和 `runs/track/exp8` 中的视频可正常读取；其他旧实验视频可能因处理被中断而缺少 MP4 尾部索引。
