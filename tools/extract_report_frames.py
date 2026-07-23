from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "runs" / "audit" / "fixed_cli" / "example.mp4"
OUT = ROOT / "submission_work" / "frame_samples"
OUT.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = [int(total * ratio) for ratio in (0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64, 0.72, 0.80, 0.88)]
frames = []
for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        continue
    path = OUT / f"frame_{idx:03d}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    frames.append((idx, Image.open(path).convert("RGB")))
cap.release()

thumb_w, thumb_h = 480, 270
sheet = Image.new("RGB", (thumb_w * 3, (thumb_h + 42) * 4), "white")
draw = ImageDraw.Draw(sheet)
font_path = "C:/Windows/Fonts/msyh.ttc"
font = ImageFont.truetype(font_path, 24)
for pos, (idx, frame) in enumerate(frames):
    x = (pos % 3) * thumb_w
    y = (pos // 3) * (thumb_h + 42)
    sheet.paste(frame.resize((thumb_w, thumb_h)), (x, y))
    draw.text((x + 8, y + thumb_h + 5), f"帧 {idx}", font=font, fill="#15324B")
sheet.save(OUT / "contact_sheet.jpg", quality=94)
print(OUT / "contact_sheet.jpg")
