from pathlib import Path
from datetime import date

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "submission_work" / "项目报告"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUT_DIR / "高速车流量计数器项目报告.docx"
ARCH_PATH = OUT_DIR / "system_architecture.png"
DATA_OVERVIEW_PATH = OUT_DIR / "dataset_overview.png"

SKY = "2E74B5"
NAVY = "15324B"
INK = "20262E"
MUTED = "66717D"
PALE = "EAF2F8"
LIGHT = "F3F6F8"
GREEN = "2F7D5A"
GOLD = "B78018"
RED = "A33A3A"
WHITE = "FFFFFF"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths_dxa, indent_dxa=120):
    total = sum(widths_dxa)
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            width = widths_dxa[min(idx, len(widths_dxa) - 1)]
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(width))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_font(run, name="Microsoft YaHei", size=10.5, color=INK, bold=False, italic=False):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.font.color.rgb = RGBColor.from_string(color)
    run.bold = bold
    run.italic = italic


def add_text(doc, text, *, bold_label=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY,
             before=0, after=6, line=1.10, color=INK, keep=False):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = line
    p.paragraph_format.keep_together = keep
    # 中文正式报告正文：首行缩进两个汉字（10.5 pt × 2 = 21 pt）。
    # 标题、表格、图注、代码块和居中文字由各自函数处理，不应用此缩进。
    if align in (WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.JUSTIFY):
        p.paragraph_format.first_line_indent = Pt(21)
    if bold_label and text.startswith(bold_label):
        r1 = p.add_run(bold_label)
        set_font(r1, bold=True, color=color)
        r2 = p.add_run(text[len(bold_label):])
        set_font(r2, color=color)
    else:
        r = p.add_run(text)
        set_font(r, color=color)
    return p


def add_code(doc, code):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.22)
    p.paragraph_format.right_indent = Inches(0.12)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.05
    p_pr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), "F4F6F8")
    p_pr.append(shd)
    r = p.add_run(code)
    set_font(r, name="Consolas", size=8.7, color="263238")
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_font(r, size={1: 16, 2: 13, 3: 11.5}[level], color=SKY if level < 3 else NAVY, bold=True)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    set_font(r, size=9, color=MUTED)
    return p


def add_table(doc, headers, rows, widths, aligns=None, header_fill=PALE):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.allow_autofit = False
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        set_cell_shading(cell, header_fill)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(header)
        set_font(r, size=9.5, color=NAVY, bold=True)
    set_repeat_table_header(table.rows[0])
    for row_idx, values in enumerate(rows):
        cells = table.add_row().cells
        if row_idx % 2 == 1:
            for cell in cells:
                set_cell_shading(cell, "FAFBFC")
        for i, value in enumerate(values):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = (aligns[i] if aligns else WD_ALIGN_PARAGRAPH.LEFT)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.08
            r = p.add_run(str(value))
            set_font(r, size=9.2)
    set_table_geometry(table, widths)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    return table


def add_status_callout(doc, label, text, fill=LIGHT, accent=GREEN):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.18)
    p.paragraph_format.right_indent = Inches(0.12)
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(9)
    p.paragraph_format.line_spacing = 1.12
    p_pr = p._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    p_pr.append(shd)
    borders = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), "18")
    left.set(qn("w:color"), accent)
    left.set(qn("w:space"), "7")
    borders.append(left)
    p_pr.append(borders)
    r1 = p.add_run(label + "  ")
    set_font(r1, color=accent, bold=True)
    r2 = p.add_run(text)
    set_font(r2)


def page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = paragraph.add_run("第 ")
    set_font(run, size=9, color=MUTED)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    tail = paragraph.add_run(" 页")
    set_font(tail, size=9, color=MUTED)


def style_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1.0)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(1.0)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    section.different_first_page_header_footer = True

    normal = doc.styles["Normal"]
    normal.font.name = "Microsoft YaHei"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Microsoft YaHei")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Microsoft YaHei")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    for level, size, before, after, color in (
        (1, 16, 16, 8, SKY),
        (2, 13, 12, 6, SKY),
        (3, 11.5, 8, 4, NAVY),
    ):
        style = doc.styles[f"Heading {level}"]
        style.font.name = "Microsoft YaHei"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Microsoft YaHei")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Microsoft YaHei")
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True

    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hp.paragraph_format.space_after = Pt(0)
    hr = hp.add_run("高速车流量计数器 · 项目成果报告")
    set_font(hr, size=8.5, color=MUTED)
    footer = section.footer
    page_number(footer.paragraphs[0])


def make_architecture():
    width, height = 1800, 760
    im = Image.new("RGB", (width, height), "#FFFFFF")
    d = ImageDraw.Draw(im)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    bold_path = Path("C:/Windows/Fonts/msyhbd.ttc")
    font = ImageFont.truetype(str(font_path), 36)
    small = ImageFont.truetype(str(font_path), 27)
    bold = ImageFont.truetype(str(bold_path if bold_path.exists() else font_path), 38)
    title = ImageFont.truetype(str(bold_path if bold_path.exists() else font_path), 46)
    d.text((70, 45), "高速车流量计数器处理流程", font=title, fill="#15324B")

    boxes = [
        (70, 220, 330, 470, "视频输入", "OpenCV 逐帧读取"),
        (390, 220, 700, 470, "目标检测", "YOLOv5\n人员/车辆类别"),
        (760, 220, 1070, 470, "多目标追踪", "DeepSORT\nTrack ID 关联"),
        (1130, 220, 1430, 470, "穿线计数", "轨迹中心穿越\n同一 ID 仅一次"),
        (1490, 220, 1730, 470, "结果输出", "标注视频 + 统计"),
    ]
    colors = ["#F3F6F8", "#EAF2F8", "#E8F3EE", "#FFF5DE", "#F1EDF8"]
    for idx, (x1, y1, x2, y2, name, desc) in enumerate(boxes):
        d.rounded_rectangle((x1, y1, x2, y2), radius=24, fill=colors[idx], outline="#B9C5D0", width=4)
        bbox = d.textbbox((0, 0), name, font=bold)
        d.text(((x1 + x2 - (bbox[2] - bbox[0])) / 2, y1 + 45), name, font=bold, fill="#15324B")
        lines = desc.split("\n")
        for line_idx, line_text in enumerate(lines):
            bb = d.textbbox((0, 0), line_text, font=small)
            d.text(((x1 + x2 - (bb[2] - bb[0])) / 2, y1 + 125 + line_idx * 48), line_text, font=small, fill="#44515E")
        if idx < len(boxes) - 1:
            ax1 = x2 + 18
            ax2 = boxes[idx + 1][0] - 18
            ay = (y1 + y2) // 2
            d.line((ax1, ay, ax2, ay), fill="#2E74B5", width=7)
            d.polygon([(ax2, ay), (ax2 - 22, ay - 15), (ax2 - 22, ay + 15)], fill="#2E74B5")
    d.text((70, 590), "核心原则：检测负责“看见”，追踪负责“识别同一目标”，穿线逻辑负责“只计一次”。", font=font, fill="#2F7D5A")
    im.save(ARCH_PATH, quality=95)


def make_dataset_overview():
    width, height = 1800, 900
    im = Image.new("RGB", (width, height), "#FFFFFF")
    d = ImageDraw.Draw(im)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    bold_path = Path("C:/Windows/Fonts/msyhbd.ttc")
    regular = ImageFont.truetype(str(font_path), 32)
    small = ImageFont.truetype(str(font_path), 27)
    bold = ImageFont.truetype(str(bold_path if bold_path.exists() else font_path), 42)
    number = ImageFont.truetype(str(bold_path if bold_path.exists() else font_path), 64)
    title = ImageFont.truetype(str(bold_path if bold_path.exists() else font_path), 48)

    d.text((70, 45), "数据集概览", font=title, fill="#15324B")
    cards = [
        ("图片", "9", "#EAF2F8"),
        ("VOC XML", "9", "#E8F3EE"),
        ("YOLO TXT", "9", "#FFF5DE"),
        ("标注框", "21", "#F1EDF8"),
    ]
    card_w, gap = 370, 45
    for idx, (label, value, fill) in enumerate(cards):
        x1 = 70 + idx * (card_w + gap)
        x2 = x1 + card_w
        d.rounded_rectangle((x1, 145, x2, 380), radius=24, fill=fill, outline="#B9C5D0", width=3)
        bb = d.textbbox((0, 0), value, font=number)
        d.text(((x1 + x2 - (bb[2] - bb[0])) / 2, 185), value, font=number, fill="#2E74B5")
        bb = d.textbbox((0, 0), label, font=regular)
        d.text(((x1 + x2 - (bb[2] - bb[0])) / 2, 300), label, font=regular, fill="#44515E")

    d.text((70, 455), "数据划分（图片数）", font=bold, fill="#15324B")
    splits = [("训练集", 6, "#2E74B5"), ("验证集", 2, "#2F7D5A"), ("测试集", 1, "#B78018")]
    max_bar = 980
    for idx, (label, value, color) in enumerate(splits):
        y = 555 + idx * 92
        d.text((90, y + 7), label, font=regular, fill="#44515E")
        d.rounded_rectangle((270, y, 270 + max_bar, y + 55), radius=18, fill="#EDF1F4")
        bar_w = int(max_bar * value / 6)
        d.rounded_rectangle((270, y, 270 + bar_w, y + 55), radius=18, fill=color)
        d.text((1290, y + 5), f"{value} 张", font=regular, fill="#15324B")
    d.text((70, 835), "类别说明：当前自建标注以 car 为主；完整系统仍支持 person、car、motorcycle、bus、truck 五类。", font=small, fill="#66717D")
    im.save(DATA_OVERVIEW_PATH, quality=96)


def cover_page(doc):
    add_text(doc, "人工智能课程项目成果报告", align=WD_ALIGN_PARAGRAPH.CENTER, after=55, color=GOLD)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("高速车流量计数器")
    set_font(r, size=28, color=NAVY, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(60)
    r = p.add_run("基于 YOLOv5 + DeepSORT 的检测、追踪与穿线计数系统")
    set_font(r, size=14, color=SKY)

    rows = [
        ("学生姓名", "____________________"),
        ("学号", "____________________"),
        ("班级 / 专业", "____________________"),
        ("指导教师", "____________________"),
        ("提交日期", date.today().strftime("%Y 年 %m 月 %d 日")),
    ]
    table = doc.add_table(rows=0, cols=2)
    table.style = "Table Grid"
    for label, value in rows:
        cells = table.add_row().cells
        cells[0].text = ""
        cells[1].text = ""
        p1 = cells[0].paragraphs[0]
        p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r1 = p1.add_run(label)
        set_font(r1, color=NAVY, bold=True)
        p2 = cells[1].paragraphs[0]
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = p2.add_run(value)
        set_font(r2)
        set_cell_shading(cells[0], PALE)
    set_table_geometry(table, [2450, 6910])
    doc.add_paragraph().paragraph_format.space_after = Pt(18)
    add_text(doc, "项目状态：已完成工程修复、命令行推理、Gradio 处理、数据转换与训练烟雾测试。", align=WD_ALIGN_PARAGRAPH.CENTER, color=MUTED)
    doc.add_page_break()


def build_report():
    make_architecture()
    make_dataset_overview()
    doc = Document()
    style_document(doc)
    cover_page(doc)

    add_heading(doc, "摘要", 1)
    add_text(doc, "本项目面向高速公路固定机位视频，构建了一个基于 YOLOv5 与 DeepSORT 的交通目标检测、持续追踪和穿线计数系统。系统可读取本地视频，对 person、car、motorcycle、bus、truck 五类目标进行检测；通过 DeepSORT 为目标分配稳定的 Track ID；当轨迹中心跨越预设计数线时完成人员与车辆分类累计，并以 Track ID 去重，避免同一目标在连续帧中重复计数。项目同时提供命令行处理、Gradio 可视化前端、Pascal VOC 标注工具、XML 转 YOLO TXT、数据集划分和模型训练入口，形成从数据准备到推理展示的完整流程。")
    add_text(doc, "经实际验证，示例视频共 473 帧、分辨率 1280×720，命令行与 Gradio 两条处理链均成功完成，最终统计人员 0、车辆 15，并生成可正常播放的 MP4。标注工具可加载 9 张图片；9 份 XML 已转换为 9 份 YOLO 标签，共 21 个标注框；1 轮小样本训练成功产出 best.pt。由于当前训练集规模很小且仅包含车辆样本，训练结果仅用于证明流程可运行，不能代表正式模型精度。")
    add_status_callout(doc, "关键词", "YOLOv5；DeepSORT；多目标追踪；穿线计数；Gradio；交通流量分析", fill="F4F6F9", accent=SKY)

    add_heading(doc, "项目成果概览", 1)
    add_table(doc, ["成果项", "对应文件 / 结果", "状态"], [
        ("完整工程", "main.py、webui.py、tracker/、yolov5/", "已完成"),
        ("数据标注", "label_tool.py、data/mydata/xml/", "已验证"),
        ("格式转换", "voc2yolo.py、data/mydata/labels/", "9/9 成功"),
        ("模型训练", "yolov5/train.py、results/training/", "1 轮跑通"),
        ("前端可视化", "webui.py（Gradio）", "已跑通"),
        ("处理后视频", "results/processed_example.mp4", "473 帧完整输出"),
    ], [1900, 5260, 2200], aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER])
    add_text(doc, "说明：压缩包内已排除虚拟环境、Git 历史、重复的原始工程和无效旧测试视频，以减小体积并避免提交内容歧义。", bold_label="说明：", color=MUTED)

    doc.add_page_break()
    add_heading(doc, "1 项目背景与需求分析", 1)
    add_heading(doc, "1.1 项目目标", 2)
    add_text(doc, "高速视频中目标数量多、运动速度快，单帧检测只能回答“当前画面有什么”，无法判断连续帧中的目标是否为同一对象。为获得可靠流量统计，系统必须同时完成目标检测、身份关联和事件计数。本项目以课程要求为约束，将数据标注、格式转换、模型训练、视频推理和前端展示整合为一套可复现工程。")
    add_heading(doc, "1.2 功能需求映射", 2)
    add_table(doc, ["课程要求", "实现方式", "验收依据"], [
        ("程序与测试视频", "OpenCV 读取，YOLOv5 推理，MP4 写出", "473 帧完整处理"),
        ("数据标注", "Tkinter 标注工具输出 Pascal VOC XML", "9 张图片可加载"),
        ("XML 转 TXT", "按统一类别顺序归一化为 YOLO 标签", "9 XML / 21 标注框"),
        ("模型训练", "YOLOv5 train.py + mydata.yaml", "1 轮训练产出权重"),
        ("追踪与计数", "DeepSORT Track ID + 中心点穿线去重", "人员/车辆分别累计"),
        ("Gradio 前端", "上传视频、处理进度、结果视频回传", "完整视频处理成功"),
    ], [1900, 4560, 2900])
    add_heading(doc, "1.3 运行环境", 2)
    add_text(doc, "工程面向 Windows 或 Linux，要求 Python 3.8 及以上、PyTorch、OpenCV、Gradio，以及 CPU 或可选 CUDA GPU。提交包不包含 venv，避免平台相关二进制文件造成体积膨胀；依赖统一记录在 requirements.txt 中。")

    doc.add_page_break()
    add_heading(doc, "2 系统总体设计", 1)
    add_text(doc, "系统采用串行视频处理架构。每一帧先经过与 YOLOv5 一致的 letterbox 预处理和 BGR→RGB 色彩转换，再执行目标检测与非极大值抑制；有效检测框送入 DeepSORT 完成状态预测、外观特征匹配和轨迹更新；最后依据轨迹中心点与计数线的相对位置变化判断穿越事件。")
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    p.add_run().add_picture(str(ARCH_PATH), width=Inches(6.35))
    add_caption(doc, "图 2-1 系统处理流程")
    add_heading(doc, "2.1 模块划分", 2)
    add_table(doc, ["模块", "职责", "主要文件"], [
        ("数据准备", "图片标注、XML 转换、训练/验证/测试划分", "label_tool.py、voc2yolo.py、split_dataset.py"),
        ("检测", "图像预处理、YOLOv5 推理、NMS", "main.py、yolov5/"),
        ("追踪", "卡尔曼滤波、外观特征、匹配与 Track ID", "tracker/"),
        ("计数", "轨迹中心缓存、穿线判断、ID 去重、分类累计", "main.py"),
        ("交互", "视频上传、进度反馈、结果回传", "webui.py"),
    ], [1500, 4560, 3300])
    add_heading(doc, "2.2 类别与数据一致性", 2)
    add_text(doc, "检测、标注转换和训练配置统一采用 person、car、motorcycle、bus、truck 的类别顺序。统一类别索引可避免 XML、TXT、YAML 与模型输出之间出现错位，这是训练链路正确性的关键。")

    doc.add_page_break()
    add_heading(doc, "3 数据标注与格式转换", 1)
    add_heading(doc, "3.1 标注工具", 2)
    add_text(doc, "label_tool.py 提供图像浏览、鼠标框选、类别选择和 Pascal VOC XML 保存能力。工具默认读取 data/mydata/images，并将 XML 输出到 data/mydata/xml。当前数据目录包含 9 张图片和 9 份 XML，可用于复现实验。")
    img = DATA_OVERVIEW_PATH
    if img.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(img), width=Inches(6.15))
        add_caption(doc, "图 3-1 数据文件、标注框数量与数据集划分")
    add_heading(doc, "3.2 XML 转 YOLO TXT", 2)
    add_text(doc, "voc2yolo.py 读取 XML 中的图像宽高、类别和边界框，将坐标转换为 YOLO 所需的中心点与宽高归一化格式：class_id x_center y_center width height。转换后共生成 9 份 TXT，包含 21 行标注，所有归一化坐标均处于 [0,1] 范围。")
    add_code(doc, "python label_tool.py\npython voc2yolo.py\npython split_dataset.py")
    add_heading(doc, "3.3 数据集划分", 2)
    add_text(doc, "split_dataset.py 生成 train.txt、val.txt 和 test.txt，当前划分为 6 张训练图、2 张验证图和 1 张测试图。该规模适合课程流程验证，但不足以支撑高可信度的精度评估。")

    add_heading(doc, "4 检测、追踪与计数实现", 1)
    add_heading(doc, "4.1 YOLOv5 检测", 2)
    add_text(doc, "main.py 负责加载模型、选择 CPU/GPU、批量预处理并执行推理。输入尺寸被规范为 [高, 宽]，确保不同调用入口与 YOLOv5 的 letterbox 逻辑一致。经过 NMS 后，仅保留人员及车辆相关类别，再交给 DeepSORT。")
    add_heading(doc, "4.2 DeepSORT 追踪", 2)
    add_text(doc, "DeepSORT 使用卡尔曼滤波预测运动状态，并用外观特征与 IOU 共同关联检测结果。主要参数包括 MAX_DIST=0.2、MAX_IOU_DISTANCE=0.7、MAX_AGE=30、N_INIT=3、NN_BUDGET=100。即使当前帧没有有效检测，程序也会推进跟踪器年龄，防止轨迹状态长期滞留。", align=WD_ALIGN_PARAGRAPH.LEFT)
    add_heading(doc, "4.3 穿线计数与去重", 2)
    add_text(doc, "计数线默认位于画面高度的 70%。程序保存每个 Track ID 上一帧的中心纵坐标，当中心点从计数线一侧移动到另一侧时判定穿线；counted_ids 集合保证同一 Track ID 只累计一次。person 类累加人员计数，car、motorcycle、bus、truck 累加车辆计数。")
    add_code(doc, "crossed_line = (previous_y is not None\n    and (previous_y - line_y) * (center_y - line_y) <= 0\n    and previous_y != center_y)\nif crossed_line and track_id not in counted_ids:\n    counted_ids.add(track_id)")
    add_status_callout(doc, "修复要点", "原逻辑容易按目标持续出现帧数累计，无法表达真实穿线事件；现版本改为轨迹位置变化判定，并用 Track ID 去重。", fill="FFF7E6", accent=GOLD)

    doc.add_page_break()
    add_heading(doc, "5 Gradio 前端与输出展示", 1)
    add_heading(doc, "5.1 前端流程", 2)
    add_text(doc, "webui.py 使用 Gradio Blocks 构建上传、处理和结果展示界面。上传视频后，前端读取帧率与分辨率，逐帧调用检测与追踪逻辑，更新进度条，并返回唯一命名的临时 MP4，避免多次任务互相覆盖。前端预处理已与命令行保持一致，包括 letterbox 和 BGR→RGB 转换。")
    add_code(doc, "python webui.py\n# 浏览器访问 http://127.0.0.1:7860")
    add_heading(doc, "5.2 可视化结果", 2)
    frame = ROOT / "submission_work" / "frame_samples" / "frame_264.jpg"
    if frame.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(str(frame), width=Inches(6.3))
        add_caption(doc, "图 5-1 实际运行截图：检测框、类别、Track ID 与穿线计数")
    add_text(doc, "该截图取自已验证的最终输出视频。画面左上角显示 Person 和 Vehicle 累计值，蓝色检测框标注车辆，同时显示类别与 Track ID；黄色横线表示计数触发位置。截图为处理中间状态 Vehicle=10，完整视频最终统计为 Vehicle=15。", color=MUTED)

    doc.add_page_break()
    add_heading(doc, "6 测试与验收结果", 1)
    add_heading(doc, "6.1 验证汇总", 2)
    add_table(doc, ["测试项", "实际结果", "结论"], [
        ("Python 语法检查", "核心 Python 文件 py_compile 通过", "通过"),
        ("标注工具初始化", "加载 9 张图片，首图可读取", "通过"),
        ("标签一致性", "9 图 / 9 XML / 9 TXT / 21 标注框", "通过"),
        ("命令行推理", "473 帧，1280×720，Person=0，Vehicle=15", "通过"),
        ("Gradio 处理链", "473 帧完整处理，Vehicle=15", "通过"),
        ("训练烟雾测试", "1 epoch，生成 best.pt（约 3.7 MB）", "通过"),
        ("独立验证", "2 images / 6 labels，mAP=0", "流程通过，精度不足"),
        ("报告版式", "LibreOffice 转 PDF 并逐页检查", "通过"),
    ], [1900, 5260, 2200], aligns=[WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER])
    add_heading(doc, "6.2 结果解释", 2)
    add_text(doc, "示例视频中没有人员穿越计数线，因此 Person=0 与场景内容一致。Vehicle=15 为完整 473 帧视频的穿线去重结果。训练部分使用 9 张图片、1 个 epoch 的最小配置，目的仅是验证数据加载、反向传播、权重保存和验证入口能够工作。验证集 mAP=0 说明该小样本权重尚不具备实际部署精度，正式应用需扩大样本规模、补充多类别和复杂天气场景，并进行多轮训练与参数调优。")
    add_heading(doc, "6.3 训练记录", 2)
    add_status_callout(doc, "单轮记录", "train/box_loss=0.11195，train/obj_loss=0.010207，train/cls_loss=0.038132；验证 mAP=0。该记录用于证明训练流程完成，不用于宣称模型精度。", fill="F4F6F9", accent=SKY)

    doc.add_page_break()
    add_heading(doc, "7 问题修复与工程完善", 1)
    add_table(doc, ["问题", "处理方式", "效果"], [
        ("计数逻辑不是真实穿线", "保存轨迹中心并比较计数线两侧位置", "按事件计数"),
        ("同一目标可能重复统计", "使用 counted_ids 对 Track ID 去重", "每条轨迹仅一次"),
        ("Gradio 预处理与 CLI 不一致", "统一 letterbox 与 BGR→RGB", "两入口结果一致"),
        ("空检测帧轨迹不老化", "无有效框时仍推进 DeepSORT", "轨迹生命周期正常"),
        ("类别顺序存在错位风险", "XML、TXT、YAML 统一五类顺序", "训练数据一致"),
        ("Pillow 新版本兼容性", "getsize 改为 getbbox", "训练可运行"),
        ("验证标签统计错误", "修正无真阳性时的标签累计逻辑", "正确显示 6 labels"),
        ("训练默认路径失效", "改为工程相对路径", "默认入口可用"),
    ], [2300, 4560, 2500])
    add_heading(doc, "7.1 工程可复现性", 2)
    add_text(doc, "提交包保留 requirements.txt、README.md、配置文件、示例输入与输出、必要检测模型和 DeepSORT 外观模型。用户无需依赖原始课程压缩包即可运行主要流程。为控制体积，未包含 venv、全部 YOLOv5 模型族和历史测试 runs；如需其他模型，可按 README 替换权重路径。")
    add_heading(doc, "7.2 已知限制", 2)
    add_text(doc, "当前计数线为固定水平线，适合固定机位和近似垂直运动方向；相机抖动、严重遮挡、目标在计数线附近往返、Track ID 切换都可能影响计数。当前自建数据几乎只有 car 类，人员、摩托车、公交车和卡车的专项训练样本不足。后续可增加方向判定、计数区域、多线统计、轨迹重识别恢复和数据库记录。")

    doc.add_page_break()
    add_heading(doc, "8 使用说明", 1)
    add_heading(doc, "8.1 安装依赖", 2)
    add_code(doc, "cd 项目工程\npip install -r requirements.txt")
    add_heading(doc, "8.2 命令行推理", 2)
    add_code(doc, "python main.py --source example.mp4 --yolo_model models/yolov5m.pt ^\n  --project runs/track --name result --exist-ok")
    add_text(doc, "处理后视频位于 runs/track/result/。提交包同时提供已验证结果 results/processed_example.mp4，可直接用于验收。", align=WD_ALIGN_PARAGRAPH.LEFT)
    add_heading(doc, "8.3 数据处理与训练", 2)
    add_code(doc, "python label_tool.py\npython voc2yolo.py\npython split_dataset.py\npython yolov5/train.py")
    add_heading(doc, "8.4 Gradio 前端", 2)
    add_code(doc, "python webui.py")
    add_text(doc, "启动后访问 http://127.0.0.1:7860，上传视频并点击开始处理。CPU 环境处理时间较长，建议保持终端窗口运行；GPU 环境可通过参数选择 CUDA 设备。")

    add_heading(doc, "9 总结", 1)
    add_text(doc, "本项目已完成课程要求的完整链路：数据标注、标签转换、数据划分、模型训练、视频检测、DeepSORT 追踪、人员/车辆分项穿线计数、结果视频生成以及 Gradio 前端展示。关键工程问题已修复，并通过完整示例视频和最小训练任务验证。最终压缩包具有清晰目录、独立运行说明和可直接查看的处理结果，可作为课程项目成果提交。")
    add_status_callout(doc, "提交前提醒", "请在封面补充姓名、学号、班级/专业和指导教师；如课程平台限制单文件大小，可在不影响验收的前提下移除 models/yolov5m.pt，并在 README 中注明权重获取方式。", fill="FFF7E6", accent=GOLD)

    doc.core_properties.title = "高速车流量计数器项目报告"
    doc.core_properties.subject = "YOLOv5 + DeepSORT 交通目标检测、追踪与计数"
    doc.core_properties.author = "课程项目提交者"
    doc.core_properties.keywords = "YOLOv5, DeepSORT, Gradio, 车流量计数"
    doc.core_properties.comments = "由项目实际代码与验证结果整理。"
    doc.save(REPORT_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    build_report()
