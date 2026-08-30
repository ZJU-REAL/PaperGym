# -*- coding: utf-8 -*-
"""Vertical bar chart of criterion leakage, light/pastel palette. Wide > tall, large fonts."""
from PIL import Image, ImageDraw, ImageFont

FONT = r"C:\Windows\Fonts\arial.ttf"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"

# (full name, label lines, leakage %, is_papergym)  — multi-line so labels never overlap
DATA = [
    ("PaperGym-20k",        ("PaperGym-20k",),             3.73, True),
    ("PaperGym-Innov",      ("PaperGym-", "Innov"),        4.71, True),
    ("PaperGym-Design",     ("PaperGym-", "Design"),       4.97, True),
    ("HealthBench",         ("HealthBench",),             11.90, False),
    ("RubricHub Science",   ("RubricHub", "Science"),     17.39, False),
    ("ResearchQA",          ("ResearchQA",),              19.22, False),
    ("ResearchPlanGen-ML",  ("ResearchPlan", "Gen-ML"),   31.29, False),
    ("ResearchPlanGen-ArXiv", ("ResearchPlan", "Gen-ArXiv"), 34.10, False),
]

PAPERG_FILL = (168, 213, 213)   # light teal
EXT_FILL = (226, 231, 235)      # light gray
TEXT = (60, 60, 66)
GRID = (225, 228, 231)
AXIS = (175, 180, 186)
MAX_V = 36.0

W, H = 1440, 600
PLOT_X0, PLOT_X1 = 100, 1410
BASELINE = 500
TOP = 72
SLOT_W = (PLOT_X1 - PLOT_X0) / len(DATA)
BAR_W = 88
Y_SCALE = (BASELINE - TOP) / MAX_V   # px per percent
N = len(DATA)


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def main():
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)

    d.text((PLOT_X0, 26), "Criterion leakage (%)", font=font(40, True), fill=TEXT)

    # horizontal grid + y ticks
    for v in (0, 10, 20, 30):
        y = BASELINE - v * Y_SCALE
        d.line([(PLOT_X0, y), (PLOT_X1, y)], fill=GRID, width=2)
        d.text((PLOT_X0 - 12, y), str(v), font=font(22), fill=(140, 140, 145), anchor="rm")

    # bars
    for i, (name, lines, val, is_pg) in enumerate(DATA):
        cx = PLOT_X0 + (i + 0.5) * SLOT_W
        bh = val * Y_SCALE
        x0, x1 = cx - BAR_W / 2, cx + BAR_W / 2
        y0 = BASELINE - bh
        fill = PAPERG_FILL if is_pg else EXT_FILL
        d.rounded_rectangle([x0, y0, x1, BASELINE], radius=10, fill=fill,
                            outline=None if is_pg else (214, 219, 224), width=2)
        # value label above the bar, with clear clearance from the bar top
        d.text((cx, y0 - 26), f"{val:.2f}%", font=font(30, True), fill=(80, 80, 86), anchor="mm")
        # two-line category label under the axis
        ly = BASELINE + 12
        for ln in lines:
            d.text((cx, ly), ln, font=font(22), fill=(80, 80, 86), anchor="ma")
            ly += 28

    # baseline
    d.line([(PLOT_X0, BASELINE), (PLOT_X1, BASELINE)], fill=AXIS, width=3)

    out = r"E:\PaperGym\docs\images\leakage.png"
    im.save(out)
    print("saved", out, im.size)


if __name__ == "__main__":
    main()
