# -*- coding: utf-8 -*-
"""Regenerate docs/images/winrate/*.png with model-name + percentage labels.

Source: the unlabeled original figures in docs/images/*.png.
Each bar is split into three colored segments (yellow / teal / blue).
We draw the model name and the specific win-rate value inside each segment.
"""
import os
from PIL import Image, ImageDraw, ImageFont

SRC = "docs/images"
DST = "docs/images/winrate"
FONT = r"C:\Windows\Fonts\arialbd.ttf"
FONT_FALLBACK = r"C:\Windows\Fonts\arial.ttf"

# (model name, value) per segment, in left-to-right order = PaperGym-20k / RubricHub / Base
DATA = {
    "overall":                 ("PaperGym-20k", "RubricHub", "Base", "58.1%", "28.2%", "13.7%"),
    "goal_alignment":          ("PaperGym-20k", "RubricHub", "Base", "55.7%", "29.0%", "15.3%"),
    "novel_insight":           ("PaperGym-20k", "RubricHub", "Base", "58.4%", "29.3%", "12.3%"),
    "scientific_soundness":    ("PaperGym-20k", "RubricHub", "Base", "57.3%", "29.0%", "13.7%"),
    "execution_quality":       ("PaperGym-20k", "RubricHub", "Base", "54.6%", "31.2%", "14.2%"),
    "expected_impact":         ("PaperGym-20k", "RubricHub", "Base", "57.1%", "29.4%", "13.5%"),
}


def load_font(size):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.truetype(FONT_FALLBACK, size)


def find_bar_band(px, w, h):
    """Rows whose colored coverage >= 0.5 form the solid bar band."""
    ys = []
    for y in range(h):
        strong = 0
        for x in range(0, w, 4):
            r, g, b = px[x, y]
            if max(r, g, b) - min(r, g, b) > 40:
                strong += 1
        if strong / ((w + 3) // 4) >= 0.5:
            ys.append(y)
    return min(ys), max(ys)


def find_segments(px, w, y):
    """Colored runs along one row -> list of (x0, x1)."""
    runs, cur, start = [], None, 0
    for x in range(w):
        r, g, b = px[x, y]
        col = (r, g, b) if max(r, g, b) - min(r, g, b) > 40 else None
        if col != cur:
            if cur is not None:
                runs.append((start, x))
            start = x
            cur = col
    if cur is not None:
        runs.append((start, w))
    # keep only runs wider than 30px (drop thin anti-aliasing strips)
    return [r for r in runs if r[1] - r[0] > 30]


def main():
    os.makedirs(DST, exist_ok=True)
    for name, (n1, n2, n3, v1, v2, v3) in DATA.items():
        im = Image.open(os.path.join(SRC, name + ".png")).convert("RGB")
        w, h = im.size
        px = im.load()
        top, bot = find_bar_band(px, w, h)
        mid = (top + bot) // 2
        segs = find_segments(px, w, mid)
        assert len(segs) == 3, f"{name}: expected 3 segments, got {segs}"

        # crop with a small vertical margin
        y0, y1 = max(0, top - 5), min(h, bot + 5)
        band = im.crop((0, y0, w, y1)).copy()
        d = ImageDraw.Draw(band)

        names = (n1, n2, n3)
        vals = (v1, v2, v3)
        name_font = load_font(27)
        val_font = load_font(38)
        for (x0, x1), nm, val in zip(segs, names, vals):
            cx = (x0 + x1) / 2
            # dark text on yellow/teal, white on blue
            r, g, b = px[cx, mid]
            text_col = (250, 250, 250) if b > r and b > g else (26, 26, 26)
            bar_top = top - y0        # bar band top within the cropped canvas
            bar_h = bot - top + 1     # bar band height
            # name line then value line, both vertically centered within the bar
            d.text((cx, bar_top + bar_h * 0.34), nm,
                   font=name_font, fill=text_col, anchor="mm")
            d.text((cx, bar_top + bar_h * 0.70), val,
                   font=val_font, fill=text_col, anchor="mm")

        out = os.path.join(DST, name + ".png")
        band.save(out)
        print(f"{name}: segs={[(a, b) for a, b in segs]} yband=({top}-{bot}) -> {out} ({band.size})")


if __name__ == "__main__":
    main()
