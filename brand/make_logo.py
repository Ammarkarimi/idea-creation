"""Profile-picture options for the "smartilearner" YouTube channel.

    python3 make_logo.py      # writes profile_option1..3.png (800x800) + profile_preview.png

Idea: smart-i-learner. The dot of the "i" is a glowing idea bulb. Topic-neutral, so it works for any subject.
"""

import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "shorts"))
from common import CYAN, PURPLE, WHITE, YELLOW, add_glow, font  # noqa: E402

SS = 2  # supersampling
SIZE = 800
N = SIZE * SS
C = N // 2
NAVY_IN, NAVY_OUT = (26, 22, 60), (8, 9, 24)
WARM = (255, 200, 60)


def background():
    yy, xx = np.mgrid[0:N, 0:N]
    r = np.sqrt((xx - C) ** 2 + (yy - C * 0.9) ** 2) / (N * 0.62)
    r = np.clip(r, 0, 1)[..., None]
    arr = np.array(NAVY_IN) * (1 - r) + np.array(NAVY_OUT) * r
    img = Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")
    # A thin gradient ring just inside YouTube's circle crop.
    ring = Image.new("L", (N, N), 0)
    ImageDraw.Draw(ring).ellipse((N * 0.045, N * 0.045, N * 0.955, N * 0.955), outline=255, width=int(N * 0.012))
    img.alpha_composite(gradient_fill(ring, alpha=0.9))
    return img


def gradient_fill(mask, c1=CYAN, c2=PURPLE, alpha=1.0):
    """Fill a mask with a diagonal cyan -> purple gradient."""
    yy, xx = np.mgrid[0:N, 0:N]
    t = np.clip((xx + yy) / (2 * N) * 1.3 - 0.15, 0, 1)[..., None]
    rgb = np.array(c1) * (1 - t) + np.array(c2) * t
    a = (np.asarray(mask).astype(np.float32) * alpha)[..., None]
    return Image.fromarray(np.concatenate([rgb, a], -1).astype(np.uint8), "RGBA")


def text_mask(text, size, center, weight="Black"):
    m = Image.new("L", (N, N), 0)
    ImageDraw.Draw(m).text(center, text, font=font(weight, size), fill=255, anchor="mm")
    return m


def glow_layer(layer, radius, strength=1.4):
    return add_glow(layer, radius=radius, strength=strength)


def bulb(img, cx, cy, r, rays=True):
    """The idea spark: a glowing bulb with a little screw base and rays."""
    lay = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    # Soft halo.
    halo = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(halo).ellipse((cx - 2.2 * r, cy - 2.2 * r, cx + 2.2 * r, cy + 2.2 * r), fill=WARM + (120,))
    img.alpha_composite(halo.filter(ImageFilter.GaussianBlur(r * 0.9)))
    if rays:
        for k in range(8):
            a = -math.pi / 2 + k * math.pi / 4
            if k in (3, 4, 5):  # no rays pointing down into the letter
                continue
            r0, r1 = r * 1.45, r * 1.95
            d.line([(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + r1 * math.cos(a), cy + r1 * math.sin(a))],
                   fill=YELLOW + (255,), width=int(r * 0.22))
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=YELLOW + (255,))
    d.ellipse((cx - r * 0.55, cy - r * 0.6, cx - r * 0.05, cy - r * 0.1), fill=(255, 250, 220, 230))  # shine
    img.alpha_composite(glow_layer(lay, int(r * 0.5), 1.2))


def finish(img, path):
    img.convert("RGB").resize((SIZE, SIZE), Image.LANCZOS).save(path, optimize=True)
    return path


# ---------------------------------------------------------------- option 1: "Si"

def option1():
    img = background()
    s = text_mask("S", int(N * 0.62), (C - N * 0.085, C + N * 0.03))
    img.alpha_composite(glow_layer(gradient_fill(s), int(N * 0.02)))
    # The "i": a rounded stem and a bulb for its dot.
    stem = Image.new("L", (N, N), 0)
    x = C + N * 0.235
    w = N * 0.095
    ImageDraw.Draw(stem).rounded_rectangle((x - w / 2, C - N * 0.04, x + w / 2, C + N * 0.235), radius=w / 2, fill=255)
    img.alpha_composite(glow_layer(gradient_fill(stem, WHITE, (200, 210, 255)), int(N * 0.015), 1.0))
    bulb(img, x, C - N * 0.165, N * 0.068)
    return img


# ---------------------------------------------------------------- option 2: "SL" (the L's stem + bulb = an "i")

def option2():
    img = background()
    s = text_mask("S", int(N * 0.56), (C - N * 0.155, C + N * 0.04))
    img.alpha_composite(glow_layer(gradient_fill(s), int(N * 0.02)))
    lm = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(lm)
    x0, w = C + N * 0.095, N * 0.09
    top, bot = C - N * 0.06, C + N * 0.235
    d.rounded_rectangle((x0, top, x0 + w, bot), radius=w / 2, fill=255)
    d.rounded_rectangle((x0, bot - w, x0 + N * 0.25, bot), radius=w / 2, fill=255)
    img.alpha_composite(glow_layer(gradient_fill(lm, WHITE, (200, 210, 255)), int(N * 0.015), 1.0))
    bulb(img, x0 + w / 2, C - N * 0.18, N * 0.066)
    return img


# ---------------------------------------------------------------- option 3: lightbulb with an "S" filament

def option3():
    img = background()
    cx, cy, r = C, C - N * 0.07, N * 0.235
    glass = Image.new("L", (N, N), 0)
    d = ImageDraw.Draw(glass)
    sw = int(N * 0.03)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=255, width=sw)
    # Neck and screw base.
    nw, ny = r * 0.52, cy + r * 0.86
    d.rectangle((cx - nw, ny, cx + nw, ny + N * 0.04), fill=0)
    d.line([(cx - nw, ny - N * 0.01), (cx - nw * 0.8, ny + N * 0.09)], fill=255, width=sw)
    d.line([(cx + nw, ny - N * 0.01), (cx + nw * 0.8, ny + N * 0.09)], fill=255, width=sw)
    for k in range(3):
        y = ny + N * 0.115 + k * N * 0.045
        ww = nw * (0.78 - 0.08 * k)
        d.rounded_rectangle((cx - ww, y - N * 0.014, cx + ww, y + N * 0.014), radius=N * 0.014, fill=255)
    img.alpha_composite(glow_layer(gradient_fill(glass), int(N * 0.018)))
    # Warm inner glow and the S filament.
    halo = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    ImageDraw.Draw(halo).ellipse((cx - r * 0.8, cy - r * 0.8, cx + r * 0.8, cy + r * 0.8), fill=WARM + (90,))
    img.alpha_composite(halo.filter(ImageFilter.GaussianBlur(r * 0.35)))
    s = text_mask("S", int(r * 1.45), (cx, cy + r * 0.02))
    fil = Image.new("RGBA", (N, N), YELLOW + (0,))
    fil.putalpha(s)
    img.alpha_composite(glow_layer(fil, int(N * 0.02), 1.6))
    return img


# ---------------------------------------------------------------- preview sheet

def circle(img, size):
    im = img.resize((size, size), Image.LANCZOS).convert("RGBA")
    m = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(m).ellipse((0, 0, size * 4 - 1, size * 4 - 1), fill=255)
    im.putalpha(m.resize((size, size), Image.LANCZOS))
    return im


def preview(paths, out):
    imgs = [Image.open(p) for p in paths]
    cell_w, row_h = 760, 420
    sheet = Image.new("RGB", (cell_w * len(imgs), row_h * 2), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    for row, bg in enumerate([(255, 255, 255), (15, 15, 15)]):
        d.rectangle((0, row * row_h, sheet.width, (row + 1) * row_h), fill=bg)
        fg = (20, 20, 20) if row == 0 else (235, 235, 235)
        for k, im in enumerate(imgs):
            x0, y0 = k * cell_w + 30, row * row_h + 30
            sheet.paste(c := circle(im, 300), (x0, y0), c)
            sheet.paste(c := circle(im, 98), (x0 + 340, y0 + 40), c)
            sheet.paste(c := circle(im, 40), (x0 + 470, y0 + 69), c)
            d.text((x0 + 340, y0 + 160), "smartilearner", font=font("Bold", 34, display=False), fill=fg)
            d.text((x0 + 340, y0 + 205), "98 px / 40 px", font=font("Regular", 26, display=False), fill=(120, 120, 120))
            d.text((x0, y0 + 320), f"Option {k + 1}", font=font("Black", 40), fill=fg)
    sheet.save(out)


def main():
    paths = [finish(fn(), os.path.join(HERE, f"profile_option{k}.png"))
             for k, fn in enumerate((option1, option2, option3), 1)]
    preview(paths, os.path.join(HERE, "profile_preview.png"))
    for p in paths:
        print(p, os.path.getsize(p) // 1024, "KB")


if __name__ == "__main__":
    main()
