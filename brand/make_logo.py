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

def option1(bg=True):
    img = background() if bg else Image.new("RGBA", (N, N), (0, 0, 0, 0))
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


# ---------------------------------------------------------------- channel banner (matches option 1)

BW, BH = 2560, 1440  # YouTube banner; the part visible on every device is the centre 1546 x 423
SAFE_W, SAFE_H = 1546, 423
TAGLINE = "Complex topics. Made simple."
CHIPS = ["AI & ML", "Programming", "Tech explained"]


def _grad(size, mask, c1=CYAN, c2=PURPLE):
    w, h = size
    t = np.clip(np.linspace(0, 1, w, dtype=np.float32), 0, 1)[None, :, None]
    rgb = np.array(c1, np.float32) * (1 - t) + np.array(c2, np.float32) * t
    rgb = np.repeat(rgb, h, axis=0)
    a = np.asarray(mask, np.float32)[..., None]
    return Image.fromarray(np.concatenate([rgb, a], -1).astype(np.uint8), "RGBA")


def banner():
    yy, xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
    r = np.clip(np.sqrt(((xx - BW / 2) / (BW * 0.55)) ** 2 + ((yy - BH / 2) / (BH * 0.6)) ** 2), 0, 1)[..., None]
    arr = np.array(NAVY_IN, np.float32) * (1 - r) + np.array(NAVY_OUT, np.float32) * r
    img = Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")
    grid = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    d = ImageDraw.Draw(grid)
    for x in range(0, BW, 64):  # faint grid, like the Shorts
        d.line([(x, 0), (x, BH)], fill=(255, 255, 255, 10))
    for y in range(0, BH, 64):
        d.line([(0, y), (BW, y)], fill=(255, 255, 255, 10))
    img.alpha_composite(grid)
    # Soft neon glows at the edges (only seen on wide screens), plus a few idea sparks.
    glow = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse((-300, -200, 700, 700), fill=CYAN + (55,))
    gd.ellipse((BW - 700, BH - 700, BW + 300, BH + 200), fill=PURPLE + (65,))
    img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(160)))
    rng = np.random.default_rng(7)
    sparks = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    sd = ImageDraw.Draw(sparks)
    for _ in range(70):
        x, y = rng.uniform(0, BW), rng.uniform(0, BH)
        if abs(x - BW / 2) < SAFE_W / 2 + 40 and abs(y - BH / 2) < SAFE_H / 2 + 60:
            continue  # keep the centre calm
        rr = rng.uniform(2, 6)
        col = [CYAN, PURPLE, YELLOW][int(rng.integers(3))]
        sd.ellipse((x - rr, y - rr, x + rr, y + rr), fill=col + (int(rng.uniform(90, 200)),))
    img.alpha_composite(add_glow(sparks, radius=6, strength=1.3))

    # The Si mark (same drawing as the profile picture, without its background).
    mark = option1(bg=False).resize((420, 420), Image.LANCZOS)
    left = (BW - SAFE_W) // 2 + 40
    img.alpha_composite(mark, (left, BH // 2 - 210 - 6))

    # Wordmark: "smart" + "i" + "learner", where the dot of the i is the idea bulb.
    fs = 150
    fw = font("Black", fs)
    parts = ["smart", "\u0131", "learner"]  # dotless i, the bulb becomes its dot
    widths = [fw.getlength(p_) for p_ in parts]
    x0 = left + 440
    base_y = BH // 2 - 30
    m = Image.new("L", (BW, BH), 0)
    md = ImageDraw.Draw(m)
    x = x0
    for p_, w_ in zip(parts, widths):
        md.text((x, base_y), p_, font=fw, fill=255, anchor="ls")
        x += w_
    word = _grad((BW, BH), m, WHITE, (205, 212, 255))
    img.alpha_composite(add_glow(word, radius=10, strength=0.9))
    i_x = x0 + widths[0] + widths[1] / 2
    bl = Image.new("RGBA", (BW, BH), (0, 0, 0, 0))
    # Reuse the bulb at banner scale: draw it on a temporary N x N canvas and paste.
    tmp = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    bulb(tmp, C, C, N * 0.075)
    tmp = tmp.resize((int(N * 0.16), int(N * 0.16)), Image.LANCZOS)
    bl.alpha_composite(tmp, (int(i_x - tmp.width / 2), int(base_y - fs * 0.67 - tmp.height / 2)))
    img.alpha_composite(bl)
    # Tagline and topic chips (topic-neutral wording).
    d = ImageDraw.Draw(img)
    d.text((x0 + 6, base_y + 70), TAGLINE, font=font("Bold", 64), fill=(170, 180, 225), anchor="lm")
    cx = x0 + 6
    cy = base_y + 160
    for k, chip in enumerate(CHIPS):
        f = font("Bold", 40)
        tw = f.getlength(chip)
        col = [CYAN, PURPLE, YELLOW][k]
        d.rounded_rectangle((cx, cy - 34, cx + tw + 56, cy + 34), radius=34, fill=(28, 26, 64, 255), outline=col, width=4)
        d.text((cx + 28 + tw / 2, cy), chip, font=f, fill=WHITE, anchor="mm")
        cx += tw + 56 + 22
    right = max(x, cx)
    return img.convert("RGB"), (left, right)


def banner_preview(img, out):
    """The banner with YouTube's device crop guides (for checking only)."""
    pv = img.copy().convert("RGBA")
    d = ImageDraw.Draw(pv)
    boxes = [((BW - 2560) / 2, (BH - 1440) / 2, 2560, 1440, "TV"), ((BW - 2560) / 2, (BH - 423) / 2, 2560, 423, "desktop max"),
             ((BW - 1546) / 2, (BH - 423) / 2, 1546, 423, "safe area (all devices)")]
    for x, y, w, h, lab in boxes:
        d.rectangle((x + 3, y + 3, x + w - 3, y + h - 3), outline=(255, 80, 80), width=6)
        d.text((x + 16, y + 10), lab, font=font("Bold", 36), fill=(255, 120, 120))
    pv.convert("RGB").resize((BW // 2, BH // 2), Image.LANCZOS).save(out)


def main():
    paths = [finish(fn(), os.path.join(HERE, f"profile_option{k}.png"))
             for k, fn in enumerate((option1, option2, option3), 1)]
    preview(paths, os.path.join(HERE, "profile_preview.png"))
    # The chosen design, under clear names for download.
    Image.open(paths[0]).save(os.path.join(HERE, "smartilearner_profile.png"), optimize=True)
    bn, (left, right) = banner()
    assert right <= (BW + SAFE_W) / 2 - 20, f"banner content runs past the safe area ({right})"
    bn.save(os.path.join(HERE, "smartilearner_banner.png"), optimize=True)
    banner_preview(bn, os.path.join(HERE, "banner_safe_area_check.png"))
    for p in paths + ["smartilearner_profile.png", "smartilearner_banner.png"]:
        p = os.path.join(HERE, p)
        print(p, os.path.getsize(p) // 1024, "KB")


if __name__ == "__main__":
    main()
