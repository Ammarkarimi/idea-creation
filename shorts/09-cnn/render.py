"""ML as a Game, Level 9: How AI Sees Images - CNNs (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> cnn_short.mp4 + thumbnail.png
    python3 render.py --preview 3,12  # just dump PNG frames at those seconds (fast check)
    python3 render.py --thumbnail     # only regenerate thumbnail.png
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import (  # noqa: E402
    CYAN, FPS, GREEN, H, MUTED, ORANGE, PINK, PURPLE, SR, W, WHITE, YELLOW, Timeline, build_confetti, build_plain,
    build_title, caption_chunks, clamp, draw_caption, draw_hud_pills, draw_level_cleared, ease_back,
    encode, font, lerp_color, mix, music_bed, pop_in, sfx_blip, sfx_boing, sfx_levelup, sfx_whoosh,
    tag, text_center, voiceover, write_wav,
)

# ---------------------------------------------------------------- the picture, the filters, the convolution

SIZE, XS, K = 8, 5, 3  # 8x8 image, 5x5 X, 3x3 filters
START, END = (0, 0), (3, 2)  # where the X sits before / after the shift


def picture(r0, c0):
    im = np.zeros((SIZE, SIZE))
    for k in range(XS):
        im[r0 + k, c0 + k] = 1
        im[r0 + k, c0 + XS - 1 - k] = 1
    return im


F_BACK = -np.ones((K, K))
np.fill_diagonal(F_BACK, 1)  # "\" : +1 on the diagonal, -1 elsewhere
F_FWD = np.fliplr(F_BACK)  # "/"
FILTERS = [F_BACK, F_FWD]
NAMES = ["\\", "/"]
M = SIZE - K + 1  # 6x6 feature maps


def conv(im, f):
    return np.array([[float((im[i:i + K, j:j + K] * f).sum()) for j in range(M)] for i in range(M)])


IMG0 = picture(*START)
MAPS0 = [conv(IMG0, f) for f in FILTERS]
IMG1 = picture(*END)
MAPS1 = [conv(IMG1, f) for f in FILTERS]
PEAKS0 = [sorted(map(tuple, np.argwhere(m == m.max()))) for m in MAPS0]
PEAKS1 = [sorted(map(tuple, np.argwhere(m == m.max()))) for m in MAPS1]
assert MAPS0[0][0, 0] == 3 and MAPS0[0][2, 2] == 3 and MAPS0[0][1, 1] == 1  # perfect match 3, X centre 1
assert all(m.max() == 3 for m in MAPS0 + MAPS1)
dr, dc = END[0] - START[0], END[1] - START[1]
assert all([(r + dr, c + dc) for r, c in p0] == p1 for p0, p1 in zip(PEAKS0, PEAKS1))  # peaks move with the X

SCRIPT = [
    "To a computer, a picture is just a grid of numbers. So how does AI see?",
    "Every pixel is a number: zero for dark, one for bright. Here's a tiny picture of an X.",
    "A convolution uses a tiny filter: a three by three grid of weights that looks for one pattern, "
    "like a diagonal line.",
    "Slide it over the picture. At each spot, multiply and add, just like a neuron.",
    "Big number? Pattern found! Those spots light up the feature map.",
    "A second filter hunts the other diagonal. Both light up? That's an X.",
    "And since the same filter slides everywhere, it finds the X wherever it is.",
    "Real networks learn their filters with backprop, and stack layers: edges become shapes, shapes become faces.",
    "That's a convolutional neural network. It's how AI spots cats, faces, and stop signs.",
    "Level nine cleared. Next: recurrent networks, AI with memory. Follow for level ten.",
]
HOOK, PIXELS, FILTER, SLIDE, FOUND, SECOND, SHIFT, STACK, CNN, CTA = range(10)

# ---------------------------------------------------------------- layout

IMG_X, IMG_Y, CELL = 40, 470, 66  # image grid: 8 x 66 = 528 px
FIL_X, FIL_Y, FCELL = 630, 516, 86  # filter grid: 3 x 86 = 258 px
MAP_Y, MCELL = 1068, 54  # feature maps: 6 x 54 = 324 px
MAP_XS = [140, 600]
ARITH = (612, 800, 1000, 1000)
CAPTION_Y = 1495
TOP_TAG_Y = 425

T: Timeline = None
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0


def word(i, prefix, nth=0):
    return T.word(i, prefix, nth)


# ---------------------------------------------------------------- state

def x_offset(t):
    """Where the X sits (row, col) - steps one cell at a time during the shift."""
    t0 = word(SHIFT, "same")
    steps = [(START[0] + min(k, dr), START[1] + min(k, dc)) for k in range(max(dr, dc) + 1)]
    if t < t0:
        return steps[0]
    k = int(clamp((t - t0) / 1.6) * (len(steps) - 1) + 1e-6)
    return steps[k]


def slide_state(t):
    """(filter index, number of positions visited, current position) or None while not sliding."""
    t0, t1 = word(SLIDE, "slide"), word(FOUND, "big") - 0.1
    if t0 <= t < t1:
        u = clamp((t - t0) / (t1 - t0))
        # Slow on the first two spots (so the arithmetic is readable), then quick.
        p = u / 0.4 * 2 if u < 0.4 else 2 + (u - 0.4) / 0.6 * (M * M - 2)
        n = min(M * M, int(p) + 1)
        return 0, n, divmod(n - 1, M)
    t2, t3 = word(SECOND, "second"), word(SECOND, "both") - 0.1
    if t2 <= t < t3:
        n = min(M * M, int(clamp((t - t2) / (t3 - t2)) * M * M) + 1)
        return 1, n, divmod(n - 1, M)
    return None


def visible_cells(fi, t):
    """How many cells of feature map fi are filled in at time t."""
    s = slide_state(t)
    if s and s[0] == fi:
        return s[1]
    start = word(SLIDE, "slide") if fi == 0 else word(SECOND, "second")
    return M * M if t >= start else 0


def current_image(t):
    return picture(*x_offset(t))


def current_maps(t):
    im = current_image(t)
    return [conv(im, f) for f in FILTERS]


def active_filter(t):
    return 1 if t >= word(SECOND, "second") and t < T.start(STACK) else 0


# ---------------------------------------------------------------- drawing

def draw_image(frame, t, d):
    im = current_image(t)
    show_nums = t >= word(PIXELS, "number")
    glow_x = word(PIXELS, "x.") <= t < T.start(FILTER) + 0.5
    for r in range(SIZE):
        for c in range(SIZE):
            x0, y0 = IMG_X + c * CELL, IMG_Y + r * CELL
            v = im[r, c]
            if t < T.start(PIXELS):  # hook: noisy grid
                v = ((r * 7 + c * 13 + int(t * 6)) % 5) / 4
            bright = lerp_color((22, 26, 54), (236, 240, 255), v)
            if glow_x and im[r, c] == 1:
                bright = lerp_color(bright, YELLOW, 0.35 + 0.25 * math.sin(t * 8))
            d.rectangle([x0 + 2, y0 + 2, x0 + CELL - 2, y0 + CELL - 2], fill=bright)
            if show_nums or t < T.start(PIXELS):
                txt = f"{int(round(v))}"
                text_center(d, (x0 + CELL / 2, y0 + CELL / 2), txt, font("Black", 32),
                            (20, 22, 44) if v > 0.5 else (88, 96, 140))
    d.rectangle([IMG_X - 3, IMG_Y - 3, IMG_X + SIZE * CELL + 3, IMG_Y + SIZE * CELL + 3], outline=(80, 88, 140),
                width=3)

    # The sliding window.
    s = slide_state(t)
    t_win = word(FILTER, "tiny")
    if s or (t_win <= t < word(SLIDE, "slide")):
        r, c = s[2] if s else (0, 0)
        x0, y0 = IMG_X + c * CELL, IMG_Y + r * CELL
        win = Image.new("RGBA", (K * CELL + 12, K * CELL + 12), (0, 0, 0, 0))
        wd = ImageDraw.Draw(win)
        wd.rectangle([0, 0, win.width - 1, win.height - 1], fill=YELLOW + (40,), outline=YELLOW + (255,), width=7)
        frame.alpha_composite(win, (x0 - 6, y0 - 6))


def draw_filter(frame, t, d):
    t0 = word(FILTER, "tiny")
    if t < t0 or t >= T.start(STACK):
        return
    fi = active_filter(t)
    f = FILTERS[fi]
    g = ease_back((t - (t0 if fi == 0 else word(SECOND, "second"))) / 0.4)
    label = f"FILTER  {NAMES[fi]}  diagonal"
    text_center(d, (FIL_X + K * FCELL / 2, FIL_Y - 34), label, font("Black", 32), YELLOW if fi == 0 else ORANGE)
    for r in range(K):
        for c in range(K):
            w = f[r, c]
            cx, cy = FIL_X + c * FCELL + FCELL / 2, FIL_Y + r * FCELL + FCELL / 2
            half = (FCELL / 2 - 4) * min(1, g)
            col = GREEN if w > 0 else PINK
            d.rounded_rectangle([cx - half, cy - half, cx + half, cy + half], radius=10, fill=col)
            if g > 0.6:
                text_center(d, (cx, cy), "+1" if w > 0 else "−1", font("Black", 36), (12, 28, 18) if w > 0 else WHITE)


def draw_arith(frame, t, d):
    """The running 'multiply and add' for the window's current spot."""
    s = slide_state(t)
    if not s:
        return
    fi, _, (r, c) = s
    im = current_image(t)
    patch = im[r:r + K, c:c + K]
    on = int((patch * (FILTERS[fi] > 0)).sum())
    off = int((patch * (FILTERS[fi] < 0)).sum())
    x0, y0, x1, y1 = ARITH
    d.rounded_rectangle(ARITH, radius=22, fill=(16, 18, 42), outline=(70, 78, 120), width=3)
    text_center(d, (x0 + 22, y0 + 36), "multiply & add:", font("Bold", 28), MUTED, anchor="lm")
    text_center(d, (x0 + 22, y0 + 82), f"bright on the line  {on} × (+1)", font("Bold", 29), GREEN, anchor="lm")
    text_center(d, (x0 + 22, y0 + 122), f"bright off the line  {off} × (−1)", font("Bold", 29), PINK, anchor="lm")
    text_center(d, (x0 + 22, y0 + 168), f"SCORE = {on - off}", font("Black", 38), YELLOW, anchor="lm")


def draw_maps(frame, t, d):
    if t < word(SLIDE, "slide") or t >= T.start(STACK) + 0.4:
        return
    maps = current_maps(t)
    s = slide_state(t)
    t_found = word(FOUND, "found!")
    for fi in range(2):
        n = visible_cells(fi, t)
        if n == 0:
            continue
        mx = MAP_XS[fi]
        col = YELLOW if fi == 0 else ORANGE
        text_center(d, (mx + M * MCELL / 2, MAP_Y - 32), f"feature map  {NAMES[fi]}", font("Black", 30), col)
        peak = maps[fi].max()
        for k in range(M * M):
            r, c = divmod(k, M)
            x0, y0 = mx + c * MCELL, MAP_Y + r * MCELL
            if k >= n:
                d.rectangle([x0 + 1, y0 + 1, x0 + MCELL - 1, y0 + MCELL - 1], outline=(50, 56, 92), width=2)
                continue
            v = maps[fi][r, c]
            fill = lerp_color((26, 30, 60), col, clamp(v / 3)) if v > 0 else (14, 16, 36)
            flash = v == peak and t >= t_found and (t < T.start(STACK))
            if flash:
                fill = lerp_color(fill, WHITE, 0.3 * abs(math.sin(t * 7)))
            d.rectangle([x0 + 1, y0 + 1, x0 + MCELL - 1, y0 + MCELL - 1], fill=fill,
                        outline=WHITE if flash else None, width=4 if flash else 0)
            txt = f"{int(v)}" if v >= 0 else f"−{int(-v)}"
            text_center(d, (x0 + MCELL / 2, y0 + MCELL / 2), txt, font("Black", 28),
                        (20, 20, 30) if v >= 2 else (MUTED if v <= 0 else WHITE))
            if s and s[0] == fi and k == n - 1:
                d.rectangle([x0, y0, x0 + MCELL, y0 + MCELL], outline=WHITE, width=4)
        d.rectangle([mx - 3, MAP_Y - 3, mx + M * MCELL + 3, MAP_Y + M * MCELL + 3], outline=(80, 88, 140), width=3)


def draw_stack(frame, t, d):
    t0 = word(STACK, "stack")
    if t < t0 - 0.2 or t >= T.start(CTA):
        return
    stages = [("pixels", word(STACK, "stack")), ("edges", word(STACK, "edges")), ("shapes", word(STACK, "shapes")),
              ("faces", word(STACK, "faces"))]
    bw, gap, y0 = 196, 38, 1090
    x = (W - (4 * bw + 3 * gap)) / 2 - 30
    for j, (name, ta) in enumerate(stages):
        g = ease_back((t - ta) / 0.4)
        if g <= 0:
            x += bw + gap
            continue
        cx, cy = x + bw / 2, y0 + bw / 2
        hb = bw / 2 * min(1, g)
        d.rounded_rectangle([cx - hb, cy - hb, cx + hb, cy + hb], radius=24, fill=(18, 22, 48),
                            outline=[CYAN, YELLOW, ORANGE, GREEN][j], width=5)
        if g > 0.7:
            col = [CYAN, YELLOW, ORANGE, GREEN][j]
            if j == 0:  # pixel grid
                for a in range(4):
                    for b in range(4):
                        v = (a + b) % 3 == 0
                        d.rectangle([cx - 60 + b * 30, cy - 60 + a * 30, cx - 34 + b * 30, cy - 34 + a * 30],
                                    fill=(236, 240, 255) if v else (40, 44, 80))
            elif j == 1:  # edges
                for (a, b, c2, e) in ((-55, -40, -5, -60), (5, -55, 55, -10), (-50, 20, 0, 55), (15, 40, 60, 30)):
                    d.line([(cx + a, cy + b), (cx + c2, cy + e)], fill=col, width=8)
            elif j == 2:  # shapes
                d.ellipse([cx - 62, cy - 50, cx - 6, cy + 6], outline=col, width=7)
                d.polygon([(cx + 34, cy - 54), (cx + 66, cy + 4), (cx + 2, cy + 4)], outline=col, width=7)
                d.rectangle([cx - 40, cy + 22, cx + 40, cy + 62], outline=col, width=7)
            else:  # face
                d.ellipse([cx - 64, cy - 64, cx + 64, cy + 64], outline=col, width=7)
                d.ellipse([cx - 32, cy - 26, cx - 14, cy - 8], fill=col)
                d.ellipse([cx + 14, cy - 26, cx + 32, cy - 8], fill=col)
                d.arc([cx - 36, cy - 10, cx + 36, cy + 40], 20, 160, fill=col, width=7)
            text_center(d, (cx, y0 + bw + 30), name, font("Black", 32), col)
        if j < 3 and t >= stages[j + 1][1]:
            ax = x + bw + 4
            d.line([(ax, cy), (ax + gap - 10, cy)], fill=WHITE, width=5)
            d.polygon([(ax + gap - 4, cy), (ax + gap - 16, cy - 9), (ax + gap - 16, cy + 9)], fill=WHITE)
        x += bw + gap


def hud(t):
    if t < T.start(PIXELS) - 0.1:
        return f"{(int(t * FPS) * 7919) % 7 - 3}", PINK
    s = slide_state(t)
    if s:
        fi, n, (r, c) = s
        v = int(current_maps(t)[fi][r, c])
        return f"{v}", GREEN if v == 3 else (CYAN if v > 0 else MUTED)
    if t >= word(SLIDE, "slide") and t < T.start(STACK):
        return "3", GREEN
    return "—", MUTED


def draw_overlays(frame, t):
    draw_hud_pills(frame, 9, *hud(t), label="SCORE")
    frame.alpha_composite(TITLE_IMG, (0, 190))
    banners = [
        (T.start(HOOK), T.start(PIXELS), "HOW DOES AI SEE?", PURPLE, WHITE),
        (T.start(PIXELS), T.start(FILTER), "A PICTURE = A GRID OF NUMBERS", CYAN, (12, 14, 30)),
        (T.start(FILTER), T.start(SLIDE), "A FILTER = 3×3 WEIGHTS", YELLOW, (20, 20, 30)),
        (T.start(SLIDE), T.start(FOUND), "SLIDE · MULTIPLY · ADD (LIKE A NEURON)", YELLOW, (20, 20, 30)),
        (T.start(FOUND), T.start(SECOND), "BIG SCORE = PATTERN FOUND!", GREEN, (12, 28, 18)),
        (T.start(SECOND), word(SECOND, "both"), "FILTER 2: THE OTHER DIAGONAL", ORANGE, (20, 20, 30)),
        (word(SECOND, "both"), T.start(SHIFT), "\\ + / BOTH FOUND = X ✓", GREEN, (12, 28, 18)),
        (T.start(SHIFT), T.start(STACK), "SAME FILTER EVERYWHERE → FINDS IT ANYWHERE", CYAN, (12, 14, 30)),
        (T.start(STACK), T.start(CNN), "LAYERS: EDGES → SHAPES → FACES", PURPLE, WHITE),
        (T.start(CNN), T.start(CTA), "CONVOLUTIONAL NEURAL NETWORK (CNN)", PURPLE, WHITE),
    ]
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 32), (W / 2, TOP_TAG_Y), t, a)

    # Number legend while introducing pixels.
    if word(PIXELS, "zero") <= t < T.start(FILTER):
        pop_in(frame, tag("0 = dark   1 = bright", (20, 20, 30), WHITE + (255,), 32), (800, 640), t,
               word(PIXELS, "zero"))
    # "FOUND" tags on the feature map peaks.
    t_f = word(FOUND, "found!")
    if t_f <= t < T.start(SECOND):
        pop_in(frame, tag("← 3 = FOUND IT!", (20, 20, 30), YELLOW + (255,), 34),
               (MAP_XS[0] + M * MCELL + 150, MAP_Y + 2.5 * MCELL), t, t_f)
    # Filters are learned.
    if word(STACK, "learn") <= t < word(STACK, "stack"):
        pop_in(frame, tag("filters are learned by backprop (Level 8)", (20, 20, 30), YELLOW + (255,), 30),
               (W / 2, 1200), t, word(STACK, "learn"))
    # What CNNs spot.
    for j, (w_, lab) in enumerate((("cats", "cats"), ("faces", "faces"), ("stop", "stop signs"))):
        ta = word(CNN, w_)
        if ta <= t < T.start(CTA):
            pop_in(frame, tag(lab, (12, 14, 30), [CYAN, GREEN, PINK][j] + (255,), 36), (230 + 280 * j, 1052), t, ta)
    draw_level_cleared(frame, t, T.start(CTA), 9, "RECURRENT NETWORKS", word(CTA, "next:"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    d = ImageDraw.Draw(frame)
    draw_image(frame, t, d)
    draw_filter(frame, t, d)
    draw_arith(frame, t, d)
    draw_maps(frame, t, d)
    draw_stack(frame, t, d)
    draw_overlays(frame, t)
    if captions:
        draw_caption(frame, t, CHUNKS, CAPTION_Y)
    if t < 0.15:
        frame.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(200 * (1 - t / 0.15)))))
    return frame


def render_frame(i):
    return render_rgba(i).convert("RGB").tobytes()


# ---------------------------------------------------------------- audio

def build_audio(voice):
    out = voice * 0.95
    mix(out, music_bed(len(out) / SR), 0.0)
    mix(out, sfx_whoosh(0.7, 0.2), 0.0)
    for k in range(int(T.start(PIXELS) * 5)):
        mix(out, sfx_blip(1000 + 90 * (k % 4), 0.03, 0.06), k / 5)
    mix(out, sfx_blip(880, 0.12, 0.13), word(PIXELS, "number"))
    mix(out, sfx_blip(1320, 0.2, 0.15), word(PIXELS, "x."))
    mix(out, sfx_boing(0.5, 0.16), word(FILTER, "tiny"))
    last = -1
    for i in range(int(word(SLIDE, "slide") * FPS), int(word(SECOND, "both") * FPS)):  # tick per window move
        s = slide_state(i / FPS)
        if s and (s[0], s[1]) != last:
            last = (s[0], s[1])
            v = MAPS0[s[0]][s[2]] if s else 0
            mix(out, sfx_blip(500 + 120 * max(0, v), 0.03, 0.06 if v < 3 else 0.12), i / FPS)
    mix(out, sfx_blip(1568, 0.3, 0.18), word(FOUND, "found!"))
    mix(out, sfx_blip(1568, 0.3, 0.18), word(SECOND, "both"))
    mix(out, sfx_whoosh(0.8, 0.16, seed=5), word(SHIFT, "same"))
    for w_ in ("edges", "shapes", "faces"):
        mix(out, sfx_blip(880, 0.12, 0.13), word(STACK, w_))
    for w_ in ("cats", "faces", "stop"):
        mix(out, sfx_blip(1175, 0.1, 0.12), word(CNN, w_))
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12, max_px=800)
    STATIC = build_plain()
    TITLE_IMG = build_title("HOW AI SEES", "ML AS A GAME  ·  EP 9")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = word(SECOND, "both") + 0.6
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "HOW AI SEES", font("Black", 110), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "Convolutions (CNNs), explained", font("Black", 56), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s")
    for fi in range(2):
        print(f"filter {NAMES[fi]}: map before shift\n{MAPS0[fi].astype(int)}\n peaks {PEAKS0[fi]} -> after shift "
              f"{PEAKS1[fi]}")
    for line in T.lines:
        print(f"  [{line['start']:5.2f}-{line['end']:5.2f}] {line['text']}")
    if args.thumbnail:
        thumbnail(os.path.join(HERE, "thumbnail.png"))
        return
    if args.preview:
        os.makedirs(os.path.join(HERE, "preview"), exist_ok=True)
        for s in args.preview.split(","):
            render_rgba(int(float(s) * FPS)).convert("RGB").save(os.path.join(HERE, "preview", f"t{float(s):05.2f}.png"))
        return
    wav = os.path.join(HERE, "audio.wav")
    write_wav(wav, build_audio(voice))
    out = os.path.join(HERE, "cnn_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
