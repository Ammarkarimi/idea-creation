"""ML as a Game, Level 1: Linear Regression (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> linear_regression_short.mp4 + thumbnail.png
    python3 render.py --preview 3,12  # just dump PNG frames at those seconds (fast check)
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
    CYAN, FPS, GREEN, H, MUTED, PINK, SR, W, WHITE, YELLOW, PlotBox,
    add_glow, build_confetti, build_title, caption_chunks, clamp, dashed_line, draw_caption,
    draw_hud_pills, draw_level_cleared, draw_training_readout, ease_back, ease_in_out, ease_out, encode, font, lerp, lerp_color, mix,
    music_bed, paste_scaled, rounded_pill, sfx_blip, sfx_boing, sfx_levelup, sfx_whoosh, text_center,
    voiceover, write_wav, zigzag,
)
from common import build_static as common_build_static  # noqa: E402

SCRIPT = [
    "This line is playing a game, and it's how machine learning begins.",
    "Every dot is a student: hours studied, and their exam score.",
    "The goal: one straight line that's as close to every dot as possible.",
    "Tie each dot to the line with a spring. Far dots pull harder.",
    "These gaps are the errors. Square them, add them up. That's the score. Lower wins.",
    "The model nudges the slope and the starting point, again and again, until the springs stop pulling.",
    "That's linear regression: y equals m x plus b.",
    "Studied six hours? It predicts about seventy-eight.",
    "Level one cleared. Next: logistic regression. Follow for level two.",
]
HOOK, DATA, GOAL, SPRINGS, SCORE, TRAIN, FORMULA, PREDICT, CTA = range(9)

# ---------------------------------------------------------------- data + model

M_TRUE, B_TRUE = 7.0, 36.0  # final fit is forced to exactly y = 7x + 36 (so 6h -> 78)
X = np.array([1.0, 1.6, 2.2, 2.8, 3.3, 4.0, 4.6, 5.2, 6.4, 7.1, 7.8, 8.6])
rng = np.random.default_rng(7)
Y = M_TRUE * X + B_TRUE + rng.normal(0, 6.5, len(X))
# Adding a linear function of x shifts the least-squares fit by exactly that function.
m_fit, b_fit = np.polyfit(X, Y, 1)
Y = Y + (M_TRUE - m_fit) * X + (B_TRUE - b_fit)
assert np.allclose(np.polyfit(X, Y, 1), [M_TRUE, B_TRUE])

M0, B0 = -1.5, 62.0  # bad starting guess


def mse(m, b):
    return float(np.mean((Y - (m * X + b)) ** 2))


def gradient_descent(steps=60, momentum=0.72, lr=0.32):
    """Momentum GD on centred x; returns (m, b) per step. Momentum gives the springy wobble."""
    xm = X.mean()
    xc = X - xm
    m, c = M0, M0 * xm + B0  # y = m*(x - xm) + c
    vm = vc = 0.0
    lr_m, lr_c = lr / (2 * np.mean(xc ** 2)), lr / 2
    out = [(m, c - m * xm)]
    for _ in range(steps):
        r = Y - (m * xc + c)
        gm, gc = -2 * np.mean(xc * r), -2 * np.mean(r)
        vm = momentum * vm - lr_m * gm
        vc = momentum * vc - lr_c * gc
        m, c = m + vm, c + vc
        out.append((m, c - m * xm))
    return out


TRAJ = gradient_descent()
MSE0, MSE_FIT = mse(M0, B0), mse(M_TRUE, B_TRUE)

# ---------------------------------------------------------------- layout

PX0, PX1, PY_TOP, PY_BOT = 150, 960, 450, 1220  # plot box (x: 0..10 h, y: 0..100 pts)
LX0, LY0, LX1, LY1 = 90, 410, 1010, 1270  # plot drawing layer (rendered at 2x)
SS = 2
CAPTION_Y = 1480


def px(x):
    return PX0 + (PX1 - PX0) * x / 10


def py(y):
    return PY_BOT - (PY_BOT - PY_TOP) * y / 100


def L(x, y):
    """Data coords -> 2x layer coords."""
    return ((px(x) - LX0) * SS, (py(y) - LY0) * SS)


def line_segment(m, b):
    """Clip y = m x + b to the plot box; returns two data points or None."""
    pts = []
    for x in (0, 10):
        pts.append((x, m * x + b))
    for yb in (0, 100):
        if abs(m) > 1e-6:
            x = (yb - b) / m
            if 0 < x < 10:
                pts.append((x, yb))
    pts = [(x, y) for x, y in pts if -1e-6 <= y <= 100 + 1e-6]
    if len(pts) < 2:
        return None
    pts.sort()
    return pts[0], pts[-1]


# ---------------------------------------------------------------- timeline (filled in main)

TL = []
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0


def start(i):
    return TL[i]["start"]


def end(i):
    return TL[i]["end"]


def word_time(i, prefix):
    for w, s, _ in TL[i]["words"]:
        if w.lower().startswith(prefix.lower()):
            return s
    raise KeyError(prefix)


def scene_of(t):
    s = 0
    for i in range(len(TL)):
        if t >= start(i) - 0.12:
            s = i
    return s


# ---------------------------------------------------------------- state

def line_params(t):
    """(m, b, alpha, reveal 0..1) of the fitted line at time t, or None."""
    if t < start(DATA) - 0.1:
        m = 22 * math.sin(2 * math.pi * 0.55 * t + 0.6)
        return m, 55 - 5 * m, clamp((start(DATA) - 0.1 - t) / 0.25), 1.0
    if t < start(GOAL):
        return None
    if t < start(TRAIN):
        reveal = ease_in_out((t - start(GOAL) - 0.2) / 0.7)
        m, b = M0, B0
        ts = t - start(SPRINGS) - 0.3
        if ts > 0:  # springs attach -> small jiggle
            m += 0.9 * math.sin(ts * 9) * math.exp(-ts * 2.5)
        return m, b, 1.0, reveal
    p = clamp((t - start(TRAIN) - 0.2) / (end(TRAIN) - start(TRAIN) - 0.2))
    f = p * (len(TRAJ) - 1)
    i = min(int(f), len(TRAJ) - 2)
    a = f - i
    m = lerp(TRAJ[i][0], TRAJ[i + 1][0], a)
    b = lerp(TRAJ[i][1], TRAJ[i + 1][1], a)
    return m, b, 1.0, 1.0


def train_step(t):
    p = clamp((t - start(TRAIN) - 0.2) / (end(TRAIN) - start(TRAIN) - 0.2))
    return int(round(p * (len(TRAJ) - 1)))


def error_display(t):
    """(text, color) for the HUD error counter."""
    if t < start(DATA) - 0.1:
        n = (int(t * FPS) * 7919) % 900 + 100
        return f"{n}", PINK
    t_add = word_time(SCORE, "add")
    if t < t_add:
        return "???", MUTED
    if t < start(TRAIN):
        return f"{int(MSE0 * ease_out((t - t_add) / 0.8))}", PINK
    m, b, _, _ = line_params(t)
    e = mse(m, b)
    good = clamp((MSE0 - e) / (MSE0 - MSE_FIT))
    return f"{int(round(e))}", lerp_color(PINK, GREEN, good ** 3)


# ---------------------------------------------------------------- drawing

def build_static():
    plot = PlotBox(PX0, PX1, PY_TOP, PY_BOT, (0, 10), (0, 100))
    return common_build_static(plot, list(range(0, 11, 2)), list(range(0, 101, 25)), "HOURS STUDIED", "EXAM SCORE")


def draw_plot_layer(t):
    lay = Image.new("RGBA", ((LX1 - LX0) * SS, (LY1 - LY0) * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    lp = line_params(t)
    sc = scene_of(t)

    # Squared errors (squares) and springs follow the current line.
    if lp and start(SPRINGS) <= t:
        m, b = lp[0], lp[1]
        fade = 1 - clamp((t - start(FORMULA)) / 0.4)
        t_sq = word_time(SCORE, "square")
        t_err = word_time(SCORE, "errors")
        flash = 1 + 0.8 * max(0, math.sin((t - t_err) * 14)) if t_err <= t < t_err + 0.7 else 1
        sq = []
        for k, (x, y) in enumerate(zip(X, Y)):
            yh = m * x + b
            r = y - yh
            if t >= t_sq and fade > 0:
                g = ease_back((t - t_sq - 0.05 * k) / 0.35) * fade
                if g > 0:
                    # Square whose side is the gap: literally the "squared error".
                    (x0, ya), (_, yb) = L(x, y), L(x, yh)
                    top, side = min(ya, yb), abs(ya - yb) * g
                    xs = (x0, x0 + side) if x < 5 else (x0 - side, x0)
                    sq.append((side, [xs[0], top, xs[1], top + side]))
            grow = ease_out((t - start(SPRINGS) - 0.25 - 0.06 * k) / 0.35) * fade
            if grow <= 0:
                continue
            p0 = L(x, y)
            p1 = L(x, lerp(y, yh, grow))
            col = lerp_color(CYAN, PINK, abs(r) / 22)
            length = abs(p1[1] - p0[1])
            coils = max(2, int(length / 34))
            d.line(zigzag(p0, p1, coils=coils, amp=16), fill=col + (int(255 * fade),),
                   width=int(7 * flash), joint="curve")

        if sq:
            sl = Image.new("RGBA", lay.size, (0, 0, 0, 0))
            sd = ImageDraw.Draw(sl)
            for _, box in sorted(sq, key=lambda q: -q[0]):
                sd.rectangle(box, fill=PINK + (int(28 * fade),))
            for _, box in sq:
                sd.rectangle(box, outline=PINK + (int(170 * fade),), width=5)
            sl.alpha_composite(lay)
            lay = sl
            d = ImageDraw.Draw(lay)

    # The line itself.
    if lp and lp[2] > 0:
        m, b, alpha, reveal = lp
        seg = line_segment(m, b)
        if seg:
            (xa, ya), (xb, yb) = seg
            xb2 = lerp(xa, xb, reveal)
            yb2 = lerp(ya, yb, reveal)
            if sc <= HOOK:
                col = PINK
            else:
                e = mse(m, b)
                col = lerp_color(PINK, GREEN, clamp((MSE0 - e) / (MSE0 - MSE_FIT)) ** 3)
            d.line([L(xa, ya), L(xb2, yb2)], fill=col + (int(255 * alpha),), width=20)

    # Prediction guide lines (drawn under the dots).
    t6 = word_time(PREDICT, "six")
    if t >= t6:
        g = ease_out((t - t6) / 0.6)
        yp = M_TRUE * 6 + B_TRUE
        dashed_line(d, L(6, 0), L(6, lerp(0, yp, g)), YELLOW + (230,), width=7, dash=26, gap=18)
        g2 = ease_out((t - t6 - 0.6) / 0.5)
        if g2 > 0:
            dashed_line(d, L(6, yp), L(lerp(6, 0, g2), yp), YELLOW + (230,), width=7, dash=26, gap=18)

    # Data dots pop in one by one.
    if t >= start(DATA):
        span = (end(DATA) - start(DATA)) * 0.75
        for k, (x, y) in enumerate(zip(X, Y)):
            s = ease_back((t - start(DATA) - span * k / len(X)) / 0.3)
            if s <= 0:
                continue
            cx, cy = L(x, y)
            r = 17 * SS * s
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CYAN + (255,))
            r2 = r * 0.55
            d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=WHITE + (255,))

    # Predicted point.
    if t >= t6 + 0.55:
        s = ease_back((t - t6 - 0.55) / 0.35)
        pulse = 1 + 0.12 * math.sin((t - t6) * 8)
        cx, cy = L(6, M_TRUE * 6 + B_TRUE)
        r = 24 * SS * s * pulse
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=YELLOW + (255,))
        r2 = r * 0.45
        d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=WHITE + (255,))

    small = lay.resize((LX1 - LX0, LY1 - LY0), Image.LANCZOS)
    return add_glow(small, radius=12, strength=1.5)


def draw_hud(frame, t):
    draw_hud_pills(frame, 1, *error_display(t))

    # Title -> formula swap.
    tf = start(FORMULA) + 0.15
    if t < tf:
        frame.alpha_composite(TITLE_IMG, (0, 255))
    else:
        g = ease_back((t - tf) / 0.45)
        card = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
        cd = ImageDraw.Draw(card)
        f = font("Black", 92)
        parts = [("y = ", WHITE), ("7", YELLOW), ("x + ", WHITE), ("36", GREEN)]
        total = sum(f.getlength(p) for p, _ in parts)
        x = (W - total) / 2
        centers = []
        for p, c in parts:
            cd.text((x, 82), p, font=f, fill=c, anchor="lm")
            centers.append(x + f.getlength(p.strip()) / 2)
            x += f.getlength(p)
        text_center(cd, (centers[1], 160), "slope (m)", font("Bold", 30), YELLOW)
        text_center(cd, (centers[3], 160), "start (b)", font("Bold", 30), GREEN)
        card = add_glow(card, radius=10, strength=1.2)
        paste_scaled(frame, card, (W / 2, 355), scale=0.6 + 0.4 * g, alpha=clamp(g))

    # Training readout under the plot.
    if start(TRAIN) <= t < start(FORMULA) + 0.1:
        lp = line_params(t)
        a = clamp((t - start(TRAIN)) / 0.3) * (1 - clamp((t - start(FORMULA)) / 0.1))
        draw_training_readout(frame, PX0, PX1, PY_BOT + 140, train_step(t), len(TRAJ) - 1,
                              f"m {lp[0]:5.1f}   b {lp[1]:5.1f}", a)

    # Prediction bubble.
    t78 = word_time(PREDICT, "seventy")
    if t >= t78 - 0.1:
        g = ease_back((t - t78 + 0.1) / 0.35)
        bub = Image.new("RGBA", (300, 150), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bub)
        rounded_pill(bd, (10, 10, 290, 140), fill=(15, 18, 40, 240), outline=YELLOW + (255,), width=5, radius=28)
        text_center(bd, (150, 58), "≈ 78", font("Black", 64), YELLOW)
        text_center(bd, (150, 112), "predicted score", font("Bold", 26), MUTED)
        paste_scaled(frame, bub, (px(6) - 235, py(78) - 115), scale=g, alpha=clamp(g))
    t6 = word_time(PREDICT, "six")
    if t >= t6:
        g = ease_back((t - t6) / 0.3)
        tag = Image.new("RGBA", (130, 64), (0, 0, 0, 0))
        td = ImageDraw.Draw(tag)
        rounded_pill(td, (0, 0, 130, 64), fill=YELLOW + (255,))
        text_center(td, (65, 32), "6 h", font("Black", 38), (20, 20, 30))
        paste_scaled(frame, tag, (px(6), PY_BOT + 34), scale=g, alpha=clamp(g))


def draw_cta(frame, t):
    draw_level_cleared(frame, t, start(CTA), 1, "LOGISTIC REGRESSION", word_time(CTA, "next"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    frame.alpha_composite(draw_plot_layer(t), (LX0, LY0))
    draw_hud(frame, t)
    draw_cta(frame, t)
    if captions:
        draw_caption(frame, t, CHUNKS, CAPTION_Y)
    # Quick white flash on the hook's first frames.
    if t < 0.15:
        frame.alpha_composite(Image.new("RGBA", (W, H), (255, 255, 255, int(200 * (1 - t / 0.15)))))
    return frame


def render_frame(i):
    return render_rgba(i).convert("RGB").tobytes()


# ---------------------------------------------------------------- audio

def build_audio(voice):
    out = voice * 0.95
    mix(out, music_bed(len(out) / SR), 0.0)
    mix(out, sfx_whoosh(0.7, 0.22), 0.0)
    span = (end(DATA) - start(DATA)) * 0.75
    for k in range(len(X)):
        mix(out, sfx_blip(600 + 40 * k, 0.1, 0.16), start(DATA) + span * k / len(X))
    mix(out, sfx_whoosh(0.6, 0.15, seed=2), start(GOAL) + 0.15)
    mix(out, sfx_boing(0.6, 0.22), start(SPRINGS) + 0.25)
    t_sq = word_time(SCORE, "square")
    for k in range(len(X)):
        mix(out, sfx_blip(900 + 30 * k, 0.07, 0.1), t_sq + 0.05 * k)
    t_add = word_time(SCORE, "add")
    for k in range(12):
        mix(out, sfx_blip(1500, 0.03, 0.06), t_add + k * 0.065)
    mix(out, sfx_whoosh(0.9, 0.2, seed=4), start(TRAIN) + 0.1)
    for k in range(0, len(TRAJ), 6):
        mix(out, sfx_blip(700 + 12 * k, 0.05, 0.07), start(TRAIN) + 0.2 + (end(TRAIN) - start(TRAIN) - 0.2) * k / len(TRAJ))
    mix(out, sfx_blip(1046, 0.25, 0.18), start(FORMULA) + 0.15)
    mix(out, sfx_blip(1320, 0.35, 0.2), word_time(PREDICT, "six") + 0.55)
    mix(out, sfx_blip(1568, 0.35, 0.18), word_time(PREDICT, "seventy") - 0.1)
    mix(out, sfx_levelup(0.24), start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global TL, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION
    voice, TL = voiceover(SCRIPT)
    CHUNKS = caption_chunks(TL)
    STATIC = build_static()
    TITLE_IMG = build_title("LINEAR REGRESSION", "ML AS A GAME  ·  EP 1")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = word_time(SCORE, "lower") + 0.3
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    text_center(d, (W / 2, 1420), "ML AS A GAME", font("Black", 124), YELLOW, stroke=8)
    text_center(d, (W / 2, 1570), "Linear Regression, explained", font("Black", 60), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s, MSE {MSE0:.0f} -> {MSE_FIT:.0f}, 6h -> {M_TRUE * 6 + B_TRUE:.0f}")
    for i, line in enumerate(TL):
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
    out = os.path.join(HERE, "linear_regression_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
