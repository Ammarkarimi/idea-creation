"""ML as a Game, Level 2: Logistic Regression (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> logistic_regression_short.mp4 + thumbnail.png
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
    CYAN, FPS, GREEN, H, MUTED, ORANGE, PINK, PURPLE, SR, W, WHITE, YELLOW, PlotBox, Timeline,
    add_glow, build_confetti, build_static, build_title, caption_chunks, clamp, dashed_line, draw_caption,
    draw_hud_pills, draw_level_cleared, draw_training_readout, ease_back, ease_in_out, ease_out, encode, font,
    lerp, lerp_color, mix, music_bed, num_to_words, paste_scaled, pop_in, rounded_pill, sfx_blip, sfx_boing, sfx_buzz,
    sfx_levelup, sfx_whoosh, tag, text_center, voiceover, write_wav,
)

# ---------------------------------------------------------------- data + models

# Hours studied and pass (1) / fail (0). Deliberate overlap (a pass at 3.2 h, a fail at 5.0 h)
# keeps the maximum-likelihood fit finite, like real data.
X = np.array([0.8, 1.5, 2.1, 2.7, 3.2, 3.6, 4.1, 4.6, 5.0, 5.5, 6.1, 6.8, 7.6, 8.4, 9.2])
Y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1], dtype=float)
# Where the dots sit before they snap to pass/fail rows (scores out of 100, as in Level 1).
rng = np.random.default_rng(11)
Y_SCATTER = np.clip(0.07 * X + 0.33 + rng.normal(0, 0.06, len(X)), 0.08, 0.95)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def log_loss(w, c):
    p = np.clip(sigmoid(w * X + c), 1e-9, 1 - 1e-9)
    return float(-np.sum(Y * np.log(p) + (1 - Y) * np.log(1 - p)))


def penalties(w, c):
    """-log(probability the model gave to the true answer), per student."""
    p = np.clip(sigmoid(w * X + c), 1e-9, 1 - 1e-9)
    return -np.log(np.where(Y == 1, p, 1 - p))


def newton_fit():
    w = c = 0.0
    for _ in range(50):
        p = sigmoid(w * X + c)
        s = p * (1 - p)
        g = np.array([np.sum((p - Y) * X), np.sum(p - Y)])
        hess = np.array([[np.sum(s * X * X), np.sum(s * X)], [np.sum(s * X), np.sum(s)]])
        w, c = np.array([w, c]) - np.linalg.solve(hess, g)
    return float(w), float(c)


OLS_M, OLS_B = np.polyfit(X, Y, 1)  # the "straight line" attempt
W_FIT, C_FIT = newton_fit()
BOUNDARY = -C_FIT / W_FIT  # where p = 50%
# Matched sigmoid: same 50% point and same slope there as the straight line (slope of sigmoid = w/4).
W_BEND = 4 * OLS_M
C_BEND = -W_BEND * (0.5 - OLS_B) / OLS_M
W0, C0 = -0.9, 4.5  # "random guess": backwards curve
X_PRED = 4.5
P_PRED = round(100 * float(sigmoid(W_FIT * X_PRED + C_FIT)))
WORST = int(np.argmax(penalties(W0, C0)))
WORST_SURE = round(100 * (1 - float(sigmoid(W0 * X[WORST] + C0))))
LOSS0, LOSS_FIT = log_loss(W0, C0), log_loss(W_FIT, C_FIT)


def gradient_descent(steps=60, lr=0.22, momentum=0.8):
    """Momentum GD on mean log loss (centred x). Returns (w, c) per step."""
    xm = X.mean()
    xc = X - xm
    w, k = W0, C0 + W0 * xm  # z = w*(x - xm) + k
    vw = vk = 0.0
    out = [(w, k - w * xm)]
    for _ in range(steps):
        p = sigmoid(w * xc + k)
        gw, gk = np.mean((p - Y) * xc), np.mean(p - Y)
        vw = momentum * vw - lr * gw / np.mean(xc ** 2) * 4
        vk = momentum * vk - lr * gk * 4
        w, k = w + vw, k + vk
        out.append((w, k - w * xm))
    return out


TRAJ = gradient_descent()
assert abs(log_loss(*TRAJ[-1]) - LOSS_FIT) < 0.02, (log_loss(*TRAJ[-1]), LOSS_FIT)
TRAJ[-1] = (W_FIT, C_FIT)  # land exactly on the fitted model

SCRIPT = [
    "Can a straight line predict pass or fail? Watch it break.",
    "Same students. But now every dot is just pass, or fail.",
    "Fit a straight line, and at ten hours it predicts over a hundred percent. That's impossible.",
    "So we bend it: squash the line through a sigmoid, so it stays between zero and one hundred percent.",
    "Its height is the chance of passing. But training starts from a random guess.",
    "The score punishes confident mistakes. Super sure, and wrong? Huge penalty.",
    "Training slides and stretches the curve, until the penalties are as small as possible.",
    "Where it crosses fifty percent is the decision line. Left fails. Right passes.",
    f"Studied four and a half hours? {num_to_words(P_PRED).capitalize()} percent chance. "
    + ("Just a pass!" if P_PRED < 60 else "Pass!"),
    "Level two cleared. Next: decision trees. Follow for level three.",
]
HOOK, DATA, LINE, BEND, HEIGHT, PENALTY, TRAIN, GATE, PREDICT, CTA = range(10)

# ---------------------------------------------------------------- layout

PLOT = PlotBox(190, 960, 450, 1220, (0, 10), (-0.35, 1.35))
px, py, L = PLOT.px, PLOT.py, PLOT.L
SS = PLOT.ss
CAPTION_Y = 1480
AX_Y = py(0)  # x-axis sits at 0%

T: Timeline = None
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0


def pct(v):
    return f"{int(round(v * 100))}%"


# ---------------------------------------------------------------- state

def curve(t):
    """Current model as (kind, params, alpha, reveal); kind is 'line' or 'mix'/'sig'."""
    t_break = T.word(HOOK, "break")
    if t < t_break + 0.05:
        reveal = ease_in_out((t - 0.2) / 0.5)
        return ("line", (OLS_M, OLS_B), 1.0, reveal)
    if t < T.start(LINE):
        return None
    t_sq = T.word(BEND, "squash")
    if t < t_sq:
        reveal = ease_in_out((t - T.start(LINE) - 0.15) / 0.6)
        return ("line", (OLS_M, OLS_B), 1.0, reveal)
    t_rand = T.word(HEIGHT, "random")
    if t < t_rand:
        a = ease_in_out((t - t_sq) / 0.9)
        return ("mix", (a, W_BEND, C_BEND), 1.0, 1.0)
    if t < T.start(TRAIN):
        a = ease_in_out((t - t_rand) / 0.5)
        return ("sig", (lerp(W_BEND, W0, a), lerp(C_BEND, C0, a)), 1.0, 1.0)
    w, c = train_params(t)
    return ("sig", (w, c), 1.0, 1.0)


def train_progress(t):
    return clamp((t - T.start(TRAIN) - 0.2) / (T.end(TRAIN) - T.start(TRAIN) - 0.1))


def train_params(t):
    f = train_progress(t) * (len(TRAJ) - 1)
    i = min(int(f), len(TRAJ) - 2)
    a = f - i
    return lerp(TRAJ[i][0], TRAJ[i + 1][0], a), lerp(TRAJ[i][1], TRAJ[i + 1][1], a)


def curve_fn(state):
    kind, prm, _, _ = state
    if kind == "line":
        m, b = prm
        return lambda x: m * x + b
    if kind == "mix":
        a, w, c = prm
        return lambda x: (1 - a) * (OLS_M * x + OLS_B) + a * sigmoid(w * x + c)
    w, c = prm
    return lambda x: sigmoid(w * x + c)


def model_params(t):
    """(w, c) when the model is a sigmoid with its own parameters, else None."""
    st = curve(t)
    if st and st[0] == "sig":
        return st[1]
    return None


def error_display(t):
    if t < T.start(DATA) - 0.1:
        n = (int(t * FPS) * 7919) % 900 + 100
        return f"{n / 10:.1f}", PINK
    t_pun = T.word(PENALTY, "punishes")
    if t < t_pun:
        return "???", MUTED
    if t < T.start(TRAIN):
        return f"{LOSS0 * ease_out((t - t_pun) / 0.8):.1f}", PINK
    e = log_loss(*train_params(t))
    good = clamp((LOSS0 - e) / (LOSS0 - LOSS_FIT))
    return f"{e:.1f}", lerp_color(PINK, GREEN, good ** 3)


def dot_pos(k, t):
    """(y, color, scale) of dot k."""
    t0 = T.start(DATA)
    span = (T.word(DATA, "just") - t0) * 0.9
    s = ease_back((t - t0 - span * k / len(X)) / 0.3)
    t_snap = T.word(DATA, "pass")
    g = ease_back((t - t_snap - 0.04 * k) / 0.45, s=1.4)
    y = lerp(Y_SCATTER[k], Y[k], g)
    col = lerp_color(CYAN, GREEN if Y[k] else PINK, clamp(g))
    return y, col, s


# ---------------------------------------------------------------- drawing

def draw_plot_layer(t):
    lay = PLOT.new_layer()
    d = ImageDraw.Draw(lay)
    sc = T.scene(t)

    # "Impossible" zones above 100% and below 0%.
    t_imp = T.word(LINE, "hundred")
    zone_a = 0.0
    if t < T.start(DATA):
        zone_a = 0.6 * clamp((t - T.word(HOOK, "break") + 0.3) / 0.2) * (1 - clamp((t - T.start(DATA) + 0.4) / 0.3))
    elif t >= t_imp:
        zone_a = clamp((t - t_imp) / 0.3) * (1 - 0.85 * clamp((t - T.word(BEND, "squash") - 0.6) / 0.6))
        if t < T.word(LINE, "impossible") + 0.6:
            zone_a *= 0.75 + 0.25 * math.sin((t - t_imp) * 12)
    if zone_a > 0.01:
        for y_lo, y_hi in ((1.0, 1.35), (-0.35, 0.0)):
            (x0, ya), (x1, yb) = L(0, y_hi), L(10, y_lo)
            d.rectangle([x0, ya, x1, yb], fill=PINK + (int(55 * zone_a),))
            for hx in range(int(x0) - 400, int(x1), 44):  # diagonal hatching
                d.line([(max(hx, x0), ya + (max(hx, x0) - hx)), (min(hx + (yb - ya), x1), ya + min(hx + (yb - ya), x1) - hx)],
                       fill=PINK + (int(70 * zone_a),), width=4)

    # Decision gate: shaded FAIL / PASS regions.
    t_dec = T.word(GATE, "decision")
    if t >= T.word(GATE, "left") - 0.05:
        a = clamp((t - T.word(GATE, "left") + 0.05) / 0.3)
        d.rectangle([*L(0, 1.0), *L(BOUNDARY, 0.0)], fill=PINK + (int(40 * a),))
    if t >= T.word(GATE, "right") - 0.05:
        a = clamp((t - T.word(GATE, "right") + 0.05) / 0.3)
        d.rectangle([*L(BOUNDARY, 1.0), *L(10, 0.0)], fill=GREEN + (int(36 * a),))
    t_fifty = T.word(GATE, "fifty")
    if t >= t_fifty:
        g = ease_out((t - t_fifty) / 0.5)
        dashed_line(d, L(0, 0.5), L(lerp(0, 10, g), 0.5), WHITE + (150,), width=5, dash=22, gap=16)
    if t >= t_dec:
        g = ease_out((t - t_dec) / 0.45)
        dashed_line(d, L(BOUNDARY, -0.12), L(BOUNDARY, lerp(-0.12, 1.12, g)), YELLOW + (255,), width=9, dash=28, gap=16)

    # Penalty bars: length = -log(prob of the true answer). Pass dots hang down, fail dots stand up.
    wc = model_params(t)
    t_pun = T.word(PENALTY, "punishes")
    if wc and t >= t_pun:
        fade = 1 - clamp((t - T.start(GATE)) / 0.4)
        pen = penalties(*wc)
        t_huge = T.word(PENALTY, "huge")
        for k in range(len(X)):
            g = ease_out((t - t_pun - 0.04 * k) / 0.4) * fade
            if g <= 0:
                continue
            length = min(pen[k], 4.0) * 0.12 * g
            y_end = Y[k] - length if Y[k] else Y[k] + length
            (bx, by0), (_, by1) = L(X[k], Y[k]), L(X[k], y_end)
            col = lerp_color(YELLOW, PINK, clamp(pen[k] / 2.5))
            width = 20
            if k == WORST and t_huge <= t < t_huge + 0.8:
                width = int(20 + 14 * abs(math.sin((t - t_huge) * 10)))
            d.rounded_rectangle([bx - width, min(by0, by1), bx + width, max(by0, by1)], radius=width,
                                fill=col + (int(215 * fade),))

    # The model: line, morphing line->S-curve, or sigmoid.
    st = curve(t)
    if st:
        f = curve_fn(st)
        reveal = st[3]
        xs = np.linspace(0, 10 * reveal, max(2, int(140 * reveal)))
        ys = np.clip([f(x) for x in xs], -0.35, 1.35)
        pts = [L(x, y) for x, y in zip(xs, ys)]
        if st[0] == "sig":
            e = log_loss(*st[1])
            col = lerp_color(PINK, GREEN, clamp((LOSS0 - e) / (LOSS0 - LOSS_FIT)) ** 3) if sc >= PENALTY else PURPLE
        elif st[0] == "mix":
            col = lerp_color(ORANGE, PURPLE, st[1][0])
        else:
            col = ORANGE
        if len(pts) > 1:
            d.line(pts, fill=col + (255,), width=20, joint="curve")
    # Hook: the line shatters into falling pieces.
    t_break = T.word(HOOK, "break")
    if t_break + 0.05 <= t < T.start(DATA):
        dt = t - t_break - 0.05
        n = 7
        for i in range(n):
            (ax, ay), (bx, by) = L(10 * i / n, OLS_M * 10 * i / n + OLS_B), L(10 * (i + 1) / n, OLS_M * 10 * (i + 1) / n + OLS_B)
            cx, cy = (ax + bx) / 2 + (i - 3) * 180 * dt, (ay + by) / 2 + 1800 * dt * dt + 120 * dt * (i % 3)
            hx, hy = (bx - ax) / 2 * 0.9, (by - ay) / 2 * 0.9
            rot = (i - 3) * 1.2 * dt
            cr, sr = math.cos(rot), math.sin(rot)
            hx, hy = hx * cr - hy * sr, hx * sr + hy * cr
            d.line([(cx - hx, cy - hy), (cx + hx, cy + hy)], fill=ORANGE + (int(255 * (1 - clamp(dt / 0.9))),), width=20)

    # Students.
    if t >= T.start(DATA):
        for k in range(len(X)):
            y, col, s = dot_pos(k, t)
            if s <= 0:
                continue
            cx, cy = L(X[k], y)
            r = 17 * SS * s
            if k == WORST and T.word(PENALTY, "super") <= t < T.start(TRAIN):
                r *= 1 + 0.25 * abs(math.sin((t - T.word(PENALTY, "super")) * 6))
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (255,))
            r2 = r * 0.55
            d.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=WHITE + (255,))

    # Probe sliding along the curve ("its height is the chance").
    t_h = T.word(HEIGHT, "height")
    t_rand = T.word(HEIGHT, "random")
    if t_h <= t < t_rand and st:
        xq = lerp(1.5, 8.0, ease_in_out((t - t_h) / max(0.5, t_rand - t_h - 0.2)))
        yq = float(curve_fn(st)(xq))
        dashed_line(d, L(xq, 0), L(xq, yq), WHITE + (200,), width=5, dash=18, gap=12)
        cx, cy = L(xq, yq)
        d.ellipse([cx - 26, cy - 26, cx + 26, cy + 26], fill=WHITE + (255,))

    # Prediction.
    t45 = T.word(PREDICT, "four")
    if t >= t45:
        yp = float(sigmoid(W_FIT * X_PRED + C_FIT))
        g = ease_out((t - t45) / 0.6)
        dashed_line(d, L(X_PRED, 0), L(X_PRED, lerp(0, yp, g)), YELLOW + (235,), width=7, dash=26, gap=18)
        if t >= t45 + 0.55:
            s = ease_back((t - t45 - 0.55) / 0.35) * (1 + 0.12 * math.sin((t - t45) * 8))
            cx, cy = L(X_PRED, yp)
            r = 24 * SS * s
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=YELLOW + (255,))
            d.ellipse([cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45], fill=WHITE + (255,))

    return PLOT.finish_layer(lay)


def draw_overlays(frame, t):
    draw_hud_pills(frame, 2, *error_display(t))

    # Title -> formula.
    tf = T.word(BEND, "sigmoid")
    if t < tf:
        frame.alpha_composite(TITLE_IMG, (0, 255))
    else:
        g = ease_back((t - tf) / 0.45)
        card = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
        cd = ImageDraw.Draw(card)
        f = font("Black", 84)
        parts = [("p = ", WHITE), ("sigmoid", PURPLE), ("(", WHITE), ("mx + b", YELLOW), (")", WHITE)]
        x = (W - sum(f.getlength(p) for p, _ in parts)) / 2
        for p, c in parts:
            cd.text((x, 82), p, font=f, fill=c, anchor="lm")
            x += f.getlength(p)
        text_center(cd, (W / 2, 160), "level 1's line, squashed into 0–100%", font("Bold", 30), MUTED)
        paste_scaled(frame, add_glow(card, radius=10, strength=1.2), (W / 2, 355), scale=0.6 + 0.4 * g, alpha=clamp(g))

    # Data scene: row labels.
    if T.start(DATA) <= t < T.start(LINE) + 0.3:
        t_snap = T.word(DATA, "pass")
        a = clamp((t - t_snap - 0.2) / 0.3) * (1 - clamp((t - T.start(LINE)) / 0.3))
        if a > 0:
            paste_scaled(frame, tag("PASS", (10, 30, 20), GREEN + (255,), 36), (px(0.9), py(1.17)), alpha=a)
            paste_scaled(frame, tag("FAIL", WHITE, PINK + (255,), 36), (px(0.9), py(-0.17)), alpha=a)

    # Straight-line scene: impossible values.
    t_h = T.word(LINE, "hundred")
    if T.start(LINE) <= t < T.word(BEND, "squash") + 0.4:
        a = 1 - clamp((t - T.word(BEND, "squash")) / 0.4)
        if t >= t_h:
            img = tag(f"{pct(OLS_M * 10 + OLS_B)} ?!", WHITE, (15, 18, 40, 240), 48, outline=PINK + (255,))
            pop_in(frame, img, (px(7.7), py(1.2)), t, t_h) if a >= 1 else paste_scaled(frame, img, (px(7.7), py(1.2)), alpha=a)
        t_i = T.word(LINE, "impossible")
        if t >= t_i:
            img = tag(f"{pct(OLS_B)} ?!", WHITE, (15, 18, 40, 240), 48, outline=PINK + (255,))
            pop_in(frame, img, (px(2.2), py(-0.2)), t, t_i) if a >= 1 else paste_scaled(frame, img, (px(2.2), py(-0.2)), alpha=a)
            imp = tag("IMPOSSIBLE", WHITE, PINK + (255,), 44)
            if a >= 1:
                pop_in(frame, imp, (px(6.6), py(-0.2)), t, t_i + 0.15)
            else:
                paste_scaled(frame, imp, (px(6.6), py(-0.2)), alpha=a)

    # Probe readout.
    t_ph = T.word(HEIGHT, "height")
    t_rand = T.word(HEIGHT, "random")
    st = curve(t)
    if t_ph <= t < t_rand and st:
        xq = lerp(1.5, 8.0, ease_in_out((t - t_ph) / max(0.5, t_rand - t_ph - 0.2)))
        yq = float(curve_fn(st)(xq))
        paste_scaled(frame, tag(pct(yq), (20, 20, 30), WHITE + (255,), 40), (px(xq), py(yq) - 70))
    if t_rand <= t < T.word(PENALTY, "punishes"):
        a = 1 - clamp((t - T.start(PENALTY)) / 0.3)
        img = tag("RANDOM GUESS", WHITE, PURPLE + (255,), 40)
        if t < t_rand + 0.3:
            pop_in(frame, img, (px(5), py(1.2)), t, t_rand)
        else:
            paste_scaled(frame, img, (px(5), py(1.2)), alpha=a)

    # Worst confident mistake.
    t_s = T.word(PENALTY, "super")
    if t_s <= t < T.start(TRAIN) + 0.3:
        a = 1 - clamp((t - T.start(TRAIN)) / 0.3)
        img = tag(f"{WORST_SURE}% sure: FAIL", WHITE, (15, 18, 40, 240), 40, outline=PINK + (255,))
        c = (px(X[WORST]) - 190, py(1.27))
        pop_in(frame, img, c, t, t_s) if a >= 1 else paste_scaled(frame, img, c, alpha=a)
        t_w = T.word(PENALTY, "wrong")
        if t >= t_w:
            img2 = tag("PASSED. WRONG!", WHITE, PINK + (255,), 36)
            c2 = (px(X[WORST]) - 190, py(1.12))
            pop_in(frame, img2, c2, t, t_w) if a >= 1 else paste_scaled(frame, img2, c2, alpha=a)

    # Training readout.
    if T.start(TRAIN) <= t < T.start(GATE) + 0.1:
        w, c = train_params(t)
        a = clamp((t - T.start(TRAIN)) / 0.3) * (1 - clamp((t - T.start(GATE)) / 0.1))
        step = int(round(train_progress(t) * (len(TRAJ) - 1)))
        draw_training_readout(frame, PLOT.x0, PLOT.x1, PLOT.y_bot + 140, step, len(TRAJ) - 1,
                              f"m {w:4.2f}   b {c:5.2f}", a)

    # Gate labels (make room for the prediction once it starts).
    t45 = T.word(PREDICT, "four")
    if t < t45:
        if t >= T.word(GATE, "left"):
            pop_in(frame, tag("FAIL", WHITE, PINK + (255,), 52), (px(BOUNDARY / 2), py(0.86)), t, T.word(GATE, "left"))
        if t >= T.word(GATE, "right"):
            pop_in(frame, tag("PASS", (10, 30, 20), GREEN + (255,), 52), (px((BOUNDARY + 10) / 2), py(0.2)), t,
                   T.word(GATE, "right"))
        t_dec = T.word(GATE, "decision")
        if t >= t_dec:
            pop_in(frame, tag(f"{BOUNDARY:.1f} h", (20, 20, 30), YELLOW + (255,), 36), (px(BOUNDARY), AX_Y + 36),
                   t, t_dec)

    # Prediction tag + bubble.
    if t >= t45:
        pop_in(frame, tag("4.5 h", (20, 20, 30), YELLOW + (255,), 36), (px(X_PRED), AX_Y + 36), t, t45)
    t_n = T.word(PREDICT, num_to_words(P_PRED).split("-")[0][:5])
    if t >= t_n - 0.1:
        g = ease_back((t - t_n + 0.1) / 0.35)
        bub = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
        bd = ImageDraw.Draw(bub)
        rounded_pill(bd, (10, 10, 290, 190), fill=(15, 18, 40, 245), outline=YELLOW + (255,), width=5, radius=28)
        text_center(bd, (150, 62), f"{P_PRED}%", font("Black", 66), YELLOW)
        text_center(bd, (150, 112), "chance to pass", font("Bold", 26), MUTED)
        if t >= T.word(PREDICT, "pass"):
            text_center(bd, (150, 158), "→ PASS", font("Black", 38), GREEN)
        paste_scaled(frame, bub, (px(X_PRED) - 180, py(0.75)), scale=g, alpha=clamp(g))

    draw_level_cleared(frame, t, T.start(CTA), 2, "DECISION TREES", T.word(CTA, "next"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    frame.alpha_composite(draw_plot_layer(t), (PLOT.layer[0], PLOT.layer[1]))
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
    mix(out, sfx_whoosh(0.7, 0.22), 0.0)
    t_b = T.word(HOOK, "break")
    mix(out, sfx_whoosh(0.5, 0.3, seed=9), t_b)
    for k in range(5):
        mix(out, sfx_blip(900 - 120 * k, 0.08, 0.12), t_b + 0.05 + 0.07 * k)
    span = (T.word(DATA, "just") - T.start(DATA)) * 0.9
    for k in range(len(X)):
        mix(out, sfx_blip(600 + 30 * k, 0.08, 0.12), T.start(DATA) + span * k / len(X))
    mix(out, sfx_boing(0.5, 0.18), T.word(DATA, "pass"))
    mix(out, sfx_whoosh(0.6, 0.15, seed=2), T.start(LINE) + 0.1)
    mix(out, sfx_buzz(0.16, 110, 0.14), T.word(LINE, "hundred"))
    mix(out, sfx_buzz(0.16, 110, 0.14), T.word(LINE, "impossible"))
    mix(out, sfx_buzz(0.3, 92, 0.14), T.word(LINE, "impossible") + 0.22)
    mix(out, sfx_boing(0.7, 0.22), T.word(BEND, "squash"))
    mix(out, sfx_blip(1046, 0.25, 0.16), T.word(BEND, "sigmoid"))
    t_r = T.word(HEIGHT, "random")
    for k in range(6):
        mix(out, sfx_blip(1400 - 160 * k, 0.04, 0.08), t_r + 0.05 * k)
    t_p = T.word(PENALTY, "punishes")
    for k in range(12):
        mix(out, sfx_blip(1500, 0.03, 0.06), t_p + k * 0.065)
    mix(out, sfx_buzz(0.4, 70, 0.2), T.word(PENALTY, "huge"))
    mix(out, sfx_whoosh(0.9, 0.2, seed=4), T.start(TRAIN) + 0.1)
    for k in range(0, len(TRAJ), 6):
        mix(out, sfx_blip(700 + 12 * k, 0.05, 0.07),
            T.start(TRAIN) + 0.2 + (T.end(TRAIN) - T.start(TRAIN) - 0.1) * k / len(TRAJ))
    mix(out, sfx_blip(1320, 0.3, 0.16), T.word(GATE, "decision"))
    mix(out, sfx_blip(700, 0.15, 0.14), T.word(GATE, "left"))
    mix(out, sfx_blip(1046, 0.15, 0.14), T.word(GATE, "right"))
    mix(out, sfx_blip(1320, 0.35, 0.2), T.word(PREDICT, "four") + 0.55)
    mix(out, sfx_blip(1568, 0.35, 0.18), T.word(PREDICT, "pass"))
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines)
    STATIC = build_static(PLOT, list(range(0, 11, 2)), [0, 0.25, 0.5, 0.75, 1.0], "HOURS STUDIED",
                          "CHANCE TO PASS", ytick_fmt=pct, xlabels_y=PLOT.y_bot)
    TITLE_IMG = build_title("LOGISTIC REGRESSION", "ML AS A GAME  ·  EP 2")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = T.word(PENALTY, "huge") + 0.5
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    text_center(d, (W / 2, 1420), "PASS OR FAIL?", font("Black", 116), YELLOW, stroke=8)
    text_center(d, (W / 2, 1570), "Logistic Regression, explained", font("Black", 58), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s | straight line: {pct(OLS_B)} at 0h, {pct(OLS_M * 10 + OLS_B)} at 10h")
    print(f"fit w={W_FIT:.3f} c={C_FIT:.3f} boundary={BOUNDARY:.2f}h | loss {LOSS0:.1f} -> {LOSS_FIT:.1f} | "
          f"worst: {X[WORST]}h, {WORST_SURE}% sure fail | 4.5h -> {P_PRED}%")
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
    out = os.path.join(HERE, "logistic_regression_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
