"""ML as a Game, Level 7: Neural Networks (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> neural_networks_short.mp4 + thumbnail.png
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
    add_glow, build_confetti, build_plain, build_static, build_title, caption_chunks, clamp, draw_caption, draw_hud_pills,
    draw_level_cleared, ease_back, ease_out, encode, font, lerp, lerp_color, mix, music_bed,
    paste_scaled, pop_in, sfx_blip, sfx_boing, sfx_buzz, sfx_levelup, sfx_whoosh, tag, text_center,
    voiceover, write_wav,
)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- the example neuron (scenes 2-3)

EX_IN = (6.0, 7.0)  # hours studied, hours slept
EX_W = (0.8, 0.5)
EX_B = -5.0
EX_PROD = (EX_IN[0] * EX_W[0], EX_IN[1] * EX_W[1])
EX_Z = EX_PROD[0] + EX_PROD[1] + EX_B
EX_P = round(100 * float(sigmoid(EX_Z)))

# ---------------------------------------------------------------- data: pass only with the right amount of sleep

_r = np.random.default_rng(7)
X = _r.uniform(0.5, 9.5, (40, 2))
Y = ((X[:, 1] >= 5) & (X[:, 1] <= 8.5)).astype(float)
N = len(Y)
BAND = (5.0, 8.5)

# One neuron = logistic regression (Newton's method) on raw hours.
_A = np.column_stack([X, np.ones(N)])
SINGLE = np.zeros(3)
for _ in range(100):
    _p = sigmoid(_A @ SINGLE)
    _hess = (_A * (_p * (1 - _p))[:, None]).T @ _A + 1e-6 * np.eye(3)
    SINGLE -= np.linalg.solve(_hess, _A.T @ (_p - Y))
SINGLE_ERR = int(((sigmoid(_A @ SINGLE) > 0.5) != Y).sum())

# A 2-2-1 network (sigmoid everywhere), trained with full-batch gradient descent + momentum.
NET_SEED = 13
STEPS = 6000


def norm(p):
    return (np.asarray(p) - 5.0) / 3.0  # inputs scaled to about -1.5..1.5 for training


def forward(params, pts):
    W1, b1, W2, b2 = params
    h = sigmoid(norm(pts) @ W1 + b1)
    return h, sigmoid(h @ W2 + b2)


def train():
    rng = np.random.default_rng(NET_SEED)
    W1, b1, W2, b2 = rng.normal(0, 0.5, (2, 2)), rng.normal(0, 0.5, 2), rng.normal(0, 0.5, 2), 0.0
    vel = [np.zeros((2, 2)), np.zeros(2), np.zeros(2), 0.0]
    xn = norm(X)
    snaps = [(W1.copy(), b1.copy(), W2.copy(), b2)]
    for _ in range(STEPS):
        h = sigmoid(xn @ W1 + b1)
        o = sigmoid(h @ W2 + b2)
        do = (o - Y) / N
        dh = np.outer(do, W2) * h * (1 - h)
        grads = [xn.T @ dh, dh.sum(0), h.T @ do, do.sum()]
        vel = [0.9 * v - 0.5 * g for v, g in zip(vel, grads)]
        W1, b1, W2, b2 = W1 + vel[0], b1 + vel[1], W2 + vel[2], b2 + vel[3]
        snaps.append((W1.copy(), b1.copy(), W2.copy(), b2))
    return snaps


_all = train()
SNAP_STEPS = np.unique(np.round(np.geomspace(1, STEPS + 1, 90)).astype(int) - 1)
SNAPS = [_all[i] for i in SNAP_STEPS]
del _all
SNAP_ERR = [int(((forward(s, X)[1] > 0.5) != Y).sum()) for s in SNAPS]
FINAL = SNAPS[-1]


def hidden_threshold(params, j):
    """Hours slept where hidden neuron j flips (at 5 h studied) and whether it fires above or below."""
    W1, b1, _, _ = params
    return 5 - 3 * b1[j] / W1[1, j], W1[1, j] > 0


H1_AT, H1_UP = hidden_threshold(FINAL, 0)
H2_AT, H2_UP = hidden_threshold(FINAL, 1)
W2F, B2F = FINAL[2], FINAL[3]
assert SNAP_ERR[-1] == 0 and SINGLE_ERR >= 6
assert H1_UP and abs(H1_AT - BAND[0]) < 0.4, (H1_AT, H1_UP)  # neuron 1: "slept enough?"
assert not H2_UP and abs(H2_AT - BAND[1]) < 0.4, (H2_AT, H2_UP)  # neuron 2: "not too much?"
assert W2F[0] > 0 and W2F[1] > 0 and W2F.sum() + B2F > 0 and max(W2F) + B2F < 0  # output = AND
for j in range(2):  # both hidden lines are (nearly) horizontal: they only care about sleep
    assert abs(FINAL[0][1, j]) > 6 * abs(FINAL[0][0, j])

SCRIPT = [
    "Every AI you've heard of is built from one tiny piece: the neuron.",
    "A neuron takes inputs, like hours studied and hours slept, and multiplies each one by a weight.",
    "Then it adds them up and squashes the total into zero to one hundred percent. That's level two's sigmoid!",
    "Now a trickier class. You pass only with the right amount of sleep. Too little or too much? Fail.",
    "One neuron can only draw one straight line. It can't catch this band.",
    "So we stack them: two hidden neurons, then one output neuron. That's a neural network.",
    "Training nudges every weight, like level one, until the mistakes shrink.",
    "Look inside. Neuron one learned: slept enough? Neuron two: not too much? "
    "The output says pass only when both say yes.",
    "Add more neurons and more layers, and it can draw almost any shape. That's deep learning.",
    "Level seven cleared. Next: backpropagation, how networks actually learn. Follow for level eight.",
]
HOOK, NEURON, SQUASH, DATA, ONE, STACK, TRAIN, INSIDE, DEEP, CTA = range(10)

# ---------------------------------------------------------------- layout

PLOT = PlotBox(290, 850, 440, 1000, (0, 10), (0, 10), layer=(200, 380, 940, 1040))
px, py, L = PLOT.px, PLOT.py, PLOT.L
SS = PLOT.ss
CAPTION_Y = 1495
TOP_TAG_Y = PLOT.y_top - 32
GP = 90  # probability-map grid

# Small network diagram under the plot.
IN_NODES = [(190, 1185), (190, 1365)]
HID_NODES = [(520, 1185), (520, 1365)]
OUT_NODE = (790, 1275)
NODE_R = 40
# Big single-neuron explainer (scenes 1-3), drawn where the plot will later be.
BIG_IN = [(230, 780), (230, 1120)]
BIG_NEURON = (600, 950)
BIG_OUT = (900, 950)

T: Timeline = None
CHUNKS = []
STATIC = None
PLAIN = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0


def word(i, prefix, nth=0):
    return T.word(i, prefix, nth)


# ---------------------------------------------------------------- state

def train_index(t):
    """Which training snapshot is shown (None before training starts)."""
    t0, t1 = word(TRAIN, "training"), T.end(TRAIN)
    if t < T.start(STACK):
        return None
    if t < t0:
        return 0
    return int(round(clamp((t - t0) / (t1 - t0)) * (len(SNAPS) - 1)))


def model_at(t):
    """('single', w) / ('net', params) / None, for the plot and the diagram."""
    if t < word(ONE, "one"):
        return None
    if t < T.start(STACK):
        return ("single", SINGLE)
    return ("net", SNAPS[train_index(t)])


def error_display(t):
    if t < T.start(NEURON) - 0.1:
        return f"{(int(t * FPS) * 7919) % 30 + 1}", PINK
    m = model_at(t)
    if m is None:
        return "—", MUTED
    if m[0] == "single":
        return f"{SINGLE_ERR}", PINK
    e = SNAP_ERR[train_index(t)]
    return f"{e}", lerp_color(PINK, GREEN, clamp(1 - e / SNAP_ERR[0]) ** 2)


def prob_grid(m):
    c = (np.arange(GP) + 0.5) / GP * 10
    gx, gy = np.meshgrid(c, c[::-1])
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    if m[0] == "single":
        p = sigmoid(np.column_stack([pts, np.ones(len(pts))]) @ m[1])
    else:
        p = forward(m[1], pts)[1]
    return p.reshape(GP, GP)


# ---------------------------------------------------------------- drawing helpers

def weight_style(w, scale):
    col = CYAN if w >= 0 else PINK
    return col, max(2, min(14, abs(w) * scale))


def edge(d, a, b, col, width, alpha=255):
    d.line([a, b], fill=col + (int(alpha),), width=int(width))


def node(d, c, r, label=None, fill=(18, 22, 48), outline=WHITE, size=30, width=5, txt=WHITE):
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=fill + (255,), outline=outline + (255,), width=width)
    if label:
        text_center(d, c, label, font("Black", size), txt)


def signal_dots(d, a, b, t, speed=1.2, n=2, col=YELLOW, r=7):
    for k in range(n):
        f = (t * speed + k / n) % 1
        x, y = lerp(a[0], b[0], f), lerp(a[1], b[1], f)
        d.ellipse([x - r, y - r, x + r, y + r], fill=col + (255,))


def mini_sigmoid(d, c, size, col, marker=None):
    """A tiny S-curve drawn inside a node; marker = z value to dot."""
    pts = []
    for i in range(41):
        z = -6 + 12 * i / 40
        pts.append((c[0] - size + 2 * size * i / 40, c[1] + size * 0.7 - 1.4 * size * float(sigmoid(z))))
    d.line(pts, fill=col + (255,), width=5)
    if marker is not None:
        zc = max(-6, min(6, marker))
        mx = c[0] - size + 2 * size * (zc + 6) / 12
        my = c[1] + size * 0.7 - 1.4 * size * float(sigmoid(zc))
        d.ellipse([mx - 9, my - 9, mx + 9, my + 9], fill=YELLOW + (255,), outline=WHITE + (255,), width=2)


# ---------------------------------------------------------------- scenes 1-3: the big neuron explainer

def draw_explainer(frame, t):
    t_end = T.start(DATA)
    if t >= t_end + 0.4:
        return
    a = 1 - clamp((t - t_end) / 0.4)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)

    if t < T.start(NEURON):  # hook: a small glowing network with signals flowing
        cols = [[(220, 800), (220, 1100)], [(470, 720), (470, 950), (470, 1180)], [(720, 800), (720, 1100)], [(920, 950)]]
        for la, lb in zip(cols, cols[1:]):
            for p in la:
                for q in lb:
                    edge(d, p, q, (70, 80, 130), 4)
                    signal_dots(d, p, q, t + p[1] * 0.001 + q[1] * 0.002, speed=0.9, n=1, col=CYAN, r=6)
        pulse = 1 + 0.08 * math.sin(t * 7)
        for layer in cols:
            for p in layer:
                node(d, p, 38 * pulse, fill=(18, 22, 48), outline=CYAN)
    else:
        t_mul = word(NEURON, "multiplies")
        t_add = word(SQUASH, "adds")
        t_sq = word(SQUASH, "squashes")
        labels = [f"studied {EX_IN[0]:g} h", f"slept {EX_IN[1]:g} h"]
        g_in = ease_out((t - T.start(NEURON) - 0.1) / 0.4)
        for k in range(2):
            col = YELLOW if t >= t_mul else MUTED
            edge(d, BIG_IN[k], BIG_NEURON, col, 10 if t >= t_mul else 6)
            if t_mul <= t < t_add + 0.4:
                signal_dots(d, BIG_IN[k], BIG_NEURON, t, speed=0.8, n=2, col=YELLOW, r=9)
        edge(d, BIG_NEURON, BIG_OUT, GREEN if t >= t_sq + 0.5 else (70, 80, 130), 10)
        for k in range(2):
            node(d, BIG_IN[k], 62 * g_in, fill=(18, 22, 48), outline=CYAN)
            if g_in > 0.6:
                text_center(d, BIG_IN[k], f"{EX_IN[k]:g}", font("Black", 52), WHITE)
                text_center(d, (BIG_IN[k][0], BIG_IN[k][1] + 92), labels[k], font("Bold", 32), MUTED)
        node(d, BIG_NEURON, 120, fill=(24, 20, 58), outline=PURPLE, width=8)
        if t < t_add:
            text_center(d, BIG_NEURON, "?", font("Black", 90), MUTED)
        elif t < t_sq:
            text_center(d, (BIG_NEURON[0], BIG_NEURON[1] - 26), "Σ", font("Black", 70), YELLOW)
            text_center(d, (BIG_NEURON[0], BIG_NEURON[1] + 44), f"= {EX_Z:.1f}", font("Black", 44), WHITE)
        else:
            mini_sigmoid(d, BIG_NEURON, 78, PURPLE, marker=EX_Z if t >= t_sq + 0.4 else None)
        if t >= t_sq + 0.6:
            g = ease_back((t - t_sq - 0.6) / 0.35)
            node(d, BIG_OUT, 70 * g, fill=GREEN, outline=WHITE)
            if g > 0.6:
                text_center(d, BIG_OUT, f"{EX_P}%", font("Black", 46), (12, 28, 18))
                text_center(d, (BIG_OUT[0], BIG_OUT[1] + 104), "chance to pass", font("Bold", 28), MUTED)
    lay = add_glow(lay, radius=10, strength=1.0)
    if a < 1:
        arr = np.asarray(lay).copy()
        arr[..., 3] = (arr[..., 3] * a).astype(np.uint8)
        lay = Image.fromarray(arr, "RGBA")
    frame.alpha_composite(lay)

    # Arithmetic labels (crisp, on top).
    if T.start(NEURON) <= t < t_end:
        t_w = word(NEURON, "weight")
        for k in range(2):
            mid = ((BIG_IN[k][0] + BIG_NEURON[0]) / 2, (BIG_IN[k][1] + BIG_NEURON[1]) / 2 + (-46 if k == 0 else 46))
            if word(NEURON, "multiplies") <= t < t_w:
                pop_in(frame, tag(f"× {EX_W[k]:g}", (20, 20, 30), YELLOW + (255,), 36), mid, t,
                       word(NEURON, "multiplies") + 0.15 * k)
            if t >= t_w:
                pop_in(frame, tag(f"{EX_IN[k]:g} × {EX_W[k]:g} = {EX_PROD[k]:g}", (20, 20, 30), YELLOW + (255,), 34),
                       mid, t, t_w + 0.2 * k)
        t_add = word(SQUASH, "adds")
        if t >= t_add:
            pop_in(frame, tag(f"{EX_PROD[0]:g} + {EX_PROD[1]:g} − {abs(EX_B):g} (bias) = {EX_Z:.1f}", WHITE,
                              (18, 22, 48, 240), 32, outline=PURPLE + (255,)), (BIG_NEURON[0], BIG_NEURON[1] - 240), t,
                   t_add)
        t_sig = word(SQUASH, "sigmoid")
        if t >= t_sig - 0.2:
            pop_in(frame, tag("LEVEL 2's SIGMOID!", (20, 20, 30), PURPLE + (255,), 34),
                   (BIG_NEURON[0] + 40, BIG_NEURON[1] + 220), t, t_sig - 0.2)
        if t >= word(NEURON, "multiplies"):
            paste_scaled(frame, tag("example weights", (20, 20, 30), MUTED + (255,), 24, pad=(12, 5)), (W / 2, 1290))


# ---------------------------------------------------------------- scenes 4+: plot + small diagram

def draw_plot_layer(t):
    lay = PLOT.new_layer()
    if t < T.start(DATA):
        return PLOT.finish_layer(lay)
    m = model_at(t)
    if m is not None:
        p = prob_grid(m)
        pink, green = np.array(PINK, np.float32), np.array(GREEN, np.float32)
        rgb = pink[None, None] * (1 - p[..., None]) + green[None, None] * p[..., None]
        img = Image.fromarray(np.dstack([rgb, np.full(p.shape, 62)]).astype(np.uint8), "RGBA")
        img = img.resize((int((PLOT.x1 - PLOT.x0) * SS), int((PLOT.y_bot - PLOT.y_top) * SS)), Image.BILINEAR)
        lay.alpha_composite(img, (int((PLOT.x0 - PLOT.layer[0]) * SS), int((PLOT.y_top - PLOT.layer[1]) * SS)))
    d = ImageDraw.Draw(lay)

    # Sweet-spot hint while introducing the data.
    t_r = word(DATA, "right")
    if t_r <= t < word(ONE, "one"):
        a = clamp((t - t_r) / 0.4)
        for yb in BAND:
            d.line([L(0, yb), L(10, yb)], fill=WHITE + (int(110 * a),), width=4)

    # The single neuron's straight line.
    if m is not None and m[0] == "single":
        w0, w1, w2 = SINGLE
        ya, yb = -(w2 + w0 * 0) / w1, -(w2 + w0 * 10) / w1
        g = ease_out((t - word(ONE, "one")) / 0.6)
        d.line([L(0, ya), L(lerp(0, 10, g), lerp(ya, yb, g))], fill=YELLOW + (255,), width=10)

    # Inside the trained network: each hidden neuron's line.
    t_one = word(INSIDE, "one")
    if t_one <= t < T.start(CTA):
        W1, b1, _, _ = FINAL
        for j, (col, tj) in enumerate(((CYAN, t_one), (ORANGE, word(INSIDE, "two:")))):
            if t < tj:
                continue
            g = ease_out((t - tj) / 0.5)

            def yline(x):
                return 5 - 3 * (b1[j] + W1[0, j] * (x - 5) / 3) / W1[1, j]
            d.line([L(0, yline(0)), L(lerp(0, 10, g), yline(lerp(0, 10, g)))], fill=col + (255,), width=9)

    # Students.
    for i in range(N):
        s = ease_back((t - T.start(DATA) - 0.012 * i) / 0.3)
        if s <= 0:
            continue
        cx, cy = L(*X[i])
        col = GREEN if Y[i] else PINK
        r = 14 * SS * s
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (255,))
        d.ellipse([cx - r * 0.45, cy - r * 0.45, cx + r * 0.45, cy + r * 0.45], fill=WHITE + (255,))
    return PLOT.finish_layer(lay, glow=10, strength=1.2)


def draw_diagram(frame, t):
    t0 = T.start(DATA)
    if t < t0:
        return
    a = clamp((t - t0) / 0.4)
    lay = Image.new("RGBA", (W, 420), (0, 0, 0, 0))  # covers y 1080..1500
    oy = 1080
    d = ImageDraw.Draw(lay)

    def P(p):
        return (p[0], p[1] - oy)

    m = model_at(t)
    deep = t >= word(DEEP, "more")
    t_stack = word(STACK, "stack")
    if deep:  # illustrative bigger network
        g = ease_out((t - word(DEEP, "more")) / 0.8)
        layers = [[(150, 1185), (150, 1365)],
                  [(350, 1150 + 55 * k) for k in range(5)],
                  [(560, 1150 + 55 * k) for k in range(5)],
                  [(780, 1275)]]
        rr = np.random.default_rng(3)
        for la, lb in zip(layers, layers[1:]):
            for p in la:
                for q in lb:
                    w = rr.normal()
                    edge(d, P(p), P(q), CYAN if w > 0 else PINK, 2 + 3 * abs(w) * g, 200 * g)
        for la in layers:
            for p in la:
                node(d, P(p), 22 if len(la) > 2 else 30, fill=(18, 22, 48), outline=PURPLE)
        for la, lb in zip(layers, layers[1:]):
            for p in la[:2]:
                signal_dots(d, P(p), P(lb[len(lb) // 2]), t, speed=1.0, n=1, col=YELLOW, r=6)
    elif t < t_stack or m is None or m[0] == "single":  # one neuron
        mid = (HID_NODES[0][0], OUT_NODE[1])
        for k in range(2):
            col, wd = weight_style(SINGLE[k], 9)
            edge(d, P(IN_NODES[k]), P(mid), col if m else MUTED, wd if m else 5)
        edge(d, P(mid), P(OUT_NODE), MUTED, 5)
        node(d, P(mid), NODE_R + 6, fill=(24, 20, 58), outline=PURPLE, width=6)
        mini_sigmoid(d, P(mid), 26, PURPLE)
        node(d, P(OUT_NODE), NODE_R, label="%", fill=(18, 22, 48), outline=WHITE, size=28)
    else:  # 2-2-1 network
        g = ease_out((t - t_stack) / 0.6)
        W1, b1, W2, b2 = m[1]
        glow = set()
        if t >= word(INSIDE, "one"):
            glow.add(0)
        if t >= word(INSIDE, "two:"):
            glow.add(1)
        both = t >= word(INSIDE, "output")
        for i in range(2):
            for j in range(2):
                col, wd = weight_style(W1[i, j], 0.45)
                edge(d, P(IN_NODES[i]), P(HID_NODES[j]), col, wd * g, 255 * g)
        for j in range(2):
            col, wd = weight_style(W2[j], 0.45)
            edge(d, P(HID_NODES[j]), P(OUT_NODE), col, wd * g, 255 * g)
        if t >= word(TRAIN, "training"):
            for i in range(2):
                for j in range(2):
                    signal_dots(d, P(IN_NODES[i]), P(HID_NODES[j]), t + 0.3 * (i + 2 * j), speed=1.1, n=1, r=6)
            for j in range(2):
                signal_dots(d, P(HID_NODES[j]), P(OUT_NODE), t + 0.5 * j, speed=1.1, n=1, r=6)
        for j in range(2):
            ring = (CYAN, ORANGE)[j] if j in glow else PURPLE
            node(d, P(HID_NODES[j]), (NODE_R + 4) * g, fill=(24, 20, 58), outline=ring, width=8 if j in glow else 6)
            if g > 0.6:
                mini_sigmoid(d, P(HID_NODES[j]), 24, ring)
        node(d, P(OUT_NODE), (NODE_R + 6) * g, fill=(GREEN if both else (24, 20, 58)), outline=GREEN if both else PURPLE,
             width=7)
        if g > 0.6:
            mini_sigmoid(d, P(OUT_NODE), 26, (12, 28, 18) if both else PURPLE)
    if not deep:
        for k, lab in enumerate(("studied", "slept")):
            node(d, P(IN_NODES[k]), NODE_R - 4, fill=(18, 22, 48), outline=CYAN)
            text_center(d, (IN_NODES[k][0] - 62, IN_NODES[k][1] - oy), lab, font("Bold", 28), MUTED, anchor="rm")
    if a < 1:
        arr = np.asarray(lay).copy()
        arr[..., 3] = (arr[..., 3] * a).astype(np.uint8)
        lay = Image.fromarray(arr, "RGBA")
    frame.alpha_composite(add_glow(lay, radius=8, strength=0.9), (0, oy))


def draw_overlays(frame, t):
    draw_hud_pills(frame, 7, *error_display(t))
    frame.alpha_composite(TITLE_IMG, (0, 190))

    banners = [
        (T.start(HOOK), T.start(NEURON), "THE BUILDING BLOCK OF AI", PURPLE, WHITE),
        (T.start(NEURON), T.start(SQUASH), "STEP 1 · MULTIPLY BY WEIGHTS", CYAN, (12, 14, 30)),
        (word(SQUASH, "adds"), word(SQUASH, "squashes"), "STEP 2 · ADD THEM UP", CYAN, (12, 14, 30)),
        (word(SQUASH, "squashes"), T.start(DATA), "STEP 3 · SQUASH TO 0–100%", CYAN, (12, 14, 30)),
        (T.start(DATA), word(DATA, "right"), "A TRICKIER CLASS", MUTED, (12, 14, 30)),
        (word(DATA, "right"), T.start(ONE), "PASS = SLEEP SWEET SPOT", GREEN, (12, 28, 18)),
        (word(ONE, "one"), T.start(STACK), f"1 NEURON = 1 STRAIGHT LINE · {SINGLE_ERR} WRONG", PINK, WHITE),
        (word(STACK, "stack"), T.start(TRAIN), "2 HIDDEN NEURONS → 1 OUTPUT", PURPLE, WHITE),
        (T.start(INSIDE), word(INSIDE, "one"), "LOOK INSIDE", PURPLE, WHITE),
        (word(INSIDE, "output"), T.start(DEEP), "BOTH SAY YES → PASS", GREEN, (12, 28, 18)),
        (word(DEEP, "more"), T.start(CTA), "MORE NEURONS + LAYERS = DEEP LEARNING", PURPLE, WHITE),
    ]
    ti = train_index(t)
    if T.start(TRAIN) <= t < T.start(INSIDE):
        step = int(SNAP_STEPS[ti]) if ti is not None else 0
        paste_scaled(frame, tag(f"TRAINING · STEP {step:,} · {SNAP_ERR[ti]} WRONG", (20, 20, 30), YELLOW + (255,), 32),
                     (W / 2, TOP_TAG_Y))
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 32), (W / 2, TOP_TAG_Y), t, a)

    # Hidden-neuron meaning labels on the plot and the diagram.
    t_one, t_two = word(INSIDE, "one"), word(INSIDE, "two:")
    if t_one <= t < word(DEEP, "more"):
        pop_in(frame, tag("① slept enough?", (12, 14, 30), CYAN + (255,), 30), (px(3.0), py(H1_AT) + 34), t, t_one)
        pop_in(frame, tag("① slept enough?", (12, 14, 30), CYAN + (255,), 26, pad=(14, 6)),
               (HID_NODES[0][0] + 150, HID_NODES[0][1] - 52), t, t_one)
    if t_two <= t < word(DEEP, "more"):
        pop_in(frame, tag("② not too much?", (12, 14, 30), ORANGE + (255,), 30), (px(7.4), py(H2_AT) - 34), t, t_two)
        pop_in(frame, tag("② not too much?", (12, 14, 30), ORANGE + (255,), 26, pad=(14, 6)),
               (HID_NODES[1][0] + 150, HID_NODES[1][1] + 52), t, t_two)
    t_out = word(INSIDE, "output")
    if t_out <= t < word(DEEP, "more"):
        pop_in(frame, tag("① AND ② → PASS", (12, 28, 18), GREEN + (255,), 28, pad=(14, 6)),
               (OUT_NODE[0], OUT_NODE[1] - 78), t, t_out)
    if word(ONE, "band") <= t < T.start(STACK):
        pop_in(frame, tag("can't bend!", WHITE, PINK + (255,), 30), (px(7.6), py(1.0)), t, word(ONE, "band"))

    draw_level_cleared(frame, t, T.start(CTA), 7, "BACKPROPAGATION", word(CTA, "next:"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    t_d = T.start(DATA)
    if t < t_d:
        frame = PLAIN.copy()
    elif t < t_d + 0.4:
        frame = Image.blend(PLAIN, STATIC, clamp((t - t_d) / 0.4))
    else:
        frame = STATIC.copy()
    frame.alpha_composite(draw_plot_layer(t), (PLOT.layer[0], PLOT.layer[1]))
    draw_explainer(frame, t)
    draw_diagram(frame, t)
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
    for k in range(int(T.start(NEURON) * 4)):
        mix(out, sfx_blip(900 + 100 * (k % 4), 0.03, 0.06), k / 4)
    mix(out, sfx_blip(880, 0.1, 0.13), word(NEURON, "multiplies"))
    mix(out, sfx_blip(1046, 0.1, 0.13), word(NEURON, "weight"))
    mix(out, sfx_blip(1175, 0.12, 0.14), word(SQUASH, "adds"))
    mix(out, sfx_boing(0.5, 0.16), word(SQUASH, "squashes"))
    mix(out, sfx_blip(1568, 0.3, 0.18), word(SQUASH, "squashes") + 0.6)
    mix(out, sfx_whoosh(0.6, 0.16, seed=5), T.start(DATA))
    for i in range(0, N, 3):
        mix(out, sfx_blip(600 + 15 * i, 0.04, 0.07), T.start(DATA) + 0.012 * i)
    mix(out, sfx_buzz(0.3, 100, 0.16), word(ONE, "band"))
    mix(out, sfx_boing(0.6, 0.18), word(STACK, "stack"))
    t0, t1 = word(TRAIN, "training"), T.end(TRAIN)
    for k in range(24):
        mix(out, sfx_blip(600 + 30 * k, 0.03, 0.06), lerp(t0, t1, k / 24))
    mix(out, sfx_blip(1046, 0.15, 0.15), word(INSIDE, "one"))
    mix(out, sfx_blip(1320, 0.15, 0.15), word(INSIDE, "two:"))
    mix(out, sfx_blip(1568, 0.3, 0.18), word(INSIDE, "output"))
    mix(out, sfx_whoosh(0.9, 0.2, seed=8), word(DEEP, "more"))
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, PLAIN, TITLE_IMG, CONFETTI, DURATION
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12, max_px=800)
    STATIC = build_static(PLOT, list(range(0, 11, 2)), list(range(0, 11, 2)), "HOURS STUDIED", "HOURS SLEPT")
    PLAIN = build_plain()
    TITLE_IMG = build_title("NEURAL NETWORKS", "ML AS A GAME  ·  EP 7")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = word(INSIDE, "both") + 0.6
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "HOW NEURONS THINK", font("Black", 90), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "Neural Networks, explained", font("Black", 56), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s | example neuron: {EX_IN[0]}x{EX_W[0]}={EX_PROD[0]}, {EX_IN[1]}x{EX_W[1]}="
          f"{EX_PROD[1]}, z={EX_Z:.1f} -> {EX_P}%")
    print(f"{N} students, {int(Y.sum())} pass | single neuron errors {SINGLE_ERR} | net errors "
          f"{SNAP_ERR[0]} -> {SNAP_ERR[-1]} over {STEPS} steps ({len(SNAPS)} snapshots)")
    print(f"hidden 1 fires when slept {'>' if H1_UP else '<'} {H1_AT:.2f}; hidden 2 fires when slept "
          f"{'>' if H2_UP else '<'} {H2_AT:.2f}; output weights {np.round(W2F, 1).tolist()} bias {B2F:.1f}")
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
    out = os.path.join(HERE, "neural_networks_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
