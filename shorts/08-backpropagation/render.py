"""ML as a Game, Level 8: Backpropagation (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> backprop_short.mp4 + thumbnail.png
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
    CYAN, FPS, GREEN, H, MUTED, PINK, PURPLE, SR, W, WHITE, YELLOW, Timeline, add_glow, build_confetti,
    build_plain, build_title, caption_chunks, clamp, draw_caption, draw_hud_pills, draw_level_cleared, ease_back,
    ease_in_out, ease_out, encode, font, lerp, lerp_color, mix, music_bed, num_to_words, pop_in, sfx_blip,
    sfx_boing, sfx_buzz, sfx_levelup, sfx_whoosh, tag, text_center, voiceover, write_wav,
)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# ---------------------------------------------------------------- the smallest network: x -> h -> o

X_IN, ANSWER, BIAS = 8.0, 1.0, -1.5  # slept 8 h; they passed; fixed output bias
W1, W2 = 0.10, 0.30


def forward(w1, w2):
    h = float(sigmoid(w1 * X_IN))
    return h, float(sigmoid(w2 * h + BIAS))


def loss(w1, w2):
    o = forward(w1, w2)[1]
    return -(ANSWER * math.log(o) + (1 - ANSWER) * math.log(1 - o))


HID, OUT = forward(W1, W2)
BLAME_OUT = OUT - ANSWER  # log-loss + sigmoid output: blame is simply prediction - answer
BLAME_W2 = BLAME_OUT * HID  # x what flowed into w2
SLOPE = HID * (1 - HID)  # slope of the hidden neuron's squash
BLAME_HID = BLAME_OUT * W2 * SLOPE
BLAME_W1 = BLAME_HID * X_IN
LR = 1.0
NEW_W1, NEW_W2 = W1 - LR * BLAME_W1, W2 - LR * BLAME_W2
NEW_HID, NEW_OUT = forward(NEW_W1, NEW_W2)

# The on-screen blame really is the gradient: check against finite differences.
_e = 1e-6
FD_W1 = (loss(W1 + _e, W2) - loss(W1 - _e, W2)) / (2 * _e)
FD_W2 = (loss(W1, W2 + _e) - loss(W1, W2 - _e)) / (2 * _e)
assert abs(FD_W1 - BLAME_W1) < 1e-6 and abs(FD_W2 - BLAME_W2) < 1e-6, (FD_W1, BLAME_W1, FD_W2, BLAME_W2)
assert BLAME_W1 < 0 and BLAME_W2 < 0 and NEW_OUT > OUT  # both weights go up; the prediction improves

# 25 nudges on this student.
TRAJ = [OUT]
_a, _b = W1, W2
for _ in range(25):
    _h, _o = forward(_a, _b)
    _d = _o - ANSWER
    _a, _b = _a - LR * _d * _b * _h * (1 - _h) * X_IN, _b - LR * _d * _h
    TRAJ.append(forward(_a, _b)[1])
assert all(b2 > a2 for a2, b2 in zip(TRAJ, TRAJ[1:]))

P0, P1 = round(100 * OUT), round(100 * NEW_OUT)
OFF = round(100 * (ANSWER - OUT))

SCRIPT = [
    "Your neural network guessed wrong. So which weight is to blame?",
    "Meet the smallest network: one input, one hidden neuron, one output, and two weights.",
    f"Forward pass: this student slept eight hours. The network says {num_to_words(P0)} percent chance to pass. "
    "But they passed!",
    f"So the output is off by {num_to_words(OFF)} points. That's its blame.",
    "Step one: the last weight's blame is the output's blame, times what flowed into it.",
    "Step two: pass the blame backward, through that weight and the hidden neuron's slope.",
    "Step three: the first weight gets its blame the same way. Blame flows backward. That's backpropagation!",
    f"Now nudge each weight against its blame, like level one. One nudge, and {num_to_words(P0)} percent becomes "
    f"{num_to_words(P1)}.",
    "Repeat that for every student, thousands of times, and the whole network learns.",
    "Level eight cleared. Next: how AI sees images. Follow for level nine.",
]
HOOK, MEET, FORWARD, OUTBLAME, STEP1, STEP2, STEP3, NUDGE, REPEAT, CTA = range(10)

# ---------------------------------------------------------------- layout

IN_P, HID_P, OUT_P = (160, 720), (540, 720), (905, 720)
IN_R, HID_R, OUT_R = 78, 108, 92
W1_MID = ((IN_P[0] + HID_P[0]) / 2, 720)
W2_MID = ((HID_P[0] + OUT_P[0]) / 2, 720)
LEDGER = (70, 1000, 1010, 1385)
CAPTION_Y = 1495
TOP_TAG_Y = 440

T: Timeline = None
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0


def word(i, prefix, nth=0):
    return T.word(i, prefix, nth)


def fmt(v, sign=False):
    s = f"{abs(v):.2f}"
    return ("−" if v < 0 else ("+" if sign else "")) + s


# ---------------------------------------------------------------- state

def nudge_progress(t):
    t0 = word(NUDGE, "nudge")
    return ease_in_out((t - t0 - 0.3) / 1.2) if t >= t0 else 0.0


def weights_at(t):
    g = nudge_progress(t)
    return lerp(W1, NEW_W1, g), lerp(W2, NEW_W2, g)


def shown_output(t):
    """Output % displayed: counts up after the nudge's forward replay."""
    t_one = word(NUDGE, "one", 1)
    if t < t_one:
        return OUT
    return lerp(OUT, NEW_OUT, ease_out((t - t_one - 0.2) / 1.0))


def ledger_rows(t):
    rows = [
        ("①", f"blame(output) = {OUT:.2f} − 1 = {fmt(BLAME_OUT)}", word(OUTBLAME, "off"), PINK),
        ("②", f"blame(w2) = {fmt(BLAME_OUT)} × {HID:.2f} = {fmt(BLAME_W2)}", word(STEP1, "times"), PINK),
        ("③", f"blame(hidden) = {fmt(BLAME_OUT)} × {W2:.2f} × {SLOPE:.2f} = {fmt(BLAME_HID)}",
         word(STEP2, "slope"), PINK),
        ("④", f"blame(w1) = {fmt(BLAME_HID)} × {X_IN:g} = {fmt(BLAME_W1)}", word(STEP3, "same"), PINK),
        ("⑤", f"nudge: w2 {W2:.2f} → {NEW_W2:.2f} · w1 {W1:.2f} → {NEW_W1:.2f}", word(NUDGE, "against"), GREEN),
    ]
    return [r for r in rows if t >= r[2]]


# ---------------------------------------------------------------- drawing helpers

def mini_sigmoid(d, c, size, col, marker=None, tangent=0.0):
    pts = []
    for i in range(41):
        z = -6 + 12 * i / 40
        pts.append((c[0] - size + 2 * size * i / 40, c[1] + size * 0.7 - 1.4 * size * float(sigmoid(z))))
    d.line(pts, fill=col + (255,), width=6)
    if marker is not None:
        zc = max(-6.0, min(6.0, marker))
        mx = c[0] - size + 2 * size * (zc + 6) / 12
        sz = float(sigmoid(zc))
        my = c[1] + size * 0.7 - 1.4 * size * sz
        if tangent > 0:  # the slope at this point
            dydx = 1.4 * sz * (1 - sz) * 12 / 2  # screen slope of the drawn curve
            ln = 60 * tangent
            dx = ln / math.sqrt(1 + dydx ** 2)
            d.line([(mx - dx, my + dydx * dx), (mx + dx, my - dydx * dx)], fill=YELLOW + (255,), width=6)
        d.ellipse([mx - 11, my - 11, mx + 11, my + 11], fill=YELLOW + (255,), outline=WHITE + (255,), width=3)


def dots(d, a, b, f, col, r=10, n=3, spread=0.12):
    """n dots travelling a->b, f in 0..1 = leading dot's position."""
    for k in range(n):
        g = f - k * spread
        if 0 <= g <= 1:
            d.ellipse([lerp(a[0], b[0], g) - r, lerp(a[1], b[1], g) - r, lerp(a[0], b[0], g) + r,
                       lerp(a[1], b[1], g) + r], fill=col + (255,))


def edge_end(p, q, r):
    """Point on the circle of radius r around p, toward q."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    n = math.hypot(dx, dy)
    return (p[0] + dx / n * r, p[1] + dy / n * r)


# ---------------------------------------------------------------- the chain diagram

def draw_chain(frame, t):
    if t >= T.start(REPEAT):
        return
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    w1, w2 = weights_at(t)
    g_meet = 1.0 if t < T.start(MEET) else ease_back((t - word(MEET, "smallest")) / 0.4)
    nudged = nudge_progress(t)
    h_val = lerp(HID, NEW_HID, ease_out((t - word(NUDGE, "one", 1) - 0.1) / 0.8)) if t >= word(NUDGE, "one", 1) else HID

    # Edges (thicker while their blame is active).
    a1, b1 = edge_end(IN_P, HID_P, IN_R), edge_end(HID_P, IN_P, HID_R)
    a2, b2 = edge_end(HID_P, OUT_P, HID_R), edge_end(OUT_P, HID_P, OUT_R)
    blame_w2 = word(STEP1, "last") <= t < T.start(NUDGE)
    blame_w1 = word(STEP3, "first") <= t < T.start(NUDGE)
    d.line([a1, b1], fill=(PINK if blame_w1 else lerp_color(MUTED, GREEN, nudged)) + (255,),
           width=int(8 + 10 * w1))
    d.line([a2, b2], fill=(PINK if blame_w2 else lerp_color(MUTED, GREEN, nudged)) + (255,),
           width=int(8 + 10 * w2))

    # Forward signals (yellow) - forward pass, and the replay after the nudge.
    for t0, dur in ((word(FORWARD, "forward"), 2.4), (word(NUDGE, "one", 1) - 0.4, 1.4)):
        if t0 <= t < t0 + dur:
            f = (t - t0) / dur * 2
            dots(d, a1, b1, clamp(f), YELLOW)
            if f > 1:
                dots(d, a2, b2, clamp(f - 1), YELLOW)
    if t < T.start(MEET):  # hook: blame pulses already racing backward
        f = (t * 0.9) % 1
        dots(d, b2, a2, f, PINK, r=9)
        dots(d, b1, a1, (f + 0.5) % 1, PINK, r=9)

    # Backward blame pulses (red), one hop per step.
    for (p, q), t0 in (((b2, a2), word(STEP1, "last")), ((b1, a1), word(STEP3, "first"))):
        if t0 <= t < t0 + 1.4:
            dots(d, p, q, clamp((t - t0) / 1.1), PINK, r=12)
    t_s2 = word(STEP2, "pass")
    if t_s2 <= t < t_s2 + 1.4:  # blame enters the hidden neuron
        dots(d, a2, (HID_P[0] + 30, HID_P[1]), clamp((t - t_s2) / 1.0), PINK, r=12)

    # Nodes.
    out_bad = OUTBLAME <= T.scene(t) < NUDGE or t < T.start(MEET)
    for p, r, ring in ((IN_P, IN_R, CYAN), (HID_P, HID_R, PURPLE), (OUT_P, OUT_R, PINK if out_bad else WHITE)):
        rr = r * g_meet
        if rr <= 1:
            continue
        pulse = 1 + 0.06 * math.sin(t * 9) if (p == OUT_P and out_bad) else 1
        d.ellipse([p[0] - rr * pulse, p[1] - rr * pulse, p[0] + rr * pulse, p[1] + rr * pulse], fill=(20, 22, 50, 255),
                  outline=ring + (255,), width=8)
    if g_meet > 0.7:
        text_center(d, IN_P, f"{X_IN:g} h", font("Black", 52), WHITE)
        slope_on = word(STEP2, "slope") <= t < T.start(STEP3) + 0.6
        mini_sigmoid(d, (HID_P[0], HID_P[1] - 6), 66, PURPLE, marker=W1 * X_IN if t >= T.start(FORWARD) else None,
                     tangent=clamp((t - word(STEP2, "slope")) / 0.4) if slope_on else 0.0)
        o_pct = round(100 * shown_output(t)) if t >= T.start(FORWARD) or t < T.start(MEET) else None
        text_center(d, OUT_P, "?" if o_pct is None else f"{o_pct}%", font("Black", 54),
                    PINK if out_bad else (GREEN if nudged > 0.5 else WHITE))
    frame.alpha_composite(add_glow(lay, radius=10, strength=1.0))

    # Labels and value tags.
    if g_meet > 0.7:
        text_center(ImageDraw.Draw(frame), (IN_P[0], IN_P[1] + 112), "slept", font("Bold", 32), MUTED)
        text_center(ImageDraw.Draw(frame), (HID_P[0], HID_P[1] + 140), "hidden", font("Bold", 32), MUTED)
        text_center(ImageDraw.Draw(frame), (OUT_P[0] - 10, OUT_P[1] + 124), "chance to pass", font("Bold", 28), MUTED)
        w_col = lambda done: (12, 28, 18) if done else (20, 20, 30)  # noqa: E731
        for mid, name, v, v0 in ((W1_MID, "w1", w1, W1), (W2_MID, "w2", w2, W2)):
            done = abs(v - v0) > 1e-3
            pop_in(frame, tag(f"{name} = {v:.2f}", w_col(done), (GREEN if done else YELLOW) + (255,), 38),
                   (mid[0], mid[1] - 80), t, word(MEET, "two") if t >= T.start(MEET) else -1)
    if t >= word(FORWARD, "eight") and t < T.start(REPEAT):
        pop_in(frame, tag(f"h = {h_val:.2f}", (20, 20, 30), WHITE + (255,), 32, pad=(14, 6)),
               (HID_P[0], HID_P[1] - 152), t, word(FORWARD, "network"))
    t_but = word(FORWARD, "but")
    if t_but <= t < T.start(NUDGE):
        pop_in(frame, tag("ANSWER: PASSED ✓", (12, 28, 18), GREEN + (255,), 34), (800, OUT_P[1] - 165), t, t_but)
    # Blame tags under each part.
    blames = [(OUT_P[0] - 10, OUT_P[1] + 180, BLAME_OUT, word(OUTBLAME, "off")),
              (W2_MID[0], W2_MID[1] + 76, BLAME_W2, word(STEP1, "times")),
              (HID_P[0], HID_P[1] + 200, BLAME_HID, word(STEP2, "slope")),
              (W1_MID[0], W1_MID[1] + 76, BLAME_W1, word(STEP3, "same"))]
    for x, y, v, ta in blames:
        if ta <= t < T.start(NUDGE) + 0.8:
            pop_in(frame, tag(f"blame {fmt(v)}", WHITE, PINK + (255,), 32, pad=(14, 6)), (x, y), t, ta)


def draw_ledger(frame, t):
    if t < word(OUTBLAME, "off") or t >= T.start(REPEAT):
        return
    d = ImageDraw.Draw(frame)
    x0, y0, x1, y1 = LEDGER
    d.rounded_rectangle(LEDGER, radius=26, fill=(16, 18, 42), outline=(70, 78, 120), width=3)
    text_center(d, (x0 + 28, y0 + 30), "BLAME LEDGER", font("Black", 26), MUTED, anchor="lm")
    text_center(d, (x1 - 28, y0 + 30), "← backward", font("Bold", 24), PINK, anchor="rm")
    rows = ledger_rows(t)
    for j, (num, text, ta, col) in enumerate(rows):
        y = y0 + 82 + j * 62
        active = j == len(rows) - 1
        g = ease_out((t - ta) / 0.3)
        if active:
            d.rounded_rectangle((x0 + 14, y - 29, x1 - 14, y + 29), radius=20, fill=(44, 40, 86))
        c = lerp_color((16, 18, 42), col if active else WHITE, g)
        text_center(d, (x0 + 48, y), num, font("Black", 36), lerp_color((16, 18, 42), col, g))
        text_center(d, (x0 + 86, y), text, font("Bold", 35), c, anchor="lm")


def draw_repeat(frame, t):
    """Scene 9: a bigger network with forward/backward pulses, and this student's prediction over 25 nudges."""
    t0 = T.start(REPEAT)
    if not (t0 <= t < T.start(CTA) + 0.3):
        return
    a = clamp((t - t0) / 0.4)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    ins, hid, out = [(170, 610), (170, 850)], [(540, 570), (540, 730), (540, 890)], [(900, 730)]
    rr = np.random.default_rng(4)
    edges = [(p, q, rr.normal()) for la, lb in ((ins, hid), (hid, out)) for p in la for q in lb]
    phase = ((t - t0) / 0.9) % 2  # 0..1 forward, 1..2 backward
    for p, q, w in edges:
        d.line([p, q], fill=(CYAN if w > 0 else PINK) + (int(200 * a),), width=int(3 + 4 * abs(w)))
        if phase < 1:
            dots(d, p, q, phase, YELLOW, r=8, n=1)
        else:
            dots(d, q, p, phase - 1, PINK, r=8, n=1)
    for la, col in ((ins, CYAN), (hid, PURPLE), (out, WHITE)):
        for p in la:
            d.ellipse([p[0] - 46, p[1] - 46, p[0] + 46, p[1] + 46], fill=(20, 22, 50, 255), outline=col + (255,), width=6)
    frame.alpha_composite(add_glow(lay, radius=8, strength=0.9))
    d2 = ImageDraw.Draw(frame)
    text_center(d2, (540, 958), "→ forward: predict      ← backward: blame", font("Bold", 32), MUTED)

    # Sparkline: this student's prediction after each nudge.
    x0, y0, x1, y1 = LEDGER
    d2.rounded_rectangle(LEDGER, radius=26, fill=(16, 18, 42), outline=(70, 78, 120), width=3)
    text_center(d2, (x0 + 28, y0 + 30), "OUR STUDENT, NUDGE BY NUDGE", font("Black", 26), MUTED, anchor="lm")
    cx0, cx1, cy0, cy1 = x0 + 90, x1 - 40, y0 + 70, y1 - 60
    for pct in (0, 50, 100):
        yy = lerp(cy1, cy0, pct / 100)
        d2.line([(cx0, yy), (cx1, yy)], fill=(50, 56, 92), width=2)
        text_center(d2, (cx0 - 14, yy), f"{pct}%", font("Bold", 22), MUTED, anchor="rm")
    n = len(TRAJ) - 1
    shown = clamp((t - t0 - 0.3) / (T.end(REPEAT) - t0 - 0.3)) * n
    pts = []
    for k in range(int(shown) + 1):
        pts.append((lerp(cx0, cx1, k / n), lerp(cy1, cy0, TRAJ[k])))
    if len(pts) > 1:
        d2.line(pts, fill=GREEN, width=6)
    if pts:
        px_, py_ = pts[-1]
        d2.ellipse([px_ - 10, py_ - 10, px_ + 10, py_ + 10], fill=YELLOW)
        k = int(shown)
        text_center(d2, (px_ - 16, py_ + (34 if TRAJ[k] > 0.75 else -34)), f"{round(100 * TRAJ[k])}%",
                    font("Black", 32), YELLOW, anchor="rm")
    text_center(d2, ((cx0 + cx1) / 2, y1 - 26), f"nudges: {int(shown)} / {n}", font("Bold", 24), MUTED)


def hud(t):
    if t < T.start(MEET) - 0.1:
        return f"{P0}%", PINK
    if t < T.start(FORWARD) + 1.0:
        return "?", MUTED
    if t < T.start(REPEAT):
        v = shown_output(t)
        return f"{round(100 * v)}%", lerp_color(PINK, GREEN, clamp((v - OUT) / (NEW_OUT - OUT)))
    n = len(TRAJ) - 1
    k = int(clamp((t - T.start(REPEAT) - 0.3) / (T.end(REPEAT) - T.start(REPEAT) - 0.3)) * n)
    return f"{round(100 * TRAJ[k])}%", GREEN


def draw_overlays(frame, t):
    draw_hud_pills(frame, 8, *hud(t), label="GUESS")
    frame.alpha_composite(TITLE_IMG, (0, 190))
    banners = [
        (T.start(HOOK), T.start(MEET), "WRONG ANSWER. WHO'S TO BLAME?", PINK, WHITE),
        (T.start(MEET), T.start(FORWARD), "THE SMALLEST NETWORK", PURPLE, WHITE),
        (T.start(FORWARD), T.start(OUTBLAME), "FORWARD PASS →", YELLOW, (20, 20, 30)),
        (T.start(OUTBLAME), T.start(STEP1), "← BACKWARD PASS: FIND THE BLAME", PINK, WHITE),
        (T.start(STEP1), T.start(STEP2), "STEP 1 · BLAME × WHAT FLOWED IN", PINK, WHITE),
        (T.start(STEP2), T.start(STEP3), "STEP 2 · × WEIGHT × SLOPE", PINK, WHITE),
        (T.start(STEP3), word(STEP3, "blame", 1), "STEP 3 · SAME FOR THE FIRST WEIGHT", PINK, WHITE),
        (word(STEP3, "blame", 1), T.start(NUDGE), "BACKPROPAGATION = BLAME FLOWS BACKWARD", PURPLE, WHITE),
        (T.start(NUDGE), T.start(REPEAT), "NUDGE AGAINST THE BLAME (LEVEL 1!)", GREEN, (12, 28, 18)),
        (T.start(REPEAT), T.start(CTA), "REPEAT × THOUSANDS = LEARNING", GREEN, (12, 28, 18)),
    ]
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 32), (W / 2, TOP_TAG_Y), t, a)
    t_one = word(NUDGE, "one", 1)
    if t_one + 1.0 <= t < T.start(REPEAT):
        pop_in(frame, tag(f"{P0}% → {P1}%", (12, 28, 18), GREEN + (255,), 42), (800, OUT_P[1] - 165), t, t_one + 1.0)
    draw_level_cleared(frame, t, T.start(CTA), 8, "HOW AI SEES IMAGES", word(CTA, "next:"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    draw_chain(frame, t)
    draw_ledger(frame, t)
    draw_repeat(frame, t)
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
    mix(out, sfx_buzz(0.3, 100, 0.16), 0.2)
    mix(out, sfx_whoosh(0.7, 0.18), 0.0)
    for k in range(3):
        mix(out, sfx_blip(700 + 200 * k, 0.08, 0.12), word(MEET, "smallest") + 0.12 * k)
    mix(out, sfx_whoosh(0.8, 0.16, seed=3), word(FORWARD, "forward"))
    mix(out, sfx_buzz(0.25, 110, 0.14), word(FORWARD, "network") + 0.3)
    mix(out, sfx_blip(1568, 0.3, 0.18), word(FORWARD, "but"))
    for ta in (word(OUTBLAME, "off"), word(STEP1, "times"), word(STEP2, "slope"), word(STEP3, "same")):
        mix(out, sfx_blip(523, 0.1, 0.14), ta)
        mix(out, sfx_blip(392, 0.12, 0.12), ta + 0.1)
    for ta in (word(STEP1, "last"), word(STEP2, "pass"), word(STEP3, "first")):
        mix(out, sfx_whoosh(0.6, 0.14, seed=7), ta)
    mix(out, sfx_boing(0.6, 0.18), word(STEP3, "blame", 1))
    mix(out, sfx_boing(0.5, 0.16), word(NUDGE, "nudge"))
    mix(out, sfx_blip(1568, 0.35, 0.2), word(NUDGE, "one", 1) + 1.0)
    t0 = T.start(REPEAT)
    for k in range(12):
        mix(out, sfx_blip(600 + 50 * k, 0.04, 0.07), lerp(t0 + 0.3, T.end(REPEAT), k / 12))
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12, max_px=800)
    STATIC = build_plain()
    TITLE_IMG = build_title("BACKPROPAGATION", "ML AS A GAME  ·  EP 8")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = word(STEP3, "blame", 1) + 0.5
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "WHO'S TO BLAME?", font("Black", 100), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "Backpropagation, explained", font("Black", 56), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s | forward: h={HID:.4f} o={OUT:.4f} | blame out {BLAME_OUT:.4f}, w2 {BLAME_W2:.4f} "
          f"(fd {FD_W2:.4f}), hidden {BLAME_HID:.4f} (slope {SLOPE:.4f}), w1 {BLAME_W1:.4f} (fd {FD_W1:.4f})")
    print(f"nudge (lr {LR}): w1 {W1} -> {NEW_W1:.4f}, w2 {W2} -> {NEW_W2:.4f}; prediction {OUT:.4f} -> {NEW_OUT:.4f} | "
          f"25 nudges: {[round(v, 2) for v in TRAJ]}")
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
    out = os.path.join(HERE, "backprop_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
