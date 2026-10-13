"""ML as a Game, Level 5: K-Nearest Neighbors (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> knn_short.mp4 + thumbnail.png
    python3 render.py --preview 3,12  # just dump PNG frames at those seconds (fast check)
    python3 render.py --thumbnail     # only regenerate thumbnail.png
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageChops, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import (  # noqa: E402
    CYAN, FPS, GREEN, H, MUTED, PINK, PURPLE, SR, W, WHITE, YELLOW, PlotBox, Timeline, add_glow,
    build_confetti, build_static, build_title, caption_chunks, clamp, draw_caption, draw_hud_pills,
    draw_level_cleared, ease_back, ease_in_out, ease_out, encode, font, lerp, lerp_color, mix, music_bed,
    num_to_words, paste_scaled, pop_in, sfx_blip, sfx_boing, sfx_buzz, sfx_levelup, sfx_whoosh, tag, text_center,
    voiceover, write_wav,
)

# ---------------------------------------------------------------- data (same students as Levels 3-4)

STUDENTS = [
    (0.8, 3.0, 0), (1.2, 4.4, 0), (0.6, 9.2, 0), (2.5, 9.6, 0), (1.7, 5.8, 0), (3.5, 6.1, 0), (1.4, 6.5, 0),
    (1.9, 8.2, 0), (2.3, 2.0, 0), (2.8, 5.0, 0), (3.1, 7.2, 0), (3.7, 3.6, 0), (2.2, 7.4, 1),
    (4.6, 2.5, 0), (5.6, 4.6, 0), (6.4, 1.8, 0), (7.8, 3.2, 0), (8.9, 4.1, 0), (9.3, 3.0, 1),
    (4.3, 6.2, 1), (5.1, 8.7, 1), (5.9, 5.4, 1), (6.7, 7.5, 1), (7.2, 9.1, 1), (8.0, 6.3, 1), (8.8, 8.0, 1),
    (9.4, 5.9, 1), (5.4, 7.1, 0),
]
DATA = np.array(STUDENTS)
X, Y = DATA[:, :2], DATA[:, 2]
N = len(Y)
WEIRD = 27
NEW = np.array([5.5, 6.5])

# ---------------------------------------------------------------- KNN

DIST = np.sqrt(((X - NEW) ** 2).sum(1))
ORDER = np.argsort(DIST)  # nearest first


def knn_vote(k):
    """(pass votes, k, prediction) for the new student."""
    p = int(Y[ORDER[:k]].sum())
    return p, k, int(p * 2 > k)


def loo_accuracy(k):
    """Hide each student, predict them from everyone else; count how many come out right."""
    ok = 0
    for i in range(N):
        d = np.sqrt(((X - X[i]) ** 2).sum(1))
        d[i] = np.inf
        nn = np.argsort(d)[:k]
        ok += int((Y[nn].mean() > 0.5) == Y[i])
    return ok


K_TEST = [1, 3, 5, 7, 9, 11, 13, 15, N - 1]  # N-1 = the whole rest of the class
SCORES = {k: loo_accuracy(k) for k in K_TEST}
BEST_K = max(K_TEST, key=lambda k: (SCORES[k], -k))
assert knn_vote(3) == (2, 3, 1) and knn_vote(1)[2] == 0 and ORDER[0] == WEIRD and knn_vote(N)[2] == 0
assert BEST_K == 5 and knn_vote(BEST_K)[2] == 1, (BEST_K, SCORES)

G = 200


def knn_map(k):
    """Vote share (0..1 pass) on a GxG grid over [0,10]^2, row 0 = slept 0."""
    c = (np.arange(G) + 0.5) / G * 10
    gx, gy = np.meshgrid(c, c)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    d = ((pts[:, None, :] - X[None]) ** 2).sum(2)
    nn = np.argsort(d, axis=1)[:, :k]
    return Y[nn].mean(1).reshape(G, G).astype(np.float32)


SCRIPT = [
    "Want to guess if a new student will pass? Just ask their closest neighbors.",
    "Here's a new student: five and a half hours studying, six and a half hours of sleep.",
    "Step one: measure how far they are from every other student.",
    "Step two: pick the K closest. Let's say K equals three.",
    "Step three: the neighbors vote. Two passed, one failed. So: pass!",
    "That's k nearest neighbors. There's no training. It just remembers everyone.",
    "But K matters. With K equals one, only the closest counts, and that's our weird student. Fail!",
    "Make K too big, and the whole class votes. Most failed, so everyone gets fail.",
    f"So we test: hide each student, and predict them from the rest. K equals {num_to_words(BEST_K)} scores best.",
    "Level five cleared. Next: k means clustering. Don't mix them up! Follow for level six.",
]
HOOK, NEWS, STEP1, STEP2, STEP3, RECAP, K1, KBIG, TEST, CTA = range(10)

# ---------------------------------------------------------------- layout

# Square plot so distances look right: a circle on screen is a circle in the data.
PLOT = PlotBox(210, 890, 440, 1120, (0, 10), (0, 10), layer=(110, 400, 960, 1160))
px, py, L = PLOT.px, PLOT.py, PLOT.L
SS = PLOT.ss
UNIT = (PLOT.x1 - PLOT.x0) / 10  # screen px per hour (same on both axes)
CAPTION_Y = 1495
TOP_TAG_Y = PLOT.y_top - 32
SLIDER_Y = 1262
CHIPS_Y = 1352
TRACK = (300, 900)

T: Timeline = None
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0
MAPS = {}


def word(i, prefix, nth=0):
    return T.word(i, prefix, nth)


# ---------------------------------------------------------------- state

def current_k(t):
    """K shown on screen (float while animating), or None before step two."""
    if t < word(STEP2, "three"):
        return None
    if t < word(K1, "one"):
        return 3.0
    if t < word(KBIG, "big"):
        return lerp(3, 1, ease_in_out((t - word(K1, "one")) / 0.5))
    if t < T.start(TEST):
        return lerp(1, N, ease_in_out((t - word(KBIG, "big")) / 0.9))
    t_b = word(TEST, num_to_words(BEST_K))
    if t < t_b:
        return float(N)
    return lerp(N, BEST_K, ease_in_out((t - t_b) / 0.6))


def circle_radius(t):
    """Neighbour circle radius in hours (encloses the K nearest)."""
    k = current_k(t)
    if k is None:
        if t < T.start(NEWS):  # hook: pulse around the 3 nearest
            return (DIST[ORDER[2]] + 0.25) * (0.8 + 0.2 * math.sin(t * 6))
        return None
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    r_lo = DIST[ORDER[lo - 1]] + 0.25
    r_hi = DIST[ORDER[hi - 1]] + 0.25
    return lerp(r_lo, r_hi, k - lo)


def k_int(t):
    k = current_k(t)
    return None if k is None else int(round(k))


def map_state(t):
    """(k, alpha) for the faint decision map, or None."""
    if word(K1, "one") <= t < word(KBIG, "big"):
        return 1, clamp((t - word(K1, "one") - 0.3) / 0.4)
    if word(KBIG, "whole") <= t < T.start(TEST) + 0.3:
        return N, clamp((t - word(KBIG, "whole")) / 0.4) * (1 - clamp((t - T.start(TEST)) / 0.3))
    t_b = word(TEST, num_to_words(BEST_K))
    if t_b <= t < T.start(CTA):
        return BEST_K, clamp((t - t_b - 0.3) / 0.5)
    return None


def hud_value(t):
    if t < T.start(NEWS) - 0.1:
        return f"{(int(t * FPS) * 7919) % 27 + 1}", PINK
    k = k_int(t)
    if k is None:
        return "?", MUTED
    return f"{k}", GREEN if k == BEST_K and t >= T.start(TEST) else CYAN


def verdict(t):
    """(text, is_pass, start time) of the current verdict, or None."""
    if word(STEP3, "so:") <= t < T.start(RECAP) + 2.5 and t < word(K1, "one"):
        return "PASS", True, word(STEP3, "so:")
    if word(K1, "fail!") <= t < word(KBIG, "big"):
        return "FAIL", False, word(K1, "fail!")
    if word(KBIG, "everyone") <= t < T.start(TEST):
        return "FAIL", False, word(KBIG, "everyone")
    t_b = word(TEST, num_to_words(BEST_K))
    if t_b + 0.6 <= t < T.start(CTA):
        return "PASS", True, t_b + 0.6
    return None


# ---------------------------------------------------------------- drawing

def map_rgba(k, alpha):
    g = MAPS[k][::-1]
    pink, green = np.array(PINK, np.float32), np.array(GREEN, np.float32)
    rgb = pink[None, None] * (1 - g[..., None]) + green[None, None] * g[..., None]
    a = np.full(g.shape, 0.3 * alpha * 255)
    img = Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), "RGBA")
    return img.resize((int((PLOT.x1 - PLOT.x0) * SS), int((PLOT.y_bot - PLOT.y_top) * SS)), Image.NEAREST)


def draw_plot_layer(t):
    lay = PLOT.new_layer()
    ms = map_state(t)
    if ms and ms[1] > 0:
        lay.alpha_composite(map_rgba(*ms), (int((PLOT.x0 - PLOT.layer[0]) * SS), int((PLOT.y_top - PLOT.layer[1]) * SS)))
    d = ImageDraw.Draw(lay)
    sx, sy = L(*NEW)
    k = k_int(t)

    # Step 1: distance lines to everyone (nearest-first labels come in the overlay).
    t_m = word(STEP1, "measure")
    if t_m <= t < word(STEP2, "pick") + 0.6:
        fade_out = clamp((t - word(STEP2, "pick")) / 0.6)
        for rank, i in enumerate(ORDER):
            g = ease_out((t - t_m - 0.05 * rank) / 0.3)
            if g <= 0:
                continue
            ex, ey = L(*X[i])
            a = 1 - fade_out if rank >= 3 else 1.0
            if a <= 0:
                continue
            d.line([(sx, sy), (lerp(sx, ex, g), lerp(sy, ey, g))], fill=WHITE + (int(150 * a),), width=4)

    # The neighbour circle.
    r = circle_radius(t)
    if r is not None and t < T.start(CTA):
        rp = r * UNIT * SS
        a = 1.0 if t >= T.start(NEWS) else 0.6
        circ = PLOT.new_layer()
        cd = ImageDraw.Draw(circ)
        cd.ellipse([sx - rp, sy - rp, sx + rp, sy + rp], fill=YELLOW + (int(18 * a),), outline=YELLOW + (int(230 * a),),
                   width=8)
        clip = Image.new("L", circ.size, 0)
        ImageDraw.Draw(clip).rectangle([*L(0, 10), *L(10, 0)], fill=255)
        circ.putalpha(ImageChops.multiply(circ.getchannel("A"), clip))
        lay.alpha_composite(circ)

    # Lines to the K chosen neighbours.
    if k is not None and t < T.start(CTA):
        for i in ORDER[:min(k, 7)]:
            d.line([(sx, sy), L(*X[i])], fill=YELLOW + (220,), width=6)

    # Students.
    pulse_all = word(RECAP, "remembers") <= t < T.start(K1)
    for i in range(N):
        s = ease_back((t - 0.15 - 0.015 * i) / 0.3)
        if s <= 0:
            continue
        cx, cy = L(*X[i])
        rr = 16 * SS * s
        if pulse_all:
            rr *= 1 + 0.2 * abs(math.sin((t - word(RECAP, "remembers")) * 6 + i * 0.4))
        inside = k is not None and i in ORDER[:k]
        col = GREEN if Y[i] else PINK
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col + (255,))
        d.ellipse([cx - rr * 0.5, cy - rr * 0.5, cx + rr * 0.5, cy + rr * 0.5], fill=WHITE + (255,))
        if inside and k <= 7:
            d.ellipse([cx - rr * 1.5, cy - rr * 1.5, cx + rr * 1.5, cy + rr * 1.5], outline=YELLOW + (255,), width=5)
        if i == WEIRD and word(K1, "weird") <= t < word(KBIG, "big"):
            q = rr * (2.2 + 0.3 * math.sin((t - word(K1, "weird")) * 9))
            d.ellipse([cx - q, cy - q, cx + q, cy + q], outline=PINK + (255,), width=8)

    # New student star.
    t0 = 0.3
    s = ease_back((t - t0) / 0.4) * (1 + (0.35 * max(0, 1 - abs(t - word(NEWS, "new") - 0.15) / 0.2)))
    if s > 0 and t < T.start(CTA):
        v = verdict(t)
        col = YELLOW if v is None else (GREEN if v[1] else PINK)
        pts = []
        for j in range(10):
            ang = -math.pi / 2 + j * math.pi / 5
            rad = (38 if j % 2 == 0 else 17) * SS * s * (1 + 0.08 * math.sin(t * 7))
            pts.append((sx + rad * math.cos(ang), sy + rad * math.sin(ang)))
        d.polygon(pts, fill=col + (255,), outline=(20, 20, 30, 255))
    return PLOT.finish_layer(lay, glow=10, strength=1.3)


def chip(text, is_pass, size=34):
    return tag(text, (12, 28, 18) if is_pass else WHITE, (GREEN if is_pass else PINK) + (255,), size, pad=(18, 10))


def draw_panel(frame, t):
    d = ImageDraw.Draw(frame)

    # Step 1: the distance formula, big and plain.
    if word(STEP1, "measure") <= t < word(STEP2, "pick"):
        a = clamp((t - word(STEP1, "measure")) / 0.3) * (1 - clamp((t - word(STEP2, "pick") + 0.3) / 0.3))
        img = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
        cd = ImageDraw.Draw(img)
        text_center(cd, (W / 2, 70), "distance = straight-line gap", font("Black", 56), WHITE)
        text_center(cd, (W / 2, 140), "√( Δstudied² + Δslept² )", font("Bold", 44), MUTED)
        paste_scaled(frame, img, (W / 2, 1300), alpha=a)

    k = current_k(t)
    in_test = T.start(TEST) <= t < T.start(CTA)
    # K slider.
    if k is not None and not in_test:
        ki = int(round(k))
        text_center(d, (100, SLIDER_Y), "K =", font("Black", 52), MUTED, anchor="lm")
        text_center(d, (205, SLIDER_Y), f"{ki}", font("Black", 64), YELLOW, anchor="lm")
        x0, x1 = TRACK
        d.rounded_rectangle((x0, SLIDER_Y - 7, x1, SLIDER_Y + 7), radius=7, fill=(60, 66, 104))
        kx = lerp(x0, x1, (k - 1) / (N - 1))
        d.rounded_rectangle((x0, SLIDER_Y - 7, kx, SLIDER_Y + 7), radius=7, fill=YELLOW)
        for tk, lab in ((1, "1"), (N, f"{N}  (everyone)")):
            tx = lerp(x0, x1, (tk - 1) / (N - 1))
            text_center(d, (tx, SLIDER_Y + 36), lab, font("Bold", 26), MUTED, anchor="mm" if tk == 1 else "rm")
        d.ellipse((kx - 20, SLIDER_Y - 20, kx + 20, SLIDER_Y + 20), fill=WHITE, outline=YELLOW, width=5)

        # Votes: chips for small K, a tally bar for big K.
        if t >= word(STEP3, "vote"):
            p, kk, _ = knn_vote(ki)
            if ki <= 7:
                t_v = word(STEP3, "vote")
                ws = [chip("PASS" if Y[i] else "FAIL", bool(Y[i])) for i in ORDER[:ki]]
                x = 120
                for j, w in enumerate(ws):
                    c = (x + w.width / 2, CHIPS_Y)
                    pop_in(frame, w, c, t, t_v + 0.15 * j if t < word(K1, "one") else 0)
                    x += w.width + 14
            else:
                bx0, bx1 = 120, 640
                d.rounded_rectangle((bx0, CHIPS_Y - 24, bx1, CHIPS_Y + 24), radius=24, fill=PINK)
                xp = bx0 + (bx1 - bx0) * p / kk
                if xp - bx0 > 30:
                    d.rounded_rectangle((bx0, CHIPS_Y - 24, xp, CHIPS_Y + 24), radius=24, fill=GREEN)
                text_center(d, (bx0 + 18, CHIPS_Y), f"PASS {p}", font("Black", 28), (12, 28, 18), anchor="lm")
                text_center(d, (bx1 - 18, CHIPS_Y), f"FAIL {kk - p}", font("Black", 28), WHITE, anchor="rm")

    v = verdict(t)
    if v and not in_test:
        pop_in(frame, tag(f"→ {v[0]}", (12, 28, 18) if v[1] else WHITE, (GREEN if v[1] else PINK) + (255,), 48),
               (800, CHIPS_Y), t, v[2], dur=0.35)

    # Test scene: leave-one-out scores per K as a bar chart.
    if in_test:
        a = clamp((t - T.start(TEST)) / 0.3)
        base, top = 1405, 1262
        bw, gap = 66, 14
        x = (W - (len(K_TEST) * bw + (len(K_TEST) - 1) * gap)) / 2 - 40
        t_h = word(TEST, "hide")
        t_b = word(TEST, num_to_words(BEST_K))
        text_center(d, (x, top - 8), "right answers on hidden students", font("Bold", 28), lerp_color((0, 0, 0), MUTED, a),
                    anchor="ls")
        for j, kk in enumerate(K_TEST):
            g = ease_out((t - t_h - 0.08 * j) / 0.4)
            hgt = (SCORES[kk] - 12) / (N - 12) * (base - top - 50) * g
            best = kk == BEST_K and t >= t_b
            col = GREEN if best else (PINK if kk in (1, N - 1) else (90, 100, 150))
            if hgt > 2:
                d.rounded_rectangle((x, base - hgt, x + bw, base), radius=10, fill=col)
            if g > 0.6:
                text_center(d, (x + bw / 2, base - hgt - 18), f"{SCORES[kk]}", font("Black", 28),
                            YELLOW if best else WHITE)
            text_center(d, (x + bw / 2, base + 22), "all" if kk == N - 1 else f"K{kk}", font("Bold", 24),
                        YELLOW if best else MUTED)
            x += bw + gap


def draw_overlays(frame, t):
    draw_hud_pills(frame, 5, *hud_value(t), label="K")
    t_r = word(RECAP, "that's")
    if t < t_r or t >= T.start(K1):
        frame.alpha_composite(TITLE_IMG, (0, 190))
    else:
        g = ease_back((t - t_r) / 0.45)
        card = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
        cd = ImageDraw.Draw(card)
        text_center(cd, (W / 2, 80), "ask the K closest → majority vote", font("Black", 56), WHITE)
        text_center(cd, (W / 2, 145), "no training: it just remembers everyone", font("Bold", 32), MUTED)
        paste_scaled(frame, add_glow(card, radius=10, strength=1.1), (W / 2, 290), scale=0.6 + 0.4 * g, alpha=clamp(g))

    # Legend (right of the plot, clear of YouTube's buttons lower down).
    lx, ly = 985, PLOT.y_top + 40
    d = ImageDraw.Draw(frame)
    for j, (lab, col) in enumerate((("PASS", GREEN), ("FAIL", PINK))):
        y = ly + j * 50
        d.ellipse((lx - 46, y - 13, lx - 20, y + 13), fill=col)
        text_center(d, (lx - 8, y), lab, font("Bold", 28), col, anchor="lm")

    # Step banners and scene tags (top of plot).
    banners = [
        (T.start(HOOK), T.start(NEWS), "WHO ARE YOUR NEIGHBORS?", PURPLE, WHITE),
        (word(STEP1, "step"), T.start(STEP2), "STEP 1 · MEASURE DISTANCES", CYAN, (12, 14, 30)),
        (word(STEP2, "step"), T.start(STEP3), "STEP 2 · PICK THE K CLOSEST", YELLOW, (20, 20, 30)),
        (word(STEP3, "step"), T.start(RECAP), "STEP 3 · NEIGHBORS VOTE", GREEN, (12, 28, 18)),
        (word(RECAP, "no"), T.start(K1), "TRAINING TIME: 0 s", PURPLE, WHITE),
        (word(K1, "one"), T.start(KBIG), "K = 1: ONLY THE CLOSEST COUNTS", PINK, WHITE),
        (word(KBIG, "big"), T.start(TEST), f"K = {N}: THE WHOLE CLASS VOTES", PINK, WHITE),
        (word(TEST, "hide"), word(TEST, num_to_words(BEST_K)) + 0.6, "HIDE ONE · PREDICT IT · REPEAT", CYAN,
         (12, 14, 30)),
        (word(TEST, num_to_words(BEST_K)) + 0.6, T.start(CTA), f"BEST: K = {BEST_K}  →  PASS", GREEN, (12, 28, 18)),
    ]
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 34), (px(5), TOP_TAG_Y), t, a)

    # New-student label.
    t_n = word(NEWS, "new")
    if t_n <= t < T.start(STEP1) + 0.3:
        pop_in(frame, tag("NEW: 5.5 h study · 6.5 h sleep", (20, 20, 30), YELLOW + (255,), 32),
               (px(NEW[0]), py(NEW[1]) - 70), t, t_n + 0.2)

    # Distance labels on the nearest few during step 1, rank badges in step 2.
    t_m = word(STEP1, "measure")
    if t_m <= t < word(STEP2, "pick"):
        for rank, i in enumerate(ORDER[:4]):
            ta = t_m + 0.05 * rank + 0.4
            if t >= ta:
                side = 1 if X[i, 0] >= NEW[0] else -1
                paste_scaled(frame, tag(f"{DIST[i]:.1f} h", (20, 20, 30), WHITE + (255,), 24, pad=(10, 4)),
                             (px(X[i, 0]) + side * 52, py(X[i, 1]) + (-26 if X[i, 1] >= NEW[1] else 26)))
    if word(STEP2, "closest") <= t < word(STEP3, "vote"):
        for rank, i in enumerate(ORDER[:3]):
            pop_in(frame, tag(f"#{rank + 1}", (20, 20, 30), YELLOW + (255,), 26, pad=(10, 5)),
                   (px(X[i, 0]) + 34, py(X[i, 1]) - 30), t, word(STEP2, "closest") + 0.2 * rank)

    draw_panel(frame, t)
    draw_level_cleared(frame, t, T.start(CTA), 5, "K-MEANS CLUSTERING", word(CTA, "next"), CONFETTI)


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
    mix(out, sfx_blip(1568, 0.3, 0.16), word(NEWS, "new"))
    t_m = word(STEP1, "measure")
    for r in range(N):
        mix(out, sfx_blip(500 + 25 * r, 0.03, 0.05), t_m + 0.05 * r)
    for r in range(3):
        mix(out, sfx_blip(880 + 220 * r, 0.08, 0.13), word(STEP2, "closest") + 0.2 * r)
    mix(out, sfx_boing(0.5, 0.16), word(STEP2, "three"))
    for j in range(3):
        mix(out, sfx_blip(523 if Y[ORDER[j]] == 0 else 1046, 0.09, 0.13), word(STEP3, "vote") + 0.15 * j)
    mix(out, sfx_blip(1568, 0.35, 0.2), word(STEP3, "so:"))
    mix(out, sfx_whoosh(0.6, 0.15, seed=5), word(K1, "one"))
    mix(out, sfx_buzz(0.35, 90, 0.18), word(K1, "fail!"))
    mix(out, sfx_whoosh(1.0, 0.18, seed=7), word(KBIG, "big"))
    mix(out, sfx_buzz(0.35, 90, 0.18), word(KBIG, "everyone"))
    t_h = word(TEST, "hide")
    for j in range(len(K_TEST)):
        mix(out, sfx_blip(600 + 70 * j, 0.06, 0.1), t_h + 0.08 * j)
    t_b = word(TEST, num_to_words(BEST_K))
    mix(out, sfx_boing(0.6, 0.18), t_b)
    mix(out, sfx_blip(1568, 0.35, 0.2), t_b + 0.6)
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION, MAPS
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12, max_px=800)
    STATIC = build_static(PLOT, list(range(0, 11, 2)), list(range(0, 11, 2)), "HOURS STUDIED", "HOURS SLEPT")
    TITLE_IMG = build_title("K-NEAREST NEIGHBORS", "ML AS A GAME  ·  EP 5")
    CONFETTI = build_confetti()
    MAPS = {k: knn_map(k) for k in (1, BEST_K, N)}
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = word(STEP3, "so:") + 0.8
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "ASK YOUR NEIGHBORS", font("Black", 86), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "K-Nearest Neighbors, explained", font("Black", 56), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s | nearest: " +
          ", ".join(f"{tuple(X[i])}:{'P' if Y[i] else 'F'}@{DIST[i]:.2f}" for i in ORDER[:5]))
    print("votes: " + " | ".join(f"K={k}: {knn_vote(k)[0]}/{k} pass" for k in (1, 3, 5, N)) +
          f" | leave-one-out: {SCORES} -> best K={BEST_K}")
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
    out = os.path.join(HERE, "knn_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
