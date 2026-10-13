"""ML as a Game, Level 3: Decision Trees (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> decision_trees_short.mp4 + thumbnail.png
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
    CYAN, FPS, GREEN, H, MUTED, ORANGE, PINK, SR, W, WHITE, YELLOW, PlotBox, Timeline,
    build_confetti, build_static, build_title, caption_chunks, clamp, dashed_line, draw_caption, draw_hud_pills,
    draw_level_cleared, ease_back, ease_in_out, ease_out, encode, font, lerp, lerp_color, mix, music_bed,
    num_to_words, paste_scaled, pop_in, sfx_blip, sfx_boing, sfx_buzz, sfx_levelup, sfx_whoosh, tag,
    text_center, voiceover, write_wav,
)

# ---------------------------------------------------------------- data + CART

# (hours studied, hours slept, passed). Three deliberate "weird students": a pass at (2.2, 7.4),
# a pass at (9.3, 3.0) and a fail at (5.4, 7.1).
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
NEW = (6.0, 4.0)  # the new student we predict for


def gini(idx):
    if len(idx) == 0:
        return 0.0
    p = Y[idx].mean()
    return 2 * p * (1 - p)


def split_score(idx, f, th):
    """Weighted Gini impurity of a split (0 = perfectly pure, 0.5 = 50/50 mess)."""
    left, right = idx[X[idx, f] < th], idx[X[idx, f] >= th]
    return (len(left) * gini(left) + len(right) * gini(right)) / len(idx)


def best_split(idx):
    """Try every midpoint on both features; return (score, feature, threshold)."""
    best = None
    for f in (0, 1):
        vals = np.unique(X[idx, f])
        for a, b in zip(vals[:-1], vals[1:]):
            th = round((a + b) / 2, 6)
            s = split_score(idx, f, th)
            if best is None or s < best[0] - 1e-12:
                best = (s, f, th)
    return best


def grow(idx, box=(0, 10, 0, 10)):
    """Full-depth CART; returns leaves as (box, class)."""
    if gini(idx) == 0:
        return [(box, int(Y[idx][0]))]
    _, f, th = best_split(idx)
    left, right = idx[X[idx, f] < th], idx[X[idx, f] >= th]
    x0, x1, y0, y1 = box
    lb, rb = ((x0, th, y0, y1), (th, x1, y0, y1)) if f == 0 else ((x0, x1, y0, th), (x0, x1, th, y1))
    return grow(left, lb) + grow(right, rb)


ALL = np.arange(N)
S1, F1, TH1 = best_split(ALL)
RIGHT = ALL[X[:, F1] >= TH1]
S2, F2, TH2 = best_split(RIGHT)
assert (F1, TH1, F2, TH2) == (0, 4.0, 1, 5.0), (F1, TH1, F2, TH2)
LO = RIGHT[X[RIGHT, 1] < TH2]
HI = RIGHT[X[RIGHT, 1] >= TH2]
MAJ = int(Y.mean() >= 0.5)
RIGHT_MAJ = int(Y[RIGHT].mean() >= 0.5)
LO_MAJ, HI_MAJ = int(Y[LO].mean() >= 0.5), int(Y[HI].mean() >= 0.5)


def predict(x, depth):
    if depth == 0:
        return MAJ
    if x[0] < TH1:
        return 0
    if depth == 1:
        return RIGHT_MAJ
    return LO_MAJ if x[1] < TH2 else HI_MAJ


ERR = [int(sum(predict(x, d) != y for x, y in zip(X, Y))) for d in (0, 1, 2)]
FULL = grow(ALL)
NEW_PRED = predict(NEW, 2)
assert NEW_PRED == 0 and LO_MAJ == 0 and HI_MAJ == 1
GINI0 = gini(ALL)


def mixed_pct(score):
    """Impurity as a 0-100% 'how mixed' meter (Gini 0.5 = 100% mixed)."""
    return int(round(score / 0.5 * 100))


SCRIPT = [
    "This AI decides your grade by playing twenty questions.",
    "Every dot is a student: hours studied, and hours slept. Green passed. Pink failed.",
    "The tree's job: ask yes or no questions that split the passes from the fails.",
    "First question. It tries every possible cut, and scores how mixed each side is.",
    f"Best cut: studied less than {num_to_words(TH1)} hours? Almost everyone there failed.",
    f"The rest are still mixed, so it asks again: slept less than {num_to_words(TH2)} hours?",
    "Now each box is almost pure. That's a decision tree.",
    "New student: six hours studying, four hours of sleep. Follow the questions. Fail! Get some sleep.",
    "But ask too many questions, and it memorizes weird students. Keep the tree short.",
    "Level three cleared. Next: random forests, a whole team of trees. Follow for level four.",
]
HOOK, INTRO, GOAL, SWEEP1, LOCK1, SWEEP2, DONE, PREDICT, OVERFIT, CTA = range(10)

# ---------------------------------------------------------------- layout

PLOT = PlotBox(190, 960, 460, 1000, (0, 10), (0, 10), layer=(90, 400, 1010, 1040))
px, py, L = PLOT.px, PLOT.py, PLOT.L
SS = PLOT.ss
DIAG = (30, 1090, 1010, 1440)  # tree-diagram layer (screen px), drawn at 2x
CAPTION_Y = 1490

# Tree diagram node positions (screen px).
NODE = {
    "root": (480, 1150),
    "L": (240, 1262),
    "R": (720, 1262),
    "RL": (590, 1374),
    "RR": (850, 1374),
}
EDGES = [("root", "L", "yes"), ("root", "R", "no"), ("R", "RL", "yes"), ("R", "RR", "no")]

T: Timeline = None
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0


# ---------------------------------------------------------------- state helpers

def word(i, prefix, nth=0):
    return T.word(i, prefix, nth)


def sweep1(t):
    """Vertical cut position during the first sweep (None outside it)."""
    t0, t1 = word(SWEEP1, "tries"), T.end(SWEEP1) + 0.1
    if t < t0:
        return None
    if t < t1:
        return lerp(0.3, 9.7, ease_in_out((t - t0) / (t1 - t0)))
    t_lock = T.start(LOCK1)
    return lerp(9.7, TH1, ease_out((t - t_lock) / 0.5)) if t < t_lock + 0.5 else TH1


def sweep2(t):
    t0, t1 = word(SWEEP2, "again"), word(SWEEP2, "slept")
    if t < t0:
        return None
    if t < t1 - 0.05:
        return lerp(0.3, 9.7, ease_in_out((t - t0) / (t1 - 0.05 - t0)))
    return lerp(9.7, TH2, ease_out((t - t1 + 0.05) / 0.45)) if t < t1 + 0.4 else TH2


def best_so_far(idx, f, pos):
    """Best split among thresholds the sweep has passed."""
    vals = np.unique(X[idx, f])
    best = None
    for a, b in zip(vals[:-1], vals[1:]):
        th = (a + b) / 2
        if th > pos:
            break
        s = split_score(idx, f, th)
        if best is None or s < best[0] - 1e-12:
            best = (s, th)
    return best


def error_display(t):
    if t < T.start(INTRO) - 0.1:
        n = (int(t * FPS) * 7919) % 20 + 3
        return f"{n}", PINK
    t_job = word(GOAL, "job")
    if t < t_job:
        return "???", MUTED
    if t < word(LOCK1, "failed"):
        return f"{int(round(ERR[0] * ease_out((t - t_job) / 0.6)))}", PINK
    if t < word(DONE, "each"):
        return f"{ERR[1]}", ORANGE
    if word(OVERFIT, "memorizes") <= t < word(OVERFIT, "short"):
        return "0", YELLOW
    return f"{ERR[2]}", GREEN


def dot_color(k, t):
    tg, tp = word(INTRO, "green"), word(INTRO, "pink")
    if Y[k] == 1:
        return lerp_color(CYAN, GREEN, clamp((t - tg) / 0.25)), 1 + 0.35 * max(0, 1 - abs(t - tg - 0.12) / 0.15)
    return lerp_color(CYAN, PINK, clamp((t - tp) / 0.25)), 1 + 0.35 * max(0, 1 - abs(t - tp - 0.12) / 0.15)


OUTLIERS = [12, 18, 27]  # indices of the three weird students


# ---------------------------------------------------------------- plot layer

def draw_plot_layer(t):
    lay = PLOT.new_layer()
    d = ImageDraw.Draw(lay)

    # Region shading once questions are locked.
    t_fail = word(LOCK1, "failed")
    if t >= t_fail:
        a = clamp((t - t_fail) / 0.3)
        d.rectangle([*L(0, 10), *L(TH1, 0)], fill=PINK + (int(45 * a),))
    t_each = word(DONE, "each")
    t_rest = word(SWEEP2, "rest")
    if t_rest <= t < t_each:
        a = clamp((t - t_rest) / 0.3) * (0.6 + 0.4 * abs(math.sin((t - t_rest) * 4)))
        d.rectangle([*L(TH1, 10), *L(10, 0)], outline=WHITE + (int(160 * a),), width=6)
    if t >= t_each:
        a = clamp((t - t_each) / 0.3)
        d.rectangle([*L(TH1, TH2), *L(10, 0)], fill=PINK + (int(45 * a),))
        d.rectangle([*L(TH1, 10), *L(10, TH2)], fill=GREEN + (int(40 * a),))

    # Overfitting: tiny full-depth leaves.
    t_many, t_short = word(OVERFIT, "many"), word(OVERFIT, "short")
    if t_many <= t < t_short + 0.4:
        a = clamp((t - t_many) / 0.25) * (1 - clamp((t - t_short) / 0.4))
        flick = 0.7 + 0.3 * abs(math.sin((t - t_many) * 10))
        for (x0, x1, y0, y1), cls in FULL:
            col = GREEN if cls else PINK
            d.rectangle([*L(x0, y1), *L(x1, y0)], fill=col + (int(55 * a * flick),), outline=YELLOW + (int(220 * a),),
                        width=5)

    # Hook: random candidate cuts flicker.
    if t < T.start(INTRO):
        r = np.random.default_rng(int(t * 5))
        a = 1 - clamp((t - T.start(INTRO) + 0.4) / 0.4)
        for _ in range(2):
            v = r.uniform(1, 9)
            if r.random() < 0.5:
                d.line([L(v, 0), L(v, 10)], fill=WHITE + (int(150 * a),), width=6)
            else:
                d.line([L(0, v), L(10, v)], fill=WHITE + (int(150 * a),), width=6)

    # Sweep 1: vertical cut across everyone.
    s1 = sweep1(t)
    if s1 is not None:
        locked = t >= T.start(LOCK1) + 0.5
        if not locked and t < T.start(LOCK1):
            d.rectangle([*L(0, 10), *L(s1, 0)], fill=WHITE + (14,))
            b = best_so_far(ALL, 0, s1)
            if b:
                dashed_line(d, L(b[1], 0), L(b[1], 10), YELLOW + (170,), width=6, dash=24, gap=16)
        col = YELLOW if locked else WHITE
        flash = 1 + 1.5 * max(0, 1 - (t - T.start(LOCK1) - 0.5) / 0.3) if locked else 1
        d.line([L(s1, 0), L(s1, 10)], fill=col + (255,), width=int(9 * flash))

    # Sweep 2: horizontal cut, right side only.
    s2 = sweep2(t)
    if s2 is not None:
        locked = t >= word(SWEEP2, "slept") + 0.4
        if not locked:
            b = best_so_far(RIGHT, 1, s2)
            if b:
                dashed_line(d, L(TH1, b[1]), L(10, b[1]), YELLOW + (170,), width=6, dash=24, gap=16)
        flash = 1 + 1.5 * max(0, 1 - (t - word(SWEEP2, "slept") - 0.4) / 0.3) if locked else 1
        d.line([L(TH1, s2), L(10, s2)], fill=(YELLOW if locked else WHITE) + (255,), width=int(9 * flash))

    # Students.
    for k in range(N):
        s = ease_back((t - 0.25 - 0.02 * k) / 0.3)
        if s <= 0:
            continue
        col, bump = dot_color(k, t)
        r = 16 * SS * s * bump
        if k in OUTLIERS and word(OVERFIT, "weird") <= t < word(OVERFIT, "short"):
            rr = r * (1.9 + 0.3 * math.sin((t - word(OVERFIT, "weird")) * 9))
            d.ellipse([L(*X[k])[0] - rr, L(*X[k])[1] - rr, L(*X[k])[0] + rr, L(*X[k])[1] + rr],
                      outline=YELLOW + (255,), width=7)
        cx, cy = L(*X[k])
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (255,))
        d.ellipse([cx - r * 0.5, cy - r * 0.5, cx + r * 0.5, cy + r * 0.5], fill=WHITE + (255,))

    # New student star.
    t_six = word(PREDICT, "six")
    if t >= t_six and t < T.start(OVERFIT) + 0.3:
        s = ease_back((t - t_six) / 0.35) * (1 - clamp((t - T.start(OVERFIT)) / 0.3))
        col = lerp_color(YELLOW, PINK, clamp((t - word(PREDICT, "fail")) / 0.2))
        cx, cy = L(*NEW)
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rad = (34 if i % 2 == 0 else 15) * SS * s * (1 + 0.1 * math.sin(t * 8))
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        d.polygon(pts, fill=col + (255,))

    return PLOT.finish_layer(lay)


# ---------------------------------------------------------------- tree diagram

def node_state(t):
    """Which nodes are visible, their labels and pop-in start times."""
    nodes = {}
    t_q = word(GOAL, "questions")
    if t < T.start(INTRO) + 0.3:  # hook: "?" teaser nodes
        tw = word(HOOK, "twenty")
        a = 1 - clamp((t - T.start(INTRO)) / 0.3)
        for i, k in enumerate(("root", "L", "R")):
            nodes[k] = ("?", "q", tw + 0.12 * i, a)
        return nodes
    if t_q <= t < word(LOCK1, "studied"):
        nodes["root"] = ("?", "q", t_q, 1.0)
    if t >= word(LOCK1, "studied"):
        nodes["root"] = (f"Studied < {TH1:g} h?", "q", word(LOCK1, "studied"), 1.0)
    if t >= word(LOCK1, "failed"):
        nodes["L"] = ("FAIL", "fail", word(LOCK1, "failed"), 1.0)
    if t >= word(SWEEP2, "slept"):
        nodes["R"] = (f"Slept < {TH2:g} h?", "q", word(SWEEP2, "slept"), 1.0)
    if t >= word(DONE, "each"):
        nodes["RL"] = ("FAIL", "fail", word(DONE, "each"), 1.0)
        nodes["RR"] = ("PASS", "pass", word(DONE, "each") + 0.15, 1.0)
    return nodes


def path_glow(t):
    """Nodes/edges lit while following the new student down the tree."""
    t_f, t_fail = word(PREDICT, "follow"), word(PREDICT, "fail")
    if not (t_f <= t < T.start(OVERFIT)):
        return set()
    steps = ["root", ("root", "R"), "R", ("R", "RL"), "RL"]
    n = int(clamp((t - t_f) / (t_fail - t_f)) * (len(steps) - 1) + 1e-6) + 1
    return set(steps[:n])


def draw_tree(frame, t):
    x0, y0, x1, y1 = DIAG
    lay = Image.new("RGBA", ((x1 - x0) * 2, (y1 - y0) * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    D = lambda p: ((p[0] - x0) * 2, (p[1] - y0) * 2)  # noqa: E731
    nodes = node_state(t)
    glow = path_glow(t)
    t_tree = word(DONE, "decision")
    pulse = 1 + 0.06 * math.sin((t - t_tree) * 10) * max(0, 1 - (t - t_tree) / 1.2) if t >= t_tree else 1

    for a, b, lab in EDGES:
        if a in nodes and b in nodes:
            g = clamp((t - nodes[b][2]) / 0.3) * nodes[b][3]
            if g <= 0:
                continue
            pa, pb = D(NODE[a]), D(NODE[b])
            pa = (pa[0], pa[1] + 70)
            pb = (pa[0] + (pb[0] - pa[0]) * g, pa[1] + (pb[1] - 70 - pa[1]) * g)
            on = (a, b) in glow
            d.line([pa, pb], fill=(YELLOW if on else MUTED) + (255,), width=14 if on else 8)
            mx, my = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
            text_center(d, (mx + (-84 if pb[0] < pa[0] else 84), my + 6), lab, font("Bold", 62),
                        (YELLOW if on else MUTED) + (int(255 * g),))

    for k, (label, kind, t0, alpha) in nodes.items():
        g = ease_back((t - t0) / 0.35)
        if g <= 0 or alpha <= 0:
            continue
        on = k in glow
        size = 100 if kind != "q" else 90
        tw = font("Black", size).getlength(label)
        wdt, hgt = (tw + 90) * g * pulse, 150 * g * pulse
        cx, cy = D(NODE[k])
        fill = {"q": (18, 22, 48), "fail": PINK, "pass": GREEN}[kind]
        outline = YELLOW if on else (CYAN if kind == "q" else fill)
        if on and kind != "q":
            fill = lerp_color(fill, WHITE, 0.25 * abs(math.sin(t * 9)))
        d.rounded_rectangle([cx - wdt / 2, cy - hgt / 2, cx + wdt / 2, cy + hgt / 2], radius=int(hgt / 2),
                            fill=fill + (int(245 * alpha),), outline=outline + (int(255 * alpha),),
                            width=10 if on else 6)
        if g > 0.6:
            tc = (12, 28, 18) if kind == "pass" else WHITE
            text_center(d, (cx, cy), label, font("Black", int(size * min(1, g) * pulse)),
                        tc + (int(255 * alpha),))

    # Overfitting: the tree sprouts lots of extra questions.
    t_many, t_short = word(OVERFIT, "many"), word(OVERFIT, "short")
    if t_many <= t < t_short + 0.4:
        a = clamp((t - t_many) / 0.2) * (1 - clamp((t - t_short) / 0.4))
        r = np.random.default_rng(5)
        for i in range(14):
            par = NODE[["L", "RL", "RR"][i % 3]]
            g = clamp((t - t_many - 0.08 * i) / 0.25) * a
            if g <= 0:
                continue
            cx = par[0] + r.uniform(-130, 130)
            cy = par[1] + r.uniform(-60, 40)
            pa, pb = D(par), D((cx, cy))
            d.line([pa, pb], fill=YELLOW + (int(180 * g),), width=5)
            rr = 28 * g
            d.ellipse([pb[0] - rr, pb[1] - rr, pb[0] + rr, pb[1] + rr], fill=YELLOW + (int(230 * g),))

    small = lay.resize((x1 - x0, y1 - y0), Image.LANCZOS)
    frame.alpha_composite(small, (x0, y0))


# ---------------------------------------------------------------- overlays

def draw_overlays(frame, t):
    draw_hud_pills(frame, 3, *error_display(t))
    frame.alpha_composite(TITLE_IMG, (0, 215))

    # Live "MIXED" meter riding on the sweeping cut.
    s1 = sweep1(t)
    if s1 is not None and t < T.start(LOCK1) + 0.5:
        score = split_score(ALL, 0, s1)
        m = mixed_pct(score)
        col = lerp_color(GREEN, PINK, clamp((m - 60) / 35))
        paste_scaled(frame, tag(f"MIXED {m}%", (12, 14, 30), col + (255,), 34), (clamp(px(s1), 300, 860), PLOT.y_top - 32))
    s2 = sweep2(t)
    if s2 is not None and t < word(SWEEP2, "slept") + 0.4:
        m = mixed_pct(split_score(RIGHT, 1, s2))
        col = lerp_color(GREEN, PINK, clamp((m - 40) / 50))
        paste_scaled(frame, tag(f"MIXED {m}%", (12, 14, 30), col + (255,), 34), (px(TH1) - 20, py(s2)))
    # Lock tags.
    t_l1 = word(LOCK1, "studied")
    if t_l1 <= t < word(SWEEP2, "rest"):
        pop_in(frame, tag(f"studied < {TH1:g} h", (20, 20, 30), YELLOW + (255,), 34), (px(TH1), PLOT.y_top - 32), t, t_l1)
    t_l2 = word(SWEEP2, "slept")
    if t_l2 + 0.3 <= t < word(DONE, "decision"):
        pop_in(frame, tag(f"slept < {TH2:g} h", (20, 20, 30), YELLOW + (255,), 34), (px(7.0), py(TH2) - 40), t, t_l2 + 0.3)

    # Prediction tags.
    t_six = word(PREDICT, "six")
    if t_six <= t < T.start(OVERFIT):
        lab = "NEW: 6 h study, 4 h sleep"
        pop_in(frame, tag(lab, (20, 20, 30), YELLOW + (255,), 32), (px(6.6), py(NEW[1]) - 62), t, t_six + 0.2)
        t_fail = word(PREDICT, "fail")
        if t >= t_fail:
            pop_in(frame, tag("FAIL  zzz", WHITE, PINK + (255,), 40), (px(6.3), py(NEW[1]) + 62), t, t_fail)

    # Overfitting tag.
    t_mem = word(OVERFIT, "memorizes")
    if t_mem <= t < word(OVERFIT, "short") + 0.3:
        a = 1 - clamp((t - word(OVERFIT, "short")) / 0.3)
        img = tag("OVERFITTING: 0 errors, learned nothing", (20, 20, 30), YELLOW + (255,), 32)
        if a >= 1:
            pop_in(frame, img, (W / 2, PLOT.y_top - 32), t, t_mem)
        else:
            paste_scaled(frame, img, (W / 2, PLOT.y_top - 32), alpha=a)
    t_short = word(OVERFIT, "short")
    if t_short <= t < T.start(CTA):
        pop_in(frame, tag("SHORT TREE = SMART TREE", (12, 28, 18), GREEN + (255,), 34), (W / 2, PLOT.y_top - 32), t,
               t_short)

    draw_level_cleared(frame, t, T.start(CTA), 3, "RANDOM FORESTS", word(CTA, "next"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    frame.alpha_composite(draw_plot_layer(t), (PLOT.layer[0], PLOT.layer[1]))
    draw_tree(frame, t)
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
    for k in range(int(T.start(INTRO) * 5)):  # random-cut clicks in the hook
        mix(out, sfx_blip(1200 + 90 * (k % 4), 0.03, 0.07), k / 5)
    tw = word(HOOK, "twenty")
    for i in range(3):
        mix(out, sfx_blip(700 + 200 * i, 0.08, 0.13), tw + 0.12 * i)
    mix(out, sfx_blip(1046, 0.12, 0.14), word(INTRO, "green"))
    mix(out, sfx_blip(523, 0.12, 0.14), word(INTRO, "pink"))
    for k in range(10):
        mix(out, sfx_blip(1500, 0.03, 0.06), word(GOAL, "job") + 0.06 * k)
    t0, t1 = word(SWEEP1, "tries"), T.end(SWEEP1) + 0.1
    for k in range(24):  # ticks as the cut passes candidate thresholds
        mix(out, sfx_blip(600 + 25 * k, 0.03, 0.06), lerp(t0, t1, k / 24))
    mix(out, sfx_boing(0.6, 0.2), T.start(LOCK1) + 0.45)
    mix(out, sfx_blip(523, 0.18, 0.15), word(LOCK1, "failed"))
    t0, t1 = word(SWEEP2, "again"), word(SWEEP2, "slept")
    for k in range(14):
        mix(out, sfx_blip(700 + 30 * k, 0.03, 0.06), lerp(t0, t1, k / 14))
    mix(out, sfx_boing(0.6, 0.2), t1 + 0.35)
    mix(out, sfx_blip(1046, 0.2, 0.16), word(DONE, "each"))
    mix(out, sfx_blip(1320, 0.25, 0.16), word(DONE, "decision"))
    mix(out, sfx_blip(1568, 0.3, 0.18), word(PREDICT, "six"))
    t_f, t_fail = word(PREDICT, "follow"), word(PREDICT, "fail")
    for k in range(5):
        mix(out, sfx_blip(660 + 110 * k, 0.07, 0.12), lerp(t_f, t_fail, k / 4))
    mix(out, sfx_buzz(0.35, 90, 0.18), t_fail)
    t_m = word(OVERFIT, "many")
    for k in range(14):
        mix(out, sfx_blip(900 + 60 * (k % 5), 0.04, 0.08), t_m + 0.08 * k)
    mix(out, sfx_whoosh(0.5, 0.18, seed=6), word(OVERFIT, "short"))
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12)  # one-line captions: the tree sits just above
    STATIC = build_static(PLOT, list(range(0, 11, 2)), list(range(0, 11, 2)), "HOURS STUDIED", "HOURS SLEPT")
    TITLE_IMG = build_title("DECISION TREES", "ML AS A GAME  ·  EP 3")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = word(DONE, "decision") + 0.8
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 200)), (0, 1440))
    text_center(d, (W / 2, 1540), "20 QUESTIONS AI", font("Black", 110), YELLOW, stroke=8)
    text_center(d, (W / 2, 1680), "Decision Trees, explained", font("Black", 58), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s | {N} students, {int(Y.sum())} passed")
    print(f"Q1: studied < {TH1} (mixed {mixed_pct(GINI0)}% -> {mixed_pct(S1)}%) | Q2: slept < {TH2} | "
          f"errors {ERR[0]} -> {ERR[1]} -> {ERR[2]} | new student -> {'PASS' if NEW_PRED else 'FAIL'} | "
          f"full tree: {len(FULL)} leaves, 0 errors")
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
    out = os.path.join(HERE, "decision_trees_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
