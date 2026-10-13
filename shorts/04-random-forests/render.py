"""ML as a Game, Level 4: Random Forests (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> random_forests_short.mp4 + thumbnail.png
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
    CYAN, FPS, GREEN, H, MUTED, PINK, PURPLE, SR, W, WHITE, YELLOW, PlotBox, Timeline, add_glow,
    build_confetti, build_static, build_title, caption_chunks, clamp, draw_caption, draw_hud_pills,
    draw_level_cleared, ease_back, ease_in_out, ease_out, encode, font, lerp, lerp_color, mix, music_bed,
    num_to_words, paste_scaled, pop_in, sfx_blip, sfx_boing, sfx_buzz, sfx_levelup, sfx_whoosh,
    tag, text_center, voiceover, write_wav,
)

# ---------------------------------------------------------------- data (same students as Level 3)

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
WEIRD = 27  # the student who failed despite studying and sleeping well
NEW = (5.5, 8.0)
N_TREES = 100
SEED = 0
G = 200  # vote-map grid resolution over [0, 10]^2

# ---------------------------------------------------------------- CART + forest


def gini(y):
    if len(y) == 0:
        return 0.0
    p = y.mean()
    return 2 * p * (1 - p)


def best_on(xs, ys, f):
    vals = np.unique(xs[:, f])
    best = None
    for a, b in zip(vals[:-1], vals[1:]):
        th = (a + b) / 2
        left = xs[:, f] < th
        s = (left.sum() * gini(ys[left]) + (~left).sum() * gini(ys[~left])) / len(ys)
        if best is None or s < best[0] - 1e-12:
            best = (s, th)
    return best


def grow(xs, ys, rng=None, box=(0, 10, 0, 10)):
    """Full-depth CART. With rng: each split may only look at one random feature (max_features=1,
    the sqrt(n_features) default for classification), falling back to the other if it can't split.
    Returns leaves as (box, class)."""
    if gini(ys) == 0:
        return [(box, int(ys[0]))]
    order = list(rng.permutation(2)) if rng is not None else [0, 1]
    cands = []
    for f in (order[:1] if rng is not None else order):
        b = best_on(xs, ys, f)
        if b:
            cands.append((b[0], f, b[1]))
    if not cands and rng is not None:
        b = best_on(xs, ys, order[1])
        if b:
            cands.append((b[0], order[1], b[1]))
    if not cands:
        return [(box, int(ys.mean() >= 0.5))]
    _, f, th = min(cands)
    left = xs[:, f] < th
    x0, x1, y0, y1 = box
    lb, rb = ((x0, th, y0, y1), (th, x1, y0, y1)) if f == 0 else ((x0, x1, y0, th), (x0, x1, th, y1))
    return grow(xs[left], ys[left], rng, lb) + grow(xs[~left], ys[~left], rng, rb)


def leaves_grid(leaves):
    """Paint leaf classes into a GxG grid (row 0 = slept 0)."""
    g = np.zeros((G, G), dtype=np.float32)
    for (x0, x1, y0, y1), c in leaves:
        g[int(round(y0 / 10 * G)):int(round(y1 / 10 * G)), int(round(x0 / 10 * G)):int(round(x1 / 10 * G))] = c
    return g


def predict(leaves, p):
    for (x0, x1, y0, y1), c in leaves:
        if x0 <= p[0] < x1 and y0 <= p[1] < y1:
            return c
    raise ValueError(p)


SINGLE = grow(X, Y)  # one plain full-depth tree on everyone
SINGLE_NEW = predict(SINGLE, NEW)
# "Change a few students": the same algorithm with 3 students removed.
VARIANT_DROPS = [(27, 3, 15), (12, 20, 8), (18, 5, 22)]
VARIANTS = [grow(np.delete(X, list(d), 0), np.delete(Y, list(d))) for d in VARIANT_DROPS]

rng = np.random.default_rng(SEED)
BAGS, TREES = [], []
for _ in range(N_TREES):
    idx = rng.integers(0, N, N)  # bootstrap: N draws with replacement
    BAGS.append(idx)
    TREES.append(grow(X[idx], Y[idx], rng))
VOTES = np.array([predict(t, NEW) for t in TREES])
N_PASS = int(VOTES.sum())
assert SINGLE_NEW == 0 and N_PASS > N_TREES // 2, (SINGLE_NEW, N_PASS)
GRIDS = np.stack([leaves_grid(t) for t in TREES])
CUM = np.cumsum(GRIDS, axis=0) / np.arange(1, N_TREES + 1)[:, None, None]  # vote map with first n trees
SINGLE_GRID = leaves_grid(SINGLE)
VARIANT_GRIDS = [leaves_grid(v) for v in VARIANTS]
SHOWN = 5  # trees shown as mini panels

SCRIPT = [
    "One decision tree is easy to fool. A whole forest? Much harder.",
    "Change just a few students, and a single tree asks totally different questions.",
    "So we grow a team. Each tree trains on a random sample of students. Some picked twice, some left out.",
    "And each question only gets to look at one random feature.",
    "So every tree overfits in its own weird way.",
    "But when they all vote, the weird bits cancel out.",
    "New student: five and a half hours studying, eight hours of sleep.",
    f"One tree says fail, fooled by one weird student. But {num_to_words(N_PASS)} out of a hundred trees say pass. "
    "Pass!",
    "That's a random forest: lots of noisy trees, one smart vote.",
    "Level four cleared. Next: k nearest neighbors. Follow for level five.",
]
HOOK, UNSTABLE, BAG, FEATURE, TEAM, VOTE, NEWS, VERDICT, RECAP, CTA = range(10)

# ---------------------------------------------------------------- layout

PLOT = PlotBox(190, 960, 460, 1000, (0, 10), (0, 10), layer=(90, 400, 1010, 1040))
px, py, L = PLOT.px, PLOT.py, PLOT.L
SS = PLOT.ss
CAPTION_Y = 1490
MINI = 160
MINI_Y = 1122
MINI_XS = [80 + i * (MINI + 16) for i in range(SHOWN)]  # left edges; last ends at 944
TALLY_Y = 1360

T: Timeline = None
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0
MINI_IMGS = []


def word(i, prefix, nth=0):
    return T.word(i, prefix, nth)


# ---------------------------------------------------------------- vote-map rendering

def grid_rgba(grid, alpha, crisp):
    """Vote fraction grid (0 fail .. 1 pass) -> RGBA image covering the plot area (2x layer size)."""
    g = grid[::-1]  # image row 0 is the top (slept 10)
    pink, green = np.array(PINK, np.float32), np.array(GREEN, np.float32)
    rgb = pink[None, None] * (1 - g[..., None]) + green[None, None] * g[..., None]
    # Fade toward transparent where the vote is split, so certainty reads as saturation.
    a = (0.45 + 0.55 * np.abs(g - 0.5) * 2) * alpha * 255
    img = Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), "RGBA")
    size = (int((PLOT.x1 - PLOT.x0) * SS), int((PLOT.y_bot - PLOT.y_top) * SS))
    return img.resize(size, Image.NEAREST if crisp else Image.BILINEAR)


def mini_image(k):
    img = Image.new("RGBA", (MINI, MINI), (14, 16, 36, 255))
    g = GRIDS[k][::-1]
    pink, green = np.array(PINK, np.float32), np.array(GREEN, np.float32)
    rgb = pink[None, None] * (1 - g[..., None]) + green[None, None] * g[..., None]
    reg = Image.fromarray(np.dstack([rgb * 0.55, np.full(g.shape, 255)]).astype(np.uint8), "RGBA")
    img.alpha_composite(reg.resize((MINI, MINI), Image.NEAREST))
    d = ImageDraw.Draw(img)
    counts = np.bincount(BAGS[k], minlength=N)
    for i in range(N):
        if counts[i] == 0:
            continue
        cx, cy = X[i, 0] / 10 * MINI, MINI - X[i, 1] / 10 * MINI
        r = 3.5 + 1.5 * (counts[i] - 1)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(GREEN if Y[i] else PINK) + (255,), outline=WHITE + (255,))
    d.rectangle([0, 0, MINI - 1, MINI - 1], outline=(90, 100, 150, 255), width=2)
    text_center(d, (8, 12), f"#{k + 1}", font("Black", 20), WHITE, anchor="lm")
    return img


# ---------------------------------------------------------------- state

def trees_shown(t):
    """How many trees are in the main vote map right now (0 = show single/variant trees)."""
    t_vote = T.start(VOTE)
    if t < t_vote:
        return 0
    return max(1, int(round(1 + (N_TREES - 1) * ease_in_out((t - t_vote - 0.2) / (T.end(VOTE) - t_vote)))))


def counter(t):
    if t < T.start(UNSTABLE) - 0.1:
        return f"{(int(t * FPS) * 7919) % 99 + 1}", PINK
    if t < T.start(TEAM):
        return "1", CYAN
    if t < T.start(VOTE):
        return f"{SHOWN}", CYAN
    n = trees_shown(t)
    return f"{n}", lerp_color(CYAN, GREEN, n / N_TREES)


def main_layer_grid(t):
    """(grid, crisp, alpha) for the main plot background."""
    if t < T.start(UNSTABLE):
        t_forest = word(HOOK, "forest")
        if t < t_forest:
            return GRIDS[int(t * 2.5) % N_TREES], True, 1.0  # flickering single trees
        return CUM[-1], False, clamp((t - t_forest) / 0.5)
    if t < word(UNSTABLE, "few"):
        return SINGLE_GRID, True, 1.0
    if t < T.start(BAG):
        k = int((t - word(UNSTABLE, "few")) / 0.8)
        return VARIANT_GRIDS[k % len(VARIANT_GRIDS)], True, 1.0
    if t < word(FEATURE, "random"):
        return None, True, 0.0
    if t < T.start(VOTE):
        return GRIDS[0], True, clamp((t - word(FEATURE, "random")) / 0.3)
    return CUM[trees_shown(t) - 1], trees_shown(t) < 3, 1.0


def dropped(t):
    """Students removed in the current 'change a few students' variant."""
    t_few = word(UNSTABLE, "few")
    if t_few <= t < T.start(BAG):
        return VARIANT_DROPS[int((t - t_few) / 0.8) % len(VARIANT_DROPS)]
    return ()


# ---------------------------------------------------------------- drawing

def draw_plot_layer(t):
    lay = PLOT.new_layer()
    grid, crisp, alpha = main_layer_grid(t)
    if grid is not None and alpha > 0:
        lay.alpha_composite(grid_rgba(grid, alpha * 0.32, crisp), (int((PLOT.x0 - PLOT.layer[0]) * SS),
                                                                   int((PLOT.y_top - PLOT.layer[1]) * SS)))
    d = ImageDraw.Draw(lay)

    # Bootstrap sample of tree #1: duplicates get count badges, left-out students fade.
    in_bag = t >= word(BAG, "random") and t < T.start(VOTE)
    counts = np.bincount(BAGS[0], minlength=N)
    drop = dropped(t)
    t_weird = word(VERDICT, "fooled")
    for k in range(N):
        s = ease_back((t - 0.2 - 0.02 * k) / 0.3)
        if s <= 0:
            continue
        cx, cy = L(*X[k])
        col = GREEN if Y[k] else PINK
        fade = 1.0
        if in_bag and counts[k] == 0:
            fade = 0.25 if t >= word(BAG, "left") else 0.5
        if k in drop:
            fade = 0.2
        r = 16 * SS * s * (1 + 0.15 * (counts[k] - 1) if in_bag and counts[k] > 1 and t >= word(BAG, "twice") else 1)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (int(255 * fade),))
        d.ellipse([cx - r * 0.5, cy - r * 0.5, cx + r * 0.5, cy + r * 0.5], fill=WHITE + (int(255 * fade),))
        if k in drop:
            q = r * 1.2
            d.line([(cx - q, cy - q), (cx + q, cy + q)], fill=WHITE + (230,), width=7)
            d.line([(cx - q, cy + q), (cx + q, cy - q)], fill=WHITE + (230,), width=7)
        if k == WEIRD and t_weird <= t < T.start(RECAP):
            rr = r * (2.0 + 0.3 * math.sin((t - t_weird) * 9))
            d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=YELLOW + (255,), width=8)

    # Single tree's fail strip around the weird student ("one tree says fail").
    t_one = word(VERDICT, "one")
    if t_one <= t < word(VERDICT, "but"):
        a = clamp((t - t_one) / 0.25)
        for (x0, x1, y0, y1), c in SINGLE:
            if x0 <= NEW[0] < x1 and y0 <= NEW[1] < y1:
                d.rectangle([*L(x0, y1), *L(x1, y0)], fill=PINK + (int(110 * a),), outline=YELLOW + (int(255 * a),),
                            width=8)

    # New student star.
    t_new = word(NEWS, "five")
    if t >= t_new and t < T.start(CTA):
        s = ease_back((t - t_new) / 0.35)
        col = lerp_color(YELLOW, GREEN, clamp((t - word(VERDICT, "pass!")) / 0.2))
        cx, cy = L(*NEW)
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            rad = (36 if i % 2 == 0 else 16) * SS * s * (1 + 0.1 * math.sin(t * 8))
            pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
        d.polygon(pts, fill=col + (255,), outline=(20, 20, 30, 255))
    return PLOT.finish_layer(lay, glow=10, strength=1.2)


def draw_minis(frame, t):
    t_team = T.start(TEAM)
    if t < t_team:
        return
    t_votes = word(VERDICT, "but")
    t_n = word(VERDICT, num_to_words(N_PASS).split("-")[0][:5])
    for k in range(SHOWN):
        t0 = t_team + 0.15 * k
        if t < t0:
            continue
        g = ease_back((t - t0) / 0.35)
        cx, cy = MINI_XS[k] + MINI / 2, MINI_Y + MINI / 2
        img = MINI_IMGS[k].copy()
        if t >= word(NEWS, "five"):
            d = ImageDraw.Draw(img)
            sx, sy = NEW[0] / 10 * MINI, MINI - NEW[1] / 10 * MINI
            d.regular_polygon((sx, sy, 9), 5, fill=YELLOW, outline=(20, 20, 30))
        paste_scaled(frame, img, (cx, cy), scale=g, alpha=clamp(g))
        if t >= t_votes + 0.12 * k:
            v = VOTES[k]
            pop_in(frame, tag("PASS" if v else "FAIL", (12, 28, 18) if v else WHITE, (GREEN if v else PINK) + (255,),
                              28, pad=(16, 8)), (cx, MINI_Y + MINI + 34), t, t_votes + 0.12 * k)
    # Tally across all 100 trees.
    if t >= t_n - 0.2:
        g = ease_out((t - t_n + 0.2) / 0.8)
        d = ImageDraw.Draw(frame)
        x0, x1 = MINI_XS[0], MINI_XS[-1] + MINI
        d.rounded_rectangle((x0, TALLY_Y, x1, TALLY_Y + 52), radius=26, fill=PINK + (255,))
        xp = x0 + (x1 - x0) * N_PASS / N_TREES * g
        if xp - x0 > 30:
            d.rounded_rectangle((x0, TALLY_Y, xp, TALLY_Y + 52), radius=26, fill=GREEN + (255,))
        text_center(d, (x0 + 24, TALLY_Y + 26), f"PASS {int(round(N_PASS * g))}", font("Black", 32), (12, 28, 18),
                    anchor="lm")
        text_center(d, (x1 - 24, TALLY_Y + 26), f"FAIL {int(round((N_TREES - N_PASS) * g))}", font("Black", 32),
                    WHITE, anchor="rm")


def draw_overlays(frame, t):
    draw_hud_pills(frame, 4, *counter(t), label="TREES")

    # Title swaps to the one-line summary in the recap.
    t_rec = word(RECAP, "lots")
    if t < t_rec:
        frame.alpha_composite(TITLE_IMG, (0, 215))
    else:
        g = ease_back((t - t_rec) / 0.45)
        card = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
        cd = ImageDraw.Draw(card)
        f = font("Black", 76)
        parts = [("100 trees", CYAN), (" → ", WHITE), ("1 vote", GREEN)]
        x = (W - sum(f.getlength(p) for p, _ in parts)) / 2
        for p, c in parts:
            cd.text((x, 80), p, font=f, fill=c, anchor="lm")
            x += f.getlength(p)
        text_center(cd, (W / 2, 150), "lots of noisy trees, one smart vote", font("Bold", 32), MUTED)
        paste_scaled(frame, add_glow(card, radius=10, strength=1.2), (W / 2, 315), scale=0.6 + 0.4 * g, alpha=clamp(g))

    top_tag_y = PLOT.y_top - 32
    # Hook tags.
    if t < T.start(UNSTABLE):
        t_f = word(HOOK, "forest")
        if t < t_f:
            paste_scaled(frame, tag("1 TREE", WHITE, PINK + (255,), 34), (W / 2, top_tag_y))
        else:
            pop_in(frame, tag(f"{N_TREES} TREES", (12, 28, 18), GREEN + (255,), 34), (W / 2, top_tag_y), t, t_f)
    # Instability.
    t_few = word(UNSTABLE, "few")
    if t_few <= t < T.start(BAG):
        k = int((t - t_few) / 0.8) % len(VARIANT_DROPS)
        paste_scaled(frame, tag(f"3 students removed  ·  version {k + 1}", (20, 20, 30), YELLOW + (255,), 32),
                     (W / 2, top_tag_y))
        pop_in(frame, tag("UNSTABLE!", WHITE, PINK + (255,), 40), (px(7.4), py(1.2)), t, word(UNSTABLE, "totally"))
    # Bootstrap.
    t_rs = word(BAG, "random")
    if t_rs <= t < T.start(FEATURE):
        paste_scaled(frame, tag("TREE #1's RANDOM SAMPLE", (20, 20, 30), YELLOW + (255,), 32), (W / 2, top_tag_y))
        if t >= word(BAG, "twice"):
            counts = np.bincount(BAGS[0], minlength=N)
            for k in np.where(counts > 1)[0]:
                paste_scaled(frame, tag(f"×{counts[k]}", (20, 20, 30), WHITE + (255,), 24, pad=(10, 4)),
                             (px(X[k, 0]) + 26, py(X[k, 1]) - 26))
    # Random feature: a "dice roll" between the two questions it may ask.
    t_ft = word(FEATURE, "only")
    if t_ft <= t < T.start(TEAM):
        rolling = t < word(FEATURE, "feature") + 0.3
        k = int(t * 8) % 2 if rolling else 1
        txt = ["CAN ONLY ASK: STUDIED?", "CAN ONLY ASK: SLEPT?"][k]
        paste_scaled(frame, tag(txt, WHITE, PURPLE + (255,), 32), (W / 2, top_tag_y))
    # Team / vote tags.
    if T.start(TEAM) <= t < T.start(VOTE):
        paste_scaled(frame, tag("EACH TREE: OWN WEIRD BOXES", (20, 20, 30), YELLOW + (255,), 32), (W / 2, top_tag_y))
    if T.start(VOTE) <= t < T.start(NEWS):
        n = trees_shown(t)
        paste_scaled(frame, tag(f"AVERAGE VOTE OF {n} TREE{'S' if n > 1 else ''}", (12, 28, 18), GREEN + (255,), 32),
                     (W / 2, top_tag_y))
    # Verdict tags.
    t_one, t_but = word(VERDICT, "one"), word(VERDICT, "but")
    if t_one <= t < t_but + 0.2:
        pop_in(frame, tag("1 TREE: FAIL", WHITE, PINK + (255,), 36), (W / 2, top_tag_y), t, t_one)
    t_pass = word(VERDICT, "pass!")
    if t_but + 0.2 <= t < T.start(CTA):
        paste_scaled(frame, tag(f"FOREST: {N_PASS}% PASS", (12, 28, 18), GREEN + (255,), 36), (W / 2, top_tag_y))
    if t_pass <= t < T.start(CTA):
        pop_in(frame, tag("PASS", (12, 28, 18), GREEN + (255,), 44), (px(NEW[0]) + 130, py(NEW[1])), t, t_pass)

    draw_minis(frame, t)
    draw_level_cleared(frame, t, T.start(CTA), 4, "K-NEAREST NEIGHBORS", word(CTA, "next"), CONFETTI)


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
    t_f = word(HOOK, "forest")
    for k in range(int(t_f * 2.5)):
        mix(out, sfx_blip(1100 + 80 * (k % 3), 0.03, 0.07), k / 2.5)
    mix(out, sfx_whoosh(0.8, 0.2, seed=3), t_f)
    t_few = word(UNSTABLE, "few")
    for k in range(int((T.start(BAG) - t_few) / 0.8) + 1):
        mix(out, sfx_buzz(0.08, 140, 0.1), t_few + 0.8 * k)
    mix(out, sfx_buzz(0.3, 90, 0.16), word(UNSTABLE, "totally"))
    t_r = word(BAG, "random")
    for k in range(10):
        mix(out, sfx_blip(800 + 50 * k, 0.04, 0.08), t_r + 0.05 * k)
    mix(out, sfx_blip(1320, 0.12, 0.12), word(BAG, "twice"))
    mix(out, sfx_blip(440, 0.15, 0.12), word(BAG, "left"))
    t_ft = word(FEATURE, "only")
    for k in range(int((word(FEATURE, "feature") + 0.3 - t_ft) * 8)):
        mix(out, sfx_blip(1500 - 200 * (k % 2), 0.025, 0.06), t_ft + k / 8)
    for k in range(SHOWN):
        mix(out, sfx_blip(600 + 120 * k, 0.08, 0.13), T.start(TEAM) + 0.15 * k)
    mix(out, sfx_whoosh(1.0, 0.2, seed=4), T.start(VOTE) + 0.1)
    for k in range(20):
        mix(out, sfx_blip(700 + 30 * k, 0.03, 0.06), lerp(T.start(VOTE) + 0.2, T.end(VOTE), k / 20))
    mix(out, sfx_blip(1568, 0.3, 0.18), word(NEWS, "five"))
    mix(out, sfx_buzz(0.3, 100, 0.16), word(VERDICT, "fail"))
    for k in range(SHOWN):
        mix(out, sfx_blip(1046 if VOTES[k] else 523, 0.07, 0.12), word(VERDICT, "but") + 0.12 * k)
    mix(out, sfx_boing(0.6, 0.18), word(VERDICT, num_to_words(N_PASS).split("-")[0][:5]))
    mix(out, sfx_blip(1568, 0.35, 0.2), word(VERDICT, "pass!"))
    mix(out, sfx_blip(1320, 0.25, 0.15), word(RECAP, "lots"))
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION, MINI_IMGS
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12, max_px=800)  # one line: mini panels sit above
    STATIC = build_static(PLOT, list(range(0, 11, 2)), list(range(0, 11, 2)), "HOURS STUDIED", "HOURS SLEPT")
    TITLE_IMG = build_title("RANDOM FORESTS", "ML AS A GAME  ·  EP 4")
    CONFETTI = build_confetti()
    MINI_IMGS = [mini_image(k) for k in range(SHOWN)]
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = word(VERDICT, "pass!") + 0.6
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 215)), (0, 1440))
    text_center(d, (W / 2, 1540), "100 TREES VOTE", font("Black", 116), YELLOW, stroke=8)
    text_center(d, (W / 2, 1680), "Random Forests, explained", font("Black", 58), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    leaves = [len(t) for t in TREES]
    print(f"duration {DURATION:.2f}s | single tree: {len(SINGLE)} leaves, says "
          f"{'PASS' if SINGLE_NEW else 'FAIL'} at {NEW} | forest (seed {SEED}): {N_PASS}/{N_TREES} PASS | "
          f"tree leaves {min(leaves)}-{max(leaves)} | first {SHOWN} votes {VOTES[:SHOWN].tolist()} | "
          f"tree #1 sample: {int((np.bincount(BAGS[0], minlength=N) == 0).sum())} left out")
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
    out = os.path.join(HERE, "random_forests_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
