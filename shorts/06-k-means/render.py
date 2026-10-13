"""ML as a Game, Level 6: K-Means Clustering (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> kmeans_short.mp4 + thumbnail.png
    python3 render.py --preview 3,12  # just dump PNG frames at those seconds (fast check)
    python3 render.py --thumbnail     # only regenerate thumbnail.png
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import (  # noqa: E402
    CYAN, FPS, GREEN, H, MUTED, ORANGE, PINK, PURPLE, SR, W, WHITE, YELLOW, PlotBox, Timeline,
    build_confetti, build_static, build_title, caption_chunks, clamp, dashed_line, draw_caption, draw_hud_pills,
    draw_level_cleared, ease_back, ease_in_out, encode, font, lerp, lerp_color, mix, music_bed,
    paste_scaled, pop_in, rounded_pill, sfx_blip, sfx_boing, sfx_levelup, sfx_whoosh, tag, text_center, voiceover,
    write_wav,
)

# ---------------------------------------------------------------- data: three loose study styles, no labels

_r = np.random.default_rng(42)


def _blob(n, xr, yr):
    return np.column_stack([_r.uniform(*xr, n), _r.uniform(*yr, n)])


X = np.vstack([
    _blob(10, (7.0, 9.5), (2.8, 5.0)),  # studies a lot, sleeps little
    _blob(10, (0.8, 3.0), (7.8, 9.6)),  # sleeps a lot, studies little
    _blob(10, (4.3, 6.5), (5.6, 7.8)),  # in between
])
DESIGNED = np.repeat([0, 1, 2], 10)  # only used to check the result; k-means never sees it
N = len(X)
K = 3
FLAG_SEED = 21  # random flag positions; see script.md


# ---------------------------------------------------------------- k-means (Lloyd's algorithm)

def kmeans_rounds(init):
    """Each round: everyone joins the closest flag, then each flag moves to its group's average.
    Returns a list of rounds: dict(flags_before, assign, switched, flags_after)."""
    flags, assign, rounds = init.copy(), None, []
    for _ in range(50):
        d = ((X[:, None] - flags[None]) ** 2).sum(2)
        a = d.argmin(1)
        switched = N if assign is None else int((a != assign).sum())
        new = np.array([X[a == k].mean(0) if (a == k).any() else flags[k] for k in range(K)])
        rounds.append(dict(flags_before=flags.copy(), assign=a.copy(), switched=switched, flags_after=new))
        if assign is not None and switched == 0:
            break
        assign, flags = a, new
    return rounds


def spread(flags, assign):
    """Total squared distance from each student to their flag: the thing k-means shrinks."""
    return float(((X - flags[assign]) ** 2).sum())


INIT = np.random.default_rng(FLAG_SEED).uniform(1, 9, (K, 2))
ROUNDS = kmeans_rounds(INIT)
NR = len(ROUNDS)
FINAL = ROUNDS[-1]["flags_before"]
FINAL_ASSIGN = ROUNDS[-1]["assign"]
SPREADS = []
for rd in ROUNDS:
    SPREADS += [spread(rd["flags_before"], rd["assign"]), spread(rd["flags_after"], rd["assign"])]
assert all(a >= b - 1e-9 for a, b in zip(SPREADS, SPREADS[1:])), SPREADS  # never goes up
assert ROUNDS[-1]["switched"] == 0
assert all(len(set(FINAL_ASSIGN[DESIGNED == g])) == 1 for g in range(3)) and len(set(FINAL_ASSIGN)) == 3
assert not all(len(set(ROUNDS[0]["assign"][DESIGNED == g])) == 1 for g in range(3))  # first try is wrong

# Name each final group by where its flag ended up.
_owl = int(np.argmax(FINAL[:, 0]))
_sleepy = int(np.argmax(FINAL[:, 1]))
_bal = ({0, 1, 2} - {_owl, _sleepy}).pop()
NAMES = {_owl: "NIGHT OWLS", _sleepy: "SLEEPYHEADS", _bal: "BALANCED"}
COLORS = [CYAN, ORANGE, PURPLE]
GRAY = (130, 138, 170)

SCRIPT = [
    "No answers this time. Can a computer find groups all by itself?",
    "Here are some students: hours studied and hours slept. No pass or fail. Just dots.",
    "K-means sorts them into K groups. Let's pick K equals three.",
    "Step one: drop three flags at random spots.",
    "Step two: every student joins the closest flag.",
    "Step three: each flag moves to the middle of its group.",
    "Then repeat. Join, move, join, move, until nobody switches.",
    "Done! It found three study styles: night owls, sleepyheads, and balanced.",
    "Don't mix it up with KNN. KNN copies answers from labeled neighbors. K-means has no answers. "
    "It invents the groups.",
    "Level six cleared. Next: neural networks. Follow for level seven.",
]
HOOK, DOTS, PICKK, STEP1, STEP2, STEP3, REPEAT, DONE, VSKNN, CTA = range(10)

# ---------------------------------------------------------------- layout (square plot, as in Level 5)

PLOT = PlotBox(210, 890, 440, 1120, (0, 10), (0, 10), layer=(110, 330, 960, 1160))
px, py, L = PLOT.px, PLOT.py, PLOT.L
SS = PLOT.ss
CAPTION_Y = 1495
TOP_TAG_Y = PLOT.y_top - 32
CARD_Y = 1236  # algorithm card top
GT = 100  # territory grid resolution

T: Timeline = None
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0
PHASES = []  # (kind, round, t0, t1)


def word(i, prefix, nth=0):
    return T.word(i, prefix, nth)


def build_phases():
    """Map each k-means step onto the narration."""
    ph = [("join", 0, word(STEP2, "every"), T.end(STEP2)), ("move", 0, word(STEP3, "each"), T.end(STEP3))]
    t0, t1 = word(REPEAT, "join"), T.end(REPEAT)
    slots = 2 * (NR - 1) - 1  # rounds 2..NR: join+move, last round only joins (nobody switches)
    dt = (t1 - t0) / slots
    for j in range(slots):
        r = 1 + j // 2
        ph.append(("join" if j % 2 == 0 else "move", r, t0 + j * dt, t0 + (j + 1) * dt))
    return ph


# ---------------------------------------------------------------- state

def flags_at(t):
    """Current flag positions (K x 2) or None before they drop; plus drop progress per flag."""
    t_drop = word(STEP1, "drop")
    if t < T.start(DOTS):  # hook: flags hop around randomly
        r = np.random.default_rng(int(t * 2.5) + 100)
        return r.uniform(1.5, 8.5, (K, 2)), np.ones(K)
    if t < t_drop:
        return None, None
    drop = np.array([ease_back((t - t_drop - 0.25 * k) / 0.45) for k in range(K)])
    pos = ROUNDS[0]["flags_before"].copy()
    for kind, r, a, b in PHASES:
        if kind != "move" or t < a:
            continue
        p = ease_in_out((t - a - 0.15 * (r == 0)) / ((b - a) * (0.55 if r == 0 else 0.8)))
        pos = ROUNDS[r]["flags_before"] + (ROUNDS[r]["flags_after"] - ROUNDS[r]["flags_before"]) * p
    return pos, drop


def assignment_at(t):
    """(color index per student or -1 for none, current round index, phase kind, phase progress)."""
    cur = -np.ones(N, int)
    info = (None, None, 0.0)
    for kind, r, a, b in PHASES:
        if t < a:
            break
        p = clamp((t - a) / (b - a))
        info = (kind, r, p)
        if kind == "join":
            prev = cur.copy()
            new = ROUNDS[r]["assign"]
            span = 0.75 if r == 0 else 0.6
            order = np.argsort(X[:, 0])  # sweep left to right in round 1
            rank = np.empty(N, int)
            rank[order] = np.arange(N)
            done = (rank / N) <= p / span if r == 0 else np.full(N, p >= 0.15)
            cur = np.where(done, new, prev)
    return cur, info


def round_number(t):
    _, (kind, r, _) = assignment_at(t)
    if r is None:
        return None
    return r + 1


# ---------------------------------------------------------------- drawing

def territory(flags, alpha):
    c = (np.arange(GT) + 0.5) / GT * 10
    gx, gy = np.meshgrid(c, c[::-1])
    d = (gx[..., None] - flags[:, 0]) ** 2 + (gy[..., None] - flags[:, 1]) ** 2
    idx = d.argmin(-1)
    pal = np.array(COLORS, np.float32)
    rgb = pal[idx]
    a = np.full(idx.shape, 40 * alpha)
    img = Image.fromarray(np.dstack([rgb, a]).astype(np.uint8), "RGBA")
    return img.resize((int((PLOT.x1 - PLOT.x0) * SS), int((PLOT.y_bot - PLOT.y_top) * SS)), Image.NEAREST)


def draw_flag(d, x, y, col, s=1.0, done=False):
    """A flag whose pole base sits exactly on the flag's (x, y) position."""
    bx, by = L(x, y)
    hgt, w = 80 * SS * s, 54 * SS * s
    d.ellipse([bx - 12 * SS * s, by - 12 * SS * s, bx + 12 * SS * s, by + 12 * SS * s], fill=col + (255,),
              outline=WHITE + (255,), width=int(3 * SS))
    d.line([(bx, by), (bx, by - hgt)], fill=WHITE + (255,), width=int(5 * SS * max(s, 0.3)))
    d.polygon([(bx, by - hgt), (bx + w, by - hgt + w * 0.38), (bx, by - hgt + w * 0.76)], fill=col + (255,),
              outline=WHITE + (255,))
    if done:
        d.ellipse([bx - 30 * SS, by - 30 * SS, bx + 30 * SS, by + 30 * SS], outline=YELLOW + (230,), width=int(3 * SS))


def draw_plot_layer(t):
    lay = PLOT.new_layer()
    flags, drop = flags_at(t)
    cur, (kind, r, p) = assignment_at(t)
    t_join0 = PHASES[0][2]
    if flags is not None and t >= t_join0:
        a = clamp((t - t_join0 - 0.6) / 0.6)
        lay.alpha_composite(territory(flags, a), (int((PLOT.x0 - PLOT.layer[0]) * SS),
                                                   int((PLOT.y_top - PLOT.layer[1]) * SS)))
    d = ImageDraw.Draw(lay)

    # "Join" lines from each student to the flag they joined.
    if kind == "join" and flags is not None and t < T.start(DONE):
        fade = 1 - clamp((p - 0.75) / 0.25) if r > 0 else 1 - clamp((t - T.end(STEP2) + 0.2) / 0.3)
        for i in range(N):
            if cur[i] >= 0:
                d.line([L(*X[i]), L(*flags[cur[i]])], fill=COLORS[cur[i]] + (int(170 * fade),), width=5)

    # "Move" step: the group average (+) and the trail the flag takes to get there.
    if kind == "move" and t < T.start(DONE):
        rd = ROUNDS[r]
        for k in range(K):
            ax, ay = L(*rd["flags_after"][k])
            g = clamp(p / 0.2)
            q = 26 * SS * g
            d.line([(ax - q, ay), (ax + q, ay)], fill=WHITE + (255,), width=int(4 * SS))
            d.line([(ax, ay - q), (ax, ay + q)], fill=WHITE + (255,), width=int(4 * SS))
            dashed_line(d, L(*rd["flags_before"][k]), L(*flags[k]), COLORS[k] + (220,), width=int(3 * SS),
                        dash=18 * SS, gap=10 * SS)

    # Students.
    for i in range(N):
        s = ease_back((t - 0.15 - 0.012 * i) / 0.3)
        if s <= 0:
            continue
        cx, cy = L(*X[i])
        col = GRAY if cur[i] < 0 else COLORS[cur[i]]
        rr = 15 * SS * s
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=col + (255,))
        d.ellipse([cx - rr * 0.45, cy - rr * 0.45, cx + rr * 0.45, cy + rr * 0.45], fill=WHITE + (255,))

    # Flags.
    if flags is not None and t < T.start(CTA):
        hook = t < T.start(DOTS)
        for k in range(K):
            s = 1.0 if hook else float(drop[k])
            if s <= 0:
                continue
            fx, fy = flags[k]
            if not hook and s < 1:  # falling in from above
                fy = fy + (1 - min(1, s)) * 3
            draw_flag(d, fx, fy, COLORS[k], s=min(1.2, s) * (0.8 if hook else 1.0),
                      done=t >= word(REPEAT, "nobody") + 0.4)
    return PLOT.finish_layer(lay, glow=10, strength=1.3)


def draw_card(frame, t):
    """The algorithm card, with the active step highlighted."""
    t0 = word(PICKK, "k-means")
    if t < t0 or t >= T.start(VSKNN):
        return
    a = clamp((t - t0) / 0.4)
    d = ImageDraw.Draw(frame)
    rows = [("1", "Drop K flags", word(STEP1, "step"), word(STEP2, "step")),
            ("2", "Everyone joins the closest flag", word(STEP2, "step"), word(STEP3, "step")),
            ("3", "Flags move to the middle", word(STEP3, "step"), T.start(REPEAT)),
            ("↻", "Repeat until nobody switches", T.start(REPEAT), T.start(DONE))]
    # During the repeat, highlight join/move rows as they happen.
    kind = assignment_at(t)[1][0] if T.start(REPEAT) <= t < T.start(DONE) else None
    for j, (num, text, ta, tb) in enumerate(rows):
        y = CARD_Y + 22 + j * 50
        on = ta <= t < tb or (kind == "join" and j == 1) or (kind == "move" and j == 2) or \
            (T.start(REPEAT) <= t < T.start(DONE) and j == 3)
        if on:
            d.rounded_rectangle((92, y - 22, 712, y + 22), radius=22, fill=(48, 52, 96))
        col = YELLOW if on else lerp_color((40, 44, 70), MUTED, a)
        text_center(d, (122, y), num, font("Black", 32), col)
        text_center(d, (150, y), text, font("Bold", 32), col if on else lerp_color((40, 44, 70), WHITE, a * 0.85),
                    anchor="lm")

    # SPREAD meter: total squared distance to your flag; it only ever shrinks.
    cur, (_, r, _) = assignment_at(t)
    flags, _ = flags_at(t)
    x0, x1, y0 = 740, 930, CARD_Y
    d.rounded_rectangle((x0, y0, x1, y0 + 196), radius=18, fill=(18, 22, 48), outline=lerp_color((18, 22, 48),
                                                                                                    (70, 78, 120), a),
                        width=3)
    text_center(d, ((x0 + x1) / 2, y0 + 30), "SPREAD", font("Bold", 28), lerp_color((18, 22, 48), MUTED, a))
    if r is not None and (cur >= 0).all():
        sp = spread(flags, cur)
        frac = sp / SPREADS[0]
        col = lerp_color(PINK, GREEN, clamp(1 - frac) ** 0.6)
        text_center(d, ((x0 + x1) / 2, y0 + 88), f"{sp:.0f}", font("Black", 60), col)
        d.rounded_rectangle((x0 + 20, y0 + 140, x1 - 20, y0 + 158), radius=9, fill=(48, 52, 96))
        d.rounded_rectangle((x0 + 20, y0 + 140, x0 + 20 + (x1 - x0 - 40) * clamp(frac), y0 + 158), radius=9,
                            fill=col)
        text_center(d, ((x0 + x1) / 2, y0 + 180), "smaller = tighter", font("Bold", 20), MUTED)
    else:
        text_center(d, ((x0 + x1) / 2, y0 + 92), "—", font("Black", 60), lerp_color((18, 22, 48), MUTED, a))


def draw_compare(frame, t):
    t0 = T.start(VSKNN)
    if not (t0 <= t < T.start(CTA)):
        return
    cards = [
        (word(VSKNN, "knn", 1), 90, "KNN · LEVEL 5", "has answers", "copies labeled neighbors", (GREEN, PINK)),
        (word(VSKNN, "k-means", 0), 530, "K-MEANS · LEVEL 6", "no answers", "invents the groups", (CYAN, ORANGE)),
    ]
    for ta, x, title, l1, l2, cols in cards:
        if t < ta:
            continue
        img = Image.new("RGBA", (410, 200), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        rounded_pill(d, (0, 0, 409, 199), fill=(18, 22, 48, 245), outline=cols[0] + (255,), width=5, radius=26)
        text_center(d, (205, 40), title, font("Black", 36), cols[0])
        for j, c in enumerate(cols):
            d.ellipse((150 + j * 70, 80, 176 + j * 70, 106), fill=c)
            if title.startswith("K-MEANS"):
                d.ellipse((185 + j * 70, 72, 197 + j * 70, 84), fill=MUTED)
        text_center(d, (205, 132), l1, font("Black", 30), WHITE)
        text_center(d, (205, 170), l2, font("Bold", 26), MUTED)
        pop_in(frame, img, (x + 205, CARD_Y + 100), t, ta, dur=0.35)


def draw_overlays(frame, t):
    rn = round_number(t)
    if t < T.start(DOTS) - 0.1:
        hud = (f"{(int(t * FPS) * 7919) % 9 + 1}", PINK)
    elif rn is None:
        hud = ("—", MUTED)
    else:
        hud = (f"{rn}", GREEN if t >= word(REPEAT, "nobody") + 0.4 else CYAN)
    draw_hud_pills(frame, 6, *hud, label="ROUND")
    frame.alpha_composite(TITLE_IMG, (0, 190))

    # Step banners at the top of the plot.
    cur, (kind, r, p) = assignment_at(t)
    rep_text = None
    if T.start(REPEAT) <= t < word(REPEAT, "nobody") + 0.4 and r is not None and r >= 1:
        sw = ROUNDS[r]["switched"] if kind == "join" else ROUNDS[r]["switched"]
        rep_text = f"ROUND {r + 1} · {'JOIN' if kind == 'join' else 'MOVE'} · switched: {sw}"
    banners = [
        (T.start(HOOK), T.start(DOTS), "NO ANSWERS. FIND THE GROUPS!", PURPLE, WHITE),
        (word(DOTS, "no"), T.start(PICKK), "NO LABELS · JUST DOTS", MUTED, (12, 14, 30)),
        (word(PICKK, "three"), T.start(STEP1), f"K = {K} GROUPS", YELLOW, (20, 20, 30)),
        (word(STEP1, "step"), T.start(STEP2), "STEP 1 · DROP 3 FLAGS AT RANDOM", CYAN, (12, 14, 30)),
        (word(STEP2, "step"), T.start(STEP3), "STEP 2 · JOIN THE CLOSEST FLAG", CYAN, (12, 14, 30)),
        (word(STEP3, "step"), T.start(REPEAT), "STEP 3 · FLAG → MIDDLE OF ITS GROUP", CYAN, (12, 14, 30)),
        (word(REPEAT, "nobody") + 0.4, T.start(VSKNN), "NOBODY SWITCHED → DONE ✓", GREEN, (12, 28, 18)),
        (T.start(VSKNN), T.start(CTA), "KNN ≠ K-MEANS", PINK, WHITE),
    ]
    if rep_text and t >= T.start(REPEAT):
        paste_scaled(frame, tag(rep_text, (20, 20, 30), YELLOW + (255,), 32), (px(5), TOP_TAG_Y))
    elif T.start(REPEAT) <= t < word(REPEAT, "join"):
        paste_scaled(frame, tag("REPEAT ↻", (20, 20, 30), YELLOW + (255,), 32), (px(5), TOP_TAG_Y))
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 32), (px(5), TOP_TAG_Y), t, a)

    # "Middle of its group" label on the first move.
    t_mid = word(STEP3, "middle")
    if t_mid <= t < T.start(REPEAT):
        rd = ROUNDS[0]
        k = int(np.argmax(np.linalg.norm(rd["flags_after"] - rd["flags_before"], axis=1)))
        ax, ay = rd["flags_after"][k]
        pop_in(frame, tag("+ = average", (20, 20, 30), WHITE + (255,), 26, pad=(12, 6)),
               (px(ax), py(ay) + 44), t, t_mid)

    # Group names.
    t_found = word(DONE, "night")
    for j, k in enumerate((_owl, _sleepy, _bal)):
        ta = t_found + [0, word(DONE, "sleepyheads") - t_found, word(DONE, "balanced") - t_found][j]
        if ta <= t < T.start(CTA):
            fx, fy = FINAL[k]
            dy = -120 if k == _bal else 150  # put each name in empty space next to its group
            pop_in(frame, tag(NAMES[k], (12, 14, 30), COLORS[k] + (255,), 32), (px(fx), py(fy) + dy), t, ta)

    draw_card(frame, t)
    draw_compare(frame, t)
    draw_level_cleared(frame, t, T.start(CTA), 6, "NEURAL NETWORKS", word(CTA, "next"), CONFETTI)


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
    for k in range(int(T.start(DOTS) * 2.5)):
        mix(out, sfx_blip(1000 + 120 * (k % 3), 0.04, 0.07), k / 2.5)
    mix(out, sfx_blip(1046, 0.15, 0.12), word(PICKK, "three"))
    t_drop = word(STEP1, "drop")
    for k in range(K):
        mix(out, sfx_boing(0.4, 0.15), t_drop + 0.25 * k + 0.3)
    for kind, r, a, b in PHASES:
        if kind == "join":
            n = 12 if r == 0 else 4
            for j in range(n):
                mix(out, sfx_blip(600 + 40 * j, 0.03, 0.06), lerp(a, b, j / n * (0.75 if r == 0 else 0.4)))
        else:
            mix(out, sfx_whoosh(min(0.6, b - a), 0.14, seed=r + 3), a)
    mix(out, sfx_blip(1568, 0.35, 0.2), word(REPEAT, "nobody") + 0.4)
    for j, w in enumerate(("night", "sleepyheads", "balanced")):
        mix(out, sfx_blip(880 + 220 * j, 0.12, 0.14), word(DONE, w))
    mix(out, sfx_blip(1046, 0.15, 0.14), word(VSKNN, "knn", 1))
    mix(out, sfx_blip(1320, 0.15, 0.14), word(VSKNN, "k-means"))
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION, PHASES
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12, max_px=800)
    STATIC = build_static(PLOT, list(range(0, 11, 2)), list(range(0, 11, 2)), "HOURS STUDIED", "HOURS SLEPT")
    TITLE_IMG = build_title("K-MEANS CLUSTERING", "ML AS A GAME  ·  EP 6")
    CONFETTI = build_confetti()
    PHASES = build_phases()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = word(DONE, "balanced") + 0.8
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "NO ANSWERS NEEDED", font("Black", 84), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "K-Means Clustering, explained", font("Black", 56), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s | {N} students | flags start at {np.round(INIT, 1).tolist()} | {NR} rounds")
    print("switched per round: " + ", ".join(str(rd["switched"]) for rd in ROUNDS) +
          " | spread (join/move): " + ", ".join(f"{s:.0f}" for s in SPREADS))
    print("final flags: " + ", ".join(f"{NAMES[k]} {np.round(FINAL[k], 1).tolist()}" for k in range(K)))
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
    out = os.path.join(HERE, "kmeans_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
