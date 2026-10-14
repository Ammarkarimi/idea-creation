"""ML as a Game, Level 13: Reinforcement Learning (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> rl_short.mp4 + thumbnail.png
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
    CYAN, FPS, GREEN, H, MUTED, ORANGE, PINK, PURPLE, SR, W, WHITE, YELLOW, Timeline, add_glow, build_confetti,
    build_plain, build_title, caption_chunks, clamp, draw_caption, draw_hud_pills, draw_level_cleared, ease_back,
    ease_in_out, ease_out, encode, font, lerp, lerp_color, mix, music_bed, num_to_words, paste_scaled, pop_in,
    sfx_blip, sfx_boing, sfx_buzz, sfx_levelup, sfx_whoosh, tag, text_center, voiceover, write_wav,
)

# ---------------------------------------------------------------- the game + real tabular Q-learning

N = 4
START, GOAL = (3, 0), (0, 3)
LAVA = {(1, 1), (2, 2)}
GOAL_R, LAVA_R, STEP_R = 10, -10, -1
ACTS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left
ALPHA, GAMMA, EPS = 0.5, 0.9, 0.2
GAMES, MAX_MOVES, SEED = 100, 60, 6


def env_step(s, a):
    r, c = s[0] + ACTS[a][0], s[1] + ACTS[a][1]
    if not (0 <= r < N and 0 <= c < N):
        r, c = s  # bumped into the wall
    if (r, c) == GOAL:
        return (r, c), GOAL_R, True
    if (r, c) in LAVA:
        return (r, c), LAVA_R, True
    return (r, c), STEP_R, False


def train():
    rng = np.random.default_rng(SEED)
    q = np.zeros((N, N, 4))
    games, snaps = [], []
    for _ in range(GAMES):
        s, path, total = START, [START], 0
        for _ in range(MAX_MOVES):
            if rng.random() < EPS:
                a = int(rng.integers(4))  # explore
            else:
                best = np.flatnonzero(q[s] == q[s].max())
                a = int(rng.choice(best))  # best known move (ties: random)
            s2, r, done = env_step(s, a)
            target = r + (0 if done else GAMMA * q[s2].max())
            q[s][a] += ALPHA * (target - q[s][a])
            total += r
            s = s2
            path.append(s)
            if done:
                break
        games.append(dict(path=path, total=total, end=s))
        snaps.append(q.copy())
    return games, snaps


GAMES_LOG, QSNAP = train()
Q = QSNAP[-1]


def greedy_path(q):
    s, path, total = START, [START], 0
    for _ in range(20):
        s, r, done = env_step(s, int(np.argmax(q[s])))
        path.append(s)
        total += r
        if done:
            break
    return path, total


BEST_PATH, BEST_TOTAL = greedy_path(Q)
BEST_MOVES = len(BEST_PATH) - 1
SHORTEST = abs(START[0] - GOAL[0]) + abs(START[1] - GOAL[1])
G1 = GAMES_LOG[0]
assert G1["end"] in LAVA  # game 1: random wandering into lava
assert BEST_PATH[-1] == GOAL and BEST_MOVES == SHORTEST and not set(BEST_PATH) & LAVA
# The value really spreads backward: squares further from the treasure turn positive later.
FIRST_POS = [next(e for e in range(GAMES) if QSNAP[e][c].max() > 0) for c in BEST_PATH[:-1]]
assert all(a > b for a, b in zip(FIRST_POS, FIRST_POS[1:]))
WINS_FIRST = sum(g["end"] == GOAL for g in GAMES_LOG[:10])
WINS_LAST = sum(g["end"] == GOAL for g in GAMES_LOG[-10:])
assert WINS_LAST > WINS_FIRST + 5
# The lava move from game 1 gets a bad score right away: 0 + 0.5 * (-10 - 0).
LAVA_FROM = G1["path"][-2]
LAVA_ACT = ACTS.index((G1["end"][0] - LAVA_FROM[0], G1["end"][1] - LAVA_FROM[1]))
LAVA_SCORE = QSNAP[0][LAVA_FROM][LAVA_ACT]
assert LAVA_SCORE == ALPHA * LAVA_R

SCRIPT = [
    "Nobody tells this robot the answer. It has to learn by playing.",
    f"Treasure: plus {num_to_words(GOAL_R)}. Lava: minus {num_to_words(-LAVA_R)}. Every move costs one point.",
    "At first, it just moves randomly, and falls in the lava.",
    "But after every move, it updates a score: how good was that move, from that square?",
    "Squares next to the treasure score high first. Then the value spreads backward, square by square.",
    "That's reinforcement learning: try, get a reward, remember.",
    "Sometimes it still tries a random move, in case there's a better path.",
    f"After a hundred games, it walks straight to the treasure in {num_to_words(BEST_MOVES)} moves.",
    "Add neural networks, and this idea learned Atari games, beat a Go world champion, and helps fine-tune "
    "chatbots.",
    "Level thirteen cleared. Next: diffusion, how AI draws pictures. Follow for level fourteen.",
]
HOOK, RULES, RANDOM, UPDATE, SPREAD, LOOP, EXPLORE, RESULT, REAL, CTA = range(10)

# ---------------------------------------------------------------- layout

CELL = 150
GX0, GY0 = (W - N * CELL) // 2, 555
PANEL_Y = 1235
CAPTION_Y = 1495
TOP_TAG_Y = 425
BASE = (36, 40, 74)

T: Timeline = None
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0


def word(i, prefix, nth=0):
    return T.word(i, prefix, nth)


def center(cell):
    r, c = cell
    return GX0 + CELL * c + CELL / 2, GY0 + CELL * r + CELL / 2


# ---------------------------------------------------------------- timing / state

def random_times():
    """Game 1 plays out between the start of its line and the word 'lava'."""
    t0, t1 = T.start(RANDOM) + 0.2, word(RANDOM, "lava")
    n = len(G1["path"]) - 1
    return [t0 + (t1 - t0) * k / n for k in range(n + 1)]


def greedy_times():
    t0 = word(RESULT, "walks")
    return [t0 + 0.25 * k for k in range(BEST_MOVES + 1)]


def game_index(t):
    """Which training game the heatmap shows (fractional, 0-based)."""
    a, b, c = T.start(SPREAD), T.start(LOOP), T.start(RESULT) - 0.4
    if t < a:
        return 0.0
    if t < b:
        return lerp(0, 29, ease_in_out((t - a) / (b - a)))
    return lerp(29, GAMES - 1, ease_in_out((t - b) / (c - b)))


def time_of_game(e):
    """First moment the heatmap shows training game e."""
    for k in np.linspace(0, 1, 600):
        t = lerp(T.start(SPREAD), T.start(RESULT) - 0.4, k)
        if game_index(t) >= e:
            return t
    return T.start(RESULT)


def values_at(t):
    g = game_index(t)
    i = int(g)
    j = min(GAMES - 1, i + 1)
    f = g - i
    return QSNAP[i].max(2) * (1 - f) + QSNAP[j].max(2) * f, QSNAP[int(round(g))]


def path_pos(times, path, t):
    """Robot position (interpolated) and how many moves are done."""
    if t <= times[0]:
        return center(path[0]), 0
    for k in range(len(path) - 1):
        if t < times[k + 1]:
            p = ease_in_out((t - times[k]) / (times[k + 1] - times[k]) / 0.8)
            (x0, y0), (x1, y1) = center(path[k]), center(path[k + 1])
            return (lerp(x0, x1, p), lerp(y0, y1, p)), k + (1 if p >= 1 else 0)
    return center(path[-1]), len(path) - 1


def partial_total(path, moves):
    total = 0
    for k in range(moves):
        nxt = path[k + 1]
        total += GOAL_R if nxt == GOAL else LAVA_R if nxt in LAVA else STEP_R
    return total


def fmt(v):
    return ("+" if v > 0 else "−" if v < 0 else "") + str(abs(v))


def hud(t):
    if t < T.start(RANDOM):
        return "0", MUTED
    if t < T.start(SPREAD):
        _, moves = path_pos(random_times(), G1["path"], t)
        v = partial_total(G1["path"], moves)
        return fmt(v), PINK if v < 0 else MUTED
    if t < greedy_times()[0]:
        v = GAMES_LOG[int(round(game_index(t)))]["total"]
        return fmt(v), GREEN if v > 0 else PINK
    _, moves = path_pos(greedy_times(), BEST_PATH, t)
    v = partial_total(BEST_PATH, moves)
    return fmt(v), GREEN if v > 0 else PINK


# ---------------------------------------------------------------- drawing

def heat_color(v):
    if v >= 0:
        return lerp_color(BASE, (34, 150, 90), clamp(v / 10))
    return lerp_color(BASE, (120, 36, 70), clamp(-v / 8))


def draw_treasure(d, cx, cy, s=1.0):
    r = 44 * s
    d.polygon([(cx, cy - r), (cx + r * 0.85, cy - r * 0.2), (cx, cy + r), (cx - r * 0.85, cy - r * 0.2)],
              fill=YELLOW, outline=(255, 250, 200), width=3)
    d.line([(cx - r * 0.85, cy - r * 0.2), (cx + r * 0.85, cy - r * 0.2)], fill=(200, 160, 30), width=3)
    d.line([(cx - r * 0.35, cy - r * 0.2), (cx, cy + r)], fill=(200, 160, 30), width=2)
    d.line([(cx + r * 0.35, cy - r * 0.2), (cx, cy + r)], fill=(200, 160, 30), width=2)


def draw_lava(d, x0, y0, t):
    d.rounded_rectangle((x0 + 6, y0 + 6, x0 + CELL - 6, y0 + CELL - 6), radius=18, fill=(190, 52, 30))
    for k in range(3):
        y = y0 + 45 + 32 * k
        pts = [(x0 + 18 + i * 6, y + 7 * math.sin(i * 0.7 + t * 4 + k)) for i in range(20)]
        d.line(pts, fill=(255, 170, 60), width=5)


def draw_robot(frame, pos, scale=1.0, alpha=1.0, sink=0.0):
    img = Image.new("RGBA", (130, 140), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.line([(65, 30), (65, 10)], fill=WHITE, width=5)
    d.ellipse((57, 2, 73, 18), fill=PINK)
    d.rounded_rectangle((20, 28, 110, 112), radius=22, fill=CYAN, outline=WHITE, width=4)
    for ex in (47, 83):
        d.ellipse((ex - 12, 54, ex + 12, 78), fill=WHITE)
        d.ellipse((ex - 5, 60, ex + 5, 72), fill=(20, 20, 30))
    d.rounded_rectangle((48, 88, 82, 96), radius=4, fill=(20, 60, 80))
    if sink > 0:
        img = img.resize((130, max(1, int(140 * (1 - sink)))))
    x, y = pos
    paste_scaled(frame, img, (x, y + 70 * sink * 0.5), scale=scale * 0.95, alpha=alpha)


def draw_arrow(d, cx, cy, a, col, size=26):
    dy, dx = ACTS[a]
    tip = (cx + dx * size, cy + dy * size)
    px, py = -dy, dx
    base = (cx - dx * size * 0.35, cy - dy * size * 0.35)
    d.polygon([tip, (base[0] + px * size * 0.6, base[1] + py * size * 0.6),
               (base[0] - px * size * 0.6, base[1] - py * size * 0.6)], fill=col)


def draw_grid(frame, t):
    t0 = T.start(HOOK)
    g = ease_out((t - t0) / 0.5)
    if g <= 0 or t >= T.start(CTA):
        return
    d = ImageDraw.Draw(frame)
    show_heat = t >= T.start(SPREAD)
    vals, qsnap = values_at(t) if show_heat else (None, None)
    show_arrows = t >= T.start(EXPLORE)
    arrows_a = clamp((t - T.start(EXPLORE)) / 0.5)
    t_rules = T.start(RULES)
    for r in range(N):
        for c in range(N):
            x0, y0 = GX0 + CELL * c, GY0 + CELL * r
            cell_g = ease_back((t - t0 - 0.03 * (r * N + c)) / 0.35)
            if cell_g <= 0:
                continue
            inset = 6 + (1 - min(1, cell_g)) * 60
            box = (x0 + inset, y0 + inset, x0 + CELL - inset, y0 + CELL - inset)
            if (r, c) in LAVA and t >= word(RULES, "lava"):
                draw_lava(d, x0, y0, t)
                continue
            fill = BASE
            if show_heat and (r, c) != GOAL:
                fill = heat_color(vals[r, c])
            d.rounded_rectangle(box, radius=18, fill=fill, outline=(60, 66, 110), width=2)
            if (r, c) == GOAL and t >= word(RULES, "treasure"):
                s = ease_back((t - word(RULES, "treasure")) / 0.4) * (1 + 0.05 * math.sin(t * 5))
                draw_treasure(d, x0 + CELL / 2, y0 + CELL / 2, s)
                continue
            if show_heat:
                v = vals[r, c]
                if abs(v) >= 0.05:
                    text_center(d, (x0 + CELL / 2, y0 + CELL - 30), f"{v:.1f}".replace("-", "−"),
                                font("Black", 30), WHITE if v > 0 else (230, 170, 190))
            if show_arrows and arrows_a > 0 and abs(qsnap[r, c]).max() > 0:
                a = int(np.argmax(qsnap[r, c]))
                draw_arrow(d, x0 + CELL / 2, y0 + CELL / 2 - 12, a, lerp_color(fill, WHITE, arrows_a))
    # Labels for the rules.
    if t_rules <= t < T.start(UPDATE):
        if t >= word(RULES, "treasure"):
            pop_in(frame, tag(f"+{GOAL_R}", (20, 20, 30), YELLOW + (255,), 34, pad=(14, 6)),
                   (center(GOAL)[0], GY0 - 30), t, word(RULES, "treasure"))
        if t >= word(RULES, "lava"):
            for cell in LAVA:
                x, y = center(cell)
                pop_in(frame, tag(f"−{-LAVA_R}", WHITE, (150, 30, 30, 255), 30, pad=(12, 5)), (x, y - 52), t,
                       word(RULES, "lava"))
        if t >= word(RULES, "every"):
            pop_in(frame, tag("every move: −1", WHITE, PURPLE + (255,), 40), (540, PANEL_Y), t, word(RULES, "every"))


def draw_trails(frame, t):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    drew = False
    # Game 1's wandering trail.
    tr = random_times()
    if tr[0] <= t < T.start(SPREAD) + 0.3:
        a = 1 - clamp((t - T.start(SPREAD)) / 0.3)
        pos, moves = path_pos(tr, G1["path"], t)
        pts = [center(p) for p in G1["path"][:moves + 1]] + [pos]
        jit = [(x + 10 * math.sin(k * 2.1), y + 10 * math.cos(k * 1.7)) for k, (x, y) in enumerate(pts)]
        d.line(jit, fill=PINK + (int(170 * a),), width=7, joint="curve")
        drew = True
    tg = greedy_times()
    if tg[0] <= t < T.start(CTA):
        pos, moves = path_pos(tg, BEST_PATH, t)
        pts = [center(p) for p in BEST_PATH[:moves + 1]] + [pos]
        d.line(pts, fill=GREEN + (230,), width=12, joint="curve")
        drew = True
    if drew:
        frame.alpha_composite(add_glow(lay, radius=8, strength=1.2))


def draw_robot_state(frame, t):
    if t < T.start(HOOK) + 0.3 or t >= T.start(CTA):
        return
    tr = random_times()
    t_lava = tr[-1]
    if t < T.start(SPREAD):
        pos, _ = path_pos(tr, G1["path"], t)
        sink = clamp((t - t_lava) / 0.5)
        draw_robot(frame, pos, scale=ease_back((t - T.start(HOOK) - 0.3) / 0.4), sink=sink, alpha=1 - 0.6 * sink)
        if t < T.start(RULES):
            pop_in(frame, tag("?", (20, 20, 30), YELLOW + (255,), 52, pad=(18, 4)), (pos[0] + 70, pos[1] - 80), t,
                   word(HOOK, "answer"))
        if t_lava <= t < t_lava + 0.6:
            d = ImageDraw.Draw(frame)
            x, y = center(G1["end"])
            p = (t - t_lava) / 0.6
            for k in range(8):
                ang = k * math.pi / 4
                rr = 20 + 70 * p
                d.ellipse((x + rr * math.cos(ang) - 9, y + rr * math.sin(ang) - 9 - 30 * p,
                           x + rr * math.cos(ang) + 9, y + rr * math.sin(ang) + 9 - 30 * p), fill=ORANGE)
        return
    tg = greedy_times()
    if t < tg[0]:
        # Training montage: the robot waits on the start square.
        draw_robot(frame, center(START), scale=0.8, alpha=0.75)
        return
    pos, moves = path_pos(tg, BEST_PATH, t)
    draw_robot(frame, pos)
    if moves == BEST_MOVES:
        pop_in(frame, tag(f"{BEST_MOVES} MOVES!", (12, 28, 18), GREEN + (255,), 40), (pos[0] - 40, pos[1] + 95), t,
               tg[-1])


def draw_update(frame, t):
    """The bad score for game 1's last move, then the update rule."""
    t0 = T.start(UPDATE)
    if not (t0 <= t < T.start(SPREAD) + 0.3):
        return
    a = 1 - clamp((t - T.start(SPREAD)) / 0.3)
    d = ImageDraw.Draw(frame)
    t_s = word(UPDATE, "score")
    if t >= t_s:
        (x0, y0), (x1, y1) = center(LAVA_FROM), center(G1["end"])
        draw_arrow(d, lerp(x0, x1, 0.5), lerp(y0, y1, 0.5), LAVA_ACT, (255, 90, 90), size=34)
        pop_in(frame, tag(f"that move: {LAVA_SCORE:.0f}".replace("-", "−"), WHITE, (150, 30, 30, 255), 32,
                          pad=(14, 6)), (x0, y0 - 52), t, t_s)
    t_h = word(UPDATE, "how")
    if t >= t_h:
        g = ease_back((t - t_h) / 0.35)
        card = Image.new("RGBA", (900, 150), (0, 0, 0, 0))
        cd = ImageDraw.Draw(card)
        cd.rounded_rectangle((4, 4, 896, 146), radius=30, fill=(28, 16, 56, 245), outline=CYAN + (255,), width=4)
        text_center(cd, (450, 44), "nudge the move's score toward:", font("Bold", 30), MUTED)
        text_center(cd, (450, 102), f"reward + {GAMMA} × next square's best", font("Black", 42), WHITE)
        paste_scaled(frame, card, (540, PANEL_Y + 40), scale=g, alpha=clamp(g) * a)


def draw_panel(frame, t):
    """Game counter, the learning loop, exploration, and the before/after stats."""
    t_sp = T.start(SPREAD)
    if t_sp <= t < greedy_times()[0]:
        g = int(round(game_index(t))) + 1
        pop_in(frame, tag(f"TRAINING GAME {g} / {GAMES}", WHITE, (36, 40, 80, 255), 36, outline=CYAN + (255,)),
               (540, PANEL_Y - 10), t, t_sp)
    if T.start(LOOP) <= t < T.start(EXPLORE):
        items = [("TRY", "try", CYAN), ("REWARD", "reward", YELLOW), ("REMEMBER", "remember", GREEN)]
        xs = [250, 540, 830]
        d = ImageDraw.Draw(frame)
        for k, (lab, w_, col) in enumerate(items):
            tw = word(LOOP, w_)
            if t >= tw:
                if k > 0:
                    text_center(d, ((xs[k - 1] + xs[k]) / 2, PANEL_Y + 85), "→", font("Black", 44), MUTED)
                pop_in(frame, tag(lab, (20, 20, 30), col + (255,), 38, pad=(20, 10)), (xs[k], PANEL_Y + 85), t, tw)
    if T.start(EXPLORE) <= t < T.start(RESULT):
        tr = word(EXPLORE, "random")
        if t >= tr:
            pop_in(frame, tag(f"{round(100 * EPS)}% of moves: random (explore)", (20, 20, 30), ORANGE + (255,), 34),
                   (540, PANEL_Y + 70), t, tr)
            pop_in(frame, tag(f"{round(100 * (1 - EPS))}%: best arrow so far", WHITE, (36, 40, 80, 255), 30),
                   (540, PANEL_Y + 140), t, tr + 0.3)
    tg = greedy_times()
    if tg[0] <= t < T.start(CTA):
        pop_in(frame, tag(f"treasure found: first 10 games {WINS_FIRST}/10 → last 10 games {WINS_LAST}/10", WHITE,
                          (36, 40, 80, 255), 30, outline=GREEN + (255,)), (540, PANEL_Y + 40), t, tg[0])


def draw_real(frame, t):
    t0 = T.start(REAL)
    if not (t0 <= t < T.start(CTA)):
        return
    frame.alpha_composite(Image.new("RGBA", (W, H), (5, 6, 16, int(205 * clamp((t - t0) / 0.3)))))
    pop_in(frame, tag("+ NEURAL NETWORKS", WHITE, PURPLE + (255,), 46), (540, 640), t, word(REAL, "neural"))
    items = [("ATARI GAMES", "atari", CYAN, 790), ("GO WORLD CHAMPION", "go", YELLOW, 930),
             ("FINE-TUNING CHATBOTS", "chatbots", GREEN, 1070)]
    for lab, w_, col, y in items:
        if t >= word(REAL, w_):
            pop_in(frame, tag(lab, (20, 20, 30), col + (255,), 46), (540, y), t, word(REAL, w_))


def draw_overlays(frame, t):
    draw_hud_pills(frame, 13, *hud(t), label="REWARD")
    frame.alpha_composite(TITLE_IMG, (0, 190))
    banners = [
        (T.start(HOOK), T.start(RULES), "NO ANSWERS GIVEN: LEARN BY PLAYING", PURPLE, WHITE),
        (T.start(RULES), T.start(RANDOM), "THE RULES: REWARDS & PENALTIES", YELLOW, (20, 20, 30)),
        (T.start(RANDOM), T.start(UPDATE), "GAME 1: RANDOM MOVES", PINK, WHITE),
        (T.start(UPDATE), T.start(SPREAD), "AFTER EVERY MOVE: UPDATE A SCORE", CYAN, (12, 14, 30)),
        (T.start(SPREAD), T.start(LOOP), "VALUE SPREADS BACKWARD FROM THE TREASURE", GREEN, (12, 28, 18)),
        (T.start(LOOP), T.start(EXPLORE), "REINFORCEMENT LEARNING", PURPLE, WHITE),
        (T.start(EXPLORE), T.start(RESULT), "EXPLORE SOMETIMES", ORANGE, (20, 20, 30)),
        (T.start(RESULT), T.start(REAL), f"AFTER {GAMES} GAMES: STRAIGHT TO THE TREASURE", GREEN, (12, 28, 18)),
        (T.start(REAL), T.start(CTA), "SAME IDEA, BIGGER GAMES", PURPLE, WHITE),
    ]
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 32), (W / 2, TOP_TAG_Y), t, a)
    draw_level_cleared(frame, t, T.start(CTA), 13, "DIFFUSION", word(CTA, "next"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    draw_grid(frame, t)
    draw_trails(frame, t)
    draw_update(frame, t)
    draw_robot_state(frame, t)
    draw_panel(frame, t)
    draw_real(frame, t)
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
    mix(out, sfx_boing(0.5, 0.16), T.start(HOOK) + 0.3)
    mix(out, sfx_blip(1568, 0.3, 0.16), word(RULES, "treasure"))
    mix(out, sfx_buzz(0.2, 110, 0.12), word(RULES, "lava"))
    tr = random_times()
    for tk in tr[1:-1]:
        mix(out, sfx_blip(520, 0.05, 0.08), tk)
    mix(out, sfx_whoosh(0.5, 0.2, seed=4), tr[-1] - 0.1)
    mix(out, sfx_buzz(0.3, 90, 0.16), tr[-1])
    mix(out, sfx_blip(880, 0.12, 0.12), word(UPDATE, "score"))
    mix(out, sfx_whoosh(0.8, 0.15, seed=6), T.start(SPREAD))
    for e in sorted(set(FIRST_POS)):
        mix(out, sfx_blip(700 + 15 * e, 0.12, 0.1), time_of_game(e))  # another best-path square turns positive
    for k, w_ in enumerate(("try", "reward", "remember")):
        mix(out, sfx_blip(660 + 220 * k, 0.1, 0.12), word(LOOP, w_))
    mix(out, sfx_boing(0.45, 0.14), word(EXPLORE, "random"))
    tg = greedy_times()
    for k, tk in enumerate(tg[1:]):
        mix(out, sfx_blip(600 + 90 * k, 0.07, 0.11), tk)
    mix(out, sfx_blip(1568, 0.35, 0.18), tg[-1] + 0.05)
    for w_ in ("atari", "go", "chatbots"):
        mix(out, sfx_blip(1175, 0.12, 0.12), word(REAL, w_))
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12, max_px=800)
    STATIC = build_plain()
    TITLE_IMG = build_title("LEARN BY PLAYING", "REINFORCEMENT LEARNING  ·  ML AS A GAME  ·  EP 13")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = min(greedy_times()[-1] + 0.6, T.start(REAL) - 0.1)
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "AI LEARNS BY PLAYING", font("Black", 82), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "Reinforcement learning, explained", font("Black", 52), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s")
    print(f"  game 1: {len(G1['path']) - 1} moves, ends in lava at {G1['end']}, total {G1['total']}")
    print(f"  lava move score after game 1: {LAVA_SCORE}")
    print(f"  best path after {GAMES} games: {BEST_MOVES} moves {BEST_PATH}, total {BEST_TOTAL}")
    print(f"  square turns positive at game (start→treasure): {[e + 1 for e in FIRST_POS]}")
    print(f"  treasure found: first 10 games {WINS_FIRST}/10, last 10 games {WINS_LAST}/10")
    print("  final best scores:\n" + str(np.round(Q.max(2), 1)))
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
    out = os.path.join(HERE, "rl_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
