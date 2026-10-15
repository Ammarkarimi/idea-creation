"""ML as a Game, Level 15: Word Embeddings (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> embeddings_short.mp4 + thumbnail.png
    python3 render.py --preview 3,12  # just dump PNG frames at those seconds (fast check)
    python3 render.py --thumbnail     # only regenerate thumbnail.png
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import (  # noqa: E402
    CYAN, FPS, GREEN, H, MUTED, ORANGE, PINK, PURPLE, SR, W, WHITE, YELLOW, Timeline, add_glow, build_confetti,
    build_plain, build_title, caption_chunks, clamp, draw_caption, draw_hud_pills, draw_level_cleared, ease_back,
    ease_in_out, encode, font, lerp, mix, music_bed, paste_scaled, pop_in, sfx_blip, sfx_boing, sfx_levelup,
    sfx_whoosh, tag, text_center, voiceover, write_wav,
)

# ---------------------------------------------------------------- the word map (hand-placed 2D embeddings)

# Each word is a list of 2 numbers: its spot on the map. Real embeddings are learned and have hundreds of numbers.
VEC = {
    "king": (-3.6, 6.2), "queen": (2.5, 5.7), "prince": (-4.6, 4.2), "princess": (1.4, 3.8),
    "man": (-3.8, 1.6), "woman": (2.2, 1.3), "boy": (-4.8, -0.3), "girl": (1.0, -0.5),
    "cat": (-6.2, -4.8), "dog": (-3.4, -5.4), "horse": (-6.0, -7.6), "cow": (-3.2, -8.2),
    "bread": (2.6, -4.9), "cheese": (5.6, -5.3), "apple": (2.9, -7.7), "pizza": (5.8, -8.3),
}
VEC = {w: np.array(v) for w, v in VEC.items()}
GROUPS = {
    "royals": (["king", "queen", "prince", "princess"], YELLOW),
    "people": (["man", "woman", "boy", "girl"], CYAN),
    "animals": (["cat", "dog", "horse", "cow"], GREEN),
    "food": (["bread", "cheese", "apple", "pizza"], ORANGE),
}
GROUP_OF = {w: g for g, (ws, _) in GROUPS.items() for w in ws}


def nearest(p, exclude=()):
    d = {w: float(np.linalg.norm(v - p)) for w, v in VEC.items() if w not in exclude}
    w = min(d, key=d.get)
    return w, d[w]


def dist(a, b):
    return float(np.linalg.norm(VEC[a] - VEC[b]))


ARROW = VEC["woman"] - VEC["man"]  # the "man -> woman" direction
KING_PT = VEC["king"] + ARROW
BOY_PT = VEC["boy"] + ARROW
KQ, KQ_D = nearest(KING_PT, exclude={"king", "man", "woman"})
BG, BG_D = nearest(BOY_PT, exclude={"boy", "man", "woman"})
assert KQ == "queen" and BG == "girl"
assert nearest(KING_PT)[0] == "queen" and nearest(BOY_PT)[0] == "girl"  # even without excluding anything
# Similar words really are close: every word's nearest neighbour is in its own group.
assert all(GROUP_OF[nearest(v, exclude={w})[0]] == GROUP_OF[w] for w, v in VEC.items())
assert dist("cat", "dog") < dist("cat", "pizza") / 3

SCRIPT = [
    "Computers can't read words, only numbers. So how does AI know king and queen are related?",
    "Give every word a spot on a map, as a list of numbers. Similar words land close together.",
    "The spots are learned from reading: words used in similar sentences get pulled together.",
    "Now directions mean something. Man to woman is one arrow.",
    "Start at king. Subtract man. Add woman.",
    "And you land right next to queen!",
    "The same arrow turns boy into girl.",
    "These number lists are called embeddings. Real ones have hundreds of numbers, "
    "and every chatbot starts with them.",
    "Level fifteen cleared. Next: Gans, a forger versus a detective. Follow for level sixteen.",
]
HOOK, MAP, LEARN, DIRECTION, MATH, RESULT, BOY, REAL, CTA = range(9)

# ---------------------------------------------------------------- layout

SX, SY = 55, 44  # px per map unit (a stretch is linear, so arrows and word math look the same on screen)
ORIGIN = (530, 560 + 6.2 * SY)
CARD_Y = 1295
CAPTION_Y = 1495
TOP_TAG_Y = 425
CAT_START = np.array([5.4, -1.9])

T: Timeline = None
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0


def word(i, prefix, nth=0):
    return T.word(i, prefix, nth)


def px(p):
    return ORIGIN[0] + SX * p[0], ORIGIN[1] - SY * p[1]


# ---------------------------------------------------------------- state

def cat_pos(t):
    """In the LEARN scene the cat starts at a random spot and gets pulled next to the dog."""
    t0, t1 = T.start(LEARN), T.start(DIRECTION)
    if not (t0 <= t < t1):
        return VEC["cat"]
    tj = word(LEARN, "pulled")
    if t < tj:
        g = ease_in_out((t - t0) / 0.5)
        return VEC["cat"] * (1 - g) + CAT_START * g
    g = ease_in_out((t - tj) / 1.0)
    return CAT_START * (1 - g) + VEC["cat"] * g


def travel(t, start, t0, dur=1.0):
    """A point moving from `start` along the man->woman arrow."""
    g = ease_in_out((t - t0) / dur) if t >= t0 else 0.0
    return start + ARROW * g, g


def probe(t):
    """The moving point (and its trail start) for the arithmetic scenes, or None."""
    if T.start(MATH) <= t < T.start(BOY):
        t_add = word(MATH, "add")
        return VEC["king"], travel(t, VEC["king"], t_add, 1.2)
    if T.start(BOY) <= t < T.start(REAL):
        return VEC["boy"], travel(t, VEC["boy"], word(BOY, "arrow") + 0.1, 1.0)
    return None


def hud(t):
    pr = probe(t)
    if pr is None:
        if t < T.start(MATH):
            return "—", MUTED
        return "queen" if t < T.start(BOY) else "girl", GREEN
    start, (p, g) = pr
    w, _ = nearest(p, exclude=set())
    return w, GREEN if g >= 1 else YELLOW


# ---------------------------------------------------------------- drawing

def draw_halos(frame, t):
    t0 = word(MAP, "similar")
    if t < t0 or t >= T.start(CTA):
        return
    a = clamp((t - t0) / 0.5) * (0.55 if t >= T.start(DIRECTION) else 1.0)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    for name, (ws, col) in GROUPS.items():
        pts = np.array([px(VEC[w]) for w in ws])
        x0, y0 = pts.min(0) - (100, 45)
        x1, y1 = pts.max(0) + (100, 45)
        d.rounded_rectangle((x0, y0, x1, y1), radius=50, fill=col + (int(46 * a),), outline=col + (int(120 * a),),
                            width=3)
    frame.alpha_composite(lay)


def draw_words(frame, t):
    t0 = T.start(MAP)
    if t >= T.start(CTA):
        return
    dim = t >= T.start(DIRECTION)
    for k, (w, v) in enumerate(VEC.items()):
        if t < t0:
            continue
        g = ease_back((t - t0 - 0.06 * k) / 0.35)
        if g <= 0:
            continue
        col = GROUPS[GROUP_OF[w]][1]
        key = w in ("king", "queen", "man", "woman", "boy", "girl")
        p = cat_pos(t) if w == "cat" else v
        if dim and not key:
            img = tag(w, WHITE, (36, 40, 74, 200), 34, pad=(14, 6))
        else:
            img = tag(w, (20, 20, 30), col + (255,), 40, pad=(16, 8))
        paste_scaled(frame, img, px(p), scale=min(1.0, g))
    # One example of "a list of numbers".
    t_n = word(MAP, "numbers")
    if t_n <= t < T.start(LEARN):
        x, y = px(VEC["king"])
        kv = VEC["king"]
        pop_in(frame, tag(f"king = ({kv[0]:.1f}, {kv[1]:.1f})".replace("-", "−"), WHITE, PURPLE + (255,), 32,
                          pad=(14, 6)), (x + 150, y - 58), t, t_n)


def draw_hook(frame, t):
    if t >= T.start(MAP):
        return
    t_k = word(HOOK, "king")
    d = ImageDraw.Draw(frame)
    pop_in(frame, tag("“king”", WHITE, (36, 40, 80, 255), 64, outline=YELLOW + (255,)), (330, 760), t, 0.3)
    text_center(d, (540, 760), "→", font("Black", 64), MUTED)
    pop_in(frame, tag("? ? ?", YELLOW, (36, 40, 80, 255), 64), (770, 760), t, word(HOOK, "only"))
    if t >= t_k:
        pop_in(frame, tag("king", (20, 20, 30), YELLOW + (255,), 52), (380, 1000), t, t_k)
        pop_in(frame, tag("queen", (20, 20, 30), YELLOW + (255,), 52), (700, 1000), t, word(HOOK, "queen"))
        if t >= word(HOOK, "related"):
            text_center(d, (540, 1000), "↔", font("Black", 56), WHITE)
            pop_in(frame, tag("related?", WHITE, PURPLE + (255,), 40), (540, 1110), t, word(HOOK, "related"))


def draw_learn(frame, t):
    t0 = T.start(LEARN)
    if not (t0 <= t < T.start(DIRECTION)):
        return
    t_s = word(LEARN, "similar")
    if t >= t_s:
        pop_in(frame, tag("the cat drinks milk", WHITE, (36, 40, 80, 255), 34, outline=GREEN + (255,)),
               (540, CARD_Y - 5), t, t_s)
        pop_in(frame, tag("the dog drinks milk", WHITE, (36, 40, 80, 255), 34, outline=GREEN + (255,)),
               (540, CARD_Y + 62), t, t_s + 0.3)
    tj = word(LEARN, "pulled")
    if tj <= t:
        d = ImageDraw.Draw(frame)
        a, b = px(cat_pos(t)), px(VEC["dog"])
        d.line([a, b], fill=GREEN, width=4)


def arrow(d, p0, p1, col, width=10, head=26):
    (x0, y0), (x1, y1) = p0, p1
    L = math.hypot(x1 - x0, y1 - y0)
    if L < 2:
        return
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    bx, by = x1 - ux * head, y1 - uy * head
    d.line([(x0, y0), (bx, by)], fill=col, width=width)
    d.polygon([(x1, y1), (bx - uy * head * 0.6, by + ux * head * 0.6), (bx + uy * head * 0.6, by - ux * head * 0.6)],
              fill=col)


def draw_arrows(frame, t):
    t0 = word(DIRECTION, "man")
    if t < t0 or t >= T.start(CTA):
        return
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    # man -> woman, growing in.
    g = ease_in_out((t - t0) / 0.8)
    arrow(d, px(VEC["man"]), px(VEC["man"] + ARROW * g), PINK + (255,))
    pr = probe(t)
    done_king = t >= T.start(BOY)
    if done_king:
        arrow(d, px(VEC["king"]), px(KING_PT), PINK + (150,), width=8)
    if pr is not None:
        start, (p, gg) = pr
        if gg > 0:
            arrow(d, px(start), px(p), PINK + (255,))
    frame.alpha_composite(add_glow(lay, radius=8, strength=1.2))
    if pr is not None:
        start, (p, gg) = pr
        x, y = px(p)
        dd = ImageDraw.Draw(frame)
        r = 16 + 4 * math.sin(t * 8)
        dd.ellipse((x - r, y - r, x + r, y + r), outline=WHITE, width=5)
    td = word(DIRECTION, "arrow")
    if T.start(DIRECTION) <= t < T.start(MATH) and t >= td:
        mx, my = px(VEC["man"] + ARROW / 2)
        pop_in(frame, tag("woman − man", WHITE, PINK + (255,), 30, pad=(12, 5)), (mx, my + 44), t, td)


def draw_math_card(frame, t):
    if T.start(MATH) <= t < T.start(BOY):
        parts = [("king", "king", YELLOW), ("− man", "subtract", CYAN), ("+ woman", "add", PINK)]
        xs = [330, 540, 760]
        for (lab, w_, col), x in zip(parts, xs):
            tw = word(MATH, w_)
            if t >= tw:
                pop_in(frame, tag(lab, (20, 20, 30), col + (255,), 44), (x, CARD_Y), t, tw)
        if t >= T.start(RESULT):
            tr = word(RESULT, "queen")
            if t >= tr:
                pop_in(frame, tag(f"≈ QUEEN  (just {KQ_D:.1f} away)", (12, 28, 18), GREEN + (255,), 44),
                       (540, CARD_Y + 70), t, tr)
    if T.start(BOY) <= t < T.start(REAL):
        tg = word(BOY, "girl")
        pop_in(frame, tag("boy − man + woman", WHITE, (36, 40, 80, 255), 40, outline=PINK + (255,)), (540, CARD_Y),
               t, T.start(BOY))
        if t >= tg:
            pop_in(frame, tag(f"≈ GIRL  ({BG_D:.1f} away)", (12, 28, 18), GREEN + (255,), 40), (540, CARD_Y + 70), t, tg)


def draw_real(frame, t):
    t0 = T.start(REAL)
    if not (t0 <= t < T.start(CTA)):
        return
    frame.alpha_composite(Image.new("RGBA", (W, H), (5, 6, 16, int(205 * clamp((t - t0) / 0.3)))))
    t_e = word(REAL, "embeddings")
    if t >= t_e:
        pop_in(frame, tag("EMBEDDING", (20, 20, 30), YELLOW + (255,), 60), (540, 660), t, t_e)
        pop_in(frame, tag("= a word as a list of numbers", WHITE, (36, 40, 80, 255), 38), (540, 760), t, t_e + 0.2)
    t_h = word(REAL, "hundreds")
    if t >= t_h:
        pop_in(frame, tag("this map: 2 numbers per word", WHITE, (36, 40, 80, 255), 36), (540, 900), t, t_h)
        pop_in(frame, tag("real models: hundreds", (20, 20, 30), CYAN + (255,), 44), (540, 985), t, t_h + 0.3)
    t_c = word(REAL, "chatbot")
    if t >= t_c:
        pop_in(frame, tag("STEP 1 OF EVERY CHATBOT", (12, 28, 18), GREEN + (255,), 44), (540, 1130), t, t_c)


def draw_overlays(frame, t):
    draw_hud_pills(frame, 15, *hud(t), label="NEAREST")
    frame.alpha_composite(TITLE_IMG, (0, 190))
    banners = [
        (T.start(HOOK), T.start(MAP), "COMPUTERS ONLY READ NUMBERS", PURPLE, WHITE),
        (T.start(MAP), T.start(LEARN), "EVERY WORD = A SPOT ON A MAP", CYAN, (12, 14, 30)),
        (T.start(LEARN), T.start(DIRECTION), "LEARNED FROM READING (SIMPLIFIED)", GREEN, (12, 28, 18)),
        (T.start(DIRECTION), T.start(MATH), "DIRECTIONS HAVE MEANING", PINK, WHITE),
        (T.start(MATH), T.start(BOY), "WORD MATH", YELLOW, (20, 20, 30)),
        (T.start(BOY), T.start(REAL), "SAME ARROW, SAME MEANING", PINK, WHITE),
        (T.start(REAL), T.start(CTA), "THESE ARE EMBEDDINGS", PURPLE, WHITE),
    ]
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 32), (W / 2, TOP_TAG_Y), t, a)
    draw_level_cleared(frame, t, T.start(CTA), 15, "GANs", word(CTA, "next"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    draw_hook(frame, t)
    draw_halos(frame, t)
    draw_arrows(frame, t)
    draw_words(frame, t)
    draw_learn(frame, t)
    draw_math_card(frame, t)
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
    mix(out, sfx_boing(0.45, 0.14), word(HOOK, "only"))
    mix(out, sfx_blip(1175, 0.15, 0.12), word(HOOK, "related"))
    for k in range(len(VEC)):
        mix(out, sfx_blip(600 + 30 * k, 0.04, 0.07), T.start(MAP) + 0.06 * k)
    mix(out, sfx_whoosh(0.6, 0.14, seed=4), word(MAP, "similar"))
    mix(out, sfx_whoosh(0.9, 0.16, seed=5), word(LEARN, "pulled"))
    mix(out, sfx_blip(1047, 0.2, 0.12), word(LEARN, "pulled") + 1.0)
    mix(out, sfx_whoosh(0.7, 0.14, seed=6), word(DIRECTION, "man"))
    for w_ in ("king", "subtract", "add"):
        mix(out, sfx_blip(880, 0.1, 0.12), word(MATH, w_))
    mix(out, sfx_whoosh(1.0, 0.16, seed=8), word(MATH, "add"))
    mix(out, sfx_blip(1568, 0.35, 0.18), word(RESULT, "queen"))
    mix(out, sfx_whoosh(0.9, 0.14, seed=9), word(BOY, "arrow") + 0.1)
    mix(out, sfx_blip(1568, 0.3, 0.16), word(BOY, "girl"))
    for w_ in ("embeddings", "hundreds", "chatbot"):
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
    TITLE_IMG = build_title("WORD EMBEDDINGS", "ML AS A GAME  ·  EP 15")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = min(word(RESULT, "queen") + 0.6, T.start(BOY) - 0.05)
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "KING − MAN + WOMAN = ?", font("Black", 74), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "Word embeddings, explained", font("Black", 54), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s")
    print(f"  woman - man = {ARROW}")
    print(f"  king - man + woman = {KING_PT} -> nearest {KQ} ({KQ_D:.2f} away)")
    print(f"  boy - man + woman = {BOY_PT} -> nearest {BG} ({BG_D:.2f} away)")
    print(f"  cat-dog {dist('cat', 'dog'):.2f}, cat-pizza {dist('cat', 'pizza'):.2f}")
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
    out = os.path.join(HERE, "embeddings_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
