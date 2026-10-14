"""ML as a Game, Level 12: Transformers & Attention (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> transformers_short.mp4 + thumbnail.png
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
    CYAN, FPS, GREEN, H, MUTED, ORANGE, PINK, PURPLE, SR, W, WHITE, YELLOW, Timeline, build_confetti, build_plain,
    build_title, caption_chunks, clamp, draw_caption, draw_hud_pills, draw_level_cleared, ease_back, ease_in_out,
    ease_out, encode, font, lerp, lerp_color, mix, music_bed, num_to_words, paste_scaled, pop_in, sfx_blip,
    sfx_boing, sfx_levelup, sfx_whoosh, tag, text_center, voiceover, write_wav,
)

# ---------------------------------------------------------------- attention for "it"

SENT = ["the", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
IT = SENT.index("it")
# Hand-made meaning vectors: [living thing, place, action].  Real models learn hundreds of these numbers.
KEYS = {"animal": [3.0, 0.0, 0.0], "street": [0.0, 3.0, 0.0], "cross": [0.0, 0.5, 1.0], "it": [0.5, 0.5, 0.0]}
DEFAULT_KEY = [0.0, 0.0, 0.2]
# What "it" is looking for, given the rest of the sentence (in a real transformer earlier layers compute this).
QUERY = {"tired": [1.5, 0.2, 0.0], "wide": [0.2, 1.5, 0.0]}


def attention(last):
    words = SENT[:-1] + [last]
    k = np.array([KEYS.get(w, DEFAULT_KEY) for w in words])
    scores = k @ np.array(QUERY[last])
    weights = np.exp(scores) / np.exp(scores).sum()  # softmax
    return words, scores, weights


WORDS_T, SCORES_T, ATT_T = attention("tired")
WORDS_W, SCORES_W, ATT_W = attention("wide")
TOP_T, TOP_W = int(np.argmax(ATT_T)), int(np.argmax(ATT_W))
assert WORDS_T[TOP_T] == "animal" and WORDS_W[TOP_W] == "street"
assert abs(ATT_T.sum() - 1) < 1e-9 and abs(ATT_W.sum() - 1) < 1e-9
P_T, P_W = round(100 * ATT_T[TOP_T]), round(100 * ATT_W[TOP_W])

SCRIPT = [
    "ChatGPT, Claude, Gemini: they're all built on one idea. The transformer.",
    "A recurrent network reads one word at a time. A transformer reads every word at once.",
    "Then each word asks: which other words matter to me? That's attention.",
    "The animal didn't cross the street because it was too tired. What does it mean?",
    "It scores every word by how well their meanings match. Softmax turns the scores into percentages.",
    f"Animal gets {num_to_words(P_T)} percent of the attention. So it means the animal.",
    "Change tired to wide, and attention jumps to street. Same word, new meaning, from context.",
    "Every word can look at every other word, so distance doesn't matter. Nothing fades.",
    "Stack dozens of these layers, train them to predict the next word on huge amounts of text, "
    "and you get a chatbot.",
    "Level twelve cleared! What should we learn next? Tell me in the comments.",
]
HOOK, PARALLEL, ASK, SENTENCE, SCORE, RESULT, SWAP, DISTANCE, STACK, CTA = range(10)

# ---------------------------------------------------------------- layout

ROWS = [SENT[0:4], SENT[4:8], SENT[8:11]]
ROW_Y = [600, 700, 800]
WORD_SIZE = 44
PANEL_Y = 930
CAPTION_Y = 1495
TOP_TAG_Y = 425


def word_positions():
    pos = []
    f = font("Black", WORD_SIZE)
    gap = 26
    for r, row in enumerate(ROWS):
        widths = [f.getlength(w) + 36 for w in row]
        x = (W - (sum(widths) + gap * (len(row) - 1))) / 2
        for wd in widths:
            pos.append((x + wd / 2, ROW_Y[r]))
            x += wd + gap
    return pos


POS = []
T: Timeline = None
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0


def word(i, prefix, nth=0):
    return T.word(i, prefix, nth)


# ---------------------------------------------------------------- state

def swap_progress(t):
    t0 = word(SWAP, "wide")
    return ease_in_out((t - t0) / 0.8) if t >= t0 else 0.0


def weights_now(t):
    g = swap_progress(t)
    return ATT_T * (1 - g) + ATT_W * g


def last_word(t):
    return "wide" if swap_progress(t) > 0.5 else "tired"


def lines_alpha(t):
    if t < word(SCORE, "scores"):
        return 0.0
    return 1 - clamp((t - T.start(DISTANCE)) / 0.4)


def hud(t):
    if t < T.start(PARALLEL) - 0.1:
        return f"{int(abs(math.sin(t * 4)) * 99)}%", PINK
    if t < word(SCORE, "softmax"):
        return "—", MUTED
    w = weights_now(t)
    i = int(np.argmax(w))
    return f"{round(100 * w[i])}%", GREEN


# ---------------------------------------------------------------- drawing

def draw_hook(frame, t):
    if t >= T.start(PARALLEL) + 0.3:
        return
    a = 1 - clamp((t - T.start(PARALLEL)) / 0.3)
    names = [("ChatGPT", word(HOOK, "chatgpt")), ("Claude", word(HOOK, "claude")), ("Gemini", word(HOOK, "gemini"))]
    t_idea = word(HOOK, "transformer")
    g = ease_in_out((t - t_idea + 0.3) / 0.7) if t >= t_idea - 0.3 else 0.0
    for j, (n, ta) in enumerate(names):
        if t < ta:
            continue
        x0, y0 = 220 + 320 * j, 640
        x, y = lerp(x0, 540, g), lerp(y0, 900, g)
        paste_scaled(frame, tag(n, WHITE, (36, 40, 80, 255), 44, pad=(22, 12), outline=CYAN + (255,)), (x, y),
                     scale=ease_back((t - ta) / 0.35) * (1 - 0.5 * g), alpha=a)
    if t >= t_idea:
        pop_in(frame, tag("TRANSFORMER", (20, 20, 30), YELLOW + (255,), 64, pad=(30, 16)), (540, 900), t, t_idea)


def draw_parallel(frame, t):
    """RNN lights cells one by one; a transformer lights them all at once."""
    t0, t1 = T.start(PARALLEL), T.start(SENTENCE)
    if not (t0 <= t < t1 + 0.3):
        return
    a = clamp((t - t0) / 0.3) * (1 - clamp((t - t1) / 0.3))
    d = ImageDraw.Draw(frame)
    labels = ["the", "cat", "sat", "on", "the", "mat"]
    t_rnn, t_tf = t0 + 0.3, word(PARALLEL, "transformer")
    for row, (name, y, col) in enumerate((("RNN", 640, PINK), ("TRANSFORMER", 900, GREEN))):
        text_center(d, (540, y - 90), name + (": one at a time" if row == 0 else ": all at once"), font("Black", 36),
                    lerp_color((10, 10, 20), col, a))
        for k, lab in enumerate(labels):
            cx = 140 + 160 * k
            if row == 0:
                on = t >= t_rnn + 0.55 * k
            else:
                on = t >= t_tf + 0.4
            fill = col if on else (36, 40, 74)
            d.rounded_rectangle((cx - 66, y - 34, cx + 66, y + 34), radius=20, fill=lerp_color((10, 10, 20), fill, a))
            text_center(d, (cx, y), lab, font("Black", 32), lerp_color((10, 10, 20), (20, 20, 30) if on else WHITE, a))
            if row == 0 and k < len(labels) - 1:
                d.line([(cx + 68, y), (cx + 92, y)], fill=lerp_color((10, 10, 20), MUTED, a), width=4)
    if t >= word(ASK, "attention"):
        pop_in(frame, tag("ATTENTION", (20, 20, 30), YELLOW + (255,), 56, pad=(26, 12)), (540, 1110), t,
               word(ASK, "attention"))
    elif t >= word(ASK, "which"):
        pop_in(frame, tag("which words matter to me?", WHITE, PURPLE + (255,), 40), (540, 1110), t, word(ASK, "which"))


def draw_sentence(frame, t):
    t0 = T.start(SENTENCE)
    if t < t0 or t >= T.start(CTA):
        return
    w = weights_now(t)
    la = lines_alpha(t)
    # Attention lines from "it" to every word (drawn first, under the cards).
    if la > 0:
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        sx, sy = POS[IT]
        g = ease_out((t - word(SCORE, "scores")) / 0.8)
        for k, (x, y) in enumerate(POS):
            if k == IT:
                continue
            ex, ey = lerp(sx, x, g), lerp(sy, y, g)
            soft = t >= word(SCORE, "softmax")
            wt = w[k] if soft else 0.3
            d.line([(sx, sy), (ex, ey)], fill=YELLOW + (int(255 * la * (0.25 + 0.75 * wt)),),
                   width=max(3, int(4 + 26 * wt)))
        frame.alpha_composite(lay)
    # Every word looks at every word (distance scene).
    t_d = T.start(DISTANCE)
    if t_d <= t < T.start(STACK) + 0.3:
        a = clamp((t - t_d) / 0.4) * (1 - clamp((t - T.start(STACK)) / 0.3))
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        for i, p in enumerate(POS):
            for q in POS[i + 1:]:
                d.line([p, q], fill=CYAN + (int(70 * a),), width=2)
        d.line([POS[0], POS[-1]], fill=GREEN + (int(255 * a),), width=7)
        frame.alpha_composite(lay)
    # Word cards.
    for k, (x, y) in enumerate(POS):
        g = ease_back((t - t0 - 0.05 * k) / 0.35)
        if g <= 0:
            continue
        wd = SENT[k] if k < len(SENT) - 1 else last_word(t)
        is_it = k == IT
        top = t >= word(SCORE, "softmax") and la > 0 and k == int(np.argmax(w))
        bg = YELLOW if is_it else (GREEN if top else (40, 44, 82))
        fg = (20, 20, 30) if (is_it or top) else WHITE
        flip = 1.0
        if k == len(SENT) - 1:
            ts = word(SWAP, "wide")
            if ts - 0.2 <= t < ts + 0.6:
                flip = abs(math.cos((t - ts + 0.2) / 0.8 * math.pi))
            if t >= ts - 0.2 and t < T.start(DISTANCE):
                bg, fg = ORANGE, (20, 20, 30)
        img = tag(wd, fg, bg + (255,), WORD_SIZE, pad=(18, 10))
        if flip < 1:
            img = img.resize((img.width, max(1, int(img.height * flip))))
        paste_scaled(frame, img, (x, y), scale=min(1.0, g))
    # Percent labels above the strongest few.
    if t >= word(SCORE, "softmax") and la > 0:
        for k in np.argsort(-w)[:2]:
            if k == IT or w[k] < 0.02:
                continue
            x, y = POS[k]
            paste_scaled(frame, tag(f"{round(100 * w[k])}%", (20, 20, 30), WHITE + (255,), 30, pad=(10, 4)),
                         (x, y - 52), alpha=la)


def draw_panel(frame, t):
    """Scores, then softmax percentages, as bars (top 5)."""
    t0 = word(SCORE, "scores")
    if t < t0 or t >= T.start(DISTANCE):
        return
    a = clamp((t - t0) / 0.4) * (1 - clamp((t - T.start(DISTANCE) + 0.3) / 0.3))
    d = ImageDraw.Draw(frame)
    soft = t >= word(SCORE, "softmax")
    g_soft = ease_in_out((t - word(SCORE, "softmax")) / 0.8) if soft else 0.0
    g = swap_progress(t)
    scores = SCORES_T * (1 - g) + SCORES_W * g
    w = weights_now(t)
    order = list(np.argsort(-(w if soft else scores)))
    order = [k for k in order if k != IT][:5]
    x0, x1, y0 = 330, 900, PANEL_Y
    title = "attention from “it” (softmax %)" if soft else "match score with “it”"
    text_center(d, (540, y0), title, font("Black", 34), lerp_color((10, 10, 20), WHITE, a))
    smax = max(scores)
    for r, k in enumerate(order):
        y = y0 + 70 + r * 70
        lab = SENT[k] if k < len(SENT) - 1 else last_word(t)
        text_center(d, (x0 - 20, y), lab, font("Black", 34), lerp_color((10, 10, 20), WHITE, a), anchor="rm")
        frac = lerp(scores[k] / smax, w[k], g_soft)
        col = GREEN if r == 0 else CYAN
        d.rounded_rectangle((x0, y - 22, x1, y + 22), radius=22, fill=lerp_color((10, 10, 20), (36, 40, 74), a))
        if frac > 0.02:
            d.rounded_rectangle((x0, y - 22, x0 + max(44, (x1 - x0) * frac), y + 22), radius=22,
                                fill=lerp_color((10, 10, 20), col, a))
        val = f"{round(100 * w[k])}%" if g_soft > 0.5 else f"{scores[k]:.1f}"
        text_center(d, (x1 + 14, y), val, font("Black", 32), lerp_color((10, 10, 20), WHITE, a), anchor="lm")


def draw_stack(frame, t):
    t0 = T.start(STACK)
    if not (t0 <= t < T.start(CTA)):
        return
    d = ImageDraw.Draw(frame)
    n = 5
    for j in range(n):
        g = ease_back((t - t0 - 0.15 * j) / 0.4)
        if g <= 0:
            continue
        y = 1300 - j * 58
        hw = 300 * min(1, g)
        d.rounded_rectangle((540 - hw, y - 24, 540 + hw, y + 24), radius=16, fill=(40 + 12 * j, 30, 90 + 18 * j),
                            outline=PURPLE, width=3)
        text_center(d, (540, y), "attention layer" if j < n - 1 else "… dozens more …", font("Bold", 28), WHITE)
    t_n = word(STACK, "next")
    if t >= t_n:
        pop_in(frame, tag("The cat sat on the ___  →  mat", (12, 28, 18), GREEN + (255,), 40), (540, 925), t, t_n)
        pop_in(frame, tag("predict the next word (illustration)", (20, 20, 30), MUTED + (255,), 26, pad=(12, 5)),
               (540, 990), t, t_n + 0.2)
    t_c = word(STACK, "chatbot")
    if t >= t_c:
        pop_in(frame, tag("= CHATBOT", (20, 20, 30), YELLOW + (255,), 48), (540, 1390), t, t_c)


def draw_overlays(frame, t):
    draw_hud_pills(frame, 12, *hud(t), label="FOCUS")
    frame.alpha_composite(TITLE_IMG, (0, 190))
    banners = [
        (T.start(HOOK), T.start(PARALLEL), "ONE IDEA BEHIND MODERN CHATBOTS", PURPLE, WHITE),
        (T.start(PARALLEL), T.start(ASK), "ONE AT A TIME vs ALL AT ONCE", CYAN, (12, 14, 30)),
        (T.start(ASK), T.start(SENTENCE), "EACH WORD: WHO MATTERS TO ME?", YELLOW, (20, 20, 30)),
        (T.start(SENTENCE), T.start(SCORE), "WHAT DOES “IT” MEAN?", PURPLE, WHITE),
        (T.start(SCORE), word(SCORE, "softmax"), "STEP 1 · SCORE EVERY WORD (MEANING MATCH)", CYAN, (12, 14, 30)),
        (word(SCORE, "softmax"), T.start(RESULT), "STEP 2 · SOFTMAX → PERCENTAGES", CYAN, (12, 14, 30)),
        (T.start(RESULT), T.start(SWAP), f"“IT” = THE ANIMAL ({P_T}%)", GREEN, (12, 28, 18)),
        (T.start(SWAP), T.start(DISTANCE), f"TIRED → WIDE: “IT” = THE STREET ({P_W}%)", ORANGE, (20, 20, 30)),
        (T.start(DISTANCE), T.start(STACK), "EVERY WORD SEES EVERY WORD → NOTHING FADES", GREEN, (12, 28, 18)),
        (T.start(STACK), T.start(CTA), "STACK LAYERS + PREDICT NEXT WORD", PURPLE, WHITE),
    ]
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 32), (W / 2, TOP_TAG_Y), t, a)
    if T.start(DISTANCE) <= t < T.start(STACK):
        pop_in(frame, tag("first ↔ last: just one step", (12, 28, 18), GREEN + (255,), 34), (540, 960), t,
               word(DISTANCE, "distance"))
    draw_level_cleared(frame, t, T.start(CTA), 12, "YOU DECIDE!", word(CTA, "what"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    draw_hook(frame, t)
    draw_parallel(frame, t)
    draw_sentence(frame, t)
    draw_panel(frame, t)
    draw_stack(frame, t)
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
    for w_ in ("chatgpt", "claude", "gemini"):
        mix(out, sfx_blip(880, 0.08, 0.12), word(HOOK, w_))
    mix(out, sfx_boing(0.5, 0.18), word(HOOK, "transformer"))
    t0 = T.start(PARALLEL) + 0.3
    for k in range(6):
        mix(out, sfx_blip(600 + 60 * k, 0.05, 0.09), t0 + 0.55 * k)
    mix(out, sfx_blip(1320, 0.25, 0.16), word(PARALLEL, "transformer") + 0.4)
    mix(out, sfx_blip(1568, 0.3, 0.16), word(ASK, "attention"))
    for k in range(len(SENT)):
        mix(out, sfx_blip(700 + 30 * k, 0.04, 0.07), T.start(SENTENCE) + 0.05 * k)
    mix(out, sfx_whoosh(0.8, 0.16, seed=5), word(SCORE, "scores"))
    mix(out, sfx_boing(0.5, 0.16), word(SCORE, "softmax"))
    mix(out, sfx_blip(1568, 0.35, 0.2), word(RESULT, "animal"))
    mix(out, sfx_whoosh(0.7, 0.18, seed=8), word(SWAP, "wide"))
    mix(out, sfx_blip(1320, 0.3, 0.18), word(SWAP, "street"))
    mix(out, sfx_whoosh(0.9, 0.14, seed=9), T.start(DISTANCE))
    for j in range(5):
        mix(out, sfx_blip(500 + 100 * j, 0.06, 0.1), T.start(STACK) + 0.15 * j)
    mix(out, sfx_blip(1568, 0.3, 0.18), word(STACK, "chatbot"))
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION, POS
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12, max_px=800)
    STATIC = build_plain()
    TITLE_IMG = build_title("TRANSFORMERS", "ML AS A GAME  ·  EP 12")
    CONFETTI = build_confetti()
    POS = word_positions()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = word(RESULT, "animal") + 0.8
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "HOW CHATBOTS READ", font("Black", 82), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "Transformers & attention, explained", font("Black", 52), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s")
    for name, words, sc, att in (("tired", WORDS_T, SCORES_T, ATT_T), ("wide", WORDS_W, SCORES_W, ATT_W)):
        print(f"  ...too {name}: " + ", ".join(f"{w_} {s:.1f}->{100 * a:.0f}%" for w_, s, a in zip(words, sc, att)))
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
    out = os.path.join(HERE, "transformers_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
