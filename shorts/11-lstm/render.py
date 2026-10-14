"""ML as a Game, Level 11: LSTM (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> lstm_short.mp4 + thumbnail.png
    python3 render.py --preview 3,12  # just dump PNG frames at those seconds (fast check)
    python3 render.py --thumbnail     # only regenerate thumbnail.png
"""

import argparse
import math
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from common import (  # noqa: E402
    CYAN, FPS, GREEN, H, MUTED, ORANGE, PINK, PURPLE, SR, W, WHITE, YELLOW, Timeline, add_glow, build_confetti,
    build_plain, build_title, caption_chunks, clamp, draw_caption, draw_hud_pills, draw_level_cleared, ease_back,
    ease_out, encode, font, lerp, lerp_color, mix, music_bed, paste_scaled, pop_in, sfx_blip, sfx_boing, sfx_buzz,
    sfx_levelup, sfx_whoosh, tag, text_center, voiceover, write_wav,
)


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# ---------------------------------------------------------------- the tiny LSTM: one memory cell

WORDS = ["I", "grew", "up", "in", "France", "and", "years", "later", "moved", "away", "so", "I", "speak"]
KEY = WORDS.index("France")


def lstm_step(word, c):
    country = 1.0 if word == "France" else 0.0
    question = 1.0 if word == "speak" else 0.0
    f = sigmoid(4.0)  # forget gate: keep (almost) all old memory
    i = sigmoid(6 * country - 3)  # input gate: write only important words
    g = math.tanh(2 * country)  # what would be written
    o = sigmoid(6 * question - 3)  # output gate: speak only when asked
    c = f * c + i * g
    return dict(f=f, i=i, g=g, o=o, c=c, h=o * math.tanh(c))


STEPS = []
_c = 0.0
for _w in WORDS:
    STEPS.append(lstm_step(_w, _c))
    _c = STEPS[-1]["c"]
C_KEY = STEPS[KEY]["c"]


def lstm_share(k):
    """How much of 'France' is still in the LSTM memory k words after it (exact: ratio of memories)."""
    return STEPS[KEY + k]["c"] / C_KEY


def rnn_share(k):
    return 0.5 ** k  # Level 10's RNN keeps half per word


N_AFTER = len(WORDS) - 1 - KEY
assert lstm_share(N_AFTER) > 0.75 and rnn_share(N_AFTER) < 0.01
assert STEPS[-1]["h"] > 0.5 and all(s["h"] < 0.1 for s in STEPS[:-1])  # only speaks at "speak"

SCRIPT = [
    "I grew up in France. Years later, I speak... what? To answer, you have to remember one word from way back.",
    "Last level, a simple recurrent network forgot the start. Its memory halved every word.",
    "Long short-term memory fixes that. It keeps a memory belt, with three gates, like valves.",
    "The forget gate decides how much old memory to keep. The input gate decides what new info to write.",
    "And the output gate decides what to say right now.",
    "Watch. France: the input gate opens, and France goes onto the belt.",
    "Filler words: the forget gate stays open, so the memory barely fades.",
    "Speak: the output gate opens, reads the belt... French!",
    "Gates let these networks remember for a long time. They powered speech recognition and translation for years.",
    "Level eleven cleared. Next: transformers, how chatbots read. Follow for level twelve.",
]
HOOK, LAST, BELT, GATES, OUTG, WATCH, FILLER, SPEAK, POWER, CTA = range(10)

# ---------------------------------------------------------------- layout

CELL = (140, 660, 940, 1000)
BELT_Y = 610
GAUGE_XS = [290, 540, 790]
GAUGE_Y0, GAUGE_Y1 = 740, 930
GAUGE_W = 74
GATE_INFO = [("FORGET", "keep", CYAN, "f"), ("INPUT", "write", GREEN, "i"), ("OUTPUT", "say", ORANGE, "o")]
WORD_Y = 1068
METER_Y = (1180, 1290)
CAPTION_Y = 1495
TOP_TAG_Y = 425

T: Timeline = None
CHUNKS = []
STATIC = None
TITLE_IMG = None
CONFETTI = None
DURATION = 0.0


def word(i, prefix, nth=0):
    return T.word(i, prefix, nth)


# ---------------------------------------------------------------- state

def step_times():
    t_w = word(WATCH, "watch")
    times = [t_w + 0.15 + 0.14 * k for k in range(KEY)]
    times.append(word(WATCH, "france:"))
    t0, t1 = T.start(FILLER) + 0.1, T.end(FILLER) - 0.1
    n_fill = len(WORDS) - KEY - 2
    times += [lerp(t0, t1, (k + 0.5) / n_fill) for k in range(n_fill)]
    times.append(word(SPEAK, "speak:"))
    return times


TIMES = []


def step_index(t):
    """Index of the word being processed (or -1 before the stream starts)."""
    n = -1
    for k, ts in enumerate(TIMES):
        if t >= ts:
            n = k
    return n


def gate_values(t):
    """(f, i, o) shown on the gauges, eased from the previous step's values."""
    k = step_index(t)
    if k < 0:
        return 0.0, 0.0, 0.0
    cur = STEPS[k]
    prev = STEPS[k - 1] if k else dict(f=0, i=0, o=0)
    g = ease_out((t - TIMES[k]) / 0.35)
    return tuple(lerp(prev[x], cur[x], g) for x in ("f", "i", "o"))


def memory_now(t):
    k = step_index(t)
    if k < 0:
        return 0.0
    prev = STEPS[k - 1]["c"] if k else 0.0
    return lerp(prev, STEPS[k]["c"], ease_out((t - TIMES[k]) / 0.45))


def shares(t):
    """(rnn share, lstm share) of 'France' still remembered, or None before 'France'."""
    k = step_index(t)
    if k < KEY:
        return None
    j = k - KEY
    g = ease_out((t - TIMES[k]) / 0.45)
    if j == 0:
        return 1.0, 1.0
    return lerp(rnn_share(j - 1), rnn_share(j), g), lerp(lstm_share(j - 1), lstm_share(j), g)


# ---------------------------------------------------------------- drawing

def draw_hook(frame, t):
    t_end = T.start(BELT)
    if t >= t_end + 0.4:
        return
    a = 1 - clamp((t - t_end) / 0.4)
    img = Image.new("RGBA", (W, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f = font("Black", 60)
    parts = [("I grew up in ", WHITE), ("France", YELLOW), (".", WHITE)]
    x = (W - sum(f.getlength(p) for p, _ in parts)) / 2
    for p, c in parts:
        d.text((x, 80), p, font=f, fill=c, anchor="lm")
        x += f.getlength(p)
    blink = "___" if int(t * 3) % 2 else "   "
    text_center(d, (W / 2, 180), f"Years later, I speak {blink}", font("Black", 60), WHITE)
    if t >= word(HOOK, "remember"):
        text_center(d, (W / 2, 262), "← one word from way back", font("Bold", 36), YELLOW)
    paste_scaled(frame, img, (W / 2, 740), alpha=a * clamp(t / 0.3))


def draw_rnn_demo(frame, t):
    """Scene 2: Level 10's memory halving, word by word."""
    t0 = T.start(LAST)
    if not (t0 <= t < T.start(BELT) + 0.3):
        return
    a = clamp((t - t0) / 0.3) * (1 - clamp((t - T.start(BELT)) / 0.3))
    k = int(clamp((t - t0 - 0.4) / (T.end(LAST) - t0 - 0.4)) * N_AFTER + 1e-6)
    draw_meter(frame, METER_Y[0], "simple RNN: France left", rnn_share(k), PINK, a)


def draw_meter(frame, y, label, v, col, a=1.0):
    d = ImageDraw.Draw(frame)
    x0, x1 = 380, 940
    text_center(d, (x0 - 16, y), label, font("Bold", 27), lerp_color((10, 10, 20), WHITE, a), anchor="rm")
    d.rounded_rectangle((x0, y - 20, x1, y + 20), radius=20, fill=lerp_color((10, 10, 20), (40, 44, 80), a))
    if v > 0.004:
        d.rounded_rectangle((x0, y - 20, x0 + max(40, (x1 - x0) * v), y + 20), radius=20,
                            fill=lerp_color((10, 10, 20), col, a))
    text_center(d, (x1 - 16, y), f"{v:.3f}" if v < 0.1 else f"{v:.2f}", font("Black", 28),
                lerp_color((10, 10, 20), WHITE, a), anchor="rm")


def draw_cell(frame, t):
    t0 = word(BELT, "belt")
    if t < t0 - 0.6 or t >= T.start(CTA):
        return
    g_in = ease_back((t - (t0 - 0.6)) / 0.5)
    lay = Image.new("RGBA", (W, 560), (0, 0, 0, 0))
    oy = 520
    d = ImageDraw.Draw(lay)
    x0, y0, x1, y1 = CELL
    k = step_index(t)
    # The memory belt (conveyor) with moving ticks.
    d.line([(30, BELT_Y - oy), (1050, BELT_Y - oy)], fill=(70, 78, 120, 255), width=16)
    shift = (t * 90) % 60
    for xx in range(30, 1050, 60):
        d.line([(xx + shift, BELT_Y - oy - 8), (xx + shift, BELT_Y - oy + 8)], fill=(110, 118, 170, 255), width=4)
    # Cell box.
    hb = (y1 - y0) / 2 * min(1, g_in)
    cy = (y0 + y1) / 2 - oy
    d.rounded_rectangle((x0, cy - hb, x1, cy + hb), radius=36, fill=(20, 22, 50, 245), outline=PURPLE + (255,),
                        width=6)
    lay = add_glow(lay, radius=8, strength=0.8)
    frame.alpha_composite(lay, (0, oy))
    if g_in < 0.8:
        return
    d = ImageDraw.Draw(frame)

    # Gauges.
    vals = gate_values(t)
    t_g = [word(GATES, "forget"), word(GATES, "input"), word(OUTG, "output")]
    for j, (name, verb, col, key) in enumerate(GATE_INFO):
        if t < t_g[j] and k < 0:
            continue
        gx = GAUGE_XS[j]
        intro = t_g[j] <= t < t_g[j] + 2.2 and k < 0
        ga = ease_back((t - t_g[j]) / 0.4) if t < t_g[j] + 0.4 else 1.0
        text_center(d, (gx, GAUGE_Y0 - 30), name, font("Black", 30), col)
        d.rounded_rectangle((gx - GAUGE_W / 2, GAUGE_Y0, gx + GAUGE_W / 2, GAUGE_Y1), radius=16, fill=(36, 40, 74),
                            outline=(YELLOW if intro else (70, 78, 120)), width=5 if intro else 3)
        v = vals[j] if k >= 0 else (0.5 + 0.5 * math.sin((t - t_g[j]) * 3)) * ga if intro else 0.0
        top = GAUGE_Y1 - (GAUGE_Y1 - GAUGE_Y0) * clamp(v)
        if top < GAUGE_Y1 - 12:  # skip slivers too thin to draw
            d.rounded_rectangle((gx - GAUGE_W / 2 + 6, top, gx + GAUGE_W / 2 - 6, GAUGE_Y1 - 6), radius=12, fill=col)
        text_center(d, (gx, GAUGE_Y1 + 32), f"{verb}: {v:.2f}" if k >= 0 else verb, font("Bold", 28), WHITE)
        # Arrows to/from the belt.
        if j < 2:
            d.line([(gx, GAUGE_Y0 - 56), (gx, BELT_Y + 14)], fill=col, width=5)
            d.polygon([(gx, BELT_Y + 8), (gx - 9, BELT_Y + 24), (gx + 9, BELT_Y + 24)], fill=col)
        else:
            d.line([(gx, BELT_Y + 14), (gx, GAUGE_Y0 - 56)], fill=col, width=5)
            d.polygon([(gx, GAUGE_Y0 - 50), (gx - 9, GAUGE_Y0 - 66), (gx + 9, GAUGE_Y0 - 66)], fill=col)
        tx = ["× keep", "+ write", "read"][j]
        text_center(d, (gx + 12, BELT_Y + 56), tx, font("Bold", 22), col, anchor="lm")

    # Memory box riding on the belt.
    c = memory_now(t)
    mx = 540
    col = lerp_color((40, 44, 80), YELLOW, clamp(c))
    d.rounded_rectangle((mx - 110, BELT_Y - 40, mx + 110, BELT_Y + 40), radius=18, fill=col, outline=WHITE, width=3)
    label = "memory" if k < KEY else "France"
    ink = (20, 20, 30) if c > 0.35 else WHITE
    text_center(d, (mx, BELT_Y - 12), label, font("Black", 28), ink)
    text_center(d, (mx, BELT_Y + 18), f"{c:.2f}", font("Black", 28), ink)


def draw_words(frame, t):
    k = step_index(t)
    if k < 0 or t >= T.start(CTA):
        return
    d = ImageDraw.Draw(frame)
    # Current word (big), previous words trailing to the left.
    for j in range(max(0, k - 3), k + 1):
        dist = k - j
        x = 540 - dist * 150
        a = 1.0 if dist == 0 else 0.55 - 0.1 * dist
        w = WORDS[j]
        key = j == KEY
        img = tag(w, (20, 20, 30) if key else WHITE, (YELLOW if key else (70, 78, 120)) + (255,),
                  44 if dist == 0 else 30, pad=(18, 10))
        if dist == 0:
            pop_in(frame, img, (x, WORD_Y), t, TIMES[j], dur=0.25)
            d.line([(540, WORD_Y - 34), (540, CELL[3] + 6)], fill=WHITE, width=4)
        else:
            paste_scaled(frame, img, (x, WORD_Y), alpha=a)
    # Comparison meters.
    sh = shares(t)
    if sh:
        draw_meter(frame, METER_Y[0], "simple RNN: France left", sh[0], PINK)
        draw_meter(frame, METER_Y[1], "LSTM: France left", sh[1], GREEN)


def draw_output(frame, t):
    t_f = word(SPEAK, "french")
    if t_f - 0.3 <= t < T.start(CTA):
        pop_in(frame, tag("says: “French!”", (12, 28, 18), GREEN + (255,), 44), (790, 540), t, t_f - 0.3)
        h = STEPS[-1]["h"]
        paste_scaled(frame, tag(f"output = {h:.2f}", (20, 20, 30), WHITE + (255,), 26, pad=(12, 5)), (790, 486))


def hud(t):
    if t < T.start(LAST) - 0.1:
        return f"{abs(math.sin(t * 5)):.2f}", PINK
    k = step_index(t)
    if k < 0:
        return "0.00", MUTED
    c = memory_now(t)
    return f"{c:.2f}", GREEN if c > 0.5 else (CYAN if c > 0.05 else MUTED)


def draw_overlays(frame, t):
    draw_hud_pills(frame, 11, *hud(t), label="MEMORY")
    frame.alpha_composite(TITLE_IMG, (0, 190))
    banners = [
        (T.start(HOOK), T.start(LAST), "REMEMBER ONE WORD FROM WAY BACK", PURPLE, WHITE),
        (T.start(LAST), T.start(BELT), "LEVEL 10: MEMORY HALVES EVERY WORD", PINK, WHITE),
        (T.start(BELT), T.start(GATES), "LSTM = MEMORY BELT + 3 GATES", PURPLE, WHITE),
        (T.start(GATES), word(GATES, "input"), "FORGET GATE: HOW MUCH OLD MEMORY TO KEEP", CYAN, (12, 14, 30)),
        (word(GATES, "input"), T.start(OUTG), "INPUT GATE: WHAT NEW INFO TO WRITE", GREEN, (12, 28, 18)),
        (T.start(OUTG), T.start(WATCH), "OUTPUT GATE: WHAT TO SAY NOW", ORANGE, (20, 20, 30)),
        (word(WATCH, "france:"), T.start(FILLER), "FRANCE → WRITTEN ONTO THE BELT", YELLOW, (20, 20, 30)),
        (T.start(FILLER), T.start(SPEAK), f"FILLER: FORGET GATE STAYS OPEN ({STEPS[0]['f']:.2f})", CYAN, (12, 14, 30)),
        (T.start(SPEAK), T.start(POWER), "OUTPUT GATE OPENS → READS THE BELT", ORANGE, (20, 20, 30)),
        (T.start(POWER), T.start(CTA), "GATES = LONG MEMORY", GREEN, (12, 28, 18)),
    ]
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 32), (W / 2, TOP_TAG_Y), t, a)
    draw_level_cleared(frame, t, T.start(CTA), 11, "TRANSFORMERS", word(CTA, "next:"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    draw_hook(frame, t)
    draw_rnn_demo(frame, t)
    draw_cell(frame, t)
    draw_words(frame, t)
    draw_output(frame, t)
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
    t0 = T.start(LAST) + 0.4
    for k in range(N_AFTER + 1):
        mix(out, sfx_blip(1000 - 70 * k, 0.04, 0.08), lerp(t0, T.end(LAST), k / N_AFTER))
    mix(out, sfx_whoosh(0.7, 0.16, seed=4), word(BELT, "belt") - 0.6)
    for ts in (word(GATES, "forget"), word(GATES, "input"), word(OUTG, "output")):
        mix(out, sfx_boing(0.45, 0.14), ts)
    for k, ts in enumerate(TIMES):
        if k == KEY:
            mix(out, sfx_blip(1568, 0.3, 0.18), ts)
        elif k == len(TIMES) - 1:
            mix(out, sfx_boing(0.5, 0.16), ts)
        else:
            mix(out, sfx_blip(700, 0.04, 0.07), ts)
    mix(out, sfx_blip(1568, 0.35, 0.2), word(SPEAK, "french"))
    mix(out, sfx_buzz(0.15, 160, 0.06), T.start(LAST) + 0.2)
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION, TIMES
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    TIMES = step_times()
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12, max_px=800)
    STATIC = build_plain()
    TITLE_IMG = build_title("LSTM", "ML AS A GAME  ·  EP 11")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = word(SPEAK, "french") + 0.6
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "MEMORY THAT LASTS", font("Black", 84), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "LSTMs, explained", font("Black", 56), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s")
    for w_, s in zip(WORDS, STEPS):
        print(f"  {w_:7s} f={s['f']:.2f} i={s['i']:.2f} g={s['g']:.2f} c={s['c']:.2f} o={s['o']:.2f} h={s['h']:.2f}")
    print(f"France left after {N_AFTER} words: LSTM {lstm_share(N_AFTER):.3f} vs simple RNN {rnn_share(N_AFTER):.4f}")
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
    out = os.path.join(HERE, "lstm_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
