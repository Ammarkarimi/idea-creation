"""ML as a Game, Level 10: Recurrent Neural Networks (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> rnn_short.mp4 + thumbnail.png
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
    ease_in_out, ease_out, encode, font, lerp, lerp_color, mix, music_bed, paste_scaled, pop_in, sfx_blip,
    sfx_boing, sfx_levelup, sfx_whoosh, tag, text_center, voiceover, write_wav,
)

# ---------------------------------------------------------------- the tiny RNN: one memory number

KEEP = 0.5  # how much of the old memory each step keeps
SCORE = {"boring": -1.0, "amazing": 1.0, "start": 0.0, "but": 0.0, "ending": 0.0}
SENT_A = ["boring", "start", "but", "amazing", "ending"]
SENT_B = ["amazing", "start", "but", "boring", "ending"]


def run(words):
    """memory_t = tanh(word_t + KEEP * memory_{t-1}), starting from 0."""
    m, out = 0.0, []
    for w in words:
        m = math.tanh(SCORE[w] + KEEP * m)
        out.append(m)
    return out


MEM_A, MEM_B = run(SENT_A), run(SENT_B)
assert sorted(SENT_A) == sorted(SENT_B)  # same words...
assert MEM_A[-1] > 0 > MEM_B[-1]  # ...opposite answers: order matters
assert np.allclose(MEM_A, [-m for m in MEM_B])
FADE = [KEEP ** k for k in range(11)]  # first word's share after k more steps

SCRIPT = [
    "Is this review good or bad? To know, you have to read it in order.",
    "A recurrent neural network reads one word at a time, and keeps a memory.",
    "At each word, the same neuron mixes the new word with half of its old memory, then passes it on.",
    "Boring: memory goes negative. Start, but: it fades. Amazing: it flips positive. Good review!",
    "Now swap it: amazing start, but boring ending. Same words, opposite answer.",
    "That's a recurrent network. Memory flows forward, so word order matters.",
    "But each step keeps only half of the old memory. After ten words, the first word is almost gone.",
    "Long sentences make simple recurrent networks forget the beginning.",
    "Level ten cleared. Next: long short-term memory, a memory that doesn't fade. Follow for level eleven.",
]
HOOK, READS, RULE, RUN_A, RUN_B, POINT, FADE_S, FORGET, CTA = range(9)

# ---------------------------------------------------------------- layout

XS = [140, 340, 540, 740, 940]
ROW_A, ROW_B = 790, 1150  # cell centres
CELL_R = 64
WORD_DY = 122
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


def fmt(v):
    return ("−" if v < 0 else "+") + f"{abs(v):.2f}"


# ---------------------------------------------------------------- state

def step_times(chain):
    """When each step of a chain is computed."""
    if chain == 0:
        return [word(RUN_A, "boring"), word(RUN_A, "start"), word(RUN_A, "but"), word(RUN_A, "amazing"),
                word(RUN_A, "good") - 0.45]
    return [word(RUN_B, "amazing"), word(RUN_B, "start"), word(RUN_B, "but"), word(RUN_B, "boring"),
            word(RUN_B, "ending")]


def steps_done(chain, t):
    return sum(1 for ts in step_times(chain) if t >= ts)


def chains_alpha(t):
    """Chains fade out when the fading chart takes over."""
    return 1 - clamp((t - T.start(FADE_S) + 0.1) / 0.4)


def hud(t):
    if t < T.start(READS) - 0.1:
        return f"{math.sin(t * 9):+.2f}", PINK
    for chain, mem in ((1, MEM_B), (0, MEM_A)):
        n = steps_done(chain, t)
        if n and t < T.start(FADE_S):
            v = mem[n - 1]
            return fmt(v), GREEN if v > 0 else PINK
    if t >= T.start(FADE_S):
        k = fade_k(t)
        return f"{FADE[k]:.3f}", lerp_color(GREEN, PINK, k / 10)
    return "0.00", MUTED


def fade_k(t):
    t0 = word(FADE_S, "after")
    return int(clamp((t - t0) / max(0.5, T.end(FADE_S) - t0)) * 10 + 1e-6) if t >= t0 else 0


# ---------------------------------------------------------------- drawing

def mem_color(v):
    return lerp_color((26, 30, 60), GREEN if v > 0 else PINK, min(1.0, abs(v) * 1.2))


def draw_chain(frame, t, chain):
    words = SENT_A if chain == 0 else SENT_B
    mem = MEM_A if chain == 0 else MEM_B
    y = ROW_A if chain == 0 else ROW_B
    t_show = T.start(READS) if chain == 0 else word(RUN_B, "swap")
    a = chains_alpha(t)
    if t < t_show or a <= 0:
        return
    lay = Image.new("RGBA", (W, 420), (0, 0, 0, 0))
    oy = y - 160
    d = ImageDraw.Draw(lay)
    times = step_times(chain)
    n = steps_done(chain, t)
    g_all = ease_out((t - t_show) / 0.5)

    # Memory arrows (start value 0 enters from the left).
    for i in range(len(words)):
        x_from = XS[i - 1] + CELL_R if i else XS[0] - CELL_R - 90
        x_to = XS[i] - CELL_R - 4
        passed = i == 0 or n >= i
        col = (mem_color(mem[i - 1]) if i and n >= i else (90, 98, 140))
        d.line([(x_from, 160), (x_to, 160)], fill=col + (int(255 * g_all),), width=8)
        d.polygon([(x_to + 4, 160), (x_to - 12, 150), (x_to - 12, 170)], fill=col + (int(255 * g_all),))
        if i and n >= i and t < times[i - 1] + 0.9:  # memory packet travelling to the next cell
            f = clamp((t - times[i - 1] - 0.15) / 0.6)
            px = lerp(x_from, x_to, f)
            d.ellipse([px - 13, 147, px + 13, 173], fill=YELLOW + (255,))
        del passed
    # Cells.
    for i, w in enumerate(words):
        g = ease_back((t - t_show - 0.08 * i) / 0.4)
        if g <= 0:
            continue
        done = i < n
        active = done and t < times[i] + 0.8
        v = mem[i] if done else 0.0
        r = CELL_R * g * (1 + (0.12 * math.sin((t - times[i]) * 10) if active else 0))
        d.ellipse([XS[i] - r, 160 - r, XS[i] + r, 160 + r], fill=(mem_color(v) if done else (22, 24, 52)) + (255,),
                  outline=(YELLOW if active else PURPLE) + (255,), width=7 if active else 5)
        if g > 0.7:
            text_center(d, (XS[i], 160), fmt(v) if done else "?", font("Black", 34 if done else 44),
                        WHITE if done else MUTED)
    frame_lay = add_glow(lay, radius=8, strength=0.9)
    if a < 1:
        arr = np.asarray(frame_lay).copy()
        arr[..., 3] = (arr[..., 3] * a).astype(np.uint8)
        frame_lay = Image.fromarray(arr, "RGBA")
    frame.alpha_composite(frame_lay, (0, oy))

    # Word cards under the cells.
    for i, w in enumerate(words):
        g = ease_back((t - t_show - 0.08 * i) / 0.4)
        if g <= 0:
            continue
        s = SCORE[w]
        col = GREEN if s > 0 else (PINK if s < 0 else (70, 78, 120))
        fg = (12, 28, 18) if s > 0 else WHITE
        card = tag(w, fg, col + (255,), 34, pad=(14, 8))
        paste_scaled(frame, card, (XS[i], y + WORD_DY), scale=min(1, g), alpha=a)
        sub = tag(f"{int(s):+d}" if s else "0", (20, 20, 30), WHITE + (230,), 26, pad=(9, 3))
        paste_scaled(frame, sub, (XS[i], y + WORD_DY + 50), scale=min(1, g), alpha=a * 0.9)
    if chain == 0 and word(READS, "memory") <= t < step_times(0)[0]:
        pop_in(frame, tag("memory starts at 0", (20, 20, 30), MUTED + (255,), 26, pad=(10, 4)),
               (XS[0] + 60, y - CELL_R - 34), t, word(READS, "memory"))

    # Verdict.
    t_v = word(RUN_A, "good") if chain == 0 else word(RUN_B, "opposite")
    if t >= t_v and a > 0:
        good = mem[-1] > 0
        txt = f"ends {fmt(mem[-1])} → {'GOOD' if good else 'BAD'} REVIEW"
        img = tag(txt, (12, 28, 18) if good else WHITE, (GREEN if good else PINK) + (255,), 34)
        if a >= 1:
            pop_in(frame, img, (W / 2, y + WORD_DY + 104), t, t_v)
        else:
            paste_scaled(frame, img, (W / 2, y + WORD_DY + 104), alpha=a)


def draw_formula(frame, t):
    t0 = word(RULE, "same")
    if t < t0 or t >= T.start(FADE_S) + 0.4:
        return
    a = clamp((t - t0) / 0.4) * chains_alpha(t)
    card = Image.new("RGBA", (980, 150), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    d.rounded_rectangle((0, 0, 979, 149), radius=26, fill=(18, 20, 46, 240), outline=PURPLE + (255,), width=4)
    f = font("Black", 40)
    parts = [("memory = squash( ", WHITE), ("word", YELLOW), (" + ", WHITE), ("½ × old memory", CYAN), (" )", WHITE)]
    x = (980 - sum(f.getlength(p) for p, _ in parts)) / 2
    for p, c in parts:
        d.text((x, 56), p, font=f, fill=c, anchor="lm")
        x += f.getlength(p)
    text_center(d, (490, 112), "squash (tanh) keeps memory between −1 and +1", font("Bold", 26), MUTED)
    paste_scaled(frame, card, (W / 2, 560), alpha=a)


def draw_fade(frame, t):
    t0 = T.start(FADE_S)
    if t < t0 or t >= T.start(CTA):
        return
    a = clamp((t - t0 - 0.2) / 0.4)
    d = ImageDraw.Draw(frame)
    x0, x1, y0, y1 = 120, 980, 640, 1180
    text_center(d, ((x0 + x1) / 2, y0 - 10), "how much of word #1 is still in memory",
                font("Bold", 32), lerp_color((0, 0, 0), WHITE, a))
    k_now = fade_k(t)
    bw = (x1 - x0) / 11
    for k, v in enumerate(FADE):
        g = ease_out((t - t0 - 0.3 - 0.06 * k) / 0.4) * a
        if g <= 0:
            continue
        hgt = (y1 - y0 - 120) * v * g
        bx = x0 + k * bw + 8
        col = lerp_color(GREEN, PINK, k / 10)
        if k > k_now and t >= word(FADE_S, "after"):
            col = lerp_color(col, (40, 44, 80), 0.6)
        d.rounded_rectangle((bx, y1 - max(hgt, 6), bx + bw - 16, y1), radius=10, fill=col)
        lab = f"{v:.3f}".rstrip("0").rstrip(".") if v < 1 else "1"
        text_center(d, (bx + (bw - 16) / 2, y1 - max(hgt, 6) - 20), lab, font("Black", 24), WHITE)
        text_center(d, (bx + (bw - 16) / 2, y1 + 26), f"{k}", font("Bold", 26), MUTED)
    text_center(d, ((x0 + x1) / 2, y1 + 68), "words later", font("Bold", 28), MUTED)
    t_f = word(FORGET, "forget")
    if t >= t_f:
        pop_in(frame, tag("FORGETS THE START", WHITE, PINK + (255,), 42), (W / 2, 1320), t, t_f)


def draw_hook(frame, t):
    """Jumbled word cards that snap into reading order."""
    if t >= T.start(READS) + 0.2:
        return
    t_order = word(HOOK, "order")
    jumbled = [(300, 1080, 0), (760, 700, 0), (250, 760, 0), (800, 1000, 0), (530, 880, 0)]  # spread out
    g = ease_in_out((t - t_order) / 0.6) if t >= t_order else 0.0
    a = 1 - clamp((t - T.start(READS)) / 0.2)
    for i, w in enumerate(SENT_A):
        jx, jy, _ = jumbled[i]
        x = lerp(jx + 20 * math.sin(t * 2 + i), XS[i], g)
        yy = lerp(jy + 14 * math.cos(t * 2.3 + i), ROW_A + WORD_DY, g)
        s = SCORE[w]
        col = GREEN if s > 0 else (PINK if s < 0 else (70, 78, 120))
        size = int(lerp(40, 34, g))  # shrink to the chain's card size as they line up
        paste_scaled(frame, tag(w, (12, 28, 18) if s > 0 else WHITE, col + (255,), size, pad=(14, 8)), (x, yy),
                     scale=ease_back(t / 0.5), alpha=a)
        if g >= 1:
            paste_scaled(frame, tag(f"{i + 1}", (20, 20, 30), YELLOW + (255,), 26, pad=(10, 4)), (x, yy - 58), alpha=a)


def draw_overlays(frame, t):
    draw_hud_pills(frame, 10, *hud(t), label="MEMORY")
    frame.alpha_composite(TITLE_IMG, (0, 190))
    banners = [
        (T.start(HOOK), T.start(READS), "GOOD OR BAD? READ IN ORDER", PURPLE, WHITE),
        (T.start(READS), T.start(RULE), "RNN = ONE WORD AT A TIME + A MEMORY", CYAN, (12, 14, 30)),
        (T.start(RULE), T.start(RUN_A), "THE SAME NEURON AT EVERY STEP", PURPLE, WHITE),
        (T.start(RUN_A), T.start(RUN_B), "BORING START, BUT AMAZING ENDING", YELLOW, (20, 20, 30)),
        (T.start(RUN_B), T.start(POINT), "SWAPPED: AMAZING START, BUT BORING ENDING", YELLOW, (20, 20, 30)),
        (T.start(POINT), T.start(FADE_S), "MEMORY FLOWS FORWARD → ORDER MATTERS", GREEN, (12, 28, 18)),
        (T.start(FADE_S), T.start(FORGET), "EACH STEP: OLD MEMORY × ½", PINK, WHITE),
        (T.start(FORGET), T.start(CTA), "SIMPLE RNNs FORGET THE BEGINNING", PINK, WHITE),
    ]
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 32), (W / 2, TOP_TAG_Y), t, a)
    if T.start(POINT) <= t < T.start(FADE_S):
        pulse = 0.5 + 0.5 * math.sin((t - T.start(POINT)) * 6)
        d = ImageDraw.Draw(frame)
        for y in (ROW_A, ROW_B):
            d.line([(XS[0] - 60, y - CELL_R - 26), (XS[-1] + 40, y - CELL_R - 26)],
                   fill=lerp_color((60, 66, 104), YELLOW, pulse), width=5)
            d.polygon([(XS[-1] + 56, y - CELL_R - 26), (XS[-1] + 38, y - CELL_R - 36), (XS[-1] + 38, y - CELL_R - 16)],
                      fill=lerp_color((60, 66, 104), YELLOW, pulse))
    draw_level_cleared(frame, t, T.start(CTA), 10, "LSTM", word(CTA, "next:"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    draw_hook(frame, t)
    draw_formula(frame, t)
    draw_chain(frame, t, 0)
    draw_chain(frame, t, 1)
    draw_fade(frame, t)
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
    mix(out, sfx_whoosh(0.6, 0.18, seed=3), word(HOOK, "order"))
    for i in range(5):
        mix(out, sfx_blip(700 + 90 * i, 0.06, 0.1), T.start(READS) + 0.08 * i)
    mix(out, sfx_boing(0.5, 0.15), word(RULE, "same"))
    for chain, mem in ((0, MEM_A), (1, MEM_B)):
        for ts, v in zip(step_times(chain), mem):
            mix(out, sfx_blip(784 + 300 * v, 0.1, 0.13), ts)
    mix(out, sfx_blip(1568, 0.3, 0.18), word(RUN_A, "good"))
    mix(out, sfx_blip(392, 0.3, 0.18), word(RUN_B, "opposite"))
    mix(out, sfx_whoosh(0.8, 0.16, seed=6), T.start(POINT))
    t0 = word(FADE_S, "after")
    for k in range(11):
        mix(out, sfx_blip(1200 - 70 * k, 0.04, 0.08), lerp(t0, T.end(FADE_S), k / 10))
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12, max_px=800)
    STATIC = build_plain()
    TITLE_IMG = build_title("RECURRENT NETS", "ML AS A GAME  ·  EP 10")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = word(RUN_B, "opposite") + 0.8
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "AI WITH MEMORY", font("Black", 104), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "Recurrent Neural Networks, explained", font("Black", 52), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s | A {SENT_A}: {[round(v, 2) for v in MEM_A]} | "
          f"B {SENT_B}: {[round(v, 2) for v in MEM_B]} | fade {[round(v, 3) for v in FADE]}")
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
    out = os.path.join(HERE, "rnn_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
