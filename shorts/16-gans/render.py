"""ML as a Game, Level 16: GANs (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> gans_short.mp4 + thumbnail.png
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
    ease_in_out, encode, font, lerp, lerp_color, mix, music_bed, num_to_words, paste_scaled, pop_in, sfx_blip,
    sfx_boing, sfx_buzz, sfx_levelup, sfx_whoosh, tag, text_center, voiceover, write_wav,
)

# ---------------------------------------------------------------- a real (tiny) GAN, trained in numpy

REAL_MEAN, REAL_SD = 8.0, 0.5  # real gold coins: weight in grams
FAKE_SD = 0.5  # the forger's natural wobble (kept fixed; the forger learns the weight it aims for)
B0 = 2.0  # the forger's first fakes
ROUNDS, D_STEPS, N = 50, 5, 512
LR_D, LR_G, SEED = 1.0, 0.15, 0


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -50, 50)))


def feats(x):
    u = (x - 5) / 3
    return np.stack([np.ones_like(u), u, u * u], -1)


def detective(w, x):
    """P(real) for a coin of weight x: logistic regression on (1, u, u^2)."""
    return sigmoid(feats(np.asarray(x, dtype=float)) @ w)


def train():
    rng = np.random.default_rng(SEED)
    b, w = B0, np.zeros(3)
    bs, ws, accs = [b], [w.copy()], [0.5]
    for _ in range(ROUNDS):
        # Detective's turn: learn to tell real (label 1) from fake (label 0).
        for _ in range(D_STEPS):
            real = rng.normal(REAL_MEAN, REAL_SD, N)
            fake = b + FAKE_SD * rng.standard_normal(N)
            fr, ff = feats(real), feats(fake)
            grad = -((1 - sigmoid(fr @ w)) @ fr - sigmoid(ff @ w) @ ff) / N
            w = w - LR_D * grad
        # How often the detective is right right now (fresh coins, half real, half fake).
        real = rng.normal(REAL_MEAN, REAL_SD, N)
        fake = b + FAKE_SD * rng.standard_normal(N)
        acc = (np.mean(detective(w, real) > 0.5) + np.mean(detective(w, fake) < 0.5)) / 2
        # Forger's turn: move the fakes toward what the detective calls "real" (maximize log D(fake)).
        z = rng.standard_normal(N)
        fake = b + FAKE_SD * z
        d = detective(w, fake)
        u = (fake - 5) / 3
        dlogd_dx = (1 - d) * (w[1] + 2 * w[2] * u) / 3
        b = b + LR_G * float(np.mean(dlogd_dx))
        bs.append(b)
        ws.append(w.copy())
        accs.append(float(acc))
    return np.array(bs), np.array(ws), np.array(accs)


BS, WS, ACCS = train()
FINAL_ACC = round(100 * float(np.mean(ACCS[-10:])))
assert ACCS[1] == 1.0  # round 1: every fake caught, every real coin passed
assert abs(BS[-1] - REAL_MEAN) < 0.3 and 45 <= FINAL_ACC <= 55
PEAK = float(BS.max())  # the forger overshoots, then settles (see-saw)

SCRIPT = [
    "Two AIs play a game. One is a forger. One is a detective.",
    f"Real gold coins weigh about {num_to_words(REAL_MEAN)} grams. "
    f"The forger's first fakes weigh about {num_to_words(B0)}.",
    "The detective learns to spot fakes. Easy: one hundred percent right.",
    "The forger learns from every catch, and makes heavier fakes.",
    "Round after round, the detective gets sharper, and the forger gets better.",
    f"After {num_to_words(ROUNDS)} rounds, the fakes weigh just like real coins. "
    f"The detective is down to a coin flip: {num_to_words(FINAL_ACC)} percent.",
    "That's a Gan: a generative adversarial network. Two rivals make each other better.",
    "Scale it up to pictures, and Gans made photo-real faces of people who don't exist.",
    "Level sixteen cleared! What should we learn next? Tell me in the comments.",
]
HOOK, SETUP, DETECT, FORGE, SEESAW, RESULT, NAME, REAL, CTA = range(9)

# ---------------------------------------------------------------- layout

PX0, PX1 = 110, 970  # x axis: 0 .. 11 grams
XMAX = 11.0
PLOT_TOP, PLOT_BOT = 760, 1150
CHAR_Y = 600
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


def gx(x):
    return PX0 + (PX1 - PX0) * x / XMAX


def gy(p):
    return PLOT_BOT - (PLOT_BOT - PLOT_TOP) * p


# ---------------------------------------------------------------- state

def round_at(t):
    """Fractional training round shown at time t (0 = nobody trained yet)."""
    a = word(DETECT, "learns")
    b, c, d = T.start(FORGE), T.start(SEESAW), word(RESULT, "fakes")
    if t < a:
        return 0.0
    if t < b:
        return clamp((t - a) / 0.8)  # detective's first turn
    if t < c:
        return lerp(1, 8, ease_in_out((t - b) / (c - b)))
    if t < d:
        return lerp(8, ROUNDS, ease_in_out((t - c) / (d - c)))
    return float(ROUNDS)


def state(t):
    r = round_at(t)
    i = min(ROUNDS - 1, int(r))
    f = r - i
    b = BS[i] * (1 - f) + BS[i + 1] * f if r >= 1 else B0  # the forger only moves after round 1
    if r < 1:
        b = B0
    w = WS[i] * (1 - f) + WS[i + 1] * f
    acc = ACCS[min(ROUNDS, int(round(r)))] if r >= 1 else None
    return r, b, w, acc


def hud(t):
    _, _, _, acc = state(t)
    if acc is None:
        return "—", MUTED
    r = state(t)[0]
    pct = FINAL_ACC if r >= ROUNDS else round(100 * acc)  # at the end: average of the last 10 rounds
    return f"{pct}%", GREEN if pct > 75 else YELLOW if pct > 60 else PINK


# ---------------------------------------------------------------- drawing

def draw_forger(frame, center, scale=1.0):
    img = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((30, 30, 170, 170), fill=PINK, outline=WHITE, width=5)
    d.rounded_rectangle((36, 70, 164, 108), radius=16, fill=(20, 20, 30))
    for ex in (72, 128):
        d.ellipse((ex - 12, 78, ex + 12, 100), fill=WHITE)
    d.arc((70, 110, 130, 150), 20, 160, fill=(20, 20, 30), width=6)
    d.polygon([(40, 44), (100, 6), (160, 44)], fill=(60, 30, 70))  # beret
    paste_scaled(frame, img, center, scale=scale)


def draw_detective(frame, center, scale=1.0):
    img = Image.new("RGBA", (220, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((30, 40, 160, 170), fill=CYAN, outline=WHITE, width=5)
    for ex in (72, 118):
        d.ellipse((ex - 11, 85, ex + 11, 107), fill=WHITE)
        d.ellipse((ex - 5, 91, ex + 5, 101), fill=(20, 20, 30))
    d.rounded_rectangle((20, 34, 170, 52), radius=8, fill=(90, 70, 40))  # hat brim
    d.rounded_rectangle((50, 4, 140, 44), radius=10, fill=(120, 95, 55))
    d.line([(150, 140), (195, 185)], fill=(160, 120, 70), width=12)  # magnifying glass
    d.ellipse((120, 108, 172, 160), outline=YELLOW, width=7)
    paste_scaled(frame, img, center, scale=scale)


def gauss(x, m, s):
    return np.exp(-0.5 * ((x - m) / s) ** 2)


def draw_plot(frame, t):
    t0 = T.start(SETUP)
    if t < t0 or t >= T.start(NAME):
        return
    a = clamp((t - t0) / 0.4)
    _, b, w, _ = state(t)
    d = ImageDraw.Draw(frame)
    # Axis.
    d.line([(PX0, PLOT_BOT), (PX1, PLOT_BOT)], fill=(120, 132, 180), width=4)
    for g in range(0, 12):
        x = gx(g)
        d.line([(x, PLOT_BOT), (x, PLOT_BOT + 12)], fill=(120, 132, 180), width=3)
        if g % 2 == 0:
            text_center(d, (x, PLOT_BOT + 36), str(g), font("SemiBold", 30), MUTED)
    text_center(d, (540, PLOT_BOT + 80), "coin weight (grams)", font("Bold", 32), (120, 132, 180))
    xs = np.linspace(0, XMAX, 400)
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ld = ImageDraw.Draw(lay)
    # Real coins (gold) and fakes (pink), as bell curves of how often each weight shows up.
    t_real, t_fake = word(SETUP, "real"), word(SETUP, "forger")
    curves = []
    for m, col, ts in ((b, PINK, t_fake), (REAL_MEAN, YELLOW, t_real)):
        if t < ts:
            continue
        g = ease_in_out((t - ts) / 0.5)
        ys = gauss(xs, m, 0.5) * 0.62 * g
        pts = [(gx(x), gy(y)) for x, y in zip(xs, ys)]
        ld.polygon(pts + [(gx(XMAX), PLOT_BOT), (gx(0), PLOT_BOT)], fill=col + (int(100 * a),))
        curves.append((pts, col))
    for pts, col in curves:  # outlines on top, so the real curve stays visible under the fakes
        ld.line(pts, fill=col + (255,), width=5)
    frame.alpha_composite(lay)
    close = abs(b - REAL_MEAN) < 1.6  # labels side by side once the curves overlap
    if t >= t_real:
        pop_in(frame, tag("REAL", (20, 20, 30), YELLOW + (255,), 30, pad=(12, 5)),
               (gx(REAL_MEAN) + (55 if close else 0), gy(0.62) - 30), t, t_real)
    if t >= t_fake:
        pop_in(frame, tag("FAKE", WHITE, PINK + (255,), 30, pad=(12, 5)), (gx(b) - (55 if close else 0), gy(0.62) - 30),
               t, t_fake)
    # The detective's opinion: P(real) for every weight.
    t_d = word(DETECT, "learns")
    if t >= t_d:
        g = ease_in_out((t - t_d) / 0.6)
        ys = detective(w, xs)
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ImageDraw.Draw(lay).line([(gx(x), gy(y * 0.95)) for x, y in zip(xs, ys)], fill=CYAN + (int(255 * g),), width=6)
        frame.alpha_composite(add_glow(lay, radius=6, strength=1.1))
        text_center(d, (PX0 + 4, gy(0.95) - 2), "detective: “real?”", font("Bold", 28), CYAN, anchor="lm")
        text_center(d, (PX0 - 12, gy(0.95)), "100%", font("Bold", 22), MUTED, anchor="rm")
        text_center(d, (PX0 - 12, gy(0.475)), "50%", font("Bold", 22), MUTED, anchor="rm")
        dd = ImageDraw.Draw(frame)
        for yy in (0.95, 0.475):
            dd.line([(PX0, gy(yy)), (PX1, gy(yy))], fill=(60, 66, 104), width=2)


def draw_characters(frame, t):
    if t >= T.start(CTA):
        return
    f_s = ease_back((t - word(HOOK, "forger")) / 0.4) if t >= word(HOOK, "forger") else 0
    d_s = ease_back((t - word(HOOK, "detective")) / 0.4) if t >= word(HOOK, "detective") else 0
    small = t >= T.start(SETUP)
    y = CHAR_Y if small else 760
    sc = 0.7 if small else 1.3
    fx, dx = (230, 850) if small else (300, 780)
    _, b, w, acc = state(t)
    if f_s > 0:
        draw_forger(frame, (fx, y), scale=sc * f_s)
        if not small:
            pop_in(frame, tag("FORGER", WHITE, PINK + (255,), 40), (fx, y + 170), t, word(HOOK, "forger"))
    if d_s > 0:
        draw_detective(frame, (dx, y), scale=sc * d_s)
        if not small:
            pop_in(frame, tag("DETECTIVE", (12, 14, 30), CYAN + (255,), 40), (dx, y + 170), t, word(HOOK, "detective"))
    if small and T.start(SETUP) <= t < T.start(NAME):
        d = ImageDraw.Draw(frame)
        text_center(d, (fx + 90, y - 30), "makes fakes", font("Bold", 28), PINK, anchor="lm")
        text_center(d, (fx + 90, y + 10), f"aims at {b:.1f} g", font("Black", 34), WHITE, anchor="lm")
        text_center(d, (dx - 90, y - 30), "real or fake?", font("Bold", 28), CYAN, anchor="rm")
    if not small and t >= word(HOOK, "game"):
        text_center(ImageDraw.Draw(frame), (540, 760), "VS", font("Black", 70), YELLOW)


def draw_rounds(frame, t):
    if not (word(DETECT, "learns") <= t < T.start(NAME)):
        return
    r, _, _, acc = state(t)
    d = ImageDraw.Draw(frame)
    y = 1290
    pop_in(frame, tag(f"ROUND {int(math.ceil(r - 1e-6))} / {ROUNDS}", WHITE, (36, 40, 80, 255), 36,
                      outline=PURPLE + (255,)), (540, y), t, word(DETECT, "learns"))
    if acc is not None:
        x0, x1 = 250, 830
        yb = y + 62
        text_center(d, (x0 - 14, yb + 7), "right", font("Bold", 28), MUTED, anchor="rm")
        d.rounded_rectangle((x0, yb, x1, yb + 16), radius=8, fill=(60, 66, 104))
        col = GREEN if acc > 0.75 else YELLOW if acc > 0.6 else PINK
        d.rounded_rectangle((x0, yb, x0 + (x1 - x0) * acc, yb + 16), radius=8, fill=col)
        d.line([((x0 + x1) / 2, yb - 8), ((x0 + x1) / 2, yb + 24)], fill=WHITE, width=3)
        text_center(d, (x1 + 14, yb + 7), f"{round(100 * acc)}%", font("Black", 30), col, anchor="lm")
    tc = word(RESULT, "coin", 1) if len([1 for w_, _, _ in T.lines[RESULT]["words"] if w_.lower().startswith("coin")]) > 1 \
        else word(RESULT, "flip")
    if t >= tc:
        pop_in(frame, tag(f"COIN FLIP: {FINAL_ACC}%", (20, 20, 30), YELLOW + (255,), 40), (540, gy(0.95) - 64), t, tc)


def draw_name(frame, t):
    t0 = T.start(NAME)
    if not (t0 <= t < T.start(CTA)):
        return
    a = clamp((t - t0) / 0.3)
    if t < T.start(REAL):
        pop_in(frame, tag("G A N", (20, 20, 30), YELLOW + (255,), 80, pad=(40, 16)), (540, 820), t, t0)
        tg = word(NAME, "generative")
        rows = [("Generative", "generative", 960), ("Adversarial", "adversarial", 1040), ("Network", "network", 1120)]
        for lab, w_, y in rows:
            if t >= word(NAME, w_):
                pop_in(frame, tag(lab, WHITE, (36, 40, 80, 255), 44, outline=PURPLE + (255,)), (540, y), t, word(NAME, w_))
        if t >= word(NAME, "rivals"):
            pop_in(frame, tag("rivals make each other better", (12, 28, 18), GREEN + (255,), 38), (540, 1250), t,
                   word(NAME, "rivals"))
        _ = tg
        return
    frame.alpha_composite(Image.new("RGBA", (W, H), (5, 6, 16, int(150 * a))))
    tr = T.start(REAL)
    pop_in(frame, tag("+ BIG NETWORKS + PICTURES", WHITE, PURPLE + (255,), 44), (540, 820), t, word(REAL, "pictures"))
    if t >= word(REAL, "faces"):
        pop_in(frame, tag("PHOTO-REAL FACES", (20, 20, 30), YELLOW + (255,), 52), (540, 960), t, word(REAL, "faces"))
        pop_in(frame, tag("of people who don't exist", WHITE, (36, 40, 80, 255), 38), (540, 1060), t,
               word(REAL, "exist"))
    _ = tr


def draw_overlays(frame, t):
    draw_hud_pills(frame, 16, *hud(t), label="RIGHT")
    frame.alpha_composite(TITLE_IMG, (0, 190))
    banners = [
        (T.start(HOOK), T.start(SETUP), "TWO AIs, ONE GAME", PURPLE, WHITE),
        (T.start(SETUP), T.start(DETECT), "REAL COINS vs FAKES", YELLOW, (20, 20, 30)),
        (T.start(DETECT), T.start(FORGE), "DETECTIVE'S TURN", CYAN, (12, 14, 30)),
        (T.start(FORGE), T.start(SEESAW), "FORGER'S TURN", PINK, WHITE),
        (T.start(SEESAW), T.start(RESULT), "TAKE TURNS, BOTH IMPROVE", PURPLE, WHITE),
        (T.start(RESULT), T.start(NAME), "FAKES = REAL → DETECTIVE GUESSES", GREEN, (12, 28, 18)),
        (T.start(NAME), T.start(REAL), "THAT'S A GAN", YELLOW, (20, 20, 30)),
        (T.start(REAL), T.start(CTA), "SAME GAME, BIGGER PICTURES", PURPLE, WHITE),
    ]
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 32), (W / 2, TOP_TAG_Y), t, a)
    draw_level_cleared(frame, t, T.start(CTA), 16, "YOU DECIDE!", word(CTA, "what"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    draw_plot(frame, t)
    draw_characters(frame, t)
    draw_rounds(frame, t)
    draw_name(frame, t)
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
    mix(out, sfx_boing(0.45, 0.14), word(HOOK, "forger"))
    mix(out, sfx_boing(0.45, 0.14), word(HOOK, "detective"))
    mix(out, sfx_blip(1320, 0.15, 0.12), word(SETUP, "real"))
    mix(out, sfx_buzz(0.18, 120, 0.12), word(SETUP, "forger"))
    mix(out, sfx_blip(1568, 0.3, 0.16), word(DETECT, "hundred"))
    # A tick each training round.
    prev = 0
    for k in range(int(T.start(FORGE) * FPS), int(word(RESULT, "fakes") * FPS) + 1):
        r = int(round_at(k / FPS))
        if r > prev:
            mix(out, sfx_blip(500 + 12 * r, 0.04, 0.06), k / FPS)
            prev = r
    mix(out, sfx_blip(1047, 0.25, 0.14), word(RESULT, "flip"))
    for w_ in ("generative", "adversarial", "network"):
        mix(out, sfx_blip(880, 0.1, 0.12), word(NAME, w_))
    mix(out, sfx_whoosh(0.7, 0.14, seed=5), word(REAL, "pictures"))
    mix(out, sfx_blip(1568, 0.3, 0.16), word(REAL, "faces"))
    mix(out, sfx_levelup(0.24), T.start(CTA))
    return out


# ---------------------------------------------------------------- main

def setup():
    global T, CHUNKS, STATIC, TITLE_IMG, CONFETTI, DURATION
    voice, lines = voiceover(SCRIPT)
    T = Timeline(lines)
    CHUNKS = caption_chunks(lines, max_words=2, max_chars=12, max_px=800)
    STATIC = build_plain()
    TITLE_IMG = build_title("GANs", "FORGER vs DETECTIVE  ·  ML AS A GAME  ·  EP 16")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    t = T.start(SEESAW) + 0.8
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "FORGER vs DETECTIVE", font("Black", 84), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "GANs, explained", font("Black", 56), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s")
    print("  round: fake weight / detective right")
    for r in (0, 1, 2, 4, 6, 8, 12, 16, 20, 25, 30, 40, 50):
        print(f"    {r:2d}: {BS[r]:.2f} g / {100 * ACCS[r]:.0f}%")
    print(f"  peak fake weight {PEAK:.2f} g; last-10-round accuracy {FINAL_ACC}%")
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
    out = os.path.join(HERE, "gans_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
