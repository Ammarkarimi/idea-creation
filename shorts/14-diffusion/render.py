"""ML as a Game, Level 14: Diffusion (YouTube Short, 1080x1920).

    python3 render.py                 # full render -> diffusion_short.mp4 + thumbnail.png
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
    encode, font, lerp, mix, music_bed, num_to_words, paste_scaled, pop_in, sfx_blip, sfx_boing, sfx_levelup,
    sfx_whoosh, tag, text_center, voiceover, write_wav,
)

# ---------------------------------------------------------------- tiny pixel-art training set (16x16 RGB, values in [-1, 1])

S, UP = 16, 8
BG_PIX = (20, 22, 48)


def _mask(draw_fn):
    im = Image.new("L", (S * UP, S * UP), 0)
    draw_fn(ImageDraw.Draw(im))
    return np.asarray(im.resize((S, S), Image.BOX)).astype(float) / 255 > 0.45


def _heart(cx, cy, r):
    def f(d):
        pts = []
        for k in range(200):
            t = 2 * math.pi * k / 200
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append(((cx + x * r / 16) * UP, (cy - y * r / 16) * UP))
        d.polygon(pts, fill=255)
    return f


def _star(cx, cy, r):
    def f(d):
        pts = []
        for k in range(10):
            rr = r if k % 2 == 0 else r * 0.45
            a = -math.pi / 2 + k * math.pi / 5
            pts.append(((cx + rr * math.cos(a)) * UP, (cy + rr * math.sin(a)) * UP))
        d.polygon(pts, fill=255)
    return f


def _face(cx, cy, r):
    def f(d):
        d.ellipse(((cx - r) * UP, (cy - r) * UP, (cx + r) * UP, (cy + r) * UP), fill=255)
        for ex in (-0.35, 0.35):
            d.ellipse(((cx + ex * r - 0.9) * UP, (cy - 0.3 * r - 1.1) * UP, (cx + ex * r + 0.9) * UP,
                       (cy - 0.3 * r + 1.1) * UP), fill=0)
        d.arc(((cx - 0.55 * r) * UP, (cy - 0.35 * r) * UP, (cx + 0.55 * r) * UP, (cy + 0.55 * r) * UP), 20, 160,
              fill=0, width=int(1.3 * UP))
    return f


def _img(draw_fn, col):
    a = np.zeros((S, S, 3))
    a[:] = BG_PIX
    a[_mask(draw_fn)] = col
    return a / 127.5 - 1


DATA = [
    ("heart", _img(_heart(8, 7.9, 6.8), PINK)), ("heart", _img(_heart(8, 7.9, 6.8), (240, 50, 50))),
    ("heart", _img(_heart(8, 8.5, 4.6), PURPLE)), ("heart", _img(_heart(8, 8.5, 4.6), ORANGE)),
    ("star", _img(_star(8, 8.6, 7.5), YELLOW)), ("star", _img(_star(8, 8.6, 5), CYAN)),
    ("smiley", _img(_face(8, 8, 7), YELLOW)), ("smiley", _img(_face(8, 8, 7), GREEN)),
]
X = np.stack([d[1] for d in DATA]).reshape(len(DATA), -1)

# ---------------------------------------------------------------- real diffusion math (cosine schedule, DDIM sampling)

T_MAX, STEPS = 1000, 20


def abar(t, s=0.008):
    """Fraction of the original picture's signal (variance) left at noise level t."""
    return math.cos((t / T_MAX + s) / (1 + s) * math.pi / 2) ** 2 / math.cos(s / (1 + s) * math.pi / 2) ** 2


def noise_pct(ab):
    """How much of the pixel values is noise: the noise's weight sqrt(1 - abar), as a %."""
    return round(100 * math.sqrt(1 - ab))


def denoise(x, ab, idx):
    """The ideal denoiser for this tiny dataset: best guess of the clean picture (posterior mean)."""
    xs = X[idx]
    lw = -((x[None] - math.sqrt(ab) * xs) ** 2).sum(1) / (2 * (1 - ab))
    w = np.exp(lw - lw.max())
    return (w / w.sum()) @ xs


def guess_noise(x, ab, idx):
    return (x - math.sqrt(ab) * denoise(x, ab, idx)) / math.sqrt(1 - ab)


def sample(seed, cls=None):
    """Start from pure static; each step removes part of the guessed noise (DDIM, 20 steps)."""
    idx = [i for i, d in enumerate(DATA) if cls is None or d[0] == cls]
    x = np.random.default_rng(seed).standard_normal(X.shape[1])
    ts = np.linspace(T_MAX, 0, STEPS + 1).round().astype(int)
    traj, pcts = [x], [noise_pct(abar(ts[0]))]
    for a, b in zip(ts[:-1], ts[1:]):
        aa, ab_ = abar(a), abar(b)
        x0 = denoise(x, aa, idx)
        eps = (x - math.sqrt(aa) * x0) / math.sqrt(1 - aa)
        x = math.sqrt(ab_) * x0 + math.sqrt(max(0.0, 1 - ab_)) * eps
        traj.append(x)
        pcts.append(noise_pct(ab_))
    k = int(np.argmin(((X - x) ** 2).mean(1)))
    return dict(traj=traj, pcts=pcts, result=k, mse=float(((X[k] - x) ** 2).mean()))


SEED_MAIN, SEED_HOOK, SEEDS_ROW = 3, 6, (3, 2, 7)
RUN_FREE = sample(SEED_MAIN)  # no prompt
RUN_HEART = sample(SEED_MAIN, "heart")  # same static, prompt "a heart"
RUN_HOOK = sample(SEED_HOOK, "heart")
RUNS_ROW = [sample(s, "heart") for s in SEEDS_ROW]
FREE_NAME = DATA[RUN_FREE["result"]][0]
assert RUN_FREE["pcts"][0] == 100 and RUN_FREE["pcts"][-1] == 0
assert FREE_NAME != "heart" and DATA[RUN_HEART["result"]][0] == "heart"
assert all(DATA[r["result"]][0] == "heart" for r in RUNS_ROW + [RUN_HOOK])
assert len({r["result"] for r in RUNS_ROW}) == 3  # different static -> different heart
assert max(r["mse"] for r in [RUN_FREE, RUN_HEART, RUN_HOOK] + RUNS_ROW) < 1e-6

# Training demo: the star, noised to 0 / 25 / 50 / 75 / 100 %.
TRAIN_IMG = X[4]
TRAIN_NOISE = np.random.default_rng(11).standard_normal(X.shape[1])
PCTS = [0, 25, 50, 75, 100]


def t_for_pct(p):
    lo, hi = 0.0, float(T_MAX)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 100 * math.sqrt(1 - abar(mid)) < p else (lo, mid)
    return hi


def noised(ab):
    return math.sqrt(ab) * TRAIN_IMG + math.sqrt(1 - ab) * TRAIN_NOISE


AB_HALF = abar(t_for_pct(50))
NOISY_HALF = noised(AB_HALF)
GUESS_HALF = guess_noise(NOISY_HALF, AB_HALF, list(range(len(DATA))))
assert noise_pct(AB_HALF) == 50

SCRIPT = [
    "Every AI image starts as pure static. Watch.",
    "Training: take a picture, and add noise, step by step, until it's just static.",
    "Then a model learns one skill: look at a noisy picture, and guess the noise.",
    "To make a new picture, start from random static, subtract a bit of the guessed noise, and repeat.",
    f"{num_to_words(STEPS).capitalize()} small clean-ups, and a picture appears!",
    "Add a prompt, like a heart, and every guess steers toward hearts.",
    "Start from different static, and you get a different heart.",
    "That's diffusion: learn to remove noise, then create from noise.",
    "Real image generators do this with huge networks, trained on millions of pictures.",
    "Level fourteen cleared! What should we learn next? Tell me in the comments.",
]
HOOK, FORWARD, GUESS, REVERSE, APPEAR, PROMPT, SEEDSC, SUMMARY, REAL, CTA = range(10)

# ---------------------------------------------------------------- layout

CANVAS = 544
CANVAS_C = (540, 840)
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


def pix_image(x, size, border=PURPLE):
    a = ((np.asarray(x).reshape(S, S, 3) + 1) * 127.5).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(a, "RGB").resize((size, size), Image.NEAREST).convert("RGBA")
    out = Image.new("RGBA", (size + 16, size + 16), (0, 0, 0, 0))
    d = ImageDraw.Draw(out)
    d.rounded_rectangle((0, 0, size + 15, size + 15), radius=18, fill=border + (255,))
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, size - 1, size - 1), radius=12, fill=255)
    out.paste(img, (8, 8), m)
    return out


# ---------------------------------------------------------------- timing / state

def run_window(i):
    """When each sampling run plays (start, end)."""
    if i == "hook":
        return T.start(HOOK) + 0.6, T.end(HOOK) - 0.4
    if i == "free":
        return word(REVERSE, "start"), word(APPEAR, "picture")
    if i == "heart":
        return word(PROMPT, "heart") + 0.2, T.end(PROMPT)
    return word(SEEDSC, "different") + 0.15 * i, word(SEEDSC, "heart") + 0.15 * i


def run_state(run, window, t):
    """Interpolated image and noise % of a sampling run at time t."""
    a, b = window
    f = clamp((t - a) / (b - a)) * STEPS
    k = min(STEPS - 1, int(f))
    g = f - k
    x = run["traj"][k] * (1 - g) + run["traj"][k + 1] * g
    pct = run["pcts"][k] * (1 - g) + run["pcts"][k + 1] * g
    return x, round(pct), min(STEPS, int(f))


def forward_ab(t):
    a, b = word(FORWARD, "add"), word(FORWARD, "static")
    p = clamp((t - a) / (b - a))
    return abar(t_for_pct(100 * p)) if p > 0 else 1.0


def hud(t):
    if t < T.start(FORWARD):
        _, p, _ = run_state(RUN_HOOK, run_window("hook"), t)
        return f"{p}%", PINK if p > 50 else GREEN
    if t < T.start(GUESS):
        return f"{noise_pct(forward_ab(t))}%", PINK
    if t < T.start(REVERSE):
        return "50%", PINK
    if t < T.start(PROMPT):
        _, p, _ = run_state(RUN_FREE, run_window("free"), t)
        return f"{p}%", PINK if p > 50 else GREEN
    if t < T.start(SEEDSC):
        _, p, _ = run_state(RUN_HEART, run_window("heart"), t)
        return f"{p}%", PINK if p > 50 else GREEN
    if t < T.start(SUMMARY):
        _, p, _ = run_state(RUNS_ROW[0], run_window(0), t)
        return f"{p}%", PINK if p > 50 else GREEN
    return "0%", GREEN


# ---------------------------------------------------------------- drawing

def draw_canvas(frame, t):
    """The big canvas: hook run, training noise, free run, prompted run, summary."""
    if t >= T.start(CTA) or T.start(GUESS) <= t < T.start(REVERSE) or T.start(SEEDSC) <= t < T.start(SUMMARY):
        return
    if t < T.start(FORWARD):
        x, _, _ = run_state(RUN_HOOK, run_window("hook"), t)
        border = PINK
    elif t < T.start(GUESS):
        x, border = noised(forward_ab(t)), CYAN
    elif t < T.start(PROMPT):
        x, _, _ = run_state(RUN_FREE, run_window("free"), t)
        border = PURPLE
    else:
        x, _, _ = run_state(RUN_HEART, run_window("heart"), t)
        border = PINK
    t0 = T.start(HOOK)
    paste_scaled(frame, pix_image(x, CANVAS, border), CANVAS_C, scale=ease_back((t - t0) / 0.4))


def draw_forward_strip(frame, t):
    if not (T.start(FORWARD) <= t < T.start(GUESS)):
        return
    d = ImageDraw.Draw(frame)
    xs = [180, 360, 540, 720, 900]
    a, b = word(FORWARD, "add"), word(FORWARD, "static")
    for k, (xc, p) in enumerate(zip(xs, PCTS)):
        tk = lerp(a, b, k / 4) if k else word(FORWARD, "picture")
        if t < tk:
            continue
        pop_in(frame, pix_image(noised(abar(t_for_pct(p))), 130, CYAN), (xc, 1215), t, tk)
        text_center(d, (xc, 1305), f"{p}% noise", font("Bold", 28), WHITE if p in (0, 100) else (200, 205, 230))
        if k:
            text_center(d, (xc - 90, 1215), "→", font("Black", 34), (150, 160, 200))


def draw_guess(frame, t):
    t0 = T.start(GUESS)
    if not (t0 <= t < T.start(REVERSE)):
        return
    d = ImageDraw.Draw(frame)
    pop_in(frame, pix_image(NOISY_HALF, 300, CYAN), (230, 820), t, t0)
    text_center(d, (230, 1010), "noisy picture (50%)", font("Bold", 30), WHITE)
    t_m = word(GUESS, "model")
    if t >= t_m:
        pop_in(frame, tag("MODEL", (20, 20, 30), YELLOW + (255,), 40, pad=(16, 12)), (540, 820), t, t_m)
        text_center(d, (540, 900), "→", font("Black", 50), (150, 160, 200))
    t_g = word(GUESS, "guess")
    if t >= t_g:
        pop_in(frame, pix_image(GUESS_HALF, 300, ORANGE), (850, 820), t, t_g)
        text_center(d, (850, 1010), "its noise guess", font("Bold", 30), WHITE)
        pop_in(frame, tag("remove the guessed noise → clean picture", WHITE, (36, 40, 80, 255), 34,
                          outline=ORANGE + (255,)), (540, 1150), t, t_g + 0.25)


def draw_steps(frame, t):
    """Step counter under the canvas while a sampling run plays."""
    if T.start(REVERSE) <= t < T.start(PROMPT):
        run, win, col = RUN_FREE, run_window("free"), PURPLE
    elif T.start(PROMPT) <= t < T.start(SEEDSC):
        run, win, col = RUN_HEART, run_window("heart"), PINK
    else:
        return
    if t < win[0]:
        return
    _, _, k = run_state(run, win, t)
    d = ImageDraw.Draw(frame)
    pop_in(frame, tag(f"CLEAN-UP STEP {k} / {STEPS}", WHITE, (36, 40, 80, 255), 36, outline=col + (255,)),
           (540, 1180), t, win[0])
    x0, x1, y = 300, 780, 1240
    d.rounded_rectangle((x0, y, x1, y + 12), radius=6, fill=(60, 66, 104))
    if k:
        d.rounded_rectangle((x0, y, x0 + (x1 - x0) * k / STEPS, y + 12), radius=6, fill=col)


def draw_prompt_and_results(frame, t):
    if T.start(APPEAR) <= t < T.start(PROMPT):
        tp = word(APPEAR, "picture")
        if t >= tp:
            pop_in(frame, tag(f"no prompt → a {FREE_NAME}", (20, 20, 30), YELLOW + (255,), 40), (540, 1320), t, tp)
    if T.start(PROMPT) <= t < T.start(SEEDSC):
        tp = word(PROMPT, "prompt")
        if t >= tp:
            s = "prompt:  a heart" if t >= word(PROMPT, "heart") else "prompt:  _"
            pop_in(frame, tag(s, WHITE, (28, 16, 56, 255), 40, outline=PINK + (255,)), (540, 1320), t, tp)


def draw_seeds(frame, t):
    t0 = T.start(SEEDSC)
    if not (t0 <= t < T.start(SUMMARY)):
        return
    d = ImageDraw.Draw(frame)
    pop_in(frame, tag("prompt:  a heart", WHITE, (28, 16, 56, 255), 36, outline=PINK + (255,)), (540, 600), t, t0)
    for i, (xc, run) in enumerate(zip((210, 540, 870), RUNS_ROW)):
        x, _, _ = run_state(run, run_window(i), t)
        paste_scaled(frame, pix_image(x, 270, PINK), (xc, 850), scale=ease_back((t - t0 - 0.1 * i) / 0.35))
        text_center(d, (xc, 1020), f"static #{i + 1}", font("Bold", 30), (200, 205, 230))


def draw_summary(frame, t):
    t0 = T.start(SUMMARY)
    if not (t0 <= t < T.start(CTA)):
        return
    t_l = word(SUMMARY, "learn")
    if t >= t_l:
        pop_in(frame, tag("TRAINING: picture → noise (learn to guess it)", (12, 14, 30), CYAN + (255,), 32),
               (540, 1210), t, t_l)
    t_c = word(SUMMARY, "create")
    if t >= t_c:
        pop_in(frame, tag("CREATING: noise → picture", (12, 28, 18), GREEN + (255,), 40), (540, 1300), t, t_c)


def draw_real(frame, t):
    t0 = T.start(REAL)
    if not (t0 <= t < T.start(CTA)):
        return
    frame.alpha_composite(Image.new("RGBA", (W, H), (5, 6, 16, int(205 * clamp((t - t0) / 0.3)))))
    items = [("HUGE NETWORKS", "huge", PURPLE, WHITE, 720), ("MILLIONS OF PICTURES", "millions", CYAN, (12, 14, 30), 880),
             ("SAME IDEA: NOISE → PICTURE", "pictures", GREEN, (12, 28, 18), 1040)]
    for lab, w_, bg, fg, y in items:
        if t >= word(REAL, w_):
            pop_in(frame, tag(lab, fg, bg + (255,), 46), (540, y), t, word(REAL, w_))


def draw_overlays(frame, t):
    draw_hud_pills(frame, 14, *hud(t), label="NOISE")
    frame.alpha_composite(TITLE_IMG, (0, 190))
    banners = [
        (T.start(HOOK), T.start(FORWARD), "HOW AI DRAWS PICTURES", PURPLE, WHITE),
        (T.start(FORWARD), T.start(GUESS), "TRAINING · STEP 1: ADD NOISE", CYAN, (12, 14, 30)),
        (T.start(GUESS), T.start(REVERSE), "TRAINING · STEP 2: GUESS THE NOISE", ORANGE, (20, 20, 30)),
        (T.start(REVERSE), T.start(APPEAR), "CREATING: REMOVE NOISE, STEP BY STEP", PURPLE, WHITE),
        (T.start(APPEAR), T.start(PROMPT), f"{STEPS} CLEAN-UPS → A PICTURE", GREEN, (12, 28, 18)),
        (T.start(PROMPT), T.start(SEEDSC), "A PROMPT STEERS THE GUESSES", PINK, WHITE),
        (T.start(SEEDSC), T.start(SUMMARY), "DIFFERENT STATIC → DIFFERENT HEART", YELLOW, (20, 20, 30)),
        (T.start(SUMMARY), T.start(REAL), "THAT'S DIFFUSION", GREEN, (12, 28, 18)),
        (T.start(REAL), T.start(CTA), "SAME IDEA, HUGE SCALE", PURPLE, WHITE),
    ]
    for a, b, text, bg, fg in banners:
        if a <= t < b:
            pop_in(frame, tag(text, fg, bg + (255,), 32), (W / 2, TOP_TAG_Y), t, a)
    draw_level_cleared(frame, t, T.start(CTA), 14, "YOU DECIDE!", word(CTA, "what"), CONFETTI)


def render_rgba(i, captions=True):
    t = i / FPS
    frame = STATIC.copy()
    draw_canvas(frame, t)
    draw_forward_strip(frame, t)
    draw_guess(frame, t)
    draw_steps(frame, t)
    draw_prompt_and_results(frame, t)
    draw_seeds(frame, t)
    draw_summary(frame, t)
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
    for name in ("hook", "free", "heart"):
        a, b = run_window(name)
        for k in range(1, STEPS + 1, 2):
            mix(out, sfx_blip(500 + 25 * k, 0.04, 0.07), lerp(a, b, k / STEPS))
        mix(out, sfx_blip(1568, 0.3, 0.16), b)
    mix(out, sfx_whoosh(1.2, 0.16, seed=3), word(FORWARD, "add"))
    mix(out, sfx_boing(0.45, 0.14), word(GUESS, "model"))
    mix(out, sfx_blip(1175, 0.15, 0.12), word(GUESS, "guess"))
    mix(out, sfx_blip(1320, 0.2, 0.14), word(PROMPT, "heart"))
    for i in range(3):
        mix(out, sfx_blip(1047 + 260 * i, 0.2, 0.13), run_window(i)[1])
    mix(out, sfx_whoosh(0.6, 0.14, seed=7), word(SUMMARY, "learn"))
    mix(out, sfx_blip(1568, 0.25, 0.14), word(SUMMARY, "create"))
    for w_ in ("huge", "millions", "pictures"):
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
    TITLE_IMG = build_title("DIFFUSION", "ML AS A GAME  ·  EP 14")
    CONFETTI = build_confetti()
    DURATION = len(voice) / SR
    return voice


def thumbnail(path):
    a, b = run_window("heart")
    t = lerp(a, b, 0.8)  # a heart coming out of the static
    img = render_rgba(int(t * FPS), captions=False)
    d = ImageDraw.Draw(img)
    img.alpha_composite(Image.new("RGBA", (W, 330), (5, 6, 16, 225)), (0, 1440))
    text_center(d, (W / 2, 1540), "HOW AI DRAWS", font("Black", 84), YELLOW, stroke=8)
    text_center(d, (W / 2, 1675), "Diffusion, explained", font("Black", 56), WHITE, stroke=6)
    img.convert("RGB").save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma-separated seconds; writes preview PNGs only")
    ap.add_argument("--thumbnail", action="store_true", help="only regenerate thumbnail.png")
    args = ap.parse_args()
    voice = setup()
    print(f"duration {DURATION:.2f}s")
    names = [f"{n}#{i}" for i, (n, _) in enumerate(DATA)]
    print(f"  noise % per step: {RUN_FREE['pcts']}")
    print(f"  seed {SEED_MAIN}: no prompt -> {names[RUN_FREE['result']]}, prompt heart -> {names[RUN_HEART['result']]}")
    print(f"  heart seeds {SEEDS_ROW} -> {[names[r['result']] for r in RUNS_ROW]}; hook -> {names[RUN_HOOK['result']]}")
    print(f"  max final MSE to nearest training image: {max(r['mse'] for r in RUNS_ROW + [RUN_FREE, RUN_HEART]):.1e}")
    print(f"  guessed vs true noise at 50%: corr {np.corrcoef(GUESS_HALF, TRAIN_NOISE)[0, 1]:.3f}")
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
    out = os.path.join(HERE, "diffusion_short.mp4")
    encode(render_frame, int(DURATION * FPS), wav, out)
    os.remove(wav)
    thumbnail(os.path.join(HERE, "thumbnail.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
