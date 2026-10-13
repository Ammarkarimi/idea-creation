"""Shared helpers for the "ML as a Game" YouTube Shorts series.

Each episode lives in its own folder and imports this module for:
voiceover (Piper TTS), caption timing, drawing primitives, procedural
music/SFX, and ffmpeg encoding.
"""

import hashlib
import json
import math
import os
import re
import subprocess
import wave
from functools import lru_cache
from multiprocessing import get_context

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1080, 1920, 30
SR = 44100  # final audio sample rate

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, ".cache")
VOICE = os.path.join(CACHE, "en_US-ryan-high.onnx")
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx"

# Palette: dark neon "game" look.
BG_TOP = (9, 12, 28)
BG_BOTTOM = (22, 10, 40)
GRID = (40, 48, 82)
AXIS = (120, 132, 180)
WHITE = (245, 247, 255)
MUTED = (150, 160, 200)
CYAN = (34, 211, 238)
PINK = (255, 77, 141)
YELLOW = (255, 225, 77)
GREEN = (74, 222, 128)
PURPLE = (168, 85, 247)
ORANGE = (251, 146, 60)

FONT_DIRS = ["/usr/share/fonts/opentype/inter", "/usr/share/fonts/truetype/dejavu"]


@lru_cache(maxsize=None)
def font(weight="Black", size=64, display=True):
    """Inter (Display) at the given weight; falls back to DejaVu Sans Bold."""
    name = f"{'InterDisplay' if display else 'Inter'}-{weight}.otf"
    for d in FONT_DIRS:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.truetype(os.path.join(FONT_DIRS[1], "DejaVuSans-Bold.ttf"), size)


# ---------------------------------------------------------------- easing

def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease_out(p):
    p = clamp(p)
    return 1 - (1 - p) ** 3


def ease_in_out(p):
    p = clamp(p)
    return 3 * p * p - 2 * p * p * p


def ease_back(p, s=1.8):
    """Overshoots slightly, then settles: good for 'pop' animations."""
    p = clamp(p) - 1
    return p * p * ((s + 1) * p + s) + 1


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    t = clamp(t)
    return tuple(int(round(lerp(a, b, t))) for a, b in zip(c1, c2))


# ---------------------------------------------------------------- voiceover

def _load_voice():
    from piper import PiperVoice

    if not os.path.exists(VOICE):
        os.makedirs(CACHE, exist_ok=True)
        for ext in ("", ".json"):
            subprocess.run(["curl", "-sSLf", "-o", VOICE + ext, VOICE_URL + ext], check=True)
    return PiperVoice.load(VOICE)


def _resample(x, sr_in, sr_out=SR):
    n_out = int(round(len(x) * sr_out / sr_in))
    return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)


def _trim(x, thresh=0.015):
    idx = np.where(np.abs(x) > thresh)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] - int(0.02 * SR))
    b = min(len(x), idx[-1] + int(0.06 * SR))
    return x[a:b]


def _word_weights(words):
    """Rough speaking-time weight per word: letters + a pause after punctuation."""
    out = []
    for w in words:
        letters = len(re.sub(r"[^A-Za-z0-9]", "", w)) or 1
        pause = 5 if w[-1] in ".?!:" else 3 if w[-1] == "," else 0
        out.append((letters + 2, pause))
    return out


def voiceover(lines, lead_in=0.35, gap=0.28, tail=1.6, length_scale=1.0):
    """Synthesize each script line and lay them out on one timeline.

    Returns (audio float32 mono @SR, timeline) where timeline is a list of
    {"text", "start", "end", "words": [(word, start, end), ...]}.
    """
    # Piper adds random noise, so every synthesis differs slightly. Cache per script so
    # re-renders (and preview vs. final) share exactly the same take and timings.
    key = hashlib.sha1(json.dumps([lines, lead_in, gap, tail, length_scale, os.path.basename(VOICE)])
                       .encode()).hexdigest()[:16]
    cached = os.path.join(CACHE, f"vo-{key}.npz")
    if os.path.exists(cached):
        z = np.load(cached)
        return z["audio"], json.loads(str(z["timeline"]))

    from piper.config import SynthesisConfig

    voice = _load_voice()
    cfg = SynthesisConfig(length_scale=length_scale)
    clips, timeline, t = [], [], lead_in
    for text in lines:
        chunks = list(voice.synthesize(text, syn_config=cfg))
        raw = np.concatenate([c.audio_float_array for c in chunks])
        clip = _trim(_resample(raw, chunks[0].sample_rate))
        dur = len(clip) / SR
        words = text.split()
        weights = _word_weights(words)
        total = sum(a + b for a, b in weights)
        wt, cur = [], t
        for w, (a, b) in zip(words, weights):
            d = dur * a / total
            wt.append((w, cur, cur + d))
            cur += d + dur * b / total
        timeline.append({"text": text, "start": t, "end": t + dur, "words": wt})
        clips.append((t, clip))
        t += dur + gap
    total_dur = t - gap + tail
    audio = np.zeros(int(total_dur * SR) + 1, dtype=np.float32)
    for start, clip in clips:
        i = int(start * SR)
        audio[i:i + len(clip)] += clip
    os.makedirs(CACHE, exist_ok=True)
    np.savez(cached, audio=audio, timeline=json.dumps(timeline))
    return audio, timeline


def caption_chunks(timeline, max_words=3, max_chars=15):
    """Group words into short caption chunks (the fast 'word-pop' style)."""
    chunks = []
    for li, line in enumerate(timeline):
        cur = []
        for w in line["words"]:
            cur.append(w)
            text = " ".join(x[0] for x in cur)
            if len(cur) >= max_words or len(text) >= max_chars or w[0][-1] in ".,:?!":
                chunks.append({"line": li, "words": cur})
                cur = []
        if cur:
            chunks.append({"line": li, "words": cur})
    for c in chunks:
        c["start"] = c["words"][0][1]
    for i, c in enumerate(chunks):
        last = c["words"][-1][2]
        nxt = chunks[i + 1]["start"] if i + 1 < len(chunks) else last + 1.0
        same_line = i + 1 < len(chunks) and chunks[i + 1]["line"] == c["line"]
        c["end"] = nxt if same_line else min(nxt, last + 0.5)
    return chunks


# ---------------------------------------------------------------- drawing

def vertical_gradient(size, top, bottom):
    w, h = size
    t = np.linspace(0, 1, h)[:, None, None]
    arr = (np.array(top)[None, None, :] * (1 - t) + np.array(bottom)[None, None, :] * t)
    arr = np.repeat(arr, w, axis=1).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def rounded_pill(draw, box, fill, outline=None, width=0, radius=None):
    x0, y0, x1, y1 = box
    r = radius if radius is not None else (y1 - y0) // 2
    draw.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def text_center(draw, xy, text, fnt, fill, stroke=0, stroke_fill=(0, 0, 0), anchor="mm"):
    draw.text(xy, text, font=fnt, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=stroke_fill)


def zigzag(p0, p1, coils=7, amp=12):
    """Points of a spring drawn between p0 and p1."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1:
        return [p0, p1]
    ux, uy = dx / length, dy / length
    nx, ny = -uy, ux
    lead = min(10 * amp / 12, length * 0.15)
    pts = [(x0, y0), (x0 + ux * lead, y0 + uy * lead)]
    n = coils * 2
    body = length - 2 * lead
    for i in range(1, n):
        s = lead + body * i / n
        side = amp if i % 2 else -amp
        pts.append((x0 + ux * s + nx * side, y0 + uy * s + ny * side))
    pts.append((x1 - ux * lead, y1 - uy * lead))
    pts.append((x1, y1))
    return pts


def dashed_line(draw, p0, p1, fill, width=4, dash=18, gap=12):
    (x0, y0), (x1, y1) = p0, p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 1:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    s = 0
    while s < length:
        e = min(s + dash, length)
        draw.line([(x0 + ux * s, y0 + uy * s), (x0 + ux * e, y0 + uy * e)], fill=fill, width=width)
        s += dash + gap


def add_glow(layer, radius=14, strength=1.6):
    """Neon glow: blurred copy of an RGBA layer composited underneath it."""
    glow = layer.filter(ImageFilter.GaussianBlur(radius))
    if strength != 1:
        a = np.asarray(glow).copy()
        a[..., 3] = np.clip(a[..., 3].astype(np.float32) * strength, 0, 255).astype(np.uint8)
        glow = Image.fromarray(a, "RGBA")
    out = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    out.alpha_composite(glow)
    out.alpha_composite(layer)
    return out


def caption_image(words, active, font_size=92, max_width=840):
    """Render a caption chunk; the currently spoken word is highlighted."""
    fnt = font("Black", font_size)
    space = fnt.getlength(" ")
    widths = [fnt.getlength(w) for w in words]
    # Greedy wrap into at most two lines.
    lines, cur, cur_w = [], [], 0
    for i, wd in enumerate(widths):
        add = wd if not cur else cur_w + space + wd
        if cur and add > max_width:
            lines.append(cur)
            cur, cur_w = [i], wd
        else:
            cur.append(i)
            cur_w = add
    lines.append(cur)
    line_h = int(font_size * 1.18)
    pad = 24
    img = Image.new("RGBA", (max_width + 2 * pad, line_h * len(lines) + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for li, idxs in enumerate(lines):
        lw = sum(widths[i] for i in idxs) + space * (len(idxs) - 1)
        x = (img.width - lw) / 2
        y = pad + li * line_h + line_h / 2
        for i in idxs:
            on = i == active
            # Soft shadow, then outlined word.
            d.text((x + 4, y + 8), words[i], font=fnt, fill=(0, 0, 0, 150), anchor="lm")
            d.text((x, y - (6 if on else 0)), words[i], font=fnt, fill=YELLOW if on else WHITE,
                   anchor="lm", stroke_width=9, stroke_fill=(0, 0, 0))
            x += widths[i] + space
    return img


def paste_scaled(base, img, center, scale=1.0, alpha=1.0):
    """Alpha-composite img onto base, centred at `center`, scaled and faded."""
    if scale <= 0.01 or alpha <= 0.01:
        return
    if abs(scale - 1) > 1e-3:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
    if alpha < 0.999:
        a = np.asarray(img).copy()
        a[..., 3] = (a[..., 3].astype(np.float32) * alpha).astype(np.uint8)
        img = Image.fromarray(a, "RGBA")
    x = max(0, int(center[0] - img.width / 2))
    y = max(0, int(center[1] - img.height / 2))
    base.alpha_composite(img, (x, y))


# ---------------------------------------------------------------- sound

def _env(n, attack=0.005, release=0.1):
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    r = np.exp(-t / max(release, 1e-4))
    return a * r


def sfx_blip(freq=880, dur=0.12, vol=0.25):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = freq * (1 + 0.5 * np.exp(-t * 40))
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.002, dur / 3) * vol).astype(np.float32)


def sfx_whoosh(dur=0.6, vol=0.18, seed=1):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n)
    # Simple one-pole low-pass with a sweeping cutoff.
    out = np.zeros(n)
    y = 0.0
    for i in range(n):
        k = 0.02 + 0.25 * math.sin(math.pi * i / n)
        y += k * (noise[i] - y)
        out[i] = y
    env = np.sin(np.linspace(0, math.pi, n)) ** 2
    return (out * env * vol / (np.abs(out).max() + 1e-9)).astype(np.float32)


def sfx_boing(dur=0.5, vol=0.2):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 220 + 90 * np.sin(2 * np.pi * 9 * t) * np.exp(-t * 6)
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.005, 0.18) * vol).astype(np.float32)


def sfx_levelup(vol=0.22):
    notes = [523.25, 659.25, 783.99, 1046.5]
    parts = []
    for i, f in enumerate(notes):
        dur = 0.12 if i < 3 else 0.5
        n = int(dur * SR)
        t = np.arange(n) / SR
        tone = np.sign(np.sin(2 * np.pi * f * t)) * 0.35 + np.sin(2 * np.pi * f * t) * 0.65
        parts.append(tone * _env(n, 0.003, dur / 2 if i < 3 else 0.25) * vol)
    return np.concatenate(parts).astype(np.float32)


def music_bed(duration, bpm=100, vol=0.07):
    """Soft synth pad (Am-F-C-G) with a gentle pulse; sits under the voice."""
    n = int(duration * SR)
    t = np.arange(n) / SR
    chords = [[220.0, 261.63, 329.63], [174.61, 220.0, 261.63], [261.63, 329.63, 392.0], [196.0, 246.94, 293.66]]
    bar = 4 * 60 / bpm
    out = np.zeros(n, dtype=np.float64)
    for ci in range(int(duration / bar) + 1):
        a, b = int(ci * bar * SR), min(n, int((ci + 1) * bar * SR))
        if a >= n:
            break
        tt = t[a:b] - t[a]
        seg = np.zeros(b - a)
        for f in chords[ci % 4]:
            seg += np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * f * 2.005 * tt)
        fade = np.minimum(1, np.minimum(tt / 0.4, (bar - tt) / 0.4))
        out[a:b] += seg * np.clip(fade, 0, 1) / 4
    # Pulse on each beat (side-chain feel).
    beat = 60 / bpm
    pulse = 0.65 + 0.35 * (1 - np.exp(-((t % beat) / 0.12)))
    # Low "kick" thump on each beat.
    kick = np.sin(2 * np.pi * 55 * (t % beat)) * np.exp(-(t % beat) * 18) * 0.6
    fade_io = np.clip(np.minimum(t / 1.5, (duration - t) / 1.5), 0, 1)
    return ((out * pulse + kick) * vol * fade_io).astype(np.float32)


def mix(base, clip, at):
    i = int(at * SR)
    if i >= len(base):
        return
    j = min(len(base), i + len(clip))
    base[i:j] += clip[: j - i]


def write_wav(path, mono):
    peak = np.abs(mono).max()
    if peak > 0.98:
        mono = mono * (0.98 / peak)
    pcm = (np.clip(mono, -1, 1) * 32767).astype(np.int16)
    stereo = np.repeat(pcm[:, None], 2, axis=1)
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(stereo.tobytes())


# ---------------------------------------------------------------- encoding

def encode(render_frame, n_frames, audio_wav, out_mp4, workers=None):
    """Render frames in parallel (fork) and pipe them into ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-i", audio_wav,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", out_mp4,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with get_context("fork").Pool(workers or os.cpu_count()) as pool:
        for i, frame in enumerate(pool.imap(render_frame, range(n_frames), chunksize=4)):
            proc.stdin.write(frame)
            if i % 150 == 0:
                print(f"  frame {i}/{n_frames}", flush=True)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")


# ---------------------------------------------------------------- episode building blocks

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
       "sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_to_words(n):
    """0..999 as spoken English (for TTS lines that must match on-screen numbers)."""
    n = int(n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    rest = n % 100
    return ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + num_to_words(rest))


class Timeline:
    """Script lines laid out in time (from voiceover()), with word lookups."""

    def __init__(self, lines):
        self.lines = lines

    def start(self, i):
        return self.lines[i]["start"]

    def end(self, i):
        return self.lines[i]["end"]

    def word(self, i, prefix, nth=0):
        hits = [s for w, s, _ in self.lines[i]["words"] if w.lower().startswith(prefix.lower())]
        if len(hits) <= nth:
            raise KeyError(prefix)
        return hits[nth]

    def scene(self, t, lead=0.12):
        s = 0
        for i in range(len(self.lines)):
            if t >= self.start(i) - lead:
                s = i
        return s


class PlotBox:
    """Maps data coords to screen pixels, plus a 2x supersampled drawing layer."""

    def __init__(self, x0, x1, y_top, y_bot, xr, yr, layer=(90, 410, 1010, 1270), ss=2):
        self.x0, self.x1, self.y_top, self.y_bot = x0, x1, y_top, y_bot
        self.xr, self.yr = xr, yr
        self.layer, self.ss = layer, ss

    def px(self, x):
        return self.x0 + (self.x1 - self.x0) * (x - self.xr[0]) / (self.xr[1] - self.xr[0])

    def py(self, y):
        return self.y_bot - (self.y_bot - self.y_top) * (y - self.yr[0]) / (self.yr[1] - self.yr[0])

    def L(self, x, y):
        """Data coords -> supersampled layer coords."""
        return ((self.px(x) - self.layer[0]) * self.ss, (self.py(y) - self.layer[1]) * self.ss)

    def new_layer(self):
        lx0, ly0, lx1, ly1 = self.layer
        return Image.new("RGBA", ((lx1 - lx0) * self.ss, (ly1 - ly0) * self.ss), (0, 0, 0, 0))

    def finish_layer(self, lay, glow=12, strength=1.5):
        lx0, ly0, lx1, ly1 = self.layer
        small = lay.resize((lx1 - lx0, ly1 - ly0), Image.LANCZOS)
        return add_glow(small, radius=glow, strength=strength)


def build_static(plot, xticks, yticks, xlabel, ylabel, ytick_fmt=str, xlabels_y=None):
    """Background gradient, game-board grid, plot grid/axes/labels and vignette."""
    img = vertical_gradient((W, H), BG_TOP, BG_BOTTOM).convert("RGBA")
    d = ImageDraw.Draw(img)
    for x in range(0, W, 60):
        d.line([(x, 0), (x, H)], fill=(255, 255, 255, 6))
    for y in range(0, H, 60):
        d.line([(0, y), (W, y)], fill=(255, 255, 255, 6))
    ax_x, ax_y = plot.px(xticks[0]), plot.py(yticks[0])
    top, right = plot.py(yticks[-1]), plot.px(xticks[-1])
    lab_y = ax_y if xlabels_y is None else xlabels_y  # x tick labels can sit below the axis line
    for gx in xticks:
        d.line([(plot.px(gx), top), (plot.px(gx), ax_y)], fill=GRID, width=2)
        text_center(d, (plot.px(gx), lab_y + 34), str(gx), font("SemiBold", 30), MUTED)
    for gy in yticks:
        d.line([(ax_x, plot.py(gy)), (right, plot.py(gy))], fill=GRID, width=2)
        text_center(d, (ax_x - 20, plot.py(gy)), ytick_fmt(gy), font("SemiBold", 30), MUTED, anchor="rm")
    d.line([(ax_x, ax_y), (right, ax_y)], fill=AXIS, width=4)
    d.line([(ax_x, top), (ax_x, ax_y)], fill=AXIS, width=4)
    text_center(d, ((ax_x + right) / 2, lab_y + 82), xlabel, font("Bold", 34), AXIS)
    lab = Image.new("RGBA", (420, 60), (0, 0, 0, 0))
    text_center(ImageDraw.Draw(lab), (210, 30), ylabel, font("Bold", 34), AXIS)
    lab = lab.rotate(90, expand=True)
    img.alpha_composite(lab, (28, int((top + ax_y) / 2 - lab.height / 2)))
    yy, xx = np.mgrid[0:H, 0:W]
    v = ((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H / 2) / (H * 0.7)) ** 2
    shade = Image.fromarray((np.clip(v, 0, 1) * 120).astype(np.uint8), "L")
    dark = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    dark.putalpha(shade)
    img.alpha_composite(dark)
    return img


def build_title(title, subtitle):
    layer = Image.new("RGBA", (W, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    text_center(d, (W / 2, 80), title, font("Black", 74), CYAN)
    text_center(d, (W / 2, 150), subtitle, font("Bold", 32), MUTED)
    return add_glow(layer, radius=12, strength=1.3)


def draw_hud_pills(frame, level, err_text, err_color):
    d = ImageDraw.Draw(frame)
    rounded_pill(d, (60, 150, 330, 234), fill=PURPLE + (235,))
    text_center(d, (195, 192), f"LEVEL {level}", font("Black", 44), WHITE)
    rounded_pill(d, (600, 150, 1020, 234), fill=(15, 18, 40, 230), outline=err_color + (255,), width=5)
    text_center(d, (705, 192), "ERROR", font("Bold", 34), MUTED)
    text_center(d, (905, 192), err_text, font("Black", 50), err_color)


def draw_caption(frame, t, chunks, y):
    for c in chunks:
        if c["start"] - 0.05 <= t < c["end"]:
            words = [w for w, _, _ in c["words"]]
            active = -1
            for k, (_, s, _) in enumerate(c["words"]):
                if s - 0.04 <= t:
                    active = k
            img = caption_image(words, active)
            pop = ease_back((t - c["start"] + 0.05) / 0.16, s=2.5)
            paste_scaled(frame, img, (W / 2, y), scale=0.85 + 0.15 * pop)
            return


def build_confetti(n=140, seed=3):
    r = np.random.default_rng(seed)
    cols = [CYAN, PINK, YELLOW, GREEN, PURPLE, ORANGE]
    return [dict(x=r.uniform(0, W), y=r.uniform(-400, -20), vx=r.uniform(-80, 80), vy=r.uniform(250, 650),
                 rot=r.uniform(0, 6.28), vr=r.uniform(-8, 8), w=r.uniform(14, 26), h=r.uniform(8, 14),
                 c=cols[int(r.integers(0, len(cols)))]) for _ in range(n)]


def draw_level_cleared(frame, t, t0, level, next_name, t_next, confetti):
    """Dim the screen, confetti, 'LEVEL N CLEARED!' banner, then the next-level card."""
    if t < t0:
        return
    g = clamp((t - t0) / 0.3)
    frame.alpha_composite(Image.new("RGBA", (W, H), (5, 6, 16, int(170 * g))))
    d = ImageDraw.Draw(frame)
    dt = t - t0
    for c in confetti:
        x = c["x"] + c["vx"] * dt
        y = c["y"] + c["vy"] * dt + 120 * dt * dt
        if y > H + 40:
            continue
        a = c["rot"] + c["vr"] * dt
        cw, ch = c["w"] / 2, c["h"] / 2 * abs(math.cos(a * 1.7))
        ca, sa = math.cos(a), math.sin(a)
        pts = [(x + ca * dx - sa * dy, y + sa * dx + ca * dy) for dx, dy in ((-cw, -ch), (cw, -ch), (cw, ch), (-cw, ch))]
        d.polygon(pts, fill=c["c"])
    s = ease_back(dt / 0.45, s=2.2)
    ban = Image.new("RGBA", (W, 330), (0, 0, 0, 0))
    bd = ImageDraw.Draw(ban)
    text_center(bd, (W / 2, 95), f"LEVEL {level}", font("Black", 120), WHITE)
    text_center(bd, (W / 2, 225), "CLEARED!", font("Black", 136), GREEN)
    paste_scaled(frame, add_glow(ban, radius=16, strength=1.4), (W / 2, 640), scale=s, alpha=clamp(s))
    if t >= t_next:
        g2 = ease_back((t - t_next) / 0.4)
        card = Image.new("RGBA", (860, 200), (0, 0, 0, 0))
        cd = ImageDraw.Draw(card)
        rounded_pill(cd, (10, 10, 850, 190), fill=(28, 16, 56, 245), outline=PURPLE + (255,), width=6, radius=40)
        text_center(cd, (430, 66), "NEXT LEVEL", font("Bold", 36), MUTED)
        text_center(cd, (430, 130), next_name, font("Black", 62), PURPLE)
        paste_scaled(frame, card, (W / 2, 950), scale=g2, alpha=clamp(g2))


def sfx_buzz(dur=0.18, freq=110, vol=0.16):
    """Low square-wave 'wrong answer' buzz."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    return (np.sign(np.sin(2 * np.pi * freq * t)) * _env(n, 0.003, dur) * vol).astype(np.float32)


def draw_training_readout(frame, x0, x1, y, step, total, right_text, alpha):
    """'TRAINING STEP n/N' + parameter readout + progress bar under the plot."""
    d = ImageDraw.Draw(frame)
    fill = lambda c: c + (int(255 * alpha),)  # noqa: E731
    text_center(d, (x0, y), f"TRAINING  STEP {step:02d}/{total}", font("Bold", 34), fill(WHITE), anchor="lm")
    text_center(d, (x1, y), right_text, font("Bold", 34), fill(MUTED), anchor="rm")
    d.rounded_rectangle((x0, y + 34, x1, y + 46), radius=6, fill=lerp_color(BG_BOTTOM, (60, 66, 104), alpha))
    p = step / total
    if p > 0:
        d.rounded_rectangle((x0, y + 34, x0 + (x1 - x0) * p, y + 46), radius=6, fill=fill(GREEN))
