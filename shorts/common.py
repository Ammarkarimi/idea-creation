"""Shared helpers for the "ML as a Game" YouTube Shorts series.

Each episode lives in its own folder and imports this module for:
voiceover (Piper TTS), caption timing, drawing primitives, procedural
music/SFX, and ffmpeg encoding.
"""

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
