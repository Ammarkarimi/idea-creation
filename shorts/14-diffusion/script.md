# ML as a Game, Level 14: Diffusion

**File:** `diffusion_short.mp4` (1080×1920, 30 fps, about 42 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:27 (three different hearts from three different statics) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- How AI turns static into a picture 🎨 #shorts
- Diffusion models explained in 40 seconds (Level 14)
- Every AI image starts as pure noise. Here's why

**Description:**
```
How do AI image generators draw? With diffusion.
Training: take a picture and add noise, step by step, until it's pure static.
A model learns one skill: look at a noisy picture and guess the noise.
Creating: start from random static, remove a bit of the guessed noise, and repeat. After 20 clean-ups a picture appears.
Add a prompt like "a heart" and every guess steers toward hearts.
Different starting static → a different heart.
That's diffusion: learn to remove noise, then create from noise.

ML as a Game · Level 14. Levels 1–13 are on the channel.
What should Level 15 be? Tell me in the comments!

#machinelearning #ai #diffusion #aiart #generativeai #shorts
```

**Pinned comment:**
> Level 15 is YOUR call. GANs? Word embeddings? Overfitting? Vote below 👇

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | Every AI image starts as pure static. Watch. | A canvas of coloured static clears into an orange heart (NOISE 100% → 0%). |
| 2 | Training: take a picture, and add noise, step by step, until it's just static. | A star gets noisier, with a strip at 0 / 25 / 50 / 75 / 100% noise. |
| 3 | Then a model learns one skill: look at a noisy picture, and guess the noise. | "noisy picture (50%)" → MODEL → "its noise guess". "remove the guessed noise → clean picture". |
| 4 | To make a new picture, start from random static, subtract a bit of the guessed noise, and repeat. | Fresh static, then "CLEAN-UP STEP k / 20" with a progress bar. |
| 5 | Twenty small clean-ups, and a picture appears! | A smiley appears. "no prompt → a smiley". |
| 6 | Add a prompt, like a heart, and every guess steers toward hearts. | "prompt: a heart". The same static as before now becomes a pink heart. |
| 7 | Start from different static, and you get a different heart. | Three statics side by side → pink, red and purple hearts. |
| 8 | That's diffusion: learn to remove noise, then create from noise. | "TRAINING: picture → noise (learn to guess it)", "CREATING: noise → picture". |
| 9 | Real image generators do this with huge networks, trained on millions of pictures. | "HUGE NETWORKS", "MILLIONS OF PICTURES", "SAME IDEA: NOISE → PICTURE". |
| 10 | Level fourteen cleared! What should we learn next? Tell me in the comments. | Confetti, then the "Next level: YOU DECIDE!" card. |

## Accuracy notes
- **Real diffusion math**, computed in `render.py` (seeded, so it's the same every render):
  - **Forward noising:** noisy = √ᾱ · picture + √(1−ᾱ) · noise, using the standard cosine noise schedule with 1000 noise levels.
  - **"NOISE %"** is the noise's weight √(1−ᾱ). It is 100% for pure static and 0% for a clean picture.
  - **Creating** uses deterministic DDIM sampling in 20 steps from pure Gaussian static. Each step estimates the clean picture, works out the noise from it, and moves to the next lower noise level. The noise % per step is 100, 100, 99, 97 … 24, 17, 9, 0.
- **The "model" is the ideal one, not a trained network.** The training set is 8 tiny 16×16 pixel-art images (4 hearts, 2 stars, 2 smileys). For such a small set, the best possible noise guess can be computed exactly (a probability-weighted average of the training images). That is what a perfectly trained model would learn.
  - **Consequence:** this toy can only reproduce its 8 training images. The code asserts each result matches one of them exactly.
  - **The 50% panel:** the guessed noise matches the true noise almost perfectly (correlation 1.000) because the toy model knows the star.
- **Real image generators** train a large neural network (a U-Net or a transformer) to predict the noise across millions of pictures. That's what lets them make genuinely new images instead of copies.
- **"Remove the guessed noise → clean picture"** is simplified. Exactly: clean = (noisy − √(1−ᾱ) · noise guess) / √ᾱ.
- **The prompt is simplified.** Here "a heart" restricts the guesses to the heart images (class conditioning). Real systems turn the text into numbers with a text encoder and steer each step with it, often using classifier-free guidance.
- **Results, asserted in code:**
  - From the same static (seed 3), no prompt gives a smiley and the prompt "a heart" gives a heart.
  - Three different statics give three different hearts.
- **20 steps** matches common fast samplers. Older diffusion models used hundreds or 1000 steps.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~1.5 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
