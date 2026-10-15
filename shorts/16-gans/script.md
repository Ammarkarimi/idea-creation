# ML as a Game, Level 16: GANs (Generative Adversarial Networks)

**File:** `gans_short.mp4` (1080×1920, 30 fps, about 41 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:26 ("COIN FLIP: 50%" with the curves on top of each other) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- Two AIs: a forger vs a detective 🕵️ #shorts
- GANs explained in 40 seconds (Level 16)
- How AI learned to fake reality

**Description:**
```
A GAN is a game between two AIs.
The forger makes fake gold coins. The detective says "real or fake?".
Real coins weigh about 8 g. The forger's first fakes weigh about 2 g, so the detective is 100% right.
The forger learns from every catch and makes heavier fakes. The detective gets sharper. They take turns.
After 50 rounds, the fakes weigh just like real coins and the detective is down to a coin flip: 50%.
Scale it up to pictures, and GANs made photo-real faces of people who don't exist.

ML as a Game · Level 16. Levels 1–15 are on the channel.
What should Level 17 be? Tell me in the comments!

#machinelearning #ai #gan #generativeai #deeplearning #shorts
```

**Pinned comment:**
> Level 17 is YOUR call. Overfitting? Gradient descent? Recommendation systems? Vote below 👇

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | Two AIs play a game. One is a forger. One is a detective. | The FORGER (masked) "VS" the DETECTIVE (hat and magnifying glass). |
| 2 | Real gold coins weigh about eight grams. The forger's first fakes weigh about two. | A weight axis from 0 to 11 g. A gold "REAL" curve at 8 g and a pink "FAKE" curve at 2 g. "aims at 2.0 g". |
| 3 | The detective learns to spot fakes. Easy: one hundred percent right. | The detective's "real?" curve appears. ROUND 1/50, right: 100%. |
| 4 | The forger learns from every catch, and makes heavier fakes. | The fake curve slides right (rounds 1→8). |
| 5 | Round after round, the detective gets sharper, and the forger gets better. | Rounds 8→50: the fakes reach 8 g and the detective's curve keeps reshaping. The accuracy bar drops from green to pink. |
| 6 | After fifty rounds, the fakes weigh just like real coins. The detective is down to a coin flip: fifty percent. | The two curves overlap. "COIN FLIP: 50%". |
| 7 | That's a Gan: a generative adversarial network. Two rivals make each other better. | "G A N" → Generative / Adversarial / Network. "rivals make each other better". |
| 8 | Scale it up to pictures, and Gans made photo-real faces of people who don't exist. | "+ BIG NETWORKS + PICTURES", "PHOTO-REAL FACES of people who don't exist". |
| 9 | Level sixteen cleared! What should we learn next? Tell me in the comments. | Confetti, then the "Next level: YOU DECIDE!" card. |

(The narration spells it "Gan" / "Gans" so the AI voice says "gan" / "ganz". Written "GANs", the voice sometimes spells the letters out.)

## Accuracy notes
- **A real GAN, trained in numpy** in `render.py` (seeded, so it's the same every render).
  - **Forger (generator):** fake = b + 0.5·noise. It learns b, the weight it aims for, starting at 2 g.
  - **Detective (discriminator):** logistic regression on (1, u, u²) with u = (weight − 5)/3. It outputs P(real).
  - **Each round:**
    1. The detective takes 5 gradient steps on the standard GAN loss (real = 1, fake = 0).
    2. The forger takes 1 step to raise log D(fake), the usual "non-saturating" generator loss.
- **Numbers from the run, asserted in code where shown:**
  - **Round 1:** the detective is right 100% of the time (asserted).
  - **Fake weight by round:** 2.11 g (1), 3.44 g (8), 6.53 g (20), 7.76 g (25), then a peak of 8.45 g around round 30. It settles at 7.92 g by round 50 (asserted within 0.3 g of 8).
  - **Detective accuracy:** 96% at round 20, 68% at round 25, and 39% at round 30, when the forger overshoots and the detective briefly gets it backwards. That's the see-saw.
  - **"Coin flip: 50%"** is the average accuracy over the last 10 rounds (asserted between 45% and 55%). At the end the HUD shows that same average; the last single round was 49%.
- **Simplified:**
  - **What the forger learns:** here it learns only the average weight; its spread is fixed at the real spread (0.5 g). When I let the forger learn its spread too, this tiny GAN showed mode collapse: the fakes squeezed onto almost one weight and the spread kept swinging. That is a real, well-known GAN failure, but too much for 40 seconds.
  - **Real GANs** use deep neural networks for both players and work on whole images.
- **History:**
  - GANs were introduced by Ian Goodfellow and colleagues in 2014.
  - StyleGAN (2019) made the famous photo-real faces of people who don't exist.
  - Since then, diffusion models (Level 14) have replaced GANs in most image generators, partly because GAN training is unstable.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~1.5 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
