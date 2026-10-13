# ML as a Game, Level 7: Neural Networks

**File:** `neural_networks_short.mp4` (1080×1920, 30 fps, about 54 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:39 (the "both say yes → PASS" moment) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- How a neural network actually "thinks" 🧠 #shorts
- Neural Networks explained in 3 steps (Level 7)
- One neuron can't do this. Two can.

**Description:**
```
A neuron does 3 things:
1) Multiply each input by a weight.
2) Add them up (plus a bias).
3) Squash the total into 0–100% with a sigmoid (Level 2's logistic regression!).

One neuron can only draw one straight line, so it can't catch a "sweet spot" band.
Stack them: two hidden neurons each learn one line ("slept enough?" and "not too much?"),
and the output neuron says PASS only when both say yes. Training nudges every weight until the mistakes shrink.
More neurons + more layers = deep learning.

ML as a Game · Level 7. Levels 1–6 are on the channel.
Next level: Backpropagation, how networks actually learn. Follow so you don't miss it.

#machinelearning #ai #neuralnetwork #deeplearning #datascience #shorts
```

**Pinned comment:**
> Quick check: what would this network say for 9.5 hours of sleep? 😴 (Hint: neuron ② 👀)

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | Every AI you've heard of is built from one tiny piece: the neuron. | A small network with signals flowing. "THE BUILDING BLOCK OF AI". |
| 2 | A neuron takes inputs, like hours studied and hours slept, and multiplies each one by a weight. | "STEP 1": inputs 6 h and 7 h, edge tags "× 0.8" / "× 0.5" that become "6 × 0.8 = 4.8" and "7 × 0.5 = 3.5" ("example weights"). |
| 3 | Then it adds them up and squashes the total into zero to one hundred percent. That's level two's sigmoid! | "STEP 2": Σ "4.8 + 3.5 − 5 (bias) = 3.3". "STEP 3": an S-curve inside the neuron → "96% chance to pass". "LEVEL 2's SIGMOID!" |
| 4 | Now a trickier class. You pass only with the right amount of sleep. Too little or too much? Fail. | 40 students appear on the plot. "PASS = SLEEP SWEET SPOT" with band hint lines. |
| 5 | One neuron can only draw one straight line. It can't catch this band. | The trained single neuron's straight line and shading. "1 NEURON = 1 STRAIGHT LINE · 8 WRONG". "can't bend!" |
| 6 | So we stack them: two hidden neurons, then one output neuron. That's a neural network. | The diagram morphs to 2 → 2 → 1. "2 HIDDEN NEURONS → 1 OUTPUT". |
| 7 | Training nudges every weight, like level one, until the mistakes shrink. | Real training: edge thickness/color follow the weights, the pass region bends into the band. "TRAINING · STEP n · k WRONG", 24 → 0. |
| 8 | Look inside. Neuron one learned: slept enough? Neuron two: not too much? The output says pass only when both say yes. | Each hidden neuron's own line on the plot (cyan ≈ 4.9 h, orange ≈ 8.5 h) with "① slept enough?" / "② not too much?". The output lights up: "① AND ② → PASS". |
| 9 | Add more neurons and more layers, and it can draw almost any shape. That's deep learning. | The diagram grows into a bigger network. "MORE NEURONS + LAYERS = DEEP LEARNING". |
| 10 | Level seven cleared. Next: backpropagation, how networks actually learn. Follow for level eight. | Confetti, then the "Next level" card. |

## Accuracy notes
- **Example neuron (scenes 2–3):** illustrative weights 0.8 and 0.5 with bias −5, labelled "example weights" on screen. The arithmetic is computed in code: 6×0.8 + 7×0.5 − 5 = 3.3, and sigmoid(3.3) = 96%.
- **Data:** 40 synthetic students (seed 7), hours studied and slept both uniform 0.5–9.5 h. A student passes if 5 ≤ slept ≤ 8.5 h (16 pass). Hours studied doesn't matter in this class.
- **One neuron:** logistic regression fitted with Newton's method. Its best straight line still gets 8 of 40 wrong.
- **Network:** a real 2-2-1 sigmoid network (inputs scaled as (hours − 5) / 3). It's trained by full-batch gradient descent with momentum (lr 0.5, momentum 0.9) for 6,000 steps from a random start (seed 13). The animation shows 75 snapshots spaced logarithmically in steps. Training errors go 24 → 0.
- **The "look inside" labels are true of the trained weights, and the code asserts it:**
  - **Hidden neuron ①** fires when slept > 4.89 h. Its sleep weight is ≫ its studied weight, so its line is nearly horizontal.
  - **Hidden neuron ②** fires when slept < 8.51 h.
  - **Output weights** are +18.2 and +19.1 with bias −27.6. Either neuron alone stays below zero, and both together are above it: a real AND.
- **Honest caveat:** out of 20 random starts I tried, 12 learned this clean solution, and the rest got stuck at 4 wrong (a local minimum). That's normal for neural networks and why people retrain or use better starts.
- **Scene 9's bigger network** is illustrative (random edge colors). It isn't trained.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~2–3 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
