# ML as a Game, Level 8: Backpropagation

**File:** `backprop_short.mp4` (1080×1920, 30 fps, about 51 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:33 (the full blame ledger) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- Backpropagation = blame flows backward 🔙 #shorts
- How neural networks ACTUALLY learn (Level 8)
- Your AI guessed wrong. Which weight is to blame?

**Description:**
```
Backpropagation on the smallest possible network, with every number on screen:
Forward pass: slept 8 h → the network guesses 22% chance to pass. But they passed.
Backward pass: blame flows from the output back to each weight, one multiplication at a time:
 1) blame(output) = prediction − answer
 2) blame(last weight) = output's blame × what flowed into it
 3) blame(hidden) = blame × weight × the hidden neuron's slope → and the same again for the first weight
Then nudge each weight against its blame (gradient descent, Level 1). One nudge: 22% → 34%.
Repeat for every student, thousands of times, and the whole network learns.

ML as a Game · Level 8. Levels 1–7 are on the channel.
Next level: how AI sees images. Follow so you don't miss it.

#machinelearning #ai #backpropagation #neuralnetwork #deeplearning #shorts
```

**Pinned comment:**
> Both weights went UP after the nudge. Can you explain why from the blame signs? 🤔👇

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | Your neural network guessed wrong. So which weight is to blame? | The network with a red "22%" and red blame pulses racing backward. "WRONG ANSWER. WHO'S TO BLAME?" |
| 2 | Meet the smallest network: one input, one hidden neuron, one output, and two weights. | Nodes pop in, then the weight tags w1 = 0.10 and w2 = 0.30. "THE SMALLEST NETWORK". |
| 3 | Forward pass: this student slept eight hours. The network says twenty-two percent chance to pass. But they passed! | Yellow signals flow right, h = 0.69, output 22%, then "ANSWER: PASSED ✓". "FORWARD PASS →". |
| 4 | So the output is off by seventy-eight points. That's its blame. | Ledger ①: blame(output) = 0.22 − 1 = −0.78. "← BACKWARD PASS". |
| 5 | Step one: the last weight's blame is the output's blame, times what flowed into it. | A red pulse goes back along w2. Ledger ②: −0.78 × 0.69 = −0.54. |
| 6 | Step two: pass the blame backward, through that weight and the hidden neuron's slope. | The pulse enters the hidden neuron and a slope tangent appears on its S-curve. Ledger ③: −0.78 × 0.30 × 0.21 = −0.05. |
| 7 | Step three: the first weight gets its blame the same way. Blame flows backward. That's backpropagation! | A pulse goes back along w1. Ledger ④: −0.05 × 8 = −0.40. "BACKPROPAGATION = BLAME FLOWS BACKWARD". |
| 8 | Now nudge each weight against its blame, like level one. One nudge, and twenty-two percent becomes thirty-four. | Ledger ⑤: w2 0.30 → 0.84, w1 0.10 → 0.50. The weights animate, a forward replay runs, and the output reads 22% → 34%. |
| 9 | Repeat that for every student, thousands of times, and the whole network learns. | A bigger network with alternating forward (yellow) and backward (pink) pulses. A sparkline shows this student over 25 nudges, 22% → 96%. |
| 10 | Level eight cleared. Next: how AI sees images. Follow for level nine. | Confetti, then the "Next level" card. |

## Accuracy notes
- **Network:** h = sigmoid(w1·x), o = sigmoid(w2·h + b), with x = 8 h, w1 = 0.10, w2 = 0.30 and a fixed bias b = −1.5. Only the two weights are trained here; in real networks the biases get nudged the same way.
- **Forward pass:** h = 0.690, o = 0.215 (22%). Answer = 1 (passed).
- **Loss:** log loss (as in Level 2). With a sigmoid output, its gradient at the output is exactly prediction − answer = −0.785. That's why the output's "blame" is that simple.
- **Chain rule, all computed in `render.py`:**
  - blame(w2) = −0.785 × 0.690 = −0.541
  - hidden slope = h(1 − h) = 0.214
  - blame(hidden) = −0.785 × 0.30 × 0.214 = −0.050
  - blame(w1) = −0.050 × 8 = −0.403
- **Finite-difference check:** the code asserts both weight blames match the numerical derivative of the loss to within 1e-6. The on-screen blame really is the gradient.
- **Nudge (learning rate 1):** w1 → 0.503, w2 → 0.841. The new prediction is 33.8% (shown as 34%).
- **Sparkline:** 25 real nudges on this single student: 22% → 96%, rising every step (asserted). Real training averages the blame over many students. The scene-9 bigger network is illustrative, with random edge colors.
- **Rounding:** the ledger shows 2 decimals, and each row's shown inputs multiply to its shown result.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~2 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
