# ML as a Game, Level 2: Logistic Regression

**File:** `logistic_regression_short.mp4` (1080×1920, 30 fps, about 47 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:27 (the penalty bars) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- Why a straight line can't predict pass or fail 🎮 #shorts
- Logistic Regression explained as a GAME (Level 2)
- 132% chance to pass?! Fixing it with one curve

**Description:**
```
A straight line can predict 132% or -18%, which is impossible for a probability.
Logistic regression squashes the line through a sigmoid so every output is a chance between 0 and 100%.
Training minimizes a penalty (log loss) that punishes confident mistakes hardest.
Where the curve crosses 50% is the decision boundary.

ML as a Game · Level 2. Missed Level 1 (linear regression)? It's on the channel.
Next level: Decision Trees. Follow so you don't miss it.

#machinelearning #ai #datascience #logisticregression #python #shorts
```

**Pinned comment:**
> At 4.5 hours the model says 55%: a pass, but barely. Would you trust it? 👇

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | Can a straight line predict pass or fail? Watch it break. | An orange line draws in, then shatters, and red "impossible" zones flash. |
| 2 | Same students. But now every dot is just pass, or fail. | Dots scatter in as in Level 1, then snap into a PASS row (green) and a FAIL row (pink). |
| 3 | Fit a straight line, and at ten hours it predicts over a hundred percent. That's impossible. | The least-squares line shows "132% ?!" and "−18% ?!", plus an IMPOSSIBLE tag. |
| 4 | So we bend it: squash the line through a sigmoid, so it stays between zero and one hundred percent. | The line morphs into an S-curve, and the title becomes `p = sigmoid(mx + b)`. |
| 5 | Its height is the chance of passing. But training starts from a random guess. | A probe slides along the curve reading out %, then the curve flips backwards ("RANDOM GUESS"). |
| 6 | The score punishes confident mistakes. Super sure, and wrong? Huge penalty. | Penalty bars appear, the error counts up to 29.3, and the 9.2 h student gets "98% sure: FAIL → PASSED. WRONG!" |
| 7 | Training slides and stretches the curve, until the penalties are as small as possible. | Gradient descent runs over 60 steps, the bars shrink, and the error drops to 4.9. |
| 8 | Where it crosses fifty percent is the decision line. Left fails. Right passes. | A 50% line and a gate at 4.4 h appear, with FAIL and PASS zones. |
| 9 | Studied four and a half hours? Fifty-five percent chance. Just a pass! | A guide line at 4.5 h and a "55% → PASS" bubble appear. |
| 10 | Level two cleared. Next: decision trees. Follow for level three. | Confetti, then the "Next level" card. |

## Accuracy notes
- **Data:** 15 synthetic students. The overlap (a pass at 3.2 h, a fail at 5.0 h) keeps the maximum-likelihood fit finite.
- **Straight line:** the real least-squares fit to the 0/1 labels. It gives −18% at 0 h and 132% at 10 h.
- **Fitted curve:** found with Newton's method: w = 1.28, b = −5.58. It crosses 50% at 4.35 h and gives 55% at 4.5 h. The spoken number is generated from the model, so voice and screen always match.
- **ERROR:** the total log loss. Penalty bar length = −log(probability given to the true answer), capped for display.
- **Training animation:** real momentum gradient descent from the "random guess" curve. It ends on the fitted model.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~2 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
