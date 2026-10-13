# ML as a Game, Level 1: Linear Regression

**File:** `linear_regression_short.mp4` (1080×1920, 30 fps, about 38 s)
**Cover:** `thumbnail.png`. In the YouTube app you can also pick a cover frame; around 0:18 (pink squares) works well.

## Upload copy

**Title** (pick one):
- Linear Regression explained as a GAME 🎮 #shorts
- This line is playing a game (it's how ML starts)
- Machine Learning Level 1: Linear Regression in 38 seconds

**Description:**
```
Linear regression = find the one line with the smallest total squared error.
Think of each data point pulling on the line with a spring: the line settles where the pulls balance.

ML as a Game · Level 1 of a series that explains one machine-learning model at a time.
Next level: Logistic Regression. Follow so you don't miss it.

#machinelearning #ai #datascience #linearregression #python #shorts
```

**Pinned comment:**
> Which model should be Level 3: decision trees, k-means, or neural networks? 👇

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | This line is playing a game, and it's how machine learning begins. | A line spins wildly, and the error counter flickers. |
| 2 | Every dot is a student: hours studied, and their exam score. | 12 dots pop in. |
| 3 | The goal: one straight line that's as close to every dot as possible. | A bad starting line draws in. |
| 4 | Tie each dot to the line with a spring. Far dots pull harder. | Springs attach and the line jiggles. |
| 5 | These gaps are the errors. Square them, add them up. That's the score. Lower wins. | Squared-error squares appear, and the error counts up to 592. |
| 6 | The model nudges the slope and the starting point, again and again, until the springs stop pulling. | Gradient descent with momentum runs over 60 steps, and the error drops to 16. |
| 7 | That's linear regression: y equals m x plus b. | The title turns into `y = 7x + 36`. |
| 8 | Studied six hours? It predicts about seventy-eight. | A guide line drops at 6 h, and an "≈ 78" bubble appears. |
| 9 | Level one cleared. Next: logistic regression. Follow for level two. | Confetti, then the "Next level" card. |

## Accuracy notes
- The data is synthetic (fixed seed). The least-squares fit is forced to be exactly `y = 7x + 36`, so 6 h gives 78.
- The springs analogy is mathematically exact: vertical springs pull the line to the least-squares fit.
- The error shown is the mean squared error (MSE).
- The training animation is real gradient descent with momentum, computed in `render.py`.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py              # video + thumbnail (~1.5 min on 4 cores)
python3 render.py --preview 5,20   # just preview frames
```
