# ML as a Game, Level 3: Decision Trees

**File:** `decision_trees_short.mp4` (1080×1920, 30 fps, about 48 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:30 (the finished tree) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- This AI grades you by playing 20 Questions 🎮 #shorts
- Decision Trees explained as a GAME (Level 3)
- How a decision tree picks its questions

**Description:**
```
A decision tree is a game of yes/no questions.
For each question it tries every possible cut and keeps the one that leaves each side least mixed (Gini impurity), then repeats on what's still mixed.
Ask too many questions and it memorizes the weird cases (overfitting), so keep trees short.

ML as a Game · Level 3. Levels 1 (linear regression) and 2 (logistic regression) are on the channel.
Next level: Random Forests, a whole team of trees. Follow so you don't miss it.

#machinelearning #ai #datascience #decisiontree #python #shorts
```

**Pinned comment:**
> 6 h studying + 4 h sleep = FAIL according to the tree. Agree or disagree? 😴👇

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | This AI decides your grade by playing twenty questions. | Random cut lines flicker across the dots, and "?" nodes pop up. |
| 2 | Every dot is a student: hours studied, and hours slept. Green passed. Pink failed. | The dots turn green or pink. |
| 3 | The tree's job: ask yes or no questions that split the passes from the fails. | ERROR counts to 10 (the "always guess fail" baseline), and a root "?" appears. |
| 4 | First question. It tries every possible cut, and scores how mixed each side is. | A vertical cut sweeps with a live "MIXED %" meter and a ghost best-so-far line. |
| 5 | Best cut: studied less than four hours? Almost everyone there failed. | The cut locks at 4 h, the root and FAIL leaf appear, and ERROR drops to 7. |
| 6 | The rest are still mixed, so it asks again: slept less than five hours? | The right side is outlined, then a horizontal cut sweeps and locks at 5 h. |
| 7 | Now each box is almost pure. That's a decision tree. | All boxes are shaded, the tree is complete, and ERROR is 3. |
| 8 | New student: six hours studying, four hours of sleep. Follow the questions. Fail! Get some sleep. | A star dot appears, its path lights up root → no → slept < 5 h? → yes → FAIL. |
| 9 | But ask too many questions, and it memorizes weird students. Keep the tree short. | A full-depth tree draws 8 tiny boxes around 3 outliers (ERROR 0, "OVERFITTING"), then snaps back. |
| 10 | Level three cleared. Next: random forests, a whole team of trees. Follow for level four. | Confetti, then the "Next level" card. |

## Accuracy notes
- **Data:** 28 synthetic students (10 passed), with 3 deliberate outliers.
- **Splits:** found by an exhaustive CART search (midpoint thresholds on both features, weighted Gini) in `render.py`. The root is studied < 4 h (mixed 92% → 65%), and the right branch is slept < 5 h. The spoken thresholds come from that search.
- **"MIXED %":** weighted Gini impurity ÷ 0.5. 100% is a 50/50 mix and 0% is pure.
- **ERROR:** the number of students the current tree misclassifies: 10 → 7 → 3.
- **Overfitting scene:** the same algorithm grown to full depth: 8 leaves, 0 training errors.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~2 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
