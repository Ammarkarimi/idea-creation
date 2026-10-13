# ML as a Game, Level 4: Random Forests

**File:** `random_forests_short.mp4` (1080×1920, 30 fps, about 44 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:33 (the vote tally) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- 100 trees vote. One tree gets fooled. 🌲 #shorts
- Random Forests explained as a GAME (Level 4)
- Why one decision tree isn't enough

**Description:**
```
A single decision tree is unstable: change a few training examples and it asks totally different questions.
A random forest trains many trees, each on a random sample of the data (some rows picked twice, some left out),
and each split only looks at a random subset of features. Every tree overfits differently, so when they vote,
the weird bits cancel out.

ML as a Game · Level 4. Levels 1–3 (linear regression, logistic regression, decision trees) are on the channel.
Next level: K-Nearest Neighbors. Follow so you don't miss it.

#machinelearning #ai #datascience #randomforest #python #shorts
```

**Pinned comment:**
> One tree said FAIL, 58 of 100 said PASS. Would you trust the single tree or the crowd? 🌲👇

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | One decision tree is easy to fool. A whole forest? Much harder. | Single-tree boxes flicker ("1 TREE"), then the smooth 100-tree vote map fades in. |
| 2 | Change just a few students, and a single tree asks totally different questions. | Three versions with 3 students removed (✕) give three very different trees. "UNSTABLE!" |
| 3 | So we grow a team. Each tree trains on a random sample of students. Some picked twice, some left out. | Tree #1's bootstrap sample: ×2/×3 badges on repeats, and left-out students fade. |
| 4 | And each question only gets to look at one random feature. | A "dice roll" between "CAN ONLY ASK: STUDIED?" and "…SLEPT?", then tree #1's boxes appear. |
| 5 | So every tree overfits in its own weird way. | Five mini panels pop in (trees #1–#5), each with different boxes. |
| 6 | But when they all vote, the weird bits cancel out. | The main map averages 1 → 100 trees and the boxes blur into a smooth vote map. The TREES counter climbs. |
| 7 | New student: five and a half hours studying, eight hours of sleep. | A star appears on the main map and in every mini panel. |
| 8 | One tree says fail, fooled by one weird student. But fifty-eight out of a hundred trees say pass. Pass! | The single tree's FAIL strip and the weird student are highlighted. The minis vote, and the tally fills to PASS 58 / FAIL 42. |
| 9 | That's a random forest: lots of noisy trees, one smart vote. | The title becomes "100 trees → 1 vote". |
| 10 | Level four cleared. Next: k nearest neighbors. Follow for level five. | Confetti, then the "Next level" card. |

## Accuracy notes
- **Data:** the same 28 students as Level 3.
- **Forest:** 100 trees, all computed in `render.py` (seed 0):
  - Each tree is a full-depth CART tree (weighted Gini) trained on a bootstrap sample (28 draws with replacement).
  - Each split considers one random feature (max_features = 1, the √2 default for classification). If that feature can't split, it falls back to the other.
- **"One tree":** the plain full-depth tree on all 28 students. Its FAIL strip (studied 5.25–5.65 h, slept ≥ 5 h) exists only because of the one student who failed at (5.4 h, 7.1 h).
- **The vote:** 58 of 100 trees vote PASS at (5.5 h, 8 h). Across 8 different forest seeds the PASS share ranged from 57% to 72%, always a PASS. So the result doesn't depend on a lucky seed.
- **What the video doesn't claim:** on this tiny toy dataset the forest isn't reliably more accurate overall than one tree, so the video makes no accuracy-percentage claim. It shows the robust part: stability and averaging away one-off quirks.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~2 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
