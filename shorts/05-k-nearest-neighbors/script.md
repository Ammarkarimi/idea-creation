# ML as a Game, Level 5: K-Nearest Neighbors

**File:** `knn_short.mp4` (1080×1920, 30 fps, about 49 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:21 (the K = 3 vote) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- K-Nearest Neighbors in 3 steps 🎮 #shorts
- Want to predict someone? Ask their neighbors. (KNN, Level 5)
- KNN explained as a GAME, and why K matters

**Description:**
```
K-Nearest Neighbors in 3 steps:
1) Measure the distance from the new point to every known point.
2) Pick the K closest.
3) Let them vote.

There's no training: KNN just remembers the data.
K matters. K = 1 trusts a single (maybe weird) neighbor; K = everyone gives everyone the same answer.
Pick K by testing: hide each student, predict them from the rest, keep the K that gets the most right.

ML as a Game · Level 5. Levels 1–4 are on the channel.
Next level: K-Means Clustering (different thing, don't mix them up!). Follow so you don't miss it.

#machinelearning #ai #datascience #knn #python #shorts
```

**Pinned comment:**
> Quick quiz: what would K = 5 say for this student? (Answer's in the video 👀)

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | Want to guess if a new student will pass? Just ask their closest neighbors. | A star appears among the students with a pulsing neighbor circle. "WHO ARE YOUR NEIGHBORS?" PASS/FAIL legend. |
| 2 | Here's a new student: five and a half hours studying, six and a half hours of sleep. | A "NEW: 5.5 h study · 6.5 h sleep" tag appears. |
| 3 | Step one: measure how far they are from every other student. | "STEP 1": lines to all 28 students, distance labels on the nearest 4, and the formula: distance = straight-line gap = √(Δstudied² + Δslept²). |
| 4 | Step two: pick the K closest. Let's say K equals three. | "STEP 2": rank badges #1, #2, #3, a circle around the 3 nearest, and the K slider at 3. |
| 5 | Step three: the neighbors vote. Two passed, one failed. So: pass! | "STEP 3": vote chips FAIL · PASS · PASS → PASS. The star turns green. |
| 6 | That's k nearest neighbors. There's no training. It just remembers everyone. | The title becomes "ask the K closest → majority vote". "TRAINING TIME: 0 s". All students pulse. |
| 7 | But K matters. With K equals one, only the closest counts, and that's our weird student. Fail! | The slider moves to 1 and the circle shrinks to the weird student (ringed), giving FAIL. The K = 1 map shows islands. |
| 8 | Make K too big, and the whole class votes. Most failed, so everyone gets fail. | The slider moves to 28 and the circle covers the plot. Tally: PASS 10 / FAIL 18 → FAIL. The map is all pink. |
| 9 | So we test: hide each student, and predict them from the rest. K equals five scores best. | "HIDE ONE · PREDICT IT · REPEAT": a bar chart of right answers per K. K = 5 wins with 23/28, and the K = 5 map appears. "BEST: K = 5 → PASS". |
| 10 | Level five cleared. Next: k means clustering. Don't mix them up! Follow for level six. | Confetti, then the "Next level" card. |

## Accuracy notes
- **Data:** the same 28 students as Levels 3–4. The new student is at (5.5 h, 6.5 h).
- **Nearest neighbors** (Euclidean distance, both axes in hours):
  - #1: (5.4, 7.1) FAIL at 0.61, the "weird student" from Levels 3–4
  - #2: (5.9, 5.4) PASS at 1.17
  - #3: (4.3, 6.2) PASS at 1.24
  - #4: (6.7, 7.5) PASS at 1.56
  - #5: (5.6, 4.6) FAIL at 1.90
- **Votes:** K = 1 → 0/1 pass (FAIL). K = 3 → 2/3 (PASS). K = 5 → 3/5 (PASS). K = 28 → 10/28 (FAIL).
- **Choosing K:** leave-one-out accuracy, i.e. each student predicted from the other 27.
  - K = 1: 16
  - K = 3: 21
  - **K = 5: 23 (best)**
  - K = 7: 20
  - K = 9: 21
  - K = 11: 22
  - K = 13: 21
  - K = 15: 21
  - all 27 others: 18
- **The plot is square on screen** (same pixels per hour on both axes), so the neighbor circle really is a circle in data space and distances look right.
- **The colored maps** are the actual K-NN vote over a 200×200 grid.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~2 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
