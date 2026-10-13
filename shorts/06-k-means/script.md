# ML as a Game, Level 6: K-Means Clustering

**File:** `kmeans_short.mp4` (1080×1920, 30 fps, about 46 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:31 (the named groups) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- The computer found the groups by itself 🚩 #shorts
- K-Means Clustering in 3 steps (Level 6)
- K-Means vs KNN: don't mix them up!

**Description:**
```
K-means finds groups in data that has NO answers (unsupervised learning):
1) Drop K flags at random.
2) Every point joins the closest flag.
3) Each flag moves to the middle (average) of its group.
Repeat until nobody switches. The "spread" (total squared distance to your flag) shrinks every round.

Don't mix it up with KNN (Level 5): KNN copies answers from labeled neighbors. K-means has no answers; it invents the groups.

ML as a Game · Level 6. Levels 1–5 are on the channel.
Next level: Neural Networks. Follow so you don't miss it.

#machinelearning #ai #datascience #kmeans #clustering #shorts
```

**Pinned comment:**
> Night owl, sleepyhead or balanced: which group are YOU in? 🦉😴⚖️

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | No answers this time. Can a computer find groups all by itself? | Gray dots, three flags hop around. "NO ANSWERS. FIND THE GROUPS!" |
| 2 | Here are some students: hours studied and hours slept. No pass or fail. Just dots. | "NO LABELS · JUST DOTS". |
| 3 | K-means sorts them into K groups. Let's pick K equals three. | The algorithm card and SPREAD meter appear. "K = 3 GROUPS". |
| 4 | Step one: drop three flags at random spots. | Card row ① lights up and three flags drop in, all on the right side (a bad start). |
| 5 | Step two: every student joins the closest flag. | Row ②: students sweep left to right, each joining the nearest flag (lines + color). Closest-flag territories appear. SPREAD 262. |
| 6 | Step three: each flag moves to the middle of its group. | Row ③: "+" marks each group's average ("+ = average") and the flags slide there with trails. SPREAD drops to 103. |
| 7 | Then repeat. Join, move, join, move, until nobody switches. | Row ↻: rounds 2–5. Banner "ROUND n · JOIN/MOVE · switched: 9 → 7 → 1 → 0". SPREAD → 22. "NOBODY SWITCHED → DONE ✓". |
| 8 | Done! It found three study styles: night owls, sleepyheads, and balanced. | Name tags pop next to each group. |
| 9 | Don't mix it up with KNN. KNN copies answers from labeled neighbors. K-means has no answers. It invents the groups. | "KNN ≠ K-MEANS". Two compare cards: "KNN · LEVEL 5: has answers" vs "K-MEANS · LEVEL 6: no answers". |
| 10 | Level six cleared. Next: neural networks. Follow for level seven. | Confetti, then the "Next level" card. |

## Accuracy notes
- **Data:** 30 synthetic students in three loose groups (seed 42). K-means never sees which group a student was generated in; that's used only to check the result.
  - studied 7–9.5 h, slept 2.8–5 h
  - studied 0.8–3 h, slept 7.8–9.6 h
  - studied 4.3–6.5 h, slept 5.6–7.8 h
- **Algorithm:** plain Lloyd's k-means in `render.py`. The flags start at uniformly random positions (seed 21): (7.2, 5.8), (6.7, 1.7), (6.0, 8.8).
- **Students who switched flag each round:** 30 (first join), 9, 7, 1, 0.
- **Spread** (sum of squared distances to your flag, after each join / move): 262, 103, 85, 71, 54, 27, 23, 22, 22, 22. It never goes up, and the code asserts this.
- **Final flags:** night owls (8.5, 4.0), sleepyheads (2.0, 8.6), balanced (5.5, 6.6). They match the three generated groups exactly.
- **Names** come from where each final flag sits: most study = night owls, most sleep = sleepyheads, the remaining group = balanced.
- **Honest caveat:** k-means can get stuck if the random start is unlucky. Real libraries run it several times from different starts and keep the result with the smallest spread (e.g. scikit-learn's `n_init`). The video uses a single start and doesn't claim otherwise.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~2 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
