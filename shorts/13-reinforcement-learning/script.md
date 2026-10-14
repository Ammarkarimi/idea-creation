# ML as a Game, Level 13: Reinforcement Learning

**File:** `rl_short.mp4` (1080×1920, 30 fps, about 49 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:34 (the green 6-move path with the arrows) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- This robot taught itself to find the treasure 🤖💎 #shorts
- Reinforcement learning explained in 50 seconds (Level 13)
- How AI learns games with NO answers given

**Description:**
```
Nobody gives this robot the answer. It learns by playing.
Rules: treasure +10, lava −10, every move −1.
Game 1: it moves randomly and falls in the lava.
After every move it updates a score: nudge it toward reward + 0.9 × the next square's best score.
Squares next to the treasure score high first, then the value spreads backward, square by square.
20% of the time it still tries a random move (exploring), in case there's a better path.
After 100 games: straight to the treasure in 6 moves. Treasure found: first 10 games 3/10 → last 10 games 10/10.
That's reinforcement learning: try, get a reward, remember.

ML as a Game · Level 13. Levels 1–12 are on the channel.
Next level: Diffusion, how AI draws pictures. Follow so you don't miss it.

#machinelearning #ai #reinforcementlearning #qlearning #gameai #shorts
```

**Pinned comment:**
> Why does every move cost −1? What would the robot do if moves were free? 🤔

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | Nobody tells this robot the answer. It has to learn by playing. | A 4×4 grid builds and the robot appears with a "?". |
| 2 | Treasure: plus ten. Lava: minus ten. Every move costs one point. | Diamond "+10", two lava squares "−10", "every move: −1". |
| 3 | At first, it just moves randomly, and falls in the lava. | Game 1 plays out: a wandering pink trail, 13 moves, then a splash into lava. REWARD −22. |
| 4 | But after every move, it updates a score: how good was that move, from that square? | A red arrow into the lava, "that move: −5". Card: "nudge the move's score toward: reward + 0.9 × next square's best". |
| 5 | Squares next to the treasure score high first. Then the value spreads backward, square by square. | Heatmap of each square's best score through training games 1→30. Green spreads out from the treasure. |
| 6 | That's reinforcement learning: try, get a reward, remember. | TRY → REWARD → REMEMBER while training continues. |
| 7 | Sometimes it still tries a random move, in case there's a better path. | Best-move arrows appear. "20% of moves: random (explore)". Training reaches game 100. |
| 8 | After a hundred games, it walks straight to the treasure in six moves. | The robot follows the arrows: green path, "6 MOVES!", REWARD +5. "treasure found: first 10 games 3/10 → last 10 games 10/10". |
| 9 | Add neural networks, and this idea learned Atari games, beat a Go world champion, and helps fine-tune chatbots. | "+ NEURAL NETWORKS", "ATARI GAMES", "GO WORLD CHAMPION", "FINE-TUNING CHATBOTS". |
| 10 | Level thirteen cleared. Next: diffusion, how AI draws pictures. Follow for level fourteen. | Confetti, then the "Next level: DIFFUSION" card. |

## Accuracy notes
- **Real tabular Q-learning**, computed in `render.py` (seeded, so it's the same every render):
  - **Update rule:** after each move, Q(square, move) ← Q + 0.5 × (reward + 0.9 × max Q(next square) − Q). The on-screen "nudge toward" means this: each update moves the score halfway toward the target. Scores start at 0.
  - **Rewards:** stepping onto the treasure gives +10 and stepping into lava gives −10 (both end the game). Every other move, including bumping into a wall, gives −1.
  - **Exploration:** ε-greedy with ε = 0.2. Ties between equal scores are broken at random.
- **The numbers on screen are the code's output, checked with asserts:**
  - **Game 1:** it ends in lava after 13 moves (reward −22). The lava move's score becomes 0.5 × (−10) = −5.
  - **The value spreads backward:** the squares on the final path turn positive at games 7, 12, 15, 21, 24 and 27, going from the treasure back to the start. The code asserts this order.
  - **Treasure found:** 3 of the first 10 games, 10 of the last 10.
  - **The final path:** following the best arrow from each square (no random moves) takes 6 moves, the shortest possible, avoids the lava and scores +5. The asserts check that it's the shortest path.
- **The cell numbers** are each square's best score, max Q. Next to the treasure it's 10, then 8.0, 6.2, 4.6, 3.1 and 1.8 along the path. That's reward + 0.9 × the next square's value at each step, as in the formula card.
- **"Add neural networks…":**
  - **Atari:** DeepMind's DQN (2015) used Q-learning with a neural network in place of the table.
  - **Go:** AlphaGo (2016), which beat Lee Sedol, combined neural networks, reinforcement learning from self-play and tree search. That's the same learn-from-reward idea with more machinery, not this exact algorithm.
  - **Chatbots:** they are commonly fine-tuned with reinforcement learning from human feedback (RLHF) and related methods. On screen this is "helps fine-tune", worded modestly on purpose.
- **Simplified:** real environments are far bigger than a 4×4 grid, so the scores can't be stored in a table. A neural network estimates them instead.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~2 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
