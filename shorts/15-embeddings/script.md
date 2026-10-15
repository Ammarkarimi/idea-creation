# ML as a Game, Level 15: Word Embeddings

**File:** `embeddings_short.mp4` (1080×1920, 30 fps, about 39 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:23 ("≈ QUEEN") also makes a good cover frame.

## Upload copy

**Title** (pick one):
- King − Man + Woman = ? 👑 How AI does word math #shorts
- Word embeddings explained in 40 seconds (Level 15)
- How AI knows "king" and "queen" are related

**Description:**
```
Computers can't read words, only numbers. So every word gets a spot on a map: a list of numbers called an embedding.
Similar words land close together (royals, people, animals, food).
The spots are learned from reading: words used in similar sentences get pulled together.
Directions mean something too. Man → woman is one arrow.
king − man + woman lands right next to queen (0.2 away).
The same arrow turns boy into girl.
Real embeddings have hundreds of numbers per word, and every chatbot starts with them.

ML as a Game · Level 15. Levels 1–14 are on the channel.
Next level: GANs, a forger versus a detective. Follow so you don't miss it.

#machinelearning #ai #embeddings #nlp #word2vec #shorts
```

**Pinned comment:**
> Guess: paris − france + italy = ? 🤔 Answer below 👇

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | Computers can't read words, only numbers. So how does AI know king and queen are related? | "king" → "? ? ?", then king ↔ queen, "related?". |
| 2 | Give every word a spot on a map, as a list of numbers. Similar words land close together. | 16 words appear on a map. "king = (−3.6, 6.2)". Clusters glow: royals, people, animals, food. |
| 3 | The spots are learned from reading: words used in similar sentences get pulled together. | "cat" jumps to a random spot. "the cat drinks milk / the dog drinks milk", then cat slides next to dog. Banner says "(SIMPLIFIED)". |
| 4 | Now directions mean something. Man to woman is one arrow. | A pink arrow from man to woman, labelled "woman − man". |
| 5 | Start at king. Subtract man. Add woman. | "king", "− man", "+ woman". A probe dot travels from king along the same arrow, and the HUD shows the nearest word live. |
| 6 | And you land right next to queen! | "≈ QUEEN (just 0.2 away)". NEAREST: queen. |
| 7 | The same arrow turns boy into girl. | The arrow from boy lands on girl. "≈ GIRL (0.2 away)". |
| 8 | These number lists are called embeddings. Real ones have hundreds of numbers, and every chatbot starts with them. | "EMBEDDING = a word as a list of numbers", "this map: 2 numbers per word", "real models: hundreds", "STEP 1 OF EVERY CHATBOT". |
| 9 | Level fifteen cleared. Next: Gans, a forger versus a detective. Follow for level sixteen. | Confetti, then the "Next level: GANs" card. |

(The narration spells it "Gans" so the AI voice says "ganz". It reads "GAN" as one syllable and sometimes spells it out. The card shows "GANs".)

## Accuracy notes
- **The map positions are hand-placed for teaching.** Each word is 2 numbers; real embeddings are learned and have hundreds of numbers.
  - I first tried learning the positions from a tiny hand-written corpus: co-occurrence counts → PPMI → SVD, the classic count-based method.
  - At that size the analogy was fragile. It held for only 13 of 20 random choices of sentences, and the 2D map was unreadable. So this episode uses hand-placed positions instead of cherry-picking a lucky run.
- **Computed and asserted in `render.py`:**
  - woman − man = (6.0, −0.3)
  - king − man + woman = (2.4, 5.9). The nearest word is queen, 0.22 away, even without excluding king, man and woman from the search.
  - boy − man + woman = (1.2, −0.6). The nearest word is girl, 0.22 away.
  - Every word's nearest neighbour is in its own group, and cat–dog (2.9) is much closer than cat–pizza (12.5).
- **"Nearest" here means straight-line distance on the map.** Real systems usually compare embeddings by cosine similarity (the angle between them).
- **The screen stretches the map a little vertically.** This is a linear stretch, so arrows and the word arithmetic look exactly as they compute.
- **Scene 3 is a simplified picture of training.** Methods like word2vec (2013) and GloVe (2014) learn embeddings from billions of words. They nudge words that appear in similar contexts toward each other.
- **On real embeddings**, king − man + woman ≈ queen is the famous word2vec result. It holds for many analogies but not all, and the input words are usually excluded from the search.
- **"Every chatbot starts with them":** transformer language models (Level 12) begin by turning each token into an embedding vector.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~1.5 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
