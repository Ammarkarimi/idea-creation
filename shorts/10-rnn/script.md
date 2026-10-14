# ML as a Game, Level 10: Recurrent Neural Networks (RNNs)

**File:** `rnn_short.mp4` (1080×1920, 30 fps, about 46.5 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:26 (both chains with GOOD / BAD verdicts) also makes a good cover frame.

**Series order from here:** Level 10 RNN → Level 11 LSTM → Level 12 Transformers.

## Upload copy

**Title** (pick one):
- AI with MEMORY: how RNNs read in order 🧠 #shorts
- Same words, opposite meaning. Here's how AI knows (Level 10)
- Recurrent Neural Networks explained in 45 seconds

**Description:**
```
A recurrent neural network (RNN) reads one word at a time and carries a memory forward.
At every step, the SAME neuron mixes the new word with half of its old memory:
  memory = squash( word + ½ × old memory )
"boring start, but amazing ending" → memory ends +0.35 → GOOD review
"amazing start, but boring ending" → memory ends −0.35 → BAD review
Same words, opposite answer: word order matters.
But each step keeps only half of the old memory. After 10 words, the first word is down to 0.001.
Simple RNNs forget the beginning of long sentences.

ML as a Game · Level 10. Levels 1–9 are on the channel.
Next level: LSTM (Long Short-Term Memory), a memory that doesn't fade. Follow so you don't miss it.

#machinelearning #ai #rnn #deeplearning #nlp #shorts
```

**Pinned comment:**
> What would the memory say for "amazing start, amazing ending"? Try the formula 👇

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | Is this review good or bad? To know, you have to read it in order. | Jumbled word cards line up in reading order 1–5. "GOOD OR BAD? READ IN ORDER". |
| 2 | A recurrent neural network reads one word at a time, and keeps a memory. | A chain of 5 identical cells above the words. "memory starts at 0". "RNN = ONE WORD AT A TIME + A MEMORY". |
| 3 | At each word, the same neuron mixes the new word with half of its old memory, then passes it on. | Formula card: memory = squash( word + ½ × old memory ). "THE SAME NEURON AT EVERY STEP". |
| 4 | Boring: memory goes negative. Start, but: it fades. Amazing: it flips positive. Good review! | Cells fill step by step: −0.76, −0.36, −0.18, +0.72, +0.35, with memory packets travelling along the arrows. "ends +0.35 → GOOD REVIEW". |
| 5 | Now swap it: amazing start, but boring ending. Same words, opposite answer. | A second chain: +0.76, +0.36, +0.18, −0.72, −0.35. "ends −0.35 → BAD REVIEW". |
| 6 | That's a recurrent network. Memory flows forward, so word order matters. | Forward arrows pulse over both chains. "MEMORY FLOWS FORWARD → ORDER MATTERS". |
| 7 | But each step keeps only half of the old memory. After ten words, the first word is almost gone. | Bar chart of word #1's share: 1, 0.5, 0.25 … 0.001 after 10 words. "EACH STEP: OLD MEMORY × ½". |
| 8 | Long sentences make simple recurrent networks forget the beginning. | "FORGETS THE START". |
| 9 | Level ten cleared. Next: long short-term memory, a memory that doesn't fade. Follow for level eleven. | Confetti, then the "Next level: LSTM" card. |

(The voiceover says "recurrent network" and "long short-term memory" in full so the AI voice doesn't stumble over the acronyms. The cards show RNN / LSTM.)

## Accuracy notes
- **The tiny RNN:** memory_t = tanh(word_t + 0.5 × memory_(t−1)), starting from 0, all computed in `render.py`. Word scores: boring = −1, amazing = +1, start / but / ending = 0.
- **Sentence A** ("boring start, but amazing ending"): −0.76, −0.36, −0.18, +0.72, **+0.35**.
- **Sentence B** ("amazing start, but boring ending"): +0.76, +0.36, +0.18, −0.72, **−0.35**.
- The code asserts both sentences contain the same words but end with opposite signs.
- **Fading:** this RNN keeps 0.5 of the old memory each step, so a word's direct share after k more steps is 0.5^k: 1, 0.5, 0.25 … 0.001 after 10 steps. tanh's squashing makes it shrink even a bit faster. This is the intuition behind the "vanishing gradient" problem that LSTMs were designed to fix.
- **Simplifications:**
  - The weights (1 for the word, 0.5 for the old memory) are hand-set for teaching. Real RNNs learn them with backpropagation through time (Level 8's idea applied across the steps).
  - Real RNNs have a memory vector of many numbers and turn words into vectors (embeddings), not single ±1 scores.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~1.5 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
