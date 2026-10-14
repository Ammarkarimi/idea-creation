# ML as a Game, Level 11: LSTM (Long Short-Term Memory)

**File:** `lstm_short.mp4` (1080×1920, 30 fps, about 49 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:35 ("says: French!") also makes a good cover frame.

## Upload copy

**Title** (pick one):
- How AI remembers a word from way back 🧠 #shorts
- LSTMs explained: a memory belt with 3 gates (Level 11)
- Why LSTMs beat simple RNNs at long sentences

**Description:**
```
"I grew up in France … years later, I speak ___?" To answer, you have to remember a word from way back.
A simple RNN (Level 10) halves its memory every word, so after 8 words only 0.004 of "France" is left.
An LSTM keeps a memory belt controlled by 3 gates (valves, each 0 to 1):
  FORGET gate: how much old memory to keep
  INPUT gate:  what new info to write
  OUTPUT gate: what to say right now
France → the input gate opens and writes it. Filler words → the forget gate stays open (0.98), so 0.86 of France survives.
"speak" → the output gate opens, reads the belt: "French!"

ML as a Game · Level 11. Levels 1–10 are on the channel.
Next level: Transformers, how chatbots read. Follow so you don't miss it.

#machinelearning #ai #lstm #deeplearning #nlp #shorts
```

**Pinned comment:**
> What would happen if the forget gate were 0.5 instead of 0.98? 🤔 (Hint: Level 10)

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | I grew up in France. Years later, I speak... what? To answer, you have to remember one word from way back. | The sentence with "France" highlighted and a blinking blank. "← one word from way back". |
| 2 | Last level, a simple recurrent network forgot the start. Its memory halved every word. | A "simple RNN: France left" meter drains 1 → 0.004. |
| 3 | Long short-term memory fixes that. It keeps a memory belt, with three gates, like valves. | A conveyor belt with a memory box, and the LSTM cell appears. "LSTM = MEMORY BELT + 3 GATES". |
| 4 | The forget gate decides how much old memory to keep. The input gate decides what new info to write. | The FORGET ("× keep") and INPUT ("+ write") gauges appear and pulse. |
| 5 | And the output gate decides what to say right now. | The OUTPUT ("read") gauge appears. |
| 6 | Watch. France: the input gate opens, and France goes onto the belt. | Words stream in. At "France": write 0.95, and the memory box becomes "France 0.92". |
| 7 | Filler words: the forget gate stays open, so the memory barely fades. | keep 0.98 on every word. The memory goes 0.92 → 0.79. Meters: RNN 0.004 vs LSTM 0.86. |
| 8 | Speak: the output gate opens, reads the belt... French! | say 0.95, output 0.63, then "says: French!". |
| 9 | Gates let these networks remember for a long time. They powered speech recognition and translation for years. | "GATES = LONG MEMORY". |
| 10 | Level eleven cleared. Next: transformers, how chatbots read. Follow for level twelve. | Confetti, then the "Next level: TRANSFORMERS" card. |

## Accuracy notes
- **Real LSTM equations** (one memory cell), computed in `render.py`:
  - f = σ(4), i = σ(6·is_country − 3), g = tanh(2·is_country), o = σ(6·is_question − 3)
  - c = f·c + i·g, h = o·tanh(c)
- **The weights are hand-set for teaching**; real LSTMs learn them (backprop through time). Real LSTMs have hundreds of memory cells working on word vectors, and their forget gate depends on the input too. Here it's a constant 0.98 to keep the picture simple.
- **Per word:** at "France" i = 0.95, g = 0.96, c = 0.92. Then 8 words at f = 0.98 each bring c to 0.79. At "speak" o = 0.95 and h = 0.63. Before "speak", h stays at about 0.03 because the output gate is nearly closed (asserted).
- **"France left":** for the LSTM this is c / c_France = 0.98⁸ ≈ 0.86. For Level 10's simple RNN it's 0.5⁸ ≈ 0.004. The code asserts LSTM > 0.75 and RNN < 0.01.
- **The "French!" readout is illustrative.** A real model would turn h into a word with a trained output layer.
- **History:** LSTMs were the standard for speech recognition and machine translation through the mid-2010s, before transformers took over (Level 12).

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~2 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
