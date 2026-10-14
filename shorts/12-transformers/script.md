# ML as a Game, Level 12: Transformers & Attention

**File:** `transformers_short.mp4` (1080×1920, 30 fps, about 50 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:26 ("it" → animal 88%) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- How ChatGPT actually reads your sentence 🤖 #shorts
- Transformers explained: attention in 50 seconds (Level 12)
- What does "it" mean? How AI figures it out

**Description:**
```
Every modern chatbot is built on the transformer.
An RNN reads one word at a time. A transformer reads every word at once,
and each word asks: which other words matter to me? That's attention.
"The animal didn't cross the street because it was too tired."
1) "it" scores every word by how well their meanings match
2) softmax turns the scores into percentages → animal 88%
Change "tired" to "wide" → street 87%. Same word, new meaning, from context.
Every word can look at every other word, so distance doesn't matter and nothing fades.
Stack dozens of these layers, train them to predict the next word on huge amounts of text → a chatbot.

ML as a Game · Level 12. Levels 1–11 are on the channel.
What should Level 13 be? Tell me in the comments!

#machinelearning #ai #transformers #chatgpt #attention #shorts
```

**Pinned comment:**
> Level 13 is YOUR call. Diffusion (image generators)? Reinforcement learning? GANs? Vote below 👇

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | ChatGPT, Claude, Gemini: they're all built on one idea. The transformer. | Three name tags converge into a "TRANSFORMER" block. |
| 2 | A recurrent network reads one word at a time. A transformer reads every word at once. | "RNN: one at a time" (cells light one by one) vs "TRANSFORMER: all at once" (all light together). |
| 3 | Then each word asks: which other words matter to me? That's attention. | "which words matter to me?", then "ATTENTION". |
| 4 | The animal didn't cross the street because it was too tired. What does it mean? | The sentence as word cards, with "it" highlighted. "WHAT DOES 'IT' MEAN?" |
| 5 | It scores every word by how well their meanings match. Softmax turns the scores into percentages. | Lines from "it" to every word. Bars show match scores (animal 4.5, street 0.6, …), then morph into softmax percentages. |
| 6 | Animal gets eighty-eight percent of the attention. So it means the animal. | "animal" glows green with 88%. "'IT' = THE ANIMAL (88%)". |
| 7 | Change tired to wide, and attention jumps to street. Same word, new meaning, from context. | "tired" flips to "wide" and the bars re-sort: street 87%. |
| 8 | Every word can look at every other word, so distance doesn't matter. Nothing fades. | All-to-all lines. "first ↔ last: just one step". |
| 9 | Stack dozens of these layers, train them to predict the next word on huge amounts of text, and you get a chatbot. | A stack of attention layers, "The cat sat on the ___ → mat" (labelled illustration), "= CHATBOT". |
| 10 | Level twelve cleared! What should we learn next? Tell me in the comments. | Confetti, then the "Next level: YOU DECIDE!" card. |

## Accuracy notes
- **Attention**, as computed in `render.py`:
  - each word has a meaning vector, its "key"
  - "it" has a "query" vector
  - score = query · key (dot product)
  - attention = softmax(scores) over all 11 words, which sums to 100% (asserted)
- **The vectors are hand-made, 3 numbers each** ([living thing, place, action]: animal = [3, 0, 0], street = [0, 3, 0], …).
  - "tired" context: query = [1.5, 0.2, 0], so animal scores 4.5 and gets 88%.
  - "wide" context: query = [0.2, 1.5, 0], so street gets 87%.
  - The code asserts the top word flips from animal to street.
- **Simplified, said honestly:**
  - In real transformers, keys, queries and values are learned vectors with hundreds of numbers, computed by learned weight matrices.
  - The query for "it" depends on the context ("tired" vs "wide") because earlier layers have already mixed information between words. Here that context-dependent query is written by hand.
  - Real models also scale scores by √(dimension), use many attention "heads" in parallel, and add position information.
- **Scene 9 is an illustration of the idea.** Large language models stack many such layers and are trained to predict the next word on huge text datasets.
- **The opening names** (ChatGPT, Claude, Gemini) refer to the general fact that these assistants are built on transformer-based models. No logos or branding are used.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~2 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
