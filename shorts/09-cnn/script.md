# ML as a Game, Level 9: How AI Sees Images (CNNs)

**File:** `cnn_short.mp4` (1080×1920, 30 fps, about 53 s)
**Cover:** `thumbnail.png`. In the YouTube app, about 0:28 (both feature maps lit) also makes a good cover frame.

## Upload copy

**Title** (pick one):
- How AI actually SEES a picture 👁️ #shorts
- Convolutions explained with one tiny X (Level 9)
- A 3×3 filter is all it takes to start seeing

**Description:**
```
To a computer, a picture is a grid of numbers (0 = dark, 1 = bright).
A convolution slides a tiny 3×3 filter of weights over it. At each spot it multiplies and adds,
just like a neuron (Level 7). A big score means "pattern found", and those spots light up the feature map.
One filter finds "\" lines, another finds "/". Both together? That's an X.
Because the same filter slides everywhere, it finds the pattern wherever it is.
Real networks learn their filters with backprop (Level 8) and stack layers: edges → shapes → faces.

ML as a Game · Level 9. Levels 1–8 are on the channel.
Next level: Recurrent Networks (RNNs), AI with memory. Follow so you don't miss it.

#machinelearning #ai #cnn #computervision #deeplearning #shorts
```

**Pinned comment:**
> Challenge: design a 3×3 filter that finds a horizontal line. Drop your numbers 👇

## Narration script
| # | Line | On screen |
|---|------|-----------|
| 1 | To a computer, a picture is just a grid of numbers. So how does AI see? | A flickering 8×8 grid of 0/1 pixels. "HOW DOES AI SEE?" |
| 2 | Every pixel is a number: zero for dark, one for bright. Here's a tiny picture of an X. | Pixels settle into an X with 0/1 in every cell and a "0 = dark 1 = bright" tag. The X glows. |
| 3 | A convolution uses a tiny filter: a three by three grid of weights that looks for one pattern, like a diagonal line. | The "\" filter appears (+1 green on the diagonal, −1 pink elsewhere), with a yellow 3×3 window on the image. |
| 4 | Slide it over the picture. At each spot, multiply and add, just like a neuron. | The window slides (slow for the first spots, then quick). A live card: "bright on the line 3 × (+1), bright off the line 0 × (−1), SCORE = 3". The feature map fills cell by cell. HUD SCORE follows. |
| 5 | Big number? Pattern found! Those spots light up the feature map. | The two 3s flash. "← 3 = FOUND IT!" |
| 6 | A second filter hunts the other diagonal. Both light up? That's an X. | The "/" filter does a quick slide and the second map fills. "\ + / BOTH FOUND = X ✓". |
| 7 | And since the same filter slides everywhere, it finds the X wherever it is. | The X steps to a new spot and both maps' peaks move with it. "SAME FILTER EVERYWHERE → FINDS IT ANYWHERE". |
| 8 | Real networks learn their filters with backprop, and stack layers: edges become shapes, shapes become faces. | "filters are learned by backprop (Level 8)", then a pixels → edges → shapes → faces stack. |
| 9 | That's a convolutional neural network. It's how AI spots cats, faces, and stop signs. | "CONVOLUTIONAL NEURAL NETWORK (CNN)" with cats / faces / stop signs tags. |
| 10 | Level nine cleared. Next: recurrent networks, AI with memory. Follow for level ten. | Confetti, then the "Next level" card. |

## Accuracy notes
- **Image:** 8×8, pixel values 0/1, holding a 5×5 X (top-left, later moved by 3 rows and 2 columns).
- **Filters:** "\" has +1 on the diagonal and −1 elsewhere; "/" is its mirror. Each feature-map cell is the sum of (patch × filter) over the 3×3 window (valid convolution, stride 1, so the maps are 6×6). This is cross-correlation, which is what deep-learning libraries call "convolution". Everything is computed with numpy in `render.py`.
- **Scores:** a perfect diagonal match scores 3 and the X's center scores 1 (asserted). With 0/1 pixels, the score is exactly "bright pixels on the line − bright pixels off the line", which is what the on-screen card shows.
- **Shift:** after moving the X by (3, 2), both maps' peak positions move by exactly (3, 2) (asserted). That's the "same filter everywhere" property (translation equivariance).
- **Simplifications, stated honestly:**
  - The filters are hand-made for teaching. Real CNNs learn their filter values with backprop, as the narration says.
  - Real CNNs also add a bias and an activation (like ReLU) after each convolution, and usually pooling. Those aren't shown.
  - The pixels → edges → shapes → faces stack is a simplified illustration of what deeper layers tend to respond to.

## Re-render
```
pip install -r ../requirements.txt
python3 render.py                 # video + thumbnail (~2 min on 4 cores)
python3 render.py --preview 5,20  # just preview frames
```
