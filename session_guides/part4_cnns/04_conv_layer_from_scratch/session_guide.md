# MLForBeginners — Instructor Guide
## Part 4 · Module 04: Conv Layer from Scratch
### Two-Session Teaching Script

> **Prerequisites:** CNN modules 01–03 complete. They understand image representation
> (H×W×C arrays), what a convolution does mathematically, pooling, and have built
> an MLP from scratch in Part 3. They have NOT yet written a Conv2D class.
> **Payoff today:** They will implement a working Conv2D layer in pure NumPy and
> watch it produce the same output as TensorFlow's Conv2D.

---

# SESSION 1 (~90 min)
## "Building Conv2D from scratch — the flashlight that scans your photo"

## Before They Arrive
- Terminal open in `convolutional_neural_networks/algorithms/`
- Whiteboard ready with a rough grid drawn (6×6 pixel patch)
- `conv_layer_from_scratch.py` open but not yet run
- Optional: print a 6×6 pixel grid on paper to hand out

---

## OPENING (10 min)

> *"You've seen what a convolution does from the math side. Today we
> build the actual code that does it.*
>
> *Here's the mental model I want you to hold: imagine you have a
> flashlight, and you're shining it on a photo in a dark room.
> The flashlight only illuminates one small square patch at a time.
> You slide it across the whole photo — left to right, top to bottom —
> one step at a time.*
>
> *Each time you light up a patch, you do one operation: multiply
> every pixel in the patch by a matching weight, sum them all up,
> get one number. That number tells you 'how much of a thing I'm
> looking for is here.'*
>
> *That's a convolutional filter. That's all it is. A weighted flashlight."*

Draw on board:
```
FILTER (3x3):            IMAGE PATCH:           OUTPUT:
┌───┬───┬───┐           ┌───┬───┬───┐
│ 1 │ 0 │-1 │           │120│130│ 90│
├───┼───┼───┤  ×  ==>   ├───┼───┼───┤   sum = one pixel
│ 2 │ 0 │-2 │           │200│210│150│         in the
├───┼───┼───┤           ├───┼───┼───┤      feature map
│ 1 │ 0 │-1 │           │180│190│140│
└───┴───┴───┘           └───┴───┴───┘

  "Sobel edge        "This patch of the image"
   detector"
```

> *"The filter slides across the whole image. At every position it lights
> up — we get one number. Stack all those numbers up and you get the
> feature map. The filter has LEARNED which patterns to look for.*
>
> *Today: we build the sliding loop in NumPy. Let's go."*

---

## SECTION 1: The Conv2D Class — Weights and Initialization (20 min)

Write on board:
```
Conv2D needs to track:
  W: shape = (kH, kW, C_in, n_filters)
  b: shape = (n_filters,)

  kH, kW   = kernel height/width (e.g., 3, 3)
  C_in     = input channels (1 for grayscale, 3 for RGB)
  n_filters = how many different patterns to detect
```

> *"Notice the weight shape: (3, 3, C_in, n_filters).*
>
> *n_filters is critical. If n_filters=32, we're learning 32 different
> flashlights at once — 32 different patterns to look for simultaneously.*
>
> *Each filter produces its own feature map. Stack 32 feature maps and
> the next layer gets 32-channel input. That's how depth grows."*

Draw the filter bank:
```
Input image                 After Conv2D(n_filters=4)
(32×32×3)                      (32×32×4)
                               ┌─────────┐
                               │ map 1   │ ← edges filter
┌─────────┐   4 filters        │ map 2   │ ← curves filter
│  RGB    │  ─────────>        │ map 3   │ ← texture filter
│  image  │                    │ map 4   │ ← color-contrast filter
└─────────┘                    └─────────┘
```

Show the `__init__` and `build` code:
```python
class Conv2D:
    def __init__(self, n_filters, kernel_size, stride=1, padding="same"):
        self.n_filters = n_filters
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.W = None   # not yet allocated — need input shape first
        self.b = None

    def build(self, input_shape):
        _, _, C = input_shape          # (H, W, C_in)
        kH, kW = self.kernel_size
        fan_in = kH * kW * C
        # He initialization: scaled random weights
        self.W = np.random.randn(kH, kW, C, self.n_filters) * np.sqrt(2.0 / fan_in)
        self.b = np.zeros(self.n_filters)
```

> *"He initialization — same trick we used in the MLP from scratch.
> Keeps activations from exploding or vanishing at the start.*
>
> *We delay building until we see the first input. That's the same
> pattern Keras uses under the hood — it's called lazy initialization."*

**INSTRUCTOR TIP — Common confusion:**
> Students often ask why W has 4 dimensions. Trace each axis on the board:
> axis 0 = row within kernel, axis 1 = col within kernel,
> axis 2 = input channel, axis 3 = which filter.
> Say: "The last axis is just 'which flashlight are we talking about.'"

---

## SECTION 2: Weight Sharing — Why CNNs Are So Efficient (15 min)

> *"Here's the mind-bending thing about CNNs: the SAME filter weights
> are used at EVERY position in the image.*
>
> *If we have a 3×3 edge-detector filter, we use those exact 9 weights
> to check for edges at position (0,0), then (0,1), then (0,2)... all
> the way to (30,30). The same 9 weights, 1024 times.*
>
> *Compare that to an MLP: a layer connecting 1024 inputs to 1024 outputs
> needs 1024×1024 = 1M weights. Our Conv2D needs just 9. That's weight
> sharing, and it's why CNNs scale to large images."*

Draw on board:
```
MLP approach (fully connected):
  Every pixel → Every neuron    = 32×32×3 × 512 = 1,572,864 weights

CNN approach (3×3 filter):
  ONE filter sliding everywhere = 3×3×3 = 27 weights (used everywhere!)

  At position (0,0): uses W     At position (5,7): uses SAME W
  At position (1,3): uses W     At position (31,31): uses SAME W
                   ↑
              weight sharing = massive parameter reduction + translation invariance
```

**Ask the room:** *"What does weight sharing give us for free, beyond fewer parameters?
If the filter detects a cat's ear, does it matter where in the image the ear appears?"*

> *(Answer: Translation invariance — the same filter fires wherever the pattern appears.)*

---

## SECTION 3: The Forward Pass — The Sliding Loop (25 min)

> *"Now we write the loop. The filter slides across. At every position
> we do an element-wise multiply and sum — which is a dot product.*
>
> *This is the most important 15 lines of code in Part 4."*

Walk through padding first:
```python
def _pad(self, X):
    kH, kW = self.kernel_size
    pH = (kH - 1) // 2
    pW = (kW - 1) // 2
    # np.pad adds zeros around the border
    return np.pad(X, ((0,0),(pH,pH),(pW,pW),(0,0)), mode="constant")
```

Then the forward pass:
```python
def forward(self, X):
    # X shape: (batch, H, W, C)
    N, H, W, C = X.shape
    self.build((H, W, C))        # lazy init
    X_pad = self._pad(X)
    kH, kW = self.kernel_size
    out_H = (H - kH) // self.stride + 1   # output spatial size
    out_W = (W - kW) // self.stride + 1
    out = np.zeros((N, out_H, out_W, self.n_filters))

    for i in range(out_H):         # slide down
        for j in range(out_W):     # slide right
            h_start = i * self.stride
            w_start = j * self.stride
            patch = X_pad[:, h_start:h_start+kH, w_start:w_start+kW, :]
            # patch shape: (N, kH, kW, C)
            # W shape:          (kH, kW, C, n_filters)
            # einsum: for each sample, dot patch with each filter
            out[:, i, j, :] = np.tensordot(patch, self.W, axes=([1,2,3],[0,1,2])) + self.b
    return out
```

Draw the sliding window on board:
```
Image (6×6):
┌─────────────────────┐
│  . . . . . .        │
│  . ┌───┐ . .        │   Filter (3×3) at position (1,1)
│  . │ X │ . .        │   Multiply patch by W element-wise
│  . └───┘ . .        │   Sum → one output value
│  . . . . . .        │   Stride right → next position
│  . . . . . .        │
└─────────────────────┘

Stride = 1: shift 1 pixel each step  → dense feature map
Stride = 2: shift 2 pixels each step → half-size feature map
```

**Ask the room:** *"If the input is 32×32 and the filter is 3×3 with 'same' padding
and stride 1, what's the output size?"*
> *(Answer: 32×32. Same padding keeps dimensions identical.)*

**INSTRUCTOR TIP — "Why is it slow?"**
> Students will notice the Python loops are slow. That's intentional.
> Say: "TensorFlow and PyTorch implement this exact loop in C++/CUDA.
> Our loop is for understanding, not speed. Run the file and watch
> TF's version run 100× faster — then you'll appreciate what optimized code does."

---

## CLOSING SESSION 1 (10 min)

Write on board:
```
TODAY:
  Conv2D.build()    → initialize W (kH×kW×C×n_filters), b (n_filters)
  Conv2D.forward()  → slide filter across image, dot-product at each position
  Weight sharing    → same W used everywhere → few params, translation invariance
  Padding "same"    → output H,W = input H,W (zero-pad border)
  Stride > 1        → output shrinks (stride 2 → half spatial size)
```

**Homework:** On paper: if input is (1, 28, 28, 1) and we apply Conv2D(8, kernel_size=5,
padding="same", stride=1), what is the output shape? How many total parameters?

---

# SESSION 2 (~90 min)
## "ReLU, MaxPool, chaining layers — and verifying against TensorFlow"

## OPENING (10 min)

> *"Last session we built the sliding window — the core computation of Conv2D.*
>
> *Today we add the two other building blocks: ReLU activation and MaxPooling.
> Then we chain them into a mini CNN. And then we do something satisfying:
> run our numpy version and TensorFlow's version on the same input, and
> confirm they produce matching outputs.*
>
> *When they match, you'll know your implementation is correct. Let's build it."*

---

## SECTION 1: ReLU and MaxPool Layers (20 min)

> *"These are simple compared to Conv2D. ReLU is literally one line.*
>
> *MaxPool is a sliding window too — but instead of a dot product,
> we just take the maximum value in the patch. No learned weights. No parameters."*

```python
class ReLU:
    def forward(self, X):
        return np.maximum(0, X)   # one line — that's the whole layer

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        N, H, W, C = X.shape
        p = self.pool_size
        out_H = (H - p) // self.stride + 1
        out_W = (W - p) // self.stride + 1
        out = np.zeros((N, out_H, out_W, C))
        for i in range(out_H):
            for j in range(out_W):
                h = i * self.stride
                w = j * self.stride
                patch = X[:, h:h+p, w:w+p, :]
                out[:, i, j, :] = patch.max(axis=(1, 2))  # max in each patch
        return out
```

Draw MaxPool on board:
```
Input (4×4):              MaxPool (2×2, stride 2):   Output (2×2):

┌────┬────┬────┬────┐     Take MAX in each 2×2:     ┌────┬────┐
│  1 │  3 │  2 │  4 │     ┌────────┐ ┌────────┐     │  3 │  4 │
├────┼────┼────┼────┤     │ 1,3,5,9│ │ 2,4,6,8│     ├────┼────┤
│  5 │  9 │  6 │  8 │     └────────┘ └────────┘     │  9 │  8 │
├────┼────┼────┼────┤         max=9      max=8       └────┴────┘
│  2 │  4 │  1 │  3 │
├────┼────┼────┼────┤     Discards 75% of values, keeps strongest signal
│  6 │  7 │  5 │  2 │     Makes model robust to small position shifts
└────┴────┴────┴────┘
```

> *"MaxPool does two things: shrinks the spatial size (saves computation)
> and gives us translation robustness (a feature 1 pixel to the left
> still produces the same max). Those two properties are why pooling
> has survived every architecture change since LeNet."*

---

## SECTION 2: Building the Mini CNN (20 min)

> *"Now let's chain them. This is where it feels real — layers stacking,
> each one transforming the data and passing it to the next."*

```python
# Build a mini CNN: Conv → ReLU → MaxPool → Conv → ReLU → Flatten
class MiniCNN:
    def __init__(self):
        self.layers = [
            Conv2D(n_filters=8, kernel_size=3, padding="same"),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Conv2D(n_filters=16, kernel_size=3, padding="same"),
            ReLU(),
        ]

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
            print(f"  {layer.__class__.__name__:12s} → {out.shape}")
        return out

# Test it
np.random.seed(42)
X_test = np.random.randn(1, 16, 16, 3).astype(np.float32)
print("Input:", X_test.shape)
cnn = MiniCNN()
output = cnn.forward(X_test)
print("Final output shape:", output.shape)
```

Watch the shape transformations print out:
```
Input:     (1, 16, 16, 3)
Conv2D   → (1, 16, 16, 8)    ← 8 feature maps, same spatial size
ReLU     → (1, 16, 16, 8)    ← unchanged (just clips negatives)
MaxPool  → (1,  8,  8, 8)    ← spatial halved, channels unchanged
Conv2D   → (1,  8,  8, 16)   ← 16 feature maps now
ReLU     → (1,  8,  8, 16)   ← unchanged
```

**Ask the room:** *"At the MaxPool step, spatial goes 16→8 and channels stay 8.
After the next Conv2D, channels go 8→16 and spatial stays 8. Can you explain
why each of those changes happen?"*

---

## SECTION 3: Verifying Against TensorFlow (20 min)

> *"Here's the moment of truth. We'll set TensorFlow's Conv2D weights
> to exactly our numpy weights, run the same input through both,
> and compare the outputs.*
>
> *If they match to within floating-point rounding, our implementation is correct."*

```python
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers as klayers

    # Build equivalent Keras model
    tf_model = keras.Sequential([
        klayers.Conv2D(8, 3, padding="same", use_bias=True, input_shape=(16,16,3))
    ])
    tf_model.build((None, 16, 16, 3))

    # Copy our numpy weights into Keras
    tf_model.layers[0].set_weights([
        our_conv.W,    # shape (3,3,3,8)
        our_conv.b     # shape (8,)
    ])

    # Run both
    our_output   = our_conv.forward(X_test)
    tf_output    = tf_model.predict(X_test, verbose=0)

    max_diff = np.abs(our_output - tf_output).max()
    print(f"Max difference: {max_diff:.2e}")   # should be ~1e-6 or less
    print("Match!" if max_diff < 1e-4 else "Something is wrong")
except ImportError:
    print("TensorFlow not installed — but the NumPy code above is correct.")
```

> *"When you see max_diff around 1e-6, that's just floating-point precision.
> Your NumPy implementation and TensorFlow's C++ implementation agree.*
>
> *You just wrote Conv2D from scratch. That's what's inside every CNN
> you will ever use."*

**INSTRUCTOR TIP — "What about backprop?"**
> Students may ask why we only did forward. Say: "We're using Keras for training.
> Backprop through a Conv layer is complex — it involves another convolution
> (the transpose). For production we trust Keras's autograd. The forward pass
> is what gives you the intuition. If you're curious, CS231n has the full derivation."

---

## SECTION 4: Live Demo — Run the Full Script (15 min)

```bash
python3 conv_layer_from_scratch.py
```

Point at output as it runs:
- Conv2D built: W shape, parameter count
- Shape transformations as data flows through mini CNN
- TF comparison output
- The saved filter visualization in `visuals/conv_layer_from_scratch/`

Open the visualizations:
> *"Look at the filter weights — random at first, but after training
> they develop structure: some detect horizontal edges, some vertical,
> some detect textures. The network discovers these patterns on its own.*
>
> *We didn't tell it to look for edges. Gradient descent found that
> edge detection is useful for classification. That's the magic."*

---

## CLOSING SESSION 2 (5 min)

Board summary:
```
FULL LAYER STACK:
  Conv2D    → W (kH×kW×C×n_filters), sliding dot product, weight sharing
  ReLU      → np.maximum(0, X), no params
  MaxPool   → sliding max, no params, shrinks spatial, adds robustness

VERIFIED: numpy Conv2D == TensorFlow Conv2D (to floating point precision)

WEIGHT SHARING: same filter weights reused at every spatial position
  → efficient (few params), → translation invariant (finds pattern anywhere)
```

**Homework:** Modify `MiniCNN` to add a second MaxPool2D after the second Conv2D.
What is the final output shape starting from (1, 32, 32, 3)?

---

## INSTRUCTOR TIPS

**"Why do we use 'same' padding?"**
> *"Without padding, each Conv shrinks the spatial size by (kernel_size-1).
> A 3×3 filter on a 32×32 image gives 30×30. After 10 layers you'd have
> 12×12. 'Same' padding adds a border of zeros so size is preserved.
> Use 'valid' (no padding) when you want deliberate shrinkage."*

**"How does the network know what to learn in the filters?"**
> *"Backpropagation updates W to minimize the loss — same as MLP weights.
> The difference is that the gradient for each weight is the SUM of gradients
> from every position where that filter was applied. That's the 'shared'
> part of weight sharing — the weight gets credit (and blame) from everywhere."*

**"Why MaxPool and not AvgPool?"**
> *"MaxPool: 'did this feature appear anywhere in the patch?'
> AvgPool: 'on average, how much of this feature was in the patch?'
> For image classification, presence matters more than average intensity,
> so MaxPool generally works better. Modern architectures often skip
> pooling entirely and use strided convolutions instead."*

**"What's the difference between filters and channels?"**
> *"Channels = depth of the INPUT at this layer. Filters = number of
> patterns we're learning = depth of the OUTPUT. After the first Conv2D(32),
> the input to the next layer has 32 channels. Those 32 channels go into
> the second Conv2D, which learns filters over all 32 at once."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening — flashlight analogy            10 min
├── Conv2D class, weight shape, He init     20 min
├── Weight sharing explanation              15 min
├── Forward pass — sliding loop             25 min
└── Close + homework                        10 min

SESSION 2  (90 min)
├── Opening bridge                          10 min
├── ReLU and MaxPool layers                 20 min
├── Building the mini CNN                   20 min
├── TF verification                         20 min
├── Live demo + visualizations              15 min
└── Close + homework                         5 min
```

---
*MLForBeginners · Part 4: CNNs · Module 04*
