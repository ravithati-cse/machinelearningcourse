# MLForBeginners — Instructor Guide
## Part 4 · Module 06: Classic CNN Architectures
### Two-Session Teaching Script

> **Prerequisites:** Module 05 complete. They can build and train a CNN in Keras,
> understand BatchNorm, Dropout, feature maps, and EarlyStopping. They are now
> ready to see the progression of ideas that shaped the entire field — from a
> 1998 bank check reader to the network that solved ImageNet in 2015.
> **Payoff today:** They will understand WHY deep networks stopped working at
> 20+ layers, and WHY skip connections fixed it. They will draw a ResNet block
> on the board themselves.

---

# SESSION 1 (~90 min)
## "The race to go deeper — LeNet to VGG, and why deep networks break"

## Before They Arrive
- Terminal open in `convolutional_neural_networks/algorithms/`
- Whiteboard with a rough timeline drawn: 1998 ... 2012 ... 2014 ... 2015
- `classic_architectures.py` open but not yet run
- Optional: print or show an image of AlexNet's original paper figure

---

## OPENING (10 min)

> *"Here's a question: if CNNs work, does a deeper CNN work better?
> More layers, more features, more powerful. Should be yes, right?*
>
> *For a while, people thought so. VGGNet in 2014 had 19 layers.
> It was better than AlexNet's 8. So researchers kept going deeper.*
>
> *But then something strange happened. At 20, 30, 50 layers, training
> started BREAKING. Not just slow — broken. Adding more layers made
> accuracy WORSE, even on the training data.*
>
> *This was terrifying. You'd expect a deeper model to at least overfit —
> to memorize training data even if it doesn't generalize. But it was
> failing even at memorization.*
>
> *Today we tell the story of how this problem was discovered, and
> how one elegant idea — skip connections — solved it completely.
> That idea is called ResNet, and it changed everything."*

Draw the timeline:
```
1998          2012          2014          2015          2017
 │             │             │             │             │
LeNet-5       AlexNet       VGGNet       ResNet-50    MobileNet
7 layers      8 layers      19 layers    50 layers    lightweight
60K params    60M params    138M params  25M params   4M params
Digits        ImageNet 1st  ImageNet 2nd ImageNet 1st Mobile apps
              (15% err)     (6.8% err)   (3.6% err)
              ← deep learning era begins here
```

---

## SECTION 1: LeNet-5 — The Original CNN (20 min)

> *"Yann LeCun, 1998. The first practical CNN. Banks used it to read
> handwritten digits on checks. It processed millions of checks per day.*
>
> *By modern standards it's tiny. But it established the template:
> alternating convolutions and pooling, ending in fully-connected layers.
> Every CNN since follows this same pattern."*

Draw on board:
```
LeNet-5 Architecture (1998):

Input         Conv1          Pool1          Conv2          Pool2         FC1      FC2    Output
(32×32×1)  → (28×28×6)   → (14×14×6)   → (10×10×16) → (5×5×16)  → 120   → 84  → 10

Filter: 5×5                 Filter: 5×5
Act: tanh                   Act: tanh
Pooling: avg                Pooling: avg

Total params: ~60,000
```

Build it in Keras:
```python
lenet = keras.Sequential([
    # Input is CIFAR-10 (32×32×3), original LeNet used 32×32×1
    layers.Conv2D(6, (5,5), activation="tanh", padding="same",
                  input_shape=(32,32,3)),
    layers.AveragePooling2D((2,2)),
    layers.Conv2D(16, (5,5), activation="tanh"),
    layers.AveragePooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(120, activation="tanh"),
    layers.Dense(84,  activation="tanh"),
    layers.Dense(10,  activation="softmax"),
], name="LeNet5")
lenet.summary()
```

> *"Notice: no BatchNorm, no Dropout — those didn't exist yet.
> Tanh instead of ReLU (ReLU wasn't standard until AlexNet).*
>
> *On CIFAR-10 this gets about 55-60% accuracy. Fine for 1998 digit
> recognition on clean grayscale images. Not enough for color photos."*

**Ask the room:** *"If this model gets 60% on CIFAR-10 today, and MLP gets 55%,
what does that tell us about what even a small CNN learns?"*

---

## SECTION 2: AlexNet — The Moment Everything Changed (20 min)

> *"AlexNet, 2012. This is the most important year in modern AI.*
>
> *Until 2012, the best ImageNet systems used hand-crafted features:
> human experts decided what to look for. Error rate: 25-26%.*
>
> *AlexNet used CNNs trained on two GTX 580 GPUs for a week.
> Error rate: 15.3%. The second place was 26%. That's not a small improvement
> — it was a different era. Everyone scrambled to understand what happened.*
>
> *Four innovations AlexNet brought to the world:*"

Write on board:
```
AlexNet Innovations (2012):

1. ReLU instead of tanh/sigmoid
   sigmoid/tanh: gradients saturate (→ 0) for large inputs
   ReLU = max(0, x): gradient is always 1 for positive inputs → trains FASTER

2. Dropout (0.5)
   Randomly zero out 50% of neurons during training
   Forces redundancy — network can't rely on any single neuron
   MASSIVE regularization effect for large networks

3. Data augmentation
   Random crops, horizontal flips during training
   Effectively multiplies dataset size by ~2,048

4. GPU training (two GPUs in parallel)
   Made training 60M parameters feasible
   Deep learning is now compute-bound (this is when GPU companies changed)
```

Simplified Keras implementation:
```python
alexnet_simple = keras.Sequential([
    layers.Conv2D(96, (3,3), strides=1, activation="relu", padding="same",
                  input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(256, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(384, (3,3), activation="relu", padding="same"),
    layers.Conv2D(384, (3,3), activation="relu", padding="same"),
    layers.Conv2D(256, (3,3), activation="relu", padding="same"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
], name="AlexNet_simplified")
```

> *"This is 60M parameters. In 1998, that would have taken years to train.
> In 2012, two GPUs, one week. In 2025 on a laptop: a few hours.*
>
> *That's hardware progress, but AlexNet showed the world the algorithm
> was right. Everyone switched to deep learning immediately."*

---

## SECTION 3: VGGNet — Simple, Deep, Everywhere (15 min)

> *"VGGNet (2014) asked: what if instead of large filters (5×5, 11×11),
> we ONLY use 3×3 filters, but go very deep?*
>
> *Mathematical insight: two 3×3 conv layers have the same receptive field
> as one 5×5 layer, but fewer parameters and more non-linearities."*

Draw on board:
```
One 5×5 conv layer:         Two 3×3 conv layers:
  params = 5×5×C×F            params = 2 × (3×3×C×F)
                                       = 2/25 × 5×5 version... wait...

Let's be precise:
  5×5 single layer:     25 × C × F parameters
  3×3 × 2 layers:       18 × C × F parameters   ← FEWER params, same receptive field

VGG-16 uses only 3×3 convolutions throughout.
Very deep (16-19 weight layers), very regular (same block pattern repeated).
```

> *"VGG was easy to understand, easy to implement, easy to extend.
> It became the default 'baseline' architecture for years.*
>
> *But it has 138 million parameters. Mostly in the fully-connected
> layers at the end. That made it slow and memory-hungry.*
>
> *And going beyond 20 layers didn't help — accuracy actually got worse.
> Nobody knew why. This mystery is what ResNet solved."*

---

## CLOSING SESSION 1 (10 min)

Board summary:
```
1998: LeNet   → template (conv → pool → FC), tanh, ~60K params
2012: AlexNet → ReLU, Dropout, GPU, ~60M params, cracked ImageNet
2014: VGGNet  → only 3×3 filters, very deep (19 layers), ~138M params

THE PROBLEM: deeper than ~20 layers → accuracy degrades even on training data
             This is NOT overfitting. Something else is wrong.
             → Session 2: what's wrong, and how ResNet fixed it.
```

**Homework:** If you stack 20 identity layers (each outputs exactly its input),
what should the training accuracy be compared to 0 layers? Why would adding
these "neutral" layers ever make things worse?

---

# SESSION 2 (~90 min)
## "The vanishing gradient and how ResNet's skip connections fix everything"

## OPENING (10 min)

> *"The homework question: adding identity layers should never hurt.
> The model can always learn to make those layers do nothing.*
>
> *But empirically, 56-layer models on CIFAR-10 performed WORSE than 20-layer
> models on training data. The deep model was doing worse at memorizing —
> that's a fundamental training failure, not overfitting.*
>
> *The culprit: gradients. The signal that flows backward through the network
> gets smaller and smaller as it passes through each layer. By the time
> it reaches Layer 1 in a 56-layer network, the gradient is so small
> the weights there barely update. The early layers stop learning.*
>
> *ResNet's solution is one of the most elegant ideas in ML:
> give the gradient a shortcut. Let it skip layers."*

---

## SECTION 1: Vanishing Gradients — The Problem Made Concrete (20 min)

> *"Backpropagation: gradients flow backward. At each layer, the gradient
> gets multiplied by the layer's weights. If those weights are small,
> gradients shrink. If they're large, gradients explode.*
>
> *Even with careful initialization and ReLU, in a 50-layer network,
> gradients multiplied 50 times can become negligibly small."*

Write on board:
```
Gradient at layer L:     ∂L/∂W_L = ∂L/∂y × ... × ∂y_L/∂W_L

In a 50-layer network:
  ∂L/∂W_1 = ∂L/∂y × (∂y_50/∂y_49) × ... × (∂y_2/∂y_1) × ∂y_1/∂W_1
                       ↑              ↑
              50 terms being multiplied together

If each term ≈ 0.9:    0.9^50 ≈ 0.005   ← gradient at Layer 1 is tiny
If each term ≈ 0.8:    0.8^50 ≈ 0.000014 ← practically zero

Layer 1 receives gradient = 0.000014 × original gradient
It updates by: 0.000014 × learning_rate × gradient ≈ basically zero
Layer 1 stops learning.
```

Draw a deep network with gradient intensity:
```
Layer:  50  49  48  ...  10   9   8  ...   3   2   1
Gradient:
        ███ ██  █        ▌    ▌   ░       ·   ·   ·

  ↑ strong                              nearly zero ↓
  gradient                                 gradient

Early layers learn features → but they don't update → network stuck
```

**Ask the room:** *"BatchNorm was supposed to help training stability.
Why doesn't it fully solve this problem?"*

> *(Answer: BatchNorm normalizes activations in the forward pass, which helps
> with internal covariate shift. But it doesn't directly fix the vanishing
> gradient in the backward pass — the chain rule multiplication still happens.)*

---

## SECTION 2: The Residual Block — The Key Insight (25 min)

> *"Here's the idea. Instead of asking a layer to learn the full transformation
> F(x), ask it to learn only the CHANGE — the residual — that needs to be
> added to the input.*
>
> *Then explicitly add the input back: output = F(x) + x.*
>
> *This is called a skip connection (or shortcut connection).*
>
> *Why does this help? Because if the layer is useless, it can learn F(x) = 0,
> and the output is just x. An identity transformation is trivial to learn.*
>
> *But more importantly: the gradient now has TWO paths backward.
> It flows through F(x), AND it flows directly through the + x path.*
> The direct path has gradient = 1. Layer 1 gets a clean signal."*

**Draw this on the board slowly. This is the most important diagram of the session.**

```
REGULAR LAYER:                    RESIDUAL BLOCK:

    x                                   x
    │                                   │────────────────┐
    ▼                                   ▼                │  ← skip connection
┌───────┐                           ┌───────┐           │  (identity shortcut)
│  F(x) │                           │  Conv │           │
│  Conv │                           │  BN   │           │
│  ReLU │                           │  ReLU │           │
└───────┘                           │  Conv │           │
    │                               │  BN   │           │
    ▼                               └───────┘           │
   out = F(x)                           │  F(x)         │
                                        ▼               │
                                    ┌───────┐           │
                                    │   +   │ ←─────────┘  F(x) + x
                                    └───────┘
                                        │
                                       ReLU
                                        │
                                       out = ReLU(F(x) + x)

Gradient backward:
  ∂loss/∂x = ∂loss/∂out × (∂F(x)/∂x + 1)   ← the +1 keeps gradient alive!
              Even if ∂F(x)/∂x ≈ 0, gradient = ∂loss/∂out × 1 = clean signal
```

> *"The +1 in the gradient. That's it. That's the whole insight.*
>
> *Even if the convolutional layers contribute nothing (F(x)=0),
> the gradient still flows. Early layers can always learn.*
>
> *ResNet-152 has 152 layers. It trains fine because every 2 layers
> has a skip connection handing gradient back directly.*
>
> *This won ImageNet 2015 with 3.6% error — better than humans (5% error)."*

---

## SECTION 3: Build a Residual Block in Keras (20 min)

```python
def residual_block(x, filters, stride=1):
    """
    Standard ResNet residual block.
    x: input tensor
    filters: number of conv filters
    stride: for downsampling (stride=2 halves spatial size)
    """
    shortcut = x   # save input for later addition

    # Main path
    x = layers.Conv2D(filters, (3,3), strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, (3,3), strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)

    # Projection shortcut: if spatial or channel size changed, we must
    # reshape the shortcut to match
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1,1), strides=stride)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add skip connection, then activate
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x
```

> *"One subtlety: if the stride is 2 or the number of channels changes,
> x and shortcut have different shapes — you can't just add them.*
>
> *The fix: a 1×1 convolution on the shortcut to match the shape.
> Called a 'projection shortcut.' It just adjusts dimensions, no real feature extraction."*

Build a small ResNet for CIFAR-10:
```python
def build_resnet_small(input_shape=(32,32,3), n_classes=10):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3,3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = residual_block(x, 32)          # 32×32×32
    x = residual_block(x, 64, stride=2) # 16×16×64
    x = residual_block(x, 128, stride=2) # 8×8×128

    x = layers.GlobalAveragePooling2D()(x)  # 128 (no Flatten needed)
    x = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, x, name="ResNet_small")

model = build_resnet_small()
model.summary()
```

> *"GlobalAveragePooling2D — a modern alternative to Flatten + Dense.
> Instead of flattening (8×8×128 = 8,192 values → big Dense layer),
> we average each feature map spatially → 128 values. Fewer parameters,
> less overfitting."*

---

## SECTION 4: Compare All Four Architectures (10 min)

Run the script and observe the comparison output:

```bash
python3 classic_architectures.py
```

Draw the comparison on board:
```
Architecture  Year  Layers  Params   CIFAR-10 Acc   Key Innovation
────────────────────────────────────────────────────────────────────
LeNet-5       1998   7       60K       ~55-60%       First practical CNN
AlexNet       2012   8       60M       ~75-80%       ReLU, Dropout, GPU
VGGNet        2014   16      138M      ~88-90%       Only 3×3 filters
ResNet-small  2015   ~10     ~500K     ~88-90%       Skip connections

Key observation: ResNet-small matches VGG accuracy with 1/276 the parameters
```

**Ask the room:** *"ResNet uses 500K parameters to match VGG's 138M parameters.
What does that tell us about VGG?"*
> *(Answer: VGG was massively over-parameterized, especially in its FC layers.
> Most parameters weren't doing useful work. ResNet showed efficient architecture
> matters more than raw parameter count.)*

---

## CLOSING SESSION 2 (5 min)

Board summary:
```
THE STORY:
  Deep networks broke because gradients vanished in early layers.
  Adding layers hurt because those layers couldn't learn — no gradient signal.

THE FIX (ResNet, 2015):
  output = ReLU(F(x) + x)      ← skip connection
  gradient has a direct highway back through the +x path
  gradient = ∂F/∂x + 1         ← always at least 1, never vanishes

ENABLES:
  ResNet-152: 152 layers, trains fine
  Better than humans on ImageNet (3.6% vs 5% error)
  Architecture still used everywhere today
```

**Homework:** Add one more residual_block with stride=2 to `build_resnet_small`.
What is the new final spatial size before GlobalAveragePooling?
Train for 20 epochs and compare accuracy to the 3-block version.

---

## INSTRUCTOR TIPS

**"Why didn't BatchNorm fix the vanishing gradient problem?"**
> *"BatchNorm normalizes activations in the FORWARD pass.
> Vanishing gradients are a BACKWARD pass problem.
> They're different problems. BatchNorm helps training stability
> generally, but doesn't give gradients a direct highway home
> the way skip connections do."*

**"Can we add skip connections to VGGNet and get ResNet?"**
> *"Essentially yes. VGGNet with skip connections every 2 layers
> is very close to ResNet. The VGG authors just didn't add them.
> ResNet's insight was realizing the identity shortcut was all you needed."*

**"What's GlobalAveragePooling vs Flatten?"**
> *"Flatten: 8×8×128 → 8192 values → Dense(512) needs 8192×512 = 4M params.
> GlobalAvgPool: average each 8×8 map → 128 values → Dense(10) needs 1,280 params.
> GlobalAvgPool reduces overfitting and speeds up training significantly.
> Modern architectures almost always use it."*

**"When would you use VGG in 2025?"**
> *"Mostly never from scratch. But pretrained VGG16 and VGG19 are still
> in many transfer learning workflows because their feature layers are
> well-understood and widely benchmarked. For new projects, prefer
> ResNet or EfficientNet."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening — the deep network mystery        10 min
├── LeNet-5: build and understand             20 min
├── AlexNet: four key innovations             20 min
├── VGGNet: only 3×3, going deep              15 min
└── Close + homework                          10 min   (+ 15 min buffer)

SESSION 2  (90 min)
├── Opening — vanishing gradient defined      10 min
├── Gradient math — why it vanishes           20 min
├── ResNet skip connection diagram            25 min
├── Build residual block in Keras             20 min
├── Compare all four architectures            10 min
└── Close + homework                           5 min
```

---
*MLForBeginners · Part 4: CNNs · Module 06*
