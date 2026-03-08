# MLForBeginners — Instructor Guide
## Part 3 · Module 07: Multi-Layer Perceptron from Scratch
### Two-Session Teaching Script

> **Prerequisites:** Modules 01–06 complete. They know neurons, forward propagation,
> backpropagation, the perceptron, and why XOR defeats a single layer. They can
> read and write basic numpy classes.
> **Payoff today:** They will build a complete neural network — forward pass,
> backward pass, mini-batch training with Adam — entirely in numpy. Then watch
> it solve XOR and create complex decision boundaries.

---

# SESSION 1 (~90 min)
## "Building the network — architecture and forward pass"

## Before They Arrive
- Terminal open in `deep_neural_networks/algorithms/`
- Whiteboard ready
- Draw the XOR diagram from Module 06 — we're solving it today
- Have a fresh numpy import ready in a Python REPL

---

## OPENING (10 min)

> *"Last session, the perceptron failed at XOR. It could only draw one line.*
>
> *Today we fix it. We're building a Multi-Layer Perceptron — the real thing.*
> *Not Keras, not sklearn. Every matrix multiplication, every activation, every
> gradient — we write it in numpy.*
>
> *When we run it at the end of session two, it will solve XOR.*
> *And then we'll throw it at a real 2D classification problem and
> watch it draw curves instead of lines.*
>
> *This is the most important session in Part 3.
> After today, you will understand what every neural network framework
> in the world is doing under the hood."*

Draw on board:
```
TODAY WE BUILD:

  x₁ ─────→ [h₁] [h₂] [h₃]  (hidden layer 1)
                   ↓
  x₂ ─────→ [h₁] [h₂] [h₃]  (hidden layer 2)
                   ↓
                 [o₁] [o₂]   (output layer)
                   ↓
               prediction

Forward pass → make prediction
Backward pass → compute gradients
Adam update → adjust weights
```

---

## SECTION 1: Architecture and Helper Functions (15 min)

Write on board:
```
NETWORK ARCHITECTURE:
  layers = [input_size, hidden1, hidden2, ..., output_size]
  Example: [2, 4, 4, 2] means:
    - 2 input features
    - 4 neurons in hidden layer 1
    - 4 neurons in hidden layer 2
    - 2 output classes (softmax)

ACTIVATION FUNCTIONS:
  Hidden layers: ReLU (fast, no vanishing gradient)
  Output layer: Softmax (multi-class probabilities)
```

Write the helper functions on board:
```python
def relu(z):
    return np.maximum(0, z)   # like a light switch — negative in, zero out

def relu_grad(z):
    return (z > 0).astype(float)   # 1 if z>0, else 0

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))  # numerically stable
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    # y_true = integer class labels [0, 1, 2, ...]
    m = y_true.shape[0]
    p = y_pred[range(m), y_true]
    return -np.mean(np.log(p + 1e-15))
```

> *"ReLU is the most popular hidden layer activation today.*
> *Think of each neuron as a light switch that only turns on for positive values.*
> *Signals below zero are suppressed. The network learns which switches to flip.*
>
> *Softmax converts the output layer into probabilities that sum to 1.*
> *If there are 10 output neurons, each one represents the probability of one class."*

**Ask the room:** *"Why subtract z.max() in the softmax? What goes wrong if we don't?"*

> (Numerical stability — if z contains large numbers like 1000, e^1000 overflows to infinity.
> Subtracting the max changes the answer for individual exponentials but not the ratio,
> because the max cancels in numerator and denominator.)

---

## SECTION 2: Weight Initialization (15 min)

Write on board:
```
WEIGHT INITIALIZATION MATTERS:

  All zeros?   → all neurons compute the same thing → symmetry problem
               neurons never differentiate from each other

  Too large?   → activations saturate, gradients vanish

  Just right?  → He initialization for ReLU:
                 W = np.random.randn(in, out) × sqrt(2/in)
```

Draw the symmetry breaking problem:
```
If all weights = 0:
  Layer output = activation(X @ 0 + 0) = activation(0) for every neuron
  All neurons identical → gradients identical → weights stay identical
  Network can never learn different features

He initialization:
  Random ≠ 0 → neurons start different → can specialize
  Scale = sqrt(2/fan_in) → keeps variance stable through ReLU layers
```

> *"The scale factor sqrt(2/fan_in) is magical. Each layer's activations
> come out with approximately unit variance — not shrinking, not exploding.
> This comes from a 2015 paper by Kaiming He.*
>
> *Without it: a 10-layer network's signals shrink to nothing or explode to infinity.*
> *With it: signals stay in a healthy range all the way through."*

**Code to write on board:**
```python
def initialize_weights(self, layer_sizes):
    self.W = []
    self.b = []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        W = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
        b = np.zeros((1, fan_out))
        self.W.append(W)
        self.b.append(b)
```

---

## SECTION 3: The Forward Pass (20 min)

Write on board:
```
FORWARD PASS — one layer at a time:

  Z[l] = A[l-1] @ W[l] + b[l]     (linear combination)
  A[l] = relu(Z[l])                 (activation)

  For the output layer:
  Z[L] = A[L-1] @ W[L] + b[L]
  A[L] = softmax(Z[L])             (probabilities)

  We SAVE Z and A for every layer — we need them in backprop
```

Draw the data flow:
```
X (batch, features)
  │
  ▼
Z[0] = X @ W[0] + b[0]    → shape: (batch, hidden1)
  │
A[0] = relu(Z[0])          → shape: (batch, hidden1)
  │
Z[1] = A[0] @ W[1] + b[1] → shape: (batch, hidden2)
  │
A[1] = relu(Z[1])          → shape: (batch, hidden2)
  │
Z[2] = A[1] @ W[2] + b[2] → shape: (batch, n_classes)
  │
A[2] = softmax(Z[2])       → shape: (batch, n_classes) — probabilities
```

> *"Every step is matrix multiplication — X @ W.*
> *A batch of 32 examples goes through all at once.*
> *The shapes tell you everything is correct.*
>
> *We store every Z and A because backprop needs to replay this computation
> in reverse order to compute gradients."*

**Ask the room:** *"If our batch has 32 examples and layer 1 has 64 neurons,
what is the shape of A[0]?"*

> ((32, 64) — 32 examples, each with 64 neuron activations)

---

## CLOSING SESSION 1 (10 min)

Board summary:
```
ARCHITECTURE:    [n_features, hidden1, hidden2, n_classes]
ACTIVATIONS:     relu (hidden), softmax (output)
INITIALIZATION:  He — W = randn × sqrt(2/fan_in)
FORWARD PASS:
  Z = A_prev @ W + b
  A = activation(Z)
  Store all Z, A for backprop
```

**Homework:** Compute the forward pass manually for this tiny network:
```
W = [[0.1, 0.2], [0.3, 0.4]], b = [0, 0]
x = [1.0, 2.0]
z = x @ W + b  → what is z?
a = relu(z)    → what is a?
```

---

# SESSION 2 (~90 min)
## "Backprop, Adam, and watching the MLP solve XOR"

## OPENING (10 min)

> *"You built the forward pass. The network can make predictions — random ones
> since the weights are random, but predictions.*
>
> *Now we need it to learn. That means computing how wrong it was
> and adjusting every weight accordingly.*
>
> *This is backpropagation — flowing the error signal backwards through
> every layer so every weight knows its responsibility.*
>
> *Then we use Adam to update the weights smarter than plain gradient descent.*
> *Run it for a few hundred epochs, and watch XOR go from 50% to 100%."*

---

## SECTION 1: The Backward Pass (30 min)

Write on board:
```
BACKPROP — working backwards from the loss:

Output layer:
  dZ[L] = A[L] - Y_onehot      (softmax + cross-entropy → clean gradient)
  dW[L] = A[L-1].T @ dZ[L] / m
  db[L] = dZ[L].mean(axis=0)

Hidden layers (chain rule backward):
  dA[l] = dZ[l+1] @ W[l+1].T         (gradient flows back through W)
  dZ[l] = dA[l] × relu_grad(Z[l])    (element-wise: gate the gradient)
  dW[l] = A[l-1].T @ dZ[l] / m
  db[l] = dZ[l].mean(axis=0)
```

Draw the gradient flow:
```
Loss ← dZ[L] ← dZ[L-1] ← dZ[L-2] ← ... ← dZ[0]
              ↑            ↑                  ↑
            dW[L]        dW[L-1]           dW[0]

Each arrow is a matrix multiply.
The relu_grad gate zeroes out the gradient where activation was negative.
```

> *"The output layer gradient is the cleanest part:*
> *dZ = predictions - true_labels. That's it.*
> *How far off were our probabilities from the correct one-hot target.*
>
> *For hidden layers, we use the chain rule:*
> *dA = gradient coming back from the next layer.*
> *dZ = dA × relu_grad — the ReLU gate: neurons that were off during forward*
> *pass contribute zero gradient. Their weights don't update.*
>
> *This flows all the way back to W[0] and b[0] — the first layer."*

**Ask the room:** *"In the relu_grad, we multiply by (Z > 0). Why does that gate the gradient?"*

> (If the neuron was off during forward pass, z < 0. relu_grad = 0 there.
> Multiplying by 0 blocks the gradient — no update for that neuron.
> This makes sense: if the neuron was silent, it didn't affect the output,
> so its gradient should be zero.)

**Common confusion — what to say:**

> *"If someone asks 'where does the 1/m come from in dW':*
> *We're averaging gradients across the mini-batch. If the batch has 32 examples,*
> *each one contributes a gradient. We average them so the weight update*
> *size doesn't depend on batch size — it stays consistent."*

---

## SECTION 2: Adam Optimizer (15 min)

Write on board:
```
GRADIENT DESCENT:  W = W - lr × dW         (simple, but slow)

ADAM (Adaptive Moment Estimation):
  m = β₁ × m + (1 - β₁) × dW             (1st moment: momentum)
  v = β₂ × v + (1 - β₂) × dW²            (2nd moment: adaptive scale)

  m̂ = m / (1 - β₁ᵗ)                       (bias correction)
  v̂ = v / (1 - β₂ᵗ)                       (bias correction)

  W = W - lr × m̂ / (√v̂ + ε)

Default: β₁=0.9, β₂=0.999, ε=1e-8, lr=0.001
```

> *"Adam is like gradient descent with memory and auto-scaling.*
>
> *The first moment m is momentum — instead of jumping based on one gradient,
> we average recent gradients. Less jittery, smoother path.*
>
> *The second moment v tracks how large gradients have been.
> If a weight's gradient is always large, scale it down automatically.
> If it's always tiny, scale it up. Every weight gets its own learning rate.*
>
> *The bias corrections (m̂, v̂) fix the fact that m and v start at zero —
> they'd be too small at the beginning without correction.*
>
> *In practice: Adam almost always works better than plain SGD.
> Use it as your default optimizer and only switch if needed."*

---

## SECTION 3: Mini-Batch Training Loop (15 min)

Write on board:
```python
def fit(self, X, y, batch_size=32, epochs=200):
    for epoch in range(epochs):
        # Shuffle data each epoch
        idx = np.random.permutation(len(X))
        X_shuffled, y_shuffled = X[idx], y[idx]

        # Mini-batches
        for start in range(0, len(X), batch_size):
            Xb = X_shuffled[start:start+batch_size]
            yb = y_shuffled[start:start+batch_size]

            # Forward pass
            output = self.forward(Xb)

            # Backward pass → get gradients
            grads = self.backward(Xb, yb, output)

            # Adam update
            self.adam_update(grads)

        # Log progress every 50 epochs
        if epoch % 50 == 0:
            loss = cross_entropy_loss(y, self.forward(X))
            acc = accuracy(y, self.forward(X))
            print(f"Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")
```

> *"Shuffle every epoch — prevents the network from memorizing order.*
> *Split into mini-batches — each batch gives one Adam update.*
> *Repeat for many epochs — the network gradually improves.*
>
> *This is identical to what Keras does under the hood with model.fit().
> You now know exactly what 'epoch' and 'batch_size' mean at the code level."*

---

## SECTION 4: Live Demo — XOR Solved (10 min)

```bash
python3 multilayer_perceptron.py
```

Watch together:
- XOR section: accuracy climbs from 50% to 100%
- Print the final predictions: [0, 1, 1, 0] — correct!
- Open the visualization showing the curved decision boundary

> *"There it is. 100% on XOR. The same problem that ended neural network
> research for ten years — we just solved it in 200 lines of numpy.*
>
> *Look at the decision boundary. It's not a straight line.*
> *It's a curved region. The hidden layers have transformed the problem
> from 'impossible with one line' to 'solvable with a curve.'*
>
> *Every deep network is doing this — transforming the space*
> *until the answer becomes obvious."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
FORWARD PASS:   Z = A_prev @ W + b   →   A = activation(Z)
BACKWARD PASS:  dZ[L] = A - Y       →   chain rule back through layers
ADAM:           momentum + adaptive learning rates per weight
MINI-BATCH:     shuffle → batches → forward → backward → update → repeat

RESULT: XOR solved, complex decision boundaries, full gradient-based learning
```

**Homework — from `multilayer_perceptron.py`:**
```python
# 1. What accuracy does the MLP reach on XOR with architecture [2,4,2]?
# 2. Change architecture to [2, 2, 2] — does it still solve XOR?
# 3. Change architecture to [2, 8, 8, 2] — how many epochs to converge?
# 4. Open the decision boundary visualization — describe in 1 sentence
#    what the hidden layer transformation did to the input space.
```

---

## INSTRUCTOR TIPS & SURVIVAL GUIDE

**"The shapes in backprop are confusing"**
> *"Always check: dW should be the same shape as W.*
> *If W is (fan_in, fan_out), then dW = A_prev.T @ dZ gives (fan_in, fan_out). Correct.*
> *When confused, print shapes. Let numpy tell you what went wrong."*

**"Why doesn't the loss always go down smoothly?"**
> *"Mini-batches introduce noise. Each batch is a random sample of the data,
> so each gradient is a noisy estimate of the true gradient.*
> *Epoch-level loss will trend down; batch-level loss will jump around.
> Adam's momentum smooths this — but some noise is expected and healthy."*

**"How is this different from what we coded in Module 03 (backprop)?"**
> *"Module 03 showed backprop for a single example and fixed small network.*
> *Now we have batches, arbitrary depth (any number of layers), He initialization,
> and Adam instead of plain SGD. The math is the same — the engineering scales up."*

**"Adam has so many variables — do I need to remember all the formulas?"**
> *"In practice, no. You set lr and use the defaults. The important insight is WHY
> Adam works: momentum smooths the path, adaptive scaling handles different weight scales.
> The exact formula lives in the code — you know what it's doing, that's enough."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook (XOR payoff)              10 min
├── Architecture and helper functions      15 min
├── Weight initialization (He)            15 min
├── Forward pass                           20 min
└── Close + homework                       10 min

SESSION 2  (90 min)
├── Opening bridge                         10 min
├── Backward pass (backprop)              30 min
├── Adam optimizer                         15 min
├── Mini-batch training loop               15 min
├── Live demo: XOR solved                  10 min
└── Close + homework                       10 min
```

---
*MLForBeginners · Part 3: Deep Neural Networks · Module 07*
