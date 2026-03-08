# MLForBeginners — Instructor Guide
## Part 3 · Module 06: Perceptron from Scratch
### Two-Session Teaching Script

> **Prerequisites:** All 5 math foundations complete. They know neurons, activations,
> forward propagation, backpropagation, loss functions. They understand gradient
> descent conceptually and can read numpy code.
> **Payoff today:** They will implement the original 1957 neural network from scratch,
> understand its legendary failure on XOR, and see exactly why we need multiple layers.

---

# SESSION 1 (~90 min)
## "1957 — the machine that learned"

## Before They Arrive
- Terminal open in `deep_neural_networks/algorithms/`
- Whiteboard ready
- Draw a large circle with arrows coming in (a single neuron diagram)
- Optional: look up a photo of Frank Rosenblatt's Mark I Perceptron machine

---

## OPENING (10 min)

> *"The year is 1957. Computers fill entire rooms. Programming means
> plugging in cables by hand. And one man — Frank Rosenblatt — builds
> a machine that can LEARN.*
>
> *He calls it the Perceptron. The New York Times runs a story.
> The Navy funds it. People think this is the beginning of artificial intelligence.*
>
> *Today you're going to build exactly what he built.*
>
> *And then we're going to watch it fail at a simple problem —
> and that failure will explain why modern deep learning exists."*

Draw on board:
```
                1957: THE PERCEPTRON
                Frank Rosenblatt, Cornell University

    x₁ ──[w₁]──┐
    x₂ ──[w₂]──┤──( Σ )──[ step ]──→ 0 or 1
    x₃ ──[w₃]──┘
               bias b

    The first learning algorithm.
    It changes history. And it will fail at XOR.
```

> *"Let's build it."*

---

## SECTION 1: The Perceptron Algorithm (20 min)

Write on board:
```
FORWARD PASS (prediction):
  z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b   (weighted sum)
  y_hat = 1 if z >= 0  else  0          (step function)

LEARNING RULE (update only on WRONG predictions):
  error = y_true - y_hat
  w = w + lr × error × x
  b = b + lr × error

Key insight: if correct (error=0) → no update at all
             if wrong (error=±1) → weights shift toward correct answer
```

> *"Unlike gradient descent which updates on every example,
> the perceptron update rule is purely reactive:
> I got it wrong, let me adjust. I got it right, leave everything alone.*
>
> *The step function is brutally simple — not sigmoid, not ReLU.
> Just a hard switch: below zero means 0, above zero means 1.
> Like a light switch, not a dimmer."*

Draw the step function:
```
output
  1 │          ████████████
    │          │
  0 │█████████ │
    └──────────┼──────────── z
               0

Hard threshold at z=0.
No gradient, no smooth curve.
```

**Ask the room:** *"Can we use backpropagation to train the step function?"*

> (No — the derivative of the step function is zero everywhere except at z=0
> where it's undefined. This is why the perceptron uses its own special
> learning rule instead of gradient descent.)

**Ask the room:** *"What's the geometric meaning of w·x + b >= 0?"*

> (It defines a hyperplane — a line in 2D, a plane in 3D.
> The perceptron draws one straight line to separate classes.)

---

## SECTION 2: Building the Perceptron Class (25 min)

Write the full class on the board piece by piece:

```python
class Perceptron:
    def __init__(self, lr=0.01, n_epochs=100):
        self.lr = lr
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = 0.0

    def predict(self, X):
        z = X @ self.weights + self.bias   # dot product
        return (z >= 0).astype(int)        # step function

    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)

        for epoch in range(self.n_epochs):
            errors = 0
            for xi, yi in zip(X, y):          # one example at a time
                y_hat = self.predict(xi.reshape(1, -1))[0]
                error = yi - y_hat
                if error != 0:                 # only update on mistakes
                    self.weights += self.lr * error * xi
                    self.bias    += self.lr * error
                    errors += 1
            if errors == 0:                    # converged!
                print(f"Converged at epoch {epoch+1}")
                break
```

> *"Walk through this together. The outer loop is epochs.
> The inner loop goes through each individual training example.
> When we get it wrong, we nudge the weights.*
>
> *This is called online learning — one example at a time.
> Modern networks use mini-batches, but the idea is the same."*

**Live Demo:**
```bash
python3 perceptron_from_scratch.py
```

Watch AND gate learning together:
```
AND truth table:
  0 AND 0 = 0
  0 AND 1 = 0
  1 AND 0 = 0
  1 AND 1 = 1      ← only both inputs on
```

> *"The perceptron learns the AND gate. Watch the weights change per epoch.
> Eventually they stabilize. Converged means it gets every training example correct.*
>
> *This was the magic of 1957 — a machine that found its own answer."*

---

## SECTION 3: AND, OR, and the Decision Boundary (15 min)

Draw on board:
```
AND GATE (linearly separable):   OR GATE (linearly separable):

  1 │    •                         1 │  •  •
    │         •  ← (1,1)=YES        │
  0 │  ×  ×                       0 │  ×  •  ← (1,0),(0,1)=YES
    └────────────                   └────────────
      0    1                          0    1

    One line can separate them.  One line can separate them.
    Perceptron WORKS.            Perceptron WORKS.
```

> *"If you can draw one straight line to separate the YES examples from the NO
> examples — the perceptron can solve it. This is called LINEAR SEPARABILITY.*
>
> *AND: one line goes between the top-right and everything else. Done.*
> *OR: one line separates the bottom-left from everything else. Done.*
>
> *The perceptron is like a student who can only draw one straight line
> on the paper to separate two groups. If one line works, great.
> If not... trouble."*

---

## CLOSING SESSION 1 (10 min)

Board summary:
```
THE PERCEPTRON:
  z = w·x + b             (weighted sum)
  y_hat = step(z)          (hard 0 or 1)
  update only when wrong   (reactive learning)

Works on: AND, OR (linearly separable)
```

**Homework:** Draw the truth table for NAND (NOT AND). Is it linearly separable?
Sketch the points on an x-y grid and try to draw a separating line.

---

# SESSION 2 (~90 min)
## "The XOR problem — and why we need layers"

## OPENING (10 min)

> *"You did your homework. AND works. OR works.*
>
> *Today we try XOR — exclusive OR. It sounds simple.*
> *XOR returns 1 when inputs are DIFFERENT.*
>
> *This destroyed the perceptron. Not just hurt it — destroyed it.*
> *A book called Perceptrons by Minsky and Papert proved mathematically that
> the single perceptron cannot solve XOR. And that was nearly the end of
> neural network research for a decade.*
>
> *Let's see why. Then let's see the fix."*

---

## SECTION 1: XOR — The Fatal Problem (20 min)

Draw on board:
```
XOR TRUTH TABLE:
  Input A  Input B  Output
     0        0       0    ← same inputs → 0
     0        1       1    ← different   → 1
     1        0       1    ← different   → 1
     1        1       0    ← same inputs → 0

Plot on grid:
  1 │   •        ×
    │
  0 │   ×        •
    └────────────────
        0        1

× = class 0   •  = class 1

Try to draw ONE straight line that separates × from •.
```

Pause. Let them try to draw the line.

> *"You can't. It's impossible. The two • points are in diagonally opposite corners.
> No single straight line separates them.*
>
> *The perceptron converges perfectly on AND and OR.*
> *On XOR, it will never converge — it will flip-flop forever.*
>
> *This isn't a software bug. It's a mathematical impossibility.
> The perceptron only works for linearly separable problems,
> and XOR is NOT linearly separable."*

**Live Demo:**
```bash
python3 perceptron_from_scratch.py
```

Watch the XOR training section. Point at the output:
- Accuracy oscillates, never reaches 100%
- "FAILED TO CONVERGE" message
- The decision boundary shown in the visualization

> *"There it is. The perceptron trying and failing. Every epoch, the line
> shifts — but there's no line that works, so it never stops moving.*
>
> *In 1969, Minsky and Papert wrote a book proving this.*
> *Funding dried up. AI winter began.*
>
> *But we know the fix now — and we'll build it next session."*

---

## SECTION 2: Why Multiple Layers Solve XOR (20 min)

Write on board:
```
THE FIX: Add a hidden layer

Layer 1 (hidden): transforms the data
Layer 2 (output): classifies the transformed data

XOR via NAND + OR:
  h₁ = NAND(x₁, x₂)   → output: [1, 1, 1, 0]
  h₂ = OR(x₁, x₂)      → output: [0, 1, 1, 1]
  y  = AND(h₁, h₂)     → output: [0, 1, 1, 0] ← that's XOR!

Each subproblem is linearly separable.
Combined: XOR is solved.
```

Draw the two-layer network:
```
                       HIDDEN LAYER
x₁ ─────────────→ [h₁] ─────────→ [output] → y
         ╲       ╱         ╲      ╱
x₂ ─────────────→ [h₂] ─────────→

The hidden layer TRANSFORMS the input space.
After transformation, the data IS linearly separable.
```

> *"The hidden layer remaps the input. After the first layer,
> the XOR pattern has been untangled into something a single line CAN separate.*
>
> *This is the insight behind the entire field of deep learning:*
> *stack layers to progressively transform data into a form that's easy
> to classify at the end.*
>
> *Every deep network you build from now on is doing this — transforming
> the representation layer by layer until the problem becomes simple."*

**Ask the room:** *"Why does adding a hidden layer break the requirement for linear separability?"*

> (Because the hidden layer is a learned non-linear transformation of the input space.
> We're not drawing a line in the original space — we're drawing a line
> in the transformed space, which can have a totally different geometry.)

---

## SECTION 3: sklearn's Perceptron — Sanity Check (15 min)

```python
from sklearn.linear_model import Perceptron

# AND gate
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0, 0, 0, 1])

clf = Perceptron(max_iter=100, random_state=42)
clf.fit(X_and, y_and)
print("AND accuracy:", clf.score(X_and, y_and))

# XOR gate
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0, 1, 1, 0])

clf2 = Perceptron(max_iter=1000, random_state=42)
clf2.fit(X_xor, y_xor)
print("XOR accuracy:", clf2.score(X_xor, y_xor))  # will not be 1.0
```

> *"Run this. AND: 100%. XOR: 50% or 75% — never 100%. Even sklearn can't do it.*
> *The algorithm is correct. The limitation is mathematical."*

**Ask the room:** *"What accuracy would a random coin flip give on XOR?"*

> (50% — two of four examples are class 0, two are class 1, balanced.
> A random predictor hits 50%. The perceptron might not even beat that.)

---

## SECTION 4: Historical Context + Bridge to MLP (10 min)

> *"Timeline:*
> *1957 — Rosenblatt invents the perceptron*
> *1969 — Minsky & Papert prove XOR failure, funding stops*
> *1986 — Rumelhart, Hinton & Williams: backpropagation for multi-layer networks*
> *Now we can train networks with many layers. XOR is trivial.*
>
> *The perceptron wasn't wrong — it was incomplete.*
> *Stack them, add non-linear activations, apply backprop.*
> *That's the Multi-Layer Perceptron — and that's next module.*
>
> *The perceptron is the ancestor of everything:*
> *GPT-4, DALL-E, AlphaFold, self-driving cars — all of it*
> *starts with this one simple idea."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
THE PERCEPTRON:               THE LIMITATION:
z = w·x + b                  Can only draw ONE line
y_hat = step(z)               Only works if data is linearly separable
update when wrong             XOR: NOT linearly separable → FAILS

THE FIX: Multi-Layer Perceptron
  Hidden layers transform the input space
  After transformation, data becomes separable
  Backpropagation trains all layers together
  → Module 07
```

**Homework — from `perceptron_from_scratch.py`:**
```python
# 1. What final accuracy does the perceptron reach on OR?
# 2. How many epochs does AND take to converge?
# 3. Modify the learning rate to 0.001 — does AND still converge?
# 4. NAND gate: y_nand = [1,1,1,0]. Does the perceptron learn it?
#    (Hint: check the truth table and whether it's linearly separable)
```

---

## INSTRUCTOR TIPS & SURVIVAL GUIDE

**"Why use step function and not sigmoid?"**
> *"Historically, Rosenblatt's 1957 perceptron used a hard threshold — step function.
> This is the original definition. Modern neurons use sigmoid or ReLU because
> they're differentiable (we can use backprop). Step function has no gradient,
> so we use the special perceptron update rule instead."*

**"Is XOR really that important? Who cares?"**
> *"XOR represents any non-linearly separable problem — which is most of reality.
> Real data (images, text, speech) is never cleanly separable with one line.
> If the perceptron couldn't solve XOR, it couldn't solve anything useful.
> Understanding XOR failure is understanding WHY deep learning was invented."*

**"If the perceptron only works for linearly separable data, what's the point of learning it?"**
> *"Three reasons: 1) historical foundation — it IS the original neural unit,
> 2) the update rule intuition carries into modern networks,
> 3) many real problems ARE linearly separable once you engineer the right features.
> Linear classifiers are still used in spam filtering, linear SVMs, etc."*

**"The visualization shows the boundary moving forever on XOR — is that a bug?"**
> *"That's expected behavior. The perceptron convergence theorem guarantees
> convergence IF the data is linearly separable. XOR is not, so the theorem
> doesn't apply and the boundary never settles. Not a bug — mathematical proof."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook (1957 story)               10 min
├── The perceptron algorithm                20 min
├── Building the class from scratch         25 min
├── AND, OR, decision boundaries            15 min
└── Close + homework                        10 min

SESSION 2  (90 min)
├── Opening bridge (XOR intro)             10 min
├── XOR: the fatal problem                  20 min
├── Why hidden layers fix it                20 min
├── sklearn Perceptron sanity check         15 min
├── Historical context + bridge to MLP     10 min
└── Close + homework                        10 min
```

---
*MLForBeginners · Part 3: Deep Neural Networks · Module 06*
