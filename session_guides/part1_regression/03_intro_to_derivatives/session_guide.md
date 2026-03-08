# MLForBeginners — Instructor Guide
## Module 3: Introduction to Derivatives  ·  Two-Session Teaching Script

> **Who this is for:** You, teaching close friends with zero ML background.
> **Tone:** Casual, curious, conversational — like explaining over coffee.
> **Goal by end of both sessions:** Everyone understands what a derivative is (slope
> at a point), how gradient descent works, and why it's the engine that makes ML learn.

---

# ─────────────────────────────────────────────
# SESSION 1  (~90 min)
# "Derivatives aren't scary — they're just slope. And you already know slope."
# ─────────────────────────────────────────────

## Before They Arrive (10 min setup)

**On your laptop, have ready:**
- Terminal ready in `MLForBeginners/regression_algorithms/math_foundations/`
- Visuals: `visuals/03_derivatives/` open in Finder
- A whiteboard — this session is very diagram-heavy
- Optional: a marble or ball to roll around for the hill metaphor

**Heads up:** This is the module people most expect to hate. Start with the hill story immediately.
Do not open with "derivatives" or "calculus" — say "slopes" instead.

---

## OPENING  (10 min)

### Hook — The hiking story

Say this out loud:

> *"Imagine you're blindfolded on a hilly landscape.*
> *Your only goal: find the lowest point. The valley.*
> *You can't see anything. You can only feel the ground under your feet.*
>
> *What do you do?*
> *You feel which direction the ground slopes downward.*
> *You take a step that way.*
> *Feel the slope again.*
> *Take another step.*
> *Repeat until the ground feels flat.*
>
> *That's it. That's gradient descent.*
> *That's how EVERY machine learning model learns.*
>
> *A derivative is just a way to measure which direction is 'downhill.'*
> *That's the entire lesson today."*

**Draw on board:**

```
        •  ← "start here, totally lost"
       / \
      /   \
─────/     \─────
             \   /
              \_/  ← "trying to find THIS: the minimum"

At each step: feel the slope → step downhill → repeat
That's gradient descent. Derivative tells you the slope.
```

> *"We are NOT doing calculus today.*
> *We're learning one concept from calculus: slope at a single point.*
> *And we're going to see exactly how ML uses it."*

---

## SECTION 1: From Lines to Curves — Slope Revisited  (20 min)

> *"You remember slope from last time: rise over run.*
> *For a STRAIGHT line, slope is constant everywhere.*
>
> *But what about a CURVE?"*

**Draw on board:**

```
STRAIGHT LINE:               CURVE:
        /                         **
       /                       *    *
      /  slope = 2            *      *
     /   everywhere        *          *
    /
                           Slope changes
                           at every point!
```

> *"On a curve, the slope changes every time you move.*
> *At the top of the hill: slope = 0 (flat at the peak)*
> *Going up: slope = positive*
> *Going down: slope = negative*
> *At the bottom of the valley: slope = 0 again*
>
> *A derivative gives you the slope at any SINGLE point on a curve.*
> *That's the whole thing."*

**Concrete example — the parabola:**

Write on board:
```
f(x) = x²

At x = 3: slope = 6     (going steeply upward)
At x = 1: slope = 2     (going gently upward)
At x = 0: slope = 0     (flat — this is the minimum!)
At x =-2: slope = -4    (going downward to the left)
```

> *"The rule for x² is: slope at any point = 2x.*
> *That's the derivative: f'(x) = 2x.*
>
> *You don't need to know WHY that rule is 2x right now.*
> *You just need to know: derivative = slope at that point.*
> *And slope = 0 means you're at a peak or a valley."*

**Ask the room:**

> *"In machine learning, we want to MINIMIZE error.*
> *That means we want to find where the 'error curve' is at its lowest point.*
> *What's the slope at the lowest point?"*

Let them answer. Someone will say zero.

> *"Exactly. Slope = 0. That's our target.*
> *We use the derivative to hunt for zero slope.*
> *When we find it, we've found the minimum error.*
> *That's how models learn to make better predictions."*

---

## SECTION 2: Gradient Descent — Rolling Downhill  (30 min)

> *"Now let's see this actually work.*
> *We're going to roll down a curve — step by step."*

**Set up the example on board:**

```
Curve: f(x) = x²
Derivative: f'(x) = 2x   (slope at any point)

START: x = 4   (somewhere on the right side of the curve)

Goal: find x = 0   (the minimum)
```

> *"Here's the algorithm. It's almost embarrassingly simple:*
>
> *STEP 1: Look at the current slope.*
> *STEP 2: Take a step in the OPPOSITE direction.*
> *STEP 3: Repeat.*
>
> *The 'step size' is called the learning rate.*
> *Let's use learning rate = 0.1."*

**Walk through the first 4 steps on the board:**

```
Step | x   | slope (2x) | new x = x - 0.1 × slope
─────┼──────┼────────────┼────────────────────────
  0  | 4.0  |    8.0     |  4.0 - 0.8 = 3.2
  1  | 3.2  |    6.4     |  3.2 - 0.64 = 2.56
  2  | 2.56 |    5.12    |  2.56 - 0.512 = 2.05
  3  | 2.05 |    4.1     |  2.05 - 0.41 = 1.64
  ...      getting smaller and smaller
```

> *"See it? We started at 4. We're rolling toward 0.*
> *After enough steps, we'll be at 0.001, 0.0001...*
> *We never quite reach 0 — but we get close enough to not matter."*

**Now run the Python version:**

```bash
python3 03_intro_to_derivatives.py
```

> *"Watch the output — it's doing exactly what we just drew.*
> *And the visuals folder has animations of the ball rolling down the curve.*"

Open `visuals/03_derivatives/` — show the slope visualization and descent plots.

> *"This is the heartbeat of machine learning.*
> *Every time ChatGPT learned to write a better sentence,*
> *gradient descent ran on a trillion-parameter curve.*
> *Same algorithm. Massively bigger scale."*

---

## SECTION 3: The Learning Rate Problem  (15 min)

> *"Here's where it gets interesting — and where beginners break things.*
>
> *The learning rate is how big a step you take each time.*
> *What happens if it's too small?"*

**Draw on board:**

```
Learning rate = 0.001 (tiny)
→ Each step is minuscule
→ You'll get there... eventually
→ But it might take a million steps = slow training

Learning rate = 2.0 (huge)
→ Slope says "go right a little"
→ You overshoot MASSIVELY, end up way left
→ Then slope says "go right" again
→ You overshoot the other way
→ You NEVER converge (it just bounces forever)

Learning rate = 0.1 (Goldilocks)
→ Steps are reasonable
→ Converges in a few dozen iterations
```

**Ask the room:**

> *"In the hiking analogy — what's a learning rate that's too large?*
> *What's a learning rate that's too small?"*

- Too large: you're taking 10-meter leaps across the valley, constantly overshooting
- Too small: you're shuffling millimeter by millimeter, never reaching the bottom

> *"One of the first things ML engineers tune is the learning rate.*
> *It's not computed. It's chosen. And choosing it well is part experience, part art."*

---

## CLOSING SESSION 1  (5 min)

### What we learned today

Write on board:

```
Derivative = slope at a single point

Slope = 0  → at a minimum (or maximum) of a curve

Gradient descent:
  1. Compute the slope
  2. Step OPPOSITE to the slope
  3. Repeat until slope ≈ 0

Learning rate:
  Too small → slow
  Too large → bounces, never converges
  Just right → finds the minimum efficiently
```

> *"Between now and next time:*
> *Try the Quick Win Challenge from the lab — just the first part.*
> *f(x) = x². Derivative is 2x. Compute the slope at x=0, x=3, x=-2.*
> *That's it. Five minutes."*

---
---

# ─────────────────────────────────────────────
# SESSION 2  (~90 min)
# "Gradient descent by hand, then watch Python do it 1000 times a second."
# ─────────────────────────────────────────────

## Opening  (10 min)

### Homework debrief

> *"Quick check — the derivative of x² is 2x.*
> *What's the slope at x=0?"*

Someone will say zero.

> *"Right. Zero. Flat. That's the minimum.*
> *At x=3?"*

Someone: "6."

> *"Exactly. We're rolling in the positive direction at x=3.*
> *So gradient descent would say: step LEFT to reduce x.*
>
> *Today we're going to run gradient descent by hand.*
> *And then we'll break it by choosing a bad learning rate — just for fun."*

---

## SECTION 1: Gradient Descent by Hand  (25 min)

Open the lab file and work through the by-hand table together:

> *"Open your notes. Let's do 5 steps of gradient descent on f(x) = x².*
> *Starting at x = 4. Learning rate = 0.1."*

**Have them fill this in — pause after each row:**

```
Step | x   | f(x) | slope | x_new
─────┼──────┼───────┼───────┼──────────────
  0  | 4.0  | 16.0  |  8.0  | 4 - 0.8 = 3.2
  1  | 3.2  | 10.24 |  6.4  | 3.2 - 0.64 = ?
  2  | ?    |  ?    |  ?    | ?
  3  | ?    |  ?    |  ?    | ?
  4  | ?    |  ?    |  ?    | ?
```

> *"Notice the function value — f(x) — is getting smaller every step.*
> *We are making progress. The error is decreasing."*

After filling in 4-5 rows:
> *"Now check your work."*

```python
x = 4.0
learning_rate = 0.1

for step in range(5):
    fx = x ** 2
    grad = 2 * x
    x_new = x - learning_rate * grad
    print(f"Step {step}: x={x:.3f}, f(x)={fx:.3f}, slope={grad:.3f}, x_new={x_new:.3f}")
    x = x_new
```

**If they got it right — celebrate loudly.** This is the algorithm that runs inside every neural network.

---

## SECTION 2: Break It — Different Learning Rates  (20 min)

> *"Now let's cause some chaos.*
> *We're going to try three different learning rates and see what happens."*

```python
import matplotlib.pyplot as plt

def gradient_descent(start, lr, steps):
    x = start
    history = [x]
    for _ in range(steps):
        x = x - lr * (2 * x)  # derivative of x²
        history.append(x)
    return history

plt.figure(figsize=(12, 4))

for i, lr in enumerate([0.1, 0.5, 1.1]):
    plt.subplot(1, 3, i+1)
    history = gradient_descent(start=4, lr=lr, steps=20)
    plt.plot(history, 'o-')
    plt.axhline(y=0, color='r', linestyle='--', label='target (x=0)')
    plt.title(f'Learning Rate = {lr}')
    plt.xlabel('Step')
    plt.ylabel('x value')
    plt.legend()

plt.tight_layout()
plt.savefig('../visuals/learning_rate_experiment.png', dpi=150)
plt.show()
```

**Before running, ask the room to predict:**
> *"What do you think happens with lr=1.1? Just guess — no wrong answers."*

After running and showing the plot:
> *"lr=0.1: smooth descent. Perfect.*
> *lr=0.5: converges, but faster and a bit bumpier.*
> *lr=1.1: chaos. It bounces between positive and negative, getting BIGGER each time.*
> *This is called divergence. The model breaks."*

> *"In real ML, if your training loss is going UP instead of down,*
> *your learning rate is too high. That's the first thing you check."*

---

## SECTION 3: 2D Gradient Descent — Preview of Real ML  (15 min)

> *"In real ML, there's not one parameter — there are millions.*
> *But the concept is exactly the same.*
> *Let's try TWO variables."*

```python
import numpy as np

x, y = 4.0, 3.0
lr = 0.1

print("Minimizing f(x, y) = x² + y²")
print("Target: (0, 0)")
print("-" * 40)

for step in range(10):
    grad_x = 2 * x    # derivative with respect to x
    grad_y = 2 * y    # derivative with respect to y

    x = x - lr * grad_x
    y = y - lr * grad_y

    f_val = x**2 + y**2
    print(f"Step {step}: x={x:.3f}, y={y:.3f}, f={f_val:.4f}")
```

> *"Two variables, two derivatives, two update steps.*
> *For a neural network with 175 billion parameters?*
> *175 billion derivatives, 175 billion updates.*
> *Same algorithm. Just massively parallel.*
> *This is called backpropagation — we'll get there much later.*
> *But it all starts here."*

---

## SECTION 4: The ML Connection  (10 min)

> *"Let me tie everything together.*
> *In linear regression, we're trying to find the slope m and intercept b*
> *that minimize the prediction error.*
>
> *The 'error curve' is bowl-shaped — like x².*
> *Gradient descent rolls down that bowl.*
> *It tweaks m and b a tiny bit at every step.*
> *Each step makes the predictions slightly more accurate.*
>
> *After hundreds or thousands of steps?*
> *The model has learned the best line.*"

**Draw on board:**

```
Error (MSE)
     │        *
     │      *   *
     │    *       *
     │  *           *
     │*               *
     └────────────────────→  (values of m and b)
                   ↑
           gradient descent
           finds this minimum
```

> *"When you run sklearn's LinearRegression or any neural network,*
> *it's doing exactly this — usually thousands of times per second.*
> *You've just seen the engine."*

---

## CLOSING SESSION 2  (10 min)

### What we now understand

Write on board, read out loud together:

```
WHAT A DERIVATIVE IS:
  slope at a single point on any curve
  f'(x) = 0  →  you're at a minimum or maximum

GRADIENT DESCENT:
  repeat until converged:
    slope = compute derivative
    parameter = parameter - learning_rate × slope

LEARNING RATE:
  too small  → slow, might never finish
  too large  → bounces, diverges, breaks
  just right → converges smoothly

WHY IT MATTERS FOR ML:
  Every time a model trains, it runs gradient descent.
  It adjusts its parameters step by step to minimize error.
  Derivatives tell it which direction to adjust.
```

### The Road Ahead

```
WHERE WE ARE:
  ✅ Algebra
  ✅ Statistics
  ✅ Derivatives + Gradient Descent

NEXT UP:
  → Linear Algebra (vectors and matrices)
     This is how multiple features get handled simultaneously.
  → Then: Linear Regression — build it from scratch
```

---

## Lab Assignment

### From `03_intro_to_derivatives_lab.md`:

**Assign the Quick Win (5 min):**
```python
def slope(x):
    return 2 * x  # derivative of x²

for x in [0, 3, -2]:
    print(f"At x={x}: slope = {slope(x)}")
```

**And the Learning Rate Chaos experiment (10 min):**
> *"Try lr=0.01, lr=0.5, lr=1.5. What happens to each?*
> *Can you find the exact learning rate where it starts to diverge?*
> *(It's somewhere between 0.9 and 1.1.)"*

> *"Optional boss challenge: the 2D gradient descent — add a third variable z.*
> *Does the algorithm still work?"*

---

# ─────────────────────────────────────────────
# INSTRUCTOR TIPS & SURVIVAL GUIDE
# ─────────────────────────────────────────────

## When People Get Confused

**"I thought derivatives were hard calculus. Why is this easy?"**
> *"You're learning the intuition, not the machinery.*
> *Professional mathematicians prove why f(x) = x² gives f'(x) = 2x.*
> *We're just using that result.*
> *For ML, you need to understand what a derivative means — not how to derive one.*
> *In practice, libraries like PyTorch and TensorFlow compute derivatives automatically.*
> *You never have to calculate them by hand in production."*

**"Why do we subtract the slope? Why not add it?"**
> *"Think about the hill.*
> *If the slope is positive (you're going uphill on the right),*
> *you want to go LEFT — decrease x.*
> *Subtracting a positive slope moves you left.*
> *If slope is negative (you're going downhill on the left side),*
> *subtracting a negative adds — moves you right.*
> *It's always 'opposite the slope.' That's why minus."*

**"What's the difference between gradient and derivative?"**
> *"For a function with one variable: same thing.*
> *For a function with many variables: gradient is the collection of all derivatives.*
> *One derivative per variable. The gradient points in the direction of steepest increase.*
> *We go the OPPOSITE direction to descend."*

**"Does ML always use gradient descent?"**
> *"For neural networks: always.*
> *For simpler models like linear regression: we can sometimes use a direct formula.*
> *We'll see that in the next module — the normal equation.*
> *But gradient descent scales to any model, any size.*
> *That's why it's the universal algorithm."*

## Energy Management

- **The by-hand table (Session 2):** Do it SLOWLY. Let people catch up.
  The aha moment hits when they see the numbers rolling toward zero.
- **If people are intimidated by math:** Keep saying "you don't need to memorize rules."
  Use the rolling-ball metaphor as many times as it takes.
- **If someone finishes early:** Ask them: "What happens if your curve has two valleys?
  Does gradient descent always find the global minimum?" (Answer: not necessarily.
  This leads to great discussion about local vs global minima.)
- **At 50 minutes:** Short break, walk around.

## The Golden Rule

> The word "calculus" should never appear in this session.
> We teach slope. We teach downhill. We teach the update rule.
> The concept is already theirs — we're just naming it properly at the end.

---

# ─────────────────────────────────────────────
# QUICK REFERENCE — Session Timing
# ─────────────────────────────────────────────

```
SESSION 1  (90 min)
├── Opening hook                  10 min
├── Slope revisited — curves      20 min
├── Gradient descent walkthrough  30 min
├── Learning rate problem         15 min
└── Close + homework               5 min

SESSION 2  (90 min)
├── Homework debrief              10 min
├── By-hand gradient descent      25 min
├── Learning rate chaos demo      20 min
├── 2D gradient descent preview   15 min
├── ML connection                 10 min
└── Close + lab                   10 min
```

---

*Generated for MLForBeginners — Module 03 · Part 1: Regression*
