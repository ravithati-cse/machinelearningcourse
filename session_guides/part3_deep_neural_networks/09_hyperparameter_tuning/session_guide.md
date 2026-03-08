# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 09: Hyperparameter Tuning
### Two-Session Teaching Script

> **Prerequisites:** Modules 01–08. They've built MLPs in NumPy and Keras,
> trained on MNIST-style problems, and seen training curves.
> **Payoff:** Learn to systematically make models better — not by luck.

---

# SESSION 1 (~90 min)
## "The 7 knobs that control your neural network"

## Before They Arrive
- Terminal open in `deep_neural_networks/algorithms/`
- Draw a "control panel" sketch on the board with 7 unlabeled dials

---

## OPENING (10 min)

> *"Imagine you're a sound engineer at a concert.*
> *You have a mixing board with dozens of knobs.*
> *Turn the wrong one and the bass drowns out the vocals.*
> *Turn the right ones just right and it's perfect.*
>
> *Training a neural network is the same.*
> *We have hyperparameters — knobs we set before training.*
> *Today we learn what each one does, and how to tune them."*

Write on board:
```
HYPERPARAMETERS (you set these):     PARAMETERS (model learns these):
  Learning rate α                      Weights W
  Number of layers                     Biases b
  Neurons per layer
  Batch size
  Epochs
  Dropout rate
  Activation function
```

---

## SECTION 1: Learning Rate — The Most Critical Knob (25 min)

> *"If you only tune ONE thing, tune this.*
>
> *Too high: the model bounces around, never converges*
> *Too low: takes forever, might get stuck*
> *Just right: smooth descent to a good minimum"*

Draw the three scenarios:
```
α too HIGH:        α too LOW:        α just RIGHT:
loss               loss              loss
│  ↗↘↗↘           │╲                │╲
│↘↗  ↘↗           │ ╲               │  ╲
│    (diverges)    │  ╲ (slow)       │    ╲___
└────────── steps  └─────── steps   └────── steps
```

```bash
python3 hyperparameter_tuning.py
```

Watch the learning rate comparison plots. Point out:
- α=0.001: stable, steady drop
- α=0.1: oscillates, may diverge
- α=0.0001: barely moves

> *"Start at 0.001. It's the community default for Adam optimizer.*
> *If your loss isn't dropping after 10 epochs, try 0.003 or 0.01.*
> *If your loss is bouncing, try 0.0003."*

---

## SECTION 2: Depth vs Width (20 min)

Write on board:
```
DEEP (many layers):          WIDE (many neurons per layer):
Input → H1 → H2 → H3 → Out  Input → [H1: 512 neurons] → Out

  Learns hierarchical          Learns broad patterns
  representations              at one level
  (edges → shapes → faces)
  Better for complex tasks     Simpler to train
```

> *"For images: deep works better (features build on features)*
> *For tabular data: width often matters more*
>
> *Rule of thumb: start with 2-3 layers, 64-256 neurons.*
> *Add layers if the model underfits. Add neurons if it's still underfitting.*
> *Never add complexity you don't need — Occam's razor."*

**Interactive:** *"For predicting house prices from 10 features — would you go deep or wide first?"*

---

## SECTION 3: Batch Size & Epochs (15 min)

```
BATCH SIZE = how many examples per gradient update

Batch=1 (SGD):     noisy but fast per epoch, good generalization
Batch=32:          balanced — community default
Batch=1024:        smooth gradients, needs more epochs, may overfit
Batch=full data:   like the normal equation, deterministic

EPOCHS = how many times you show the full dataset

Too few → underfitting (didn't learn enough)
Too many → overfitting (memorized training data)
Solution: Early Stopping!
```

---

## CLOSING SESSION 1 (5 min)

```
THE 7 KNOBS:
  Learning rate → most critical, try 0.001 first
  Layers        → start shallow, add if underfitting
  Neurons       → start 64-128, double if underfitting
  Batch size    → 32 is a safe default
  Epochs        → use Early Stopping
  Dropout       → 0.2-0.5 if overfitting
  Activation    → ReLU hidden, sigmoid/softmax output
```

---

# SESSION 2 (~90 min)
## "Grid search, learning rate schedules, and professional workflow"

## OPENING (5 min)

> *"Last session we learned what each knob does.*
> *Today: how to search over them systematically.*
> *And a pro technique — changing the learning rate as training progresses."*

---

## SECTION 1: Manual Grid Search (25 min)

> *"Grid search means trying every combination of hyperparameters*
> *and picking the best one."*

```python
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'n_layers':      [1, 2, 3],
    'n_neurons':     [32, 64, 128],
}
# Total combinations: 3 × 3 × 3 = 27 experiments
```

> *"27 experiments. Each takes 30 seconds. Total: 13 minutes.*
> *That's affordable for small networks.*
> *For big networks (GPT-4): each experiment costs $1M+.*
> *That's why hyperparameter research is so important at scale."*

Open the results table from the output. Find the winner together.

> *"What pattern do you see? Usually learning rate matters most.*
> *Then architecture. Batch size has the least effect.*
> *That's the general hierarchy — good to internalize."*

---

## SECTION 2: Learning Rate Schedules (20 min)

> *"Instead of a fixed learning rate, what if we start high and slow down?*
>
> *Like driving: full speed on the highway, slow down near the destination.*
> *We explore fast at first, then fine-tune carefully."*

Write on board:
```
COMMON SCHEDULES:

Step Decay:       lr = lr × 0.5 every 10 epochs
                  Simple, predictable

Exponential:      lr = lr₀ × e^(−decay × epoch)
                  Smooth decrease

Cosine Annealing: lr follows a cosine curve down to 0
                  Widely used in modern LLMs

ReduceLROnPlateau: lr drops when validation loss stops improving
                   Most practical for day-to-day use
```

```python
# In Keras — one line:
from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
model.fit(..., callbacks=[reduce_lr])
```

---

## SECTION 3: Professional Hyperparameter Workflow (15 min)

Write this on the board — they should photograph it:

```
STEP 1: Start with defaults
  lr=0.001, layers=2, neurons=64, batch=32, dropout=0.2

STEP 2: Train → check learning curves
  Is it overfitting? → more dropout, fewer neurons
  Is it underfitting? → more layers, more neurons, fewer epochs

STEP 3: Tune learning rate first (biggest impact)
  Try: 0.0001, 0.001, 0.01

STEP 4: Tune architecture
  Try adding one layer at a time

STEP 5: Use ReduceLROnPlateau + EarlyStopping always

STEP 6: Final validation on held-out test set ONCE
  (If you peek at test data during tuning, you're cheating)
```

---

## CLOSING SESSION 2 (10 min)

```
HYPERPARAMETER TUNING IN PRACTICE:
  → There is no perfect set of hyperparameters
  → Start with defaults, iterate based on learning curves
  → Learning rate is the most important knob
  → Use ReduceLROnPlateau + EarlyStopping in every project
  → Automated tools (Keras Tuner, Optuna) can search for you
```

**Homework:** Take the Keras MLP from module 08. Try 3 different learning rates and 2 different architectures. Which combination gives the best validation accuracy? Write it down with your reasoning.

---

## INSTRUCTOR TIPS

**"How do companies tune GPT-scale models?"**
> *"They use expensive ablations — train a small version, tune it,*
> *then scale up with those settings. The field of 'scaling laws'*
> *studies exactly which hyperparameters matter at scale.*
> *The Chinchilla paper (2022) showed that many LLMs were undertrained —*
> *they needed more data, not bigger models."*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening hook           10 min
├── Learning rate          25 min
├── Depth vs width         20 min
├── Batch size + epochs    15 min
└── Close                   5 min  (+ 15 min buffer)

SESSION 2  (90 min)
├── Opening bridge          5 min
├── Grid search            25 min
├── LR schedules           20 min
├── Pro workflow           15 min
└── Close + homework       15 min  (+ 10 min buffer)
```

---
*MLForBeginners · Part 3: Deep Neural Networks · Module 09*
