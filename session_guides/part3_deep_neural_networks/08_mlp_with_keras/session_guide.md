# MLForBeginners — Instructor Guide
## Part 3 · Module 08: MLP with Keras
### Two-Session Teaching Script

> **Prerequisites:** Module 07 complete — they just built an MLP from scratch in
> numpy. They know every piece: forward pass, backprop, Adam, mini-batches. They
> understand what `model.fit()` is actually doing internally.
> **Payoff today:** They discover that everything they built in ~200 lines of numpy
> can be replaced by ~10 lines of Keras — plus GPU support, automatic differentiation,
> professional callbacks, and model saving. The hard work of Module 07 pays off here.

---

# SESSION 1 (~90 min)
## "Why Keras exists — and how to use it right"

## Before They Arrive
- Terminal open in `deep_neural_networks/algorithms/`
- Run `python3 -c "import tensorflow; print(tensorflow.__version__)"` to confirm TF works
- If TF not installed: `pip install tensorflow` — do this before class starts
- Whiteboard ready

---

## OPENING (10 min)

> *"Last module, you wrote 200 lines of numpy to build an MLP.*
> *Forward pass, backward pass, Adam, mini-batch training.*
> *Every matrix multiply, every gradient, by hand.*
>
> *Today, we replace all of it with this:"*

Write on board:
```python
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(2, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_split=0.2)
```

> *"Ten lines. Same algorithm. Same math.*
> *But now you have GPU support, automatic differentiation,
> callbacks, model saving, and the full production toolchain.*
>
> *This only feels magical because you did it from scratch first.*
> *Everyone who hasn't done Module 07 is pressing buttons they don't understand.*
> *You know exactly what 'compile' does. You know what 'fit' is doing.*
> *That knowledge makes you a better engineer — not just a framework user."*

---

## SECTION 1: Why Keras Over NumPy (15 min)

Write on board:
```
NUMPY MLP (Module 07):          KERAS MLP (today):
───────────────────────         ─────────────────
~200 lines                      ~10 lines
Manual backprop                 Automatic differentiation
CPU only                        CPU + GPU + TPU
Manual Adam code                Dozens of optimizers built in
No callbacks                    EarlyStopping, LR scheduling, checkpoints
Manual saving                   model.save() / keras.models.load_model()
```

> *"Keras is built on TensorFlow, which uses a computation graph.*
> *When you write `Dense(64, activation='relu')`, TF builds a symbolic graph.*
> *During training, it traces the forward pass and automatically derives
> all the backward pass gradients — this is called autograd or autodiff.*
>
> *This means if you invent a new layer architecture, you don't need to
> derive its gradient by hand. You just write the forward pass and TF handles the rest.*
>
> *This is how researchers iterate so fast in deep learning today.*
> *New idea → code forward pass → autograd handles gradients → test immediately."*

**Ask the room:** *"In our numpy MLP, what would break if we added a 5th hidden layer?"*

> (We'd need to carefully extend the forward loop and backward loop, handling the
> correct layer indices and shapes. In Keras, you just add `layers.Dense(...)` to the list.)

---

## SECTION 2: The Sequential API (20 min)

Write on board and explain each line:
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build the model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(n_features,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(n_classes, activation='softmax')
])

# Inspect the architecture
model.summary()
```

Draw what `model.summary()` shows:
```
Model: "sequential"
_________________________________________________________________
Layer (type)          Output Shape         Param #
=================================================================
dense (Dense)         (None, 128)          1152        ← (10 features × 128) + 128 bias
dropout (Dropout)     (None, 128)          0
dense_1 (Dense)       (None, 64)           8256        ← (128 × 64) + 64 bias
dropout_1 (Dropout)   (None, 64)           0
dense_2 (Dense)       (None, 2)            130
=================================================================
Total params: 9,538
Trainable params: 9,538
```

> *"Count the parameters. Dense(128) with 10 input features:*
> *10 weights per neuron × 128 neurons = 1280 weights, plus 128 biases = 1408.*
> *Every one of those is a floating point number the model will adjust.*
>
> *The 'None' in the output shape means 'any batch size.'*
> *Keras doesn't know in advance how many examples you'll send at once.*
>
> *This summary is your first sanity check — before you train anything,
> print the summary and make sure the shapes look right."*

**Ask the room:** *"Our model.summary() shows 9,538 parameters. How does
Keras find the gradient for each one of those 9,538 numbers?"*

> (Autograd — TF builds a computation graph. On the backward pass,
> it uses the chain rule through the graph automatically.
> No manual derivation required.)

---

## SECTION 3: compile() — Setting Up Training (15 min)

Write on board:
```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

Explain each argument:
```
optimizer: the update rule
  'adam'       → Adam with default lr=0.001
  'sgd'        → vanilla stochastic gradient descent
  keras.optimizers.Adam(lr=0.0001)  → Adam with custom lr

loss: what to minimize
  'sparse_categorical_crossentropy'  → multi-class, integer labels
  'categorical_crossentropy'         → multi-class, one-hot labels
  'binary_crossentropy'              → binary classification
  'mse'                              → regression

metrics: what to display during training (doesn't affect optimization)
  ['accuracy']           → fraction correct
  ['accuracy', 'auc']    → multiple metrics
```

> *"compile() is where you tell Keras three things:*
> *1. How to update weights (optimizer)*
> *2. What score to minimize (loss)*
> *3. What to show you during training (metrics)*
>
> *The loss is what actually drives learning.*
> *Metrics are just for your information — they don't affect gradients."*

**Ask the room:** *"If I'm predicting house prices — a regression problem —
what loss function should I use?"*

> ('mse' — mean squared error. Binary/categorical crossentropy are for classification.)

---

## CLOSING SESSION 1 (10 min)

Board summary:
```
KERAS WORKFLOW:
  1. Build:    keras.Sequential([layers.Dense(...), ...])
  2. Inspect:  model.summary()
  3. Configure: model.compile(optimizer, loss, metrics)

WHY KERAS:
  ✓ Autodiff — no manual backprop
  ✓ GPU/TPU support
  ✓ Clean API — same network in 10 lines vs 200
  ✓ Production tooling
```

**Homework:** Run this and count the parameters manually to verify summary output:
```python
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(5,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])
model.summary()
# Compute: how many parameters in layer 1? Layer 2? Layer 3?
```

---

# SESSION 2 (~90 min)
## "fit(), callbacks, history, and saving"

## OPENING (10 min)

> *"Yesterday we built the model and configured the training.*
> *Today we actually train it — and learn all the professional tools
> that come with Keras.*
>
> *Callbacks are the big unlock: automated responses to training events.*
> *Early stopping. Learning rate reduction on plateau. Model checkpointing.*
> *These aren't advanced features — they're what everyone uses in production.*
>
> *We'll also plot the training history so you can visually diagnose
> what's happening during training, and save/load models for deployment."*

---

## SECTION 1: model.fit() — Training the Network (20 min)

Write on board:
```python
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,      # hold out 20% for validation
    verbose=1,                  # show progress bars
    callbacks=[...]             # we'll add these next
)
```

Show what the output looks like:
```
Epoch 1/100
480/480 [==============================] - 0s 2ms/step
  loss: 0.6821 - accuracy: 0.5833 - val_loss: 0.6743 - val_accuracy: 0.6125

Epoch 50/100
480/480 [==============================] - 0s 1ms/step
  loss: 0.2103 - accuracy: 0.9187 - val_loss: 0.2341 - val_accuracy: 0.9050
```

> *"Every line tells you:*
> *loss — how wrong the model is on training data (lower = better)*
> *accuracy — fraction correct on training data*
> *val_loss — how wrong on validation data (the real performance number)*
> *val_accuracy — fraction correct on validation data*
>
> *Train and val should go down together.*
> *If train keeps improving but val plateaus or rises — you're overfitting.*
> *This is exactly what we talked about in Module 05 (Regularization)."*

**Ask the room:** *"After 100 epochs, train accuracy = 97%, val accuracy = 71%.
What do you add to your model?"*

> (Dropout, L2 regularization, or reduce model complexity.)

---

## SECTION 2: Callbacks — Smart Training (25 min)

Write on board:
```python
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,        # multiply lr by 0.5 when plateau detected
        patience=7,
        min_lr=1e-6
    ),
    ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True
    )
]

history = model.fit(
    X_train, y_train,
    epochs=500,           # set high — EarlyStopping decides when to halt
    callbacks=callbacks,
    validation_split=0.2
)
```

Explain each callback:
```
EarlyStopping:
  Watch val_loss. If it doesn't improve for 15 epochs, stop.
  restore_best_weights=True: rewind to the best epoch's weights.

ReduceLROnPlateau:
  Watch val_loss. If stuck for 7 epochs, halve the learning rate.
  Often unsticks a stalled model without overfitting.

ModelCheckpoint:
  Every time val_loss improves, save the model to disk.
  Even if training crashes later, you have the best weights saved.
```

> *"These three callbacks are the standard recipe used in production.*
> *You almost always want all three together.*
>
> *Set epochs to 500 or 1000. Don't think about it.*
> *EarlyStopping will halt at the right time.*
>
> *ReduceLROnPlateau is subtle but powerful: when the model is stuck,
> smaller steps can find a better path. Like switching from hiking boots
> to ballet flats when the terrain gets precise.*
>
> *ModelCheckpoint protects you — if your laptop dies during hour 3 of training,
> you still have the best model checkpoint on disk."*

**Ask the room:** *"ReduceLROnPlateau with factor=0.5 fires three times.
Starting lr = 0.001. What is the learning rate after the third reduction?"*

> (0.001 × 0.5 × 0.5 × 0.5 = 0.000125)

---

## SECTION 3: Plotting Training History (15 min)

Write on board:
```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_title('Loss Over Epochs')
ax1.set_xlabel('Epoch')
ax1.legend()

# Accuracy curves
ax2.plot(history.history['accuracy'], label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
ax2.set_title('Accuracy Over Epochs')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=300)
```

Draw the four possible training curve patterns on board:
```
PATTERN 1 (good):          PATTERN 2 (overfit):
  loss                        loss
   \  train                    \ train
    \ val  (parallel)           \
     \_____                      \       val
                                  ↑ diverges

PATTERN 3 (underfit):      PATTERN 4 (unstable):
  loss                        loss
   ─────── (flat, no           /\/\/\/\/\
   train       learning)
   ─────── val                both noisy, no clear trend
```

> *"Plot these every time. Pattern 1: you're done.*
> *Pattern 2: add regularization from Module 05.*
> *Pattern 3: bigger model, more epochs, check data.*
> *Pattern 4: lower learning rate, add BatchNorm.*
>
> *The curves tell you exactly what to do next."*

---

## SECTION 4: Save and Load Models (10 min)

Write on board:
```python
# Save the entire model (architecture + weights + optimizer state)
model.save('my_model.h5')
model.save('my_model/')          # SavedModel format (preferred)

# Load it back
loaded_model = keras.models.load_model('my_model.h5')
loaded_model.predict(X_new)      # ready to use immediately

# Save just the weights
model.save_weights('weights.h5')

# Load just the weights (need to rebuild architecture first)
model.load_weights('weights.h5')
```

> *"model.save() saves everything: the architecture, the weights, the optimizer state.*
> *You can send this file to a colleague or deploy it to a server.*
>
> *This is deployment. Real ML systems load a saved model and call predict().*
> *You never retrain in production — you train once, save, deploy, predict.*
>
> *The h5 format is a single file. The SavedModel format is a directory.*
> *SavedModel is preferred for TensorFlow Serving (production deployment)."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
FULL KERAS WORKFLOW:
  1. Build:       Sequential([Dense(...), Dropout, Dense(...)])
  2. Summarize:   model.summary()
  3. Compile:     model.compile(optimizer, loss, metrics)
  4. Callbacks:   EarlyStopping + ReduceLROnPlateau + ModelCheckpoint
  5. Train:       history = model.fit(..., callbacks=callbacks)
  6. Evaluate:    model.evaluate(X_test, y_test)
  7. Plot:        plot history.history['loss'] and ['val_loss']
  8. Save:        model.save('model.h5')
  9. Load + use:  keras.models.load_model() → .predict()
```

**Homework — from `mlp_with_keras.py`:**
```python
# 1. Run the script. What val_accuracy does the model reach on make_circles?
# 2. Add a third hidden Dense(32) layer. Does accuracy improve or decrease?
# 3. Change EarlyStopping patience from 15 to 5. How many epochs does it train for?
# 4. What does model.summary() say the total parameter count is?
```

---

## INSTRUCTOR TIPS & SURVIVAL GUIDE

**"TensorFlow won't install / import errors"**
> *"Use: `pip install tensorflow`. If on Apple Silicon Mac: `pip install tensorflow-macos`.*
> *If still failing: `pip install tensorflow-cpu`.*
> *The module has try/except ImportError — it will show the code even without TF installed.*
> *But install TF before running this module for real results."*

**"What's the difference between h5 and SavedModel format?"**
> *"h5 is a single file, older HDF5 format, works everywhere.*
> *SavedModel is TensorFlow's native format, better for production deployment.*
> *For this course, h5 is fine. In production, use SavedModel.*"

**"When should I use Functional API instead of Sequential?"**
> *"Sequential is one input → one output, straight line of layers.*
> *Functional API handles: multiple inputs, multiple outputs, skip connections (ResNets).*
> *For everything in this course, Sequential works.*
> *You'll encounter Functional API when we get to advanced architectures in later parts."*

**"Why does model.fit return a history object?"**
> *"history.history is a dictionary: keys are metric names, values are lists — one entry per epoch.*
> *It's just Python. history.history['val_loss'][0] is val_loss after epoch 1.*
> *You can plot it, compute statistics, save it to a file.*
> *Always save the history — you want to know how training went."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening: numpy vs Keras              10 min
├── Why Keras exists                      15 min
├── Sequential API + model.summary()     20 min
├── model.compile()                       15 min
└── Close + homework                      10 min

SESSION 2  (90 min)
├── Opening bridge                        10 min
├── model.fit() and reading output        20 min
├── Callbacks (three-callback recipe)     25 min
├── Plotting training history             15 min
├── Save and load                         10 min
└── Close + homework                      10 min
```

---
*MLForBeginners · Part 3: Deep Neural Networks · Module 08*
