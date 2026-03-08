# MLForBeginners — Instructor Guide
## Part 4 · Module 05: CNN with Keras
### Two-Session Teaching Script

> **Prerequisites:** Module 04 complete. They have built Conv2D from scratch
> in NumPy, understand weight sharing, forward pass shapes, ReLU, and MaxPool.
> They also know Keras from Part 3 (MLP with Keras). Today they combine both.
> **Payoff today:** They will train a real CNN on 60,000 color images and
> visualize what features each layer has learned to detect.

---

# SESSION 1 (~90 min)
## "Building the CNN architecture in Keras — layer by layer"

## Before They Arrive
- Terminal open in `convolutional_neural_networks/algorithms/`
- CIFAR-10 downloaded if possible (first run downloads automatically)
- `cnn_with_keras.py` open but not yet run
- Whiteboard with a blank stack of 5 horizontal boxes labeled: Conv2D → MaxPool → BatchNorm → Conv2D → Dense

---

## OPENING (10 min)

> *"Last session you built Conv2D from scratch. You slid a loop across
> pixels and did dot products. You know exactly what's happening inside.*
>
> *Today we use Keras — and instead of 80 lines of numpy loops,
> the whole conv layer is one line of code. The math is identical.*
>
> *But today also introduces something new: data augmentation,
> batch normalization, and callbacks. These are the techniques that
> take a model from 'works in a notebook' to 'actually good in practice.'*
>
> *We're also working on CIFAR-10 — 60,000 real color photos across
> 10 classes. Airplane. Cat. Truck. Dog. This is the real deal."*

Draw on board:
```
CIFAR-10 at a glance:
  60,000 images   (50K train, 10K test)
  32 × 32 pixels
  3 channels (RGB)
  10 classes:

  ✈ airplane    🚗 automobile   🐦 bird       🐱 cat    🦌 deer
  🐶 dog        🐸 frog         🐴 horse      🚢 ship   🚛 truck

Why hard?
  32×32 is tiny — details are blurry
  Cats and dogs look similar
  Intra-class variation: a cat facing left vs. right looks very different
  MLP on this: ~55% accuracy   CNN on this: ~90%+ accuracy
```

---

## SECTION 1: Loading and Normalizing CIFAR-10 (15 min)

> *"Before we build the model, we need to understand what we're feeding it.*
>
> *CIFAR-10 images are (32, 32, 3) arrays of integers from 0 to 255.
> Neural networks train better when inputs are in [0, 1]. Simple divide by 255."*

```python
from tensorflow import keras
from tensorflow.keras import layers

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
y_train = y_train.flatten()
y_test  = y_test.flatten()

print(f"X_train: {X_train.shape}  dtype: {X_train.dtype}")  # (50000, 32, 32, 3) uint8
print(f"Pixel range: {X_train.min()} to {X_train.max()}")   # 0 to 255

# Normalize
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

print(f"After normalize: {X_train.min():.1f} to {X_train.max():.1f}")  # 0.0 to 1.0

# Validation split
X_tr, X_val = X_train[:45000], X_train[45000:]
y_tr, y_val = y_train[:45000], y_train[45000:]
```

**Ask the room:** *"Why do we split 50K into 45K train and 5K validation?
What would happen if we used the test set to pick hyperparameters?"*

> *(Answer: Test set contamination — the test set would no longer be a fair
> estimate of generalization. The model effectively "sees" the test data
> through our tuning choices.)*

Draw the splits:
```
50,000 training images:
┌───────────────────────────────────────┬───────────┐
│            45,000 TRAIN               │  5K VAL   │
└───────────────────────────────────────┴───────────┘
                                                         + 10,000 TEST (untouched)
Used to update weights    Used to tune HPs    Used once at the very end
```

---

## SECTION 2: Building the CNN Architecture (25 min)

> *"Now for the fun part. I want you to watch how we stack layers
> and think about what happens to the data shape at each step.*
>
> *I'll show you the full model, then we'll trace the shapes together."*

Write the model on board AND in code simultaneously:

```python
def build_cnn(input_shape=(32, 32, 3), n_classes=10):
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, (3,3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3,3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3,3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return model
```

Trace shapes on board:
```
Layer                     Output Shape       Parameters
─────────────────────────────────────────────────────────
Input                     (32, 32,  3)
Conv2D(32, 3×3, same)     (32, 32, 32)       3×3×3×32 + 32  = 896
BatchNorm                 (32, 32, 32)       128 (scale, shift, mean, var)
ReLU                      (32, 32, 32)       0
Conv2D(32, 3×3, same)     (32, 32, 32)       3×3×32×32 + 32 = 9,248
MaxPool(2×2)              (16, 16, 32)       0
Dropout(0.25)             (16, 16, 32)       0
Conv2D(64, 3×3, same)     (16, 16, 64)       3×3×32×64 + 64 = 18,496
Conv2D(64, 3×3, same)     (16, 16, 64)       3×3×64×64 + 64 = 36,928
MaxPool(2×2)              ( 8,  8, 64)       0
Flatten                   (4096,)            0
Dense(512)                (512,)             4096×512 + 512 = 2,097,664
Dense(10, softmax)        (10,)              512×10 + 10    = 5,130
─────────────────────────────────────────────────────────
Total trainable parameters: ~2.2 million
```

> *"Two observations:
> 1. The spatial size shrinks (32→16→8) while depth grows (3→32→64).
>    We're trading spatial resolution for feature richness.
> 2. Most parameters are in the Dense layer at the end — the classifier.
>    The conv layers are the cheap feature extractors."*

---

## SECTION 3: Batch Normalization — Why It Matters (15 min)

> *"BatchNorm is everywhere in modern CNNs. Let's understand it.*
>
> *Problem: as data flows through layers, the distribution shifts.
> Layer 3's input distribution changes every time Layer 2's weights update.
> This makes training unstable and slow. We call this 'internal covariate shift.'*
>
> *BatchNorm fix: after each linear transformation, normalize the activations
> to have mean=0 and std=1 within the current batch. Then let two learned
> parameters (gamma, beta) scale and shift back to whatever's useful."*

Draw on board:
```
Without BatchNorm:
  x ──→ Conv2D ──→ wildly varying values ──→ ReLU ──→ next layer confused

With BatchNorm:
  x ──→ Conv2D ──→ BN: normalize batch ──→ ReLU ──→ stable, predictable input

BN formula (per feature, per batch):
  μ_B = mean(x)            ← batch mean
  σ_B = std(x)             ← batch std
  x̂ = (x - μ_B) / σ_B    ← normalize
  y = γ × x̂ + β           ← scale and shift (γ, β are LEARNED)

Benefits:
  • Allows higher learning rates (faster convergence)
  • Acts as mild regularizer (batch mean/std adds noise)
  • Reduces sensitivity to weight initialization
```

**INSTRUCTOR TIP — "What's the order: BN before or after ReLU?"**
> Original paper: Conv → BN → ReLU. Some modern work uses Conv → ReLU → BN.
> The difference is small in practice. We use Conv → BN → ReLU because
> it normalizes before the activation, which the paper argues is more principled.
> Don't let students get stuck on this — it barely matters empirically.

---

## SECTION 4: Data Augmentation — Teaching the Model to Generalize (15 min)

> *"Here's a problem: we have 45,000 training images. A CNN with 2M parameters
> can memorize all of them. We need to stop that.*
>
> *Data augmentation: instead of feeding the same images every epoch,
> we randomly transform them on the fly. Horizontal flip, slight zoom,
> small rotation. The model sees slightly different versions each time.*
>
> *It can't memorize — it has to learn the actual features."*

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    horizontal_flip=True,       # flip left-right (50% chance)
    width_shift_range=0.1,      # shift up to 10% horizontally
    height_shift_range=0.1,     # shift up to 10% vertically
    rotation_range=15,          # rotate up to 15 degrees
    zoom_range=0.1,             # zoom in/out up to 10%
    fill_mode="nearest"         # fill new pixels with nearest neighbor
)
datagen.fit(X_tr)
```

Draw the augmentation effects:
```
Original:           H-flip:             Small rotation:     Slight zoom:
┌────────┐          ┌────────┐           ┌────────┐          ┌──────┐
│  cat   │  ─────>  │  tac   │    or     │  cat~  │    or    │  CAT │
│ facing │          │ facing │           │  tilted│          │ close│
│  left  │          │  right │           │        │          │  up  │
└────────┘          └────────┘           └────────┘          └──────┘

All are the same cat — the model must learn "cat-ness", not exact pixel positions
```

**Ask the room:** *"Would you use vertical flip for CIFAR-10? Why or why not?"*
> *(Answer: Probably not for most classes. An upside-down truck is not a truck
> the model will ever see in real life. Augmentations should reflect realistic variations.)*

---

## CLOSING SESSION 1 (10 min)

Board summary:
```
WHAT WE BUILT:
  Input (32×32×3)
  → Conv blocks: feature extraction (spatial shrinks, depth grows)
  → BatchNorm: stabilize training, normalize activations
  → Dropout: force redundancy, prevent memorization
  → Dense + Softmax: classification head

DATA PREP:
  Normalize to [0,1]
  Train/val/test splits (no test contamination!)
  Augmentation: see more variations of each image
```

**Homework:** In the Keras model above, what does `model.summary()` show for the
total number of parameters? Why is the Dense(512) layer the biggest contributor?

---

# SESSION 2 (~90 min)
## "Training, callbacks, evaluation, and feature map visualization"

## OPENING (10 min)

> *"We have an architecture and augmented data. Now we train.*
>
> *Today's additions are practical ones that every ML engineer uses:
> EarlyStopping so we don't waste compute, a confusion matrix so we
> know exactly which classes the model confuses, and my personal favorite —
> feature map visualization. We're going to look INSIDE the trained CNN
> and see what patterns each filter has learned to detect.*
>
> *That last one is genuinely cool. Let's get there."*

---

## SECTION 1: Compile and Train with Callbacks (20 min)

```python
model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=10,           # stop if no improvement for 10 epochs
        restore_best_weights=True  # roll back to best model, not last model
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,            # halve the learning rate
        patience=5,            # after 5 epochs of no improvement
        min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        "best_cnn_model.keras",
        monitor="val_accuracy",
        save_best_only=True
    )
]

history = model.fit(
    datagen.flow(X_tr, y_tr, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=100,              # EarlyStopping will cut this short
    callbacks=callbacks,
    verbose=1
)
```

> *"Three callbacks doing real work:*
>
> *EarlyStopping: stops training when val_accuracy hasn't improved for 10 epochs.
> Saves time AND prevents overfitting — best of both worlds.*
>
> *ReduceLROnPlateau: when we plateau, cut the learning rate in half.
> It's like when you're almost at the minimum of the loss surface —
> take smaller steps so you don't overshoot.*
>
> *ModelCheckpoint: save the best weights to disk. Even if we overfit
> later, we restore to the best point. 'restore_best_weights=True' in
> EarlyStopping does this automatically in memory."*

**Ask the room:** *"If EarlyStopping fires at epoch 47 with patience=10,
when did validation accuracy last improve?"*
> *(Answer: Epoch 37. It waited 10 epochs after that.)*

---

## SECTION 2: Evaluating with a Confusion Matrix (20 min)

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
```

Draw a simplified confusion matrix on board:
```
             Predicted →
             plane  auto  bird  cat  deer  dog  frog  horse  ship  truck
Actual  plane [890   10    12    5    3     2    1     4      61    12  ]
        cat   [  2   15    45  680   30   120    8    15      3     2  ]

                                     ↑
                        cat confused with dog most often
                        (they look similar at 32×32)

Diagonal = correct predictions
Off-diagonal = confusions between classes
```

> *"This is gold. Accuracy says '87% correct.' The confusion matrix tells us
> WHERE the 13% errors are.*
>
> *Cat and dog are the hardest — the model confuses them constantly.
> That makes total sense: a 32×32 image of a cat and dog can look nearly identical.*
>
> *Planes and ships are easy — very different shapes.*
>
> *When you build your own classifier, always look at the confusion matrix.
> It tells you which classes to collect more data for."*

**Ask the room:** *"If you had budget to collect 500 more training images,
which class would you pick? Why?"*

---

## SECTION 3: Visualizing Feature Maps (25 min)

> *"This is the coolest part. We're going to take a trained CNN and look
> at what each convolutional layer 'sees' when it processes an image.*
>
> *We'll create a model that outputs the activations at intermediate layers —
> Keras calls this a 'feature extraction model.' Then we'll plot the
> activation maps for a single input image."*

```python
# Build a model that outputs intermediate activations
layer_names = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3"]
feature_extractor = keras.Model(
    inputs=model.inputs,
    outputs=[model.get_layer(name).output for name in layer_names]
)

# Pick one test image
img = X_test[0:1]   # shape (1, 32, 32, 3)
feature_maps = feature_extractor.predict(img)

# Plot first 8 feature maps from Layer 1
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for idx, ax in enumerate(axes.flat):
    ax.imshow(feature_maps[0][0, :, :, idx], cmap="viridis")
    ax.set_title(f"Filter {idx}")
    ax.axis("off")
```

Draw what feature maps look like on board:
```
Original image (airplane):    After Conv Block 1 (32 feature maps):

┌────────────────┐            Filter 0:  Filter 1:  Filter 2:  Filter 3:
│                │            ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
│    ══════      │            │ edge │   │ wing │   │ sky  │   │ body │
│   /      \     │  ──────>   │ resp │   │ resp │   │ tone │   │shape │
│  /__________\  │            └──────┘   └──────┘   └──────┘   └──────┘
│                │
└────────────────┘            Each filter highlights something different!
                              Some detect edges, some detect textures,
                              some respond to color regions.

                              Later layers: more abstract, harder to interpret
                              Final layers: "looks like airplane" vs "looks like truck"
```

> *"Early layers: you can see edges, colors, simple textures.
> They're interpretable to a human eye.*
>
> *Middle layers: more complex — combinations of edges. Harder to label.*
>
> *Deep layers: highly abstract. The network has its own 'concepts'
> that don't have human-readable names. That's both fascinating and
> a bit unsettling — the model works, but we can't fully explain it.*
>
> *This is an active research area called 'explainability.' For now,
> knowing the general pattern — early = simple, deep = abstract —
> is what matters."*

---

## SECTION 4: Saving and Reloading the Model (5 min)

```python
# Save
model.save("cifar10_cnn.keras")
print("Model saved!")

# Reload — identical to original
loaded_model = keras.models.load_model("cifar10_cnn.keras")
test_acc = loaded_model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Reloaded model accuracy: {test_acc:.4f}")

# Single image inference
def predict_image(model, img_array):
    """Predict class for one (32,32,3) image array."""
    img_input = img_array[np.newaxis, ...]  # add batch dim
    probs = model.predict(img_input, verbose=0)[0]
    pred_class = CLASS_NAMES[np.argmax(probs)]
    confidence = probs.max()
    return pred_class, confidence
```

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
TRAINING:
  Adam optimizer + sparse_categorical_crossentropy
  EarlyStopping     → stops when val_accuracy plateaus (saves time)
  ReduceLROnPlateau → shrinks LR when progress stalls
  ModelCheckpoint   → saves best weights automatically

EVALUATION:
  Accuracy        → overall score (simple but limited)
  Confusion matrix → which classes are confused (actionable)
  Per-class report → precision/recall per class

INSPECTION:
  Feature extraction model → see intermediate activations
  Early layers = simple features, late layers = abstract concepts
```

**Homework — extend the experiment:**
```python
# Try one of these modifications and report what changes:
# 1. Remove BatchNormalization from all layers. Does training stability change?
# 2. Add a third conv block (Conv2D(128) + Conv2D(128) + MaxPool).
#    Does accuracy improve? Does training time change?
# 3. Change horizontal_flip to vertical_flip in augmentation. What effect?
```

---

## INSTRUCTOR TIPS

**"Why sparse_categorical_crossentropy and not categorical_crossentropy?"**
> *"'Sparse' means your labels are integers (0, 1, 2...). Regular categorical
> expects one-hot vectors ([0,0,1,0...]).  We use sparse because it's
> simpler — no need to convert labels. Both compute the same loss."*

**"Model accuracy is stuck at 60% — what's wrong?"**
> *"Check in order: (1) Did you normalize the inputs to [0,1]?
> (2) Is the learning rate too high or too low?
> (3) Is the batch size too small (noisy gradients)?
> (4) Did BatchNorm come before or after activation?
> Most 'stuck training' issues are normalization or learning rate."*

**"Should we always use EarlyStopping?"**
> *"Yes, almost always. The only reason not to is if you have a fixed
> compute budget and want to use every epoch. But restore_best_weights=True
> means you're never actually hurting yourself — worst case you just
> trained a bit longer than needed."*

**"What's the difference between Dropout(0.25) and Dropout(0.5)?"**
> *"0.25 = drop 25% of neurons randomly per batch.
> 0.5 = drop 50%. Use higher dropout in fully connected layers (Dense),
> lower in conv layers. Conv layers have fewer parameters and are
> already regularized by weight sharing, so they need less dropout."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening — CIFAR-10 overview             10 min
├── Load, normalize, split data             15 min
├── Build CNN architecture, trace shapes    25 min
├── BatchNorm explanation                   15 min
├── Data augmentation                       15 min
└── Close + homework                        10 min

SESSION 2  (90 min)
├── Opening bridge                          10 min
├── Compile, callbacks, train               20 min
├── Confusion matrix evaluation             20 min
├── Feature map visualization               25 min
├── Save and reload model                    5 min
└── Close + homework                        10 min
```

---
*MLForBeginners · Part 4: CNNs · Module 05*
