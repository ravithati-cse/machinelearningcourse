# MLForBeginners — Instructor Guide
## Part 4 · Module 08: CIFAR-10 Classifier (Project 1)
### Single 120-Minute Project Session

> **Prerequisites:** All 4 CNN algorithm modules complete (conv from scratch,
> Keras CNN, classic architectures, transfer learning). They can build,
> train, and evaluate CNNs. Today they apply everything to a real benchmark.
> **Payoff today:** They will build THREE classifiers (MLP baseline, CNN, transfer
> learning), compare them side by side, and see exactly why CNNs beat MLPs
> on color image classification. They will visualize the actual learned filters.

---

## Before They Arrive
- Terminal open in `convolutional_neural_networks/projects/`
- `cifar10_classifier.py` open but not yet run
- Whiteboard ready with "MLP vs CNN vs Transfer Learning" as three columns
- TensorFlow installed and confirmed working
- Dataset will auto-download on first run (~170MB) — do this beforehand if on slow internet:
  ```python
  from tensorflow import keras; keras.datasets.cifar10.load_data()
  ```

---

## OPENING (10 min)

> *"You have all the tools. Today is about using them together.*
>
> *CIFAR-10 is the benchmark that every serious CNN paper tests on.
> 60,000 real color photos across 10 categories. Airplanes. Cats. Trucks.*
>
> *We're going to do something deliberate today: build a bad model first.*
> *On purpose. We'll use an MLP — no convolutions — and see it struggle.*
> *Then we'll build a CNN and watch the gap.*
> *Then we'll apply transfer learning and push further.*
>
> *By the end you'll have built three classifiers on the same dataset
> and have a very concrete understanding of WHAT each approach gives you.*
>
> *This is how professional ML engineers work: build a baseline,
> improve systematically, understand why each improvement helps."*

Draw the plan on board:
```
PROJECT PLAN:
┌─────────────────┬──────────────────┬───────────────────────┐
│   STEP 1        │    STEP 2        │      STEP 3           │
│   MLP Baseline  │  CNN from Keras  │  Transfer Learning    │
│   ~55% accuracy │  ~85-87% target  │  ~88-90% target       │
│                 │                  │                       │
│  "Let's see how │  "Now convolutions│ "Now 1.2M pre-trained │
│   bad flat is"  │   do the work"   │  images help us"      │
└─────────────────┴──────────────────┴───────────────────────┘

After all three: confusion matrix, per-class analysis, filter visualization
```

---

## PART 1: Load and Explore CIFAR-10 (15 min)

Run the data exploration section together:

```bash
python3 cifar10_classifier.py
```

Stop after the data loading output and discuss. Write what you see on board:

```
CIFAR-10 statistics:
  Training:  X_train = (50000, 32, 32, 3)   y_train = (50000,)
  Test:      X_test  = (10000, 32, 32, 3)   y_test  = (10000,)
  Pixel range: 0 to 255 (uint8)
  Classes: balanced — each class has exactly 5,000 training images

Compare to MNIST:
  MNIST: (28, 28, 1) — grayscale digits, very clean
  CIFAR: (32, 32, 3) — color photos, MUCH more variation
```

> *"Look at those sample images when the script opens the visualization.*
>
> *32×32 pixels is TINY. Scroll closer on the screen. That horse is basically
> a brown blur. That cat could be a dog. That ship could be a truck.*
>
> *Human accuracy on CIFAR-10 is about 94%. We should be able to get
> close to that with a good CNN. Our MLP won't."*

Draw what a single CIFAR image looks like as data:
```
One CIFAR-10 image = array of shape (32, 32, 3):

  Pixel (0, 0):  R=120, G=85, B=60   ← top-left corner, brownish
  Pixel (0, 1):  R=115, G=82, B=58
  ...
  Pixel (16, 16): R=200, G=180, B=120  ← middle, sky region (lighter)
  ...
  Pixel (31, 31): R=80, G=60, B=40    ← bottom-right

  Flattened for MLP: 32×32×3 = 3072 numbers per image
  For CNN: kept as (32, 32, 3) — spatial structure preserved
```

**Ask the room:** *"If we flatten the 32×32×3 image for an MLP, what spatial
information do we lose?"*
> *(Answer: The fact that pixel (5,6) is next to pixel (5,7) — neighbor relationships.
> A pixel in the top-left and a pixel in the bottom-right become 'equally close'
> to every weight in the Dense layer. CNNs preserve this structure.)*

---

## PART 2: MLP Baseline — Establish the Ceiling of a Bad Model (15 min)

> *"We build the MLP not to win. We build it to understand what problem
> convolutions solve. Watch how flat it gets."*

```python
mlp_model = keras.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),   # 3072 inputs
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
], name="MLP_Baseline")

mlp_model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
```

Run it (or show the output from the script):
```
MLP training output (typical):
  Epoch 1:  val_acc: 0.42
  Epoch 5:  val_acc: 0.49
  Epoch 10: val_acc: 0.52
  Epoch 20: val_acc: 0.54
  ...
  Final test accuracy: ~55%
```

> *"55%. That's not random (random would be 10% for 10 classes).
> But it's not good either.*
>
> *The MLP is treating every pixel independently.
> It can't see that the pixels in the top-right corner form a wing shape.
> It can only see: 'there are some high-value pixels somewhere.'*
>
> *Also notice: the MLP has 512×3072 = 1.5M parameters just in the first layer.
> It has a LOT of parameters but still gets 55%.
> Our CNN will have fewer parameters and do much better.*
>
> *That's the power of inductive bias — building the right assumption
> (locality and translation invariance) into the architecture."*

---

## PART 3: CNN with BatchNorm and Data Augmentation (25 min)

> *"Now let's do it properly. Two conv blocks, BatchNorm, Dropout,
> and data augmentation. This is production-quality code for CIFAR-10."*

Walk through building the model (students should have seen most of this in Module 05):

```python
def build_production_cnn(input_shape=(32,32,3), n_classes=10):
    model = keras.Sequential([
        # Conv Block 1
        layers.Conv2D(32, (3,3), padding="same", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, (3,3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Conv Block 2
        layers.Conv2D(64, (3,3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3,3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation="softmax"),
    ])
    return model
```

Draw the shape flow:
```
Input (32×32×3)
  → Conv(32) + BN + ReLU  → (32×32×32)
  → Conv(32) + BN + ReLU  → (32×32×32)
  → MaxPool(2×2)          → (16×16×32)
  → Dropout(0.25)
  → Conv(64) + BN + ReLU  → (16×16×64)
  → Conv(64) + BN + ReLU  → (16×16×64)
  → MaxPool(2×2)          → (8×8×64)
  → Dropout(0.25)
  → Flatten               → (4096,)
  → Dense(512)            → (512,)
  → Dropout(0.5)
  → Dense(10, softmax)    → (10,)   ← one probability per class
```

Set up augmentation and train:
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=15,
    zoom_range=0.1,
)
datagen.fit(X_tr)

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=10,
                                   restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
]

cnn_history = cnn_model.fit(
    datagen.flow(X_tr, y_tr, batch_size=64),
    validation_data=(X_val, y_val),
    epochs=100,
    callbacks=callbacks,
)
```

Watch training output together. Point out:
- Early epochs: rapid improvement (25%→50%→70%)
- Middle: slower improvement, BN helping stability
- EarlyStopping fires when val_accuracy plateaus

Typical result: **~85-87% test accuracy**

> *"Compare: MLP 55%, CNN 87%. That's 32 percentage points from ONE architectural
> change — switching from Dense to Conv layers.*
>
> *The CNN didn't get more parameters. It just got the RIGHT structure:
> local connectivity and weight sharing."*

---

## PART 4: Transfer Learning Comparison (15 min)

> *"Can we do better? MobileNetV2 was trained on 1.2 million images.
> Let's use it on our 45,000."*

Show the transfer learning code (students wrote this in Module 07):

```python
base = keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),    # upsample 32×32 to 96×96 for MobileNetV2
    include_top=False,
    weights="imagenet"
)
base.trainable = False

tl_model = keras.Sequential([
    layers.Lambda(lambda x: tf.image.resize(x, (96, 96))),  # resize layer
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax"),
])
```

> *"We upsample from 32×32 to 96×96 because MobileNetV2 was designed for
> larger images. We're stretching our small images, but the pretrained
> features still extract useful information.*
>
> *A Lambda layer handles the resize inside the model —
> it applies tf.image.resize to each batch on the fly."*

Typical result: **~88-90% test accuracy**

Write the three-way comparison on board:
```
MODEL               PARAMS    TRAINING TIME    TEST ACCURACY
───────────────────────────────────────────────────────────────
MLP baseline        ~2M       ~5 min           ~55%
CNN (from scratch)  ~2.2M     ~30 min          ~85-87%
Transfer (MobileNet) ~3.4M*   ~15 min          ~88-90%

* most params are frozen — only ~330K trainable
  so effectively: 330K trainable, trains FASTER than our scratch CNN
```

---

## PART 5: Confusion Matrix and Per-Class Analysis (15 min)

> *"Accuracy is a single number. The confusion matrix tells a story.*
>
> *Which classes does our best model confuse? Let's find out."*

```python
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred = np.argmax(tl_model.predict(X_test), axis=1)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=CIFAR_CLASSES))
```

Draw a simplified confusion matrix on board, highlighting the interesting cells:
```
                Predicted →
                plane auto bird cat deer dog frog horse ship truck
Actual plane  [ 891   13   12   5    3    2   1    4    57    12  ]
       cat    [   3   18   38  672   25  118   8   15    3     0  ]
       dog    [   2    8   26  115   18  793   9   15    5     9  ]
       ship   [  52    9    2   1    0    0   0    1   923    12  ]
       truck  [   8   72    3   0    0    0   1    2    11   903  ]

Easiest class: ship (92.3% correct — distinctive shape)
Hardest class: cat  (67.2% correct — confused with dog 11.8% of time)
2nd hardest:   dog  (79.3% correct — confused with cat 11.5% of time)
```

> *"Cat vs dog is the famous failure mode of CIFAR-10 classifiers.*
> *At 32×32, a sitting cat and sitting dog look almost identical.*
> *The model doesn't fail because it's stupid — it fails because
> the input literally doesn't have enough information to be certain.*
>
> *What could we do? Collect more cat and dog images.
> Use higher-resolution images (64×64 or 128×128 instead of 32×32).*
> *The confusion matrix tells us exactly where to invest.*"

**Ask the room:** *"Looking at the ship row — 52 ships were predicted as airplanes.
Why might that happen?"*
> *(Answer: Both have elongated shapes and often appear against a sky-like background.
> The model doesn't have enough context to distinguish without seeing the water below.)*

---

## PART 6: Filter Visualization (10 min)

> *"Final piece: let's look inside our trained CNN and see what the
> first layer's filters have learned to detect."*

```python
# Get weights of the first conv layer
first_conv = cnn_model.layers[0]
weights = first_conv.get_weights()[0]   # shape: (3, 3, 3, 32)

# Plot the 32 filters as RGB images
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for idx, ax in enumerate(axes.flat):
    w = weights[:, :, :, idx]
    # Normalize to [0, 1] for display
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)
    ax.imshow(w)
    ax.axis("off")
plt.suptitle("Learned first-layer filters (32 filters, 3×3×3 each)")
```

Open the saved visualization and discuss:
> *"These 3×3 filters look noisy and small — hard to interpret at first.*
>
> *But look carefully. Some filters are oriented — one axis bright,
> the other dark. Those detect edges. Some are uniform — those detect color.
> Some show a diagonal stripe — those detect diagonal edges.*
>
> *A CNN trained from scratch on CIFAR-10 discovered edge detection
> on its own. We didn't tell it to look for edges. Gradient descent
> found that edge detection is useful for classifying airplanes and trucks.*
>
> *That's what learning means in this context: discovering structure
> that helps the task."*

---

## CLOSING (5 min)

Write on board:
```
PROJECT 1 COMPLETE:

Three approaches on CIFAR-10 (60,000 color images, 10 classes):
  MLP baseline:    ~55%  — no spatial structure, pixel soup
  CNN from scratch: ~87%  — local connectivity + weight sharing
  Transfer learning: ~90%  — 1.2M ImageNet images + our 50K

Key learnings:
  1. Architecture matters more than parameter count
  2. Always build a baseline first — measure the gain from each addition
  3. Confusion matrix > accuracy for understanding failure modes
  4. Cat vs dog is hard at 32×32 — the pixels literally don't tell the full story
  5. Learned filters = edge/texture/color detectors, discovered by gradient descent

You can now build production-quality image classifiers on standard benchmarks.
Next: build one on YOUR OWN images.
```

**Homework:**
```python
# Exploration challenge: pick ONE of these and report results next session:

# Option A: Add a third conv block (Conv64→Conv128→MaxPool) to the from-scratch CNN.
#           Does accuracy improve? How does training time change?

# Option B: Try augmentation with cutout (randomly zero out 8×8 patches).
#           Does regularization help or hurt?

# Option C: Change the optimizer from Adam to SGD with momentum=0.9.
#           Which converges faster? Which gives better final accuracy?
```

---

## INSTRUCTOR TIPS

**"Why does transfer learning beat from-scratch with FEWER trainable parameters?"**
> *"Because the 3.4M frozen parameters are USEFUL parameters — they already
> know how to detect edges, shapes, textures. Our 2.2M scratch parameters
> need to discover all of that from 45,000 images. The pretrained model
> starts with knowledge; the scratch model starts with noise."*

**"My CNN is stuck at 72% accuracy — what's wrong?"**
> *"Check in order:
> 1. Did you normalize to [0,1]? (most common mistake)
> 2. Is BatchNorm BEFORE or AFTER the activation? (should be before)
> 3. Is your learning rate too high? (try 0.0005)
> 4. Are you using data augmentation? (turn it off first to debug)
> 5. Is EarlyStopping firing too early? (increase patience to 15)"*

**"Can this approach work on larger images?"**
> *"Yes, but you need to adjust:
> More conv blocks (spatial needs to shrink more before Flatten)
> Larger batch size or smaller learning rate
> More data augmentation
> Transfer learning becomes even MORE important at larger image sizes"*

**"What makes CIFAR-10 harder than MNIST?"**
> *"Three things: color (3 channels vs 1), real-world variation (a cat can
> look many ways, a digit looks basically one way), and inter-class similarity
> (cat/dog, plane/ship look similar). MNIST has clean, centered, grayscale
> digits with consistent orientation. CIFAR has the messiness of the real world."*

**"Is 90% accuracy on CIFAR-10 good?"**
> *"Yes for this setup. The state of the art is ~99% using large models
> and extensive augmentation (EfficientNet + AutoAugment).
> Our goal was to demonstrate the MLP→CNN→Transfer improvement.
> 90% in 20 minutes of training is genuinely impressive."*

---

## Quick Reference

```
PROJECT SESSION (120 min)
├── Opening — project overview, three models   10 min
├── Load and explore CIFAR-10                  15 min
├── MLP baseline — establish floor             15 min
├── CNN with BatchNorm + augmentation          25 min
├── Transfer learning comparison               15 min
├── Confusion matrix + per-class analysis      15 min
├── Filter visualization                       10 min
└── Closing summary + homework                  5 min
                                          ──────────
                                           110 min + 10 min buffer
```

---
*MLForBeginners · Part 4: CNNs · Module 08*
