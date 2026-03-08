# MLForBeginners — Instructor Guide
## Part 4 · Module 07: Transfer Learning
### Two-Session Teaching Script

> **Prerequisites:** Module 06 complete. They understand LeNet through ResNet,
> skip connections, vanishing gradients, and can build CNNs in Keras. They know
> what pretrained weights means (the module introduces using them). Today they
> learn to USE pre-trained ImageNet models rather than train from scratch.
> **Payoff today:** They will take a model trained on 1.2 million images and
> adapt it to a new task with 200 images — in minutes, not days.

---

# SESSION 1 (~90 min)
## "What gets transferred — the feature hierarchy of a pretrained CNN"

## Before They Arrive
- Terminal open in `convolutional_neural_networks/algorithms/`
- `transfer_learning.py` open but not yet run
- Whiteboard ready — draw two columns: "Hire an expert" vs "Train from scratch"
- Optional: have a photo of a dog and a cat on screen — we'll use the hiring analogy

---

## OPENING (10 min)

> *"Imagine you need to build a system to identify X-ray scans —
> normal vs abnormal. You have 500 labeled X-ray images.*
>
> *Option A: hire a junior radiologist with no experience.
> Train them from scratch on your 500 scans. They've never seen
> images before. They need to learn: what is contrast? what is an edge?
> what is a shape? Then learn what makes bones look normal.*
>
> *Option B: hire an experienced radiologist who has already read
> 1.2 million images. They already know how images work, how shapes work,
> how textures work. You just need to teach them what your specific scans
> look like. 500 examples is plenty.*
>
> *Transfer learning is Option B. We're hiring an expert.*
>
> *The expert is a CNN trained on ImageNet — 1.2 million images, 1000 categories,
> 2 weeks on 8 GPUs. We use their brain. We just add a new head for our task."*

Draw on board:
```
TRAINING FROM SCRATCH:                TRANSFER LEARNING:

Random weights                        Pre-trained ImageNet weights
(random noise)                        (knows edges, shapes, textures)
    │                                     │
    ▼                                     ▼
Train on YOUR data                    Freeze base → train only top layers
  → needs lots of data                  → works with small datasets
  → takes a long time                   → converges very fast
  → often fails with < 1000 samples     → often great with 100-500 samples

When to use from scratch:             When to use transfer learning:
  • Data is very different from         • Data is similar to natural images
    natural images (radar, seismic)     • Dataset is small (< 10K images)
  • You have millions of images         • Training time is limited
  • The domain is truly unique          • Almost always — default choice
```

---

## SECTION 1: What Gets Transferred — The Feature Hierarchy (20 min)

> *"To understand transfer learning, you need to understand WHAT is being transferred.*
>
> *A CNN trained on ImageNet learns features in a hierarchy.
> Early layers are universal — they work for any image.
> Late layers are task-specific — they recognize ImageNet categories.*
>
> *When we transfer, we KEEP the universal layers and REPLACE the specific ones."*

Draw on board (this diagram is central — take time with it):
```
  Input image
      │
      ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Block 1                                                             │
  │  Learns: edges (horizontal, vertical, diagonal)                      │
  │          color contrasts, brightness gradients                       │
  │  ✓ HIGHLY TRANSFERABLE — same for any image, anywhere, always        │
  └──────────────────────────────────────────────────────────────────────┘
      │
      ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Block 2                                                             │
  │  Learns: corners, curves, simple shapes                              │
  │          texture patterns (stripes, dots, grids, repeating patterns) │
  │  ✓ VERY TRANSFERABLE — universal image grammar                       │
  └──────────────────────────────────────────────────────────────────────┘
      │
      ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Block 3                                                             │
  │  Learns: object parts (eyes, wheels, windows, buttons)               │
  │          complex textures (fur, glass, wood, fabric)                 │
  │  ~ MODERATELY TRANSFERABLE — depends on domain similarity            │
  └──────────────────────────────────────────────────────────────────────┘
      │
      ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Block 4+                                                            │
  │  Learns: high-level semantics ("this looks like a dog", "wheel-like")│
  │          ImageNet-specific combinations of parts                     │
  │  ✗ LESS TRANSFERABLE — too specific to ImageNet categories           │
  └──────────────────────────────────────────────────────────────────────┘
      │
      ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │  Classification head (Dense + Softmax → 1000 classes)                │
  │  ✗ NOT TRANSFERABLE — predicts ImageNet classes, not your classes    │
  │  We always REMOVE this and add our own head                          │
  └──────────────────────────────────────────────────────────────────────┘
```

> *"When we do transfer learning, we throw away the classification head —
> it outputs 1000 ImageNet categories, we don't want that.*
>
> *We keep everything else. Then we add our own Dense layers at the top
> that output OUR classes.*
>
> *Whether we freeze or fine-tune the base depends on our dataset size.
> That's the key decision — we'll cover it in the next section."*

**Ask the room:** *"You want to classify 3 types of skin lesions from photos.
A doctor labeled 300 images. Should you train from scratch or use transfer learning?"*

> *(Answer: Transfer learning, definitely. Skin lesion photos are natural images —
> the edge and texture features from ImageNet are directly useful.
> 300 images is not enough to train from scratch.)*

---

## SECTION 2: Loading a Pretrained Model in Keras (15 min)

> *"Keras ships with pretrained models built in. MobileNetV2 and ResNet50
> are the two we'll use. Let's see how to load them."*

```python
from tensorflow import keras
from tensorflow.keras import layers

# Load MobileNetV2 WITHOUT the classification head (include_top=False)
base_model = keras.applications.MobileNetV2(
    input_shape=(128, 128, 3),   # we can resize to whatever we need
    include_top=False,            # drop the 1000-class head
    weights="imagenet"            # download pretrained ImageNet weights
)

print(f"MobileNetV2 base: {base_model.count_params():,} parameters")
base_model.summary()
```

> *"include_top=False: this is the critical flag. Without the top,
> we get the feature extractor — the convolutional base — without
> the classification layers.*
>
> *weights='imagenet': downloads ~14MB of weights trained on 1.2M images.
> First run downloads. Subsequent runs use cached file.*
>
> *MobileNetV2: designed for mobile devices. 3.4M parameters.
> Fast, lightweight, still very accurate. Great default choice.*
>
> *ResNet50: the classic. 25M parameters. Deeper and more powerful.
> Use when accuracy matters more than speed."*

Compare popular options:
```
Model         Params    Size(MB)  Top-5 Acc    Speed    Best for
───────────────────────────────────────────────────────────────────
MobileNetV2   3.4M      14MB      90.1%        Fast     Small datasets, mobile
ResNet50      25.6M     98MB      92.9%        Medium   Accuracy-focused work
VGG16         138M      528MB     90.4%        Slow     Legacy benchmarks
EfficientNetB0 5.3M     29MB      93.3%        Fast     Best efficiency/accuracy
InceptionV3   23.9M     92MB      93.9%        Medium   Complex images
```

---

## SECTION 3: Two Modes — Feature Extraction vs Fine-Tuning (20 min)

> *"This is the key strategic decision in transfer learning.*
>
> *Mode 1 — Feature Extraction: freeze ALL pretrained layers.
> Only train your new classification head. Fast, stable, best for small datasets.*
>
> *Mode 2 — Fine-Tuning: unfreeze the top layers of the base AND train your head.
> The top layers adjust to your specific data. More powerful but needs more data
> and a very low learning rate."*

Write on board:
```
FEATURE EXTRACTION MODE:                 FINE-TUNING MODE:
(freeze base)                            (unfreeze top N layers)

┌─────────────────────────┐              ┌─────────────────────────┐
│  MobileNetV2 base       │  ← FROZEN   │  MobileNetV2 base       │
│  (ImageNet weights,     │             │  Bottom layers: FROZEN   │
│  weights DON'T update)  │             │  Top layers: ← TRAINING  │
└─────────────────────────┘             └─────────────────────────┘
           │                                        │
┌─────────────────────────┐              ┌─────────────────────────┐
│  GlobalAveragePooling2D │  ← TRAINING  │  GlobalAveragePooling2D │  ← TRAINING
│  Dense(256, relu)       │             │  Dense(256, relu)        │
│  Dropout(0.3)           │             │  Dropout(0.3)            │
│  Dense(n_classes)       │             │  Dense(n_classes)        │
└─────────────────────────┘              └─────────────────────────┘

lr = 0.001 (normal)                     lr = 0.0001 (10x lower!)
Epochs: 10-20                           Epochs: 5-15 additional
Use when: < 1000 samples                Use when: > 1000 samples, high accuracy needed
```

> *"Why lower learning rate for fine-tuning?*
>
> *The pretrained weights are already in a good place — they took 2 weeks
> to find on 1.2M images. If you train with a high learning rate, you
> destroy that knowledge in a few batches. Use a tiny learning rate
> so the weights nudge gently toward your task, not leap away from
> what they already know.*
>
> *Rule of thumb: fine-tuning learning rate = 1/10 of feature extraction rate."*

**INSTRUCTOR TIP — Common mistake:**
> Students often try fine-tuning immediately on tiny datasets (50 images)
> and get terrible results. They've destroyed the pretrained weights without
> enough data to relearn. Rule: if you have fewer than 500 images per class,
> feature extraction only. Fine-tuning needs at least 1000+ samples total.

---

## CLOSING SESSION 1 (15 min — demo)

Run the feature hierarchy visualization:
```bash
python3 transfer_learning.py
```

Watch the output — it prints the layer-by-layer description of what gets transferred.

Open the visuals in `convolutional_neural_networks/visuals/transfer_learning/`.

Board summary:
```
WHAT TRANSFERS:
  Blocks 1-2: edges, textures → ALWAYS transfer (keep frozen)
  Blocks 3-4: shapes, parts  → usually transfer (keep or fine-tune)
  Head:       ImageNet classes → NEVER transfer (replace with your head)

MODE 1: Feature Extraction
  base_model.trainable = False
  Fast, works with tiny datasets, great baseline

MODE 2: Fine-Tuning
  Unfreeze top layers, use tiny learning rate (0.0001)
  More powerful, needs more data, do AFTER feature extraction
```

**Homework:** Load MobileNetV2 with `include_top=False`. Print `base_model.layers`
and count how many layers are in the convolutional base. At which layer index
does Block 5 start? (Hint: look for 'block_5' in layer names.)

---

# SESSION 2 (~90 min)
## "Feature extraction, fine-tuning — and when to use each"

## OPENING (10 min)

> *"Last session we understood what gets transferred and the two modes.*
>
> *Today we write the actual code. Step by step:
> load MobileNetV2, freeze it, add our head, train.
> Then unfreeze the top layers, drop the learning rate, fine-tune.*
>
> *We'll also see the data size experiment: same model architecture
> tested on 200 images vs 2000 images vs 20000 images.
> When does fine-tuning help? When does it hurt?*
>
> *This is the decision-making knowledge that makes you effective in practice."*

---

## SECTION 1: Feature Extraction — Full Code (20 min)

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_feature_extractor(base_model, n_classes, input_shape=(128, 128, 3)):
    """
    Feature extraction: freeze ALL of base_model, train only the head.
    """
    # Step 1: Freeze the entire base
    base_model.trainable = False

    # Step 2: Build our new classification head
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)   # training=False: keep BN in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model

# Build and compile
base = keras.applications.MobileNetV2(input_shape=(128,128,3),
                                       include_top=False, weights="imagenet")
model = build_feature_extractor(base, n_classes=5)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# Trainable params: ~330K (our head only)
# Non-trainable params: ~2.2M (the frozen base)
```

> *"Two numbers in model.summary() matter: trainable and non-trainable.*
>
> *Non-trainable: the frozen pretrained weights. They forward-pass data
> through, but gradients don't update them.*
>
> *Trainable: our small head — ~330K parameters. That's all that updates.*
>
> *We're training 330K parameters instead of 3.4M. 10× faster, 10× less data needed."*

Point out `training=False`:
> *"When you pass training=False to the base model, BatchNorm layers
> use their stored running statistics instead of computing batch statistics.
> This is important: if you allow BN to compute new statistics on your small
> dataset, it will corrupt the pretrained normalization. Always use
> training=False when the base is frozen."*

---

## SECTION 2: Fine-Tuning — Unfreeze the Top (20 min)

> *"After feature extraction converges, we unlock the top layers
> and let them adjust. This is the fine-tuning phase.*
>
> *The exact number of layers to unfreeze is a hyperparameter.
> Common practice: unfreeze the last 1-3 blocks (20-40% of the base).*
>
> *Critical: use a very small learning rate. We're nudging, not relearning."*

```python
def fine_tune(model, base_model, fine_tune_at_layer=100, learning_rate=1e-4):
    """
    Fine-tune: unfreeze top layers of base_model, train with low LR.
    Call AFTER feature extraction is complete.
    """
    # Unfreeze the base model
    base_model.trainable = True

    # Freeze all layers BEFORE fine_tune_at_layer
    for layer in base_model.layers[:fine_tune_at_layer]:
        layer.trainable = False

    # Recompile with a very low learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Count trainable params now
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"Fine-tuning: {trainable:,} trainable parameters")
    return model

# Phase 1: Feature extraction
history_fe = model.fit(train_dataset, epochs=20, validation_data=val_dataset,
                       callbacks=[keras.callbacks.EarlyStopping(patience=5,
                                  restore_best_weights=True)])

# Phase 2: Fine-tuning
model = fine_tune(model, base, fine_tune_at_layer=100, learning_rate=1e-4)
history_ft = model.fit(train_dataset, epochs=10, validation_data=val_dataset,
                       callbacks=[keras.callbacks.EarlyStopping(patience=5,
                                  restore_best_weights=True)])
```

Draw the two-phase training curve:
```
Accuracy
  │
  │                        ╭────── fine-tune (phase 2)
  │             ╭──────────╯
  │       ╭─────╯ ← feature extraction (phase 1) converges
  │  ╭────╯
  │──╯
  └────────────────────────────────── Epochs
       phase 1 (fast)        phase 2 (slow, careful)

Phase 1: big jumps, rapid learning (new head learning from scratch)
Phase 2: small improvements, gradual tuning (base weights adjusting)
```

---

## SECTION 3: The Dataset Size Decision (20 min)

> *"Now the most practical question: given YOUR dataset, which strategy?*
>
> *Let's see the experiment from the script: we test three dataset sizes
> and compare feature extraction vs fine-tuning vs training from scratch."*

```bash
python3 transfer_learning.py
```

Watch and discuss the comparison plot. Draw the summary on board:

```
Dataset size     From Scratch    Feature Extract   Fine-Tune
──────────────────────────────────────────────────────────────
50  images/class   ~35%            ~75-80%           ~65%  ←worse! (overfits)
200 images/class   ~45%            ~80-85%            ~83%
1000 images/class  ~65%            ~84-87%            ~88%
5000 images/class  ~78%            ~87-89%            ~91%

KEY INSIGHT:
  Small dataset (< 200/class):  feature extraction WINS over fine-tuning
  Medium dataset:               fine-tuning slightly better
  Large dataset (> 5000/class): fine-tuning clearly better, scratch becomes viable
```

> *"With 50 images per class, fine-tuning was WORSE than feature extraction.
> Why? Because we gave the top layers permission to change, but didn't have
> enough data to guide those changes. They drifted randomly from their
> ImageNet initialization.*
>
> *The frozen feature extractor, however, already knows how to extract good
> features. 50 images is enough to train a small Dense head.*
>
> *Rule: always start with feature extraction. Only switch to fine-tuning
> if you have enough data and feature extraction has plateaued."*

**Ask the room:** *"You're building a classifier for 5 types of satellite imagery.
You have 400 images per class. ImageNet contains no satellite images.
Should you still use transfer learning?"*

> *(Answer: Probably yes — the low-level edge and texture detectors still transfer.
> But fine-tuning more layers will be important since the domain is different.
> Start with feature extraction, evaluate, then fine-tune more aggressively
> than you would for natural photo datasets.)*

---

## SECTION 4: Quick Comparison — MobileNetV2 vs ResNet50 (10 min)

```python
# Load ResNet50 (slightly different normalization preprocessing)
resnet_base = keras.applications.ResNet50(
    input_shape=(128, 128, 3),
    include_top=False,
    weights="imagenet"
)
```

Draw comparison:
```
Property         MobileNetV2         ResNet50
──────────────────────────────────────────────
Parameters         3.4M               25.6M
Download size      14MB               98MB
Inference speed    Fast               Medium
ImageNet accuracy  90.1%              92.9%
Best use case      Mobile, small DS   Accuracy-focused
Depth              53 layers          50 layers
Architecture       Inverted residuals Standard residuals

When to pick MobileNetV2:
  • Dataset < 5000 images
  • Deployment on edge devices
  • Prototyping quickly

When to pick ResNet50:
  • Dataset > 5000 images
  • Server-side deployment
  • Need maximum accuracy
```

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
TRANSFER LEARNING CHEATSHEET:

1. Always start with a pretrained base (MobileNetV2 is a great default)
2. always set include_top=False, add your own Dense head
3. Phase 1 — Feature Extraction:
     base_model.trainable = False
     learning_rate = 0.001
     Train until converged
4. Phase 2 — Fine-Tuning (if > 1000 samples):
     Unfreeze top 30-40% of base
     learning_rate = 0.0001 (10x smaller)
     Train 5-15 more epochs
5. Always use training=False for frozen base (preserves BN statistics)
```

**Homework — build your own:**
```python
# Find any 3-class image dataset online (Kaggle, Google Images, etc.)
# Minimum: 50 images per class
# Build a transfer learning classifier:
#   1. Load and resize images to 128×128
#   2. Feature extraction with MobileNetV2
#   3. Evaluate on held-out test set
#   4. Report: train acc, val acc, test acc, confusion matrix
```

---

## INSTRUCTOR TIPS

**"How do I know WHICH layers to unfreeze for fine-tuning?"**
> *"Print base_model.layers and look for block names.
> MobileNetV2 has blocks 1-16. Unfreeze blocks 14-16 first (last ~40 layers).
> ResNet50 has res5 as the last residual group — unfreeze that first.
> Start conservative. If accuracy keeps improving, unfreeze more."*

**"What if my images are very different from ImageNet?"**
> *"Medical images (X-ray, MRI), satellite imagery, microscopy.
> The first 1-2 blocks STILL transfer well — edge and texture detection
> is universal. But you'll want to fine-tune more aggressively.
> Consider unfreezing 50-70% of the base instead of 30-40%.
> And you'll need more data — at least 1000 images per class."*

**"Can I use transfer learning from a video model?"**
> *"Yes. There are pretrained models for video (C3D, I3D, SlowFast)
> and audio (VGGish for audio spectrograms). The same principle applies:
> the early layers learn universal features, use them.*
>
> *This is one of the most exciting frontiers in ML right now —
> transfer across modalities."*

**"Why is training=False important?"**
> *"BatchNorm has two modes: training (estimates mean/std from batch)
> and inference (uses stored running mean/std). If your fine-tuned
> batch is 32 images of cats, the batch mean/std of the frozen BN layers
> will look very different from the ImageNet distribution they were
> estimated on. Using training=False preserves the ImageNet statistics
> that the frozen weights expect to see."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening — hiring the expert analogy      10 min
├── Feature hierarchy diagram                20 min
├── Loading pretrained models in Keras       15 min
├── Two modes: extract vs fine-tune diagram  20 min
├── Live demo + visuals                      15 min
└── Close + homework                         10 min

SESSION 2  (90 min)
├── Opening bridge                           10 min
├── Feature extraction — full code           20 min
├── Fine-tuning — unfreeze top layers        20 min
├── Dataset size experiment                  20 min
├── MobileNetV2 vs ResNet50 comparison       10 min
└── Close + homework                         10 min
```

---
*MLForBeginners · Part 4: CNNs · Module 07*
