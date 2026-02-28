"""
🔄 CONVOLUTIONAL NEURAL NETWORKS — Algorithm 4: Transfer Learning
=================================================================

Learning Objectives:
  1. Understand what transfer learning is and why it works so well
  2. Use pretrained ImageNet weights (MobileNetV2, ResNet50) as feature extractors
  3. Implement feature extraction mode: freeze base, train top classifier only
  4. Implement fine-tuning mode: unfreeze top layers and train with low LR
  5. Know when to use transfer learning vs training from scratch
  6. Compare: scratch vs feature-extract vs fine-tune on a small dataset
  7. Understand domain similarity and data size considerations

YouTube Resources:
  ⭐ Andrew Ng — Transfer Learning (deeplearning.ai) https://www.youtube.com/watch?v=yofjFQddwHE
  ⭐ Sentdex — Transfer Learning Keras https://www.youtube.com/watch?v=19LLLaGLaptop
  📚 TensorFlow — Transfer learning guide https://www.tensorflow.org/tutorials/images/transfer_learning

Time Estimate: 70-85 minutes
Difficulty: Intermediate-Advanced
Prerequisites: classic_architectures.py, cnn_with_keras.py
Key Concepts: pretrained weights, ImageNet, frozen layers, fine-tuning, domain adaptation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs("../visuals/transfer_learning", exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🔄 ALGORITHM 4: TRANSFER LEARNING")
print("=" * 70)
print()
print("The big idea: Don't train from scratch if someone already did it better.")
print()
print("ImageNet: 1.2 million images, 1000 categories, 2 weeks on 8 GPUs")
print("Transfer learning: steal those weights, apply to YOUR problem.")
print()
print("Why it works:")
print("  Layer 1-3:  edges, textures, color blobs (universal — work for anything)")
print("  Layer 4-8:  shapes, parts (somewhat universal)")
print("  Layer 9-12: car wheels, cat ears (task-specific — replace these)")
print()
print("Result: with 200 images you can beat a CNN trained on 50,000 from scratch.")
print()


# ======================================================================
# SECTION 1: The Concept — What Gets Transferred?
# ======================================================================
print("=" * 70)
print("SECTION 1: WHAT GETS TRANSFERRED — FEATURE HIERARCHY")
print("=" * 70)
print()

print("  A CNN trained on ImageNet learns a feature hierarchy:")
print()
print("  Input Image")
print("      │")
print("      ▼  ─────────────────────────────────────────────────")
print("   Block 1  → Edges (horizontal, vertical, diagonal)")
print("              → Color contrasts, brightness gradients")
print("              [HIGHLY TRANSFERABLE — same for any image]")
print("      │")
print("      ▼  ─────────────────────────────────────────────────")
print("   Block 2  → Corners, curves, simple shapes")
print("              → Texture patterns (stripes, dots, grids)")
print("              [VERY TRANSFERABLE — universal image grammar]")
print("      │")
print("      ▼  ─────────────────────────────────────────────────")
print("   Block 3  → Object parts (eyes, wheels, windows)")
print("              → Complex textures (fur, glass, fabric)")
print("              [MODERATELY TRANSFERABLE — depends on domain]")
print("      │")
print("      ▼  ─────────────────────────────────────────────────")
print("   Block 4+ → High-level semantic features")
print("              → Class-specific representations")
print("              [LOW TRANSFER — replace for your task]")
print("      │")
print("      ▼  ─────────────────────────────────────────────────")
print("   Output   → 1000 ImageNet classes")
print("              [ALWAYS REPLACED — your task has different classes]")
print()

print("  When to use what:")
print()
print("  ┌─────────────────┬──────────────┬──────────────────────────────┐")
print("  │ Data Size       │ Domain       │ Strategy                     │")
print("  ├─────────────────┼──────────────┼──────────────────────────────┤")
print("  │ Very small      │ Similar      │ Feature extraction only      │")
print("  │ (<1k images)    │              │ (train linear classifier)    │")
print("  ├─────────────────┼──────────────┼──────────────────────────────┤")
print("  │ Medium          │ Similar      │ Feature extract → fine-tune  │")
print("  │ (1k-10k)        │              │ (unfreeze top layers)        │")
print("  ├─────────────────┼──────────────┼──────────────────────────────┤")
print("  │ Large           │ Similar      │ Fine-tune all layers         │")
print("  │ (10k-100k)      │              │ (lower LR throughout)        │")
print("  ├─────────────────┼──────────────┼──────────────────────────────┤")
print("  │ Any size        │ Very diff.   │ Train from scratch           │")
print("  │                 │ (e.g., MRI)  │ OR use domain-specific model │")
print("  └─────────────────┴──────────────┴──────────────────────────────┘")
print()


# ======================================================================
# SECTION 2: Transfer Learning From Scratch (Conceptual Numpy)
# ======================================================================
print("=" * 70)
print("SECTION 2: FROM SCRATCH — WEIGHT FREEZING MECHANICS")
print("=" * 70)
print()
print("How 'frozen layers' work in practice:")
print()
print("  During training, for each layer we decide:")
print("  • trainable=True  → gradients computed, weights updated")
print("  • trainable=False → gradients NOT computed, weights FIXED")
print()


class FrozenLayer:
    """Simple demonstration of a 'frozen' vs 'trainable' layer."""

    def __init__(self, weights, trainable=True):
        self.W = weights.copy()
        self.trainable = trainable
        self._grad = None

    def forward(self, x):
        self._input = x
        return x @ self.W

    def backward(self, grad_output):
        if self.trainable:
            # Compute gradient w.r.t. weights — will be used to UPDATE them
            self._grad = self._input.T @ grad_output
        else:
            # Frozen: don't compute weight gradient, don't update
            self._grad = None   # explicitly None
        # Always pass gradient backward (for layers below to receive)
        return grad_output @ self.W.T

    def update(self, lr=0.01):
        if self.trainable and self._grad is not None:
            self.W -= lr * self._grad
            return True  # updated
        return False  # frozen — no update


# Simulate a tiny 2-layer network
print("  Simulation: 2-layer network — Layer 1 frozen, Layer 2 trainable")
print()

np.random.seed(42)
layer1 = FrozenLayer(np.random.randn(4, 4) * 0.1, trainable=False)
layer2 = FrozenLayer(np.random.randn(4, 2) * 0.1, trainable=True)

W1_before = layer1.W.copy()
W2_before = layer2.W.copy()

# Forward
X = np.random.randn(8, 4)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

h = layer1.forward(X)
out = layer2.forward(h)

# Fake gradient (normally from loss)
grad = np.random.randn(*out.shape) * 0.1

# Backward
grad_h = layer2.backward(grad)
layer1.backward(grad_h)

# Update
updated1 = layer1.update(lr=0.01)
updated2 = layer2.update(lr=0.01)

W1_change = np.abs(layer1.W - W1_before).mean()
W2_change = np.abs(layer2.W - W2_before).mean()

print(f"  Layer 1 (frozen):    updated={updated1}, avg weight change={W1_change:.6f}")
print(f"  Layer 2 (trainable): updated={updated2}, avg weight change={W2_change:.6f}")
print()
print(f"  ✓ Layer 1 weights unchanged (frozen)")
print(f"  ✓ Layer 2 weights updated by training")
print()

print("  Key efficiency benefit of frozen layers:")
print("  • No gradient computation through frozen layers (saves memory + speed)")
print("  • Only need to store gradients for trainable parameters")
print("  • Pretrained features preserved EXACTLY as-is")
print()


# ======================================================================
# SECTION 3: MobileNetV2 — Feature Extraction
# ======================================================================
print("=" * 70)
print("SECTION 3: MOBILENETV2 AS FEATURE EXTRACTOR")
print("=" * 70)
print()
print("MobileNetV2: efficient CNN designed for mobile devices")
print("  • 3.4 million parameters (vs ResNet50's 25M)")
print("  • Trained on ImageNet: 1000 classes, 1.2M images")
print("  • Input: (224, 224, 3) RGB images")
print("  • Output without top: (7, 7, 1280) feature maps → flatten to 62,720")
print()
print("Feature extraction strategy:")
print("  1. Load MobileNetV2 WITH pretrained ImageNet weights")
print("  2. Remove the top classification layer (include_top=False)")
print("  3. Freeze ALL base model weights")
print("  4. Add YOUR classification head on top")
print("  5. Train ONLY the new head")
print()

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2, ResNet50
    TF_AVAILABLE = True
    tf.random.set_seed(42)
    print(f"  TensorFlow {tf.__version__} detected")
    print()
except ImportError:
    print("  TensorFlow not installed. Run: pip install tensorflow")
    print("  Architecture diagrams and comparisons will still be shown.")
    print()

# ---- Small flowers dataset (5 classes) ----
print("  Dataset: TensorFlow Flowers")
print("  5 classes: daisy, dandelion, roses, sunflowers, tulips")
print("  ~3,600 images (~720 per class) — small enough to demo TL advantage")
print()

if TF_AVAILABLE:
    # Download flowers dataset
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(
        "flower_photos",
        origin=dataset_url,
        untar=True,
        cache_dir=os.path.expanduser("~/.keras/datasets")
    )
    import pathlib
    data_dir = pathlib.Path(data_dir)

    IMG_SIZE  = (160, 160)
    BATCH     = 32
    AUTOTUNE  = tf.data.AUTOTUNE
    CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH,
    )

    print(f"  Train batches: {len(train_ds)}, Val batches: {len(val_ds)}")
    print(f"  Image shape per batch: {IMG_SIZE + (3,)}")
    print()

    # Performance: cache + prefetch
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Preprocessing: normalize to [-1, 1] (MobileNetV2 expects this)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Data augmentation for small dataset
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
    ], name="data_augmentation")

    # ---- Build Feature Extractor Model ----
    print("  Building Feature Extraction model:")
    print()

    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,        # Remove ImageNet classifier
        weights="imagenet"        # Use pretrained weights
    )
    base_model.trainable = False  # FREEZE ALL BASE LAYERS

    inputs     = keras.Input(shape=IMG_SIZE + (3,))
    x          = data_augmentation(inputs)
    x          = preprocess_input(x)
    x          = base_model(x, training=False)   # training=False keeps BN frozen
    x          = layers.GlobalAveragePooling2D()(x)
    x          = layers.Dropout(0.2)(x)
    outputs    = layers.Dense(5, activation="softmax")(x)

    feature_extract_model = keras.Model(inputs, outputs, name="MobileNetV2_FeatureExtract")

    frozen_params    = sum(~v.trainable for v in feature_extract_model.variables)
    trainable_params = sum(v.trainable for v in feature_extract_model.variables)

    print(f"  Total params:     {feature_extract_model.count_params():,}")
    print(f"  Trainable:        {sum(np.prod(v.shape) for v in feature_extract_model.trainable_variables):,}")
    print(f"  Non-trainable:    {sum(np.prod(v.shape) for v in feature_extract_model.non_trainable_variables):,}")
    print()
    print("  Only the Dense(5) head is trainable — 6,405 params vs 2.2M base")
    print()

    feature_extract_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("  Training Phase 1: Feature extraction (frozen base)...")
    history_fe = feature_extract_model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        verbose=1
    )
    fe_acc = max(history_fe.history["val_accuracy"])
    print(f"\n  Best val accuracy (feature extraction): {fe_acc:.4f} ({fe_acc*100:.1f}%)")
    print()

else:
    print("  [TensorFlow not installed — showing architecture only]")
    print()
    print("  Feature Extractor Architecture:")
    print("    Input (160, 160, 3)")
    print("    → DataAugmentation (RandomFlip, Rotate, Zoom)")
    print("    → preprocess_input (normalize to [-1, 1])")
    print("    → MobileNetV2 base (FROZEN: 2,257,984 params)")
    print("    → GlobalAveragePooling2D → (1280,)")
    print("    → Dropout(0.2)")
    print("    → Dense(5, softmax)   ← ONLY trainable layer")
    print()
    print("  Expected val accuracy: ~85-92% in 10 epochs")
    print()


# ======================================================================
# SECTION 4: Fine-Tuning — Unfreeze Top Layers
# ======================================================================
print("=" * 70)
print("SECTION 4: FINE-TUNING — UNFREEZE TOP LAYERS")
print("=" * 70)
print()
print("Fine-tuning strategy:")
print("  After feature extraction converges, unfreeze the LAST few blocks")
print("  of the base model and continue training with a MUCH lower LR.")
print()
print("  WHY lower LR? The pretrained weights are already good.")
print("  A high LR would 'catastrophically forget' the ImageNet knowledge.")
print("  We want to nudge them gently toward our specific domain.")
print()
print("  Rule of thumb: fine-tuning LR ≈ initial LR / 10")
print()
print("  MobileNetV2 structure (155 layers):")
print("  → Layers 0-100:   Early features (KEEP FROZEN)")
print("  → Layers 100-155: Late features  (UNFREEZE for fine-tuning)")
print()

if TF_AVAILABLE:
    # Build a fresh copy for fine-tuning demo
    base_model_ft = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model_ft.trainable = False   # Start frozen

    inputs_ft  = keras.Input(shape=IMG_SIZE + (3,))
    x_ft       = data_augmentation(inputs_ft)
    x_ft       = preprocess_input(x_ft)
    x_ft       = base_model_ft(x_ft, training=False)
    x_ft       = layers.GlobalAveragePooling2D()(x_ft)
    x_ft       = layers.Dropout(0.2)(x_ft)
    outputs_ft = layers.Dense(5, activation="softmax")(x_ft)
    fine_tune_model = keras.Model(inputs_ft, outputs_ft, name="MobileNetV2_FineTune")

    fine_tune_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Phase 1: Feature extraction (5 epochs to warm up the head)
    print("  Phase 1: Warming up classifier head (5 epochs)...")
    history_warmup = fine_tune_model.fit(
        train_ds, epochs=5, validation_data=val_ds, verbose=1
    )
    print()

    # Phase 2: Unfreeze top layers and fine-tune
    FINE_TUNE_AT = 100   # Unfreeze from layer 100 onwards
    base_model_ft.trainable = True

    # Re-freeze all layers before fine_tune_at
    for layer in base_model_ft.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    trainable_count = sum(
        np.prod(v.shape) for v in fine_tune_model.trainable_variables
    )
    print(f"  Phase 2: Fine-tuning layers {FINE_TUNE_AT}+ of MobileNetV2")
    print(f"  Now trainable: {trainable_count:,} parameters")
    print()

    # Recompile with lower LR
    fine_tune_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # 100x lower!
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    initial_epochs = 5
    fine_tune_epochs = 10
    total_epochs = initial_epochs + fine_tune_epochs

    print(f"  Fine-tuning for {fine_tune_epochs} more epochs (LR=1e-5)...")
    history_ft = fine_tune_model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=initial_epochs,
        validation_data=val_ds,
        verbose=1
    )
    ft_acc = max(history_ft.history["val_accuracy"])
    print(f"\n  Best val accuracy (fine-tuned): {ft_acc:.4f} ({ft_acc*100:.1f}%)")
    print()

    # Combine histories for plotting
    acc_all     = history_warmup.history["accuracy"]     + history_ft.history["accuracy"]
    val_acc_all = history_warmup.history["val_accuracy"] + history_ft.history["val_accuracy"]
    loss_all    = history_warmup.history["loss"]         + history_ft.history["loss"]
    val_loss_all= history_warmup.history["val_loss"]     + history_ft.history["val_loss"]

else:
    print("  [TensorFlow not installed — showing code structure only]")
    print()
    print("  Phase 1 code:")
    print("    base_model.trainable = False")
    print("    model.compile(optimizer=Adam(lr=0.001), ...)")
    print("    model.fit(train_ds, epochs=5, ...)   # warm up head")
    print()
    print("  Phase 2 code:")
    print("    base_model.trainable = True")
    print("    for layer in base_model.layers[:100]:")
    print("        layer.trainable = False")
    print("    model.compile(optimizer=Adam(lr=1e-5), ...)  # 100x lower LR!")
    print("    model.fit(train_ds, epochs=10, initial_epoch=5, ...)")
    print()
    print("  Expected accuracy improvement: ~3-5% over feature extraction")
    print()


# ======================================================================
# SECTION 5: ResNet50 Quick Demo
# ======================================================================
print("=" * 70)
print("SECTION 5: RESNET50 — A DEEPER PRETRAINED BACKBONE")
print("=" * 70)
print()
print("ResNet50 vs MobileNetV2:")
print()
print("  ┌─────────────────┬──────────────┬──────────────┬─────────────┐")
print("  │ Model           │ Params       │ Top-1 Acc    │ Mobile?     │")
print("  ├─────────────────┼──────────────┼──────────────┼─────────────┤")
print("  │ MobileNetV2     │ 3.4M         │ 71.3%        │ ✅ Yes       │")
print("  │ ResNet50        │ 25.6M        │ 74.9%        │ ❌ No        │")
print("  │ EfficientNetB0  │ 5.3M         │ 77.1%        │ ✅ Yes       │")
print("  │ EfficientNetB7  │ 66M          │ 84.4%        │ ❌ No        │")
print("  │ VGG16           │ 138M         │ 71.3%        │ ❌ No        │")
print("  └─────────────────┴──────────────┴──────────────┴─────────────┘")
print()
print("  All available via keras.applications:")
print("    • keras.applications.MobileNetV2()")
print("    • keras.applications.ResNet50()")
print("    • keras.applications.EfficientNetB0()")
print("    • keras.applications.VGG16()")
print("    • keras.applications.InceptionV3()")
print()
print("  Same API for all — just swap the class name!")
print()

if TF_AVAILABLE:
    print("  Building ResNet50 feature extractor (for comparison)...")
    base_resnet = ResNet50(
        input_shape=(160, 160, 3),
        include_top=False,
        weights="imagenet"
    )
    base_resnet.trainable = False

    inputs_rn  = keras.Input(shape=(160, 160, 3))
    x_rn       = tf.keras.applications.resnet50.preprocess_input(inputs_rn)
    x_rn       = base_resnet(x_rn, training=False)
    x_rn       = layers.GlobalAveragePooling2D()(x_rn)
    x_rn       = layers.Dropout(0.2)(x_rn)
    outputs_rn = layers.Dense(5, activation="softmax")(x_rn)
    resnet_model = keras.Model(inputs_rn, outputs_rn, name="ResNet50_FeatureExtract")

    resnet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(f"  ResNet50 base params: {sum(np.prod(v.shape) for v in resnet_model.non_trainable_variables):,}")
    print(f"  Trainable (head):     {sum(np.prod(v.shape) for v in resnet_model.trainable_variables):,}")
    print()
    print("  Training ResNet50 feature extractor (5 epochs)...")
    history_rn = resnet_model.fit(
        train_ds, epochs=5, validation_data=val_ds, verbose=1
    )
    rn_acc = max(history_rn.history["val_accuracy"])
    print(f"\n  ResNet50 val accuracy: {rn_acc:.4f} ({rn_acc*100:.1f}%)")
    print()

else:
    print("  ResNet50 Architecture (frozen feature extractor):")
    print()
    print("    Input (160, 160, 3)")
    print("    → resnet50.preprocess_input  (normalize: subtract ImageNet mean)")
    print("    → ResNet50 base (FROZEN: 23,587,712 params)")
    print("    → GlobalAveragePooling2D → (2048,)")
    print("    → Dropout(0.2)")
    print("    → Dense(5, softmax)          ← 10,245 trainable params")
    print()
    print("  ResNet50 typically gives ~87-93% on flowers dataset")
    print()


# ======================================================================
# SECTION 6: Training From Scratch (Baseline Comparison)
# ======================================================================
print("=" * 70)
print("SECTION 6: BASELINE — TRAINING CNN FROM SCRATCH")
print("=" * 70)
print()
print("For a fair comparison: train a custom CNN on the same flowers data")
print("with NO pretrained weights.")
print()

if TF_AVAILABLE:
    scratch_model = keras.Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dense(5, activation="softmax"),
    ], name="CNN_FromScratch")

    scratch_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    scratch_model.summary()
    print()

    print("  Training from scratch (15 epochs)...")
    history_scratch = scratch_model.fit(
        train_ds,
        epochs=15,
        validation_data=val_ds,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        )],
        verbose=1
    )
    scratch_acc = max(history_scratch.history["val_accuracy"])
    print(f"\n  From-scratch val accuracy: {scratch_acc:.4f} ({scratch_acc*100:.1f}%)")
    print()

else:
    print("  From-Scratch CNN:")
    print("    Conv(32) → Pool → Conv(64) → Pool → Conv(128) → Pool")
    print("    → Conv(128) → GAP → Dropout → Dense(256) → Dense(5)")
    print()
    print("  With only ~3,600 images:")
    print("  → Typically converges to 65-75% accuracy")
    print("  → Transfer learning usually gets 85-93%")
    print("  → 15-20% accuracy gap from only ~3,600 training images!")
    print()


# ======================================================================
# SECTION 7: Summary Comparison
# ======================================================================
print("=" * 70)
print("SECTION 7: COMPARISON SUMMARY")
print("=" * 70)
print()

if TF_AVAILABLE:
    results = {
        "From Scratch":       scratch_acc,
        "MobileNetV2 FeatEx": fe_acc,
        "MobileNetV2 FineTune": ft_acc,
        "ResNet50 FeatEx":    rn_acc,
    }

    print(f"  {'Method':<28} {'Val Accuracy':>14}")
    print(f"  {'─' * 28} {'─' * 14}")
    for method, acc in sorted(results.items(), key=lambda x: x[1]):
        bar = "█" * int(acc * 30)
        print(f"  {method:<28} {acc:.4f} ({acc*100:.1f}%)")
    print()

    best = max(results, key=results.get)
    print(f"  Best: {best} → {results[best]*100:.1f}%")
    print()

else:
    print("  Typical results on Flowers dataset (~3,600 images):")
    print()
    print(f"  {'Method':<28} {'Typical Val Accuracy'}")
    print(f"  {'─' * 28} {'─' * 20}")
    print(f"  {'From Scratch':<28} 65–75%")
    print(f"  {'MobileNetV2 FeatEx':<28} 85–90%")
    print(f"  {'MobileNetV2 Fine-Tune':<28} 88–93%")
    print(f"  {'ResNet50 FeatEx':<28} 87–93%")
    print()

print("  Key takeaways:")
print("  1. Transfer learning crushes from-scratch on small datasets")
print("  2. Feature extraction is faster; fine-tuning squeezes more accuracy")
print("  3. MobileNetV2 is excellent for resource-constrained environments")
print("  4. ResNet50 may edge ahead on complex tasks despite more params")
print("  5. EfficientNet often best accuracy/param tradeoff in practice")
print()


# ======================================================================
# SECTION 8: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 8: VISUALIZATIONS")
print("=" * 70)
print()


# --- PLOT 1: Transfer Learning Strategies Diagram ---
print("Generating: Transfer learning strategy diagram...")

fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle("Transfer Learning Strategies", fontsize=16, fontweight="bold")

strategies = [
    {
        "title": "Feature Extraction",
        "blocks": [
            ("ImageNet\nWeights", "frozen", 0.7),
            ("Block 1\n(edges)", "frozen", 0.6),
            ("Block 2\n(shapes)", "frozen", 0.5),
            ("Block 3\n(parts)", "frozen", 0.4),
            ("Block 4\n(semantic)", "frozen", 0.3),
            ("YOUR HEAD\nDense(N)", "trainable", 0.2),
        ],
        "note": "Train only classifier head\nFast, good for very small data",
    },
    {
        "title": "Fine-Tuning",
        "blocks": [
            ("ImageNet\nWeights", "frozen", 0.7),
            ("Block 1\n(edges)", "frozen", 0.6),
            ("Block 2\n(shapes)", "frozen", 0.5),
            ("Block 3\n(parts)", "trainable_low", 0.4),
            ("Block 4\n(semantic)", "trainable_low", 0.3),
            ("YOUR HEAD\nDense(N)", "trainable", 0.2),
        ],
        "note": "Unfreeze top layers\nLow LR (1e-5), best accuracy",
    },
    {
        "title": "From Scratch",
        "blocks": [
            ("Random\nInit", "trainable", 0.7),
            ("Block 1", "trainable", 0.6),
            ("Block 2", "trainable", 0.5),
            ("Block 3", "trainable", 0.4),
            ("Block 4", "trainable", 0.3),
            ("YOUR HEAD\nDense(N)", "trainable", 0.2),
        ],
        "note": "All random init\nNeeds large dataset",
    },
]

colors = {
    "frozen":        "#4A90D9",
    "trainable_low": "#F5A623",
    "trainable":     "#7ED321",
}
labels = {
    "frozen":        "Frozen",
    "trainable_low": "Trainable (low LR)",
    "trainable":     "Trainable",
}

for ax, strat in zip(axes, strategies):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(strat["title"], fontsize=13, fontweight="bold", pad=12)

    for label, mode, y in strat["blocks"]:
        color = colors[mode]
        rect = mpatches.FancyBboxPatch(
            (0.15, y - 0.04), 0.7, 0.08,
            boxstyle="round,pad=0.01",
            facecolor=color, edgecolor="white", linewidth=2, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(0.5, y, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")
        # Arrow connecting blocks (except last)
        if y > 0.2:
            ax.annotate("", xy=(0.5, y - 0.05), xytext=(0.5, y - 0.11),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    ax.text(0.5, 0.08, strat["note"], ha="center", va="center",
            fontsize=9, color="#333333", style="italic",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

# Legend
legend_elements = [
    mpatches.Patch(facecolor=colors["frozen"],        label="Frozen (no update)"),
    mpatches.Patch(facecolor=colors["trainable_low"], label="Trainable (low LR)"),
    mpatches.Patch(facecolor=colors["trainable"],     label="Trainable (normal LR)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("../visuals/transfer_learning/strategies_diagram.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: strategies_diagram.png")


# --- PLOT 2: Model Accuracy Comparison ---
print("Generating: Accuracy comparison chart...")

methods = [
    "From Scratch\n(custom CNN)",
    "MobileNetV2\nFeature Extract",
    "MobileNetV2\nFine-Tune",
    "ResNet50\nFeature Extract",
]

if TF_AVAILABLE:
    accuracies = [scratch_acc, fe_acc, ft_acc, rn_acc]
else:
    # Representative typical values
    accuracies = [0.70, 0.88, 0.91, 0.90]

bar_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Transfer Learning: Accuracy Comparison on Flowers Dataset",
             fontsize=14, fontweight="bold")

# Bar chart
bars = axes[0].bar(methods, accuracies, color=bar_colors, edgecolor="white",
                   linewidth=2, width=0.6)
axes[0].set_ylim(0.4, 1.05)
axes[0].set_ylabel("Validation Accuracy", fontsize=12)
axes[0].set_title("Method Comparison", fontsize=12)
axes[0].axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% threshold")
axes[0].tick_params(axis="x", labelsize=8)
axes[0].grid(axis="y", alpha=0.3)
axes[0].legend(fontsize=9)

for bar, acc in zip(bars, accuracies):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{acc:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

# Relative improvement
improvements = [(a - accuracies[0]) * 100 for a in accuracies]
imp_colors   = ["#95A5A6"] + ["#2ECC71" if i > 0 else "#E74C3C" for i in improvements[1:]]

bars2 = axes[1].bar(methods, improvements, color=imp_colors, edgecolor="white", linewidth=2, width=0.6)
axes[1].set_ylabel("Accuracy Gain over From-Scratch (%)", fontsize=11)
axes[1].set_title("Improvement over Baseline", fontsize=12)
axes[1].axhline(y=0, color="black", linewidth=1.5)
axes[1].tick_params(axis="x", labelsize=8)
axes[1].grid(axis="y", alpha=0.3)

for bar, imp in zip(bars2, improvements):
    y_pos = bar.get_height() + 0.3 if imp >= 0 else bar.get_height() - 1.5
    axes[1].text(
        bar.get_x() + bar.get_width() / 2, y_pos,
        f"{imp:+.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

plt.tight_layout()
plt.savefig("../visuals/transfer_learning/accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: accuracy_comparison.png")


# --- PLOT 3: Fine-Tuning History ---
print("Generating: Fine-tuning training history...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fine-Tuning Training Progress (MobileNetV2 on Flowers)",
             fontsize=14, fontweight="bold")

if TF_AVAILABLE:
    epochs_fe = list(range(1, len(acc_all) + 1))

    axes[0].plot(epochs_fe, acc_all,     color="steelblue",  linewidth=2, label="Train Acc")
    axes[0].plot(epochs_fe, val_acc_all, color="darkorange", linewidth=2, linestyle="--", label="Val Acc")
    axes[0].axvline(x=initial_epochs, color="red", linestyle=":", linewidth=2,
                    label=f"Fine-tuning starts (epoch {initial_epochs})")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy over Epochs")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(epochs_fe, loss_all,     color="steelblue",  linewidth=2, label="Train Loss")
    axes[1].plot(epochs_fe, val_loss_all, color="darkorange", linewidth=2, linestyle="--", label="Val Loss")
    axes[1].axvline(x=initial_epochs, color="red", linestyle=":", linewidth=2,
                    label=f"Fine-tuning starts (epoch {initial_epochs})")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss over Epochs")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

else:
    # Synthetic illustration
    epochs_show   = list(range(1, 16))
    phase1_epochs = 5

    # Simulated curves
    fe_acc_sim  = [0.60, 0.74, 0.80, 0.83, 0.85] + [0.87, 0.88, 0.89, 0.90, 0.90, 0.91, 0.91, 0.92, 0.92, 0.93]
    fe_vacc_sim = [0.58, 0.73, 0.79, 0.82, 0.85] + [0.86, 0.87, 0.88, 0.88, 0.89, 0.89, 0.90, 0.90, 0.91, 0.91]

    axes[0].plot(epochs_show, fe_acc_sim,  color="steelblue",  linewidth=2, label="Train Acc")
    axes[0].plot(epochs_show, fe_vacc_sim, color="darkorange", linewidth=2, linestyle="--", label="Val Acc")
    axes[0].axvline(x=phase1_epochs, color="red", linestyle=":", linewidth=2,
                    label="Fine-tuning starts (epoch 5)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Simulated: Accuracy over Epochs")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    fe_loss_sim  = [1.4, 0.9, 0.7, 0.55, 0.45] + [0.42, 0.38, 0.35, 0.33, 0.31, 0.30, 0.29, 0.28, 0.27, 0.27]
    fe_vloss_sim = [1.5, 0.95, 0.75, 0.58, 0.48] + [0.45, 0.42, 0.39, 0.38, 0.36, 0.35, 0.34, 0.33, 0.32, 0.32]

    axes[1].plot(epochs_show, fe_loss_sim,  color="steelblue",  linewidth=2, label="Train Loss")
    axes[1].plot(epochs_show, fe_vloss_sim, color="darkorange", linewidth=2, linestyle="--", label="Val Loss")
    axes[1].axvline(x=phase1_epochs, color="red", linestyle=":", linewidth=2,
                    label="Fine-tuning starts (epoch 5)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Simulated: Loss over Epochs")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    # Annotation
    for ax in axes:
        ax.annotate("Feature\nExtraction", xy=(2.5, 0.05 if ax == axes[0] else 1.35),
                    ha="center", fontsize=9, color="steelblue",
                    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5))
        ax.annotate("Fine-\nTuning", xy=(10, 0.05 if ax == axes[0] else 1.35),
                    ha="center", fontsize=9, color="darkorange",
                    bbox=dict(boxstyle="round", facecolor="moccasin", alpha=0.5))

plt.tight_layout()
plt.savefig("../visuals/transfer_learning/finetuning_history.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: finetuning_history.png")


# ======================================================================
# SECTION 9: When NOT to Use Transfer Learning
# ======================================================================
print()
print("=" * 70)
print("SECTION 9: WHEN NOT TO USE TRANSFER LEARNING")
print("=" * 70)
print()
print("Transfer learning is NOT always the answer:")
print()
print("  ❌ Medical imaging (X-rays, MRI, CT):")
print("     → Very different from natural photos (ImageNet domain)")
print("     → Early layers: edges are similar, but what matters is different")
print("     → Solution: use RadImageNet or domain-specific pretrained models")
print()
print("  ❌ Satellite / aerial imagery:")
print("     → Top-down view, very different statistics from ground-level photos")
print("     → Solution: look for pretrained remote sensing models")
print()
print("  ❌ Microscopy / scientific imaging:")
print("     → Completely different visual domain")
print("     → May need to train from scratch or use domain-specific model")
print()
print("  ❌ You have 10M+ labeled images:")
print("     → At that scale, training from scratch often beats TL")
print("     → Or TL just provides a faster convergence start point")
print()
print("  ✅ Transfer learning shines when:")
print("     → You have < 10k training images")
print("     → Your images look like natural photos")
print("     → You need quick prototyping (hours vs weeks)")
print("     → You have limited compute budget")
print()


# ======================================================================
# SECTION 10: Practical Tips
# ======================================================================
print("=" * 70)
print("SECTION 10: PRACTICAL TIPS FOR TRANSFER LEARNING")
print("=" * 70)
print()
print("  Tip 1: Always use include_top=False")
print("    → The top layer is ImageNet's 1000-class head — useless for you")
print("    → Add your own Dense(n_classes, softmax)")
print()
print("  Tip 2: Match preprocessing to the backbone")
print("    → MobileNetV2: expects [-1, 1] via preprocess_input()")
print("    → ResNet50: expects subtracted ImageNet mean via preprocess_input()")
print("    → Getting this wrong = garbage accuracy")
print()
print("  Tip 3: Use training=False for BatchNorm in base during feature extraction")
print("    → base_model(x, training=False)")
print("    → Keeps BatchNorm stats from ImageNet frozen (don't recompute on your data)")
print()
print("  Tip 4: Fine-tune from top down")
print("    → Unfreeze later layers first, keep early layers frozen")
print("    → Early layers (edges/textures) are already universal")
print()
print("  Tip 5: Use a much lower learning rate for fine-tuning")
print("    → 10x–100x lower than initial training LR")
print("    → Prevents catastrophic forgetting of pretrained knowledge")
print()
print("  Tip 6: Data augmentation is crucial for small datasets")
print("    → RandomFlip, RandomRotation, RandomZoom, RandomBrightness")
print("    → Keras layers.RandomFlip() etc. run on-GPU automatically")
print()
print("  Tip 7: GlobalAveragePooling2D > Flatten for TL")
print("    → Reduces overfitting on small datasets")
print("    → More robust to input size variations")
print()
print("  Tip 8: Compare multiple backbones cheaply")
print("    → Run feature extraction (frozen base) for 5 epochs each")
print("    → Pick the best, THEN fine-tune")
print("    → EfficientNetB0/B2 often wins on accuracy/speed tradeoff")
print()


print()
print("=" * 70)
print("ALGORITHM 4: TRANSFER LEARNING COMPLETE!")
print("=" * 70)
print()
print("What you learned:")
print("  ✓ CNNs learn universal features (edges → shapes → semantics)")
print("  ✓ ImageNet pretrained models are universal visual feature extractors")
print("  ✓ Feature extraction: freeze base, train small head (fastest)")
print("  ✓ Fine-tuning: unfreeze top layers, use low LR (best accuracy)")
print("  ✓ Transfer learning beats from-scratch by 15-20% on small datasets")
print("  ✓ MobileNetV2 for speed, ResNet50 for accuracy, EfficientNet for balance")
print()
print("3 Visualizations saved to: ../visuals/transfer_learning/")
print("  1. strategies_diagram.png     — Feature extract vs fine-tune vs scratch")
print("  2. accuracy_comparison.png    — Method comparison bar chart")
print("  3. finetuning_history.png     — Training curves across both phases")
print()
print("Next: Project 1 → CIFAR-10 Classifier")
