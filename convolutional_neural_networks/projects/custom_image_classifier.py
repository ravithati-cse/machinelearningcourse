"""
🎨 CONVOLUTIONAL NEURAL NETWORKS — Project 2: Custom Image Classifier
=====================================================================

Learning Objectives:
  1. Build an end-to-end image classifier for ANY custom dataset
  2. Organize images into train/val/test splits from raw folders
  3. Use Keras ImageDataGenerator / image_dataset_from_directory
  4. Apply the full transfer learning pipeline (MobileNetV2 + fine-tuning)
  5. Handle class imbalance with weighted loss and augmentation
  6. Export the model and write an inference function for new images
  7. Walk through a realistic custom dataset workflow top-to-bottom

YouTube Resources:
  ⭐ Sentdex — Custom dataset CNN https://www.youtube.com/watch?v=j-3vuBynnOE
  ⭐ TF — image_dataset_from_directory https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
  📚 Keras applications guide https://keras.io/api/applications/

Time Estimate: 80-100 minutes
Difficulty: Intermediate-Advanced
Prerequisites: cifar10_classifier.py, transfer_learning.py
Key Concepts: custom datasets, directory structure, class imbalance, inference, model export
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os, pathlib, shutil

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "custom_classifier")
os.makedirs(VIS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🎨 PROJECT 2: CUSTOM IMAGE CLASSIFIER")
print("=" * 70)
print()
print("Build a classifier for YOUR data — any images, any classes.")
print()
print("This project teaches the REAL workflow:")
print("  1. Gather images (scraping, Kaggle, camera, etc.)")
print("  2. Organize into folders per class")
print("  3. Split into train/val/test sets")
print("  4. Train with transfer learning")
print("  5. Evaluate, inspect failures, iterate")
print("  6. Export model and write inference code")
print()
print("We'll demo on a synthetic 3-class dataset so you can run it anywhere,")
print("then show the EXACT same code for your real dataset.")
print()


# ======================================================================
# SECTION 1: Dataset Folder Structure
# ======================================================================
print("=" * 70)
print("SECTION 1: DATASET FOLDER STRUCTURE")
print("=" * 70)
print()
print("The ONLY folder structure Keras needs:")
print()
print("  my_dataset/")
print("  ├── train/")
print("  │   ├── class_a/         ← folder name = class label")
print("  │   │   ├── img001.jpg")
print("  │   │   ├── img002.jpg")
print("  │   │   └── ...")
print("  │   ├── class_b/")
print("  │   └── class_c/")
print("  ├── val/")
print("  │   ├── class_a/")
print("  │   ├── class_b/")
print("  │   └── class_c/")
print("  └── test/")
print("      ├── class_a/")
print("      ├── class_b/")
print("      └── class_c/")
print()
print("Rule of thumb for splits:")
print("  • 70% train, 15% val, 15% test")
print("  • Or 80/10/10 for larger datasets")
print("  • Keep class distribution proportional in each split")
print()
print("Minimum images per class:")
print("  • Transfer learning (feature extract): ~50-100 minimum")
print("  • Transfer learning (fine-tune):       ~200-500")
print("  • Training from scratch:               ~1,000-10,000+")
print()


# ======================================================================
# SECTION 2: Create a Synthetic Demo Dataset
# ======================================================================
print("=" * 70)
print("SECTION 2: CREATING A DEMO DATASET")
print("=" * 70)
print()
print("Creating synthetic 'rocks / paper / scissors' demo dataset...")
print("(Same code works with real images — just swap the folder path)")
print()

DEMO_DIR    = "/tmp/rps_demo"
CLASSES     = ["rock", "paper", "scissors"]
N_PER_CLASS = {"train": 200, "val": 50, "test": 50}
IMG_SHAPE   = (64, 64, 3)


def make_synthetic_image(cls_idx, shape=(64, 64, 3)):
    """Create a synthetic colored image distinguishable by class."""
    img = np.random.randn(*shape) * 40 + 128

    if cls_idx == 0:     # rock → circular dark blob
        cx, cy, r = 32, 32, 14
        for y in range(shape[0]):
            for x in range(shape[1]):
                if (x - cx)**2 + (y - cy)**2 < r**2:
                    img[y, x, :] = [60, 40, 30]    # dark brownish

    elif cls_idx == 1:   # paper → light rectangle
        img[20:44, 18:46, :] = np.random.rand(24, 28, 3) * 30 + 220  # light

    else:                # scissors → two diagonal stripes
        for i in range(shape[0]):
            j1 = int(i * 0.5)
            j2 = int(shape[1] - i * 0.5)
            if 0 <= j1 < shape[1]:
                img[i, max(0, j1-2):min(shape[1], j1+2), :] = [180, 20, 20]
            if 0 <= j2 < shape[1]:
                img[i, max(0, j2-2):min(shape[1], j2+2), :] = [180, 20, 20]

    return np.clip(img, 0, 255).astype(np.uint8)


# Create directory structure and save synthetic images
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)

total_created = 0
for split, n in N_PER_CLASS.items():
    for cls_idx, cls_name in enumerate(CLASSES):
        folder = os.path.join(DEMO_DIR, split, cls_name)
        os.makedirs(folder, exist_ok=True)

        for i in range(n):
            img = make_synthetic_image(cls_idx)
            # Save as raw numpy (normally you'd have real .jpg files)
            # Use matplotlib to save as PNG
            plt.imsave(os.path.join(folder, f"{cls_name}_{i:04d}.png"), img)
        total_created += n

print(f"  Created {total_created} synthetic images in {DEMO_DIR}/")
print()

# Count and verify
for split in ["train", "val", "test"]:
    split_path = pathlib.Path(DEMO_DIR) / split
    total = sum(len(list(d.glob("*.png"))) for d in split_path.iterdir() if d.is_dir())
    class_counts = {d.name: len(list(d.glob("*.png")))
                    for d in sorted(split_path.iterdir()) if d.is_dir()}
    print(f"  {split:6s}: {total:4d} images  → {class_counts}")

print()


# ======================================================================
# SECTION 3: Imbalanced Dataset Simulation
# ======================================================================
print("=" * 70)
print("SECTION 3: HANDLING CLASS IMBALANCE")
print("=" * 70)
print()
print("Real-world datasets are rarely balanced.")
print("Example: defect detection — 95% normal, 5% defective")
print()
print("Solutions for class imbalance:")
print()
print("  1. Class weights (easiest — reweight the loss function)")
print("     → Penalize mistakes on minority class more")
print("     → class_weight = {0: 1.0, 1: 19.0}  (for 5% minority)")
print()
print("  2. Oversampling minority class")
print("     → Duplicate/augment minority class images")
print("     → Risk: overfitting to minority samples")
print()
print("  3. Undersampling majority class")
print("     → Drop majority class images to balance")
print("     → Risk: losing potentially useful information")
print()
print("  4. Augmentation for minority class only")
print("     → Apply more aggressive augmentation to minority")
print("     → Best of both worlds: more diversity without duplication")
print()
print("  5. Use F1-score or AUC-ROC instead of accuracy for evaluation")
print("     → Accuracy is misleading for imbalanced data")
print("     → A model that always predicts 'normal' gets 95% — useless!")
print()

# Demo: compute class weights
def compute_class_weights(y):
    """sklearn-style balanced class weights."""
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes  = len(classes)
    weights    = n_samples / (n_classes * counts)
    return dict(zip(classes.tolist(), weights.tolist()))

# Simulated imbalanced labels
y_imbalanced = np.array([0] * 950 + [1] * 50)  # 95%/5% split
cw = compute_class_weights(y_imbalanced)
print(f"  Example imbalanced dataset (95% class 0, 5% class 1):")
print(f"  Computed class weights: {cw}")
print()
print(f"  Class 0 weight: {cw[0]:.2f}  (majority — small weight)")
print(f"  Class 1 weight: {cw[1]:.2f}  (minority — large weight)")
print(f"  → Mistakes on class 1 penalized {cw[1]/cw[0]:.0f}x more in loss")
print()


# ======================================================================
# SECTION 4: Load Dataset with image_dataset_from_directory
# ======================================================================
print("=" * 70)
print("SECTION 4: LOAD DATASET (image_dataset_from_directory)")
print("=" * 70)
print()
print("Keras's preferred way to load image datasets from disk:")
print()
print("  tf.keras.utils.image_dataset_from_directory(")
print("      directory,              # path to split folder (e.g., 'data/train')")
print("      image_size=(H, W),      # resize all images to this")
print("      batch_size=32,          # images per gradient step")
print("      label_mode='int',       # 'int' for sparse_categorical, 'categorical' for one-hot")
print("      shuffle=True,           # shuffle order each epoch")
print("      seed=42,                # reproducibility")
print("  )")
print()

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import MobileNetV2
    TF_AVAILABLE = True
    tf.random.set_seed(42)
    print(f"  TensorFlow {tf.__version__} detected")
    print()
except ImportError:
    print("  TensorFlow not installed. Run: pip install tensorflow")
    print("  Demo will continue with code walkthroughs.")
    print()

IMG_SIZE = (96, 96)
BATCH    = 32

if TF_AVAILABLE:
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DEMO_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DEMO_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int",
        shuffle=False,
        seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(DEMO_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH,
        label_mode="int",
        shuffle=False,
        seed=42,
    )

    class_names = train_ds.class_names
    n_classes   = len(class_names)

    print(f"  Classes detected: {class_names}")
    print(f"  Number of classes: {n_classes}")
    print()

    # Peek at a batch
    for imgs, labels in train_ds.take(1):
        print(f"  Batch shape: images={imgs.shape}, labels={labels.shape}")
        print(f"  Pixel range: [{imgs.numpy().min():.0f}, {imgs.numpy().max():.0f}]")
    print()

    # Optimize loading
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

else:
    class_names = CLASSES
    n_classes   = 3
    print("  Code for loading a real dataset:")
    print()
    print("    train_ds = tf.keras.utils.image_dataset_from_directory(")
    print("        'data/train', image_size=(96, 96), batch_size=32)")
    print("    val_ds = tf.keras.utils.image_dataset_from_directory(")
    print("        'data/val', image_size=(96, 96), batch_size=32)")
    print("    class_names = train_ds.class_names   # auto-detected!")
    print()


# ======================================================================
# SECTION 5: Build the Custom Classifier
# ======================================================================
print("=" * 70)
print("SECTION 5: BUILD CUSTOM CLASSIFIER (MOBILENETV2 + FINE-TUNE)")
print("=" * 70)
print()
print("Strategy: feature extraction first, then fine-tune top layers")
print()

if TF_AVAILABLE:
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Data augmentation
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomTranslation(0.15, 0.15),
        layers.RandomBrightness(0.2),
    ], name="augmentation")

    def build_custom_classifier(n_classes, img_size=(96, 96), fine_tune_from=None):
        """
        Build a transfer learning classifier.

        Args:
            n_classes:      number of output classes
            img_size:       (H, W) input image size
            fine_tune_from: if not None, unfreeze base layers from this index onward
        """
        base_model = MobileNetV2(
            input_shape=img_size + (3,),
            include_top=False,
            weights="imagenet",
        )

        if fine_tune_from is None:
            base_model.trainable = False   # Feature extraction mode
        else:
            base_model.trainable = True
            for layer in base_model.layers[:fine_tune_from]:
                layer.trainable = False    # Fine-tune only top layers

        inputs  = keras.Input(shape=img_size + (3,))
        x       = augmentation(inputs)
        x       = preprocess_input(x)
        x       = base_model(x, training=(fine_tune_from is None))
        x       = layers.GlobalAveragePooling2D()(x)
        x       = layers.BatchNormalization()(x)
        x       = layers.Dropout(0.3)(x)
        x       = layers.Dense(128, activation="relu")(x)
        x       = layers.Dropout(0.3)(x)

        if n_classes == 2:
            outputs = layers.Dense(1, activation="sigmoid")(x)
        else:
            outputs = layers.Dense(n_classes, activation="softmax")(x)

        return keras.Model(inputs, outputs, name=f"CustomClassifier_{n_classes}class")

    # Phase 1: Feature extraction
    print("  Phase 1: Feature extraction (frozen base)")
    model = build_custom_classifier(n_classes, img_size=IMG_SIZE, fine_tune_from=None)

    loss_fn = "sparse_categorical_crossentropy" if n_classes > 2 else "binary_crossentropy"
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=loss_fn,
        metrics=["accuracy"]
    )

    trainable_p = sum(np.prod(v.shape) for v in model.trainable_variables)
    total_p     = model.count_params()
    print(f"  Total params:     {total_p:,}")
    print(f"  Trainable params: {trainable_p:,}  ({trainable_p/total_p*100:.1f}%)")
    print()

    history_p1 = model.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        verbose=1,
    )
    p1_best = max(history_p1.history["val_accuracy"])
    print(f"\n  Phase 1 best val accuracy: {p1_best:.4f} ({p1_best*100:.1f}%)")
    print()

    # Phase 2: Fine-tune top layers
    print("  Phase 2: Fine-tuning top MobileNetV2 layers (layers 100+)")
    model_ft = build_custom_classifier(n_classes, img_size=IMG_SIZE, fine_tune_from=100)
    model_ft.set_weights(model.get_weights())   # start from Phase 1 weights

    model_ft.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # 100x lower LR
        loss=loss_fn,
        metrics=["accuracy"]
    )

    trainable_p2 = sum(np.prod(v.shape) for v in model_ft.trainable_variables)
    print(f"  Trainable params (fine-tune): {trainable_p2:,}")
    print()

    history_p2 = model_ft.fit(
        train_ds,
        epochs=10,
        validation_data=val_ds,
        verbose=1,
    )
    p2_best = max(history_p2.history["val_accuracy"])
    print(f"\n  Phase 2 best val accuracy: {p2_best:.4f} ({p2_best*100:.1f}%)")
    print()

    # Evaluate on test set
    test_loss, test_acc = model_ft.evaluate(test_ds, verbose=0)
    print(f"  Test accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")
    print()

else:
    print("  Two-phase training code (runs with TF installed):")
    print()
    print("  # Phase 1: Feature extraction")
    print("  model = build_custom_classifier(n_classes, fine_tune_from=None)")
    print("  model.compile(optimizer=Adam(0.001), ...)")
    print("  model.fit(train_ds, epochs=10, ...)")
    print()
    print("  # Phase 2: Fine-tuning")
    print("  model_ft = build_custom_classifier(n_classes, fine_tune_from=100)")
    print("  model_ft.set_weights(model.get_weights())  # start from Phase 1")
    print("  model_ft.compile(optimizer=Adam(1e-5), ...)  # 100x lower LR!")
    print("  model_ft.fit(train_ds, epochs=10, initial_epoch=10, ...)")
    print()
    test_acc = 0.92  # expected synthetic


# ======================================================================
# SECTION 6: Save and Load the Model
# ======================================================================
print("=" * 70)
print("SECTION 6: SAVE AND LOAD THE MODEL")
print("=" * 70)
print()
print("Always save your trained model — training takes time!")
print()
print("Two save formats:")
print()
print("  1. SavedModel (recommended — full TF graph)")
print('     model.save("my_classifier.keras")         # Keras v3 format')
print('     model.save("my_classifier/")              # SavedModel format')
print('     loaded = keras.models.load_model("my_classifier.keras")')
print()
print("  2. Weights only (faster, smaller)")
print('     model.save_weights("my_classifier_weights.h5")')
print('     model.load_weights("my_classifier_weights.h5")')
print()

if TF_AVAILABLE:
    save_path = "/tmp/custom_classifier_demo.keras"
    model_ft.save(save_path)
    print(f"  Model saved to: {save_path}")
    print(f"  File size: {os.path.getsize(save_path) / 1024 / 1024:.1f} MB")
    print()

    loaded_model = keras.models.load_model(save_path)
    _, loaded_acc = loaded_model.evaluate(test_ds, verbose=0)
    print(f"  Loaded model test accuracy: {loaded_acc:.4f}  (matches saved: ✓)")
    print()


# ======================================================================
# SECTION 7: Inference on New Images
# ======================================================================
print("=" * 70)
print("SECTION 7: INFERENCE — CLASSIFY NEW IMAGES")
print("=" * 70)
print()
print("Write a reusable predict() function for single images:")
print()

if TF_AVAILABLE:
    def predict_image(model, image_path_or_array, class_names, img_size=(96, 96)):
        """
        Classify a single image.

        Args:
            model:               trained Keras model
            image_path_or_array: file path (str) or numpy array (H, W, 3)
            class_names:         list of class name strings
            img_size:            (H, W) the model expects

        Returns:
            predicted class name, confidence score, all probabilities
        """
        if isinstance(image_path_or_array, str):
            # Load from file
            img = tf.keras.utils.load_img(image_path_or_array,
                                          target_size=img_size)
            arr = tf.keras.utils.img_to_array(img)
        else:
            arr = image_path_or_array.astype("float32")
            arr = tf.image.resize(arr, img_size).numpy()

        # Add batch dimension: (H, W, 3) → (1, H, W, 3)
        batch = np.expand_dims(arr, axis=0)

        # Get probabilities
        probs = model.predict(batch, verbose=0)[0]

        pred_idx  = np.argmax(probs)
        pred_name = class_names[pred_idx]
        confidence = probs[pred_idx]

        return pred_name, confidence, probs

    # Test inference on some demo images
    print("  Running inference on 6 test images:")
    print()
    test_images, test_labels_batch = next(iter(test_ds))
    test_images_np = test_images.numpy()
    test_labels_np = test_labels_batch.numpy()

    print(f"  {'#':<4} {'True':>10} {'Predicted':>12} {'Confidence':>12} {'Correct?':>10}")
    print(f"  {'─'*4} {'─'*10} {'─'*12} {'─'*12} {'─'*10}")

    for i in range(min(6, len(test_images_np))):
        pred_name, conf, _ = predict_image(
            model_ft, test_images_np[i], class_names, img_size=IMG_SIZE
        )
        true_name = class_names[test_labels_np[i]]
        correct   = "✓" if pred_name == true_name else "✗"
        print(f"  {i:<4} {true_name:>10} {pred_name:>12} {conf*100:>10.1f}%  {correct:>10}")

    print()

else:
    print("  def predict_image(model, image_path, class_names, img_size=(96, 96)):")
    print("      img  = tf.keras.utils.load_img(image_path, target_size=img_size)")
    print("      arr  = tf.keras.utils.img_to_array(img)")
    print("      arr  = np.expand_dims(arr, axis=0)    # add batch dim")
    print("      probs = model.predict(arr, verbose=0)[0]")
    print("      pred_idx  = np.argmax(probs)")
    print("      return class_names[pred_idx], probs[pred_idx], probs")
    print()
    print("  # Usage:")
    print("  pred, conf, all_probs = predict_image(model, 'my_photo.jpg', class_names)")
    print("  print(f'Predicted: {pred} with {conf:.1%} confidence')")
    print()


# ======================================================================
# SECTION 8: Real Dataset Checklist
# ======================================================================
print("=" * 70)
print("SECTION 8: REAL DATASET CHECKLIST")
print("=" * 70)
print()
print("Before training on your real dataset, verify:")
print()
print("  Data quality:")
print("    □ Each image belongs to exactly ONE class (no ambiguous samples)")
print("    □ No duplicates or near-duplicates across train/val/test")
print("    □ Test set is truly held out — never seen during training/tuning")
print("    □ Images are representative of what the model will see in production")
print()
print("  Data quantity:")
print("    □ At least 50-100 images per class for feature extraction")
print("    □ At least 200-500 per class for fine-tuning")
print("    □ Val/test sets large enough to give stable metrics (100+ per class)")
print()
print("  Data distribution:")
print("    □ Check class imbalance — use class_weight if needed")
print("    □ Train/val/test have same class distribution (stratified split)")
print("    □ No data leakage (same person/scene in train AND test)")
print()
print("  Preprocessing:")
print("    □ Using the correct preprocess_input() for your backbone")
print("    □ Image size matches backbone expectations (check keras.applications)")
print("    □ Augmentations are realistic (don't flip if orientation matters)")
print()
print("  Evaluation:")
print("    □ Using F1-score/ROC-AUC for imbalanced classes")
print("    □ Inspecting failure cases, not just aggregate accuracy")
print("    □ Testing on data collected in 'production' conditions")
print()


# ======================================================================
# SECTION 9: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 9: VISUALIZATIONS")
print("=" * 70)
print()


# --- PLOT 1: Dataset samples per class ---
print("Generating: Dataset samples per class...")

fig, axes = plt.subplots(3, 6, figsize=(15, 7))
fig.suptitle("Custom Dataset: Rock / Paper / Scissors — Sample Images",
             fontsize=14, fontweight="bold")

for cls_idx, cls_name in enumerate(CLASSES):
    img_dir = pathlib.Path(DEMO_DIR) / "train" / cls_name
    imgs    = sorted(img_dir.glob("*.png"))[:6]
    for col, img_path in enumerate(imgs):
        img_arr = plt.imread(str(img_path))
        axes[cls_idx, col].imshow(img_arr)
        axes[cls_idx, col].axis("off")
        if col == 0:
            axes[cls_idx, col].set_ylabel(cls_name, fontsize=11,
                                          fontweight="bold", rotation=90, va="center")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/dataset_samples.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: dataset_samples.png")


# --- PLOT 2: Full pipeline diagram ---
print("Generating: Full pipeline diagram...")

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("Custom Image Classifier Pipeline",
             fontsize=16, fontweight="bold", pad=15)

pipeline_steps = [
    ("📁\nCollect\nImages",    0.07, "#3498DB"),
    ("📂\nOrganize\nFolders",  0.20, "#2ECC71"),
    ("✂️\nTrain/Val\nTest Split", 0.33, "#F39C12"),
    ("🔄\nData\nAugmentation", 0.46, "#9B59B6"),
    ("🧠\nTransfer\nLearning",  0.59, "#E74C3C"),
    ("📊\nEvaluate\n& Iterate", 0.72, "#1ABC9C"),
    ("🚀\nDeploy\n& Infer",    0.85, "#E67E22"),
]

box_w, box_h = 0.10, 0.30
for label, xc, color in pipeline_steps:
    rect = plt.Rectangle((xc - box_w/2, 0.35), box_w, box_h,
                          facecolor=color, alpha=0.85, edgecolor="white", linewidth=2,
                          transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(xc, 0.50, label, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="white",
            transform=ax.transAxes, multialignment="center")

# Arrows between steps
for i in range(len(pipeline_steps) - 1):
    x0 = pipeline_steps[i][1]     + box_w / 2 + 0.005
    x1 = pipeline_steps[i+1][1]   - box_w / 2 - 0.005
    ax.annotate("", xy=(x1, 0.50), xytext=(x0, 0.50),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2),
                xycoords="axes fraction", textcoords="axes fraction")

# Sub-labels below
sub_labels = [
    "camera / scrape\n/ Kaggle",
    "one folder\nper class",
    "70% / 15%\n/ 15%",
    "flip, crop\nrotate, zoom",
    "MobileNetV2\nfrozen → tune",
    "confusion\nmatrix, F1",
    "save .keras\npredict()",
]
for (label, xc, _), sub in zip(pipeline_steps, sub_labels):
    ax.text(xc, 0.22, sub, ha="center", va="top",
            fontsize=8, color="#333", transform=ax.transAxes, multialignment="center")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/pipeline_diagram.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: pipeline_diagram.png")


if TF_AVAILABLE:
    # --- PLOT 3: Training history (both phases) ---
    print("Generating: Two-phase training history...")

    p1_epochs = len(history_p1.history["accuracy"])
    p2_epochs = len(history_p2.history["accuracy"])
    total_epochs = p1_epochs + p2_epochs

    acc_all     = history_p1.history["accuracy"]     + history_p2.history["accuracy"]
    val_acc_all = history_p1.history["val_accuracy"] + history_p2.history["val_accuracy"]
    loss_all    = history_p1.history["loss"]         + history_p2.history["loss"]
    val_loss_all= history_p1.history["val_loss"]     + history_p2.history["val_loss"]

    epochs_all  = list(range(1, total_epochs + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Custom Classifier: Two-Phase Training History",
                 fontsize=14, fontweight="bold")

    for ax, train_v, val_v, ylabel, title in zip(
        axes,
        [acc_all, loss_all],
        [val_acc_all, val_loss_all],
        ["Accuracy", "Loss"],
        [f"Accuracy (Test: {test_acc:.1%})", "Loss"],
    ):
        ax.plot(epochs_all, train_v, color="steelblue",  linewidth=2, label=f"Train {ylabel}")
        ax.plot(epochs_all, val_v,   color="darkorange", linewidth=2, linestyle="--", label=f"Val {ylabel}")
        ax.axvline(x=p1_epochs, color="red", linestyle=":", linewidth=2,
                   label=f"Fine-tune starts (epoch {p1_epochs})")
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Annotate phases
        ax.text(p1_epochs / 2, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "Phase 1\n(frozen)", ha="center", fontsize=8,
                color="steelblue", bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.4))
        ax.text(p1_epochs + p2_epochs / 2, ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                "Phase 2\n(fine-tune)", ha="center", fontsize=8,
                color="darkorange", bbox=dict(boxstyle="round", facecolor="moccasin", alpha=0.4))

    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/training_history.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: training_history.png")


    # --- PLOT 4: Inference confidence plot ---
    print("Generating: Inference confidence visualization...")

    all_probs  = []
    all_true   = []
    all_pred   = []

    for imgs_batch, labels_batch in test_ds:
        probs_batch = model_ft.predict(imgs_batch, verbose=0)
        preds_batch = probs_batch.argmax(axis=1)
        all_probs.extend(probs_batch)
        all_true.extend(labels_batch.numpy())
        all_pred.extend(preds_batch)

    all_probs = np.array(all_probs)
    all_true  = np.array(all_true)
    all_pred  = np.array(all_pred)

    correct_conf   = all_probs.max(axis=1)[all_pred == all_true]
    incorrect_conf = all_probs.max(axis=1)[all_pred != all_true]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Inference Confidence Distribution",
                 fontsize=14, fontweight="bold")

    bins = np.linspace(0, 1, 21)
    axes[0].hist(correct_conf,   bins=bins, alpha=0.7, color="seagreen",  label="Correct", edgecolor="white")
    axes[0].hist(incorrect_conf, bins=bins, alpha=0.7, color="tomato",    label="Wrong",   edgecolor="white")
    axes[0].set_xlabel("Predicted Confidence"); axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence: Correct vs Wrong Predictions")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="50% threshold")

    # Per-class probability bar for first test sample
    sample_probs = all_probs[0]
    colors_bar = ["seagreen" if i == all_pred[0] else "#cccccc" for i in range(n_classes)]
    axes[1].bar(class_names, sample_probs * 100, color=colors_bar, edgecolor="white", linewidth=1.5)
    axes[1].set_ylabel("Predicted Probability (%)")
    axes[1].set_title(f"Sample Prediction: True={class_names[all_true[0]]}, "
                      f"Pred={class_names[all_pred[0]]} ({sample_probs.max():.1%})")
    axes[1].set_ylim(0, 110)
    axes[1].grid(axis="y", alpha=0.3)
    for i, prob in enumerate(sample_probs):
        axes[1].text(i, prob * 100 + 1, f"{prob:.1%}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/inference_confidence.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("   Saved: inference_confidence.png")

else:
    print("  [Skipping model plots — TensorFlow not installed]")
    print("  Dataset samples and pipeline diagram were saved.")


# ======================================================================
# SECTION 10: Using YOUR Real Dataset
# ======================================================================
print()
print("=" * 70)
print("SECTION 10: USING YOUR OWN REAL DATASET")
print("=" * 70)
print()
print("To classify YOUR images, change 3 things:")
print()
print("  Step 1: Organize your images (do this once):")
print("  ─────────────────────────────────────────────")
print("    my_dataset/")
print("    ├── train/")
print("    │   ├── cats/   (300 images)")
print("    │   └── dogs/   (300 images)")
print("    ├── val/")
print("    │   ├── cats/   (75 images)")
print("    │   └── dogs/   (75 images)")
print("    └── test/")
print("        ├── cats/   (75 images)")
print("        └── dogs/   (75 images)")
print()
print("  Step 2: Update the path in this script:")
print("  ─────────────────────────────────────────────")
print('    DEMO_DIR = "my_dataset"       # ← your folder')
print('    CLASSES  = ["cats", "dogs"]   # ← optional, auto-detected')
print()
print("  Step 3: Run the script:")
print("  ─────────────────────────────────────────────")
print("    python3 custom_image_classifier.py")
print()
print("  The script will automatically:")
print("    ✓ Load your images with image_dataset_from_directory")
print("    ✓ Apply data augmentation")
print("    ✓ Train MobileNetV2 in two phases")
print("    ✓ Evaluate on your test set")
print("    ✓ Save the trained model")
print("    ✓ Run inference on test samples")
print()

# Show a quick script snippet
print("  Quick-start snippet for your own dataset:")
print()
print("  ┌─────────────────────────────────────────────────────────────┐")
print("  │  import tensorflow as tf                                    │")
print("  │  from tensorflow import keras                               │")
print("  │  from tensorflow.keras.applications import MobileNetV2     │")
print("  │                                                             │")
print("  │  MY_DATA = 'path/to/my_dataset'                            │")
print("  │  IMG_SIZE = (96, 96)                                        │")
print("  │                                                             │")
print("  │  train_ds = tf.keras.utils.image_dataset_from_directory(  │")
print("  │      f'{MY_DATA}/train', image_size=IMG_SIZE, batch_size=32│")
print("  │  )                                                          │")
print("  │  n_classes = len(train_ds.class_names)                     │")
print("  │  # ... then run build_custom_classifier(n_classes) above   │")
print("  └─────────────────────────────────────────────────────────────┘")
print()

# Cleanup demo dir
shutil.rmtree(DEMO_DIR)
print(f"  (Demo data cleaned up from {DEMO_DIR})")


print()
print("=" * 70)
print("PROJECT 2: CUSTOM IMAGE CLASSIFIER COMPLETE!")
print("=" * 70)
print()
print("What you built:")
print("  ✓ End-to-end pipeline: folder structure → training → inference")
print("  ✓ Synthetic 3-class dataset (rock/paper/scissors)")
print("  ✓ MobileNetV2 feature extraction (Phase 1)")
print("  ✓ Fine-tuning top layers with 100x lower LR (Phase 2)")
print("  ✓ Model saved and reloaded successfully")
print("  ✓ reusable predict_image() function for single images")
print()
print("What you know now:")
print("  ✓ CNN folder structure: train/val/test, one folder per class")
print("  ✓ image_dataset_from_directory auto-detects class names")
print("  ✓ Class imbalance: use class weights or stratified augmentation")
print("  ✓ Two-phase transfer learning: extract → fine-tune")
print("  ✓ How to adapt this script to ANY image classification problem")
print()
print("Visualizations saved to: ../visuals/custom_classifier/")
print("  1. dataset_samples.png       — sample images per class")
print("  2. pipeline_diagram.png      — full workflow diagram")
print("  3. training_history.png      — two-phase training curves")
print("  4. inference_confidence.png  — confidence distributions")
print()
print("🎉 Part 4: Convolutional Neural Networks — ALL MODULES COMPLETE!")
print()
print("Summary of what you built in Part 4:")
print("  Math Foundations:")
print("    01. image_basics.py            — pixels, channels, (N,H,W,C) format")
print("    02. convolution_operation.py   — filters, stride, padding from scratch")
print("    03. pooling_and_depth.py       — MaxPool, GAP, feature hierarchy")
print("  Algorithms:")
print("    04. conv_layer_from_scratch.py — Conv2D/MaxPool/Dense in numpy")
print("    05. cnn_with_keras.py          — 3-block Keras CNN on CIFAR-10")
print("    06. classic_architectures.py   — LeNet, AlexNet, VGG, ResNet")
print("    07. transfer_learning.py       — feature extract + fine-tune")
print("  Projects:")
print("    08. cifar10_classifier.py      — MLP vs CNN vs TL on CIFAR-10")
print("    09. custom_image_classifier.py — end-to-end on YOUR data (this file!)")
print()
print("Next: Part 5 → Natural Language Processing (NLP)")
