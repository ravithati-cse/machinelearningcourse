# 🎓 MLForBeginners — Instructor Guide
## Part 4 · Module 09: Custom Image Classifier (Project + Part 4 Capstone)
### Single 120-Minute Session

> **The Part 4 graduation project.**
> They've mastered convolutions, pooling, classic architectures, and transfer learning.
> Now they build a classifier on THEIR OWN categories — images they chose.
> This is the moment CNN knowledge becomes a personal superpower.

---

# SESSION (120 min)
## "Build a classifier for anything you care about"

## Before They Arrive
- Terminal open in `convolutional_neural_networks/projects/`
- Ask each student to think of 2-3 image categories they find interesting
- Examples ready: cats vs dogs, cars vs bikes, fruits, flowers, faces

---

## OPENING (10 min)

> *\"Every CNN project we've done so far used pre-packaged datasets.*
> *MNIST: here are 60,000 labeled digits.*
> *CIFAR-10: here are 60,000 labeled photos.*
>
> *Real-world ML doesn't work that way.*
> *Your boss doesn't hand you a perfectly labeled dataset.*
> *They say: 'Can you build something that identifies our product defects?'*
>
> *Today you do exactly that.*
> *You pick the categories. You provide (or find) the images.*
> *You build the classifier.*
> *This is what it actually means to apply computer vision.\"*

Write on board:
```
THE REAL ML WORKFLOW:
  1. Define the problem (what do we need to classify?)
  2. Collect / organize data (the hard part)
  3. Build & train the model
  4. Evaluate & iterate
  5. Deploy

  Today: we do steps 1-4 with YOUR chosen categories.
```

---

## SECTION 1: Defining Your Problem (10 min)

Go around the room — each person names their 2-3 categories.

Ideas to suggest if they're stuck:
```
EASY (distinctive visual differences):
  cats vs dogs
  apples vs oranges vs bananas
  cars vs motorcycles vs bicycles

MEDIUM:
  different dog breeds (3-4)
  indoor vs outdoor scenes
  daytime vs nighttime photos

HARD (subtle differences):
  different flower species
  art styles (impressionism vs cubism)
  satellite: forest vs urban vs water
```

> *\"The harder the problem, the more data and the better architecture you need.*
> *For today: pick something with clear visual differences.*
> *Rule of thumb: if YOU can't tell them apart at a glance, neither can the model.\"*

---

## SECTION 2: Data Collection & Organization (15 min)

```bash
python3 custom_image_classifier.py
```

While the framework loads, explain data requirements:

> *\"For each class you need at minimum 50-100 images.*
> *More is always better. 1,000 per class = solid.*
>
> *Where to get images:*
> *— Google Images (download manually or via scripts)*
> *— Open datasets: ImageNet, Open Images, Flickr*
> *— Your own photos (most meaningful!)*
>
> *Data quality beats quantity.*
> *100 clean, correctly labeled images > 1,000 messy ones.\"*

Write the folder structure on board:
```
data/
  custom_classifier/
    train/
      class_a/  ← 80% of images
        img001.jpg
        img002.jpg
      class_b/
        img001.jpg
    val/
      class_a/  ← 10% of images
      class_b/
    test/
      class_a/  ← 10% of images
      class_b/
```

> *\"Keras's ImageDataGenerator reads this structure automatically.*
> *Folder name = class label. Simple and powerful.\"*

---

## SECTION 3: Transfer Learning in Practice (20 min)

> *\"For a custom classifier with limited data, transfer learning is non-negotiable.*
> *We use a pretrained backbone — already knows edges, shapes, textures —*
> *and add our own head for our specific classes.\"*

Show the architecture live:

```python
# The pattern every real project uses
base_model = MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False   # frozen — keeps ImageNet knowledge

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(base_model.input, predictions)
```

> *\"Phase 1: Train ONLY the new head. Fast. Converges in 5-10 epochs.*
> *Phase 2: Unfreeze the top layers. Fine-tune EVERYTHING. Slow, but gets extra %.*
>
> *This two-phase approach is standard industry practice.*
> *Google uses it. OpenAI uses it. Now you do too.\"*

---

## SECTION 4: Data Augmentation — Multiplying Your Dataset (15 min)

Write on board:
```
THE AUGMENTATION TOOLKIT:

Original image → apply random transforms → more training variety

Horizontal flip    → double your dataset instantly
Random rotation    → ±15 degrees
Zoom               → 0.8x to 1.2x
Width/height shift → ±10% of image size
Brightness adjust  → ±20%

Result: model learns the concept, not the exact pixel arrangement
```

> *\"Data augmentation is legal 'cheating'.*
> *You don't have 1,000 images — but with augmentation you simulate*
> *thousands of variations from your 100 images.*
>
> *The model sees each image many times, but slightly different each time.*
> *Like studying flashcards: flip them, rotate them, quiz yourself in different orders.*
> *You learn the concept, not the card.\"*

```python
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.2,
    preprocessing_function=preprocess_input
)
```

---

## SECTION 5: Training and Evaluation (20 min)

Watch the training together:

> *\"Phase 1 accuracy climbs fast — the head learns quickly.*
> *Phase 2 is slower — we're adjusting the backbone too.*
>
> *What accuracy would satisfy you for your use case?*
> *A toy classifier: 80% is fine.*
> *Medical diagnosis: 99%+ required.*
> *The bar depends entirely on the stakes.\"*

Open the confusion matrix and sample predictions.

**Interactive:** Go around the room:
> *\"Tell me one correct prediction and one incorrect prediction from your results.*
> *Why do you think the model got confused on that one?\"*

Common causes of errors:
```
TYPICAL FAILURE MODES:
  Low quality images (blurry, dark, cropped wrong)
  Class imbalance (more dogs than cats)
  Unusual viewpoints not seen in training
  Background clutter confusing the model
  Very similar inter-class appearance
```

---

## CLOSING + PART 4 GRADUATION (10 min)

Write on board:

```
PART 4 COMPLETE — CNNs MASTERED

You can now:
  ✅ Explain convolution, pooling, feature maps
  ✅ Build CNNs from scratch AND with Keras
  ✅ Understand classic architectures (VGG, ResNet, Inception)
  ✅ Apply transfer learning with fine-tuning
  ✅ Use data augmentation effectively
  ✅ Build a classifier for ANY set of image categories

WHAT COMES NEXT:
  Part 5: Natural Language Processing
  → What if instead of pixels, our input is words?
  → The same deep learning ideas, but now for text.
  → Sentiment analysis, spam detection, sequence models.
```

> *\"You just built a system that can learn to distinguish*
> *anything visually separable.*
>
> *That drone footage analyzer, that quality control system,*
> *that medical image screener — they all start here.*
> *You now have the toolkit to build them.\"*

**Graduation moment:** Each person takes a photo with their phone, runs it through their classifier, shares the result with the group.

---

## INSTRUCTOR TIPS

**"How many images do I really need?"**
> *"With transfer learning: as few as 50-100 per class can work.*
> *With augmentation: you can get away with even fewer.*
> *Rule: if your val accuracy climbs during training, you have enough.*
> *If it plateaus at 60-70%, you need more data.\"*

**"What if I'm stuck at 70% accuracy?"**
> *"Check your data first: are images correctly labeled?*
> *Are categories genuinely visually distinct?*
> *Then: more data, stronger augmentation, unfreeze more layers.*
> *Finally: try a larger backbone (EfficientNetB4 instead of MobileNetV2).\"*

**"When would I NOT use transfer learning?"**
> *"If your domain is very different from ImageNet — like satellite imagery,*
> *X-rays, or microscopy. But even then, ImageNet features often generalize better*
> *than random initialization. Always try transfer learning first.\"*

---

## Quick Reference
```
Single Session (120 min)
├── Opening + motivation       10 min
├── Define your problem        10 min
├── Data collection setup      15 min
├── Transfer learning review   20 min
├── Data augmentation          15 min
├── Training + evaluation      20 min
└── Part 4 graduation          10 min
```

---
*MLForBeginners · Part 4: CNNs · Module 09 (Capstone)*
