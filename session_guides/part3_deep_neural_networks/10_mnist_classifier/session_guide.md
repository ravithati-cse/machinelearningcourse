# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 10: MNIST Digit Classifier (Project)
### Single 120-Minute Session

> **The moment they've been building toward.**
> They can now classify handwritten digits from images using a neural network.
> This is the "Hello World" of deep learning — make it feel like a milestone.

---

# SESSION (120 min)
## "Teaching a computer to read handwriting"

## Before They Arrive
- Terminal open in `deep_neural_networks/projects/`
- Write the digits 0-9 on the whiteboard in your own handwriting
- Optional: print some handwritten digit images from the MNIST dataset

---

## OPENING (10 min)

> *"Write any digit from 0 to 9 on a piece of paper.*
> *Now — how does your brain know what digit it is?*
>
> *You've seen thousands of examples of each digit.*
> *Your brain learned patterns: 0 has a loop, 1 is a vertical line,*
> *8 has two loops stacked...*
>
> *Today we teach a neural network to do the same thing.*
> *60,000 training images. 10,000 test images.*
> *The same dataset that launched modern deep learning in the 1990s.*
> *And you're going to do it in about 20 lines of Keras."*

---

## SECTION 1: The MNIST Dataset (15 min)

```bash
python3 mnist_digit_classifier.py
```

While it downloads/loads:

> *"MNIST: Modified National Institute of Standards and Technology.*
> *28×28 pixel grayscale images of handwritten digits.*
> *10 classes: digits 0 through 9.*
>
> *Each image = 784 numbers (28×28 pixels), each 0-255.*
> *That's our input. The output: which digit is it?"*

Write on board:
```
IMAGE (28×28 pixels):        NEURAL NETWORK:
┌─────────────────┐
│ 0  0  0  0  0   │     784 inputs
│ 0  0 255 255 0  │ →   [Hidden: 128 neurons]  → 10 outputs
│ 0 255  0 255 0  │     [Hidden:  64 neurons]     (one per digit)
│ 0  0 255 255 0  │
│ 0  0  0  0  0   │     Output with highest value = predicted digit
└─────────────────┘
```

> *"We flatten the 28×28 grid into a list of 784 numbers.*
> *Normalize to 0-1 (divide by 255).*
> *Then feed into our familiar MLP."*

---

## SECTION 2: Training Live (20 min)

Watch the training progress together:

> *"See the loss dropping? The accuracy climbing?*
> *Each epoch the network sees all 60,000 images and adjusts.*
>
> *What accuracy do you think we'll hit?*
> *A human gets ~98%. Random guessing = 10%.*
> *Our simple MLP usually hits 97-98%.*
>
> *That's genuinely impressive for a network we understand completely."*

As each epoch prints, celebrate small wins:
- Epoch 1: "It's already learning!"
- Epoch 5: "Over 95%!"
- Final: "97%+ — better than many humans at speed!"

---

## SECTION 3: Confusion Matrix — Where Does It Fail? (20 min)

Open the confusion matrix visualization.

> *"The confusion matrix tells us exactly where the model makes mistakes.*
> *Dark diagonal = good. Bright off-diagonal = confusion."*

Point out the interesting off-diagonal cells:

> *"4s and 9s get confused — they look similar.*
> *3s and 8s too. 1s and 7s.*
>
> *These are the SAME confusions humans make when reading messy handwriting.*
> *The model is making human-like errors — that's actually a good sign."*

**Ask the room:** *"Which two digits do YOU think look most alike? Look at the matrix."*

---

## SECTION 4: Visualizing Right and Wrong (20 min)

Open the visualization of correct vs incorrect predictions.

> *"Let's look at the ones it got wrong.*
>
> *[Show examples] — see this? Even I'm not sure what digit that is.*
> *The model gets it wrong, but honestly... I might too.*
>
> *And this one — that's clearly a 3 but the model said 8.*
> *Why? Look at the curve — it's very loopy.*
> *The network learned 'loopy = 8' a bit too aggressively."*

This creates intuition for:
- What "hard" examples look like
- The limits of a simple MLP on image data
- Why we'll need CNNs (Part 4) for harder image tasks

> *"Our MLP treats pixels independently — it doesn't know that*
> *nearby pixels form shapes. CNNs will fix that.*
> *But 97% on MNIST? Not bad for a few layers of simple math."*

---

## SECTION 5: Make a Prediction on Your Own Writing (15 min)

If you have a drawing tablet or webcam setup:
> *"Let's draw a digit and see what the model predicts."*

Otherwise use the built-in test image display and let them pick numbers to test:

```python
# Pick any test image
idx = 42   # let each person pick a number
image = X_test[idx]
label = y_test[idx]
prediction = model.predict(image.reshape(1, -1))[0].argmax()
print(f"True label: {label}, Predicted: {prediction}")
```

Go around the room — each person picks a test index, they predict first, then check.

---

## CLOSING (10 min)

Write on board:
```
MNIST CLASSIFIER — WHAT WE BUILT:
  60,000 training images → 97%+ accuracy on unseen digits
  Simple 2-layer MLP → better than many humans at speed

WHAT WE LEARNED:
  → Image data = flatten pixels + normalize
  → Confusion matrix shows WHERE the model fails
  → Error analysis is as important as accuracy
  → Simple MLPs work well on simple images (MNIST)
  → We'll need CNNs for harder tasks (cats vs dogs)

NEXT: tabular_deep_learning.py
  → Does deep learning beat traditional ML on structured data?
  → Sometimes. Let's find out when.
```

**High-five moment.** This deserves one.

---

## INSTRUCTOR TIPS

**"Why is MNIST considered 'too easy' now?"**
> *"Modern networks hit 99.8%+ on MNIST.*
> *It's been 'solved' — so the field moved to harder datasets:*
> *CIFAR-10, ImageNet (1.2M images, 1000 classes).*
> *But MNIST is still perfect for learning the pipeline."*

**"What if we used a CNN instead of MLP?"**
> *"CNN gets ~99.5% on MNIST. MLP gets ~97-98%.*
> *The 2% gap doesn't sound like much but at scale it matters.*
> *That's Part 4 — CNNs are built for image patterns."*

---

## Quick Reference
```
Single Session (120 min)
├── Opening hook             10 min
├── Dataset exploration      15 min
├── Training live            20 min
├── Confusion matrix         20 min
├── Right vs wrong viz       20 min
├── Custom predictions       15 min
└── Close                    20 min
```

---
*MLForBeginners · Part 3: Deep Neural Networks · Module 10*
