# 🎓 MLForBeginners — Instructor Guide
## Part 4 · Module 12: Autoencoders
### Two-Session Teaching Script

> **The bridge between unsupervised learning and deep learning.**
> No labels. The network compresses data into a bottleneck, then reconstructs it.
> Applications: denoising, anomaly detection, feature learning, generation.

---

# SESSION 1 (~90 min)
## "The self-taught compressor — unsupervised deep learning"

## Before They Arrive
- Terminal open in `deep_neural_networks/algorithms/`
- Draw on the board: a funnel getting narrow then widening again

---

## OPENING (10 min)

> *"Quick question: if I gave you a 784-pixel image of the digit '3'*
> *and told you to describe it in just 32 numbers — could you?*
>
> *That's exactly what an autoencoder learns to do.*
> *No one tells it WHICH 32 numbers to use.*
> *It figures that out through compression and reconstruction.*
>
> *Here's the twist that makes it unsupervised:*
> *the input IS the target.*
> *You feed it an image of a '3'. You ask it to reconstruct a '3'.*
> *No label. No 'this is a 3'. Just: match the input.*
>
> *Through that pressure — forced to pass through a 32-number bottleneck —*
> *it learns the essence of handwritten digits.*"

Write on board:
```
STANDARD NETWORK:        AUTOENCODER:
  Input → predict label    Input → COMPRESS → RECONSTRUCT → Input
  Needs labels             No labels needed! Input = Target

  Loss = cross_entropy     Loss = MSE(input, reconstruction)
         (need 'cat'=0)           (just compare pixel-by-pixel)
```

> *"This puts autoencoders in the unsupervised family (like Part 3).*
> *But they're built from the deep learning tools we've spent Part 4 learning.*
> *They're the bridge."*

---

## SECTION 1: The Architecture in Detail (20 min)

Open `01_architecture_concept.png` from the visuals folder.

> *"This is a CONCEPTUAL diagram — not a loss curve, not a scatter plot.*
> *It shows the actual shape of the network.*
> *Let's read it together."*

Walk through the diagram:
```
INPUT (784)
  → Dense 256 (ReLU)     ← encoder starts
  → Dense 128 (ReLU)
  → Dense 32  (ReLU)     ← BOTTLENECK: the compressed code
  → Dense 128 (ReLU)     ← decoder starts
  → Dense 256 (ReLU)
  → Dense 784 (Sigmoid)  ← output: pixel values 0–1
```

> *"Notice: the decoder mirrors the encoder.*
> *The encoder compresses. The decoder expands.*
> *Think of a camera lens focusing light to a point, then a projector expanding it back.*
>
> *The 32-number bottleneck is called the LATENT CODE or LATENT SPACE.*
> *Those 32 numbers are everything the network knows about the image.*
> *What does it choose to put there? That's what we discover.*"

---

## SECTION 2: From Scratch — Feel Every Weight (30 min)

```bash
python3 autoencoder.py
```

Watch the toy 2D → 1D → 2D autoencoder train:

> *"We start simpler than MNIST.*
> *Our data: points on a circle in 2D.*
> *Our bottleneck: 1 single number.*
>
> *Think about it: one number describes a point on a circle.*
> *What IS that one number?*
> *(Let them guess — it's the angle)*
>
> *The network discovers 'angle' by itself.*
> *Nobody told it about angles. It figured out that the circle*
> *has one degree of freedom, and that's the most useful thing to encode.*
>
> *This is PCA's non-linear cousin.*
> *PCA finds linear directions of variance.*
> *The autoencoder finds non-linear directions of variance."*

Watch loss decrease. Ask:
> *"What should happen to the loss?*
> *(Down). What does it mean when loss plateaus?*
> *(The network has learned what it can — more epochs won't help.)*"

---

## SECTION 3: What Does the Bottleneck Force? (15 min)

Write on board:
```
THE COMPRESSION PRESSURE:

Without bottleneck: network could just copy input (shortcut!)
With bottleneck:    MUST compress → can't copy → must understand

This is like studying for an exam with an open book vs closed book.
Open book: you look everything up (copy).
Closed book: you have to ACTUALLY UNDERSTAND to reconstruct.

The bottleneck = closed book.
The reconstruction quality = your grade.

BIG bottleneck (512 neurons): easy → lazy learning
SMALL bottleneck (2 neurons):  hard → must find true structure
TOO SMALL:                      impossible → network fails
```

> *"The bottleneck size is a hyperparameter.*
> *How small is too small?*
> *For MNIST: 32 works well. You can try 2 — it reconstructs blurry but readable digits.*
> *For face images: you need more — faces have more variation than digits."*

---

## CLOSING SESSION 1 (5 min)

```
SESSION 1 SUMMARY:
  Autoencoder = encoder + bottleneck + decoder
  Training: input = target, loss = MSE (no labels!)
  Bottleneck = compressed latent code
  Compression forces understanding (no shortcut)
  Connects Part 3 (unsupervised) to Part 4 (deep learning)
```

**Homework:**
> *"Think of three use cases where you'd want*
> *compressed representations of data.*
> *Hint: what's expensive to store? What's expensive to transmit?"*

---

# SESSION 2 (~90 min)
## "Denoising, anomaly detection, and the latent space"

## OPENING (5 min)

> *"Session 1: what autoencoders are and how they're built.*
> *Today: what they're used for.*
> *Three killer applications: denoising, anomaly detection, latent space exploration.*
> *Each one builds directly on the bottleneck idea."*

---

## SECTION 1: Denoising Autoencoder (25 min)

Open `02_reconstructions.png`.

> *"Three rows. Top: clean original. Middle: noisy (we added Gaussian noise).*
> *Bottom: what the denoising autoencoder outputs.*
>
> *The noisy row barely looks like digits.*
> *The bottom row: clean. Sharp. Correct.*
>
> *How? The training trick:"*

Write on board:
```
STANDARD AE TRAINING:     DENOISING AE TRAINING:
  Input:  clean image       Input:  noisy image  (noise added)
  Target: clean image       Target: clean image  (same as always)
  ─────────────────────     ──────────────────────────────────
  Learns to copy self       Learns to REMOVE noise

  Loss = MSE(clean, recon)  Loss = MSE(clean, denoised_recon)
```

> *"The network never sees the same noisy image twice.*
> *Different noise is added randomly each batch.*
> *So it can't memorize the noise — it has to learn to ignore it.*
>
> *Use cases: satellite imagery, medical scans (MRI/CT noise),*
> *audio cleanup, old photograph restoration.*
> *Same architecture. Different training data."*

**Ask the room:**
> *"What if we used dropout as noise instead of Gaussian noise?*
> *We'd randomly zero out pixels.*
> *The network learns: 'if some pixels are missing, reconstruct from context.'*
> *That's exactly how BERT works in NLP — we'll see this in Part 7!"*

---

## SECTION 2: Anomaly Detection (20 min)

Open `03_latent_space_and_anomaly.png` — right panel.

> *"This autoencoder was trained ONLY on the digit '0'.*
> *Nothing else. No 1s, 2s, 3s.*
> *Now we test it on all ten digits.*
>
> *Look at the reconstruction error bars.*
> *Digit '0': very low error — it's seen this before.*
> *Digit '1', '2', ...: high error — completely foreign to it.*
>
> *The reconstruction error IS the anomaly score.*
> *High error = 'I've never seen anything like this' = anomaly.*"

Write on board:
```
AUTOENCODER ANOMALY DETECTION:
  Train:  only on NORMAL data (digit 0 / normal transactions / healthy sensors)
  Deploy: run new data through
  Score:  reconstruction_error = MSE(original, reconstruction)

  Low error  → looks familiar → NORMAL
  High error → looks foreign  → ANOMALY ⚠️

WHY IT WORKS:
  The bottleneck is shaped to the NORMAL data manifold.
  Anomalies don't fit that shape → can't be reconstructed well.

CONNECTS TO PART 3:
  Isolation Forest:   "anomalies are easy to isolate"
  LOF:                "anomalies have low density"
  Autoencoder:        "anomalies have high reconstruction error"
  → Same goal, different mechanism
```

> *"This is used in production at banks (fraud), data centers (server failure),*
> *manufacturing (defective parts), healthcare (rare diseases).*
> *One trained model. No labeled anomalies required."*

---

## SECTION 3: The Latent Space (20 min)

Open `03_latent_space_and_anomaly.png` — middle panel.

> *"This is my favorite part.*
> *We forced the bottleneck down to just 2 dimensions.*
> *Now we can plot every digit as a point in 2D.*
> *Each color is a digit class — but the network NEVER SAW these labels.*
>
> *Look at the structure:*
> *Zeros cluster together. Ones cluster together. Fours and nines near each other.*
> *(Makes sense — they look similar: both have vertical lines.)*
>
> *The autoencoder learned the geometry of handwritten digits*
> *by trying to compress and reconstruct them.*
> *No human labeling. No supervised signal. Just:*
> *'compress to 2 numbers and reconstruct back.'*
>
> *This is what PCA does for linear data.*
> *The autoencoder does it for complex, non-linear data like images."*

Write:
```
LATENT SPACE PROPERTIES:
  • Nearby points = similar-looking inputs
  • Clusters = naturally similar groups (found without labels!)
  • Moving smoothly through latent space = morphing between examples
  • Can interpolate: encode A, encode B, blend the codes → blend the images

THE GENERATIVE LEAP:
  If we know the latent space is structured...
  What if we SAMPLE from it and DECODE?
  → Generate new images that look like training data!
  That's a Variational Autoencoder (VAE).
  The conceptual ancestor of Stable Diffusion and DALL-E.
```

---

## CLOSING SESSION 2 (10 min)

```
AUTOENCODERS — COMPLETE:
  ✅ Architecture: encoder → bottleneck → decoder
  ✅ Unsupervised: input = target, no labels
  ✅ Denoising: noisy input → clean output
  ✅ Anomaly detection: high reconstruction error = anomaly
  ✅ Latent space: compressed representation with structure

THE BIG PICTURE:
  Part 3 Unsupervised: find structure without labels
    K-Means, PCA, DBSCAN, Isolation Forest
  Part 4 Deep Learning: learn representations
    Perceptron, MLP, Keras, Hyperparameter tuning
  Autoencoder = WHERE THESE TWO WORLDS MEET
    Unsupervised goal (find structure)
    + Deep learning method (neural network)
```

---

## About the Visuals

> *"Notice that visualization 1 is a conceptual diagram — boxes, arrows,*
> *the shape of the network — not just a loss curve.*
> *Always look for these architectural diagrams when learning new models.*
> *They tell you: what information flows where.*
> *Draw them yourself for any new architecture you encounter."*

---

## INSTRUCTOR TIPS

**"What's a Variational Autoencoder (VAE)?"**
> *"A regular autoencoder maps each input to ONE point in latent space.*
> *A VAE maps each input to a DISTRIBUTION (mean + variance) in latent space.*
> *Then it samples from that distribution before decoding.*
> *This forces the latent space to be smooth and continuous —*
> *so you CAN sample from it and generate realistic new images.*
> *Think: 'generate a new digit' by sampling a point and decoding.*
> *We'll cover VAEs briefly in Part 8 (LLMs), where they appear in diffusion models."*

**"How is this different from PCA?"**
> *"PCA: linear transformation, closed-form solution, fast, interpretable.*
> *Autoencoder: non-linear, learned by gradient descent, slower, harder to interpret.*
> *PCA: same projection no matter the data.*
> *Autoencoder: projection shaped to YOUR dataset.*
> *Rule: try PCA first (fast). If structure is non-linear: autoencoder.*
> *Same rule as: try linear regression before neural networks."*

**"Can I use the bottleneck features for downstream tasks?"**
> *"Yes! This is called representation learning.*
> *Train autoencoder on all your data (unlabeled).*
> *Then use the bottleneck output as features for a classifier.*
> *Especially useful when you have lots of unlabeled data*
> *but few labeled examples — semi-supervised learning.*
> *BERT (Part 7) does this at massive scale."*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening — the self-compressor       10 min
├── Architecture diagram walkthrough    20 min
├── From scratch — toy circle           30 min
├── What the bottleneck forces          15 min
└── Close + homework                     5 min  (+ 10 min buffer)

SESSION 2  (90 min)
├── Opening                              5 min
├── Denoising autoencoder               25 min
├── Anomaly detection                   20 min
├── Latent space exploration            20 min
└── Close + big picture                 10 min  (+ 10 min buffer)
```

---
*MLForBeginners · Part 4: Deep Neural Networks · Module 12*
