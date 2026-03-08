# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 09: Dimensionality Reduction (t-SNE & UMAP)
### Two-Session Teaching Script

> **When PCA isn't enough — non-linear structure.**
> t-SNE reveals clusters hidden in 100+ dimensions.
> UMAP is faster and also usable as ML preprocessing.
> These are the visualization tools of modern ML research.

---

# SESSION 1 (~90 min)
## "t-SNE — seeing what PCA can't show you"

## Before They Arrive
- Terminal open in `unsupervised_learning/algorithms/`
- Pull up a t-SNE visualization of MNIST from Google/Wikipedia for reference

---

## OPENING (10 min)

> *"Every ML paper you read will have a t-SNE plot.*
> *It looks like a scattered galaxy of colored dots.*
> *You've probably seen them without knowing what they are.*
>
> *This is what a t-SNE of the MNIST digits looks like:*
> *(show the reference image)*
> *10 distinct blobs — one per digit class — with no overlap.*
> *The model didn't know the labels. It just found the structure.*
>
> *PCA on those same digits gives you a smeared blob.*
> *Because MNIST has non-linear structure: pixel combinations*
> *form a curved manifold in 784-dimensional space.*
> *PCA can't unfold that. t-SNE can.*
>
> *Today: why, how, and when to use t-SNE."*

---

## SECTION 1: The Non-Linear Problem (15 min)

Draw on board:
```
SWISS ROLL DATA:
  In 3D: looks like a rolled-up carpet
  In 2D (PCA): squished mess — loses all structure
  In 2D (t-SNE): unrolled — the spiral is visible!

Why:
  PCA = linear projection (rotate and squish)
  t-SNE = non-linear embedding (preserve LOCAL structure)

PCA preserves: global distances
t-SNE preserves: local neighborhoods (who's near whom)
```

```bash
python3 dimensionality_reduction.py
```

Watch the three projections of the digits dataset build: PCA → t-SNE → UMAP.

---

## SECTION 2: How t-SNE Works (Intuition) (30 min)

> *"t-SNE doesn't preserve distances. It preserves NEIGHBORHOODS.*
> *Here's the key idea:"*

Write on board:
```
t-SNE: t-distributed Stochastic Neighbor Embedding

Step 1: HIGH-DIMENSIONAL similarities
  For each pair of points (i,j): P(j|i) = how likely j is a neighbor of i
  Based on Gaussian distribution centered at i
  Perplexity parameter controls "neighborhood size" (usually 5-50)

Step 2: LOW-DIMENSIONAL initialization
  Start with random 2D layout

Step 3: OPTIMIZE
  Find a 2D layout where Q(i,j) (2D probabilities) ≈ P(i,j) (HD probabilities)
  Minimize KL divergence: KL(P || Q)
  Points that are neighbors in HD → keep them close in 2D
  Points that aren't neighbors → push apart

The "t" in t-SNE:
  Uses t-distribution (heavier tails) for Q
  Prevents the "crowding problem" (points getting squished to center)
```

> *"The result: local structure is preserved.*
> *If two points are neighbors in 784D, they'll be close in 2D.*
> *Global structure: less reliable — don't read too much into*
> *the distances BETWEEN clusters."*

---

## SECTION 3: Perplexity and What It Controls (15 min)

Write on board:
```
PERPLEXITY = effective number of neighbors to preserve

Low perplexity (5):   Very local structure, many small clusters
                      Often shows too much fragmentation
Medium (30-50):       Balanced — usually best choice
High perplexity (100): More global, clusters merge

Rule: perplexity = 5 to 50, default 30 is good for most datasets

ALWAYS RUN MULTIPLE PERPLEXITIES and compare.
Different settings can show completely different structure.
t-SNE is exploratory — it's a lens, not a truth.
```

---

## CLOSING SESSION 1 (10 min)

```
SESSION 1 SUMMARY:
  t-SNE: non-linear embedding for visualization
  Preserves local neighborhoods, not global distances
  Perplexity: neighborhood size (30 is a safe default)
  Use: ONLY for visualization — not for ML features
  Limit: slow (O(n²)), not for datasets > 10K points
```

---

# SESSION 2 (~90 min)
## "UMAP — faster, better, and actually useful for ML"

## OPENING (5 min)

> *"t-SNE: great for visualization, but slow and only for visualization.*
> *UMAP: faster, better global structure, AND can be used as ML preprocessing.*
> *Today: UMAP and the final comparison of all three methods."*

---

## SECTION 1: UMAP vs t-SNE (25 min)

Write comparison on board:
```
                    PCA      t-SNE      UMAP
────────────────────────────────────────────────
Speed              Fast     Slow       Fast
Preserves local    Good     Excellent  Excellent
Preserves global   Good     Poor       Good
Non-linear data    Poor     Good       Excellent
For visualization  OK       Best       Best
For ML features    YES      NO         YES
Deterministic      YES      NO         NO (mostly)
Handles new data   YES      NO         YES
Scales to large    YES      NO         YES (approx.)
```

```python
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)
    # Can also transform new data:
    X_new_umap = reducer.transform(X_new)
except ImportError:
    print("Install: pip install umap-learn")
```

> *"UMAP is theoretically grounded in Riemannian geometry and topology.*
> *But you don't need to understand that to use it.*
> *Use UMAP when: visualization + you might also want to use it for features.*
> *Use t-SNE when: you ONLY need visualization and have < 10K points."*

---

## SECTION 2: When Each Method Shines (20 min)

Show three datasets side by side:

```
BLOBS: PCA wins (linear separation, fast)
SWISS ROLL: UMAP wins (non-linear manifold)
MNIST 784D: t-SNE and UMAP both excellent
```

> *"For your workflow:*
> *1. Always start with PCA for a quick look.*
> *2. If structure is unclear → run t-SNE or UMAP.*
> *3. If you need the reduced features for downstream ML → UMAP.*
> *4. Final paper-quality visualization → t-SNE or UMAP, run several times.*
>
> *In research: both often shown to be comprehensive.*
> *In production: UMAP (can transform new incoming data)."*

---

## SECTION 3: Practical Tips (15 min)

```
t-SNE PRACTICAL GUIDE:
  → perplexity: 5-50 (start with 30)
  → n_iter: at least 1000 (default 1000 is fine)
  → Run multiple times with different random seeds
  → Cluster SHAPES and DISTANCES between clusters: not meaningful
  → Cluster EXISTENCE and relative densities: meaningful

UMAP PRACTICAL GUIDE:
  → n_neighbors: 5-50 (analog of perplexity, default 15)
  → min_dist: 0.0-1.0 (how tightly packed clusters are, default 0.1)
  → metric: 'euclidean' for numbers, 'cosine' for text
  → More stable across runs than t-SNE

BOTH: normalize your data first with StandardScaler
```

---

## CLOSING SESSION 2 (15 min)

```
DIMENSIONALITY REDUCTION — COMPLETE:

  PCA:    linear, fast, interpretable, global structure
  t-SNE:  non-linear, visualization only, slow
  UMAP:   non-linear, fast, can use for ML features

  Decision guide:
  → n < 1000, visualization: t-SNE
  → n > 1000, visualization: UMAP
  → Features for ML: PCA (linear) or UMAP (non-linear)
  → Interpretability needed: PCA (components have meaning)

  Coming up: putting it all together in the projects.
  Next: Customer Segmentation — K-Means + PCA in a real business context.
```

---

## INSTRUCTOR TIPS

**"Can I use t-SNE features to train a classifier?"**
> *"No. Two reasons:*
> *1. t-SNE can't transform new data (no .transform() method)*
> *2. t-SNE results change every run — your test set would be incompatible*
> *For features: PCA or UMAP. t-SNE: visualization only."*

**"Why does t-SNE sometimes show 'false clusters'?"**
> *"t-SNE can split one real cluster into multiple clusters.*
> *High perplexity often fixes this.*
> *Always validate: run K-Means on original data and color t-SNE by those labels.*
> *If the K-Means clusters align with t-SNE clusters: real structure.*
> *If they don't: artifact of t-SNE."*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening — the t-SNE plot         10 min
├── Non-linear problem               15 min
├── How t-SNE works                  30 min
├── Perplexity tuning                15 min
└── Close                            10 min

SESSION 2  (90 min)
├── Opening                           5 min
├── UMAP vs t-SNE comparison         25 min
├── When each shines                 20 min
├── Practical tips                   15 min
└── Close                            15 min  (+ 10 min buffer)
```

---
*MLForBeginners · Part 3: Unsupervised Learning · Module 09*
