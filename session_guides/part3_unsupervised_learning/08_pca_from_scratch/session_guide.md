# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 08: PCA from Scratch
### Two-Session Teaching Script

> **The most important dimensionality reduction technique in all of ML.**
> PCA = find directions of maximum variance, project data there.
> You built the math in Modules 02-03. Now you build the algorithm.

---

# SESSION 1 (~90 min)
## "PCA — turning 100 features into 2 you can actually see"

## Before They Arrive
- Terminal open in `unsupervised_learning/algorithms/`
- Have Module 03 eigenvector notes handy for reference

---

## OPENING (10 min)

> *"Raise your hand if you've ever worked with a dataset with more than 10 columns.*
> *50 columns? 100? 1000?*
>
> *How do you visualize that? You can't.*
> *How do you find patterns? Hard.*
> *How do you know which features matter? Unknown.*
>
> *PCA takes all of that and collapses it.*
> *100 features → 2 features that capture 90% of the information.*
> *Now you can plot it. Now you can see the structure.*
> *Now clustering, classification, everything gets easier.*
>
> *Let's build it from scratch — you have all the math from Modules 02 and 03."*

Write on board:
```
PCA IN ONE SENTENCE:
  Find the directions of maximum variance in your data,
  then project everything onto those directions.

USE CASES:
  • Visualization (100D → 2D to plot)
  • Noise reduction (remove low-variance components)
  • Speed up ML (fewer features = faster training)
  • Remove multicollinearity (correlated features → orthogonal PCs)
  • Image compression (face reconstruction with fewer components)
```

---

## SECTION 1: The 5-Step Algorithm (25 min)

Write step by step, relating back to Modules 02-03:

```
PCA FROM SCRATCH — 5 STEPS:

Step 1: CENTER
  X_centered = X - mean(X, axis=0)
  "Move the data to the origin"
  Why: PCA finds directions of spread, not position

Step 2: COVARIANCE MATRIX (Module 02)
  C = X_centered.T @ X_centered / (n-1)
  "How do features vary together?"

Step 3: EIGENDECOMPOSITION (Module 03)
  eigenvalues, eigenvectors = np.linalg.eigh(C)
  "Find the directions of maximum spread"

Step 4: SORT
  Sort eigenvalues descending, reorder eigenvectors accordingly
  "PC1 = most important direction, PC2 = second most..."

Step 5: PROJECT
  X_pca = X_centered @ eigenvectors[:, :k]
  "Express data in terms of the new directions"
```

```bash
python3 pca_from_scratch.py
```

---

## SECTION 2: Building the PCA Class (30 min)

Live code:
```python
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components_ = None       # the eigenvectors (PC directions)
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        C = X_centered.T @ X_centered / (len(X) - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(C)
        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.components_ = eigenvectors[:, :self.n_components].T
        total_var = eigenvalues.sum()
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / total_var
        return self

    def transform(self, X):
        X_centered = X - self.mean_
        return X_centered @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_pca):
        return X_pca @ self.components_ + self.mean_
```

---

## CLOSING SESSION 1 (5 min)

```
SESSION 1 SUMMARY:
  PCA = find directions of max variance (eigenvectors of covariance matrix)
  5 steps: center → covariance → eigendecompose → sort → project
  Components: orthogonal (uncorrelated) directions
  inverse_transform: go back to original space (lossy if k < n_features)
```

---

# SESSION 2 (~90 min)
## "Explained variance, reconstruction, and when PCA fails"

## OPENING (5 min)

> *"Session 1: we built PCA.*
> *Today: how to choose how many components to keep,*
> *and when PCA is the right tool (and when it isn't)."*

---

## SECTION 1: Scree Plot and Explained Variance (20 min)

> *"How many components do you keep?*
> *The scree plot and cumulative variance curve tell you."*

```
EXPLAINED VARIANCE TABLE (example — Iris, 4 features):
  PC1: 72.96% variance (captures sepal/petal separation)
  PC2: 22.85% variance (captures within-species variation)
  PC3:  3.67% variance (mostly noise)
  PC4:  0.52% variance (noise)

Cumulative:
  2 components: 95.8% ✓ almost everything
  3 components: 99.5% ✓ all signal
  4 components: 100%   = original data (no compression)

RULE: Keep enough components for 90-95% cumulative variance.
For visualization: always use 2 (or 3 for 3D plots).
```

Show the scree plot from the program. Point to the elbow.

---

## SECTION 2: Reconstruction — Seeing What's Lost (20 min)

> *"PCA is lossy compression.*
> *We can go back (inverse_transform), but we lose the low-variance components.*
> *On the digits dataset: watch how quality improves with more components."*

```
DIGITS: 64 features (8×8 pixels)

Components  Variance  Reconstruction quality
─────────────────────────────────────────────
1           12%       Blurry silhouette
4           33%       Rough digit shape
10          55%       Recognizable digits
20          74%       Clear digits
40          90%       Near-perfect
64          100%      Original (no compression)
```

> *"At 20 components we kept 74% of the information*
> *and reduced dimensions from 64 to 20 — 3x compression.*
> *At 10 components: 6x compression, still recognizable.*
> *This is what image compression (JPEG) is fundamentally doing.*
> *PCA is the conceptual ancestor of JPEG."*

---

## SECTION 3: PCA vs sklearn — and When Not to Use PCA (20 min)

```python
from sklearn.decomposition import PCA

# Select enough components for 95% variance automatically
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)
print(f"Components needed: {pca.n_components_}")
```

**When PCA fails:**
```
PCA IS LINEAR — it can only rotate and scale.

PCA on Swiss Roll dataset:
  Original: 3D spiral manifold
  After PCA: uninterpretable 2D mess
  Because the structure is NON-LINEAR

For non-linear data use:
  t-SNE   — visualization of non-linear structure
  UMAP    — faster, also preserves global structure
  (Module 09: Dimensionality Reduction)

PCA also struggles with:
  Categorical features → MCA (Multiple Correspondence Analysis)
  Sparse data          → TruncatedSVD (no centering step)
  Non-Gaussian data    → ICA (Independent Component Analysis)
```

---

## CLOSING SESSION 2 (10 min)

```
PCA FROM SCRATCH — COMPLETE:
  Algorithm: center → covariance → eigen → sort → project
  Explained variance: how much each component captures
  Scree plot: choose k at the elbow
  Reconstruction: inverse_transform (lossy)
  Limitation: linear only — use t-SNE/UMAP for non-linear

  What you can now do:
  ✅ Reduce any dataset to 2D for visualization
  ✅ Remove correlated features before classification
  ✅ Compress and reconstruct image data
  ✅ Speed up any downstream ML algorithm
```

---

## INSTRUCTOR TIPS

**"Should I always do PCA before clustering?"**
> *"Not always. PCA for K-Means: often helps — removes noise and speeds up computation.*
> *But PCA destroys non-linear structure. If using DBSCAN on non-linear data,*
> *skip PCA (or use UMAP to reduce dimensions first instead).*
> *Rule: PCA before K-Means, UMAP before DBSCAN on complex data."*

**"What's TruncatedSVD vs PCA?"**
> *"PCA centers the data first (subtracts mean). TruncatedSVD does not.*
> *For sparse data (like TF-IDF matrices with thousands of zeros),*
> *centering destroys sparsity — memory explodes.*
> *sklearn's TruncatedSVD handles sparse matrices directly.*
> *For dense, continuous data: use PCA. For sparse text: TruncatedSVD."*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening — the dimensionality problem  10 min
├── 5-step algorithm                      25 min
├── Build PCA class from scratch          30 min
└── Close                                  5 min  (+ 20 min buffer)

SESSION 2  (90 min)
├── Opening                                5 min
├── Scree plot + explained variance       20 min
├── Reconstruction on digits              20 min
├── sklearn + when PCA fails              20 min
└── Close                                 10 min  (+ 15 min buffer)
```

---
*MLForBeginners · Part 3: Unsupervised Learning · Module 08*
