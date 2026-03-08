# 🎓 MLForBeginners — Instructor Guide
## Part 3 · Module 03: Eigenvectors for PCA
### Two-Session Teaching Script

> **The final math key before we build PCA.**
> Eigenvectors sound scary. They're not.
> "Directions in space that a matrix only stretches — never rotates."
> The covariance matrix's eigenvectors ARE the principal components.

---

# SESSION 1 (~90 min)
## "Eigenvectors — directions that don't spin"

## Before They Arrive
- Terminal open in `unsupervised_learning/math_foundations/`
- Draw a 2D coordinate system with an arrow pointing diagonally

---

## OPENING (10 min)

> *"I want to show you something weird first.*
>
> *Take the matrix [[3, 1], [0, 2]].*
> *Multiply it by the vector [1, 0].*
> *Result: [3, 0]. Same direction, just stretched by 3.*
>
> *Multiply by [0, 1]:*
> *Result: [0, 2]. Same direction, stretched by 2.*
>
> *Multiply by [1, 1]:*
> *Result: [4, 2]. DIFFERENT direction! It rotated.*
>
> *[1,0] and [0,1] are special vectors: the matrix only stretches them.*
> *Those are the eigenvectors.*
> *3 and 2 are the eigenvalues — the stretch factors."*

Write on board:
```
EIGENVECTOR: a vector v where Av = λv
  A  = a matrix (like the covariance matrix)
  v  = the eigenvector (a special direction)
  λ  = the eigenvalue (how much it's stretched)

"Eigen" is German for "own" or "characteristic"
→ The matrix's OWN special directions
```

---

## SECTION 1: Visual Intuition (25 min)

> *"Every matrix transforms space. Most vectors get both stretched AND rotated.*
> *Eigenvectors only get stretched."*

Draw on board:
```
BEFORE matrix:           AFTER matrix:
         ↑ v₂                    ↑↑ Av₂ (stretched, same direction)
         |                       |
    ──── + ────              ──────── + ──────── (stretched, same direction)
         |                       |
                    → v₁

Most vectors rotate:  v → Av (different direction)
Eigenvectors:         v → λv (same direction, just scaled)
```

Open the program and watch the visualization:
```bash
python3 03_eigenvectors_for_pca.py
```

> *"See the arrows on the scatter plot? Those are the eigenvectors*
> *of the data's covariance matrix.*
> *They point in the directions of maximum and minimum spread.*
> *The longest arrow is the FIRST principal component."*

---

## SECTION 2: Computing Eigenvectors (20 min)

> *"You don't compute eigenvectors by hand.*
> *Not even data scientists do. That's what numpy is for.*
> *But you should know what the algorithm is finding."*

```python
import numpy as np

C = np.array([[3, 2],     # covariance matrix
              [2, 2]])

eigenvalues, eigenvectors = np.linalg.eigh(C)
# eigh for symmetric matrices (covariance matrices are always symmetric)

print("Eigenvalues:", eigenvalues)    # [0.76, 4.24]
print("Eigenvectors:", eigenvectors)  # columns are eigenvectors
```

> *"Two eigenvalues: 4.24 and 0.76.*
> *First component (λ=4.24) captures most variance.*
> *Second component (λ=0.76) captures the rest.*
> *Total variance: 4.24 + 0.76 = 5.0*
> *First component: 4.24/5.0 = 85% of total variance.*
>
> *That's the explained variance ratio.*
> *Scree plot shows this — how many components to keep?"*

---

## CLOSING SESSION 1 (5 min)

```
SESSION 1 SUMMARY:
  Eigenvector: direction a matrix only stretches, never rotates
  Eigenvalue: the stretch factor for that eigenvector
  For PCA: eigenvectors of the covariance matrix = principal components
  Eigenvalues = how much variance each component captures
```

---

# SESSION 2 (~90 min)
## "From eigenvectors to PCA — putting it all together"

## OPENING (5 min)

> *"Session 1: what eigenvectors are.*
> *Today: how they become PCA.*
> *By the end, you'll be able to derive PCA from scratch mathematically.*
> *Not many people can do that. You will."*

---

## SECTION 1: The Full PCA Pipeline in Math (30 min)

Write step by step:

```
PCA FROM SCRATCH — 5 STEPS:

Step 1: CENTER the data
  X_centered = X - mean(X)
  (subtract mean of each feature)

Step 2: COVARIANCE MATRIX
  C = (X_centered.T @ X_centered) / (n - 1)
  C is shape (n_features × n_features)

Step 3: EIGENDECOMPOSITION
  eigenvalues, eigenvectors = np.linalg.eigh(C)
  eigenvectors columns are the principal component directions

Step 4: SORT by eigenvalue (descending)
  idx = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[idx]
  eigenvectors = eigenvectors[:, idx]

Step 5: PROJECT data
  X_pca = X_centered @ eigenvectors[:, :k]
  (keep only top k eigenvectors)
```

> *"That's PCA. 5 steps. All numpy.*
> *Let's code it together."*

Live code:
```python
def pca_from_scratch(X, n_components=2):
    # Step 1: Center
    X_c = X - X.mean(axis=0)
    # Step 2: Covariance matrix
    C = (X_c.T @ X_c) / (len(X) - 1)
    # Step 3: Eigendecomposition
    vals, vecs = np.linalg.eigh(C)
    # Step 4: Sort descending
    idx = np.argsort(vals)[::-1]
    vecs = vecs[:, idx]
    vals = vals[idx]
    # Step 5: Project
    return X_c @ vecs[:, :n_components], vals
```

---

## SECTION 2: The Scree Plot and Explained Variance (20 min)

> *"The scree plot answers: how many components do I keep?"*

```
EXPLAINED VARIANCE RATIO:
  ratio[i] = eigenvalue[i] / sum(all eigenvalues)

Example (iris, 4 features):
  Component 1: 92.5% of variance
  Component 2:  5.3% of variance
  Component 3:  1.7% of variance
  Component 4:  0.5% of variance

"Elbow" in the scree plot → keep components before the elbow

RULES OF THUMB:
  → Keep components that explain 80-95% cumulative variance
  → Stop where the curve flattens ("elbow")
  → For visualization: always use 2 or 3 components
```

---

## SECTION 3: sklearn vs From Scratch (15 min)

> *"You've built PCA from scratch. Now the 2-line version:"*

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)

print(pca.explained_variance_ratio_)
# [0.925, 0.053]
```

> *"Same result. sklearn's PCA uses SVD (Singular Value Decomposition)*
> *instead of eigendecomposition — numerically more stable for high dimensions.*
> *But conceptually identical."*

---

## CLOSING SESSION 2 (10 min)

```
EIGENVECTORS FOR PCA — COMPLETE:
  Eigenvector: special direction, only stretched by a matrix
  Eigenvalue: stretch factor (= variance captured)
  PCA = find eigenvectors of covariance matrix
  Scree plot: pick the elbow

  What you can now do:
  → Reduce 100 features to 2 for visualization
  → Remove noise (low-variance components)
  → Deal with multicollinearity in regression
  → Speed up any downstream ML algorithm

  Next: K-Means from scratch — using these foundations to cluster.
```

---

## INSTRUCTOR TIPS

**"Do I need to understand the math to use PCA?"**
> *"To USE sklearn's PCA: no.*
> *To KNOW when it's appropriate, why it works, and debug when it fails: yes.*
> *This module gives you that understanding.*
> *The 5-step derivation makes you dangerous in any data science interview."*

**"What's SVD vs eigendecomposition?"**
> *"Both decompose a matrix into components.*
> *Eigendecomposition requires a square matrix — works for covariance matrix.*
> *SVD works for any matrix — so sklearn uses SVD for efficiency.*
> *The principal components you get are the same either way."*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening — matrix × vector       10 min
├── Visual intuition                 25 min
├── Computing with numpy             20 min
└── Close                            5 min  (+ 30 min buffer)

SESSION 2  (90 min)
├── Opening                          5 min
├── PCA pipeline in math            30 min
├── Scree plot + explained variance  20 min
├── sklearn comparison              15 min
└── Close                           10 min  (+ 10 min buffer)
```

---
*MLForBeginners · Part 3: Unsupervised Learning · Module 03*
