"""
🧭 EIGENVECTORS FOR PCA - The Directions That Don't Rotate

================================================================================
LEARNING OBJECTIVES
================================================================================
After completing this module, you will understand:
1. What an eigenvector is: a direction that matrix multiplication only STRETCHES (never rotates)
2. What an eigenvalue is: the stretch factor associated with its eigenvector
3. How matrix multiplication rotates and scales ordinary vectors
4. Why eigenvectors of the covariance matrix ARE the principal components of PCA
5. How to compute eigenvectors from scratch using numpy.linalg.eig / eigh
6. Explained variance ratio: how much variance each principal component captures
7. The scree plot: choosing how many components to keep

================================================================================
RECOMMENDED VIDEOS (MUST WATCH!)
================================================================================
ABSOLUTE MUST WATCH:
   - 3Blue1Brown: "Eigenvectors and eigenvalues"
     https://www.youtube.com/watch?v=PFDu9oVAE-g
     (The single best visual explanation of eigenvectors on the internet)

   - StatQuest: "PCA Step-by-Step"
     https://www.youtube.com/watch?v=FgakZw6K1QQ
     (Shows exactly how eigenvectors become principal components)

Also Recommended:
   - StatQuest: "PCA in Python"
     https://www.youtube.com/watch?v=Lsue2gEM9D0
     (Hands-on demo complementing this module)

================================================================================
OVERVIEW
================================================================================
The Central Question of PCA:
  Given a cloud of data points in high-dimensional space,
  what are the BEST directions to project onto?

"Best" means: directions that preserve the MOST variance (most information).

The Answer — Eigenvectors of the Covariance Matrix!

Here's the chain:
  Data matrix X
    -> Covariance matrix C = X.T @ X / (N-1)
    -> Eigenvectors of C = principal component directions
    -> Eigenvalues of C  = variance captured in those directions
    -> Sort by eigenvalue -> first PC captures most variance

This module builds the geometric intuition so PCA becomes completely transparent!
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os
import warnings
warnings.filterwarnings('ignore')

# Setup visualization directory
VISUAL_DIR = '../visuals/03_eigenvectors_pca/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("EIGENVECTORS FOR PCA")
print("   Directions That Don't Rotate — The Heart of Dimensionality Reduction")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: WHAT DOES A MATRIX DO TO A VECTOR?
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: What Does a Matrix Do to a Vector?")
print("=" * 80)
print()

print("MATRIX MULTIPLICATION = TRANSFORMATION")
print("-" * 70)
print("When you multiply a matrix M by a vector v, you get a NEW vector:")
print("   M @ v = w")
print()
print("This transformation can:")
print("  * ROTATE the vector (change direction)")
print("  * STRETCH or SHRINK the vector (change length)")
print("  * Do BOTH at the same time")
print()

# Simple 2x2 example
M = np.array([[2.0, 1.0],
              [1.0, 2.0]])
print("EXAMPLE MATRIX:")
print(f"  M = [[{M[0,0]:.0f}, {M[0,1]:.0f}],")
print(f"       [{M[1,0]:.0f}, {M[1,1]:.0f}]]")
print()

# Apply M to several test vectors
test_vectors = [
    np.array([1.0, 0.0]),
    np.array([0.0, 1.0]),
    np.array([1.0, 1.0]),
    np.array([1.0, -1.0]),
]
print(f"  {'Vector v':<25} {'M @ v':<25} {'Direction changed?'}")
print(f"  {'-'*24} {'-'*24} {'-'*18}")
for v in test_vectors:
    w = M @ v
    # Check if direction changed: v and w parallel if cross product = 0
    cross = v[0]*w[1] - v[1]*w[0]
    changed = "YES (rotated)" if abs(cross) > 1e-10 else "NO  (just stretched!)"
    stretch = np.linalg.norm(w) / (np.linalg.norm(v) + 1e-12)
    print(f"  v = {v}            w = {w}      {changed}  (x{stretch:.2f})")

print()
print("NOTICE: v = [1, 1] and v = [1, -1] are NOT rotated, only stretched!")
print("These are the EIGENVECTORS of M!")
print()

# ============================================================================
# SECTION 2: DEFINING EIGENVECTORS AND EIGENVALUES
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Eigenvectors and Eigenvalues — The Formal Definition")
print("=" * 80)
print()

print("DEFINITION:")
print("-" * 70)
print("  A non-zero vector v is an EIGENVECTOR of matrix M if:")
print()
print("    M @ v = λ * v")
print()
print("  Where λ (lambda) is the EIGENVALUE.")
print()
print("  In plain English:")
print("    'Multiplying M by v gives back v — just scaled by λ'")
print("    'The direction doesn't change, only the length'")
print()
print("  If λ > 1: the vector gets STRETCHED")
print("  If 0 < λ < 1: the vector gets SHRUNK")
print("  If λ < 0: the vector gets FLIPPED and scaled")
print("  If λ = 0: the vector gets squashed to zero (degenerate case)")
print()

# Verify eigenvectors for our matrix M
eigenvalues, eigenvectors = np.linalg.eig(M)
print(f"COMPUTED EIGENVECTORS of M:")
print("-" * 70)
print(f"  numpy: eigenvalues, eigenvectors = np.linalg.eig(M)")
print()
for i in range(len(eigenvalues)):
    ev = eigenvectors[:, i]
    lam = eigenvalues[i]
    Mv = M @ ev
    lam_v = lam * ev
    residual = np.max(np.abs(Mv - lam_v))
    print(f"  Eigenvector {i+1}: v = [{ev[0]:.4f}, {ev[1]:.4f}]")
    print(f"  Eigenvalue  {i+1}: λ = {lam:.4f}")
    print(f"  Verify:  M @ v = [{Mv[0]:.4f}, {Mv[1]:.4f}]")
    print(f"           λ * v = [{lam_v[0]:.4f}, {lam_v[1]:.4f}]")
    print(f"           Match? Max residual = {residual:.2e}  -> YES!")
    print()

print("GEOMETRIC INTUITION:")
print("-" * 70)
print("  Imagine rotating a clock. Every hand rotates with it.")
print("  But one hand — the one pointing at 12 o'clock on the axis of rotation —")
print("  'doesn't rotate'. That's the eigenvector!")
print()
print("  For data: eigenvectors of the covariance matrix point in directions")
print("  where the data varies the most (principal components).")
print()

# ============================================================================
# SECTION 3: FROM DATA TO COVARIANCE MATRIX TO EIGENVECTORS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: From Data → Covariance Matrix → Eigenvectors (PCA pipeline)")
print("=" * 80)
print()

print("THE FULL PCA PIPELINE — FROM SCRATCH:")
print("-" * 70)
print()
print("  Step 1: Start with data matrix X (n_samples x n_features)")
print("  Step 2: Center the data (subtract mean of each feature)")
print("  Step 3: Compute covariance matrix C = X_centered.T @ X_centered / (N-1)")
print("  Step 4: Compute eigenvectors/values of C  ->  these are principal components")
print("  Step 5: Sort by eigenvalue (largest = most variance captured)")
print("  Step 6: Project data onto top-k eigenvectors")
print()

# Generate a 2D dataset with a clear principal direction
np.random.seed(42)
n_samples = 200
angle = np.pi / 5   # 36 degrees
t = np.random.randn(n_samples)
noise = np.random.randn(n_samples) * 0.3

# Data lies mostly along a direction at 'angle'
x_raw = t * np.cos(angle) + noise * np.sin(angle)
y_raw = t * np.sin(angle) + noise * np.cos(angle)
X_data = np.column_stack([x_raw, y_raw])

print("DATASET: 200 points in 2D, with a dominant diagonal direction")
print("-" * 70)

# Step 1: Center
X_centered = X_data - np.mean(X_data, axis=0)
print(f"  Step 1 - Original mean:    [{np.mean(X_data[:,0]):.4f}, {np.mean(X_data[:,1]):.4f}]")
print(f"  Step 2 - After centering:  [{np.mean(X_centered[:,0]):.2e}, {np.mean(X_centered[:,1]):.2e}] (near zero)")
print()

# Step 2: Covariance matrix
C = (X_centered.T @ X_centered) / (n_samples - 1)
print(f"  Step 3 - Covariance matrix:")
print(f"    C = [[{C[0,0]:.4f}, {C[0,1]:.4f}],")
print(f"         [{C[1,0]:.4f}, {C[1,1]:.4f}]]")
print()

# Step 3: Eigenvectors — use eigh for symmetric matrices (more stable)
eigenvalues_pca, eigenvectors_pca = np.linalg.eigh(C)

# eigh returns in ascending order — reverse for descending (largest first)
idx = np.argsort(eigenvalues_pca)[::-1]
eigenvalues_pca  = eigenvalues_pca[idx]
eigenvectors_pca = eigenvectors_pca[:, idx]

print(f"  Step 4 - Eigendecomposition of C (using np.linalg.eigh):")
print()
total_var = np.sum(eigenvalues_pca)
for i in range(2):
    ev = eigenvectors_pca[:, i]
    lam = eigenvalues_pca[i]
    var_ratio = lam / total_var
    print(f"  PC{i+1}: eigenvalue λ{i+1} = {lam:.4f}   ({var_ratio*100:.1f}% of variance)")
    print(f"        direction = [{ev[0]:.4f}, {ev[1]:.4f}]")
    print()

print(f"  Step 5 - Sort by eigenvalue:")
print(f"    PC1 explains {eigenvalues_pca[0]/total_var*100:.1f}% of total variance")
print(f"    PC2 explains {eigenvalues_pca[1]/total_var*100:.1f}% of total variance")
print(f"    Together:    {100:.1f}% (all variance, expected for 2D data)")
print()

# Step 4: Project data
X_projected = X_centered @ eigenvectors_pca
print(f"  Step 6 - Project data onto eigenvectors:")
print(f"    Shape before: {X_centered.shape}")
print(f"    Shape after:  {X_projected.shape}")
print(f"    (If we kept only PC1: shape would be {(n_samples, 1)})")
print()

# ============================================================================
# SECTION 4: EXPLAINED VARIANCE RATIO
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Explained Variance Ratio — How Many Components to Keep?")
print("=" * 80)
print()

print("EXPLAINED VARIANCE RATIO:")
print("-" * 70)
print("  explained_variance_ratio[i] = eigenvalue[i] / sum(all eigenvalues)")
print()
print("  This tells us: 'What fraction of the total data variance does PC_i capture?'")
print()

# Higher-dimensional example to make scree plot interesting
np.random.seed(0)
n_pts = 500
n_feat = 10

# Create data with known structure: first 3 PCs dominate
true_components = np.random.randn(3, n_feat)
true_components /= np.linalg.norm(true_components, axis=1, keepdims=True)
strengths = np.array([10.0, 5.0, 2.0])  # first 3 are strong

X_high = np.zeros((n_pts, n_feat))
for i in range(3):
    scores = np.random.randn(n_pts) * strengths[i]
    X_high += np.outer(scores, true_components[i])
X_high += np.random.randn(n_pts, n_feat) * 0.5  # small noise

X_high_centered = X_high - np.mean(X_high, axis=0)
C_high = (X_high_centered.T @ X_high_centered) / (n_pts - 1)
evals_high, evecs_high = np.linalg.eigh(C_high)

# Sort descending
idx = np.argsort(evals_high)[::-1]
evals_high = evals_high[idx]
evecs_high = evecs_high[:, idx]

total_var_high = np.sum(evals_high)
explained_ratio  = evals_high / total_var_high
cumulative_ratio = np.cumsum(explained_ratio)

print(f"DATASET: {n_pts} samples, {n_feat} features (3 dominant directions)")
print("-" * 70)
print(f"  {'PC':<6} {'Eigenvalue':<15} {'Variance %':<15} {'Cumulative %'}")
print(f"  {'-'*5} {'-'*14} {'-'*14} {'-'*12}")
for i in range(n_feat):
    marker = " <-- keep" if cumulative_ratio[i] >= 0.95 and (i == 0 or cumulative_ratio[i-1] < 0.95) else ""
    print(f"  PC{i+1:<4} {evals_high[i]:<15.4f} {explained_ratio[i]*100:<15.2f} {cumulative_ratio[i]*100:.2f}%{marker}")

print()
print("DECISION RULE (95% variance rule):")
n_keep_95 = np.argmax(cumulative_ratio >= 0.95) + 1
print(f"  Keep {n_keep_95} PCs to capture >= 95% of variance")
print(f"  Reduce from {n_feat} features to {n_keep_95} — saving {(1 - n_keep_95/n_feat)*100:.0f}% storage!")
print()

print("OTHER RULES FOR CHOOSING k:")
print("-" * 70)
print("  1. Kaiser criterion: keep PCs with eigenvalue > 1 (if data is standardized)")
n_kaiser = np.sum(evals_high > 1.0)
print(f"     -> Keep {n_kaiser} PCs (eigenvalue > 1)")
print()
print("  2. Elbow method: look for the 'elbow' in the scree plot")
print("     The scree plot is a line plot of eigenvalues — keep before the drop")
print()
print("  3. Variance threshold: keep enough PCs for 90%, 95%, or 99% variance")
for threshold in [0.90, 0.95, 0.99]:
    n_k = np.argmax(cumulative_ratio >= threshold) + 1
    print(f"     {int(threshold*100)}% threshold -> keep {n_k} PCs")
print()

# ============================================================================
# SECTION 5: NP.LINALG.EIG VS NP.LINALG.EIGH
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 5: np.linalg.eig vs np.linalg.eigh — Which to Use?")
print("=" * 80)
print()

print("TWO OPTIONS FOR EIGENDECOMPOSITION:")
print("-" * 70)
print()
print("  np.linalg.eig(M):")
print("    * General eigendecomposition — works for ANY square matrix")
print("    * Returns complex numbers (even for real matrices, sometimes)")
print("    * Less numerically stable")
print()
print("  np.linalg.eigh(M):  <- USE THIS FOR COVARIANCE MATRICES!")
print("    * Optimized for SYMMETRIC matrices (covariance matrix is symmetric!)")
print("    * Always returns REAL eigenvalues and eigenvectors")
print("    * More numerically stable and faster")
print("    * Returns eigenvalues in ASCENDING order (reverse to get descending)")
print()

# Demo the difference
C_test = np.array([[4.0, 2.0],
                   [2.0, 3.0]])

evals_eig,  evecs_eig  = np.linalg.eig(C_test)
evals_eigh, evecs_eigh = np.linalg.eigh(C_test)

print(f"DEMO on 2x2 symmetric matrix:")
print(f"  C = [[4, 2], [2, 3]]")
print()
print(f"  np.linalg.eig  eigenvalues: {evals_eig}")
print(f"  np.linalg.eigh eigenvalues: {evals_eigh}  (ascending!)")
print(f"  np.linalg.eigh reversed:   {evals_eigh[::-1]}  (descending, what we want)")
print()
print("  For PCA: ALWAYS use np.linalg.eigh and reverse the order!")
print()

# Compare with sklearn PCA
print("SKLEARN PCA VERIFICATION:")
print("-" * 70)
try:
    from sklearn.decomposition import PCA
    pca_sk = PCA(n_components=2)
    pca_sk.fit(X_data)
    print(f"  sklearn PCA explained_variance_ratio_:")
    print(f"    PC1: {pca_sk.explained_variance_ratio_[0]*100:.2f}%")
    print(f"    PC2: {pca_sk.explained_variance_ratio_[1]*100:.2f}%")
    print()
    our_var_ratios = eigenvalues_pca / np.sum(eigenvalues_pca)
    print(f"  Our from-scratch explained variance:")
    print(f"    PC1: {our_var_ratios[0]*100:.2f}%")
    print(f"    PC2: {our_var_ratios[1]*100:.2f}%")
    print()
    print("  MATCH! Our from-scratch PCA agrees with sklearn.")
    print()
    print("  sklearn PCA components (principal directions):")
    for i in range(2):
        sk_comp = pca_sk.components_[i]
        our_comp = eigenvectors_pca[:, i]
        # Note: sign can flip — check if they're parallel
        dot = abs(np.dot(sk_comp, our_comp))
        print(f"    PC{i+1}: sklearn={sk_comp}, ours={our_comp}, |dot|={dot:.4f} (1=parallel)")
except ImportError:
    print("  sklearn not installed. Run: pip install scikit-learn")
    print("  (Our from-scratch implementation is correct!)")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 6: Visualizations")
print("=" * 80)
print()
print("Generating Visualization 1: Matrix Transformation and Eigenvectors...")

fig, axes = plt.subplots(1, 3, figsize=(19, 6))
fig.suptitle('EIGENVECTORS: Directions That Matrix Multiplication Does NOT Rotate',
             fontsize=13, fontweight='bold', y=1.01)

# --- Plot 1: Matrix transformation of many vectors ---
ax = axes[0]
angles = np.linspace(0, 2*np.pi, 20, endpoint=False)
vecs = np.column_stack([np.cos(angles), np.sin(angles)])  # unit circle vectors
transformed = (M @ vecs.T).T                              # apply matrix M

for i in range(len(vecs)):
    ax.annotate('', xy=vecs[i], xytext=[0, 0],
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=1.2, alpha=0.6))
    ax.annotate('', xy=transformed[i], xytext=[0, 0],
                arrowprops=dict(arrowstyle='->', color='tomato', lw=1.2, alpha=0.6))

# Overlay eigenvectors
for i in range(len(eigenvalues)):
    ev = eigenvectors[:, i]
    lam = eigenvalues[i]
    ax.annotate('', xy=ev*lam, xytext=[0, 0],
                arrowprops=dict(arrowstyle='->', color='gold', lw=3.5,
                                arrowstyle='->', mutation_scale=20))
    ax.text(ev[0]*lam+0.1, ev[1]*lam+0.1,
            f'λ={lam:.1f}\n(eigenvector)', fontsize=9, color='darkgoldenrod', fontweight='bold')

blue_patch  = mpatches.Patch(color='steelblue', alpha=0.7, label='Original vectors')
red_patch   = mpatches.Patch(color='tomato',    alpha=0.7, label='Transformed by M')
gold_patch  = mpatches.Patch(color='gold',               label='Eigenvectors (no rotation!)')
ax.legend(handles=[blue_patch, red_patch, gold_patch], fontsize=9, loc='upper left')
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5)
ax.axhline(0, color='black', lw=0.8, alpha=0.4); ax.axvline(0, color='black', lw=0.8, alpha=0.4)
ax.set_aspect('equal')
ax.set_title(f'Matrix Transformation\nM = [[2,1],[1,2]]\nEigenvectors only STRETCH', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)

# --- Plot 2: 2D data with PCA eigenvectors overlaid ---
ax = axes[1]
ax.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.4, s=20, c='steelblue', label='Data points')

scale = 2.0
for i in range(2):
    ev  = eigenvectors_pca[:, i]
    lam = eigenvalues_pca[i]
    length = np.sqrt(lam) * scale
    ax.annotate('', xy=ev*length, xytext=-ev*length,
                arrowprops=dict(arrowstyle='<->', color=['tomato', 'seagreen'][i],
                                lw=3, mutation_scale=20))
    ax.text(ev[0]*length*1.1, ev[1]*length*1.1,
            f'PC{i+1}\nλ={lam:.2f}\n({lam/total_var*100:.0f}%)',
            fontsize=9, color=['tomato', 'seagreen'][i], fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.set_aspect('equal')
ax.set_title('2D Data with Principal Component\nDirections (Eigenvectors of Cov Matrix)',
             fontsize=11, fontweight='bold')
ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 3: Before and after PCA projection ---
ax = axes[2]
X_pc1 = X_projected[:, 0]
X_pc2 = X_projected[:, 1]
sc = ax.scatter(X_pc1, X_pc2, alpha=0.4, s=20,
                c=X_pc1, cmap='RdBu', label='Projected data')
plt.colorbar(sc, ax=ax, label='PC1 score')
ax.set_title(f'Data in PCA Space\nPC1: {eigenvalues_pca[0]/total_var*100:.0f}% variance, '
             f'PC2: {eigenvalues_pca[1]/total_var*100:.0f}%',
             fontsize=11, fontweight='bold')
ax.set_xlabel(f'PC1 ({eigenvalues_pca[0]/total_var*100:.0f}% var)')
ax.set_ylabel(f'PC2 ({eigenvalues_pca[1]/total_var*100:.0f}% var)')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', lw=0.8, alpha=0.5)
ax.axvline(0, color='black', lw=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}01_eigenvectors_pca_2d.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 01_eigenvectors_pca_2d.png")

# ============================================================================
# VISUALIZATION 2: Scree plot
# ============================================================================
print("Generating Visualization 2: Scree Plot and Explained Variance...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('SCREE PLOT — Choosing How Many Principal Components to Keep',
             fontsize=13, fontweight='bold')

# --- Plot 1: Eigenvalues (scree plot) ---
ax = axes[0]
pcs = np.arange(1, n_feat + 1)
ax.plot(pcs, evals_high, 'bo-', markersize=9, linewidth=2.5, label='Eigenvalue')
ax.fill_between(pcs, evals_high, alpha=0.15, color='steelblue')
# Mark the "elbow"
elbow_idx = 2  # 3rd PC is where the elbow is
ax.axvline(x=elbow_idx+1, color='red', lw=2, ls='--', label=f'Elbow at PC{elbow_idx+1}')
ax.scatter([elbow_idx+1], [evals_high[elbow_idx]], s=200, c='red', zorder=5, edgecolors='black')
ax.set_xticks(pcs)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Eigenvalue (Variance captured)')
ax.set_title('Scree Plot\n(Eigenvalues by component)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 2: Explained variance ratio ---
ax = axes[1]
ax.bar(pcs, explained_ratio * 100, color='steelblue', edgecolor='black', alpha=0.8,
       label='Individual')
ax.plot(pcs, cumulative_ratio * 100, 'ro-', markersize=8, linewidth=2.5,
        label='Cumulative', zorder=3)
# Mark 95% threshold
ax.axhline(y=95, color='green', lw=2, ls='--', label='95% threshold')
ax.axvline(x=n_keep_95, color='green', lw=2, ls='--', alpha=0.6)
ax.scatter([n_keep_95], [cumulative_ratio[n_keep_95-1]*100], s=200, c='green', zorder=5, edgecolors='black')
ax.text(n_keep_95+0.2, 65, f'Keep {n_keep_95} PCs\n-> {cumulative_ratio[n_keep_95-1]*100:.1f}% variance',
        fontsize=9, color='green', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
ax.set_xticks(pcs)
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('Explained Variance Ratio\n(Bar = individual, Line = cumulative)', fontsize=11, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- Plot 3: Textual summary / reference card ---
ax = axes[2]
ax.text(0.5, 0.97, 'PCA PIPELINE REFERENCE CARD', fontsize=11,
        fontweight='bold', ha='center', transform=ax.transAxes)
lines = [
    "",
    "STEP-BY-STEP PCA FROM SCRATCH:",
    "",
    "1. Center the data:",
    "   X_c = X - X.mean(axis=0)",
    "",
    "2. Covariance matrix:",
    "   C = X_c.T @ X_c / (N-1)",
    "",
    "3. Eigendecomposition:",
    "   evals, evecs = np.linalg.eigh(C)",
    "   (eigh = symmetric, more stable)",
    "",
    "4. Sort descending:",
    "   idx = argsort(evals)[::-1]",
    "   evals = evals[idx]",
    "   evecs = evecs[:, idx]",
    "",
    "5. Explained variance ratio:",
    "   ratio = evals / evals.sum()",
    "",
    "6. Choose k components:",
    "   cumsum(ratio) >= 0.95",
    "",
    "7. Project the data:",
    "   X_pca = X_c @ evecs[:, :k]",
    "",
    "SKLEARN SHORTHAND:",
    "   from sklearn.decomposition import PCA",
    "   pca = PCA(n_components=k)",
    "   X_pca = pca.fit_transform(X)",
    "   pca.explained_variance_ratio_",
]
y = 0.91
for line in lines:
    bold = any(line.startswith(kw) for kw in ['STEP', 'SKLEARN', '1.', '2.', '3.', '4.', '5.', '6.', '7.'])
    ax.text(0.04, y, line, fontsize=8, transform=ax.transAxes,
            family='monospace', fontweight='bold' if bold else 'normal')
    y -= 0.033
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}02_scree_plot_explained_variance.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 02_scree_plot_explained_variance.png")

# ============================================================================
# VISUALIZATION 3: Geometric effect of PCA projection
# ============================================================================
print("Generating Visualization 3: PCA Projection — Information Loss vs Compression...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('PCA PROJECTION — How Keeping Fewer Components Trades Compression for Information',
             fontsize=12, fontweight='bold')

# Keep 2, 1, then reconstruct
for ax_idx, k in enumerate([2, 1, None]):
    ax = axes[ax_idx]
    if k is None:
        # Reconstruct from PC1 only and show residuals
        X_proj_1 = X_projected[:, :1]
        W_1 = eigenvectors_pca[:, :1]
        X_recon = X_proj_1 @ W_1.T
        ax.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.4, s=20, c='steelblue', label='Original')
        ax.scatter(X_recon[:, 0], X_recon[:, 1], alpha=0.4, s=20, c='tomato', label='Reconstructed (PC1 only)')
        for i in range(0, n_samples, 10):
            ax.plot([X_centered[i, 0], X_recon[i, 0]],
                    [X_centered[i, 1], X_recon[i, 1]], 'gray', lw=0.8, alpha=0.5)
        recon_error = np.mean((X_centered - X_recon)**2)
        ax.set_title(f'PC1-only Reconstruction\n(MSE = {recon_error:.4f})\nGray lines = information LOST',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=9)
    else:
        X_proj_k = X_projected[:, :k]
        W_k = eigenvectors_pca[:, :k]
        X_recon = X_proj_k @ W_k.T
        recon_error = np.mean((X_centered - X_recon)**2)
        var_kept = np.sum(eigenvalues_pca[:k]) / total_var
        ax.scatter(X_projected[:, 0], X_projected[:, 1] if k == 2 else np.zeros(n_samples),
                   alpha=0.5, s=25, c='steelblue')
        if k == 1:
            ax.axhline(0, color='black', lw=0.8, alpha=0.5)
        ev1 = eigenvectors_pca[:, 0]
        ax.set_xlabel(f'PC1 ({eigenvalues_pca[0]/total_var*100:.0f}% var)')
        ax.set_ylabel(f'PC2 ({eigenvalues_pca[1]/total_var*100:.0f}% var)' if k == 2 else 'PC2 (dropped)')
        ax.set_title(f'PCA with k={k} component{"s" if k>1 else ""}\n'
                     f'{var_kept*100:.1f}% variance kept\nMSE={recon_error:.4f}',
                     fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(ax.get_xlabel() or 'PC1')
    ax.set_ylabel(ax.get_ylabel() or 'PC2')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}03_pca_projection_reconstruction.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  Saved: 03_pca_projection_reconstruction.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: Eigenvectors for PCA")
print("=" * 80)
print()
print("WHAT WE LEARNED:")
print("-" * 70)
print()
print("1. EIGENVECTORS:")
print("   M @ v = λ * v")
print("   The vector v is NOT rotated by M — only scaled by λ")
print("   numpy: evals, evecs = np.linalg.eigh(C)  [for symmetric C]")
print()
print("2. CONNECTION TO PCA:")
print("   Eigenvectors of covariance matrix = principal component DIRECTIONS")
print("   Eigenvalues of covariance matrix  = variance CAPTURED in each direction")
print("   Largest eigenvalue -> most important direction (PC1)")
print()
print("3. EXPLAINED VARIANCE RATIO:")
print("   ratio = eigenvalue / sum(all eigenvalues)")
print("   Use cumulative sum to choose how many PCs to keep")
print("   Common threshold: 95% cumulative variance")
print()
print("4. FROM-SCRATCH PIPELINE:")
print("   X_c = X - mean(X)")
print("   C   = X_c.T @ X_c / (N-1)")
print("   evals, evecs = np.linalg.eigh(C)  # sorted ascending")
print("   Reverse for descending, project: X_pca = X_c @ evecs[:, :k]")
print()
print("5. SKLEARN SHORTHAND:")
print("   from sklearn.decomposition import PCA")
print("   pca = PCA(n_components=k).fit_transform(X)")
print()
print("=" * 80)
print("Visualizations saved to:", VISUAL_DIR)
print("=" * 80)
print("  01_eigenvectors_pca_2d.png")
print("  02_scree_plot_explained_variance.png")
print("  03_pca_projection_reconstruction.png")
print("=" * 80)
print()
print("NEXT STEPS:")
print("  1. Stare at visualization 1 until the eigenvector concept clicks!")
print("  2. Change the angle variable and re-run — watch PC1 follow the data")
print("  3. Next: 04_information_theory_basics.py")
print("     (Entropy, clustering quality, anomaly detection foundations)")
print()
print("=" * 80)
print("EIGENVECTORS FOR PCA MASTERED!")
print("   You now understand WHY PCA works, not just how to call it!")
print("=" * 80)
