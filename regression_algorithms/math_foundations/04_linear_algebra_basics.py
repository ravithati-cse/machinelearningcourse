"""
📐 LINEAR ALGEBRA BASICS - The Language of Machine Learning

================================================================================
LEARNING OBJECTIVES
================================================================================
After completing this module, you will understand:
1. What vectors are and how to work with them
2. What matrices are and why they're useful for data
3. Dot products and what they mean geometrically
4. Matrix multiplication basics
5. Matrix transpose operation
6. Why linear algebra is essential for machine learning

================================================================================
📺 RECOMMENDED VIDEOS (WATCH THESE!)
================================================================================
⭐ MUST WATCH (The best visual explanations ever created):
   - 3Blue1Brown: "Essence of Linear Algebra" - Full Series
     https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab

     Key chapters:
     • Chapter 1: "Vectors, what even are they?"
       https://www.youtube.com/watch?v=fNk_zzaMoSs
     • Chapter 3: "Matrix multiplication as composition"
       https://www.youtube.com/watch?v=XkY2DOUCWMU
     • Chapter 9: "Dot products and duality"
       https://www.youtube.com/watch?v=LyGKycYT2v0

Also Recommended:
   - Khan Academy: "Introduction to Matrices"
     https://www.youtube.com/watch?v=0oGJTQCy4cQ

   - StatQuest: "Matrix Algebra for Deep Learning"
     https://www.youtube.com/watch?v=kYB8IZa5AuE

================================================================================
OVERVIEW
================================================================================
Linear algebra is the mathematical language of machine learning!

Think about it:
- Your dataset is a MATRIX (rows = samples, columns = features)
- Each data point is a VECTOR
- Predictions involve DOT PRODUCTS
- Neural networks are just matrix multiplications!

This module makes linear algebra VISUAL and INTUITIVE.
No memorizing formulas - we'll see what everything means geometrically!
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
warnings.filterwarnings('ignore')

# Setup visualization directory
VISUAL_DIR = '../visuals/04_linear_algebra/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("📐 LINEAR ALGEBRA BASICS")
print("   Vectors, Matrices, and the Math Behind ML")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: VECTORS - ORDERED LISTS OF NUMBERS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Vectors - Arrows in Space")
print("=" * 80)
print()

print("WHAT IS A VECTOR?")
print("-" * 70)
print("A vector is simply an ordered list of numbers!")
print()
print("Two ways to think about vectors:")
print("  1. As a list: [3, 2] - just two numbers in order")
print("  2. As an arrow: pointing from origin to the point (3, 2)")
print()

print("EXAMPLE VECTORS:")
print("-" * 70)

# 2D vectors
v1 = np.array([3, 2])
v2 = np.array([1, 4])
v3 = np.array([-2, 3])

print("2D Vectors (two numbers):")
print(f"  v1 = {v1}  → Point at (3, 2)")
print(f"  v2 = {v2}  → Point at (1, 4)")
print(f"  v3 = {v3} → Point at (-2, 3)")
print()

# 3D vectors
v4 = np.array([1, 2, 3])
v5 = np.array([2, -1, 4])

print("3D Vectors (three numbers):")
print(f"  v4 = {v4}  → Point at (1, 2, 3)")
print(f"  v5 = {v5} → Point at (2, -1, 4)")
print()

print("VECTOR OPERATIONS:")
print("-" * 70)

# Vector addition
print("1. VECTOR ADDITION: Add corresponding elements")
print(f"   v1 + v2 = {v1} + {v2}")
print(f"           = [{v1[0]} + {v2[0]}, {v1[1]} + {v2[1]}]")
print(f"           = {v1 + v2}")
print(f"   Geometric meaning: Place vectors tip-to-tail")
print()

# Scalar multiplication
scalar = 2
print("2. SCALAR MULTIPLICATION: Multiply each element by a number")
print(f"   {scalar} × v1 = {scalar} × {v1}")
print(f"          = [{scalar} × {v1[0]}, {scalar} × {v1[1]}]")
print(f"          = {scalar * v1}")
print(f"   Geometric meaning: Stretch or shrink the vector")
print()

# Vector magnitude (length)
print("3. MAGNITUDE (Length): How long is the vector?")
magnitude_v1 = np.linalg.norm(v1)
print(f"   ||v1|| = √(3² + 2²) = √(9 + 4) = √13 ≈ {magnitude_v1:.2f}")
print(f"   Formula: ||v|| = √(v₁² + v₂² + ... + vₙ²)")
print(f"   It's the Pythagorean theorem!")
print()

print("WHY VECTORS MATTER FOR ML:")
print("-" * 70)
print("  • Each data point is a vector!")
print("    Example: [house_size=1500, bedrooms=3, age=10] is a 3D vector")
print("  • Features are components of the vector")
print("  • Dataset = collection of vectors")
print()

# ============================================================================
# VISUALIZATION 1: Understanding Vectors
# ============================================================================
print("📊 Generating Visualization 1: Vectors...")

fig = plt.figure(figsize=(16, 10))
fig.suptitle('📐 UNDERSTANDING VECTORS: Arrows in Space', fontsize=16, fontweight='bold')

# Plot 1: 2D vectors as arrows
ax1 = plt.subplot(2, 3, 1)
origin = [0, 0]

# Draw vectors
ax1.quiver(*origin, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
           color='red', width=0.01, label='v1 = [3, 2]')
ax1.quiver(*origin, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.01, label='v2 = [1, 4]')
ax1.quiver(*origin, v3[0], v3[1], angles='xy', scale_units='xy', scale=1,
           color='green', width=0.01, label='v3 = [-2, 3]')

# Annotate endpoints
ax1.scatter([v1[0], v2[0], v3[0]], [v1[1], v2[1], v3[1]], s=80, zorder=5)
ax1.text(v1[0]+0.2, v1[1]+0.2, f'({v1[0]}, {v1[1]})', fontsize=9)
ax1.text(v2[0]+0.2, v2[1]+0.2, f'({v2[0]}, {v2[1]})', fontsize=9)
ax1.text(v3[0]-0.5, v3[1]+0.2, f'({v3[0]}, {v3[1]})', fontsize=9)

ax1.set_xlim(-3, 5)
ax1.set_ylim(-1, 5)
ax1.set_xlabel('x', fontsize=10)
ax1.set_ylabel('y', fontsize=10)
ax1.set_title('Vectors as Arrows', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)
ax1.set_aspect('equal')

# Plot 2: Vector addition
ax2 = plt.subplot(2, 3, 2)
v_sum = v1 + v2

# Draw individual vectors
ax2.quiver(*origin, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
           color='red', width=0.01, alpha=0.6, label='v1 = [3, 2]')
ax2.quiver(*origin, v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.01, alpha=0.6, label='v2 = [1, 4]')

# Draw v2 starting from tip of v1 (tip-to-tail)
ax2.quiver(v1[0], v1[1], v2[0], v2[1], angles='xy', scale_units='xy', scale=1,
           color='blue', width=0.01, alpha=0.3, linestyle='--')

# Draw sum
ax2.quiver(*origin, v_sum[0], v_sum[1], angles='xy', scale_units='xy', scale=1,
           color='purple', width=0.015, label=f'v1+v2 = {v_sum}')

ax2.scatter([v_sum[0]], [v_sum[1]], s=100, color='purple', zorder=5, marker='*')
ax2.text(v_sum[0]+0.3, v_sum[1], f'({v_sum[0]}, {v_sum[1]})', fontsize=9, fontweight='bold')

ax2.set_xlim(-1, 6)
ax2.set_ylim(-1, 7)
ax2.set_xlabel('x', fontsize=10)
ax2.set_ylabel('y', fontsize=10)
ax2.set_title('Vector Addition (Tip-to-Tail)', fontsize=11, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='k', linewidth=0.5)
ax2.axvline(x=0, color='k', linewidth=0.5)
ax2.set_aspect('equal')

# Plot 3: Scalar multiplication
ax3 = plt.subplot(2, 3, 3)

scalars = [0.5, 1, 2, 3]
colors = ['lightblue', 'blue', 'darkblue', 'purple']

for s, c in zip(scalars, colors):
    scaled = s * v1
    ax3.quiver(*origin, scaled[0], scaled[1], angles='xy', scale_units='xy', scale=1,
               color=c, width=0.008, label=f'{s}×v1 = {scaled}')

ax3.set_xlim(-1, 10)
ax3.set_ylim(-1, 7)
ax3.set_xlabel('x', fontsize=10)
ax3.set_ylabel('y', fontsize=10)
ax3.set_title('Scalar Multiplication (Scaling)', fontsize=11, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='k', linewidth=0.5)
ax3.axvline(x=0, color='k', linewidth=0.5)
ax3.set_aspect('equal')

# Plot 4: Vector magnitude
ax4 = plt.subplot(2, 3, 4)

ax4.quiver(*origin, v1[0], v1[1], angles='xy', scale_units='xy', scale=1,
           color='red', width=0.015, label=f'v1 = {v1}')

# Draw magnitude as dashed line from origin to tip
ax4.plot([0, v1[0]], [0, v1[1]], 'k--', linewidth=2, label=f'||v1|| = {magnitude_v1:.2f}')

# Draw right triangle to show Pythagorean theorem
ax4.plot([0, v1[0]], [0, 0], 'g--', linewidth=1, alpha=0.6)
ax4.plot([v1[0], v1[0]], [0, v1[1]], 'b--', linewidth=1, alpha=0.6)

ax4.text(v1[0]/2, -0.3, f'{v1[0]}', fontsize=10, ha='center', color='green')
ax4.text(v1[0]+0.3, v1[1]/2, f'{v1[1]}', fontsize=10, color='blue')
ax4.text(v1[0]/2 - 0.5, v1[1]/2 + 0.3, f'√({v1[0]}² + {v1[1]}²) = {magnitude_v1:.2f}',
         fontsize=9, fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax4.set_xlim(-0.5, 4)
ax4.set_ylim(-0.5, 3)
ax4.set_xlabel('x', fontsize=10)
ax4.set_ylabel('y', fontsize=10)
ax4.set_title('Vector Magnitude (Length)', fontsize=11, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='k', linewidth=0.5)
ax4.axvline(x=0, color='k', linewidth=0.5)
ax4.set_aspect('equal')

# Plot 5: 3D vector visualization
ax5 = plt.subplot(2, 3, 5, projection='3d')

# Draw 3D vectors
ax5.quiver(0, 0, 0, v4[0], v4[1], v4[2], color='red', arrow_length_ratio=0.15,
           linewidth=2, label=f'v4 = {v4}')
ax5.quiver(0, 0, 0, v5[0], v5[1], v5[2], color='blue', arrow_length_ratio=0.15,
           linewidth=2, label=f'v5 = {v5}')

ax5.set_xlabel('x', fontsize=9)
ax5.set_ylabel('y', fontsize=9)
ax5.set_zlabel('z', fontsize=9)
ax5.set_title('3D Vectors', fontsize=11, fontweight='bold')
ax5.legend()
ax5.set_xlim(0, 3)
ax5.set_ylim(-2, 3)
ax5.set_zlim(0, 5)

# Plot 6: Key concepts
ax6 = plt.subplot(2, 3, 6)
ax6.text(0.5, 0.95, 'VECTOR KEY CONCEPTS', fontsize=12, fontweight='bold',
         ha='center', transform=ax6.transAxes)

concepts = [
    "📊 What: Ordered list of numbers",
    "",
    "📈 Notation: v = [v₁, v₂, ..., vₙ]",
    "",
    "➕ Addition:",
    "   [a, b] + [c, d] = [a+c, b+d]",
    "",
    "✖️  Scalar multiplication:",
    "   k × [a, b] = [ka, kb]",
    "",
    "📏 Magnitude (length):",
    "   ||v|| = √(v₁² + v₂² + ... + vₙ²)",
    "",
    "🎯 ML Connection:",
    "   • Data point = vector",
    "   • Features = components",
    "   • [size, bedrooms, age] is a vector!"
]

y_pos = 0.85
for line in concepts:
    if line.startswith(('📊', '📈', '➕', '✖️', '📏', '🎯')):
        weight = 'bold'
        size = 10
    else:
        weight = 'normal'
        size = 9
    ax6.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax6.transAxes,
             family='monospace', fontweight=weight)
    y_pos -= 0.049

ax6.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}01_vectors.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 01_vectors.png")
print()

# ============================================================================
# SECTION 2: MATRICES - TABLES OF NUMBERS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Matrices - Organizing Data")
print("=" * 80)
print()

print("WHAT IS A MATRIX?")
print("-" * 70)
print("A matrix is a rectangular table of numbers!")
print("Think of it like a spreadsheet or a table in Excel.")
print()

print("EXAMPLE MATRIX:")
print("-" * 70)

# Create a sample matrix
A = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print("Matrix A:")
print(A)
print()
print(f"Shape: {A.shape[0]} rows × {A.shape[1]} columns")
print(f"      (also written as {A.shape[0]}×{A.shape[1]} matrix)")
print()

print("MATRIX NOTATION:")
print("-" * 70)
print("  • A[i, j] = element in row i, column j")
print("  • Row indexing starts at 0 in Python (1 in math textbooks)")
print()
print("Examples from matrix A:")
print(f"  A[0, 0] = {A[0, 0]}  (first row, first column)")
print(f"  A[0, 2] = {A[0, 2]}  (first row, third column)")
print(f"  A[2, 1] = {A[2, 1]}  (third row, second column)")
print()

print("REAL-WORLD EXAMPLE: Housing Dataset as a Matrix")
print("-" * 70)

# Create a sample dataset
house_data = np.array([
    [1500, 3, 10],  # House 1: 1500 sqft, 3 bedrooms, 10 years old
    [2000, 4, 5],   # House 2: 2000 sqft, 4 bedrooms, 5 years old
    [1200, 2, 15],  # House 3: 1200 sqft, 2 bedrooms, 15 years old
    [1800, 3, 8]    # House 4: 1800 sqft, 3 bedrooms, 8 years old
])

print("Each row = one house (data point, sample)")
print("Each column = one feature (attribute)")
print()
print("House Dataset Matrix:")
print("       Size  Bed  Age")
print("     ┌──────────────────┐")
for i, house in enumerate(house_data, 1):
    print(f"  H{i} │ {house[0]:5.0f}  {house[1]}   {house[2]:3.0f}  │")
print("     └──────────────────┘")
print()
print(f"Shape: {house_data.shape[0]} samples × {house_data.shape[1]} features")
print()

print("MATRIX OPERATIONS:")
print("-" * 70)

# Matrix addition
B = np.array([
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3]
])

print("1. MATRIX ADDITION (element-wise):")
print("   A + B:")
print(A)
print("     +")
print(B)
print("     =")
print(A + B)
print()

# Scalar multiplication
print("2. SCALAR MULTIPLICATION:")
scalar = 2
print(f"   {scalar} × A:")
print(A)
print(f"     × {scalar} = ")
print(scalar * A)
print()

# ============================================================================
# VISUALIZATION 2: Understanding Matrices
# ============================================================================
print("📊 Generating Visualization 2: Matrices...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('📊 UNDERSTANDING MATRICES: Tables of Data', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Matrix structure diagram
ax = axes[0, 0]
matrix_display = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# Create color map
cmap = plt.cm.Blues
norm = plt.Normalize(vmin=matrix_display.min(), vmax=matrix_display.max())

# Display matrix as heatmap
im = ax.imshow(matrix_display, cmap=cmap, norm=norm, aspect='auto')

# Add values
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, matrix_display[i, j],
                      ha="center", va="center", color="black", fontsize=16, fontweight='bold')

# Labels
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Col 0', 'Col 1', 'Col 2'], fontsize=10)
ax.set_yticklabels(['Row 0', 'Row 1', 'Row 2'], fontsize=10)
ax.set_title('Matrix Structure (3×3)', fontsize=12, fontweight='bold')

# Annotations
ax.text(3.5, 0, '← Row 0', fontsize=9, va='center')
ax.text(3.5, 1, '← Row 1', fontsize=9, va='center')
ax.text(3.5, 2, '← Row 2', fontsize=9, va='center')

ax.text(0, 3.3, '↑\nCol 0', fontsize=8, ha='center')
ax.text(1, 3.3, '↑\nCol 1', fontsize=8, ha='center')
ax.text(2, 3.3, '↑\nCol 2', fontsize=8, ha='center')

# Plot 2: Dataset as matrix
ax = axes[0, 1]

# Display house dataset
im = ax.imshow(house_data, cmap='YlOrRd', aspect='auto')

# Add values
for i in range(house_data.shape[0]):
    for j in range(house_data.shape[1]):
        text = ax.text(j, i, int(house_data[i, j]),
                      ha="center", va="center", color="black", fontsize=11, fontweight='bold')

ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2, 3])
ax.set_xticklabels(['Size (sqft)', 'Bedrooms', 'Age (years)'], fontsize=9)
ax.set_yticklabels(['House 1', 'House 2', 'House 3', 'House 4'], fontsize=9)
ax.set_title('ML Dataset as Matrix\n(4 samples × 3 features)', fontsize=11, fontweight='bold')

ax.text(3.7, 1.5, '← Rows = Samples\n   (data points)', fontsize=9, va='center')
ax.text(1, 4.2, '↑\nColumns = Features\n(attributes)', fontsize=8, ha='center')

# Plot 3: Matrix shapes visualization
ax = axes[1, 0]

shapes = [
    ("Vector\n(1D)", np.array([[1, 2, 3]]).T, 0.5),
    ("Row Vector\n(1×3)", np.array([[1, 2, 3]]), 2),
    ("3×3\nMatrix", np.array([[1,2,3],[4,5,6],[7,8,9]]), 4.5)
]

for name, mat, x_pos in shapes:
    # Display small matrix
    rows, cols = mat.shape
    cell_size = 0.3

    for i in range(rows):
        for j in range(cols):
            rect = plt.Rectangle((x_pos + j*cell_size, 2 - i*cell_size),
                                cell_size, cell_size,
                                facecolor='lightblue', edgecolor='black', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x_pos + j*cell_size + cell_size/2,
                   2 - i*cell_size + cell_size/2,
                   str(mat[i, j]), ha='center', va='center', fontsize=9)

    ax.text(x_pos + cols*cell_size/2, 2.8, name, ha='center', fontsize=10, fontweight='bold')
    ax.text(x_pos + cols*cell_size/2, 1.3, f'{rows}×{cols}', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

ax.set_xlim(0, 6)
ax.set_ylim(0, 3.5)
ax.set_title('Different Matrix Shapes', fontsize=12, fontweight='bold')
ax.axis('off')

# Plot 4: Key concepts
ax = axes[1, 1]
ax.text(0.5, 0.95, 'MATRIX KEY CONCEPTS', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

concepts = [
    "📊 What: Rectangular table of numbers",
    "",
    "📐 Shape: m × n",
    "   • m = number of rows",
    "   • n = number of columns",
    "",
    "📍 Element: A[i, j]",
    "   • i = row index",
    "   • j = column index",
    "",
    "➕ Addition: Same shape matrices only",
    "   Add corresponding elements",
    "",
    "✖️  Scalar mult: Multiply each element",
    "",
    "🎯 ML Connection:",
    "   • Rows = Data samples",
    "   • Columns = Features",
    "   • Entire dataset = One big matrix!",
    "",
    "Example: 1000 houses, 10 features",
    "         → 1000×10 matrix"
]

y_pos = 0.85
for line in concepts:
    if line.startswith(('📊', '📐', '📍', '➕', '✖️', '🎯', 'Example')):
        weight = 'bold'
        size = 9.5
    else:
        weight = 'normal'
        size = 8.5
    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.042

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}02_matrices.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 02_matrices.png")
print()

# ============================================================================
# SECTION 3: DOT PRODUCT - THE HEART OF ML
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Dot Product - The Most Important Operation")
print("=" * 80)
print()

print("WHAT IS A DOT PRODUCT?")
print("-" * 70)
print("The dot product combines two vectors into a single number.")
print("It's used EVERYWHERE in machine learning!")
print()

print("FORMULA:")
print("-" * 70)
print("For vectors a = [a₁, a₂, a₃] and b = [b₁, b₂, b₃]:")
print()
print("  a · b = a₁×b₁ + a₂×b₂ + a₃×b₃")
print()
print("Steps:")
print("  1. Multiply corresponding elements")
print("  2. Add all the products together")
print("  3. Result is a single number!")
print()

print("EXAMPLE 1: Computing a dot product")
print("-" * 70)

a = np.array([2, 3, 4])
b = np.array([1, 0, 2])

print(f"a = {a}")
print(f"b = {b}")
print()
print("Step-by-step:")
print(f"  a · b = (2×1) + (3×0) + (4×2)")
print(f"        = {2*1} + {3*0} + {4*2}")
print(f"        = {2*1 + 3*0 + 4*2}")
print()

# Using NumPy
dot_product = np.dot(a, b)
print(f"Using NumPy: np.dot(a, b) = {dot_product}")
print()

print("EXAMPLE 2: Making a prediction with dot product!")
print("-" * 70)
print("This is how linear regression makes predictions!")
print()

# House features
house = np.array([1500, 3, 10])  # size, bedrooms, age
print(f"House features: {house}")
print("  [size=1500 sqft, bedrooms=3, age=10 years]")
print()

# Model weights (coefficients)
weights = np.array([150, 10000, -500])  # price per sqft, per bedroom, per year
print(f"Model weights: {weights}")
print("  [150 $/sqft, 10000 $/bedroom, -500 $/year]")
print()

# Prediction = dot product!
prediction = np.dot(house, weights)

print("Prediction = dot product:")
print(f"  house · weights = (1500×150) + (3×10000) + (10×-500)")
print(f"                  = {1500*150} + {3*10000} + {10*-500}")
print(f"                  = ${prediction:,}")
print()
print("✅ Predicted price: ${:,}".format(prediction))
print()

print("GEOMETRIC MEANING:")
print("-" * 70)
print("The dot product measures how ALIGNED two vectors are:")
print()
print("  • Large positive: vectors point in similar direction")
print("  • Zero: vectors are perpendicular (90°)")
print("  • Large negative: vectors point in opposite directions")
print()

# Show examples
print("Examples:")
v1 = np.array([1, 0])
v2 = np.array([1, 0])
v3 = np.array([0, 1])
v4 = np.array([-1, 0])

print(f"  v1 · v2 = {np.dot(v1, v2):>3}  (parallel, same direction)")
print(f"  v1 · v3 = {np.dot(v1, v3):>3}  (perpendicular)")
print(f"  v1 · v4 = {np.dot(v1, v4):>3}  (parallel, opposite direction)")
print()

# ============================================================================
# VISUALIZATION 3: Dot Product
# ============================================================================
print("📊 Generating Visualization 3: Dot Product...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('⭐ DOT PRODUCT: The Heart of Machine Learning', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Dot product calculation
ax = axes[0, 0]

a_example = np.array([3, 2])
b_example = np.array([1, 4])

# Visualization of calculation
ax.text(0.5, 0.9, 'DOT PRODUCT CALCULATION', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

calc_text = [
    f"a = {a_example}",
    f"b = {b_example}",
    "",
    "a · b = a₁×b₁ + a₂×b₂",
    "",
    f"     = {a_example[0]}×{b_example[0]} + {a_example[1]}×{b_example[1]}",
    "",
    f"     = {a_example[0]*b_example[0]} + {a_example[1]*b_example[1]}",
    "",
    f"     = {np.dot(a_example, b_example)}",
    "",
    "✅ Result: Single number!"
]

y_pos = 0.75
for line in calc_text:
    if '=' in line and '·' in line:
        weight = 'bold'
        size = 11
    elif line.startswith('✅'):
        weight = 'bold'
        size = 11
        color = 'green'
    else:
        weight = 'normal'
        size = 10
        color = 'black'

    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight, color=color if 'color' in locals() else 'black')
    y_pos -= 0.06

ax.axis('off')

# Plot 2: Geometric interpretation
ax = axes[0, 1]

# Different angle pairs
test_vecs = [
    (np.array([2, 0]), np.array([1, 0]), 'Same direction'),
    (np.array([2, 0]), np.array([0, 1]), 'Perpendicular'),
    (np.array([2, 0]), np.array([-1, 0]), 'Opposite direction'),
]

origin = [0, 0]
colors = ['green', 'blue', 'red']

for i, (v1, v2, label) in enumerate(test_vecs):
    y_offset = i * 2.5

    # Draw vectors
    ax.quiver(*origin, v1[0], v1[1] + y_offset, angles='xy', scale_units='xy', scale=1,
              color=colors[i], width=0.01, alpha=0.7)
    ax.quiver(*origin, v2[0], v2[1] + y_offset, angles='xy', scale_units='xy', scale=1,
              color=colors[i], width=0.01, linestyle='--', alpha=0.7)

    dot = np.dot(v1, v2)
    ax.text(2.5, 0.5 + y_offset, f'{label}\na·b = {dot}',
            fontsize=9, ha='left', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

ax.set_xlim(-2.5, 5)
ax.set_ylim(-1, 6)
ax.set_title('Geometric Meaning of Dot Product', fontsize=12, fontweight='bold')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)

# Plot 3: ML prediction example
ax = axes[1, 0]

ax.text(0.5, 0.95, 'ML PREDICTION = DOT PRODUCT', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

pred_text = [
    "Making a house price prediction:",
    "",
    "Features (house):",
    "  [1500, 3, 10]",
    "  size  bed age",
    "",
    "Weights (model):",
    "  [150, 10000, -500]",
    "  $/sqft $/bed $/yr",
    "",
    "Prediction = features · weights",
    "",
    " = 1500×150 + 3×10000 + 10×(-500)",
    " = 225,000 + 30,000 - 5,000",
    " = $250,000",
    "",
    "🎯 Every ML prediction uses",
    "   dot products!"
]

y_pos = 0.85
for line in pred_text:
    if line.startswith(('Features', 'Weights', 'Prediction')):
        weight = 'bold'
        size = 10
    elif line.startswith('🎯'):
        weight = 'bold'
        size = 10
        color = 'darkgreen'
    else:
        weight = 'normal'
        size = 9
        color = 'black'

    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight, color=color if 'color' in locals() else 'black')
    y_pos -= 0.046

ax.axis('off')

# Plot 4: Key concepts
ax = axes[1, 1]
ax.text(0.5, 0.95, 'DOT PRODUCT KEY CONCEPTS', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

concepts = [
    "📐 Formula: a·b = Σ(aᵢ × bᵢ)",
    "",
    "📊 Input: Two vectors (same length)",
    "",
    "📍 Output: One number (scalar)",
    "",
    "🔢 Calculation:",
    "   1. Multiply corresponding elements",
    "   2. Sum all products",
    "",
    "📏 Geometric meaning:",
    "   Measures vector alignment",
    "   • Positive: similar direction",
    "   • Zero: perpendicular",
    "   • Negative: opposite direction",
    "",
    "🎯 ML Applications:",
    "   • Making predictions",
    "   • Measuring similarity",
    "   • Computing distances",
    "   • Neural network layers",
    "",
    "💡 Most important operation in ML!"
]

y_pos = 0.87
for line in concepts:
    if line.startswith(('📐', '📊', '📍', '🔢', '📏', '🎯', '💡')):
        weight = 'bold'
        size = 9.5
    else:
        weight = 'normal'
        size = 8.5
    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.037

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}03_dot_product.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 03_dot_product.png")
print()

# ============================================================================
# SECTION 4: MATRIX OPERATIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Matrix Multiplication and Transpose")
print("=" * 80)
print()

print("MATRIX MULTIPLICATION:")
print("-" * 70)
print("When we multiply matrices, we use dot products!")
print()

# Small example
A_small = np.array([
    [1, 2],
    [3, 4]
])

B_small = np.array([
    [5, 6],
    [7, 8]
])

print("Example:")
print("A =")
print(A_small)
print()
print("B =")
print(B_small)
print()

C = np.dot(A_small, B_small)

print("A × B = ")
print(C)
print()
print("How it works:")
print("  C[0,0] = A[row 0] · B[col 0] = [1,2] · [5,7] = 1×5 + 2×7 = 19")
print("  C[0,1] = A[row 0] · B[col 1] = [1,2] · [6,8] = 1×6 + 2×8 = 22")
print("  C[1,0] = A[row 1] · B[col 0] = [3,4] · [5,7] = 3×5 + 4×7 = 43")
print("  C[1,1] = A[row 1] · B[col 1] = [3,4] · [6,8] = 3×6 + 4×8 = 50")
print()

print("MATRIX TRANSPOSE:")
print("-" * 70)
print("Transpose = flip rows and columns")
print()

original = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

transposed = original.T

print("Original (2×3):")
print(original)
print()
print("Transposed (3×2):")
print(transposed)
print()
print("Notice: Rows became columns, columns became rows!")
print()

# ============================================================================
# VISUALIZATION 4: Matrix Operations
# ============================================================================
print("📊 Generating Visualization 4: Matrix Operations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('🔄 MATRIX OPERATIONS: Multiplication and Transpose',
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Matrix multiplication visualization
ax = axes[0, 0]

# Display the multiplication
ax.text(0.5, 0.95, 'MATRIX MULTIPLICATION', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

mult_text = [
    "A × B = C",
    "",
    "A (2×2)    B (2×2)    C (2×2)",
    "[1 2]   ×  [5 6]   =  [19 22]",
    "[3 4]      [7 8]      [43 50]",
    "",
    "Each element C[i,j] is:",
    "  A[row i] · B[col j]",
    "",
    "Example: C[0,0] = 19",
    "  = A[row 0] · B[col 0]",
    "  = [1,2] · [5,7]",
    "  = 1×5 + 2×7",
    "  = 5 + 14 = 19 ✅"
]

y_pos = 0.85
for line in mult_text:
    if line.startswith(('A ×', 'Each', 'Example', '=')):
        weight = 'bold'
        size = 10
    else:
        weight = 'normal'
        size = 9
    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.055

ax.axis('off')

# Plot 2: Transpose visualization
ax = axes[0, 1]

# Create visual representation
original_mat = np.array([[1, 2, 3], [4, 5, 6]])
transposed_mat = original_mat.T

y_start = 0.7
ax.text(0.5, 0.9, 'MATRIX TRANSPOSE', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

ax.text(0.25, y_start, 'Original (2×3):', fontsize=10, fontweight='bold',
        ha='center', transform=ax.transAxes)
ax.text(0.75, y_start, 'Transposed (3×2):', fontsize=10, fontweight='bold',
        ha='center', transform=ax.transAxes)

# Draw matrices
for i in range(2):
    for j in range(3):
        rect1 = plt.Rectangle((0.05 + j*0.12, 0.45 - i*0.15), 0.1, 0.12,
                             facecolor='lightblue', edgecolor='black', linewidth=1.5,
                             transform=ax.transAxes)
        ax.add_patch(rect1)
        ax.text(0.1 + j*0.12, 0.51 - i*0.15, str(original_mat[i, j]),
               ha='center', va='center', fontsize=11, fontweight='bold',
               transform=ax.transAxes)

        rect2 = plt.Rectangle((0.55 + i*0.12, 0.45 - j*0.15), 0.1, 0.12,
                             facecolor='lightgreen', edgecolor='black', linewidth=1.5,
                             transform=ax.transAxes)
        ax.add_patch(rect2)
        ax.text(0.6 + i*0.12, 0.51 - j*0.15, str(transposed_mat[j, i]),
               ha='center', va='center', fontsize=11, fontweight='bold',
               transform=ax.transAxes)

# Arrow
ax.annotate('', xy=(0.52, 0.5), xytext=(0.43, 0.5),
           arrowprops=dict(arrowstyle='->', lw=3, color='red'),
           transform=ax.transAxes)
ax.text(0.475, 0.53, 'Transpose\n(flip)', ha='center', fontsize=9,
       fontweight='bold', transform=ax.transAxes)

ax.text(0.5, 0.15, 'Rows → Columns,  Columns → Rows',
        ha='center', fontsize=10, style='italic', transform=ax.transAxes)

ax.axis('off')

# Plot 3: ML application
ax = axes[1, 0]

ax.text(0.5, 0.95, 'ML APPLICATION: BATCH PREDICTIONS', fontsize=11, fontweight='bold',
        ha='center', transform=ax.transAxes)

ml_text = [
    "Predict prices for 3 houses at once!",
    "",
    "X (3 houses × 2 features):",
    "  [1000  2]  ← House 1",
    "  [1500  3]  ← House 2",
    "  [2000  4]  ← House 3",
    "   sqft bed",
    "",
    "Weights (2×1):",
    "  [150]  ← $/sqft",
    "  [10000] ← $/bedroom",
    "",
    "Predictions = X × Weights",
    "",
    "  [1000×150 + 2×10000  ]   [170,000]",
    "= [1500×150 + 3×10000  ] = [255,000]",
    "  [2000×150 + 4×10000  ]   [340,000]",
    "",
    "🎯 One matrix multiplication",
    "   → All predictions at once!"
]

y_pos = 0.87
for line in ml_text:
    if line.startswith(('X ', 'Weights', 'Predictions', '🎯')):
        weight = 'bold'
        size = 9.5
    else:
        weight = 'normal'
        size = 8.5
    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.042

ax.axis('off')

# Plot 4: Key concepts
ax = axes[1, 1]
ax.text(0.5, 0.95, 'MATRIX OPERATIONS: KEY POINTS', fontsize=11, fontweight='bold',
        ha='center', transform=ax.transAxes)

concepts = [
    "MULTIPLICATION:",
    "• C = A × B",
    "• C[i,j] = A[row i] · B[col j]",
    "• A is (m×n), B is (n×p)",
    "  → C is (m×p)",
    "• Inner dimensions must match!",
    "",
    "TRANSPOSE:",
    "• Notation: Aᵀ or A.T",
    "• Flips rows ↔ columns",
    "• (m×n) → (n×m)",
    "• (Aᵀ)ᵀ = A",
    "",
    "ML USAGE:",
    "• X × W = predictions",
    "• Process many samples at once",
    "• Efficient computation",
    "• Core of neural networks!",
    "",
    "💡 Matrix ops make ML fast!"
]

y_pos = 0.87
for line in concepts:
    if line.startswith(('MULTIPLICATION', 'TRANSPOSE', 'ML USAGE', '💡')):
        weight = 'bold'
        size = 10
    else:
        weight = 'normal'
        size = 8.5
    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.039

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}04_matrix_operations.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 04_matrix_operations.png")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ SUMMARY: Linear Algebra for Machine Learning")
print("=" * 80)
print()

print("📊 VECTORS:")
print("   • Ordered list of numbers: [v₁, v₂, ..., vₙ]")
print("   • Geometric: arrow from origin to point")
print("   • ML: Each data point is a vector")
print()

print("📐 MATRICES:")
print("   • Rectangular table of numbers")
print("   • Shape: m rows × n columns")
print("   • ML: Entire dataset is a matrix!")
print("   • Rows = samples, Columns = features")
print()

print("⭐ DOT PRODUCT:")
print("   • a · b = Σ(aᵢ × bᵢ)")
print("   • Combines two vectors → one number")
print("   • ML: Used for predictions, similarities")
print("   • Most important operation in ML!")
print()

print("🔄 MATRIX MULTIPLICATION:")
print("   • C = A × B")
print("   • Uses dot products of rows and columns")
print("   • ML: Process many samples simultaneously")
print()

print("↔️  TRANSPOSE:")
print("   • Flip rows and columns")
print("   • (m×n) becomes (n×m)")
print("   • Notation: Aᵀ or A.T")
print()

print("🤖 WHY IT MATTERS:")
print("   • ML is built on linear algebra")
print("   • Data = vectors in a matrix")
print("   • Predictions = matrix multiplications")
print("   • Understanding this unlocks all of ML!")
print()

print("=" * 80)
print("📁 Visualizations saved to:", VISUAL_DIR)
print("=" * 80)
print("✅ 01_vectors.png")
print("✅ 02_matrices.png")
print("✅ 03_dot_product.png")
print("✅ 04_matrix_operations.png")
print("=" * 80)
print()

print("🎓 NEXT STEPS:")
print("   1. Review all visualizations - especially dot product!")
print("   2. Watch 3Blue1Brown's Linear Algebra series (absolute must!)")
print("   3. Try creating your own vectors and matrices in NumPy")
print("   4. Move to next module: 05_probability_basics.py")
print("   5. Then you're ready for actual regression algorithms!")
print()

print("💡 KEY TAKEAWAY:")
print("   Linear algebra turns machine learning into simple matrix operations.")
print("   Once you see data as matrices, everything clicks!")
print()

print("=" * 80)
print("🎉 Module Complete! You now speak the language of ML!")
print("=" * 80)
