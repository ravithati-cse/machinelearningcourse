"""
ALGEBRA BASICS FOR MACHINE LEARNING
====================================

This module teaches fundamental algebra concepts needed for understanding linear regression.
Don't worry if you've forgotten your math - we'll rebuild it from scratch with lots of visuals!

LEARNING OBJECTIVES:
-------------------
1. Understand what variables, constants, and equations are
2. Master the linear equation: y = mx + b
3. Interpret slopes (m) and intercepts (b)
4. Plot and visualize linear relationships
5. See why this matters for machine learning

📺 RECOMMENDED VIDEOS (watch these for deeper understanding):
---------------------------------------------------------
⭐ Khan Academy: "Slope and y-intercept"
   https://www.khanacademy.org/math/algebra/x2f8bb11595b61c86:forms-of-linear-equations

⭐ 3Blue1Brown: "Essence of Linear Algebra preview"
   https://www.youtube.com/watch?v=fNk_zzaMoSs

⭐ Organic Chemistry Tutor: "Linear Equations"
   https://www.youtube.com/watch?v=2UrcOKr7HAw

VISUAL-FIRST APPROACH:
---------------------
This file generates several visualizations that will be saved to visuals/01_algebra/
You can run this file and study the images to understand concepts visually!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Set up visual style for educational clarity
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Create output directory for visualizations
VISUAL_DIR = '../visuals/01_algebra/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 70)
print("ALGEBRA BASICS: Building Blocks for Machine Learning")
print("=" * 70)
print()

# ============================================================================
# SECTION 1: WHAT ARE VARIABLES AND CONSTANTS?
# ============================================================================

print("SECTION 1: Variables and Constants")
print("-" * 70)
print()
print("VARIABLE: A symbol (like x or y) that can represent different values")
print("  Example: x could be 1, 2, 3, or any number")
print()
print("CONSTANT: A fixed value that doesn't change")
print("  Examples: 5, -3, 100")
print()

# Example demonstration
x_values = [1, 2, 3, 4, 5]
print(f"If x takes different values: {x_values}")
print(f"And we compute 2 * x, we get: {[2 * x for x in x_values]}")
print("Notice: 2 is a constant (always 2), but x is a variable (changes)")
print()

# ============================================================================
# SECTION 2: THE LINEAR EQUATION - y = mx + b
# ============================================================================

print("SECTION 2: The Most Important Equation in Machine Learning")
print("-" * 70)
print()
print("THE LINEAR EQUATION: y = mx + b")
print()
print("Let's break down each component:")
print("  y  = the OUTPUT (what we're predicting)")
print("  x  = the INPUT (what we know)")
print("  m  = the SLOPE (how much y changes when x increases by 1)")
print("  b  = the INTERCEPT (value of y when x = 0)")
print()
print("REAL-WORLD EXAMPLE:")
print("  Predicting house prices: y = 150x + 50000")
print("  y (price in $) = 150 * x (square feet) + 50000 (base price)")
print("  → A house with 0 sqft would cost $50,000 (the intercept b)")
print("  → Each additional square foot adds $150 (the slope m)")
print()

# ============================================================================
# VISUALIZATION 1: ANATOMY OF y = mx + b (INFOGRAPHIC)
# ============================================================================

print("Creating Infographic: Anatomy of y = mx + b...")

fig, ax = plt.subplots(1, 1, figsize=(14, 10))

# Title
fig.suptitle('📊 ANATOMY OF THE LINEAR EQUATION: y = mx + b',
             fontsize=20, fontweight='bold', y=0.98)

# Create text boxes explaining each component
components = [
    {
        'label': 'y',
        'name': 'OUTPUT',
        'desc': 'The value we want to predict or calculate',
        'example': 'House Price ($)',
        'color': '#FF6B6B',
        'position': (0.1, 0.75)
    },
    {
        'label': '=',
        'name': 'EQUALS',
        'desc': 'Means "is calculated as"',
        'example': '',
        'color': '#4ECDC4',
        'position': (0.3, 0.75)
    },
    {
        'label': 'm',
        'name': 'SLOPE',
        'desc': 'How much y changes per unit increase in x',
        'example': '$150 per sqft',
        'color': '#95E1D3',
        'position': (0.4, 0.75)
    },
    {
        'label': 'x',
        'name': 'INPUT',
        'desc': 'The known value we use to predict',
        'example': 'House Size (sqft)',
        'color': '#F38181',
        'position': (0.55, 0.75)
    },
    {
        'label': '+',
        'name': 'PLUS',
        'desc': 'Add this next component',
        'example': '',
        'color': '#4ECDC4',
        'position': (0.7, 0.75)
    },
    {
        'label': 'b',
        'name': 'INTERCEPT',
        'desc': 'Starting value when x = 0',
        'example': '$50,000 base',
        'color': '#AA96DA',
        'position': (0.8, 0.75)
    }
]

# Draw component boxes
for comp in components:
    # Main label box
    box = FancyBboxPatch(comp['position'], 0.08, 0.12,
                         boxstyle="round,pad=0.01",
                         facecolor=comp['color'],
                         edgecolor='black', linewidth=2,
                         transform=ax.transAxes, zorder=2)
    ax.add_patch(box)

    # Large label
    ax.text(comp['position'][0] + 0.04, comp['position'][1] + 0.06,
            comp['label'], fontsize=32, fontweight='bold',
            ha='center', va='center', transform=ax.transAxes)

    # Component name
    ax.text(comp['position'][0] + 0.04, comp['position'][1] - 0.05,
            comp['name'], fontsize=12, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)

    # Description
    ax.text(comp['position'][0] + 0.04, comp['position'][1] - 0.10,
            comp['desc'], fontsize=9,
            ha='center', va='top', transform=ax.transAxes,
            wrap=True, style='italic')

    # Example
    if comp['example']:
        ax.text(comp['position'][0] + 0.04, comp['position'][1] - 0.20,
                f"Ex: {comp['example']}", fontsize=10, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Add complete example at bottom
example_text = """
COMPLETE EXAMPLE:  y = 150x + 50000

If a house has x = 1000 square feet:
y = 150(1000) + 50000
y = 150,000 + 50,000
y = $200,000

If a house has x = 1500 square feet:
y = 150(1500) + 50000
y = 225,000 + 50,000
y = $275,000
"""

ax.text(0.5, 0.30, example_text, fontsize=12,
        ha='center', va='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightblue',
                 edgecolor='darkblue', linewidth=2, alpha=0.8),
        family='monospace')

# Add "Why This Matters" box
why_text = """
🎯 WHY THIS MATTERS FOR MACHINE LEARNING:

Linear Regression is about finding the BEST values for m and b!
Given data points, ML algorithms figure out:
  • What slope (m) fits the data best?
  • What intercept (b) fits the data best?

This simple equation is the foundation of machine learning!
"""

ax.text(0.5, 0.08, why_text, fontsize=11,
        ha='center', va='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightgreen',
                 edgecolor='darkgreen', linewidth=2, alpha=0.7))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}01_anatomy_of_linear_equation.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {VISUAL_DIR}01_anatomy_of_linear_equation.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: UNDERSTANDING SLOPE (m)
# ============================================================================

print("Creating Visualization: Understanding Slope...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('📈 UNDERSTANDING SLOPE (m): How y Changes with x',
             fontsize=18, fontweight='bold')

# Generate x values
x = np.linspace(0, 10, 100)

# Different slopes to demonstrate
slopes = [0.5, 1, 2, -0.5, -1, -2]
titles = ['Gentle Positive (m=0.5)', 'Moderate Positive (m=1)', 'Steep Positive (m=2)',
          'Gentle Negative (m=-0.5)', 'Moderate Negative (m=-1)', 'Steep Negative (m=-2)']
colors = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']

for idx, (ax, m, title, color) in enumerate(zip(axes.flat, slopes, titles, colors)):
    b = 2  # Keep intercept constant
    y = m * x + b

    # Plot the line
    ax.plot(x, y, color=color, linewidth=3, label=f'y = {m}x + {b}')
    ax.axhline(y=b, color='gray', linestyle='--', alpha=0.5, label=f'Intercept b={b}')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Add "rise over run" visualization for positive slopes
    if m > 0 and idx < 3:
        x1, x2 = 2, 4
        y1, y2 = m * x1 + b, m * x2 + b

        # Horizontal line (run)
        ax.plot([x1, x2], [y1, y1], 'k--', linewidth=2)
        ax.text((x1 + x2) / 2, y1 - 0.5, f'run = {x2 - x1}',
               ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        # Vertical line (rise)
        ax.plot([x2, x2], [y1, y2], 'k--', linewidth=2)
        ax.text(x2 + 0.5, (y1 + y2) / 2, f'rise = {y2 - y1:.1f}',
               ha='left', fontsize=9, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

        # Slope calculation
        ax.text(0.5, 0.95, f'slope = rise/run = {y2-y1:.1f}/{x2-x1} = {m}',
               transform=ax.transAxes, ha='center', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('x (Input)', fontsize=10)
    ax.set_ylabel('y (Output)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 10)

    # Add interpretation
    if m > 0:
        interp = f"As x increases, y increases\n(positive relationship)"
    elif m < 0:
        interp = f"As x increases, y decreases\n(negative relationship)"
    else:
        interp = f"y doesn't change with x\n(no relationship)"

    ax.text(0.5, 0.05, interp, transform=ax.transAxes,
           ha='center', va='bottom', fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}02_understanding_slope.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {VISUAL_DIR}02_understanding_slope.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: UNDERSTANDING INTERCEPT (b)
# ============================================================================

print("Creating Visualization: Understanding Intercept...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('📍 UNDERSTANDING INTERCEPT (b): Where the Line Crosses y-axis',
             fontsize=18, fontweight='bold')

# Keep slope constant, vary intercept
m = 1  # Constant slope
intercepts = [-2, 0, 3]
x = np.linspace(-5, 5, 100)

for idx, (ax, b) in enumerate(zip(axes, intercepts)):
    y = m * x + b

    # Plot the line
    ax.plot(x, y, linewidth=3, label=f'y = {m}x + {b}', color=f'C{idx}')

    # Highlight the intercept point
    ax.plot(0, b, 'ro', markersize=15, label=f'Intercept = {b}', zorder=5)
    ax.axhline(y=b, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=2, alpha=0.7, label='y-axis')

    # Add annotation
    ax.annotate(f'When x=0, y={b}', xy=(0, b), xytext=(1.5, b+1),
               fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))

    ax.set_title(f'Intercept b = {b}', fontweight='bold', fontsize=13)
    ax.set_xlabel('x (Input)', fontsize=11)
    ax.set_ylabel('y (Output)', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 8)

    # Add interpretation
    ax.text(0.5, 0.05, f'The line crosses the y-axis at y = {b}',
           transform=ax.transAxes, ha='center', va='bottom', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}03_understanding_intercept.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {VISUAL_DIR}03_understanding_intercept.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: INTERACTIVE EXPLORATION - HOW m AND b WORK TOGETHER
# ============================================================================

print("Creating Visualization: How m and b Work Together...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('🎨 HOW SLOPE (m) AND INTERCEPT (b) WORK TOGETHER',
             fontsize=18, fontweight='bold')

x = np.linspace(0, 10, 100)

# Configuration: [m, b, title, description]
configs = [
    [2, 1, 'Steep & Low Start', 'High slope, low intercept\nRapid growth from low base'],
    [0.5, 5, 'Gentle & High Start', 'Low slope, high intercept\nSlow growth from high base'],
    [-1, 8, 'Decreasing from High', 'Negative slope, high intercept\nDecreasing trend'],
    [1.5, 0, 'Moderate Through Origin', 'Moderate slope, zero intercept\nPasses through (0, 0)']
]

for ax, (m, b, title, desc) in zip(axes.flat, configs):
    y = m * x + b

    # Plot the line
    ax.plot(x, y, linewidth=3, label=f'y = {m}x + {b}', color='darkblue')

    # Highlight intercept
    ax.plot(0, b, 'ro', markersize=12, label=f'Intercept = {b}', zorder=5)

    # Show slope triangle
    if m != 0:
        x1, x2 = 2, 4
        y1, y2 = m * x1 + b, m * x2 + b
        ax.plot([x1, x2, x2, x1], [y1, y1, y2, y1], 'r--', linewidth=2, alpha=0.7)
        ax.text((x1 + x2) / 2, y1 - 0.3, f'Δx = {x2-x1}', ha='center', fontsize=9)
        ax.text(x2 + 0.2, (y1 + y2) / 2, f'Δy = {y2-y1:.1f}', ha='left', fontsize=9)

    # Add grid and axes
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.legend(fontsize=10, loc='best')
    ax.set_xlim(0, 10)

    # Add description
    ax.text(0.5, 0.05, desc, transform=ax.transAxes,
           ha='center', va='bottom', fontsize=9,
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}04_slope_intercept_together.png',
            dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Saved: {VISUAL_DIR}04_slope_intercept_together.png")
plt.close()

# ============================================================================
# SECTION 3: REAL-WORLD PRACTICE
# ============================================================================

print()
print("SECTION 3: Practice with Real-World Examples")
print("-" * 70)
print()

# Example 1: Temperature conversion
print("EXAMPLE 1: Converting Fahrenheit to Celsius")
print("Formula: C = (5/9)(F - 32)")
print("This can be rewritten as: C = (5/9)F - 160/9")
print("In the form y = mx + b:")
print("  y (Celsius) = 0.556 * x (Fahrenheit) - 17.78")
print()

fahrenheit = [32, 50, 68, 86, 104]
celsius = [(5/9) * (f - 32) for f in fahrenheit]

print(f"Fahrenheit: {fahrenheit}")
print(f"Celsius:    {[f'{c:.1f}' for c in celsius]}")
print()

# Example 2: Taxi fare
print("EXAMPLE 2: Taxi Fare Calculation")
print("Formula: Fare = $3 (base) + $2 per mile")
print("In the form y = mx + b:")
print("  y (Total Fare) = 2 * x (Miles) + 3")
print()

miles = [0, 1, 2, 5, 10]
fares = [2 * m + 3 for m in miles]

print(f"Miles:  {miles}")
print(f"Fare:   {['$' + str(f) for f in fares]}")
print()

# ============================================================================
# SECTION 4: WHY THIS MATTERS FOR MACHINE LEARNING
# ============================================================================

print("SECTION 4: Connection to Machine Learning")
print("-" * 70)
print()
print("🎯 THE BIG PICTURE:")
print()
print("In Linear Regression, we have DATA POINTS (x, y)")
print("Our goal: Find the BEST LINE (y = mx + b) that fits the data")
print()
print("Machine Learning finds:")
print("  • The optimal slope (m) - how much y changes with x")
print("  • The optimal intercept (b) - the starting value")
print()
print("Example: Predicting house prices from size")
print("  • Data: [(1000 sqft, $150k), (1500 sqft, $225k), (2000 sqft, $300k)]")
print("  • ML finds: y = 150x + 0  (approx)")
print("  • Meaning: Each sqft adds $150, with no base price")
print()
print("That's it! Linear regression is just finding m and b!")
print()

# ============================================================================
# TRY IT YOURSELF!
# ============================================================================

print("=" * 70)
print("✅ SUMMARY: What You Learned")
print("=" * 70)
print()
print("1. Variables (x, y) represent changing values")
print("2. Constants (numbers) are fixed values")
print("3. Linear equation: y = mx + b")
print("4. Slope (m) controls how steep the line is")
print("5. Intercept (b) controls where the line starts")
print("6. ML = finding the best m and b for your data!")
print()
print("📊 Visual files created in:", VISUAL_DIR)
print("   - 01_anatomy_of_linear_equation.png")
print("   - 02_understanding_slope.png")
print("   - 03_understanding_intercept.png")
print("   - 04_slope_intercept_together.png")
print()
print("🎓 NEXT STEPS:")
print("   1. Review the generated visualizations")
print("   2. Watch the recommended YouTube videos")
print("   3. Try changing the values of m and b in the code above")
print("   4. Move on to 02_statistics_fundamentals.py")
print()
print("=" * 70)

# ============================================================================
# OPTIONAL: Create a simple interactive example
# ============================================================================

def calculate_y(m, x, b):
    """Calculate y = mx + b"""
    return m * x + b

print("\n🔧 INTERACTIVE CALCULATOR:")
print("-" * 70)
print("Try some values yourself!")
print()
print("Example: If m=2, x=5, b=3")
print(f"Then y = {calculate_y(2, 5, 3)}")
print()
print("Try editing the code to test your own values!")
print()

# That's all! You've mastered the basics of linear equations!
# Next: Learn statistics to understand data better!
