"""
📐 INTRODUCTION TO DERIVATIVES - The Foundation of Machine Learning Optimization

================================================================================
LEARNING OBJECTIVES
================================================================================
After completing this module, you will understand:
1. What a slope is and how to calculate "rise over run"
2. What a derivative represents (slope at a single point)
3. How to find minimums and maximums (where slope = 0)
4. The intuition behind gradient descent
5. Why derivatives are crucial for machine learning

================================================================================
📺 RECOMMENDED VIDEOS (WATCH THESE!)
================================================================================
⭐ MUST WATCH:
   - 3Blue1Brown: "Essence of Calculus Chapter 2 - The paradox of the derivative"
     https://www.youtube.com/watch?v=9vKqVkMQHKk
     (Best visual explanation of derivatives ever made!)

Also Recommended:
   - StatQuest: "Gradient Descent, Step-by-Step"
     https://www.youtube.com/watch?v=sDv4f4s2SB8
     (Clear explanation of how ML models learn)

   - Khan Academy: "Introduction to derivatives"
     https://www.youtube.com/watch?v=5yfh5cf4-0w
     (Step-by-step basics)

   - Welch Labs: "Neural Networks Demystified Part 2"
     https://www.youtube.com/watch?v=H-ybCx8gt-8
     (Gradient descent visualized)

================================================================================
OVERVIEW
================================================================================
Don't worry - we're NOT doing heavy calculus! We're learning just enough to
understand how machine learning models "learn" by following slopes downhill.

Think of it this way:
- You're standing on a hill and want to get to the bottom
- You look around to see which direction slopes downward
- You take a step in that direction
- Repeat until you reach the bottom
- That's gradient descent! And derivatives tell you which way is "downward"

This module uses LOTS of visualizations to make these concepts crystal clear.
We'll see slopes, tangent lines, and watch a ball roll down a curve!
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')

# Setup visualization directory
VISUAL_DIR = '../visuals/03_derivatives/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("📐 INTRODUCTION TO DERIVATIVES")
print("   Understanding Slopes, Rates of Change, and How ML Models Learn")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: SLOPE - RISE OVER RUN
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: SLOPE - The Foundation")
print("=" * 80)
print()

print("WHAT IS SLOPE?")
print("-" * 70)
print("Slope measures how steep a line is.")
print("It tells us: 'For every step to the right, how many steps up or down?'")
print()
print("FORMULA: slope = rise / run = Δy / Δx = (y₂ - y₁) / (x₂ - x₁)")
print()
print("Where:")
print("  • rise (Δy) = change in y = y₂ - y₁")
print("  • run (Δx) = change in x = x₂ - x₁")
print("  • Δ (delta) = 'change in'")
print()

print("EXAMPLE 1: Finding slope between two points")
print("-" * 70)
# Two points
x1, y1 = 2, 3
x2, y2 = 6, 11

print(f"Point 1: ({x1}, {y1})")
print(f"Point 2: ({x2}, {y2})")
print()
print("Step-by-step calculation:")
print(f"  rise (Δy) = y₂ - y₁ = {y2} - {y1} = {y2 - y1}")
print(f"  run (Δx)  = x₂ - x₁ = {x2} - {x1} = {x2 - x1}")
print(f"  slope     = rise / run = {y2 - y1} / {x2 - x1} = {(y2 - y1) / (x2 - x1)}")
print()
print(f"✅ The slope is {(y2 - y1) / (x2 - x1)}")
print(f"   Interpretation: For every 1 unit we move right, we move up {(y2 - y1) / (x2 - x1)} units")
print()

print("DIFFERENT TYPES OF SLOPES:")
print("-" * 70)
print("  • Positive slope: Line goes upward (↗) - y increases as x increases")
print("  • Negative slope: Line goes downward (↘) - y decreases as x increases")
print("  • Zero slope: Flat line (→) - y stays the same")
print("  • Undefined slope: Vertical line (↑) - x stays the same")
print()

# ============================================================================
# VISUALIZATION 1: Understanding Slope (Rise over Run)
# ============================================================================
print("📊 Generating Visualization 1: Rise over Run Diagram...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('📐 UNDERSTANDING SLOPE: Rise Over Run', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Detailed slope calculation
ax = axes[0, 0]
x_points = np.array([2, 6])
y_points = np.array([3, 11])
ax.plot(x_points, y_points, 'b-', linewidth=2, label='Line')
ax.scatter(x_points, y_points, color='red', s=100, zorder=5)

# Annotate points
ax.annotate(f'Point 1: ({x1}, {y1})', xy=(x1, y1), xytext=(x1-1, y1-1.5),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
ax.annotate(f'Point 2: ({x2}, {y2})', xy=(x2, y2), xytext=(x2+1, y2+1),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Draw rise and run
ax.plot([x2, x2], [y1, y2], 'g--', linewidth=2, label=f'Rise = {y2-y1}')
ax.plot([x1, x2], [y1, y1], 'r--', linewidth=2, label=f'Run = {x2-x1}')

# Add rise/run labels
ax.text(x2 + 0.3, (y1 + y2) / 2, f'Rise\n= {y2-y1}', fontsize=11, color='green', fontweight='bold')
ax.text((x1 + x2) / 2, y1 - 0.8, f'Run = {x2-x1}', fontsize=11, color='red', fontweight='bold')

ax.text(4, 1, f'Slope = Rise/Run = {y2-y1}/{x2-x1} = {(y2-y1)/(x2-x1)}',
        fontsize=12, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.set_title('Slope Calculation: Rise Over Run', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 8)
ax.set_ylim(0, 13)

# Plot 2: Different slope types
ax = axes[0, 1]
x = np.linspace(0, 10, 100)

ax.plot(x, 0.5*x + 2, 'g-', linewidth=2, label='Positive (↗): slope = 0.5')
ax.plot(x, -0.5*x + 8, 'r-', linewidth=2, label='Negative (↘): slope = -0.5')
ax.plot(x, np.ones_like(x) * 5, 'b-', linewidth=2, label='Zero (→): slope = 0')

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.set_title('Types of Slopes', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Plot 3: Steepness comparison
ax = axes[1, 0]
ax.plot(x, 0.3*x + 1, '-', linewidth=2, label='Gentle: slope = 0.3', color='lightgreen')
ax.plot(x, 0.7*x + 1, '-', linewidth=2, label='Medium: slope = 0.7', color='orange')
ax.plot(x, 1.5*x + 1, '-', linewidth=2, label='Steep: slope = 1.5', color='darkred')

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.set_title('Slope Values and Steepness', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 5)
ax.set_ylim(0, 10)

# Plot 4: Slope as rate of change
ax = axes[1, 1]
ax.text(0.5, 0.8, 'SLOPE AS RATE OF CHANGE', fontsize=14, fontweight='bold',
        ha='center', transform=ax.transAxes)
ax.text(0.5, 0.65, 'Slope tells us how fast y changes with respect to x',
        fontsize=11, ha='center', transform=ax.transAxes, style='italic')

examples = [
    "📈 If slope = 2:",
    "   → When x increases by 1, y increases by 2",
    "",
    "📉 If slope = -3:",
    "   → When x increases by 1, y decreases by 3",
    "",
    "→ If slope = 0:",
    "   → When x changes, y stays the same",
    "",
    "Real-world examples:",
    "• Speed: distance/time (slope of position graph)",
    "• Price increase: $/year (slope of price graph)",
    "• Temperature change: °/hour"
]

y_pos = 0.52
for line in examples:
    ax.text(0.5, y_pos, line, fontsize=10, ha='center', transform=ax.transAxes,
            family='monospace')
    y_pos -= 0.05

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}01_rise_over_run.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 01_rise_over_run.png")
print()

# ============================================================================
# SECTION 2: FROM SLOPE TO DERIVATIVE
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: What is a Derivative?")
print("=" * 80)
print()

print("THE PROBLEM WITH CURVES:")
print("-" * 70)
print("We just learned how to find the slope of a straight line.")
print("But what about a CURVE? The slope keeps changing!")
print()
print("Example: f(x) = x²")
print("  • At x = 1, the curve is going up gently")
print("  • At x = 5, the curve is going up steeply")
print("  • The slope is different at every point!")
print()

print("THE SOLUTION: DERIVATIVES")
print("-" * 70)
print("A derivative is the slope at a SINGLE POINT on a curve.")
print()
print("How do we find it?")
print("  1. Pick a point on the curve")
print("  2. Draw a tangent line (a line that just touches the curve at that point)")
print("  3. The slope of that tangent line is the derivative!")
print()
print("NOTATION:")
print("  • f(x) = the function (e.g., f(x) = x²)")
print("  • f'(x) = the derivative (read as 'f prime of x')")
print("  • Also written as: df/dx or dy/dx")
print()

print("EXAMPLE: Derivative of f(x) = x²")
print("-" * 70)
print("Function: f(x) = x²")
print("Derivative: f'(x) = 2x")
print()
print("What does this mean?")
print("  • At any point x, the slope of the curve is 2x")
print()
print("Let's calculate slopes at different points:")

test_points = [0, 1, 2, 3, 4]
print(f"{'x':<10} {'f(x) = x²':<15} {'f\'(x) = 2x (slope)':<25} {'Interpretation'}")
print("-" * 75)
for x in test_points:
    fx = x**2
    fpx = 2*x
    if fpx > 0:
        direction = f"going up at rate {fpx}"
    elif fpx < 0:
        direction = f"going down at rate {abs(fpx)}"
    else:
        direction = "flat (slope = 0)"
    print(f"{x:<10} {fx:<15} {fpx:<25} {direction}")

print()
print("📌 KEY INSIGHT:")
print("   The derivative tells us the RATE OF CHANGE at each point.")
print("   For f(x) = x², the rate of change gets faster as x increases!")
print()

# ============================================================================
# VISUALIZATION 2: Tangent Lines and Derivatives
# ============================================================================
print("📊 Generating Visualization 2: Tangent Lines and Derivatives...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('🎯 DERIVATIVES: Slope at a Point', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Function with tangent lines at different points
ax = axes[0, 0]
x = np.linspace(-1, 5, 200)
y = x**2

ax.plot(x, y, 'b-', linewidth=3, label='f(x) = x²')

# Draw tangent lines at specific points
tangent_points = [0, 1, 2, 3]
colors = ['red', 'green', 'orange', 'purple']

for point, color in zip(tangent_points, colors):
    # Point on curve
    y_point = point**2
    ax.scatter([point], [y_point], color=color, s=100, zorder=5)

    # Tangent line
    slope = 2 * point  # derivative of x² is 2x
    x_tangent = np.linspace(point - 1, point + 1, 50)
    y_tangent = slope * (x_tangent - point) + y_point

    ax.plot(x_tangent, y_tangent, '--', color=color, linewidth=2, alpha=0.7,
            label=f'x={point}: slope={slope}')

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('f(x)', fontsize=11)
ax.set_title('Tangent Lines at Different Points', fontsize=12, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 15)

# Plot 2: The derivative function
ax = axes[0, 1]
x = np.linspace(-3, 3, 200)
original = x**2
derivative = 2*x

ax.plot(x, original, 'b-', linewidth=2, label='f(x) = x² (original)')
ax.plot(x, derivative, 'r-', linewidth=2, label="f'(x) = 2x (derivative)")
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axvline(x=0, color='black', linewidth=0.5)

ax.scatter([0], [0], color='green', s=150, zorder=5, marker='*',
           label='Minimum (slope=0)')

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.set_title('Original Function vs Its Derivative', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(-6, 10)

# Plot 3: Geometric interpretation
ax = axes[1, 0]
x_point = 2
y_point = x_point**2
slope = 2 * x_point

x = np.linspace(0, 4, 200)
y = x**2
ax.plot(x, y, 'b-', linewidth=3)
ax.scatter([x_point], [y_point], color='red', s=200, zorder=5)

# Draw tangent
x_tangent = np.linspace(x_point - 1, x_point + 1.5, 50)
y_tangent = slope * (x_tangent - x_point) + y_point
ax.plot(x_tangent, y_tangent, 'r--', linewidth=2, label=f'Tangent: slope = {slope}')

# Show rise and run for tangent
run_point = x_point + 1
rise_point = slope * 1
ax.plot([x_point, run_point], [y_point, y_point], 'g--', linewidth=2)
ax.plot([run_point, run_point], [y_point, y_point + rise_point], 'g--', linewidth=2)

ax.text(x_point + 0.5, y_point - 0.5, f'run = 1', fontsize=10, color='green')
ax.text(run_point + 0.2, y_point + rise_point/2, f'rise = {slope}', fontsize=10, color='green')

ax.annotate(f'Point ({x_point}, {y_point})\nSlope = {slope}',
            xy=(x_point, y_point), xytext=(x_point - 1.2, y_point + 3),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('f(x)', fontsize=11)
ax.set_title('Derivative = Slope of Tangent Line', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 4)
ax.set_ylim(-1, 10)

# Plot 4: Key facts about derivatives
ax = axes[1, 1]
ax.text(0.5, 0.85, 'KEY FACTS ABOUT DERIVATIVES', fontsize=13, fontweight='bold',
        ha='center', transform=ax.transAxes)

facts = [
    "📐 Derivative = Slope at a point",
    "",
    "📊 If f'(x) > 0:",
    "   → Function is INCREASING",
    "   → Going uphill",
    "",
    "📉 If f'(x) < 0:",
    "   → Function is DECREASING",
    "   → Going downhill",
    "",
    "🎯 If f'(x) = 0:",
    "   → Function is FLAT",
    "   → Could be a minimum or maximum!",
    "",
    "Common derivatives to know:",
    "• f(x) = x²     → f'(x) = 2x",
    "• f(x) = x³     → f'(x) = 3x²",
    "• f(x) = 2x + 3 → f'(x) = 2"
]

y_pos = 0.72
for line in facts:
    ax.text(0.5, y_pos, line, fontsize=10, ha='center', transform=ax.transAxes,
            family='monospace')
    y_pos -= 0.045

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}02_tangent_lines_derivatives.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 02_tangent_lines_derivatives.png")
print()

# ============================================================================
# SECTION 3: FINDING MINIMUMS AND MAXIMUMS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Finding Minimums and Maximums")
print("=" * 80)
print()

print("WHY DO WE CARE ABOUT MINIMUMS?")
print("-" * 70)
print("In machine learning, we want to MINIMIZE ERROR!")
print("  • We make predictions")
print("  • We calculate how wrong we are (error)")
print("  • We want to find the MINIMUM error")
print()
print("Derivatives help us find minimums and maximums!")
print()

print("THE KEY INSIGHT:")
print("-" * 70)
print("At a minimum or maximum:")
print("  • The function is FLAT")
print("  • The slope is ZERO")
print("  • The derivative equals zero: f'(x) = 0")
print()

print("EXAMPLE: Finding the minimum of f(x) = x² - 4x + 5")
print("-" * 70)
print("Function: f(x) = x² - 4x + 5")
print("Derivative: f'(x) = 2x - 4")
print()
print("To find minimum, set derivative equal to zero:")
print("  f'(x) = 0")
print("  2x - 4 = 0")
print("  2x = 4")
print("  x = 2")
print()
print("At x = 2:")
print(f"  f(2) = (2)² - 4(2) + 5 = 4 - 8 + 5 = 1")
print()
print("✅ The minimum is at point (2, 1)")
print()

print("HOW DO WE KNOW IT'S A MINIMUM (NOT A MAXIMUM)?")
print("-" * 70)
print("Check the derivative on both sides:")
print("  • At x = 1 (left of minimum): f'(1) = 2(1) - 4 = -2 (negative, going down)")
print("  • At x = 2 (at minimum): f'(2) = 2(2) - 4 = 0 (flat)")
print("  • At x = 3 (right of minimum): f'(3) = 2(3) - 4 = 2 (positive, going up)")
print()
print("Going down → flat → going up = MINIMUM! ✅")
print()

# ============================================================================
# VISUALIZATION 3: Finding Minimums with Derivatives
# ============================================================================
print("📊 Generating Visualization 3: Finding Minimums...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('🎯 FINDING MINIMUMS: Where Derivative = 0', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Function with minimum marked
ax = axes[0, 0]
x = np.linspace(-1, 5, 200)
y = x**2 - 4*x + 5

ax.plot(x, y, 'b-', linewidth=3, label='f(x) = x² - 4x + 5')
ax.scatter([2], [1], color='red', s=300, zorder=5, marker='*', label='Minimum at (2, 1)')

# Draw tangent line at minimum (flat line)
ax.plot([0.5, 3.5], [1, 1], 'r--', linewidth=2, alpha=0.7, label='Tangent: slope = 0')

ax.annotate('MINIMUM\n(2, 1)\nSlope = 0',
            xy=(2, 1), xytext=(3, 3),
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('f(x)', fontsize=11)
ax.set_title('Minimum Point: Where Slope = 0', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 5)
ax.set_ylim(0, 8)

# Plot 2: Derivative function showing where it crosses zero
ax = axes[0, 1]
x = np.linspace(-1, 5, 200)
derivative = 2*x - 4

ax.plot(x, derivative, 'r-', linewidth=3, label="f'(x) = 2x - 4")
ax.axhline(y=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax.axvline(x=2, color='green', linewidth=2, linestyle='--', alpha=0.7, label='x = 2 (where derivative = 0)')

ax.scatter([2], [0], color='red', s=200, zorder=5, marker='*')

# Shade regions
ax.fill_between(x[x < 2], derivative[x < 2], alpha=0.3, color='orange', label='f\'(x) < 0 (decreasing)')
ax.fill_between(x[x > 2], derivative[x > 2], alpha=0.3, color='lightgreen', label='f\'(x) > 0 (increasing)')

ax.annotate('Derivative = 0\nMinimum here!',
            xy=(2, 0), xytext=(3, -2),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel("f'(x)", fontsize=11)
ax.set_title('Derivative Function: Crosses Zero at Minimum', fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 5)
ax.set_ylim(-6, 6)

# Plot 3: Both functions together
ax = axes[1, 0]
x = np.linspace(-1, 5, 200)
original = x**2 - 4*x + 5
derivative = 2*x - 4

ax.plot(x, original, 'b-', linewidth=2, label='f(x) = x² - 4x + 5')
ax.plot(x, derivative, 'r-', linewidth=2, label="f'(x) = 2x - 4")

ax.axhline(y=0, color='black', linewidth=0.5)
ax.axvline(x=2, color='green', linewidth=2, linestyle='--', alpha=0.5)

ax.scatter([2], [1], color='blue', s=150, zorder=5, marker='o', label='Min of f(x)')
ax.scatter([2], [0], color='red', s=150, zorder=5, marker='o', label="Zero of f'(x)")

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('y', fontsize=11)
ax.set_title('Original and Derivative Together', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 5)
ax.set_ylim(-6, 10)

# Plot 4: Step-by-step process
ax = axes[1, 1]
ax.text(0.5, 0.9, 'FINDING MINIMUMS: STEP-BY-STEP', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

steps = [
    "STEP 1: Start with function",
    "        f(x) = x² - 4x + 5",
    "",
    "STEP 2: Find the derivative",
    "        f'(x) = 2x - 4",
    "",
    "STEP 3: Set derivative equal to zero",
    "        f'(x) = 0",
    "        2x - 4 = 0",
    "",
    "STEP 4: Solve for x",
    "        2x = 4",
    "        x = 2",
    "",
    "STEP 5: Find y-value",
    "        f(2) = (2)² - 4(2) + 5",
    "             = 4 - 8 + 5 = 1",
    "",
    "RESULT: Minimum at (2, 1) ✅",
    "",
    "Why this matters for ML:",
    "We use this same process to minimize",
    "prediction error!"
]

y_pos = 0.78
for line in steps:
    if line.startswith('STEP') or line.startswith('RESULT'):
        weight = 'bold'
        size = 10
    else:
        weight = 'normal'
        size = 9
    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.037

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}03_finding_minimums.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 03_finding_minimums.png")
print()

# ============================================================================
# SECTION 4: GRADIENT DESCENT - HOW MACHINE LEARNING LEARNS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Gradient Descent - How ML Models Learn")
print("=" * 80)
print()

print("THE BIG PICTURE:")
print("-" * 70)
print("Imagine you're blindfolded on a hill and want to reach the bottom:")
print("  1. Feel the ground around you to find which way slopes down")
print("  2. Take a step in that direction")
print("  3. Repeat until you reach the bottom")
print()
print("That's EXACTLY how machine learning models learn!")
print()

print("GRADIENT DESCENT ALGORITHM:")
print("-" * 70)
print("Goal: Find the minimum of a function (minimum error)")
print()
print("Process:")
print("  1. Start at a random point")
print("  2. Calculate the derivative (which way is downhill?)")
print("  3. Move in the opposite direction of the derivative")
print("  4. Repeat until derivative ≈ 0 (we're at the bottom!)")
print()

print("MATHEMATICAL FORMULA:")
print("-" * 70)
print("  x_new = x_old - α × f'(x_old)")
print()
print("Where:")
print("  • x_old = current position")
print("  • f'(x_old) = derivative at current position (slope)")
print("  • α (alpha) = learning rate (how big a step to take)")
print("  • x_new = new position after taking a step")
print()

print("EXAMPLE: Using gradient descent on f(x) = x² - 4x + 5")
print("-" * 70)
print("We know the minimum is at x = 2, but let's pretend we don't!")
print("We'll start at x = 5 and use gradient descent to find the minimum.")
print()

# Gradient descent simulation
def f(x):
    return x**2 - 4*x + 5

def f_prime(x):
    return 2*x - 4

x = 5.0  # Starting point
alpha = 0.1  # Learning rate
steps = []

print(f"{'Step':<8} {'x':<12} {'f(x)':<12} {'f\'(x)':<12} {'Action'}")
print("-" * 75)

for i in range(15):
    fx = f(x)
    fpx = f_prime(x)
    steps.append((x, fx, fpx))

    action = f"Move from {x:.3f} to {x - alpha * fpx:.3f}"
    print(f"{i:<8} {x:<12.4f} {fx:<12.4f} {fpx:<12.4f} {action}")

    # Update x
    x = x - alpha * fpx

    # Stop if derivative is very small (we're at minimum)
    if abs(fpx) < 0.01:
        print()
        print(f"✅ Converged at x = {x:.4f}, f(x) = {f(x):.4f}")
        print(f"   (True minimum is at x = 2, f(2) = 1)")
        break

print()
print("🎯 KEY OBSERVATIONS:")
print("  • We started at x = 5 (far from minimum)")
print("  • Each step moved us closer to x = 2")
print("  • The derivative got smaller as we approached the minimum")
print("  • When derivative ≈ 0, we stopped (found the minimum!)")
print()

# ============================================================================
# VISUALIZATION 4: Gradient Descent Animation
# ============================================================================
print("📊 Generating Visualization 4: Gradient Descent...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('⚡ GRADIENT DESCENT: Walking Downhill to the Minimum',
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Gradient descent path
ax = axes[0, 0]
x_range = np.linspace(-1, 6, 200)
y_range = x_range**2 - 4*x_range + 5

ax.plot(x_range, y_range, 'b-', linewidth=3, label='f(x) = x² - 4x + 5')

# Plot the steps
x = 5.0
alpha = 0.1
path_x = [x]
path_y = [f(x)]

for i in range(15):
    fpx = f_prime(x)
    x = x - alpha * fpx
    path_x.append(x)
    path_y.append(f(x))
    if abs(fpx) < 0.01:
        break

ax.plot(path_x, path_y, 'ro-', markersize=6, linewidth=2, alpha=0.7, label='Gradient descent path')
ax.scatter([path_x[0]], [path_y[0]], color='green', s=200, zorder=5, marker='o', label='Start')
ax.scatter([path_x[-1]], [path_y[-1]], color='red', s=300, zorder=5, marker='*', label='End (minimum)')

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('f(x)', fontsize=11)
ax.set_title('Gradient Descent Path', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 6)
ax.set_ylim(0, 10)

# Plot 2: Learning rate comparison
ax = axes[0, 1]
ax.plot(x_range, y_range, 'b-', linewidth=2, alpha=0.3)

learning_rates = [0.05, 0.1, 0.3]
colors = ['green', 'orange', 'red']

for lr, color in zip(learning_rates, colors):
    x = 5.0
    path_x = [x]
    path_y = [f(x)]

    for i in range(20):
        fpx = f_prime(x)
        x = x - lr * fpx
        path_x.append(x)
        path_y.append(f(x))
        if abs(fpx) < 0.01:
            break

    ax.plot(path_x, path_y, 'o-', color=color, markersize=4, linewidth=1.5,
            alpha=0.7, label=f'α = {lr} ({len(path_x)} steps)')

ax.scatter([2], [1], color='red', s=200, zorder=5, marker='*', label='Target minimum')

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('f(x)', fontsize=11)
ax.set_title('Effect of Learning Rate (α)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 6)
ax.set_ylim(0, 10)

# Plot 3: Derivative as direction indicator
ax = axes[1, 0]

positions = [0.5, 2, 4]
for pos in positions:
    # Plot point on curve
    ax.plot(x_range, y_range, 'b-', linewidth=2, alpha=0.5)
    y_pos = f(pos)
    ax.scatter([pos], [y_pos], s=150, zorder=5)

    # Calculate derivative
    slope = f_prime(pos)

    # Draw arrow showing gradient descent direction
    if slope > 0:
        arrow_dir = -0.5
        text_msg = f"f'({pos}) = {slope:.1f} > 0\n← Move LEFT"
    elif slope < 0:
        arrow_dir = 0.5
        text_msg = f"f'({pos}) = {slope:.1f} < 0\n→ Move RIGHT"
    else:
        arrow_dir = 0
        text_msg = f"f'({pos}) = 0\n✅ At minimum!"

    if arrow_dir != 0:
        ax.arrow(pos, y_pos + 0.5, arrow_dir, 0, head_width=0.3, head_length=0.2,
                fc='red', ec='red', linewidth=2)

    ax.text(pos, y_pos + 2, text_msg, fontsize=9, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

ax.set_xlabel('x', fontsize=11)
ax.set_ylabel('f(x)', fontsize=11)
ax.set_title('Derivative Tells Us Which Direction to Move', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 6)
ax.set_ylim(0, 12)

# Plot 4: Summary infographic
ax = axes[1, 1]
ax.text(0.5, 0.92, 'GRADIENT DESCENT ALGORITHM', fontsize=13, fontweight='bold',
        ha='center', transform=ax.transAxes)

summary = [
    "Goal: Find minimum of f(x)",
    "",
    "Algorithm:",
    "1️⃣  Pick random starting point x",
    "2️⃣  Calculate derivative f'(x)",
    "3️⃣  Update: x ← x - α·f'(x)",
    "4️⃣  Repeat until f'(x) ≈ 0",
    "",
    "Key parameters:",
    "• α (learning rate): step size",
    "  - Too small: slow convergence",
    "  - Too large: might overshoot",
    "  - Just right: fast convergence ✅",
    "",
    "Why it works:",
    "• If f'(x) > 0: x decreases (go left)",
    "• If f'(x) < 0: x increases (go right)",
    "• Moving opposite to gradient → downhill",
    "",
    "Machine Learning Application:",
    "Replace f(x) with ERROR function,",
    "and x with model parameters.",
    "Gradient descent finds parameters",
    "that MINIMIZE ERROR! 🎯"
]

y_pos = 0.82
for line in summary:
    if line.startswith(('Goal', 'Algorithm', 'Key', 'Why', 'Machine')):
        weight = 'bold'
        size = 10
    elif any(line.startswith(emoji) for emoji in ['1️⃣', '2️⃣', '3️⃣', '4️⃣', '•']):
        weight = 'normal'
        size = 9
    else:
        weight = 'normal'
        size = 9

    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.035

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}04_gradient_descent.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 04_gradient_descent.png")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ SUMMARY: What You Learned")
print("=" * 80)
print()

print("📐 SLOPE (Rise over Run):")
print("   • Slope = Δy / Δx = (y₂ - y₁) / (x₂ - x₁)")
print("   • Measures steepness of a line")
print("   • Tells us rate of change")
print()

print("📊 DERIVATIVE (Slope at a Point):")
print("   • Derivative = slope of tangent line at a point")
print("   • For f(x) = x², the derivative is f'(x) = 2x")
print("   • Tells us instantaneous rate of change")
print()

print("🎯 FINDING MINIMUMS:")
print("   • At minimum: derivative = 0 (flat point)")
print("   • Process: Set f'(x) = 0 and solve for x")
print("   • Check: going down → flat → going up = minimum")
print()

print("⚡ GRADIENT DESCENT:")
print("   • Algorithm: x_new = x_old - α × f'(x_old)")
print("   • Iteratively moves toward minimum")
print("   • Used by ALL machine learning algorithms to learn!")
print("   • Learning rate (α) controls step size")
print()

print("🤖 CONNECTION TO MACHINE LEARNING:")
print("   • ML models have ERROR functions")
print("   • Goal: minimize error")
print("   • Gradient descent finds best parameters")
print("   • Derivatives tell us which direction reduces error")
print("   • This is how neural networks, linear regression, etc. all learn!")
print()

print("=" * 80)
print("📁 Visualizations saved to:", VISUAL_DIR)
print("=" * 80)
print("✅ 01_rise_over_run.png")
print("✅ 02_tangent_lines_derivatives.png")
print("✅ 03_finding_minimums.png")
print("✅ 04_gradient_descent.png")
print("=" * 80)
print()

print("🎓 NEXT STEPS:")
print("   1. Review the visualizations - they tell the whole story!")
print("   2. Watch the recommended YouTube videos (especially 3Blue1Brown)")
print("   3. Try modifying the code: change functions, learning rates, starting points")
print("   4. Move to next module: 04_linear_algebra_basics.py")
print()

print("💡 KEY TAKEAWAY:")
print("   Derivatives are the foundation of machine learning optimization.")
print("   Every time a model 'learns', it's using gradient descent with derivatives!")
print()

print("=" * 80)
print("🎉 Module Complete! You now understand how ML models learn!")
print("=" * 80)
