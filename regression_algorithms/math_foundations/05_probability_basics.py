"""
🎲 PROBABILITY BASICS - Understanding Uncertainty in Machine Learning

================================================================================
LEARNING OBJECTIVES
================================================================================
After completing this module, you will understand:
1. What probability means and how to interpret it (0 to 1 scale)
2. Probability distributions and what they tell us
3. The normal distribution (bell curve) and why it's everywhere
4. The 68-95-99.7 rule for normal distributions
5. How randomness and noise affect data
6. Why probability matters for machine learning

================================================================================
📺 RECOMMENDED VIDEOS (WATCH THESE!)
================================================================================
⭐ MUST WATCH:
   - StatQuest: "Probability is not Likelihood"
     https://www.youtube.com/watch?v=pYxNSUDSFH4
     (Clears up a common confusion!)

   - StatQuest: "The Normal Distribution, Clearly Explained!!!"
     https://www.youtube.com/watch?v=rzFX5NWojp0
     (Best explanation of the bell curve)

Also Recommended:
   - Khan Academy: "Introduction to Probability"
     https://www.youtube.com/watch?v=uzkc-qNVoOk

   - Crash Course Statistics: "Probability"
     https://www.youtube.com/watch?v=OyddY7DlV58
     (Fun and engaging intro)

   - 3Blue1Brown: "Binomial distributions"
     https://www.youtube.com/watch?v=8idr1WZ1A7Q
     (Beautiful visualizations)

================================================================================
OVERVIEW
================================================================================
Machine learning is all about dealing with UNCERTAINTY!

Real-world questions:
- Will this customer buy? (Probability)
- How confident are we in this prediction? (Uncertainty)
- Is this data point unusual? (Distributions)
- Why don't all points fit the line perfectly? (Randomness/noise)

Probability helps us answer these questions mathematically.
Don't worry - we focus on intuition, not complex formulas!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Setup visualization directory
VISUAL_DIR = '../visuals/05_probability/'
os.makedirs(VISUAL_DIR, exist_ok=True)

print("=" * 80)
print("🎲 PROBABILITY BASICS")
print("   Understanding Uncertainty and Randomness")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: WHAT IS PROBABILITY?
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Understanding Probability")
print("=" * 80)
print()

print("WHAT IS PROBABILITY?")
print("-" * 70)
print("Probability measures how likely something is to happen.")
print("It's always a number between 0 and 1 (or 0% to 100%)")
print()
print("THE PROBABILITY SCALE:")
print("  0.0 (0%)   = Impossible - will never happen")
print("  0.25 (25%) = Unlikely - but could happen")
print("  0.5 (50%)  = Even odds - coin flip")
print("  0.75 (75%) = Likely - probably will happen")
print("  1.0 (100%) = Certain - will definitely happen")
print()

print("REAL-WORLD EXAMPLES:")
print("-" * 70)

examples = [
    ("Sun rising tomorrow", 1.0, "100%", "Certain"),
    ("Flipping heads on fair coin", 0.5, "50%", "Even odds"),
    ("Rolling a 6 on a die", 1/6, "16.7%", "Unlikely"),
    ("Drawing an Ace from deck", 4/52, "7.7%", "Rare"),
    ("Finding a unicorn", 0.0, "0%", "Impossible")
]

print(f"{'Event':<30} {'Probability':<15} {'Percent':<10} {'Description'}")
print("-" * 75)
for event, prob, percent, desc in examples:
    print(f"{event:<30} {prob:<15.3f} {percent:<10} {desc}")

print()

print("BASIC PROBABILITY RULES:")
print("-" * 70)
print("1. All probabilities sum to 1")
print("   Example: P(heads) + P(tails) = 0.5 + 0.5 = 1.0")
print()
print("2. Probability of NOT happening = 1 - P(happening)")
print("   Example: If P(rain) = 0.3, then P(no rain) = 1 - 0.3 = 0.7")
print()
print("3. Probabilities can't be negative or greater than 1")
print("   Must be: 0 ≤ P ≤ 1")
print()

# Simulate coin flips to show probability
print("EXPERIMENT: Flipping a coin 1000 times")
print("-" * 70)

n_flips = 1000
flips = np.random.choice(['Heads', 'Tails'], size=n_flips)
heads_count = np.sum(flips == 'Heads')
heads_probability = heads_count / n_flips

print(f"Results after {n_flips} flips:")
print(f"  Heads: {heads_count} times")
print(f"  Tails: {n_flips - heads_count} times")
print(f"  Probability of heads: {heads_probability:.3f}")
print(f"  (Close to theoretical 0.500!)")
print()

# ============================================================================
# VISUALIZATION 1: Probability Scale and Examples
# ============================================================================
print("📊 Generating Visualization 1: Probability Fundamentals...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('🎲 UNDERSTANDING PROBABILITY: From 0 to 1', fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Probability scale
ax = axes[0, 0]

# Draw scale
scale_y = 0.5
ax.plot([0, 1], [scale_y, scale_y], 'k-', linewidth=3)

# Mark points on scale
probs = [0, 0.25, 0.5, 0.75, 1.0]
labels = ['Impossible\n0%', 'Unlikely\n25%', 'Even\n50%', 'Likely\n75%', 'Certain\n100%']
colors = ['red', 'orange', 'yellow', 'lightgreen', 'darkgreen']

for prob, label, color in zip(probs, labels, colors):
    ax.scatter([prob], [scale_y], s=300, color=color, edgecolor='black', linewidth=2, zorder=5)
    ax.text(prob, scale_y - 0.15, label, ha='center', fontsize=9, fontweight='bold')

ax.set_xlim(-0.1, 1.1)
ax.set_ylim(0, 1)
ax.set_title('The Probability Scale', fontsize=12, fontweight='bold')
ax.axis('off')

# Add examples
example_events = [
    ("🌞 Sun rises tomorrow", 1.0),
    ("🎲 Roll a 6", 1/6),
    ("🪙 Flip heads", 0.5),
    ("🦄 Find a unicorn", 0.0)
]

y_pos = 0.85
for event, prob in example_events:
    ax.plot([prob, prob], [scale_y + 0.05, y_pos], 'k--', alpha=0.3, linewidth=1)
    ax.text(prob, y_pos + 0.02, f'{event}\nP = {prob:.2f}', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    y_pos -= 0.15

# Plot 2: Coin flip simulation
ax = axes[0, 1]

# Simulate increasing flips and track running probability
flip_counts = [10, 50, 100, 500, 1000, 5000]
running_probs = []

for n in flip_counts:
    flips = np.random.choice([0, 1], size=n)  # 0=Tails, 1=Heads
    prob = np.mean(flips)
    running_probs.append(prob)

ax.plot(flip_counts, running_probs, 'bo-', markersize=8, linewidth=2, label='Observed probability')
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Theoretical (0.5)')

ax.set_xlabel('Number of flips', fontsize=11)
ax.set_ylabel('Probability of heads', fontsize=11)
ax.set_title('Law of Large Numbers\n(More flips → closer to true probability)', fontsize=11, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.35, 0.65)

# Plot 3: Dice rolling probabilities
ax = axes[1, 0]

# Theoretical probabilities for dice
outcomes = np.arange(1, 7)
probabilities = np.ones(6) / 6  # Each outcome has probability 1/6

ax.bar(outcomes, probabilities, color='skyblue', edgecolor='black', linewidth=1.5, alpha=0.8)

for outcome, prob in zip(outcomes, probabilities):
    ax.text(outcome, prob + 0.01, f'{prob:.3f}\n({prob*100:.1f}%)',
            ha='center', fontsize=9, fontweight='bold')

ax.axhline(y=1/6, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Equal probability')

ax.set_xlabel('Dice outcome', fontsize=11)
ax.set_ylabel('Probability', fontsize=11)
ax.set_title('Fair Dice: All Outcomes Equally Likely', fontsize=12, fontweight='bold')
ax.set_xticks(outcomes)
ax.set_ylim(0, 0.25)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Plot 4: Key concepts
ax = axes[1, 1]
ax.text(0.5, 0.95, 'PROBABILITY KEY CONCEPTS', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

concepts = [
    "📏 Range: 0 ≤ P ≤ 1",
    "   (or 0% to 100%)",
    "",
    "🎯 Interpretation:",
    "   • P = 0: Impossible",
    "   • P = 0.5: Even odds",
    "   • P = 1: Certain",
    "",
    "➕ Sum rule:",
    "   All probabilities sum to 1",
    "",
    "↔️  Complement:",
    "   P(not A) = 1 - P(A)",
    "",
    "📊 Law of Large Numbers:",
    "   More trials → closer to",
    "   theoretical probability",
    "",
    "🤖 ML Connection:",
    "   • Predict probabilities",
    "   • Measure confidence",
    "   • Understand uncertainty"
]

y_pos = 0.87
for line in concepts:
    if line.startswith(('📏', '🎯', '➕', '↔️', '📊', '🤖')):
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
plt.savefig(f'{VISUAL_DIR}01_probability_fundamentals.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 01_probability_fundamentals.png")
print()

# ============================================================================
# SECTION 2: PROBABILITY DISTRIBUTIONS
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Probability Distributions")
print("=" * 80)
print()

print("WHAT IS A PROBABILITY DISTRIBUTION?")
print("-" * 70)
print("A probability distribution shows us:")
print("  • All possible outcomes")
print("  • How likely each outcome is")
print()
print("Think of it as a map of probabilities!")
print()

print("TYPES OF DISTRIBUTIONS:")
print("-" * 70)
print()
print("1. UNIFORM DISTRIBUTION: All outcomes equally likely")
print("   Example: Rolling a fair die - each number (1-6) has P = 1/6")
print()

# Generate uniform distribution
uniform_data = np.random.uniform(0, 10, 1000)
print("   Generated 1000 random numbers between 0 and 10")
print(f"   Min: {uniform_data.min():.2f}, Max: {uniform_data.max():.2f}")
print(f"   Mean: {uniform_data.mean():.2f} (should be around 5)")
print()

print("2. NORMAL DISTRIBUTION (Bell Curve): Most common in nature!")
print("   • Symmetric around the mean")
print("   • Most values near the center")
print("   • Fewer values at the extremes")
print("   Examples: Heights, test scores, measurement errors")
print()

# Generate normal distribution
normal_data = np.random.normal(loc=100, scale=15, size=1000)
print("   Generated 1000 random numbers from normal distribution")
print(f"   Mean = 100, Standard deviation = 15")
print(f"   Actual mean: {normal_data.mean():.2f}")
print(f"   Actual std: {normal_data.std():.2f}")
print()

# ============================================================================
# VISUALIZATION 2: Probability Distributions
# ============================================================================
print("📊 Generating Visualization 2: Probability Distributions...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('📊 PROBABILITY DISTRIBUTIONS: Patterns of Randomness',
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Uniform distribution
ax = axes[0, 0]

uniform_data = np.random.uniform(0, 10, 10000)
ax.hist(uniform_data, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)

# Theoretical uniform line
ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Theoretical uniform')

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Probability density', fontsize=11)
ax.set_title('Uniform Distribution\n(All values equally likely)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax.text(5, 0.12, 'All heights approximately equal\n→ Uniform!',
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 2: Normal distribution
ax = axes[0, 1]

normal_data = np.random.normal(loc=0, scale=1, size=10000)
ax.hist(normal_data, bins=50, color='lightgreen', edgecolor='black', alpha=0.7, density=True)

# Theoretical normal curve
x = np.linspace(-4, 4, 100)
ax.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=3, label='Theoretical normal')

ax.set_xlabel('Value (standard deviations from mean)', fontsize=10)
ax.set_ylabel('Probability density', fontsize=11)
ax.set_title('Normal Distribution (Bell Curve)\n(Most common in nature!)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(-4, 4)

# Plot 3: Comparing distributions
ax = axes[1, 0]

# Generate different distributions
x = np.linspace(-5, 5, 1000)

normal_pdf = stats.norm.pdf(x, 0, 1)
wide_normal_pdf = stats.norm.pdf(x, 0, 2)
shifted_normal_pdf = stats.norm.pdf(x, 2, 1)

ax.plot(x, normal_pdf, 'b-', linewidth=2, label='Normal (μ=0, σ=1)')
ax.plot(x, wide_normal_pdf, 'r-', linewidth=2, label='Wide (μ=0, σ=2)')
ax.plot(x, shifted_normal_pdf, 'g-', linewidth=2, label='Shifted (μ=2, σ=1)')

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Probability density', fontsize=11)
ax.set_title('Normal Distributions with Different Parameters', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 5)

ax.text(0, 0.45, '← Higher, narrower\n   (smaller σ)',
        fontsize=8, bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax.text(-2.5, 0.15, '← Shorter, wider\n   (larger σ)',
        fontsize=8, bbox=dict(boxstyle='round', facecolor='lightpink'))

# Plot 4: Key concepts
ax = axes[1, 1]
ax.text(0.5, 0.95, 'DISTRIBUTION KEY CONCEPTS', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

concepts = [
    "📊 What: Shows all possible values",
    "   and their probabilities",
    "",
    "📐 Uniform distribution:",
    "   • All values equally likely",
    "   • Flat shape",
    "   • Example: Fair dice",
    "",
    "🔔 Normal distribution:",
    "   • Bell-shaped curve",
    "   • Symmetric around mean (μ)",
    "   • Spread controlled by std dev (σ)",
    "   • Most common in real data!",
    "",
    "🎯 Parameters:",
    "   • μ (mu): mean/center",
    "   • σ (sigma): standard deviation",
    "",
    "🤖 ML Connection:",
    "   • Errors often normally distributed",
    "   • Understand data patterns",
    "   • Detect outliers"
]

y_pos = 0.87
for line in concepts:
    if line.startswith(('📊', '📐', '🔔', '🎯', '🤖')):
        weight = 'bold'
        size = 9.5
    else:
        weight = 'normal'
        size = 8.5
    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.04

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}02_probability_distributions.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 02_probability_distributions.png")
print()

# ============================================================================
# SECTION 3: THE NORMAL DISTRIBUTION (BELL CURVE)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: The Normal Distribution - The Most Important Distribution")
print("=" * 80)
print()

print("WHY IS THE NORMAL DISTRIBUTION SO IMPORTANT?")
print("-" * 70)
print("It appears EVERYWHERE in real life:")
print("  • Heights of people")
print("  • Test scores")
print("  • Measurement errors")
print("  • Blood pressure")
print("  • IQ scores")
print("  • And many more!")
print()

print("THE 68-95-99.7 RULE (Empirical Rule)")
print("-" * 70)
print("This is THE most important rule for normal distributions:")
print()
print("  • 68% of data within 1 standard deviation of mean")
print("  • 95% of data within 2 standard deviations of mean")
print("  • 99.7% of data within 3 standard deviations of mean")
print()

# Example with IQ scores
mean_iq = 100
std_iq = 15

print("EXAMPLE: IQ Scores")
print("-" * 70)
print(f"Mean (μ) = {mean_iq}")
print(f"Standard deviation (σ) = {std_iq}")
print()

print("Using the 68-95-99.7 rule:")
print(f"  • 68% of people have IQ between {mean_iq - std_iq} and {mean_iq + std_iq}")
print(f"    (μ ± 1σ = {mean_iq} ± {std_iq})")
print()
print(f"  • 95% of people have IQ between {mean_iq - 2*std_iq} and {mean_iq + 2*std_iq}")
print(f"    (μ ± 2σ = {mean_iq} ± {2*std_iq})")
print()
print(f"  • 99.7% of people have IQ between {mean_iq - 3*std_iq} and {mean_iq + 3*std_iq}")
print(f"    (μ ± 3σ = {mean_iq} ± {3*std_iq})")
print()

print("WHAT THIS MEANS:")
print("  • If someone has IQ = 130 (2σ above mean), they're in top ~2.5%!")
print("  • If someone has IQ = 145 (3σ above mean), they're in top ~0.15%!")
print("  • Values beyond 3σ are very rare (outliers)")
print()

print("STANDARD NORMAL DISTRIBUTION (Z-scores):")
print("-" * 70)
print("Special case where μ = 0 and σ = 1")
print("We can convert ANY normal distribution to standard normal!")
print()
print("Formula: z = (x - μ) / σ")
print("  • z = number of standard deviations from the mean")
print("  • Also called a 'z-score'")
print()

# Example
person_iq = 130
z_score = (person_iq - mean_iq) / std_iq
print(f"Example: Person with IQ = {person_iq}")
print(f"  z = ({person_iq} - {mean_iq}) / {std_iq}")
print(f"  z = {z_score:.2f}")
print(f"  → This person is {z_score:.2f} standard deviations above average")
print()

# ============================================================================
# VISUALIZATION 3: Normal Distribution and 68-95-99.7 Rule
# ============================================================================
print("📊 Generating Visualization 3: Normal Distribution...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('🔔 THE NORMAL DISTRIBUTION: The Bell Curve',
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: The 68-95-99.7 rule
ax = axes[0, 0]

mu, sigma = 0, 1
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x, mu, sigma)

ax.plot(x, y, 'b-', linewidth=3, label='Normal distribution')

# Shade areas
ax.fill_between(x, 0, y, where=(x >= -1) & (x <= 1),
                alpha=0.3, color='green', label='68% (±1σ)')
ax.fill_between(x, 0, y, where=(x >= -2) & (x <= -1),
                alpha=0.2, color='orange')
ax.fill_between(x, 0, y, where=(x >= 1) & (x <= 2),
                alpha=0.2, color='orange', label='95% (±2σ)')
ax.fill_between(x, 0, y, where=(x >= -3) & (x <= -2),
                alpha=0.1, color='red')
ax.fill_between(x, 0, y, where=(x >= 2) & (x <= 3),
                alpha=0.1, color='red', label='99.7% (±3σ)')

# Mark standard deviations
for i in range(-3, 4):
    ax.axvline(x=i, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    if i != 0:
        ax.text(i, -0.02, f'{i}σ', ha='center', fontsize=9)

ax.axvline(x=0, color='black', linewidth=2, label='Mean (μ)')

ax.set_xlabel('Standard deviations from mean', fontsize=11)
ax.set_ylabel('Probability density', fontsize=11)
ax.set_title('68-95-99.7 Rule (Empirical Rule)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(-4, 4)
ax.set_ylim(-0.05, 0.45)

# Add percentage labels
ax.text(0, 0.25, '68%', ha='center', fontsize=12, fontweight='bold', color='darkgreen')
ax.text(1.5, 0.1, '95%', ha='center', fontsize=11, fontweight='bold', color='darkorange')
ax.text(2.5, 0.03, '99.7%', ha='center', fontsize=10, fontweight='bold', color='darkred')

# Plot 2: Real-world example (IQ scores)
ax = axes[0, 1]

mu_iq, sigma_iq = 100, 15
x_iq = np.linspace(mu_iq - 4*sigma_iq, mu_iq + 4*sigma_iq, 1000)
y_iq = stats.norm.pdf(x_iq, mu_iq, sigma_iq)

ax.plot(x_iq, y_iq, 'b-', linewidth=3)

# Shade and mark regions
regions = [
    (mu_iq - sigma_iq, mu_iq + sigma_iq, 'green', '68%'),
    (mu_iq - 2*sigma_iq, mu_iq + 2*sigma_iq, 'orange', '95%'),
]

for lower, upper, color, label in regions:
    ax.fill_between(x_iq, 0, y_iq, where=(x_iq >= lower) & (x_iq <= upper),
                    alpha=0.2, color=color)

# Mark key IQ values
key_values = [70, 85, 100, 115, 130, 145]
for val in key_values:
    ax.axvline(x=val, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(val, -0.001, str(val), ha='center', fontsize=9)

ax.axvline(x=mu_iq, color='black', linewidth=2)

ax.set_xlabel('IQ Score', fontsize=11)
ax.set_ylabel('Probability density', fontsize=11)
ax.set_title('Real Example: IQ Scores\n(μ=100, σ=15)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(40, 160)

# Add annotations
ax.annotate('Average\nIQ = 100', xy=(100, 0.027), xytext=(100, 0.020),
            fontsize=9, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.annotate('Very rare\n(< 0.15%)', xy=(145, 0.0005), xytext=(150, 0.005),
            fontsize=8, ha='center',
            bbox=dict(boxstyle='round', facecolor='pink', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red'))

# Plot 3: Z-scores
ax = axes[1, 0]

# Generate sample data
np.random.seed(42)
sample_data = np.random.normal(100, 15, 1000)

# Convert to z-scores
z_scores = (sample_data - np.mean(sample_data)) / np.std(sample_data)

ax.hist(z_scores, bins=50, color='lightblue', edgecolor='black', alpha=0.7, density=True)

# Overlay standard normal
x_z = np.linspace(-4, 4, 100)
ax.plot(x_z, stats.norm.pdf(x_z, 0, 1), 'r-', linewidth=3, label='Standard Normal (μ=0, σ=1)')

ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Z-score (standard deviations from mean)', fontsize=10)
ax.set_ylabel('Probability density', fontsize=11)
ax.set_title('Z-Scores: Standardized Normal Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(-4, 4)

ax.text(0, 0.45, 'z = (x - μ) / σ\nConvert any data\nto standard scale!',
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 4: Key concepts
ax = axes[1, 1]
ax.text(0.5, 0.95, 'NORMAL DISTRIBUTION: KEY FACTS', fontsize=11, fontweight='bold',
        ha='center', transform=ax.transAxes)

concepts = [
    "🔔 Shape: Bell curve (symmetric)",
    "",
    "📊 Parameters:",
    "   • μ (mean): center",
    "   • σ (std dev): spread",
    "",
    "📏 68-95-99.7 Rule:",
    "   • 68% within μ ± 1σ",
    "   • 95% within μ ± 2σ",
    "   • 99.7% within μ ± 3σ",
    "",
    "🎯 Z-score formula:",
    "   z = (x - μ) / σ",
    "   → Standardizes any normal distribution",
    "",
    "🌟 Why it matters:",
    "   • Most common in nature",
    "   • Basis for many statistical tests",
    "   • Errors in ML often normal",
    "",
    "🤖 ML Applications:",
    "   • Detect outliers (beyond 3σ)",
    "   • Understand prediction errors",
    "   • Confidence intervals"
]

y_pos = 0.87
for line in concepts:
    if line.startswith(('🔔', '📊', '📏', '🎯', '🌟', '🤖')):
        weight = 'bold'
        size = 9
    else:
        weight = 'normal'
        size = 8
    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.038

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}03_normal_distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 03_normal_distribution.png")
print()

# ============================================================================
# SECTION 4: RANDOMNESS AND NOISE IN DATA
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Randomness and Noise in Machine Learning")
print("=" * 80)
print()

print("WHY PERFECT PREDICTIONS ARE IMPOSSIBLE:")
print("-" * 70)
print("Real data has NOISE - random variation we can't explain or predict.")
print()
print("Sources of noise:")
print("  • Measurement errors (thermometer not perfectly accurate)")
print("  • Unknown factors (weather affects sales, but we didn't measure it)")
print("  • Truly random events (person's mood when taking test)")
print("  • Data entry mistakes")
print()

print("EXAMPLE: House prices with noise")
print("-" * 70)

# Generate data with noise
np.random.seed(42)
true_sizes = np.linspace(1000, 3000, 50)
true_prices = 150 * true_sizes + 50000  # True relationship

# Add noise
noise = np.random.normal(0, 30000, 50)  # Random noise with std dev = $30,000
observed_prices = true_prices + noise

print("True relationship: Price = 150 × Size + 50,000")
print("But we observe prices with random noise added!")
print()
print("Sample data:")
print(f"{'Size (sqft)':<15} {'True Price':<15} {'Noise':<15} {'Observed Price'}")
print("-" * 70)
for i in range(5):
    print(f"{true_sizes[i]:<15.0f} ${true_prices[i]:<14,.0f} ${noise[i]:<14,.0f} ${observed_prices[i]:<,.0f}")

print("...")
print()
print("This is why data points don't fall perfectly on a line!")
print("Our job in ML: find the best line despite the noise.")
print()

print("RESIDUALS = NOISE:")
print("-" * 70)
print("Residual = Observed value - Predicted value")
print("         = The part we can't explain")
print("         = Noise!")
print()
print("In good models, residuals should:")
print("  • Be normally distributed (bell curve)")
print("  • Be centered around 0 (no systematic errors)")
print("  • Have constant variance (same spread everywhere)")
print()

# ============================================================================
# VISUALIZATION 4: Noise and Randomness
# ============================================================================
print("📊 Generating Visualization 4: Noise in Data...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('🎲 RANDOMNESS AND NOISE in Machine Learning',
             fontsize=16, fontweight='bold', y=0.995)

# Plot 1: Data with and without noise
ax = axes[0, 0]

ax.scatter(true_sizes, true_prices, color='green', s=30, alpha=0.7, label='True (no noise)', marker='s')
ax.scatter(true_sizes, observed_prices, color='blue', s=30, alpha=0.7, label='Observed (with noise)')

# Plot true line
ax.plot(true_sizes, true_prices, 'g--', linewidth=2, label='True relationship')

ax.set_xlabel('House size (sqft)', fontsize=11)
ax.set_ylabel('Price ($)', fontsize=11)
ax.set_title('Effect of Noise on Data', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax.text(1500, 450000, 'Green squares: Perfect data\nBlue dots: Real data with noise',
        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Plot 2: Noise distribution
ax = axes[0, 1]

ax.hist(noise, bins=30, color='orange', edgecolor='black', alpha=0.7, density=True)

# Overlay normal distribution
x_noise = np.linspace(noise.min(), noise.max(), 100)
ax.plot(x_noise, stats.norm.pdf(x_noise, 0, 30000), 'r-', linewidth=3,
        label='Normal(μ=0, σ=30,000)')

ax.axvline(x=0, color='black', linestyle='--', linewidth=2)

ax.set_xlabel('Noise ($)', fontsize=11)
ax.set_ylabel('Probability density', fontsize=11)
ax.set_title('Distribution of Noise\n(Usually Normal!)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax.text(0, max(stats.norm.pdf(x_noise, 0, 30000)) * 0.7,
        'Noise centered at 0\n→ No systematic bias',
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# Plot 3: Signal vs Noise
ax = axes[1, 0]

# Create signal and noise
x_time = np.linspace(0, 10, 200)
signal = np.sin(x_time)  # True pattern
noise_component = np.random.normal(0, 0.2, 200)  # Random noise
observed = signal + noise_component  # What we actually see

ax.plot(x_time, signal, 'g-', linewidth=3, label='Signal (true pattern)', alpha=0.8)
ax.plot(x_time, observed, 'b-', linewidth=1, alpha=0.6, label='Signal + Noise (observed)')

ax.fill_between(x_time, signal - 0.2, signal + 0.2, alpha=0.2, color='red',
                label='Noise range (±σ)')

ax.set_xlabel('Time', fontsize=11)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Signal vs Noise', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

ax.text(5, -1.5, 'ML Goal: Extract the signal from noisy data!',
        ha='center', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# Plot 4: Key concepts
ax = axes[1, 1]
ax.text(0.5, 0.95, 'NOISE IN ML: KEY CONCEPTS', fontsize=12, fontweight='bold',
        ha='center', transform=ax.transAxes)

concepts = [
    "🎲 What: Random variation in data",
    "",
    "📊 Sources:",
    "   • Measurement errors",
    "   • Unknown factors",
    "   • Truly random events",
    "   • Human mistakes",
    "",
    "📏 Typical distribution:",
    "   • Usually NORMAL (bell curve)",
    "   • Centered at 0",
    "   • Constant variance",
    "",
    "🎯 Impact on ML:",
    "   • Makes perfect predictions impossible",
    "   • Data doesn't fall on perfect line",
    "   • Need to find best fit despite noise",
    "",
    "🔍 Residuals:",
    "   • Residual = Observed - Predicted",
    "   • Should look like random noise",
    "   • If not, model is missing something!",
    "",
    "💡 Key insight:",
    "   Don't chase perfection -",
    "   some error is unavoidable!"
]

y_pos = 0.87
for line in concepts:
    if line.startswith(('🎲', '📊', '📏', '🎯', '🔍', '💡')):
        weight = 'bold'
        size = 9
    else:
        weight = 'normal'
        size = 8
    ax.text(0.5, y_pos, line, fontsize=size, ha='center', transform=ax.transAxes,
            family='monospace', fontweight=weight)
    y_pos -= 0.036

ax.axis('off')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}04_noise_and_randomness.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ Saved: 04_noise_and_randomness.png")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("✅ SUMMARY: Probability for Machine Learning")
print("=" * 80)
print()

print("🎲 PROBABILITY BASICS:")
print("   • Scale: 0 (impossible) to 1 (certain)")
print("   • All probabilities sum to 1")
print("   • Complement: P(not A) = 1 - P(A)")
print()

print("📊 PROBABILITY DISTRIBUTIONS:")
print("   • Show all possible outcomes and their likelihoods")
print("   • Uniform: all outcomes equally likely")
print("   • Normal: bell-shaped, most common in nature")
print()

print("🔔 NORMAL DISTRIBUTION:")
print("   • Parameters: μ (mean), σ (standard deviation)")
print("   • 68-95-99.7 rule:")
print("     - 68% within μ ± 1σ")
print("     - 95% within μ ± 2σ")
print("     - 99.7% within μ ± 3σ")
print("   • Z-score: z = (x - μ) / σ")
print()

print("🎲 NOISE AND RANDOMNESS:")
print("   • Real data has random variation")
print("   • Usually normally distributed")
print("   • Makes perfect predictions impossible")
print("   • Residuals should look like random noise")
print()

print("🤖 WHY IT MATTERS FOR ML:")
print("   • Understand uncertainty in predictions")
print("   • Detect outliers (values beyond 3σ)")
print("   • Evaluate if residuals look random")
print("   • Make probabilistic predictions")
print("   • Confidence intervals and hypothesis testing")
print()

print("=" * 80)
print("📁 Visualizations saved to:", VISUAL_DIR)
print("=" * 80)
print("✅ 01_probability_fundamentals.png")
print("✅ 02_probability_distributions.png")
print("✅ 03_normal_distribution.png")
print("✅ 04_noise_and_randomness.png")
print("=" * 80)
print()

print("🎓 NEXT STEPS:")
print("   1. Review all visualizations - especially the 68-95-99.7 rule!")
print("   2. Watch StatQuest videos on probability and normal distribution")
print("   3. Practice: Calculate z-scores for different scenarios")
print("   4. You've completed ALL math foundations! 🎉")
print("   5. Ready for: algorithms/linear_regression_intro.py")
print()

print("💡 KEY TAKEAWAY:")
print("   Probability helps us quantify uncertainty.")
print("   Normal distribution appears everywhere in ML - master it!")
print()

print("=" * 80)
print("🎉 MATH FOUNDATIONS COMPLETE!")
print("   You now have all the math needed for machine learning!")
print("=" * 80)
