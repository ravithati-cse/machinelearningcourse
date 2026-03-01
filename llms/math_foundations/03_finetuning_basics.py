"""
Fine-Tuning Large Language Models
===================================
Learning Objectives:
  1. Understand why fine-tuning is needed beyond prompting
  2. Implement Supervised Fine-Tuning (SFT) loss from scratch
  3. Build LoRA (Low-Rank Adaptation) from scratch in NumPy
  4. Compare parameter counts: full fine-tuning vs LoRA vs adapters
  5. Understand RLHF conceptually (reward model + PPO)
  6. Know when to use prompting vs fine-tuning vs LoRA
YouTube: Search "LoRA explained fine-tuning LLM parameter efficient"
Time: ~25 min | Difficulty: Advanced | Prerequisites: Parts 3-6
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "03_finetuning_basics")
os.makedirs(VIS_DIR, exist_ok=True)

# ==========================================================================
print("\n" + "="*70)
print("SECTION 1: THE FINE-TUNING SPECTRUM")
print("="*70)
# ==========================================================================

print("""
When adapting a pre-trained LLM to a specific task, you have many options.
The core trade-off: how many parameters to train vs. how much data you need.

Fine-Tuning Strategy Comparison
--------------------------------
Strategy          | Trainable Params | Data Needed | Speed   | Quality
------------------|------------------|-------------|---------|----------
Prompting         | 0                | 0-10 shots  | Fast    | Good
Prefix Tuning     | ~0.1%            | ~1K         | Fast    | Better
LoRA (r=8)        | ~0.2%            | ~1K-10K     | Medium  | Very Good
Adapters          | ~0.5-1%          | ~1K-10K     | Medium  | Very Good
Full Fine-tuning  | 100%             | ~10K-1M     | Slow    | Best
RLHF              | 100%+RM          | Human labels| Slowest | Aligned
""")

print("When to use each strategy:")
print("-" * 50)
print("""
  PROMPTING        - No model access needed. Quick experiments. General tasks.
                     Best when: GPT-4/Claude already does it well with examples.

  PREFIX TUNING    - Prepend learned 'soft prompt' tokens to the input.
                     Best when: You have ~1K labeled examples, memory-constrained.

  LoRA (r=8)       - Add tiny low-rank update matrices to each attention layer.
                     Best when: Customizing an open model (Llama, Mistral, etc.)
                     with 1K-10K examples. The sweet spot for most use cases.

  ADAPTERS         - Insert small bottleneck modules between transformer layers.
                     Best when: Multi-task: train one adapter per task, swap them.

  FULL FINE-TUNING - Update every weight in the network.
                     Best when: Data is abundant (>100K), need maximum quality.

  RLHF             - SFT + Reward Model + PPO to align with human preferences.
                     Best when: You need safe, helpful, harmless behavior.
                     Used by: InstructGPT, Claude, ChatGPT, Llama 2 Chat.
""")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 2: SUPERVISED FINE-TUNING (SFT) LOSS")
print("="*70)
# ==========================================================================

print("""
SFT Loss: Train on (instruction, response) pairs but compute loss ONLY
on the response tokens — never on the instruction/prompt tokens.

Why? If you compute loss on the instruction too, the model learns to
predict the question, not the answer. That wastes training signal.

Loss formula:
  L_SFT = -1/|R| * sum_{t in R} log P(token_t | token_{<t})

Where R = set of response token positions (masked from instruction).
""")

def sft_loss(logits, targets, response_mask):
    """
    Compute SFT cross-entropy loss only on response tokens.

    Args:
        logits:        (T, V) token prediction logits
        targets:       (T,) ground truth token ids
        response_mask: (T,) boolean, True where we compute loss (response only)
    Returns:
        scalar loss
    """
    T, V = logits.shape
    # Numerically stable softmax
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    # Log probability of correct token at each position
    log_probs = np.log(probs[np.arange(T), targets] + 1e-9)
    # Average only over response positions
    loss = -log_probs[response_mask].mean()
    return loss

# Demo with a tiny vocabulary and sequence
np.random.seed(42)
V = 32          # vocabulary size (tiny)
T = 7           # sequence length (7 tokens)

# Simulated logits from a tiny model
logits = np.random.randn(T, V)

# Token ids (made up):
# "Classify sentiment: " → token ids [5, 12, 3, 8]  (instruction, 4 tokens)
# "positive"             → token ids [17, 22, 6]     (response, 3 tokens)
targets = np.array([5, 12, 3, 8, 17, 22, 6])

# Mask: True only for response tokens (last 3)
instruction_len = 4
response_mask = np.array([False, False, False, False, True, True, True])

print("Demo sequence:")
print(f"  Instruction tokens (positions 0-3): ids={targets[:instruction_len].tolist()}  mask=False")
print(f"  Response tokens    (positions 4-6): ids={targets[instruction_len:].tolist()}  mask=True")
print()

# Loss computed only on response
loss_sft = sft_loss(logits, targets, response_mask)

# For comparison: naively compute loss over ALL tokens
all_mask = np.ones(T, dtype=bool)
loss_naive = sft_loss(logits, targets, all_mask)

print(f"  SFT loss (response tokens only): {loss_sft:.4f}")
print(f"  Naive loss (all tokens):          {loss_naive:.4f}")
print()
print("  Training on instruction would teach the model to predict the question,")
print("  not the answer! SFT masking focuses every gradient step on the response.")
print()
print("  In practice, datasets look like:")
print('    [INST] Classify this review as positive or negative: "Great movie!" [/INST]')
print('    Positive   <- ONLY this token contributes to the loss')

# ==========================================================================
print("\n" + "="*70)
print("SECTION 3: LoRA FROM SCRATCH")
print("="*70)
# ==========================================================================

print("""
LoRA: Low-Rank Adaptation (Hu et al., 2021)
---------------------------------------------
Key insight: pre-trained weight matrices have low "intrinsic rank" — the
meaningful updates during fine-tuning live in a low-dimensional subspace.

Instead of updating W (d_out × d_in), we learn:
    ΔW = (α/r) * B @ A

Where:
  W  is frozen   (d_out × d_in)  — the pre-trained weights
  A  is trained  (r × d_in)      — random Gaussian init
  B  is trained  (d_out × r)     — zero init (so ΔW=0 at start!)

B initialized to zero so the model starts identical to the pre-trained base.
Only A and B are updated; W never changes.

Forward pass: y = xW^T + (α/r) * x @ A^T @ B^T
""")

class LoRALinear:
    """
    LoRA: Low-Rank Adaptation of a Linear Layer.
    W' = W + (alpha/r) * B @ A
    W frozen (d_out x d_in), A trained (r x d_in), B trained (d_out x r).
    B initialized to zeros so initial delta is 0 (preserves pre-trained behavior).
    """
    def __init__(self, d_in, d_out, rank=8, alpha=16.0, seed=42):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(d_out, d_in) * 0.02    # frozen pre-trained weights
        self.A = rng.randn(rank, d_in) * 0.01      # trainable
        self.B = np.zeros((d_out, rank))             # trainable, init 0
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.d_in = d_in
        self.d_out = d_out
        # Stored for backward pass
        self._last_x = None
        self._last_Ax = None

    def forward(self, x):
        """x: (batch, d_in) -> (batch, d_out)"""
        self._last_x = x
        base_out = x @ self.W.T                      # (batch, d_out)
        self._last_Ax = x @ self.A.T                 # (batch, rank)
        lora_out = self._last_Ax @ self.B.T          # (batch, d_out)
        return base_out + self.scale * lora_out

    def backward(self, d_out, lr=0.01):
        """
        d_out: (batch, d_out) — gradient flowing back from loss.
        Updates only A and B; W stays frozen.
        """
        batch = len(self._last_x)
        # Gradient for B: dL/dB = (d_out^T @ last_Ax) / batch * scale
        dB = (d_out.T @ self._last_Ax) / batch * self.scale
        # Gradient for A: propagate through B, then to A
        dAx = (d_out @ self.B) * self.scale           # (batch, rank)
        dA = (dAx.T @ self._last_x) / batch
        # Update only A and B (W stays frozen)
        self.B -= lr * dB
        self.A -= lr * dA

    def param_count(self):
        base = self.W.size
        lora = self.A.size + self.B.size
        return base, lora, f"{100.0*lora/(base+lora):.2f}%"

    def merge(self):
        """Merge LoRA weights into W for efficient inference (no overhead)."""
        return self.W + self.scale * (self.B @ self.A)


# ------ Demo: binary classification with LoRA ------
np.random.seed(0)
d_in, d_out, rank = 64, 64, 8
lora = LoRALinear(d_in, d_out, rank=rank, alpha=16.0)

base_params, lora_params, pct = lora.param_count()
print(f"Layer size: d_in={d_in}, d_out={d_out}, rank={rank}")
print(f"  Full weight matrix W:   {base_params:,} parameters")
print(f"  LoRA matrices A+B:      {lora_params:,} parameters ({pct} of full)")
print(f"  Savings:                {100-float(pct[:-1]):.1f}% fewer trainable params")
print()

# Training task: learn to separate two classes
# Class 0: input has high activation in first half of features
# Class 1: input has high activation in second half of features
batch = 32
losses = []
print("Training LoRA for 100 steps (binary classification)...")

for step in range(100):
    # Generate batch
    labels = (np.random.rand(batch) > 0.5).astype(int)
    x = np.random.randn(batch, d_in) * 0.1
    # Class signal injected
    x[labels == 0, :d_in//2] += 1.5
    x[labels == 1, d_in//2:] += 1.5

    # Forward
    out = lora.forward(x)                             # (batch, d_out)
    # Simple linear readout: mean of first half vs second half
    logit = out[:, :d_out//2].mean(axis=1) - out[:, d_out//2:].mean(axis=1)
    # Sigmoid + BCE loss
    prob = 1 / (1 + np.exp(-np.clip(logit, -10, 10)))
    eps = 1e-7
    loss_val = -np.mean(labels * np.log(prob + eps) + (1-labels) * np.log(1-prob + eps))
    losses.append(loss_val)

    # Backward: gradient of BCE through linear readout
    d_logit = (prob - labels) / batch                 # (batch,)
    d_out_mat = np.zeros((batch, d_out))
    d_out_mat[:, :d_out//2] = d_logit[:, None] / (d_out//2)
    d_out_mat[:, d_out//2:] = -d_logit[:, None] / (d_out//2)

    lora.backward(d_out_mat, lr=0.05)

    if (step+1) % 20 == 0:
        print(f"  Step {step+1:3d}/100  loss={loss_val:.4f}")

print()
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss:   {losses[-1]:.4f}")
print(f"  Loss reduced by {100*(losses[0]-losses[-1])/losses[0]:.1f}%")
print()

# Show merge
W_orig = lora.W.copy()
W_merged = lora.merge()
delta_norm = np.linalg.norm(W_merged - W_orig)
print(f"  |ΔW| after 100 LoRA steps: {delta_norm:.4f}")
print("  Merged weight can replace W for zero-overhead inference.")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 4: PARAMETER EFFICIENCY TABLE")
print("="*70)
# ==========================================================================

print("""
How many parameters does LoRA actually train?

For a weight matrix of size (d × d):
  Full fine-tuning:  d² parameters
  LoRA (rank r):     2 * r * d  parameters  (A: r×d, B: d×r)

For BERT-base hidden size d=768:
""")

d = 768
print(f"  {'rank':>6} | {'LoRA params':>12} | {'% of full':>10} | {'% saved':>10}")
print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")
for r in [2, 4, 8, 16, 32, 64]:
    base = d * d
    lora_p = 2 * r * d
    pct = 100 * lora_p / base
    savings = 100 - pct
    print(f"  {r:>6} | {lora_p:>12,} | {pct:>9.2f}% | {savings:>9.1f}%")

print()
print("Across common model hidden sizes (single attention projection layer):")
print()
print(f"  {'d (hidden)':>12} | {'Full (d²)':>12} | {'LoRA r=8':>12} | {'% of full':>10}")
print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
for d_size, model in [(768, "BERT-base"), (1024, "BERT-large"), (2048, "Llama-7B"), (4096, "Llama-13B/70B"), (8192, "Llama-70B")]:
    full = d_size * d_size
    lora_p = 2 * 8 * d_size
    pct = 100 * lora_p / full
    print(f"  {d_size:>12,} | {full:>12,} | {lora_p:>12,} | {pct:>9.2f}%  ({model})")

print()
print("At rank=8, LoRA trains roughly 0.2% of parameters of each weight matrix.")
print("In practice, LoRA is applied to Q, K, V, and output projections, so")
print("total trainable params for a 7B model ≈ 20-40 million (vs 7 billion).")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 5: RLHF OVERVIEW")
print("="*70)
# ==========================================================================

print("""
RLHF: Reinforcement Learning from Human Feedback
--------------------------------------------------
Used by InstructGPT, ChatGPT, Claude, Llama 2 Chat to align LLMs with
human values: helpful, harmless, and honest.

The 3-stage pipeline:

STAGE 1 — Supervised Fine-Tuning (SFT)
  - Collect high-quality (instruction, response) pairs from human labelers
  - Fine-tune the base LLM on this curated dataset using cross-entropy loss
  - Result: SFT model that can follow instructions but not yet "aligned"
  - InstructGPT used ~13K curated prompts for this stage

STAGE 2 — Reward Model Training
  - Show human labelers K completions for the same prompt
  - They rank them: A > B > C > D (preference pairs extracted)
  - Train a regression head on top of LLM: (prompt, response) → scalar reward
  - Loss: maximize P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
  - The reward model learns what "good" responses look like from humans
  - InstructGPT: ~33K prompts, each with ~4-9 ranked completions

STAGE 3 — PPO Fine-Tuning (Policy Optimization)
  - The SFT model is the policy π_RL that generates responses y given prompt x
  - Objective: maximize expected reward while staying close to SFT model:
      J(π_RL) = E[r(x,y)] - β · KL(π_RL || π_SFT)

  - KL penalty (β ≈ 0.02) is CRITICAL:
    Without it, the model "games" the reward model — finds degenerate outputs
    that get high reward scores but are nonsensical to humans.
    (Like a student who memorizes exactly what the teacher wants to hear.)

  - PPO (Proximal Policy Optimization) clips updates to prevent instability
  - Result: a model that is both capable AND aligned

InstructGPT Data Recipe (OpenAI, 2022):
  - Base model: GPT-3 (175B parameters)
  - SFT data:   13,000 curated (instruction, response) pairs
  - RM data:    33,000 prompts × ~4-9 ranked responses = ~130K comparisons
  - PPO steps:  optimized for hundreds of thousands of steps
  - Outcome:    humans preferred InstructGPT over GPT-3 175B by 85%!
                even though InstructGPT is 100x SMALLER (1.3B params with SFT)

Key lesson: alignment matters more than raw scale for user-facing models.
""")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 6: WHEN TO USE WHAT (DECISION TREE)")
print("="*70)
# ==========================================================================

print("""
Decision guide: choosing your LLM adaptation strategy

START
  └─ Do you have access to the model weights?
       ├─ NO → Use prompting (zero-shot / few-shot / chain-of-thought)
       │         Examples: GPT-4 API, Claude API, Gemini API
       │         Tips: structured prompts, few-shot examples, CoT reasoning
       └─ YES (open weights: Llama, Mistral, Falcon, etc.)
              └─ Do you have labeled training data?
                   ├─ NO → Use prompting or prefix-tuning
                   │         Try structured prompts first; if insufficient,
                   │         prefix-tuning needs no labels (unsupervised)
                   └─ YES
                          └─ How much labeled data?
                               ├─ < 1K examples
                               │    → LoRA (r=4 or r=8)
                               │      Small rank prevents overfitting
                               │      Training time: hours on 1 GPU
                               │
                               ├─ 1K – 100K examples
                               │    → LoRA (r=16) or Adapters
                               │      Larger rank captures more task signal
                               │      Adapters good for multi-task scenarios
                               │
                               └─ > 100K examples
                                    → Full fine-tuning or RLHF
                                      Need quality + alignment? → RLHF
                                      Need best task accuracy?  → Full FT
                                      Requires multi-GPU setup (DeepSpeed, FSDP)

Rule of thumb:
  Start with prompting → if not good enough, try LoRA → then full fine-tuning
  RLHF is only needed when alignment/safety is a core requirement.
""")

# ==========================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)
# ==========================================================================

# ---- VIS 1: Fine-Tuning Spectrum Bar Chart ----
strategies = ["Prompting", "Prefix Tuning", "LoRA (r=8)", "Adapters", "Full Fine-tune", "RLHF"]
trainable_pct = [0.0, 0.1, 0.2, 0.75, 100.0, 100.0]
data_needed = [0.001, 1, 5, 5, 100, 500]   # in K examples (log scale)
colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#795548"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Fine-Tuning Strategy Comparison", fontsize=16, fontweight="bold", y=1.01)

# Left: % trainable params
y_pos = np.arange(len(strategies))
bars1 = ax1.barh(y_pos, trainable_pct, color=colors, edgecolor="white", linewidth=0.8, height=0.6)
ax1.set_xscale("symlog", linthresh=0.05)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(strategies, fontsize=11)
ax1.set_xlabel("Trainable Parameters (%)", fontsize=11)
ax1.set_title("Trainable Parameters\n(log scale)", fontsize=12, fontweight="bold")
ax1.set_xlim(-0.01, 150)
ax1.axvline(x=0, color="gray", linewidth=0.5)
for bar, val in zip(bars1, trainable_pct):
    label = f"{val}%" if val > 0 else "0%"
    ax1.text(max(val + 0.5, 0.05), bar.get_y() + bar.get_height()/2,
             label, va="center", fontsize=9, color="black")
ax1.grid(axis="x", alpha=0.3)
ax1.set_facecolor("#fafafa")

# Right: data needed
bars2 = ax2.barh(y_pos, data_needed, color=colors, edgecolor="white", linewidth=0.8, height=0.6)
ax2.set_xscale("log")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(strategies, fontsize=11)
ax2.set_xlabel("Labeled Examples Needed (K, log scale)", fontsize=11)
ax2.set_title("Training Data Required\n(log scale)", fontsize=12, fontweight="bold")
data_labels = ["~0", "~1K", "~5K", "~5K", "~100K", "~500K"]
for bar, label in zip(bars2, data_labels):
    ax2.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
             label, va="center", fontsize=9, color="black")
ax2.grid(axis="x", alpha=0.3)
ax2.set_facecolor("#fafafa")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_finetuning_spectrum.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/01_finetuning_spectrum.png")

# ---- VIS 2: LoRA Mechanics ----
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("LoRA: Low-Rank Adaptation Mechanics", fontsize=15, fontweight="bold")

# (a) Architecture diagram
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("(a) Weight Decomposition", fontsize=11, fontweight="bold")

# Frozen W block
w_rect = mpatches.FancyBboxPatch((0.5, 2), 4, 6, boxstyle="round,pad=0.1",
                                   facecolor="#BBDEFB", edgecolor="#1565C0", linewidth=2)
ax.add_patch(w_rect)
ax.text(2.5, 5.0, "W\n(frozen)\nd × d", ha="center", va="center",
        fontsize=12, fontweight="bold", color="#0D47A1")

# LoRA A block (thin horizontal)
a_rect = mpatches.FancyBboxPatch((6.0, 7.0), 3.0, 1.0, boxstyle="round,pad=0.1",
                                   facecolor="#FFCCBC", edgecolor="#BF360C", linewidth=2)
ax.add_patch(a_rect)
ax.text(7.5, 7.5, "A  (r × d)", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#BF360C")

# LoRA B block (thin vertical)
b_rect = mpatches.FancyBboxPatch((6.0, 2.0), 1.0, 4.5, boxstyle="round,pad=0.1",
                                   facecolor="#F8BBD0", edgecolor="#880E4F", linewidth=2)
ax.add_patch(b_rect)
ax.text(6.5, 4.25, "B\n(d×r)", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#880E4F")

# Arrows and annotations
ax.annotate("", xy=(6.1, 7.0), xytext=(4.6, 5.5),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
ax.annotate("", xy=(6.1, 4.5), xytext=(6.9, 7.0),
            arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
ax.text(5.1, 6.0, "x @ A^T", fontsize=8, color="#333", rotation=30)
ax.text(7.2, 5.8, "@ B^T", fontsize=8, color="#333", rotation=-70)

ax.text(2.5, 0.8, "+ (α/r) ×", ha="center", fontsize=11, color="#555", style="italic")
ax.text(6.5, 0.8, "ΔW", ha="center", fontsize=12, color="#BF360C", fontweight="bold")
ax.text(5.0, 1.4, "=", ha="center", fontsize=14, color="black", fontweight="bold")
ax.text(9.0, 0.8, "← trainable", ha="center", fontsize=8, color="#BF360C")
ax.text(2.5, 1.4, "← frozen", ha="center", fontsize=8, color="#0D47A1")

# (b) Training loss curve
ax2 = axes[1]
smooth_losses = []
window = 5
for i in range(len(losses)):
    start = max(0, i - window)
    smooth_losses.append(np.mean(losses[start:i+1]))

ax2.plot(range(1, 101), losses, color="#BDBDBD", linewidth=1, alpha=0.7, label="Raw loss")
ax2.plot(range(1, 101), smooth_losses, color="#FF5722", linewidth=2.5, label="Smoothed loss")
ax2.fill_between(range(1, 101), smooth_losses, alpha=0.15, color="#FF5722")
ax2.set_xlabel("Training Step", fontsize=11)
ax2.set_ylabel("BCE Loss", fontsize=11)
ax2.set_title("(b) LoRA Training Loss\n(100 steps, d=64, rank=8)", fontsize=11, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
ax2.set_facecolor("#fafafa")
ax2.text(50, max(losses)*0.95, f"Final loss: {losses[-1]:.3f}", ha="center",
         fontsize=9, color="#333",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc"))

# (c) Parameter counts for various ranks
ax3 = axes[2]
d_vis = 768
ranks_vis = [2, 4, 8, 16, 32, 64]
lora_pcts = [100 * (2*r*d_vis) / (d_vis*d_vis) for r in ranks_vis]
bar_colors = plt.cm.Oranges(np.linspace(0.35, 0.85, len(ranks_vis)))
bars = ax3.bar([str(r) for r in ranks_vis], lora_pcts, color=bar_colors,
               edgecolor="white", linewidth=0.8)
ax3.axhline(y=100, color="#F44336", linewidth=1.5, linestyle="--", label="Full fine-tune (100%)")
ax3.set_xlabel("LoRA Rank (r)", fontsize=11)
ax3.set_ylabel("Trainable Params (% of full)", fontsize=11)
ax3.set_title("(c) LoRA Parameter Efficiency\n(d=768, BERT-base hidden size)", fontsize=11, fontweight="bold")
ax3.legend(fontsize=9)
ax3.set_ylim(0, 115)
ax3.grid(axis="y", alpha=0.3)
ax3.set_facecolor("#fafafa")
for bar, pct in zip(bars, lora_pcts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{pct:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_lora_mechanics.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/02_lora_mechanics.png")

# ---- VIS 3: RLHF Pipeline ----
fig, ax = plt.subplots(figsize=(15, 6))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_title("RLHF: Reinforcement Learning from Human Feedback — 3-Stage Pipeline",
             fontsize=14, fontweight="bold", pad=15)

stage_configs = [
    {
        "x": 0.3, "color_face": "#E3F2FD", "color_edge": "#1565C0",
        "title": "Stage 1: SFT",
        "title_color": "#0D47A1",
        "lines": [
            "Input: (instruction, response)",
            "curated by human labelers",
            "",
            "Loss: cross-entropy",
            "on response tokens only",
            "",
            "Output: SFT Model",
            "(instruction-following)"
        ]
    },
    {
        "x": 5.3, "color_face": "#E8F5E9", "color_edge": "#2E7D32",
        "title": "Stage 2: Reward Model",
        "title_color": "#1B5E20",
        "lines": [
            "Input: (prompt, response A/B/...)",
            "humans rank K completions",
            "",
            "Loss: Bradley-Terry ranking",
            "sigmoid(r_chosen - r_rejected)",
            "",
            "Output: Reward Model (RM)",
            "(prompt, response) → scalar"
        ]
    },
    {
        "x": 10.3, "color_face": "#FFF3E0", "color_edge": "#E65100",
        "title": "Stage 3: PPO Fine-Tuning",
        "title_color": "#BF360C",
        "lines": [
            "Policy: π_RL (starts from SFT)",
            "Env: RM provides reward signal",
            "",
            "Objective:",
            "max E[r(x,y)]",
            "  - β · KL(π_RL || π_SFT)",
            "",
            "Output: Aligned LLM"
        ]
    }
]

box_w, box_h = 4.5, 6.5
for cfg in stage_configs:
    rect = mpatches.FancyBboxPatch(
        (cfg["x"], 0.5), box_w, box_h,
        boxstyle="round,pad=0.2",
        facecolor=cfg["color_face"],
        edgecolor=cfg["color_edge"],
        linewidth=2.5
    )
    ax.add_patch(rect)
    ax.text(cfg["x"] + box_w/2, box_h - 0.2, cfg["title"],
            ha="center", va="top", fontsize=12, fontweight="bold",
            color=cfg["title_color"])
    ax.plot([cfg["x"] + 0.3, cfg["x"] + box_w - 0.3], [box_h - 0.65, box_h - 0.65],
            color=cfg["color_edge"], linewidth=1.2, alpha=0.5)
    for i, line in enumerate(cfg["lines"]):
        ax.text(cfg["x"] + box_w/2, box_h - 1.1 - i*0.65, line,
                ha="center", va="top", fontsize=8.5, color="#333",
                fontstyle="italic" if line.startswith("Loss") or line.startswith("Objective") else "normal")

# Arrows between stages
for x_start in [4.8, 9.8]:
    ax.annotate("", xy=(x_start + 0.45, 3.75), xytext=(x_start, 3.75),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=2.5,
                                mutation_scale=18))

# KL annotation
ax.text(7.5, 0.1, "KL penalty β prevents reward hacking — keeps π_RL close to π_SFT",
        ha="center", va="bottom", fontsize=9, color="#555", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#F9A825", alpha=0.9))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_rlhf_pipeline.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/03_rlhf_pipeline.png")

# ==========================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
# ==========================================================================

print("""
Key Takeaways — Fine-Tuning LLMs
----------------------------------
1. SFT Loss: mask instruction tokens; compute cross-entropy only on responses.
   Otherwise the model learns to predict questions, not answers.

2. LoRA: add two tiny matrices (A and B) alongside frozen pre-trained weights.
   - B initialized to zero → no change at start of training
   - Only A and B are updated; W stays frozen
   - At rank=8 for a 768-dim model: trains 0.26% of full parameters
   - Can merge B@A into W after training for zero-overhead inference

3. Parameter efficiency matters enormously:
   - A 7B model fine-tuned with LoRA (r=16) trains ~40M params instead of 7B
   - Fits on a single consumer GPU (RTX 3090/4090)
   - Often matches full fine-tune quality on task-specific benchmarks

4. RLHF is 3 stages: SFT → Reward Model → PPO
   - KL penalty prevents reward hacking (critical!)
   - This is what makes models "helpful, harmless, honest"

5. Decision rule: prompting first → LoRA if insufficient → full FT if data-rich
""")

print(f"\nVisualizations saved to: {VIS_DIR}/")
print("  01_finetuning_spectrum.png")
print("  02_lora_mechanics.png")
print("  03_rlhf_pipeline.png")
