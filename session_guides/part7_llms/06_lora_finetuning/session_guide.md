# MLForBeginners — Instructor Guide
## Part 7 · Module 06: LoRA — Low-Rank Adaptation for Efficient Fine-Tuning
### Two-Session Teaching Script

> **Prerequisites:** Module 05 complete. They just built a full language model
> from scratch — they understand weight matrices, gradient descent, and the
> scale of modern LLMs. Module 03 (fine-tuning basics) gives them context on
> why fine-tuning matters. Part 6 Transformers for the architectural background.
> **Payoff today:** They will implement LoRA from scratch, understand the exact
> math, and see why it enables fine-tuning on consumer hardware.

---

# SESSION 1 (~90 min)
## "The scale problem — and a brilliant mathematical shortcut"

## Before They Arrive
- Terminal open in `llms/algorithms/`
- Whiteboard ready — draw a thick textbook on the left side of the board
- Have actual sticky notes on the desk (for the analogy)

---

## OPENING (10 min)

Pick up the sticky notes from the desk. Hold them up.

> *"Here's a question: if you needed to add information to a textbook —
> some corrections, some updates, some new context — what would you do?*
>
> *Option 1: Rewrite the entire textbook from scratch. Every page.*
> *Option 2: Add sticky notes to the relevant pages.*
>
> *Sticky notes are LoRA."*

Write on board: LoRA = sticky notes on the textbook

> *"You have a massive pre-trained LLM — GPT-3 at 175 billion parameters.
> You want to fine-tune it to work better for your company's customer support.*
> *Option 1: Retrain all 175 billion parameters. Cost: $100,000+. Time: weeks.*
> *Option 2: Add a small set of 'sticky note' parameters that modify the behavior
> while the original weights stay frozen. Cost: $10–100. Time: hours.*
>
> *That's the LoRA insight. And today you're going to see exactly how it works —
> the linear algebra, the initialization trick, and the forward pass."*

---

## SECTION 1: The Scale Problem — Why Full Fine-tuning Is Impractical (15 min)

Draw on board:
```
GPT-3: 175 BILLION PARAMETERS

Full fine-tuning memory requirements:
  Model weights:         175B × 4 bytes (float32) =   700 GB
  Gradients:             175B × 4 bytes            =   700 GB
  Adam optimizer states: 175B × 8 bytes (2 states) = 1,400 GB
  TOTAL:                                             ~ 2,800 GB

Available hardware:
  NVIDIA A100 (flagship, ~$15K):  80 GB VRAM
  Number of A100s needed:         ~35 GPUs
  Estimated cost per fine-tuning run: $100,000+

CONCLUSION: Full fine-tuning of frontier models is unavailable to
99.9% of practitioners, teams, and companies.
```

> *"This is the real-world constraint. When OpenAI fine-tunes GPT-4, they
> have clusters of thousands of A100s. When your startup needs to fine-tune
> a model for your domain — you need a smarter approach."*

**Ask the room:** *"What are some approaches you could imagine for reducing
fine-tuning cost? Don't think about LoRA yet — just brainstorm."*

Expected answers: freeze most layers and only train the last few, train on
smaller models, use lower precision (float16/int8), distillation.

> *"All of those work. LoRA is a different idea entirely — it doesn't reduce
> which layers you train, it reduces the RANK of the update for each layer.
> And the reason it works is rooted in a fascinating property of how neural
> networks actually learn."*

---

## SECTION 2: The Core Insight — Low-Rank Hypothesis (20 min)

> *"Here's the key research finding from Hu et al. 2021:*
>
> *When you fine-tune a large pre-trained model, the update matrix ΔW
> has low intrinsic rank. The fine-tuning adapts the model in a much
> lower-dimensional subspace than the full parameter space.*
>
> *In other words: even though ΔW has shape d×k (potentially 4096×4096 = 16M
> parameters), the meaningful variation in that matrix can be captured by a
> rank-r approximation where r is just 4, 8, or 16."*

Draw the key equation on the board — make it big:
```
FULL FINE-TUNING:
  W' = W + ΔW
       ↑   ↑
  frozen  d × k parameters (millions)

LORA:
  W' = W + B · A
       ↑   ↑   ↑
  frozen  d×r  r×k
           └─────┘
         r(d+k) parameters (much smaller)

  r = rank (typically 4, 8, or 16)
  r << min(d, k)
```

Work through the parameter count example:
```
Attention projection: d = k = 4096 (LLaMA-7B style)

Full ΔW: 4096 × 4096 = 16,777,216 parameters

LoRA with r=8:
  A: 8 × 4096    =    32,768 parameters
  B: 4096 × 8   =    32,768 parameters
  Total:              65,536 parameters
  Ratio:              65,536 / 16,777,216 = 0.39%

LoRA with r=1:
  Total: 2 × 4096 = 8,192 parameters = 0.05%!
```

> *"Less than half a percent of the original parameters. And the research
> shows: for many tasks, LoRA with r=8 matches or even slightly exceeds
> full fine-tuning quality. The low-rank hypothesis turns out to be true."*

**Ask the room:** *"Why do we initialize B to all zeros? Think about the
forward pass: h = Wx + (α/r) · B · A · x. What does B=0 give us?"*

Expected answer: If B=0, then B·A·x = 0 at initialization. The model starts
identical to the pre-trained model. Training begins from the pre-trained
behavior, not from a random perturbation.

Write on board:
```
INITIALIZATION:
  A ~ Normal(0, 0.02)     ← random (provides varied gradient signal)
  B = zeros               ← zero (LoRA output = 0 at start)

  Combined: B·A = 0 at init
  → Model starts exactly as the pre-trained model
  → Training nudges B·A away from zero gradually
```

---

## SECTION 3: LoRALinear Implementation from Scratch (30 min)

> *"Now let's look at the actual implementation. I'm going to walk through the
> LoRALinear class — forward pass, backward pass, and weight merging.*
>
> *This is the exact same logic used in Hugging Face PEFT and Microsoft's
> original LoRA codebase — just without the PyTorch abstraction."*

Trace through on the board as you explain:
```python
class LoRALinear:
    def __init__(self, d_in, d_out, rank=4, alpha=1.0):
        # Base weights — FROZEN, never updated
        self.W = np.random.randn(d_in, d_out) * 0.02

        # LoRA matrices — TRAINED
        self.A = np.random.randn(rank, d_out) * 0.02  # random init
        self.B = np.zeros((d_in, rank))                # zero init

        self.rank = rank
        self.alpha = alpha           # scaling factor
        self.scale = alpha / rank    # applied to LoRA output

    def forward(self, x):
        base_output = x @ self.W        # W·x (frozen)
        lora_output = x @ self.B @ self.A  # x·B·A (trained)
        return base_output + self.scale * lora_output
```

> *"The forward pass is simple: base output plus scaled LoRA output.*
> *The base weights never receive gradients — they are never updated.*
> *Only A and B receive gradients. That's 0.39% of the parameters."*

Show the backward pass logic:
```python
    def backward(self, dout, x):
        # Gradient for base weights: NONE (frozen)

        # Gradient for LoRA matrices:
        lora_input = x @ self.B         # intermediate
        dA = lora_input.T @ dout * self.scale
        dB = dout @ self.A.T @ ... * self.scale
        return dA, dB
```

**Run the demo:**
```bash
python3 lora_finetuning.py
```

Watch the output for:
- The trainable parameter count (should be tiny vs base model)
- Training loss decreasing
- The final merge step

---

## CLOSING SESSION 1 (5 min)

Board summary:
```
LORA:
  Problem:      Full fine-tuning of 175B params = impractical
  Insight:      Fine-tuning updates have low intrinsic rank
  Solution:     Learn ΔW ≈ B·A where r << min(d,k)

  Init:         B=0 (model starts = pretrained), A=random
  Training:     Only B, A updated — base W frozen
  Parameters:   0.1–1% of full fine-tuning

ANALOGY:
  Full fine-tuning = rewriting the entire textbook
  LoRA = adding sticky notes to key pages
```

**Homework:** If a model has a single weight matrix of shape 2048×2048,
how many parameters does LoRA save with r=4? What about r=1?
Calculate both and write down the percentage savings.

---

# SESSION 2 (~90 min)
## "Training, merging, and the parameter efficiency frontier"

## OPENING (10 min)

> *"Last session we derived the math and built LoRALinear. Today we're going
> to train it, look at the parameter efficiency tables, and understand how
> to merge LoRA weights back into the base model for deployment.*
>
> *The merge step is elegant — it's one of those moments where the math just
> works out beautifully."*

---

## SECTION 1: Training a Classifier with LoRA (20 min)

> *"In the module, we train a 2-layer classifier where only the LoRA adapters
> are trained — the base linear layers are frozen. This simulates fine-tuning
> a large pre-trained model: the base captures general knowledge, LoRA
> adapts it to the specific task."*

Trace through the training loop with the module running:
```bash
python3 lora_finetuning.py
```

Watch together:
- The frozen vs trainable parameter count printed at the start
- Loss per epoch
- Final accuracy

> *"Notice: we're achieving good accuracy with just the LoRA parameters
> being updated. The base model's general representations are doing the
> heavy lifting — LoRA is steering them toward our specific task."*

**Ask the room:** *"If we increase rank r, what happens to accuracy? To
the number of trainable parameters? Is there always a tradeoff?"*

Expected answer: Higher r = more expressiveness = potentially better accuracy,
but more parameters. At some point, performance saturates (the low-rank
hypothesis — most useful information is captured at r=8 or so).

---

## SECTION 2: Merging LoRA Weights for Deployment (20 min)

> *"At inference time, you don't want to keep the base model and the LoRA
> adapters separate. Every forward pass would require two matrix multiplications
> instead of one — slower and more memory.*
>
> *LoRA provides a clean solution: merge the adapters into the base weights.*"

Write on board:
```
BEFORE MERGE (two separate components):
  h = W·x + scale·B·A·x
  ↑                       ↑
  base mat-mul          LoRA mat-mul
  (slow: two operations)

MERGE (algebraically equivalent):
  h = (W + scale·B·A)·x
  = W_merged · x

  W_merged = W + (alpha/rank) · B · A
           ← computed ONCE, stored once

AFTER MERGE:
  h = W_merged · x
  → single matrix multiplication at inference
  → zero additional cost vs un-adapted model!
```

> *"This is the beauty of LoRA: during training, you pay a tiny parameter
> overhead. At deployment, you merge the adapters into the base weights
> and the merged model is IDENTICAL in size and speed to the original.*
>
> *No inference overhead. That's why it's preferred in production over
> other adapter methods that attach extra modules."*

Show in code:
```python
def merge_lora(W_base, B, A, alpha, rank):
    delta_W = (alpha / rank) * B @ A
    return W_base + delta_W   # ← merged, shape unchanged
```

---

## SECTION 3: Parameter Efficiency Tables (20 min)

Draw the full efficiency table:
```
MODEL SIZE vs LoRA PARAMETERS (r=8, single attention projection per layer)

Model      Layers  d_model  Params (base)  LoRA params   Ratio
----------------------------------------------------------------
Tiny       2       64       ~50K           ~4K           8.0%
GPT-2 sm   12      768      124M           ~300K         0.24%
GPT-2 xl   48      1600     1.5B           ~2.5M         0.17%
LLaMA-7B   32      4096     7B             ~8M           0.11%
GPT-3      96      12288    175B           ~100M         0.06%

Key insight: As models get LARGER, LoRA's efficiency IMPROVES.
The ratio decreases — you adapt a larger model with proportionally
fewer parameters. That's counter-intuitive but explains why LoRA
is most impactful for the largest models.
```

> *"For LLaMA-7B — a model you can run on a laptop with quantization —
> LoRA fine-tuning with r=8 updates 8 million parameters instead of 7 billion.*
> *That's the difference between needing a $15,000 GPU cluster and doing it
> on a $1,000 consumer GPU."*

**Ask the room:** *"If LoRA adds so little capacity — how can it match full
fine-tuning? Isn't it doing less?"*

Expected answer: The pre-trained model already contains the general knowledge.
Fine-tuning doesn't teach the model everything from scratch — it primarily
adjusts weights by small amounts. Those small adjustments happen to be
low-rank. So LoRA's low-rank approximation captures the same adjustments
with far fewer parameters.

---

## SECTION 4: Viewing the Visualizations (15 min)

```bash
open llms/visuals/lora_finetuning/
```

Walk through together:
- LoRA architecture diagram: base W (frozen) + B·A (trained)
- Parameter efficiency bar chart: full fine-tuning vs LoRA ranks 1, 4, 8, 16
- Training curves: loss and accuracy over epochs
- Weight update analysis: singular values of ΔW (should be low-rank)

For the singular value plot:

> *"This is the evidence for the low-rank hypothesis. Look at the singular
> values of the trained LoRA update matrix B·A. The first few are large —
> those capture the meaningful adaptation. The rest are near zero.*
> *This is what 'low rank' means visually: most variance is explained by
> a few directions in the parameter space."*

---

## CLOSING SESSION 2 (5 min)

Board summary:
```
COMPLETE LORA PICTURE:
  Math:      ΔW ≈ B·A,  r << d
  Init:      B=0, A=random → model starts at pretrained baseline
  Training:  Only B, A updated (base W frozen, no gradients)
  Merge:     W_merged = W + (α/r) · B·A (done once, before deployment)
  Inference: W_merged · x — identical speed to original model
  Efficiency: 0.1–1% of full fine-tuning parameters

IN PRACTICE (Hugging Face PEFT):
  from peft import LoraConfig, get_peft_model
  config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"])
  peft_model = get_peft_model(base_model, config)
  # only LoRA params have requires_grad=True
```

**Homework — from the module:** Try LoRA with ranks 1, 4, 8, 16.
Record accuracy and trainable parameter count for each.
At what rank does accuracy stop improving for this task?

---

## INSTRUCTOR TIPS

**"Why not just freeze the early layers and fine-tune the last few?"**
> *"That's a valid strategy — 'partial fine-tuning'. It works but has a
> different tradeoff. You get full expressiveness in the last layers, but the
> frozen layers may create a bottleneck. LoRA spreads adaptation across all
> layers uniformly, which tends to work better for tasks requiring nuanced
> adjustments throughout the network."*

**"What does alpha do? I thought rank controlled everything."**
> *"Alpha scales the LoRA contribution: output = Wx + (alpha/rank) · BAx.
> It's a hyperparameter like learning rate. If alpha=rank, the scale factor
> is 1 and LoRA adds at full strength. Lower alpha = more conservative update.
> In practice, alpha=2×rank is common. It decouples rank (capacity) from
> update magnitude (strength)."*

**"When should I use LoRA over full fine-tuning?"**
> *"Simple rule: if you can fit full fine-tuning in your budget (GPU-hours,
> memory), full fine-tuning may give marginally better results. If you
> can't — which is almost always the case for models >1B parameters —
> LoRA is the go-to. For models under 100M params, full fine-tuning is
> usually practical."*

**"Can LoRA be applied to non-attention layers?"**
> *"Yes! Any linear (weight matrix) layer can get LoRA adapters — attention
> projections (Q, K, V, output), MLP layers, even embedding layers.
> In practice, applying LoRA only to Q and V projections (as in the original
> paper) often gives 90%+ of the benefit at half the LoRA parameter count."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening + sticky note analogy      10 min
├── Scale problem — why GPT-3 costs    15 min
├── Low-rank hypothesis + math         20 min
├── LoRALinear implementation          30 min
└── Close + parameter savings homework  5 min

SESSION 2  (90 min)
├── Opening bridge                     10 min
├── Training classifier with LoRA      20 min
├── Merging weights for deployment     20 min
├── Parameter efficiency tables        20 min
├── Viewing visualizations             15 min
└── Close + rank ablation homework      5 min
```

---
*MLForBeginners · Part 7: LLMs · Module 06*
