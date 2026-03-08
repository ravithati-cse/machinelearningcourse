# MLForBeginners — Instructor Guide
## Module 3: Fine-Tuning Basics  ·  Two-Session Teaching Script

> **Who this is for:** You, teaching close friends who completed Modules 1-2 and Parts 1-6.
> **Tone:** Measured, conceptually precise. This is the most technically dense module
> in the math foundations series. Go slowly on the math. The LoRA analogy is gold —
> use it early and often.
> **Goal by end of both sessions:** Everyone understands when and why to fine-tune,
> can compute SFT loss from scratch, understands LoRA mathematically, and has a
> clear decision tree for choosing a fine-tuning strategy.

---

# ─────────────────────────────────────────────
# SESSION 1  (~90 min)
# "When prompting isn't enough — and the spectrum of options"
# ─────────────────────────────────────────────

## Before They Arrive (10 min setup)

**On your laptop, have ready:**
- Terminal open in `MLForBeginners/llms/math_foundations/`
- `03_finetuning_basics.py` visible but not run
- Whiteboard ready — you will fill it heavily this session
- Optional: print the strategy comparison table from Section 1 for reference

**Key pedagogical note:** The audience knows gradient descent and cross-entropy
from Parts 3-4. They know attention layers from Part 6. Don't re-explain those.
The new material is: what changes during fine-tuning, what stays frozen,
and why the low-rank math of LoRA is so elegant.

---

## OPENING  (10 min)

### Hook — The textbook analogy

Say this naturally:

> *"Let me give you an analogy. You have the world's most comprehensive
> medical textbook — every disease, every treatment, every interaction.*
>
> *But your hospital needs it re-written in plain language that patients
> can understand. And it needs to always recommend consulting a doctor.
> And it should never give advice that could be harmful.*
>
> *Option 1: Re-write the entire textbook from scratch.*
> *— Takes years. Costs millions. You lose everything that was already good.*
>
> *Option 2: Write margin notes on the existing textbook.*
> *— Takes days. The original knowledge stays intact. You just steer it.*
>
> *Fine-tuning an LLM is Option 2.*
> *And LoRA — which we'll get to today — is the art of writing the
> smallest possible margin notes that change the behavior you need.*"

---

## SECTION 1: The Fine-Tuning Spectrum  (25 min)

> *"There are many ways to adapt a pre-trained LLM. They form a spectrum
> from 'touch nothing' to 'update everything'."*

**Write this table on the board slowly. Explain each row:**

```
Strategy         Trainable %   Data Needed   Cost      Quality
──────────────────────────────────────────────────────────────────
Prompting        0%            0-10 examples  Free      Good
Prefix Tuning    ~0.1%         ~1K            Fast      Better
LoRA (r=8)       ~0.2%         1K-10K         Medium    Very Good
Adapters         ~0.5-1%       1K-10K         Medium    Very Good
Full Fine-tuning 100%          10K-1M+        Expensive Best
RLHF             100%+RM       Human labels   Very exp. Aligned
──────────────────────────────────────────────────────────────────
```

> *"Walk me through this. Prompting — you've done that. Zero trainable parameters.*
>
> *Full fine-tuning — every weight gets updated. Same as training from scratch,
> except you START from a very good initialization.*
>
> *The problem: GPT-3 has 175 billion parameters.*
> *Storing gradients for all of them at float32 = 700GB of memory.*
> *Adam optimizer states = another 1.4TB.*
> *You need an industrial GPU cluster and $100K just to fine-tune once.*
>
> *LoRA solves this. We get to the math in Session 2.*
> *For now — 0.2% of parameters, almost as good as full fine-tuning.*"

### When to use each

> *"Decision rule:"*

**Write on board:**

```
DECISION TREE:
  Model already does it well with examples?
  → PROMPTING

  Need custom format / domain vocabulary / consistent style?
  Have 1K-10K labeled examples?
  → LoRA

  Need absolute best quality?
  Have 100K+ examples and a GPU cluster?
  → Full Fine-Tuning

  Need the model to be safe, helpful, harmless?
  → RLHF (after SFT)
```

**Ask the room:**
> *"You work at a law firm. You want a model to draft contract clauses
> in your firm's specific style, using your precedent library.*
> *You have 5,000 example clauses. What do you use?"*

Answer: LoRA. 5K examples is in the sweet spot. Style/format changes are
exactly what LoRA handles well. Full fine-tuning would be overkill.

---

## SECTION 2: Supervised Fine-Tuning (SFT) Loss  (25 min)

> *"Before we talk about LoRA, let's understand the loss we're optimizing.*
>
> *Pre-training loss:*
> *Given a sequence of text, predict every next token.*
> *Loss is computed over ALL token positions.*
>
> *SFT loss is different. We have (instruction, response) pairs.*
> *We ONLY compute loss on the response tokens.*"

**Write on board:**

```
SEQUENCE:  [INSTRUCTION TOKENS] [RESPONSE TOKENS]
            ─────────────────────  ────────────────
            "Translate to French:  "Je mange une
            I eat an apple"        pomme"

PRE-TRAINING: compute loss on ALL positions (instruction + response)

SFT:          compute loss ONLY on response tokens
              mask the instruction tokens out of the loss

Why?
  If we include instruction tokens in the loss,
  the model learns to predict "Translate to French: ..."
  → learns to ask questions, not answer them
  → wastes training signal
```

**Show the math:**

```
SFT Loss:
  L_SFT = -1/|R| × Σ_{t ∈ R} log P(token_t | token_{<t})

  Where:
    R = set of response token positions
    |R| = number of response tokens (NOT total sequence length)
    token_{<t} = all tokens before position t (instruction + prior response)
```

> *"The model still SEES the instruction during forward pass.*
> *It just doesn't get trained to predict those tokens.*
> *The instruction is context, not target."*

**Common confusion — address directly:**

> *"Someone will ask: but if we don't compute loss on the instruction,
> does the model learn to understand instructions?*
>
> *Yes — because the response is conditioned on the instruction.*
> *The model must attend to the instruction to predict good response tokens.*
> *It learns instruction comprehension as a side effect of learning
> to produce good responses."*

**Walk through a numerical example:**

```
Suppose response is 5 tokens: ["Je", "mange", "une", "pomme", "."]
Logits at each position (probability of correct token):

Position  Token    log P(correct)
────────────────────────────────
   1      "Je"     -0.4   (P = 0.67)
   2      "mange"  -0.8   (P = 0.45)
   3      "une"    -0.3   (P = 0.74)
   4      "pomme"  -1.2   (P = 0.30)
   5      "."      -0.1   (P = 0.90)

L_SFT = -(-0.4 + -0.8 + -0.3 + -1.2 + -0.1) / 5
       = -(−2.8) / 5
       = 0.56

Perplexity = exp(0.56) = 1.75
```

> *"Run this a few thousand times on real instruction-response pairs,
> backpropagate, update weights — and you get a model that follows instructions.*
>
> *That's SFT. One forward pass. One backward pass. Repeat.*"

---

## CLOSING SESSION 1  (10 min)

Write on board:

```
SESSION 1 RECAP
═══════════════════════════════════════════════════════════
Fine-tuning spectrum: Prompting → LoRA → Full FT → RLHF

SFT Loss:
  - (instruction, response) pairs
  - Compute loss ONLY on response tokens
  - Instruction is context, not target
  - Same cross-entropy loss you know from Part 3

Decision rule:
  0-10 examples    → prompting
  1K-10K examples  → LoRA
  100K+ examples   → full fine-tuning
  Safety alignment → RLHF
═══════════════════════════════════════════════════════════
```

> *"Next session: the math behind LoRA.*
> *And I promise — once you understand it, it will seem obvious.*
> *The insight is almost embarrassingly simple once you see it."*

---
---

# ─────────────────────────────────────────────
# SESSION 2  (~90 min)
# "LoRA: elegant math, tiny footprint, big results"
# ─────────────────────────────────────────────

## Opening  (5 min)

> *"Before we write any code today — let's do the intuition.*
>
> *Imagine you have a textbook with 10 million sentences.*
> *To teach a new skill, do you need to re-write all 10 million sentences?*
> *Or can you add a thin appendix of 50 pages that steers the reader?*
>
> *LoRA is the 50-page appendix.*"

---

## SECTION 1: LoRA Mathematics  (30 min)

> *"Here's the key insight. When you fine-tune a model,*
> *the change in each weight matrix ΔW is surprisingly low-rank.*
>
> *'Low-rank' means: even though ΔW has millions of entries,
> you can capture most of its behavior with a product of two much smaller matrices."*

**Write on board step by step:**

```
Standard fine-tuning:
  W' = W + ΔW
  Forward pass: h = W'·x = W·x + ΔW·x

  W is (d × k) — for LLaMA-7B attention: (4096 × 4096) = 16M parameters
  ΔW is also (d × k) = 16M more parameters to store and optimize

LoRA approximation:
  ΔW ≈ B · A
  where B is (d × r) and A is (r × k), r << min(d, k)

  Forward pass: h = W·x + (α/r) · B · A · x
                    └── frozen ──┘  └─── trainable ───┘
```

**Show the parameter count:**

```
Full ΔW:  d × k   = 4096 × 4096 = 16,777,216 parameters

LoRA A:   r × k   = 8 × 4096    =     32,768 parameters
LoRA B:   d × r   = 4096 × 8   =     32,768 parameters
─────────────────────────────────────────────────────
LoRA total:         65,536 parameters  =  0.39% of full update
```

> *"You reduce the trainable parameters by 99.6% with comparable quality.*
> *That means training fits on a single GPU instead of a server farm."*

**Explain the initialization choices — these matter:**

```
Why B is initialized to ZERO:
  At training start: ΔW = B·A = 0·A = 0
  → model starts EXACTLY where pre-training left it
  → no random disruption of hard-won knowledge

Why A is initialized from N(0, σ²):
  B is zero, so B's gradient is: ∂L/∂B = grad_output · A^T
  If A were also zero, this gradient would be zero → no learning
  A must be non-zero to give B non-zero gradients

Why the α/r scaling factor:
  Makes the update magnitude independent of your choice of r
  → changing r from 4 to 16 doesn't require retuning learning rate
```

**Draw on board:**

```
BASE MODEL LAYER (frozen)
┌─────────────────────────┐
│  W (4096 × 4096)        │  ← never touched during fine-tuning
└─────────────────────────┘
           │
           + (add)
           │
   ┌───────────┐
   │  A (r×k) │  ← trainable: initialized randomly
   └───────────┘
           │ (matrix multiply)
   ┌───────────┐
   │  B (d×r) │  ← trainable: initialized to ZERO
   └───────────┘
           │
         ΔW = B·A (0.39% of parameters)
```

**Ask the room:**
> *"Why is it called LOW-RANK? What does rank mean here?"*

Walk them through: rank of a matrix = number of linearly independent rows/columns.
ΔW has rank at most r. We're constraining the update to live in a low-dimensional
subspace of the full parameter space.

---

## SECTION 2: Run the Script  (20 min)

```bash
python3 03_finetuning_basics.py
```

**Walk through output as it prints. Key moments:**

- **Strategy comparison table**: "See how LoRA r=8 gives 0.39% trainable params
  but competes with full fine-tuning on most benchmarks."

- **SFT loss computation**: "Watch the response mask — True for response tokens,
  False for instruction tokens. Loss only computed where True."

- **LoRA from scratch**: "This is the real B·A forward pass. NumPy only.
  Verify: B starts at all zeros, update is exactly ΔW = B·A."

**Open the visuals:**
- Show the LoRA architecture diagram (B and A matrices annotating W)
- Show the parameter efficiency chart across different ranks
- Show the RLHF pipeline diagram

---

## SECTION 3: RLHF Overview  (15 min)

> *"One more piece. Once you have SFT, the model follows instructions.*
> *But it's not necessarily helpful, harmless, or honest.*
> *It might follow the instruction 'How do I pick a lock?' very well.*
>
> *RLHF is how you align the model to human values.*"

**Draw on board:**

```
RLHF PIPELINE:
══════════════════════════════════════════════════

STEP 1: Supervised Fine-Tuning (SFT)
  (instruction) → SFT model → response
  Train on human-written ideal responses

STEP 2: Train a Reward Model (RM)
  Human is shown two responses A and B for same prompt
  Human picks: "A is better" or "B is better"
  Train a classifier: RM(prompt, response) → scalar score

STEP 3: Optimize with PPO (Proximal Policy Optimization)
  Use the reward model as the "judge"
  LLM generates response → RM scores it → update LLM weights
  Repeat until LLM maximizes RM score

RESULT: A model that generates responses humans prefer
        Used by: InstructGPT, ChatGPT, Claude, LLaMA-2 Chat
```

> *"The tricky part: you can overfit the reward model.*
> *If the LLM gets too good at maximizing the reward model's score,
> it starts finding weird loopholes — technically scoring high but
> actually producing bad responses.*
>
> *PPO has a KL penalty that prevents the policy from drifting too far
> from the SFT model. You're not trying to maximize reward alone —
> you're trying to maximize reward while staying sane."*

---

## CLOSING SESSION 2  (10 min)

Write on board:

```
COMPLETE FINE-TUNING SUMMARY
═══════════════════════════════════════════════════════════
SFT Loss:    cross-entropy on response tokens only

LoRA math:
  ΔW ≈ B·A  where B∈R^{d×r}, A∈R^{r×k}, r << min(d,k)
  B initialized to zero (no disruption at start)
  A initialized randomly (gives non-zero gradients)
  Saves 99%+ of parameters vs full fine-tuning

RLHF:
  SFT → Reward Model (human preferences) → PPO optimization
  Used to make models helpful, harmless, honest

Decision Tree:
  Prompting → LoRA → Full FT → RLHF
  (more data and compute needed at each step →)
═══════════════════════════════════════════════════════════
```

> *"Next: Retrieval-Augmented Generation.*
> *Fine-tuning teaches the model new behavior.*
> *RAG teaches the model new FACTS.*
> *And it works without any training at all."*

---

## INSTRUCTOR TIPS & SURVIVAL GUIDE

### When People Get Confused

**"Why does LoRA work if the rank is so low?"**
> *"Empirical research shows that the intrinsic rank of fine-tuning updates
> is surprisingly small. The model was already initialized close to a good
> solution by pre-training. The fine-tuning update just steers it slightly —
> that steering lives in a low-dimensional space."*

**"What's the difference between LoRA and adapters?"**
> *"Adapters insert tiny new modules (bottleneck layers) between transformer
> blocks. LoRA adds low-rank matrices to EXISTING weight matrices.
> Both achieve similar parameter efficiency, but LoRA can be 'merged' into
> the original weights at inference time — zero added latency.
> Adapters always add a small overhead."*

**"Can you use LoRA with RLHF?"**
> *"Yes — LoRA + RLHF is common. You SFT with LoRA, then RLHF with LoRA.
> This makes the entire pipeline much cheaper. The key insight:
> you're doing a small adaptation from a strong pre-trained base,
> not training from scratch."*

### Energy Management

- **Session 2 is math-heavy.** If anyone looks glazed after the LoRA math section,
  do the live demo BEFORE the RLHF section. Running code re-energizes.
- **The LoRA architecture diagram is critical.** Draw it slowly, label every dimension.
  This is the "aha" moment for most people.
- **If they want to go deeper:** Point to `06_lora_finetuning.py` in the algorithms
  section — it implements the full forward and backward pass from scratch.

---

# QUICK REFERENCE — Session Timing

```
SESSION 1  (90 min)
├── Opening hook                  10 min
├── Section 1: FT spectrum        25 min
├── Section 2: SFT loss           25 min
├── Numerical example walkthrough 20 min
└── Close + preview               10 min

SESSION 2  (90 min)
├── Opening intuition              5 min
├── Section 1: LoRA math          30 min
├── Section 2: Live demo          20 min
├── Section 3: RLHF overview      15 min
├── Visuals walkthrough           10 min
└── Close + next steps            10 min
```

---

*Generated for MLForBeginners — Module 03 · Part 7: Large Language Models*
