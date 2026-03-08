# MLForBeginners — Instructor Guide
## Module 1: How LLMs Work  ·  Two-Session Teaching Script

> **Who this is for:** You, teaching close friends who have completed Parts 1-6.
> **Tone:** Excited, collegial — you are now going to explain the thing they have been
> using for months. They know DNNs, CNNs, NLP, and the full Transformer architecture.
> **Goal by end of both sessions:** Everyone understands the decoder-only GPT architecture,
> can explain tokenization, scaling laws, and perplexity, and has watched a live
> BPE tokenizer and next-token predictor run.

---

# ─────────────────────────────────────────────
# SESSION 1  (~90 min)
# "Autocomplete on steroids — the decoder-only transformer revealed"
# ─────────────────────────────────────────────

## Before They Arrive (10 min setup)

**On your laptop, have ready:**
- Terminal open in `MLForBeginners/llms/math_foundations/`
- `01_how_llms_work.py` loaded (do NOT run yet — save it for the live demo)
- `visuals/01_how_llms_work/` folder ready to open after the run
- The model scale table printed or written on a slip of paper for reference
- Whiteboard or large notepad, several markers

**Room vibe:** This is the capstone topic. Make it feel like a reveal. You've been
building toward this for six parts. Start on time, open with energy.

---

## OPENING  (10 min)

### Hook — The thing they already know

Say this naturally, not rushed:

> *"Quick question. How many of you have used ChatGPT, or Claude, or Gemini
> in the last week? Hands up."*

Everyone raises their hand.

> *"Now — how many of you actually know what's happening inside?
> Like, mechanically, what is the computer doing when you hit send?"*

Pause. Most hands go down.

> *"Today that changes. You've been using this thing for months.
> After today you will understand exactly — not approximately, EXACTLY —
> how it works. The architecture, the training objective, even the math.
> Because you already know all the pieces. We built them in Parts 3 through 6.*
>
> *A transformer. You built that. Self-attention. You built that.
> Cross-entropy loss. You've been computing that since Part 3.*
>
> *An LLM is just those pieces, assembled at a ridiculous scale,
> trained on the entire internet, with one single objective:
> predict the next word.*
>
> *That's it. That is the whole secret.*
>
> *Autocomplete on steroids."*

**Write on the whiteboard:**

```
LLM = Decoder-Only Transformer
      + Massive Data
      + One training objective: predict next token
```

---

## SECTION 1: What is a Large Language Model?  (20 min)

### The training objective

> *"Let's be precise about that training objective.*
>
> *The model sees a sequence of tokens — words, or pieces of words.
> It tries to predict the NEXT token. Every single training step.*
>
> *Formally — and I know you've seen this before:"*

**Write on the board:**

```
Loss = -1/T × Σ log P(t_i | t_1, t_2, ..., t_{i-1})

t_i        = the next token we're predicting
t_1 ... t_{i-1} = everything that came before
T          = total number of tokens in this sequence
log P(...)  = how confident we were about the right answer
```

> *"That's it. Cross-entropy loss. Same formula you used in Part 3 for
> MNIST. Same formula. Just applied to text, at trillion-token scale.*
>
> *The model sees: 'The cat sat on the ___'
> It outputs a probability distribution over every word in its vocabulary.
> 'mat' gets high probability, 'Jupiter' gets very low probability.
> We penalize it for being wrong and update the weights.*
>
> *Do that 10 trillion times across all the text ever written on the internet.
> What do you get? Something that can write code, pass the bar exam,
> explain quantum physics — from learning to predict the next word."*

**Ask the room:**
> *"Does it feel like that should be enough? Like — just predicting words?
> Shout out your intuition."*

Let them debate for 2-3 minutes. This is a genuinely interesting question.
Some will say no. Guide them:

> *"The argument for 'yes' is this: to predict the next word in a
> medical textbook perfectly, you have to understand medicine.
> To predict the next word in code, you have to understand logic.
> The task forces the model to build an internal model of the world."*

### The architecture: Decoder-only Transformer

> *"Now — you built a full Encoder-Decoder Transformer in Part 6.
> LLMs drop the encoder. They only keep the decoder, slightly modified."*

**Draw on the whiteboard:**

```
ENCODER-DECODER (Part 6)          DECODER-ONLY (GPT / LLMs)
────────────────────────          ──────────────────────────
Input → Encoder → Context         Input tokens
              ↓                         ↓
        Decoder → Output          Embedding + Positional Encoding
                                         ↓
                                  ┌─────────────────────┐
                                  │  Transformer Block × N │
                                  │  ┌───────────────┐   │
                                  │  │ Masked Self-  │   │
                                  │  │ Attention     │   │
                                  │  └───────────────┘   │
                                  │  ┌───────────────┐   │
                                  │  │  Feed-Forward │   │
                                  │  └───────────────┘   │
                                  └─────────────────────┘
                                         ↓
                                  Linear + Softmax
                                         ↓
                                  P(next token)
```

> *"The KEY difference: MASKED self-attention.*
>
> *In Part 6's encoder, every token could attend to every other token —
> past AND future. Great for understanding a sentence you already have.*
>
> *Here, at training time, each token can ONLY attend to tokens before it.
> The future is masked out. This is called causal masking.*
>
> *Why? Because at inference time, you're generating one token at a time.
> You don't have the future yet. So the model must learn without it."*

**Pause and ask:**
> *"Think about Part 6. Which part of what we built does this remind you of?"*

They should connect it to the GPT decoder module from Part 6.

---

## SECTION 2: Tokenization — From Text to Numbers  (20 min)

> *"Before the transformer sees anything, text has to become numbers.
> You know about word embeddings from Part 5. But LLMs don't use whole words.*
>
> *They use SUBWORD tokens. Let me show you why."*

**Write on board:**

```
Problem with word-level tokens:
  "running", "runner", "runs", "ran"
  → 4 separate vocabulary entries
  → no connection between them

Problem with character-level tokens:
  "the" → t, h, e  (3 tokens for a tiny word)
  Long sequences → slow, expensive attention

Solution: Subword tokens (BPE)
  "running" → "run" + "ning"
  "runner"  → "run" + "ner"
  "runs"    → "run" + "s"
  Common root, shared!
```

> *"The algorithm is called Byte-Pair Encoding, BPE. Here's the intuition:*
>
> *Start with every character as its own token.*
> *Find the most common pair of adjacent tokens.*
> *Merge them into a single new token.*
> *Repeat until you have a vocabulary of the target size — say 50,000 tokens."*

**Write the worked example:**

```
Corpus: "low low low lower lowest"

Start:  l o w   l o w   l o w   l o w e r   l o w e s t
Step 1: Most frequent pair: (l, o) → "lo"
        lo w   lo w   lo w   lo w e r   lo w e s t
Step 2: Most frequent pair: (lo, w) → "low"
        low   low   low   low e r   low e s t
Step 3: Pair (low, e) → "lowe"
        low   low   low   lowe r   lowe s t

Final vocabulary includes: l, o, w, e, r, s, t, lo, low, lowe, lower...
```

> *"Real LLMs do this with a vocabulary of 50K-100K tokens.*
> *GPT-4 uses ~100K. That's how 'gpt' becomes one token, but
> 'ChatGPT' might be two: 'Chat' + 'GPT'."*

**Run the script — Section 1 & 2:**

```bash
python3 01_how_llms_work.py
```

Point at the BPE output as it prints. Then open the generated visuals folder.

> *"See that table — it shows how merge rules compress the corpus.
> And that visualization shows the token boundary map.
> Before our transformer sees a single word — this has already happened."*

---

## SECTION 3: The Model Scale Table  (15 min)

**Write on board (or point to printout):**

```
Model        Parameters    Training Tokens    Key Capability
─────────────────────────────────────────────────────────────
GPT-1        117M          ~1B                Basic coherence
GPT-2        1.5B          ~10B               Coherent paragraphs
GPT-3        175B          300B               Few-shot in-context learning
Chinchilla   70B           1.4T               Compute-optimal training
LLaMA-2      70B           2T                 Open, instruction-tuned
GPT-4        ~1T est.      ~13T est.          Complex reasoning, multimodal
```

> *"Notice something weird here. Chinchilla has FEWER parameters than GPT-3
> but is trained on FAR more data. And it actually performs better.*
>
> *This shocked the field in 2022. Most big labs had been making models
> bigger — more parameters — assuming that was always the lever to pull.*
>
> *The Chinchilla paper showed: for a given compute budget, you want to
> roughly match parameter count and token count.*
>
> *The rule of thumb: train on at least 20 tokens per parameter.*
> *GPT-3: 175B params × 20 = 3.5T tokens needed. But it only saw 300B.*
> *It was starved of data.*"

**Ask the room:**
> *"If you were starting an LLM project right now with a fixed compute budget,
> what's the first question you'd ask? Bigger model or more data?"*

Guide toward: it depends on your budget. The Chinchilla scaling law gives you the
optimal ratio. More parameters alone won't save you.

---

## CLOSING SESSION 1  (10 min)

### What we covered

Write on board:

```
SESSION 1 RECAP
───────────────────────────────────────────────────────
LLM = Decoder-Only Transformer
      + one objective: predict next token (cross-entropy)
      + causal masking (can't look at the future)

Tokenization = BPE subword tokens
      + merge most frequent character pairs
      + result: ~50K-100K token vocabulary

Scale matters:
      + Chinchilla: 20 tokens per parameter is compute-optimal
      + Bigger model is NOT always better if undertrained
```

> *"Before next session — spend 5 minutes playing with a tokenizer.*
> *Go to platform.openai.com/tokenizer. Type in some English text.*
> *Try 'ChatGPT', your name, some Python code.*
> *See where the word boundaries land.*
>
> *It's weirder than you expect. And that weirdness matters —
> it directly affects how the model thinks about things."*

---
---

# ─────────────────────────────────────────────
# SESSION 2  (~90 min)
# "Emergent abilities, perplexity, and the training pipeline"
# ─────────────────────────────────────────────

## Opening  (10 min)

### Tokenizer show-and-tell

> *"Who played with the tokenizer? What did you find?"*

Go around the room. Affirm everything. Common interesting findings:
- 'ChatGPT' is 3 tokens: 'Chat', 'G', 'PT'
- Spaces are often part of the token: ' the' is one token
- Code has very different tokenization than prose

> *"Today we go deeper: emergent abilities, perplexity as a metric,
> and the full pipeline from raw pre-training to the ChatGPT you use."*

---

## SECTION 1: Perplexity — What's the Model's Score?  (15 min)

> *"So we train with cross-entropy loss. How do we report it to humans?*
>
> *Perplexity. It's just the exponent of the loss:"*

**Write on board:**

```
Loss (L)    = -1/T × Σ log P(t_i | context)
Perplexity  = exp(L)

Intuition: the "effective number of choices" the model is confused among.

Perplexity = 1    → perfect prediction every time (impossible in practice)
Perplexity = 10   → model acts as if choosing uniformly from 10 options
Perplexity = 100  → very confused, 100-option dice roll
Perplexity = 50K  → random guessing from full vocab (terrible)
```

> *"A well-trained LLM on English text gets perplexity around 10-20.*
> *That means for every token, it effectively narrows 50,000 choices
> down to about 15 plausible next tokens. Pretty good.*
>
> *On code, perplexity is much lower — code is more structured,
> more predictable than natural language."*

**Ask the room:**
> *"What would perplexity look like on random noise — a sequence of
> random tokens? Think about it mathematically."*

Answer: perplexity approaches vocabulary size (~50K) because every prediction
is equally wrong.

---

## SECTION 2: Emergent Abilities  (20 min)

> *"Here's one of the most surprising results in AI research.*
>
> *You'd expect that as a model gets bigger, all its abilities
> improve gradually and proportionally. A model twice as big is
> twice as good at everything.*
>
> *That's NOT what happens."*

**Draw on the board:**

```
Capability vs. Model Scale

Accuracy
   │                         ┌─── Few-shot math
   │                    ┌────┘
   │               ┌────┘          ← capabilities appear
   │    ────────────               ← suddenly, not gradually
   │
   └──────────────────────────────► Parameters
        1B      10B     100B
```

> *"Some capabilities are essentially ZERO below a certain scale threshold.*
> *Then suddenly — as if a switch was flipped — they appear.*
>
> *Examples:*
> *- 3-digit arithmetic: absent below ~50B params, then suddenly good*
> *- In-context learning: GPT-2 showed traces of it, GPT-3 it appeared strongly*
> *- Chain-of-thought reasoning: emerged around 100B+ params*
>
> *Nobody predicted this. The scaling curve for loss is smooth and predictable.
> But which TASKS appear is a phase transition.*
>
> *We don't fully understand why. The current best hypothesis:
> some tasks require the model to simultaneously execute multiple
> sub-skills, and only at large scale does it have enough capacity
> to hold all of them at once."*

**Ask the room:**
> *"Does this change how you think about AI progress?
> Like — what does it mean if the next scale jump produces capabilities
> we genuinely didn't predict?"*

Let this discussion breathe for 5 minutes. It's important.

---

## SECTION 3: Pre-training, Fine-tuning, RLHF, Prompting  (20 min)

> *"One more crucial thing to understand before we start building:
> the training pipeline has multiple stages. The ChatGPT you use
> has been through ALL of them."*

**Draw on the board:**

```
STAGE 1: PRE-TRAINING
  Data: entire internet (trillions of tokens)
  Objective: next-token prediction
  Result: base model that completes text — but can be toxic,
          unreliable, and doesn't follow instructions

  ↓

STAGE 2: SUPERVISED FINE-TUNING (SFT)
  Data: human-written (instruction → ideal response) pairs
        ~10K-100K examples
  Objective: same cross-entropy, but only on response tokens
  Result: model that understands instruction format
          and tries to be helpful

  ↓

STAGE 3: RLHF (Reinforcement Learning from Human Feedback)
  Data: human preference rankings between two model responses
        "Which is better, A or B?"
  Objective: train a reward model, then use PPO to optimize for it
  Result: responses aligned with human preferences — safer,
          more helpful, more honest

  ↓

STAGE 4: PROMPTING (inference time)
  No training! Just craft the input text to guide the model.
  Cost: zero dollars. Happens in real time.
```

> *"Everything after Stage 1 is 'alignment' — teaching the model
> to behave the way humans want it to. Without it, the raw pre-trained
> model is powerful but unpredictable and often harmful.*
>
> *The actual ML research at companies like Anthropic and OpenAI
> is largely about Stages 2 and 3. Stage 1 is mostly engineering."*

### Common confusion to address:

> *"People often ask: is ChatGPT 'just autocomplete'?*
>
> *Technically yes — at inference it's still doing next-token prediction.
> But the fine-tuning and RLHF stages have shaped the DISTRIBUTION
> of which next tokens it predicts.*
>
> *It now predicts tokens that are helpful, honest, and safe —
> not just statistically likely given the internet.*
>
> *The base model and the aligned model have the same architecture.
> The difference is entirely in the weights.*"

---

## SECTION 4: Live Demo — run the full script  (15 min)

> *"Let's let it run all the way through."*

```bash
python3 01_how_llms_work.py
```

Walk through the output as it prints. Key moments to narrate:

- **Model scale table**: "Look at the token counts — Chinchilla's 1.4T tokens
  on a 70B model. Compare to GPT-3's 300B tokens on 175B. Undertrained."

- **BPE tokenizer demo**: "Watch the merge rules happen. Each line is one merge step.
  The vocabulary grows from characters to subwords."

- **Perplexity computation**: "Starting high, training brings it down.
  The shape of this curve is universal across all LLMs."

**Open the visuals folder** and show:
- The scaling law visualization
- The BPE token boundary map
- The perplexity vs. training steps chart

---

## CLOSING SESSION 2  (10 min)

### Full two-session recap board

Write on board:

```
WHAT WE NOW KNOW ABOUT LLMs
═══════════════════════════════════════════════════════════
Architecture:
  Decoder-only Transformer, causal (masked) self-attention
  N blocks of: Masked Attention → Feed-Forward → Layer Norm

Training:
  Objective: next-token prediction, cross-entropy loss
  Metric: perplexity = exp(loss)

Tokenization:
  BPE subword tokens, ~50K-100K vocabulary
  Merges frequent character pairs iteratively

Scale:
  Chinchilla: 20 tokens per parameter for compute-optimal training
  Emergent abilities: some capabilities appear suddenly at threshold

Pipeline:
  Pre-training → SFT → RLHF → Prompting
  Each stage serves a different purpose
```

> *"Next session: Prompt Engineering.*
>
> *Everything we just learned is stuff you can't change — it's baked
> into the pre-trained model. But there's one thing you CAN control:
> what you put into the input.*
>
> *And it turns out, that matters enormously.*
> *We're going to make the same model look brilliant or confused
> just by changing the text we give it."*

---

## INSTRUCTOR TIPS & SURVIVAL GUIDE

### When People Get Confused

**"How does predicting words produce reasoning?"**
> *"Think about it this way: to predict the next word in a calculus
> textbook accurately, you need to 'know' calculus. The task forces
> the model to build whatever internal representation is needed
> to make accurate predictions. Reasoning emerges from that."*

**"What's the difference between GPT and BERT?"**
> *"BERT is encoder-only. It sees the WHOLE sentence and fills in masked words.
> Great for understanding. GPT is decoder-only. It only sees past context
> and generates future tokens. Great for generation. You built both in Part 6."*

**"Why don't we just use characters instead of subword tokens?"**
> *"Attention is O(n²) in sequence length. A 1000-word document is
> ~5000 characters but only ~1300 tokens. Character-level = 4× longer
> sequences = 16× more expensive attention computation. That cost
> is prohibitive at training scale."*

**"Is there an upper limit to what scale can do?"**
> *"This is the trillion-dollar question. The honest answer is we don't know.
> Emergent abilities keep appearing at new scales. But physical limits exist —
> you can only fit so many GPUs on Earth. We'll see."*

### Energy Management

- **30-min mark:** Take a break. Scaling laws and architecture are dense.
- **If excitement is high after the emergence discussion:** Let it run. This is
  the most important philosophical moment in the course.
- **If someone is lost on the transformer architecture:** Remind them Part 6
  covered it in depth. Offer to walk them through the Part 6 visual after class.
- **If they want to go deeper on RLHF:** Tell them Part 3 covers it in detail
  (finetuning_basics module). Today is overview.

### The Golden Rule for This Module

> Everything connects back to what they already know.
> Never present LLMs as magic — always reduce to the mechanisms
> from Parts 3-6. The goal is demystification, not mystification.

---

# QUICK REFERENCE — Session Timing

```
SESSION 1  (90 min)
├── Opening hook                  10 min
├── Section 1: What is an LLM    20 min
├── Section 2: Tokenization       20 min
├── Section 3: Scale table        15 min
├── Live demo (partial run)       15 min
└── Close + homework              10 min

SESSION 2  (90 min)
├── Homework debrief              10 min
├── Section 1: Perplexity         15 min
├── Section 2: Emergent abilities 20 min
├── Section 3: Training pipeline  20 min
├── Section 4: Full live demo     15 min
└── Close + preview of S2         10 min
```

---

# WHAT'S COMING

```
Module 1 (these sessions):  How LLMs Work  ← YOU ARE HERE
Module 2 (next session):    Prompt Engineering
                            — same frozen model, wildly different results
                            — zero-shot, few-shot, chain-of-thought
Module 3:                   Fine-Tuning Basics + LoRA
                            — how to make a model yours
Module 4:                   Retrieval-Augmented Generation
                            — giving the model a cheat sheet
──────────────────────────────────────────────────────────
ALGORITHMS:
  Build MiniGPT in pure NumPy (the crown jewel of this course)
  LoRA from scratch
  RAG pipeline from scratch

PROJECTS:
  Q&A system with RAG
  LLM-powered classifier
```

---

*Generated for MLForBeginners — Module 01 · Part 7: Large Language Models*
