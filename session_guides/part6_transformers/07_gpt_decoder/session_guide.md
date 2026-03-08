# 🎓 MLForBeginners — Instructor Guide
## Part 6 · Module 07: GPT Decoder
### Two-Session Teaching Script

> **Prerequisites:** Modules 01–06. They understand attention, multi-head attention,
> positional encoding, and the full encoder architecture (BERT style).
> **Payoff:** Understand how GPT works — the architecture behind ChatGPT.

---

# SESSION 1 (~90 min)
## "From BERT encoder to GPT decoder — the other half of the Transformer"

## Before They Arrive
- Terminal open in `transformers/algorithms/`
- Draw a side-by-side: BERT (bidirectional) vs GPT (left-to-right)

---

## OPENING (10 min)

> *\"Last module: BERT.*
> *BERT is an encoder — it reads the whole sentence at once,*
> *like a reader who gets to see the whole page before answering.*
>
> *Today: GPT.*
> *GPT is a decoder — it generates text one word at a time,*
> *never looking at future words.*
> *Like typing — you don't know what comes next when you type each letter.*
>
> *Same attention mechanism. Same transformer layers.*
> *But one crucial difference in the attention mask.*
> *That difference is what makes GPT generate.\"*

Write on board:
```
BERT (Encoder):          GPT (Decoder):
  [CLS] I love [MASK]      I love →  ?
   ↕    ↕    ↕    ↕         ↓        ↓
  reads ALL positions     reads ONLY past positions
  bidirectional           left-to-right (causal)

Both use self-attention.
The MASK is the only structural difference.
```

---

## SECTION 1: Causal (Masked) Self-Attention (25 min)

> *\"In BERT, every token can attend to every other token.*
> *In GPT, each token can ONLY attend to itself and tokens BEFORE it.*
>
> *Why? Because GPT is generative.*
> *If token 5 could see token 6 during training,*
> *it would 'cheat' — look at the answer before predicting it.\"*

Write the attention mask on board:
```
ATTENTION MASK (4-token example):

        I   love  coffee  every
I       ✓    ✗     ✗       ✗
love    ✓    ✓     ✗       ✗
coffee  ✓    ✓     ✓       ✗
every   ✓    ✓     ✓       ✓

Upper triangle = MASKED (set to -∞ before softmax)
After softmax: masked positions become 0
→ "coffee" cannot attend to "every" — it hasn't appeared yet
```

> *\"This masking is implemented as a single matrix operation.*
> *Computationally free — just set some values to -infinity.*
> *But conceptually: this one change turns an encoder into a generator.\"*

```bash
python3 gpt_decoder.py
```

Watch the masked attention visualization load.

---

## SECTION 2: The GPT Architecture Stack (20 min)

Write the full stack on board:
```
INPUT TOKENS:
  "The cat sat"

TOKEN EMBEDDINGS + POSITIONAL ENCODING:
  Same as BERT

DECODER BLOCK × N (repeated):
  1. Masked Multi-Head Self-Attention
     (each token attends only to past tokens)
  2. Layer Norm
  3. Feed-Forward Network
  4. Layer Norm

OUTPUT HEAD:
  Linear layer → vocabulary size (50,257 for GPT-2)
  Softmax → probability over all words

PREDICTION:
  Most likely next token
```

> *\"GPT-2 uses 12 of these decoder blocks.*
> *GPT-3 uses 96.*
> *GPT-4: unknown, but estimated 100+.*
> *More layers = better at longer-range patterns.*
> *The architecture is the same — just scaled up.\"*

---

## SECTION 3: How GPT Generates Text (20 min)

> *\"This is the magic. Watch carefully.\"*

Write the generation loop:
```
AUTOREGRESSIVE GENERATION:

Start: "The cat"
Pass through GPT → probability over vocabulary
Sample: "sat"

Now input: "The cat sat"
Pass through GPT → probability over vocabulary
Sample: "on"

Now input: "The cat sat on"
Pass through GPT → probability
Sample: "the"

... repeat until <end-of-sequence> token or max length

The model never sees the "future" — it generates it.
```

> *\"This is called autoregressive generation.*
> *Each new token becomes part of the input for the next step.*
>
> *Critically: GPT is just predicting the next word.*
> *It was trained on internet text: 'given these words, what comes next?'*
>
> *And through this simple objective on 400 billion tokens of text,*
> *it learned grammar, facts, reasoning, and how to answer questions.*
> *The emergent capabilities surprised even the researchers.\"*

---

## SECTION 4: Temperature and Sampling Strategies (10 min)

> *\"When we sample the next token, HOW we sample matters.*
>
> *Greedy: always pick the most likely word.*
> *Result: repetitive, boring text.*
>
> *Temperature sampling: scale the logits before softmax.*
> *Temperature = 1.0: sample from true distribution.*
> *Temperature = 0.5: more confident, more predictable.*
> *Temperature = 1.5: more random, more creative.*
>
> *Top-k sampling: only sample from the k most likely tokens.*
> *Top-p (nucleus): only sample from tokens summing to p probability.*
>
> *ChatGPT uses top-p sampling with p ≈ 0.9-0.95.*
> *That's the 'temperature' slider you see in APIs.\"*

---

## CLOSING SESSION 1 (5 min)

```
SESSION 1 SUMMARY:
  GPT = Transformer decoder
  Key difference: causal attention mask (no peeking at future)
  Generation = autoregressive: one token at a time
  Training = next-token prediction on massive text
  Temperature controls creativity vs predictability
```

**Homework:** *\"Think of 3 prompts you'd give to GPT.*
*What do you think it would generate?*
*Why does the phrasing of your prompt change the output?*
*We'll explore that in Session 2.\"*

---

# SESSION 2 (~90 min)
## "GPT in practice — generation, prompting, and the limits"

## OPENING (5 min)

> *\"Last session: HOW GPT generates.*
> *Today: what GPT is actually good at, where it fails, and how we'll use it.*
>
> *And: you'll run GPT-2 locally — the full generative model.*
> *Same architecture as ChatGPT, just much smaller.\"*

---

## SECTION 1: Running GPT-2 (25 min)

```python
# GPT-2 generation — runs locally
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=100, temperature=0.8, do_sample=True)
print(tokenizer.decode(outputs[0]))
```

Try several prompts together:
```
Prompts to try:
  "Once upon a time in a forest"
  "The most important thing about machine learning is"
  "Scientists discovered that"
  "In the year 2050, humans will"
```

After each generation:

> *\"Notice: it's fluent. It sounds like it was written by a person.*
> *Is it factually accurate? Not necessarily.*
> *GPT-2 has no access to truth — it predicts what SOUNDS right.*
>
> *This is the key limitation: fluency ≠ accuracy.*
> *Hallucination is a fundamental property of autoregressive language models.\"*

---

## SECTION 2: The Training Objective — Just Next Token Prediction (20 min)

> *\"GPT was trained on a simple self-supervised objective.*
> *No human labels needed.*
>
> *Just: given these words, predict the next one.*
> *Then check. Adjust weights. Repeat.*
> *On 400 billion words. (GPT-3)*
>
> *Through this, it learned:*
> *— Grammar (to predict grammatically valid continuations)*
> *— Facts (to predict factually plausible continuations)*
> *— Reasoning (to predict logically consistent continuations)*
> *— Code (GitHub was in the training data)*
>
> *This is why 'just predicting the next word' turned out to be*
> *the most powerful ML objective ever discovered.\"*

Write on board:
```
THE SCALING HYPOTHESIS (validated):
  More parameters + more data + more compute
  = better next-token prediction
  = better at almost every language task

  GPT-1:   117M parameters (2018)
  GPT-2:   1.5B parameters (2019)
  GPT-3:   175B parameters (2020)
  GPT-4:   ~1T+ parameters (estimated, 2023)

  Each step: qualitatively different capabilities emerged.
```

---

## SECTION 3: Encoder vs Decoder — When to Use Which (15 min)

Write the guide:
```
USE AN ENCODER (BERT style) WHEN:
  You need to UNDERSTAND text
  → Sentiment analysis
  → Text classification
  → Named entity recognition
  → Question answering (extract from passage)
  → Semantic similarity

USE A DECODER (GPT style) WHEN:
  You need to GENERATE text
  → Story generation
  → Code completion
  → Summarization (generate summary from scratch)
  → Translation (generate target language)
  → Dialogue / chatbots

BOTH (Encoder-Decoder, T5 / BART):
  → Summarization (attend to full input, generate output)
  → Translation
  → Any task framed as "input text → output text"
```

> *\"When someone says 'use GPT for classification' — they can.*
> *Frame it as: 'The sentiment of this review is...' → complete the sentence.*
> *But BERT is more efficient for classification tasks.*
> *Use the right tool.\"*

---

## CLOSING SESSION 2 (10 min)

```
GPT DECODER — COMPLETE:
  Architecture: stacked masked self-attention blocks
  Training: next-token prediction (self-supervised)
  Generation: autoregressive, one token at a time
  Control: temperature, top-k, top-p sampling

  GPT = language modeler, fluent but not factually reliable
  BERT = language understander, great at classification tasks
  T5/BART = both, great for text-to-text tasks
```

**Looking ahead:**
> *\"Next module: we take BERT and fine-tune it for text classification.*
> *That's where the power of pretrained transformers becomes undeniable.*
> *State of the art on most NLP tasks. In an afternoon.\"*

---

## INSTRUCTOR TIPS

**"Is GPT-2 the same as ChatGPT?"**
> *"Same decoder architecture. Very different scale and training.*
> *GPT-2: 1.5B parameters, next-token prediction only.*
> *ChatGPT (GPT-3.5/4): 175B+, PLUS RLHF — reinforcement learning from human feedback.*
> *RLHF aligns the model to be helpful, harmless, and honest.*
> *That's what turns a next-token predictor into an assistant.\"*

**"Why does it sometimes generate nonsense?"**
> *"GPT predicts likely continuations — not true statements.*
> *'The capital of Australia is...' → probably continues with 'Sydney' (most mentioned city)*
> *Correct answer: Canberra. GPT-2 gets this wrong.*
> *Larger models are more accurate, but hallucination never fully disappears.*
> *It's a fundamental property of the objective.\"*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening bridge             10 min
├── Causal attention           25 min
├── GPT architecture stack     20 min
├── Autoregressive generation  20 min
├── Temperature + sampling     10 min
└── Close + homework            5 min

SESSION 2  (90 min)
├── Opening                     5 min
├── Run GPT-2 locally          25 min
├── Training objective         20 min
├── Encoder vs decoder guide   15 min
└── Close                      10 min  (+ 15 min buffer)
```

---
*MLForBeginners · Part 6: Transformers · Module 07*
