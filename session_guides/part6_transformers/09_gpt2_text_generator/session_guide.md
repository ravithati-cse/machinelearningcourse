# 🎓 MLForBeginners — Instructor Guide
## Part 6 · Module 09: GPT-2 Text Generator (Project + Part 6 Capstone)
### Single 120-Minute Session

> **The Part 6 graduation project.**
> They've mastered attention, multi-head attention, positional encoding,
> encoder-decoder architecture, BERT fine-tuning, and GPT decoding.
> Today: run GPT-2 and generate real text — creative writing, code, conversation.
> Then: a full Part 6 graduation to mark the Transformers milestone.

---

# SESSION (120 min)
## "Generate text with GPT-2 — and understand every line"

## Before They Arrive
- Terminal open in `transformers/projects/`
- GPU preferred — CPU works but generation is slower
- Prepare 5-6 diverse prompts in a text file

---

## OPENING (10 min)

> *\"This module is the most fun one in the course.*
> *You're going to watch a machine write.*
>
> *GPT-2 was released by OpenAI in 2019.*
> *At the time, OpenAI called it 'too dangerous to release fully'*
> *because of its ability to generate convincing fake text.*
> *Now we run it locally on our laptops.*
>
> *You understand how it works — every layer, every attention head.*
> *Today you drive it.*
> *Let's see what it can do.\"*

Start the program immediately while talking:

```bash
python3 gpt2_text_generator.py
```

---

## SECTION 1: The Model Behind the Words (15 min)

While GPT-2 downloads (only first time):

> *\"GPT-2 has 1.5 billion parameters in its largest version.*
> *We're using GPT-2-medium: 345 million parameters.*
> *Still massive. 24 decoder blocks, 1024 dimensions, 16 attention heads.*
>
> *Each of those 345 million numbers was tuned to predict*
> *the next word on 40 gigabytes of internet text.*
>
> *When we give it a prompt, it doesn't 'look up' anything.*
> *It computes — forward pass through 24 layers —*
> *and produces a probability distribution over its 50,257-word vocabulary.*
> *Then samples from that distribution.*
> *Then repeats.*
>
> *That's it. Repeat that 100 times = a paragraph.\"*

Write on board:
```
GPT-2 MEDIUM:
  Parameters:      345 million
  Decoder layers:  24
  Hidden size:     1024
  Attention heads: 16
  Vocabulary:      50,257 tokens
  Training data:   40GB of internet text (WebText)

One forward pass = one token = ~1 word
Generate 200 tokens ≈ write 150 words
```

---

## SECTION 2: Generation — Live Experiments (35 min)

Run a series of prompts with the group. For each, take predictions first, then generate:

**Round 1 — Creative writing:**
```python
prompt = "In the year 2150, the last human city floated above"
# Temperature: 0.8, max_length: 150
```

> *\"What do you predict it'll write?*
> *One word that will definitely appear? One word that won't?\"*
> *(Generate, read, discuss.)*

**Round 2 — Factual continuation:**
```python
prompt = "The theory of relativity, proposed by Albert Einstein, states that"
```

> *\"GPT-2 knows facts — it read Wikipedia.*
> *But does it get this right? Is it accurate?*
> *Remember: it predicts what SOUNDS likely, not what's TRUE.\"*

**Round 3 — Temperature comparison:**

Same prompt, three temperatures:
```python
prompt = "The best way to learn machine learning is to"
# Run with temperature 0.3, 0.8, and 1.5
```

> *\"Temperature 0.3: predictable, maybe repetitive.*
> *Temperature 0.8: balanced — creative but coherent.*
> *Temperature 1.5: wild, sometimes nonsensical.*
>
> *The 'creativity dial' is literally just one number in the sampling function.\"*

**Round 4 — Their prompts:**

Go around the room — each person gives one prompt. Generate it.

---

## SECTION 3: Prompt Engineering Basics (15 min)

> *\"Notice how the quality of output depends heavily on HOW you phrase the prompt.*
> *This is why 'prompt engineering' is a real skill.\"*

Write on board:
```
PROMPT ENGINEERING PRINCIPLES (even for GPT-2):

1. BE SPECIFIC
   Bad:  "Write about science"
   Good: "The discovery of penicillin in 1928 by Alexander Fleming"

2. SET THE TONE
   Bad:  "Continue this story"
   Good: "In a tense, thriller-style narrative, the detective entered"

3. PRIME WITH FORMAT
   Bad:  "What is machine learning?"
   Good: "Machine learning, defined simply, is the process of"

4. GIVE CONTEXT
   Bad:  "The result was surprising"
   Good: "After running the experiment three times with the same outcome,"

GPT completes whatever pattern you start.
Start well → finish well.
```

> *\"ChatGPT uses all these tricks — but now applied to a model 100x larger.*
> *The principles are identical. You already understand them.\"*

---

## SECTION 4: Limitations and What Comes Next (10 min)

> *\"Let's be honest about GPT-2's limits.*
>
> *1. Knowledge cutoff: it only knows what was on the internet in 2019.*
> *2. No factual grounding: it makes up plausible-sounding facts.*
> *3. No memory: each generation starts fresh — it doesn't 'remember' past runs.*
> *4. No reasoning: it pattern-matches, it doesn't think.*
>
> *These are exactly the limitations that Part 7 addresses.*
>
> *RAG (Retrieval Augmented Generation): give the model real-time facts.*
> *Fine-tuning: make it expert in a specific domain.*
> *RLHF: align it to be helpful and accurate, not just fluent.*
>
> *You're one part away from understanding all of that.\"*

---

## PART 6 GRADUATION (15 min)

Write slowly on board:

```
PART 6 COMPLETE — TRANSFORMERS MASTERED 🎓

You can now explain:
  ✅ Self-attention: "Which words should I focus on?"
  ✅ Multi-head attention: many perspectives simultaneously
  ✅ Positional encoding: injecting word order into attention
  ✅ Encoder: BERT — bidirectional understanding
  ✅ Decoder: GPT — left-to-right generation
  ✅ Fine-tuning: point pretrained models at specific tasks

You have built:
  ✅ Transformer from scratch (NumPy)
  ✅ BERT text classifier (fine-tuned, state-of-the-art)
  ✅ GPT-2 text generator (running locally)

THE JOURNEY SO FAR:
  Algebra → Statistics → Linear Regression
  → Classification → Neural Networks
  → CNNs → NLP → Transformers

WHAT COMES NEXT:
  Part 7: Large Language Models
  → How does GPT-3 / GPT-4 / Llama differ from GPT-2?
  → LoRA: fine-tune a 7B model on a laptop
  → RAG: give LLMs access to your own documents
  → Build a production Q&A system from scratch
```

> *\"You now understand the architecture powering:*
> *ChatGPT, Claude, Gemini, GitHub Copilot, Midjourney's text encoder,*
> *Google Translate, Bing Search, every AI assistant.*
>
> *Not as a black box — as a sequence of matrix multiplications*
> *and attention mechanisms that you can draw from memory.*
>
> *That puts you in the top 1% of people who work with these systems.*
>
> *Part 7 is the final frontier.*
> *How do the MASSIVE models work — the ones with 7 billion, 70 billion, 1 trillion parameters?*
> *How do you use them without a supercomputer?*
> *Let's find out.\"*

**Graduation moment:** Each person generates their favorite prompt one last time. Screenshot or note the best output. This is their Part 6 artifact.

---

## INSTRUCTOR TIPS

**"Can GPT-2 write code?"**
> *"Yes — it was trained on GitHub data.*
> *Try: 'def fibonacci(n):' and watch it complete the function.*
> *GPT-2 writes reasonable Python for simple tasks.*
> *Codex (the model behind GitHub Copilot) was specifically fine-tuned on code — much better.*
> *But the base capability comes from the same architecture.\"*

**"How much would it cost to train GPT-2 from scratch today?"**
> *"Approximately $50,000–$100,000 in cloud compute.*
> *GPT-3: estimated $4.6 million.*
> *GPT-4: estimated $100 million+.*
> *This is why 'fine-tune existing models' beats 'train from scratch' for 99% of use cases.\"*

**"What's the difference between GPT-2 and Llama 2/3?"**
> *"Same decoder architecture. Llama 2-7B has 7 billion parameters vs 1.5B.*
> *More layers, larger hidden dimensions, trained on better and more data.*
> *Plus: group query attention, rotary positional embeddings — small improvements.*
> *The core idea is identical. You already understand Llama.\"*

---

## Quick Reference
```
Single Session (120 min)
├── Opening + start program     10 min
├── Architecture recap          15 min
├── Live generation rounds      35 min
├── Prompt engineering          15 min
├── Limitations + Part 7 bridge 10 min
└── Part 6 graduation           15 min
```

---
*MLForBeginners · Part 6: Transformers · Module 09 (Capstone)*
