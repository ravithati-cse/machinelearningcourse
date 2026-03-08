# MLForBeginners — Instructor Guide
## Module 2: Prompt Engineering  ·  Two-Session Teaching Script

> **Who this is for:** You, teaching close friends who completed Module 1 and Parts 1-6.
> **Tone:** Immediately practical. This is the one module they can apply at 9am tomorrow
> with ChatGPT open. Make it feel like a toolkit they're walking away with.
> **Goal by end of both sessions:** Everyone can write zero-shot, few-shot, and
> chain-of-thought prompts, understands what temperature controls, and has a
> personal prompt engineering checklist.

---

# ─────────────────────────────────────────────
# SESSION 1  (~90 min)
# "Same model. Wildly different results. Why?"
# ─────────────────────────────────────────────

## Before They Arrive (10 min setup)

**On your laptop, have ready:**
- Terminal open in `MLForBeginners/llms/math_foundations/`
- `02_prompt_engineering.py` loaded but NOT run yet
- A browser tab open with ChatGPT or Claude (you'll demo live)
- Whiteboard ready

**Mindset prep:** This is the most immediately practical session in the
entire course. Everyone will be able to use what they learn TODAY.
That energy should be in the room. If anyone is tired — they won't be for long.

---

## OPENING  (10 min)

### Hook — Same model, different behavior

Open your browser to ChatGPT or Claude. Do this live:

**First, type this:**
> `Summarize AI.`

Show the output. It will be generic and vague.

**Then, type this:**
> `You are a senior editor at MIT Technology Review. Summarize the concept of
> Artificial Intelligence for a general adult reader in exactly 3 bullet points.
> Each bullet must be under 20 words. Use no jargon. Start each bullet with a
> verb in present tense.`

Show the output. It will be dramatically better.

Then say:

> *"Same model. Same weights. Same hardware. Same price per token.*
>
> *The ONLY difference was the text we put in.*
>
> *That's prompt engineering: the practice of crafting inputs to a frozen,
> pre-trained LLM to get dramatically better outputs — without touching
> a single weight.*
>
> *Well-designed prompts can close 80% of the gap to a fine-tuned model.*
> *On a per-hour basis, this is the highest-ROI ML skill you can develop.*
> *And you can start using it today."*

---

## SECTION 1: What is Prompt Engineering?  (10 min)

> *"Let me give you the mental model. When you last session's LLM is doing
> inference, it's looking at all the text in its input and asking:
> 'What text is most likely to come next, given all of this?'*
>
> *Your prompt IS that input. It shapes the probability distribution
> over every possible next token.*
>
> *Think of it like this:"*

**Draw on the board:**

```
"Summarize AI."
 └──────────────────────────────────────────┐
                                             ▼
                                  [ LLM: 50K token distribution ]
                                             │
                                             ▼ (samples from peak)
                              "AI stands for Artificial Intelligence..."
                              → Generic, unfocused


"You are an MIT editor. Summarize AI in 3 bullets, <20 words each..."
 └──────────────────────────────────────────┐
                                             ▼
                          [ SAME LLM, DIFFERENT distribution ]
                                             │ (same model, different peak!)
                                             ▼
                         "• AI systems learn from data..."
                         → Specific, structured, expert-level
```

> *"The prompt re-shapes the distribution. Not the model — the prompt.*
>
> *Three things that always help:*
> *1. Give the model a ROLE — 'You are a...'*
> *2. Give it a FORMAT — 'Respond in 3 bullets'*
> *3. Give it CONSTRAINTS — 'Each under 15 words, no jargon'"*

---

## SECTION 2: Zero-Shot Prompting  (15 min)

> *"Zero-shot means: no examples in the prompt. Just a description of the task.*
>
> *The model has to rely entirely on its pre-trained knowledge about
> what that task means and how it's usually done."*

**Write on board:**

```
ZERO-SHOT TEMPLATE:
────────────────────────────────────────────────────────────
[Role/Persona]
[Task description]
[Format constraints]
[Input]

EXAMPLE:
"You are a sentiment analysis expert.
 Classify the following customer review as Positive, Negative, or Neutral.
 Respond with a single word only.
 Review: 'The camera quality is amazing but battery life is terrible.'"
────────────────────────────────────────────────────────────
```

> *"Works when: the task is common, the instructions are unambiguous,
> and you have zero examples. Very fast. Very cheap.*
>
> *Fails when: the task is unusual, the output format matters a lot,
> or the model needs to see what 'correct' looks like."*

**Live mini-exercise:**

> *"Let's all write a zero-shot prompt for this task:*
> *'Extract all names of people and companies from this news sentence.'*
>
> *You have 3 minutes. Write it on paper or your laptop.*
> *Then we'll compare."*

Have 2-3 people share theirs. Compare role clarity, format specification,
constraint specificity. Show how small wording choices change outputs.

---

## SECTION 3: Few-Shot Prompting  (20 min)

> *"Now the good stuff. Few-shot: give the model examples.*
>
> *This is called 'in-context learning' — the model learns from examples
> you put directly in the prompt, with no weight updates at all.*
>
> *It works because — remember how it was trained?*
> *Next-token prediction on the entire internet.*
> *The internet is full of patterns like 'here are examples, now do the same thing'.*
> *The model has seen millions of those patterns. So it knows how to follow them."*

**Write on board:**

```
FEW-SHOT TEMPLATE:
────────────────────────────────────────────────────────────
[Role / Task description]

Examples:
  Input: [example 1 input]
  Output: [example 1 correct output]

  Input: [example 2 input]
  Output: [example 2 correct output]

  Input: [example 3 input]
  Output: [example 3 correct output]

Now classify:
  Input: [new input]
  Output:
────────────────────────────────────────────────────────────
```

> *"Notice: we stop the last 'Output:' blank.*
> *The model completes it based on the pattern it saw in the examples.*
> *You are literally programming the model with examples instead of code."*

**Worked example on board:**

```
"Classify each customer issue as: billing / technical / shipping / other

Input: 'I was charged twice for my subscription last month.'
Output: billing

Input: 'The app crashes every time I open my account settings.'
Output: technical

Input: 'My order was supposed to arrive Tuesday and still hasn't shown up.'
Output: shipping

Input: 'I love the product but want to suggest a new feature.'
Output:"
```

> *"What does the model output? Almost certainly: 'other'.*
> *The examples trained it, in-context, on what each category means.*
>
> *Critical tips for few-shot:*
> *- Use 3-8 examples (more isn't always better)*
> *- Make examples REPRESENTATIVE of the real distribution*
> *- Include edge cases if the model keeps getting them wrong*
> *- The last example is the most influential — choose it carefully"*

**Ask the room:**
> *"If you wanted the model to classify medical vs non-medical questions,
> what examples would you pick? Think about edge cases.*
> *'What medications can I take for a headache?' — medical or not?"*

Let them debate. This gets into interesting territory about where category
boundaries sit and why examples need to represent that.

---

## SECTION 4: Chain-of-Thought Prompting  (20 min)

> *"Here's one of the most impactful discoveries in all of LLM research.*
> *And it came from a paper that Google published in 2022.*
>
> *On reasoning tasks — math, logic, multi-step problems — if you ask the model
> to think out loud before giving the answer, accuracy goes WAY up.*
>
> *On the GSM8K math benchmark:*"

**Write on board:**

```
Standard prompting:                    Chain-of-Thought:
──────────────────                     ───────────────────
Q: Roger has 5 balls.                  Q: Roger has 5 balls.
   He buys 2 more cans                    He buys 2 more cans
   of 3 balls each.                       of 3 balls each.
   How many does he have?                 How many does he have?

A: 11                                  A: Roger starts with 5 balls.
                                          Each can has 3 balls.
                                          2 cans × 3 = 6 balls.
                                          5 + 6 = 11 balls.
                                          The answer is 11.
──────────────────────────────────────────────────────────
GPT-3 standard:  17% correct           GPT-3 with CoT:  58% correct
```

> *"Forcing the model to write out intermediate steps dramatically improves
> reasoning accuracy. Why?*
>
> *Remember: it's a next-token predictor.*
> *By generating intermediate reasoning tokens, each subsequent token
> is conditioned on that reasoning.*
> *The model 'thinks' by writing.*
>
> *The magic phrase: 'Let's think step by step.'*
> *This alone, added to almost any reasoning prompt, improves results.*
> *Google researchers call it the 'zero-shot chain of thought' trigger."*

**Live demo — do this in the browser:**

Paste the Roger ball problem to ChatGPT with and without "Let's think step by step."
Show the difference in output structure.

**Ask the room:**
> *"Where else would this help? Give me a real example from your work or life
> where you'd want the model to reason step by step."*

Common good answers: debugging code, medical differential diagnosis, legal analysis,
financial projections with multiple variables.

---

## CLOSING SESSION 1  (10 min)

Write on board:

```
PROMPT ENGINEERING TOOLKIT — SESSION 1
═══════════════════════════════════════════════════════
Technique          When to Use              Magic Phrase
───────────────────────────────────────────────────────
Zero-shot          Common tasks, fast       [Role] + [Format]
Few-shot           Custom format, accuracy  "Input: ... Output: ..."
Chain-of-thought   Math, logic, reasoning   "Let's think step by step"
═══════════════════════════════════════════════════════
```

> *"Homework — not optional, this is actually fun:*
>
> *Open ChatGPT right now. Pick one task you did this week.*
> *Write three versions of the prompt:*
> *1. Zero-shot, bad version (vague, no constraints)*
> *2. Zero-shot, good version (role + format + constraints)*
> *3. Few-shot version (2-3 examples)*
>
> *Screenshot all three outputs. Next session we compare."*

---
---

# ─────────────────────────────────────────────
# SESSION 2  (~90 min)
# "Structured outputs, temperature, and the full framework"
# ─────────────────────────────────────────────

## Opening  (10 min)

### Homework show-and-tell

> *"Let's see the screenshots. Who had the biggest difference between
> their bad zero-shot and their good zero-shot?"*

Have 2-3 people share. This is almost always entertaining and reinforcing.
Point out what made the good prompts work: specificity, role, format.

> *"Today we do structured outputs, temperature — what it actually means —
> and I'll give you a checklist you can use forever."*

---

## SECTION 1: Structured Output Prompting  (20 min)

> *"In real products, you usually don't want a chatty paragraph response.
> You want data you can parse — JSON, a table, a specific format.*
>
> *Prompting can do that."*

**Write on board:**

```
STRUCTURED OUTPUT PROMPT:
─────────────────────────────────────────────────────────────
"You are a data extraction assistant.
 Extract the following fields from the text below.
 Respond with ONLY valid JSON. No other text before or after.
 JSON schema:
 {
   "company": string,
   "funding_amount_usd": number,
   "investors": [string],
   "product_description": string
 }

 Text: 'Acme AI raised $50M in Series B from Sequoia and a16z.
        Their product helps hospitals predict patient readmissions.'"
─────────────────────────────────────────────────────────────

Expected output:
{
  "company": "Acme AI",
  "funding_amount_usd": 50000000,
  "investors": ["Sequoia", "a16z"],
  "product_description": "Helps hospitals predict patient readmissions"
}
```

> *"This is how you build real pipelines on top of LLMs.*
> *The model's output is parsed directly by your code.*
> *If the model diverges from the format, it breaks your pipeline.*
>
> *Best practices for structured output prompts:*"

**Write on board:**

```
1. State the format early ("Respond with ONLY valid JSON")
2. Provide the exact schema with field names and types
3. Say what NOT to include ("No preamble, no explanation")
4. Include a few-shot example if format is complex
5. Use model providers' JSON mode if available (OpenAI, Anthropic)
```

**Run the script — structured output sections:**

```bash
python3 02_prompt_engineering.py
```

Point at the structured output comparison as it prints.

---

## SECTION 2: Temperature and Sampling  (20 min)

> *"Everyone has seen the 'temperature' setting in LLM APIs.*
> *Let me tell you what it actually does mathematically."*

**Write on board:**

```
At every token position, the model outputs LOGITS:
  (raw scores for each token in the vocabulary)

  "mat": 4.2
  "rug": 2.1
  "floor": 1.8
  "cat": 0.3
  ... (50K more entries)

Softmax converts to probabilities:
  P(token) = exp(logit) / Σ exp(all logits)

Temperature τ modifies this:
  P(token) = exp(logit / τ) / Σ exp(logit_i / τ)
```

**Draw the effect:**

```
τ = 0.1 (very low — COLD):          τ = 1.0 (default):
  "mat" → 99.9%                       "mat" → 68%
  "rug" → 0.1%                        "rug" → 12%
  (almost always picks top token)     "floor" → 9%
                                      (realistic distribution)

τ = 2.0 (high — HOT):
  "mat" → 35%
  "rug" → 25%
  "floor" → 20%
  "cat" → 12%
  (much more random)
```

> *"Low temperature → deterministic, predictable, safe.*
> *High temperature → creative, diverse, sometimes nonsensical.*
>
> *The right setting depends on your task:"*

**Write this table:**

```
Task                        Temperature    Why
──────────────────────────────────────────────────────────
Code generation             0.0 - 0.2      Correctness over creativity
Factual Q&A                 0.0 - 0.3      Don't want hallucination
Summarization               0.3 - 0.5      Some variation is fine
Creative writing            0.7 - 1.0      Diversity is the point
Brainstorming               1.0 - 1.5      Maximum variety
```

**Ask the room:**
> *"What temperature would you use for a chatbot that answers customer support
> questions about a product? Why?"*

Correct answer: 0.0-0.3. Accuracy matters more than creativity.
You do NOT want "creative" hallucinations in customer support.

---

## SECTION 3: Prompt Engineering Checklist  (15 min)

> *"Here's the framework I use every time I write a serious prompt.*
> *Write this down."*

**Slowly write on board, explaining each point:**

```
PROMPT ENGINEERING CHECKLIST
══════════════════════════════════════════════════════════

1. ROLE          Is there a persona that primes the right knowledge?
                 "You are a senior Python engineer."

2. TASK          One clear verb-noun statement of the task.
                 "Extract all error codes."

3. CONTEXT       What does the model need to know to succeed?
                 Background, constraints, edge cases.

4. FORMAT        Exactly how should the output look?
                 JSON, bullets, table, plain text, max N words.

5. EXAMPLES      If the task is nuanced, show 2-3 worked examples.
                 (Few-shot when zero-shot fails.)

6. CHAIN-OF-THOUGHT  For reasoning tasks: "Think step by step."

7. NEGATIVE CONSTRAINTS  What NOT to do.
                 "Do not use jargon. Do not include preamble."

8. TEMPERATURE   Set explicitly based on task type (see table above).
```

> *"Every professional prompt engineer uses some version of this checklist.*
> *You don't need all 8 every time. But check each one before you deploy a prompt.*
>
> *The most commonly forgotten: #7 (negative constraints) and #4 (format).*
> *Models often know WHAT to say but not HOW you want it formatted.*
> *Tell them explicitly."*

---

## SECTION 4: Full Live Demo  (15 min)

Run the complete script and open visuals:

```bash
python3 02_prompt_engineering.py
```

Walk through the comparison charts:
- Zero-shot vs. few-shot accuracy on sentiment classification
- Temperature effect on output diversity
- Chain-of-thought accuracy improvement

> *"Look at this comparison chart — same model, zero-shot 72% accuracy,
> few-shot with 5 examples 91% accuracy.*
> *19 percentage points, zero training, zero cost to update weights.*
> *Just the text in the input."*

---

## CLOSING SESSION 2  (10 min)

Write on board:

```
COMPLETE PROMPT ENGINEERING TOOLKIT
══════════════════════════════════════════════════════════
Zero-shot:          Role + Task + Format + Constraints
Few-shot:           Examples as in-context training data
Chain-of-thought:   "Think step by step" for reasoning
Structured output:  JSON schema + "ONLY output JSON"
Temperature:        Low for accuracy, high for creativity

8-point checklist: Role, Task, Context, Format, Examples,
                   CoT, Negative constraints, Temperature
══════════════════════════════════════════════════════════
```

> *"Honestly — use this today. Open whatever LLM you use.*
> *Pick the most annoying prompt you have.*
> *Run it through the 8-point checklist.*
>
> *I guarantee you'll see a meaningful improvement.*
>
> *Next session: Fine-tuning — what to do when prompting isn't enough."*

---

## INSTRUCTOR TIPS & SURVIVAL GUIDE

### When People Get Confused

**"When should I few-shot vs zero-shot?"**
> *"Start with zero-shot. If accuracy is low OR the output format is wrong,
> add examples. 3-5 examples usually covers 80% of the improvement.
> More than 8 examples rarely helps without diminishing returns."*

**"Can I make the model do anything with prompting?"**
> *"No. Two hard limits:*
> *1. Knowledge cutoff — it can't know facts after training*
> *2. Safety filters — aligned models will refuse harmful requests*
>
> *Prompting is powerful within those bounds. Outside them,
> you need RAG (for knowledge) or a different model (for restrictions)."*

**"Is this really 'AI' or just pattern matching?"**
> *"This debate is ongoing in the research community. What we can say:
> the model generalizes far beyond its training data in ways that
> look like reasoning. Whether that IS reasoning or just very sophisticated
> pattern completion — that's a philosophy question. The practical answer:
> it works, use it."*

### Energy Management

- **This is the most energizing session in Part 7.** People leave with skills.
- Do the live ChatGPT demo in the opening. Real-time is more compelling than slides.
- If people want to try prompts themselves — ENCOURAGE this. Phones out, all trying.
- If someone finds a creative prompt that works unexpectedly well — celebrate it.
- Budget 10 extra minutes if the temperature discussion gets philosophical.

---

# QUICK REFERENCE — Session Timing

```
SESSION 1  (90 min)
├── Opening (live ChatGPT demo)   10 min
├── Section 1: What is PE?        10 min
├── Section 2: Zero-shot          15 min
├── Section 3: Few-shot           20 min
├── Section 4: Chain-of-thought   20 min
└── Close + homework              15 min

SESSION 2  (90 min)
├── Homework show-and-tell        10 min
├── Section 1: Structured output  20 min
├── Section 2: Temperature        20 min
├── Section 3: PE Checklist       15 min
├── Section 4: Full live demo     15 min
└── Close + next steps            10 min
```

---

# WHAT'S COMING

```
Module 1:  How LLMs Work           ✅
Module 2:  Prompt Engineering      ← YOU ARE HERE
Module 3:  Fine-Tuning Basics
           — when prompting isn't enough
           — SFT loss from scratch
           — RLHF overview
Module 4:  Retrieval-Augmented Generation
           — solving the knowledge cutoff problem
───────────────────────────────────────────────────
ALGORITHMS: MiniGPT in NumPy, LoRA from scratch
PROJECTS:   Q&A system, LLM-powered classifier
```

---

*Generated for MLForBeginners — Module 02 · Part 7: Large Language Models*
