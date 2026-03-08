# 🎓 MLForBeginners — Instructor Guide
## Part 6 · Module 08: BERT Text Classifier (Project)
### Single 120-Minute Session

> **Prerequisites:** All of Part 6 Modules 01–07.
> They understand attention, BERT architecture, GPT decoding.
> **Payoff:** Fine-tune BERT and beat every NLP model they've built before.
> This is the moment transformers become real tools.

---

# SESSION (120 min)
## "Fine-tune BERT — state-of-the-art text classification in one afternoon"

## Before They Arrive
- Terminal open in `transformers/projects/`
- Have the Part 5 accuracy numbers handy for comparison
- GPU access if possible (CPU will work, just slower)

---

## OPENING (10 min)

> *\"Every NLP model we've built so far has been trained FROM SCRATCH.*
> *We started with random weights and learned everything from our small dataset.*
>
> *Today is different.*
>
> *BERT has already read the entire English Wikipedia.*
> *It's already read BookCorpus — thousands of books.*
> *It has 110 million parameters tuned on billions of words.*
>
> *We're going to take that knowledge and point it at our specific task.*
> *Not teach it English from scratch — it already knows English.*
> *Just teach it: for THIS kind of text, output THIS label.*
>
> *That's fine-tuning. And the results are remarkable.\"*

Write on board:
```
PRETRAIN (already done, by Google):
  BERT reads 3.3 billion words
  Learns language, grammar, facts, context
  Time: weeks on 64 TPUs

FINE-TUNE (we do today):
  Take pretrained BERT
  Add classification head
  Train on OUR labeled data (hours, not weeks)
  BERT adapts to our task

Result: state-of-the-art accuracy, hours not weeks.
```

---

## SECTION 1: What Fine-Tuning Looks Like (15 min)

```bash
python3 bert_text_classifier.py
```

While it downloads:

> *\"Fine-tuning BERT takes the pretrained model and adds a single linear layer.*
> *That layer maps BERT's 768-dimensional output to our number of classes.*
>
> *During fine-tuning, we train BOTH the new head AND update BERT's weights slightly.*
> *'Slightly' is key — if we update too aggressively, BERT 'forgets' its language knowledge.*
> *This is called catastrophic forgetting.\"*

Write the architecture:
```
INPUT: "This movie was absolutely incredible"

TOKENIZE → BERT ENCODING:
  [CLS] This movie was absolutely incredible [SEP]
     ↓    ↓     ↓    ↓      ↓          ↓       ↓
  BERT processes all tokens with self-attention

POOLING:
  Take [CLS] token's representation
  (BERT was trained so [CLS] summarizes the whole sentence)

CLASSIFICATION HEAD:
  [CLS vector (768d)] → Linear(768, num_classes) → Softmax

OUTPUT: P(positive) = 0.97, P(negative) = 0.03
```

> *\"The [CLS] token is a special 'summary' token BERT inserts at the start.*
> *Through training, BERT learned to encode sentence-level meaning there.*
> *We extract that summary and classify it.*
> *That's fine-tuning in one sentence: extract BERT's summary, classify it.\"*

---

## SECTION 2: The Training Loop (20 min)

Watch the training output together.

> *\"Notice how fast it improves.*
> *Epoch 1: already 85%+ accuracy — BERT's pretrained knowledge is doing the heavy lifting.*
> *By Epoch 3: 93-95%.*
>
> *Compare to our Part 5 NLP models:*
> *TF-IDF + Logistic Regression: ~85%*
> *LSTM with embeddings: ~90%*
> *BERT fine-tuned: ~94%+*
>
> *Same dataset. Same task. BERT wins.*
> *And it reached good performance in 3 epochs instead of 30.\"*

Fill in the comparison table:
```
MODEL                    Accuracy   Epochs   Time
─────────────────────────────────────────────────
TF-IDF + Logistic           85%      1        5s
LSTM + Embeddings           90%      30      3 min
BERT fine-tuned             94%+     3       10 min
```

> *\"That 4% gap from 90% to 94% is massive in practice.*
> *At 1 million examples/day, that's 40,000 fewer errors.*
> *That's why every major tech company moved to transformers.\"*

---

## SECTION 3: Tokenization — BERT's Special Vocabulary (15 min)

> *\"BERT doesn't tokenize by spaces.*
> *It uses WordPiece tokenization — breaking words into subword units.\"*

Write examples:
```
WordPiece tokenization:

"playing"     → ["play", "##ing"]
"unbelievable" → ["un", "##believ", "##able"]
"ChatGPT"     → ["Chat", "##GP", "##T"]
"tokenization" → ["token", "##ization"]

The "##" means: this piece continues the previous word.

WHY SUBWORDS?
  Handles rare words: "supercalifragilistic" still gets tokenized
  Handles new words: "ChatGPT" wasn't in the vocabulary (2018)
  Compact vocabulary: 30,000 tokens covers most of English

BERT's vocabulary: 30,522 tokens
GPT-2's vocabulary: 50,257 tokens (byte-pair encoding)
```

> *\"Show of hands: who here thought it just splits by spaces?*
> *Most people do. Subword tokenization is one of the key insights*
> *that makes transformers robust to new and unusual words.\"*

---

## SECTION 4: What BERT Gets Right (That LSTMs Get Wrong) (15 min)

> *\"Let's look at specific examples where BERT clearly wins.\"*

Write on board:
```
AMBIGUOUS SENTENCES:

"I saw the man with the telescope"
  → Who has the telescope? Me or the man?
  BERT: looks at all words simultaneously → resolves correctly
  LSTM: reads left to right → struggles with this ambiguity

"The bank was steep"  vs  "The bank closed early"
  → "bank" means river bank vs financial bank
  BERT: bidirectional context → different embeddings for each
  LSTM: context helps but limited
  TF-IDF: SAME vector for "bank" always → fails completely

"Not bad"
  → Positive sentiment (double negative)
  BERT: attention between "not" and "bad" → understands negation
  Naive Bayes: "bad" = negative → wrong
```

> *\"BERT's bidirectional attention is what makes it understand*
> *context in a fundamentally deeper way.*
> *The sentence is not just a bag of words — it's a web of relationships.\"*

---

## SECTION 5: Live Testing (15 min)

Test custom sentences together:

```python
test_texts = [
    "The customer service was absolutely terrible and I want a refund",
    "Not the worst experience I've had, but could be better",
    "This is genuinely one of the most impressive products I've ever used",
    "I wouldn't say I loved it but I didn't hate it either",  # hard case
    "The food was cold but the service was warm and friendly",  # mixed
]
```

For each:
1. Have room predict: positive or negative?
2. Run and show BERT's confidence
3. Discuss the hard cases

> *\"The last two are genuinely difficult.*
> *'Not the worst' — is that positive or negative?*
> *BERT's confidence will be lower on ambiguous examples.*
> *That calibrated uncertainty is itself useful information.\"*

---

## CLOSING (10 min)

Write on board:
```
BERT FINE-TUNING — KEY TAKEAWAYS:

  Why it works:
  → Massive pretraining = built-in language understanding
  → Fine-tuning adapts that knowledge to your task in hours

  When to use it:
  → Any text classification task with < 100K examples
  → When state-of-the-art accuracy matters
  → When you have a GPU (or can rent one for a few hours)

  When NOT to use it:
  → Millions of examples + simple patterns → Logistic Regression is fine
  → Need < 1ms prediction latency → BERT is too slow (use distillation)
  → No GPU → use TF-IDF + SVM, it's surprisingly competitive

NEXT: GPT-2 Text Generator (the generative counterpart)
```

---

## INSTRUCTOR TIPS

**"What's the difference between BERT-base and BERT-large?"**
> *"BERT-base: 12 layers, 768 dimensions, 110M parameters.*
> *BERT-large: 24 layers, 1024 dimensions, 340M parameters.*
> *Large is better on benchmarks, but 2-3x slower.*
> *For most tasks: BERT-base is the practical choice.\"*

**"What about DistilBERT?"**
> *"DistilBERT: 40% smaller, 60% faster, retains 97% of BERT's performance.*
> *Knowledge distillation: a small model trained to mimic a large one.*
> *For production where speed matters: DistilBERT is often the right choice.*
> *We'll touch on this in Part 7 (LLMs).\"*

**"How do I fine-tune for my own data?"**
> *"Same script, different dataset.*
> *Replace the training data with your labeled examples.*
> *If you have < 500 examples: might underfit, consider more augmentation.*
> *If you have > 10K examples: you'll get excellent results.*
> *Sweet spot: 1K-50K examples per class.\"*

---

## Quick Reference
```
Single Session (120 min)
├── Opening hook                10 min
├── Fine-tuning architecture    15 min
├── Live training + comparison  20 min
├── Tokenization deep dive      15 min
├── BERT vs LSTM analysis       15 min
├── Live testing                15 min
└── Closing + next module       10 min (+ 20 min buffer)
```

---
*MLForBeginners · Part 6: Transformers · Module 08*
