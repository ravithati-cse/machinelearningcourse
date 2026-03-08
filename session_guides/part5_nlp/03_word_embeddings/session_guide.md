# MLForBeginners — Instructor Guide
## Part 5, Module 3: Word Embeddings  ·  Two-Session Teaching Script

> **Who this is for:** You, teaching close friends who completed Modules 1-2.
> **They already know:** TF-IDF, cosine similarity, vocabulary building.
> **Tone:** This module has the "wow" moments — lean into them.
> **Goal:** Everyone understands the distributional hypothesis, the Word2Vec
> intuition, and can load and use pretrained embeddings in code.

---

# ─────────────────────────────────────────────
# SESSION 1  (~90 min)
# "Words on a map — teaching the computer what words mean"
# ─────────────────────────────────────────────

## Before They Arrive (10 min setup)

**On your laptop, have ready:**
- Terminal open in `MLForBeginners/nlp/math_foundations/`
- The file `03_word_embeddings.py` open in your editor
- Visuals folder `nlp/visuals/03_word_embeddings/` open in Finder
- A large blank area on the whiteboard for the "word map" diagram

---

## OPENING  (10 min)

### Hook — The broken dictionary

> *"Last module, TF-IDF gave every word a vector in a 50,000-dimensional space.*
> *Here's the problem: ask TF-IDF how similar 'cat' and 'dog' are.*
>
> *cosine_similarity(cat_vector, dog_vector) = 0.0*
>
> *They're completely unrelated. Orthogonal. As far apart as 'cat' and 'algorithm'.*
>
> *But WE know: cats and dogs are both domestic animals, both pets, both
> featured in memes. They're much more similar to each other than to 'algorithm'.*
>
> *TF-IDF is a great frequency counter. But it's a broken dictionary.*
> *It has no idea what any word means.*
>
> *Today: we fix that. Word embeddings are how we teach the computer
> that 'cat' and 'dog' are neighbors. That 'king' and 'queen' are related.
> That 'Paris' is to 'France' as 'Rome' is to 'Italy'."*

**Write on board:**

```
TF-IDF:   "cat"  →  [0, 0, 1, 0, 0, 0, ..., 0]  (50000-d, one-hot, no meaning)
Embedding: "cat"  →  [0.32, -0.14, 0.78, ...]     (300-d, dense, meaningful)

cosine(cat, dog)  TF-IDF → 0.0    Embedding → 0.92  ✓
cosine(cat, car)  TF-IDF → 0.0    Embedding → 0.07  ✓
```

---

## SECTION 1: The Distributional Hypothesis  (20 min)

### "You shall know a word by the company it keeps"

> *"This is a quote from a linguist named J.R. Firth, 1957.*
> *The idea: words that appear in similar contexts have similar meanings.*
>
> *Let me show you."*

**Write on board:**

```
CONTEXTS FOR "cat":
  "the ___ sat on the mat"
  "feed the ___"
  "the ___ chased the mouse"
  "my ___ is sleeping on the sofa"
  "the ___ purred contentedly"

CONTEXTS FOR "dog":
  "the ___ sat by the door"
  "feed the ___"
  "the ___ chased the ball"
  "my ___ is sleeping by the fire"
  "the ___ barked at the mailman"
```

> *"Almost identical. 'Feed the ___' works for both. 'My ___ is sleeping' works
> for both. 'The ___ chased' works for both.*
>
> *This means in any large body of text, 'cat' and 'dog' appear in very similar
> surrounding contexts. A model trained to predict context words will therefore
> learn very similar internal representations for them.*
>
> *That's the distributional hypothesis — and it's the entire basis of Word2Vec,
> GloVe, and even modern BERT and GPT representations."*

**Ask the room:**

> *"What would be a word that appears in very DIFFERENT contexts from 'cat'?
> Think about what kinds of sentences it appears in."*

Good answers: algorithm, mortgage, parliament, neutron. Let them explain why.

> *"Correct — 'algorithm' never appears in 'my ___ is sleeping.'
> Their context distributions are completely different.
> So their embedding vectors will be far apart."*

### The word map analogy

**Draw a rough 2D map on the board:**

```
        ROYALTY
    king  queen
      \    /
       \/
       /\
      /  \
    man   woman
       (GENDER)

    paris  rome  london
       (CAPITALS)

    cat   dog  rabbit
       (PETS)

    car   bus  train
       (VEHICLES)
```

> *"Imagine placing every word in your vocabulary as a dot on a map.*
> *Similar words cluster together. Related words are near each other.*
> *Unrelated words are far apart.*
>
> *That map is what a word embedding IS.*
> *Each word's coordinates on that map = its embedding vector.*
> *300 dimensions instead of 2, but same idea."*

---

## SECTION 2: Word2Vec — The Idea  (25 min)

### Skip-gram architecture

> *"Word2Vec comes in two flavors: CBOW and Skip-gram.*
> *CBOW: given surrounding words, predict the center word.*
> *Skip-gram: given the center word, predict the surrounding words.*
> *Skip-gram is more commonly used — it handles rare words better."*

**Draw on board:**

```
TEXT: "the cat sat on the mat"

SKIP-GRAM TRAINING PAIRS (window size = 2):
  Center: "cat"  → Predict: ["the", "sat"]
  Center: "sat"  → Predict: ["cat", "on", "the"]
  Center: "on"   → Predict: ["sat", "the", "mat"]

SKIP-GRAM ARCHITECTURE:
  Input:  one-hot("cat")         →   [0,0,1,0,0,0,...]    (V-dim)
  W_in:   embedding matrix       →   shape (V × D)          ← THE EMBEDDINGS
  h:      embedding for "cat"    →   [0.32, -0.14, 0.78]   (D-dim)
  W_out:  output matrix          →   shape (D × V)
  Output: softmax over vocab     →   P("sat" | "cat") = 0.23, etc.

TRAINING: maximize P(context | center)
RESULT:   W_in rows = word embeddings (words used in similar contexts → similar rows)
```

> *"The brilliant insight: we never explicitly train a meaning-learning model.*
> *We train a PREDICTION model: predict which words appear nearby.*
> *As a side effect of getting good at prediction, the internal weights
> (W_in) become meaningful representations. The embedding is the side effect.*
>
> *It's unsupervised — no human labels needed. Just raw text.*
> *Google trained Word2Vec on 100 billion words from Google News.*
> *The resulting embeddings knew about royalty, geography, grammar."*

### The famous analogy

**Write on board:**

```
VECTOR ARITHMETIC IN EMBEDDING SPACE:

  vector("king") - vector("man") + vector("woman") ≈ vector("queen")

  vector("Paris") - vector("France") + vector("Italy") ≈ vector("Rome")

  vector("running") - vector("run") + vector("swim") ≈ vector("swimming")

This is MEANING encoded as GEOMETRY.
```

> *"This actually works. It's not hand-crafted. The model discovered it
> purely from patterns in text. 'King' and 'man' differ by exactly
> the gender direction in embedding space. Add 'woman' — you get 'queen'.*
>
> *When people first saw this in 2013, it felt like magic.*
> *It's not magic. It's a remarkable consequence of the distributional hypothesis
> applied at massive scale."*

**Ask the room:**

> *"Can you think of other analogies that should work in embedding space?
> Country:Capital? Animal:Sound? Job:Workplace?"*

Let them brainstorm for 3 minutes — it's energizing and fun.

---

## SECTION 3: Live Demo — Word2Vec Training  (15 min)

```bash
python3 03_word_embeddings.py
```

Walk through the output as it runs. Point at Section 2:

> *"This is training a tiny Word2Vec from scratch on 6 sentences.*
> *Not enough to be useful, but enough to see the mechanics.*
> *Look at the training loss decreasing — same gradient descent we've
> been doing since Part 1, just applied to a prediction task."*

Point at the cosine similarity outputs:

> *"Even with our tiny corpus, 'cat' and 'dog' have higher similarity
> than 'cat' and 'mat' — even though 'cat' and 'mat' rhyme and co-occur.*
> *The model has correctly learned animal kinship beats phonetic similarity."*

**Open the visuals:**

> *"Look at the PCA visualization. We've compressed 300 dimensions to 2
> so we can draw it. Words that should cluster together do cluster together.*
> *Even on our small corpus — animals together, actions together.*
> *On Google's 100B-word corpus, this map is breathtaking."*

---

## CLOSING SESSION 1  (10 min)

```
SESSION 1 RECAP:
─────────────────────────────────────────────
TF-IDF problem:
  "cat" and "dog" look identical (both unrelated to everything)

Distributional hypothesis:
  words in similar contexts have similar meaning
  "You shall know a word by the company it keeps" — Firth, 1957

Word2Vec:
  train model to predict context words from center word (Skip-gram)
  side effect: embedding matrix W_in = word meanings
  word = point in 300-dimensional space
  similar words = nearby points

Key result:
  king - man + woman ≈ queen
  arithmetic in embedding space = semantic relationships
─────────────────────────────────────────────
```

---
---

# ─────────────────────────────────────────────
# SESSION 2  (~90 min)
# "Pretrained embeddings, GloVe, FastText, and using them"
# ─────────────────────────────────────────────

## Opening  (10 min)

### The "don't train from scratch" lesson

> *"In Part 3, we learned that training a CNN from scratch on a small dataset
> gives mediocre results. We used transfer learning — pretrained ImageNet weights.*
>
> *Same principle here.*
>
> *Training Word2Vec on your 1,000 reviews? You'll get bad embeddings.*
> *Loading GloVe trained on 840 billion Wikipedia + web text tokens? Excellent.*
>
> *Today we learn how to use pretrained embeddings and why they're
> the default starting point for any NLP problem."*

---

## SECTION 4: Pretrained Embeddings — GloVe  (25 min)

### What is GloVe?

> *"GloVe — Global Vectors for Word Representation — comes from Stanford.*
> *Trained on 840B tokens from Common Crawl (the internet) and Wikipedia.*
> *Vocabulary of 2.2 million words. Each word → 300-dimensional vector.*
>
> *The key difference from Word2Vec: GloVe uses global co-occurrence statistics
> rather than a sliding window. But for practical purposes — same idea,
> similar quality, both widely used."*

**Write on board:**

```
AVAILABLE PRETRAINED EMBEDDINGS:
  GloVe (Stanford):  glove.6B.100d.txt  (6B tokens, 100-d vectors)
                     glove.840B.300d.txt (840B tokens, 300-d vectors)
  Word2Vec (Google): word2vec-google-news-300  (3M words, 300-d)
  FastText (Meta):   wiki-news-300d-1M          (1M words, 300-d)

LOADING IN PYTHON:
  glove = {}
  with open("glove.6B.100d.txt") as f:
      for line in f:
          parts = line.split()
          word = parts[0]
          vector = np.array(parts[1:], dtype=float)
          glove[word] = vector

  # Now: glove["cat"] = array([ 0.32, -0.14, ...], shape=(100,))
```

Point at Section 3 of the script output:

> *"Our script loads a mini version of GloVe with 50 representative words.*
> *Look at the nearest neighbors for 'cat': dog, rabbit, kitten, puppy.*
> *For 'king': queen, prince, throne, royal.*
> *The pretrained model has absorbed 840 billion tokens of human knowledge."*

### Analogies in action

Point at the analogy output:

> *"king - man + woman. The script computes the vector arithmetic and then
> finds the nearest word in the vocabulary.*
> *Result: 'queen'. Every time.*
>
> *This is one of the most reproduced results in ML.*
> *The geometry of language. Encoded in floating point numbers."*

**Ask the room:**

> *"What would fail? When would embedding arithmetic break down?"*

Good answers:
- Ambiguous words ("bank" = river bank vs financial institution)
- Cultural context that changed (slang)
- Domain-specific jargon not in the training corpus

---

## SECTION 5: FastText — Subword Embeddings  (15 min)

### The OOV problem — solved

> *"GloVe and Word2Vec have a fatal flaw: OOV.*
> *A word not in their training vocabulary gets no vector.*
> *'cryptocurrency', 'COVID', 'deepfake' — all OOV in 2013 GloVe.*
>
> *FastText solves this with subword embeddings.*
> *Instead of learning one vector per word, it learns vectors for character n-grams.*
> *The word vector = sum of its subword vectors."*

**Write on board:**

```
FastText for "playing":
  "playing" → subwords of length 3-6:
  pla, lay, ayi, yin, ing, play, layi, ayin, ying,
  playi, layin, aying, laying, playing

  vector("playing") = mean([v("pla"), v("lay"), ..., v("playing")])

ADVANTAGE: "playful" shares subwords with "playing"
           → similar vectors even if "playful" was rare in training

EVEN BETTER: "COVID19" never seen in training
           → subwords might include "vid", "cov" etc.
           → at least some signal vs pure zero for OOV words
```

> *"FastText is Meta's (Facebook) contribution. It's the recommended default
> when you have domain-specific text with unusual vocabulary.*
> *Medical, legal, code — any specialized domain benefits from subword embeddings."*

---

## SECTION 6: Visualizing Embedding Space — t-SNE and PCA  (15 min)

Open the visuals:

> *"We can't visualize 300 dimensions. So we use dimensionality reduction.*
>
> *PCA: finds the two directions that preserve the most variance.*
> *Fast, linear, deterministic.*
>
> *t-SNE: finds a 2D layout that preserves NEIGHBORHOODS.*
> *Slow, nonlinear, stochastic (run it twice, get slightly different results).*
> *But the clusters it reveals are more interpretable."*

Point at the t-SNE plot in the visuals:

> *"See how royalty words cluster? Animals cluster? Capitals cluster?*
> *This is 300 dimensions projected to 2 by an algorithm trying to preserve
> which words are near each other.*
>
> *The king-queen gender direction is actually visible as a consistent
> offset between male and female words.*
>
> *This is geometry. This is what the model learned from text alone."*

---

## CLOSING SESSION 2  (10 min)

### Recap board

```
SESSION 2 RECAP:
─────────────────────────────────────────────
Pretrained Embeddings:
  train once on massive corpus, reuse everywhere
  GloVe: global co-occurrence, 840B tokens
  Word2Vec: skip-gram, 100B Google News tokens
  FastText: subword, handles OOV words

Loading in production:
  glove["cat"] = np.array([...])   # 300-dim vector
  Use as lookup table in your pipeline

Embedding in Keras:
  Embedding(vocab_size, embed_dim)
  trainable=True  → learn from scratch
  trainable=False → freeze pretrained weights
  Load GloVe → fine-tune on your task

Visualization:
  PCA  → fast, linear, preserves variance
  t-SNE → slow, nonlinear, preserves neighborhoods
─────────────────────────────────────────────
```

### The road ahead

```
WHERE WE ARE:
  ✅ Module 1: Text Preprocessing
  ✅ Module 2: BoW + TF-IDF
  ✅ Module 3: Word Embeddings (this module)

PROBLEM WE'VE JUST NOTICED:
  We now have great word vectors.
  But we still have no way to handle WORD ORDER.

  "The dog bit the man" and "The man bit the dog":
  Both have identical sets of word vectors.
  Embeddings don't know which came first.

NEXT UP:
  → Module 4: RNN Intuition
    Read sequences left-to-right.
    Maintain a "memory" of what was seen.
    Handle position and order.
```

---

## Homework

```
EXERCISES:

1. The word map:
   Load the GloVe mini-embeddings from the script
   (or use the provided approximate vectors in Section 3 output).

   Pick 5 pairs of words you'd expect to be similar
   and 5 pairs you'd expect to be different.
   Compute cosine similarities and see if your intuitions hold.

2. Vector arithmetic:
   Try these analogies using the pretrained embeddings:
     a) "man" - "king" + "queen" → should ≈ "woman"
     b) "walk" - "walking" + "swimming" → should ≈ "swim"
     c) "france" - "paris" + "london" → should ≈ "england" or "britain"

   Do they work? Which ones fail? Why do you think that is?

3. Reflection:
   You're building a sentiment analyzer for medical reviews.
   (Patients reviewing hospitals, doctors, treatments.)
   Would you use GloVe, Word2Vec, or FastText?
   Would you use pretrained weights or train from scratch?
   Write 3-4 sentences justifying your choice.
```

---

# ─────────────────────────────────────────────
# INSTRUCTOR TIPS & SURVIVAL GUIDE
# ─────────────────────────────────────────────

## When People Get Confused

**"Why does vector arithmetic work? That seems like magic."**
> *"It's not magic — it's a consequence of training at scale.*
> *Think about it: in 840B words of text, every time 'king' appears,
> similar words appear around it as when 'queen' appears, BUT with one consistent
> difference: the surrounding 'he'/'his' vs 'she'/'her' words.*
> *The model encodes that gender direction as a consistent offset vector.*
> *At enough scale, these patterns become reliable geometric relationships."*

**"When would you ever train embeddings from scratch?"**
> *"Domain-specific text that is very different from the web.*
> *Medical notes, legal contracts, code, scientific papers, ancient texts.*
> *GloVe trained on Wikipedia won't capture the relationships in pathology reports.*
> *In those cases, train from scratch on your domain data, or use
> BioBERT / LegalBERT — domain-specific pretrained models."*

**"What's the difference between an embedding and the output of a DNN?"**
> *"Great question. In our Part 3 DNNs, every layer output is technically
> a 'representation' or 'embedding' in the loose sense.*
> *Word embeddings specifically are trained so that GEOMETRIC DISTANCE = SEMANTIC DISTANCE.*
> *That's the special property. Regular DNN hidden states don't have that constraint."*

**"Do embeddings capture grammar?"**
> *"Partially. They capture morphology well (run ≈ running in structure).*
> *They capture some syntax (adjective vectors cluster together).*
> *But they're weak on sentence-level grammar — that's what attention and
> transformers in Part 6 are designed for."*

## Energy Management

- **The king-queen analogy moment** is the high point of this session. Slow down there.
- **t-SNE plots** are almost always a crowd pleaser. Open them full screen.
- **If someone asks about BERT:** "BERT does this at the sentence level.
  Instead of word2vec's fixed vector per word, BERT gives different vectors
  for the same word in different contexts. 'bank' the river and 'bank' the institution
  get completely different BERT vectors. That's Part 6."

---

# ─────────────────────────────────────────────
# QUICK REFERENCE — Session Timing
# ─────────────────────────────────────────────

```
SESSION 1  (90 min)
├── Opening hook — broken dictionary   10 min
├── Section 1: Distributional hyp.    20 min
├── Section 2: Word2Vec / Skip-gram   25 min
├── Section 3: Live demo              15 min
└── Close + preview                   20 min

SESSION 2  (90 min)
├── "Don't train from scratch" lesson  10 min
├── Section 4: GloVe + analogies      25 min
├── Section 5: FastText subword       15 min
├── Section 6: PCA / t-SNE visuals    15 min
├── Closing recap board               10 min
└── Homework + road ahead             15 min
```

---

*Generated for MLForBeginners — Module 03 · Part 5: NLP*
