# MLForBeginners — Instructor Guide
## Part 5, Module 2: Bag of Words & TF-IDF  ·  Two-Session Teaching Script

> **Who this is for:** You, teaching close friends who completed Module 1 (Text Processing).
> **They already know:** Tokenization, vocabulary building, integer encoding.
> **Tone:** Casual and mathematical — this is where the elegance kicks in.
> **Goal:** Everyone understands BoW, TF-IDF from first principles, and
> can build both from scratch and verify against scikit-learn.

---

# ─────────────────────────────────────────────
# SESSION 1  (~90 min)
# "Turning text into numbers — the Bag of Words way"
# ─────────────────────────────────────────────

## Before They Arrive (10 min setup)

**On your laptop, have ready:**
- Terminal open in `MLForBeginners/nlp/math_foundations/`
- The file `02_bag_of_words_tfidf.py` open in your editor
- Visuals folder `nlp/visuals/02_bag_of_words_tfidf/` open in Finder
- A whiteboard with enough space for a matrix diagram

---

## OPENING  (10 min)

### Hook — The search engine question

> *"Here's a puzzle. Google indexes hundreds of billions of web pages.*
> *When you type 'best pasta recipe', it comes back in 0.3 seconds with
> the most relevant pages in the world.*
>
> *How does it know which pages are relevant?*
> *It can't read. It sees bytes, not meaning.*
>
> *The answer — at least conceptually — is that every page gets turned into
> a vector of numbers. Your query also becomes a vector. And then it's just
> a geometry problem: find the vectors closest to your query vector.*
>
> *Today we build the first version of that. It's called Bag of Words.
> It's fifty years old and still gets used in production every day."*

**Draw on the board:**

```
document → vector of numbers → compare vectors → find similar docs
   ↑
That's the entire idea.
```

---

## SECTION 1: Bag of Words From Scratch  (25 min)

### The core idea

**Write on the board:**

```
CORPUS (our document collection):
  doc0: "the cat sat on the mat"
  doc1: "the dog sat on the log"
  doc2: "the cat and dog are friends"

VOCABULARY (sorted unique words):
  [and, are, cat, dog, friends, log, mat, on, sat, the]
   0    1    2    3    4        5    6    7   8    9

DOCUMENT-TERM MATRIX:
         and  are  cat  dog  friends  log  mat  on  sat  the
  doc0:   0    0    1    0      0      0    1    1   1    2
  doc1:   0    0    0    1      0      1    0    1   1    2
  doc2:   1    1    1    1      1      0    0    0   0    1
```

> *"Each document becomes a row. Each word in the vocabulary is a column.*
> *The value is how many times that word appears in that document.*
>
> *We call it a 'bag' because ORDER IS GONE.*
> *'Dog bites man' and 'Man bites dog' produce IDENTICAL vectors.*
> *The bag has the same items regardless of how you arrange them."*

**Ask the room:**

> *"Is that a problem? When would losing order matter?
> When would it NOT matter?"*

Let them think. Good answers:
- Matters: sentiment ("not good" vs "good not"), questions
- Doesn't matter: spam detection, topic classification, document similarity

> *"For topic classification — is this article about sports, politics, or tech? —
> word counts without order work extremely well. We'll see that in Module 5.*
>
> *For understanding 'I didn't love this movie' vs 'I loved this movie'?
> BoW fails. That's why we have LSTMs and transformers."*

### The sparsity problem

**Point at the matrix you drew:**

> *"Look at how many zeros there are. Real vocabulary: 50,000 words.*
> *A typical document uses maybe 200 unique words.*
> *That means each document vector is 49,800 zeros and 200 small numbers.*
>
> *This is called a SPARSE matrix. 99.6% zeros.*
> *It's actually fine computationally — sparse matrix formats store only the
> non-zeros. But it hints at a deeper problem: raw counts mislead."*

---

## SECTION 2: Why Raw Counts Mislead  (15 min)

### The "the" problem

> *"Let's say you have 100 news articles. The word 'the' appears in every
> single one of them, hundreds of times each. According to raw counts,
> 'the' is the most important word in your entire corpus.*
>
> *But 'the' tells you absolutely nothing about what any specific article is
> ABOUT. It's useless for distinguishing a sports article from a tech article.*
>
> *Meanwhile, the word 'quarterback' appears in only 3 articles —
> and those 3 articles are definitely about American football."*

**Write on board:**

```
PROBLEM WITH RAW COUNTS:
  "the"          → appears everywhere → completely uninformative
  "quarterback"  → appears in 3 docs  → extremely informative

Raw counts REWARD common words.
We want to REWARD RARE words that are specific to a document.
```

> *"This insight — that rare words are more informative than common words —
> is the entire mathematical intuition behind TF-IDF.*
> *Let's derive it."*

---

## SECTION 3: TF-IDF From First Principles  (25 min)

### Term Frequency (TF)

**Write on board:**

```
TERM FREQUENCY (TF):
  How often does word w appear in document d?
  Normalized by document length so long docs don't dominate.

  TF(w, d) = count(w in d) / total_words_in_d

  Example: "the" appears 5 times in a 50-word document
  TF("the", doc) = 5 / 50 = 0.10
```

> *"TF is just the frequency of a word in a single document, normalized
> so a 1000-word document isn't automatically ranked higher than a 50-word one."*

### Inverse Document Frequency (IDF)

> *"Now the clever part. We want words that are RARE ACROSS THE WHOLE CORPUS
> to get a HIGHER score. Words that appear everywhere get a LOWER score.*
>
> *How do we encode 'how surprising is this word'?
> We use a logarithm of a ratio."*

**Write on board — build it step by step:**

```
IDF(w) = log( N / df(w) )

  N     = total number of documents in corpus
  df(w) = number of documents that contain word w

EXAMPLES (N = 100 documents):
  "the"          → df = 100  → IDF = log(100/100) = log(1.0) = 0.0
  "algorithm"    → df = 20   → IDF = log(100/20)  = log(5.0) ≈ 1.61
  "quarterback"  → df = 3    → IDF = log(100/3)   = log(33.3)≈ 3.51
  "backpropagation" → df = 1 → IDF = log(100/1)   = log(100) = 4.61

INTERPRETATION:
  IDF = 0.0  →  completely useless (in every document)
  IDF = 4.6  →  extremely specific (in only 1 document)
  IDF = "how surprising is it to see this word?"
```

> *"The logarithm prevents extreme cases. A word in 1 document vs 2 documents
> shouldn't get double the weight. Log compresses the scale.
> This is the same intuition as decibels for sound.*
>
> *Side note: in practice, you add 1 to avoid log(0) when a word appears
> in every document. scikit-learn uses log(1 + N / (1 + df)) + 1.
> Same idea, just more numerically stable."*

### TF-IDF = TF × IDF

**Write on board:**

```
TF-IDF(w, d) = TF(w, d)  ×  IDF(w)

              = (word frequency in doc)
                ×  (how surprising is this word across corpus)

HIGH TF-IDF:  word appears often in THIS doc, rarely in other docs
              → this word is CHARACTERISTIC of this document

LOW TF-IDF:   word appears everywhere (stopwords)
              OR word appears rarely in this doc
              → this word is NOT characteristic of this document
```

**Ask the room:**

> *"So TF-IDF of 'the' in any document will always be exactly what?"*

Wait for "zero." Confirm.

> *"Zero. Because IDF('the') = 0. Multiplied by anything = 0.*
> *The math automatically handles what we'd do manually with stopwords.*
> *This is why TF-IDF is so elegant."*

---

## CLOSING SESSION 1  (5 min)

```
SESSION 1 RECAP:
─────────────────────────────────────────────
Bag of Words:
  document → count vector over vocabulary
  "bag" = order lost, only counts remain
  sparse: most entries are zero

TF-IDF:
  TF  = how often word appears in THIS doc
  IDF = log(N / df) = how surprising is this word?
  TF-IDF = TF × IDF

  high → word is characteristic of this doc
  zero → word appears everywhere (stopwords)
─────────────────────────────────────────────
```

---
---

# ─────────────────────────────────────────────
# SESSION 2  (~90 min)
# "From TF-IDF vectors to document similarity and classification"
# ─────────────────────────────────────────────

## Opening  (10 min)

### Quick check

> *"Before we go further — can someone tell me in one sentence why TF-IDF
> gives 'the' a score of zero?"*

Expected answer: IDF('the') is log(N/N) = log(1) = 0.

> *"Perfect. Today we're going to: (1) implement TF-IDF from scratch,
> (2) verify it against scikit-learn, (3) use it to compare documents,
> and (4) use it to classify text. This is a complete practical workflow."*

---

## SECTION 4: Implement TF-IDF + Verify Against sklearn  (20 min)

> *"Let's run the script and walk through the scratch implementation."*

```bash
python3 02_bag_of_words_tfidf.py
```

As it runs, point at Section 3 output:

> *"See our scratch implementation vs sklearn side by side.*
> *They match — good. The slight differences come from sklearn's
> smoothing formula. Both give the same relative rankings, which
> is what matters for classification."*

Point at the document-term matrix printout:

> *"Look at Section 4. 'The' has TF-IDF zero in every document.*
> *'Friends' has a high score only in doc2 where it appears.*
> *That's the TF-IDF contract: high score = this word is distinctive
> for this specific document."*

---

## SECTION 5: Cosine Similarity  (20 min)

### Why cosine, not Euclidean distance?

**Draw on board:**

```
SHORT DOC:  [0, 3, 0, 2, 0]      (3 uses of word A, 2 of word B)
LONG DOC:   [0, 9, 0, 6, 0]      (9 uses of word A, 6 of word B)

Euclidean distance: very large (vectors are far apart)
Cosine similarity:  = 1.0 (vectors point in EXACT same direction)

The long doc is just the short doc repeated three times.
Same topic. Cosine correctly identifies them as identical.
Euclidean doesn't.
```

> *"Cosine similarity measures the ANGLE between two vectors.*
> *Not their magnitude — their direction.*
> *A document using 'machine' and 'learning' 10 times is the same topic
> as one using them 100 times. Cosine sees through the length difference.*
>
> *Formula:*"

**Write on board:**

```
cos(A, B) = (A · B) / (|A| × |B|)

         = dot product
           ────────────────
           product of norms

Range: -1 to +1  (with TF-IDF vectors: 0 to +1, all positive)
1.0 = identical direction (same topic)
0.0 = completely orthogonal (different topics)
```

**Ask the room:**

> *"Which pair of our 6 documents should have the highest cosine similarity?
> The two cat documents? The two dog-related ones?"*

Let them predict. Then show the similarity matrix from the script output.

> *"This is how recommendation engines work. Your watch history becomes a vector.
> Every movie is a vector. Recommend the movies with highest cosine similarity
> to your history vector. Same math, different domain."*

---

## SECTION 6: N-grams  (10 min)

### Beyond single words

> *"BoW treats every word independently.*
> *But 'New York' means something completely different from 'new' and 'York'.*
> *'Machine learning' is a concept that neither 'machine' nor 'learning' alone captures.*
>
> *N-grams let us use PAIRS (bigrams) or TRIPLES (trigrams) of words as tokens."*

**Write on board:**

```
"machine learning is great"

UNIGRAMS (n=1):  machine, learning, is, great
BIGRAMS  (n=2):  machine_learning, learning_is, is_great
TRIGRAMS (n=3):  machine_learning_is, learning_is_great

In sklearn: TfidfVectorizer(ngram_range=(1, 2))
  → uses both unigrams and bigrams
  → vocabulary explodes but captures phrases

COMMON PRACTICE: ngram_range=(1, 2)
  → captures single words AND adjacent pairs
  → good tradeoff between coverage and vocabulary size
```

> *"N-grams are one of the cheapest ways to improve classifier accuracy on
> text data. Going from unigrams to bigrams typically adds 1-3% accuracy
> with no architectural change."*

---

## SECTION 7: BoW/TF-IDF for Classification  (15 min)

> *"Let's put it all together. TF-IDF vectors → Logistic Regression.*
> *This is the industry standard baseline for text classification.*
> *It's simple, interpretable, and surprisingly hard to beat."*

Point at Section 7 output:

> *"Look at the top features per class the script prints.*
> *For sports: 'game', 'team', 'player', 'championship'.*
> *For tech: 'software', 'algorithm', 'data', 'model'.*
>
> *You can read what the model learned. This interpretability is one of
> the biggest advantages of TF-IDF + Logistic Regression.*
> *With a deep learning model you'd have to use SHAP or attention maps
> to get this. Here it's just the top coefficients."*

**Ask the room:**

> *"When would you use TF-IDF + LogReg over an LSTM or BERT?
> Think about speed, interpretability, data size."*

Guide them to:
- Small datasets → TF-IDF wins (LSTMs overfit without data)
- Need interpretability → TF-IDF wins
- Latency-critical systems → TF-IDF wins
- Accuracy matters most, have data and compute → BERT/LSTM wins

---

## CLOSING SESSION 2  (10 min)

### Recap board

```
SESSION 2 RECAP:
─────────────────────────────────────────────
TF-IDF Implementation:
  scratch implementation ≈ sklearn (same math)
  smoothing prevents log(0)

Cosine Similarity:
  measures direction not magnitude
  perfect for comparing documents of different lengths
  foundation of recommendation systems

N-grams:
  bigrams capture phrases ("machine_learning")
  ngram_range=(1,2) is the standard starting point

Classification:
  TF-IDF → TfidfVectorizer → sparse matrix
  sparse matrix → LogisticRegression → predictions
  top coefficients = most important words per class
─────────────────────────────────────────────
```

### The road ahead

```
WHERE WE ARE:
  ✅ Module 1: Text Preprocessing
  ✅ Module 2: BoW + TF-IDF (this module)

LIMITATION WE'RE ABOUT TO FIX:
  TF-IDF knows "cat" and "dog" are different positions in vocabulary.
  But it doesn't know they're RELATED (both animals).
  cosine_similarity(cat_vector, dog_vector) = 0 in TF-IDF space.

NEXT UP:
  → Module 3: Word Embeddings
    "cat" → [0.32, -0.14, 0.78, ...]  (300 numbers)
    "dog" → [0.29, -0.11, 0.81, ...]  (similar numbers!)
    cosine_similarity(cat, dog) ≈ 0.92  ✓
```

---

## Homework

```
EXERCISES (pick 2 of 3):

1. TF-IDF by hand:
   Documents:
     d1 = "I love cats and cats love me"
     d2 = "dogs love to play fetch"
     d3 = "cats and dogs are both great pets"

   Compute TF-IDF for 'cats' in d1. Then for 'love' in d1.
   Which is higher? Why?

2. Cosine similarity:
   A = [1, 2, 3]
   B = [2, 4, 6]    (A × 2)
   C = [3, 0, 0]

   Compute cosine_similarity(A, B) and cosine_similarity(A, C).
   Explain the result in plain English.

3. Build it:
   Take 10 of your own text examples (any topic).
   Run TF-IDF on them using sklearn's TfidfVectorizer.
   Print the top 5 most distinctive words for 3 different documents.
   Do the results make intuitive sense?
```

---

# ─────────────────────────────────────────────
# INSTRUCTOR TIPS & SURVIVAL GUIDE
# ─────────────────────────────────────────────

## When People Get Confused

**"Why use log in IDF?"**
> *"Without log: a word in 1 doc (out of 100) gets IDF = 100.*
> *A word in 2 docs gets IDF = 50. The ratio is 2x for just one extra doc.*
> *With log: 1 doc → IDF = 4.6, 2 docs → IDF = 3.9. Much more stable.*
> *Log compresses extreme values the same way decibels compress sound intensity."*

**"Why not just use TF or just IDF?"**
> *"TF alone: long documents get artificially high scores.*
> *IDF alone: ignores how often the word actually appears in the document.*
> *TF × IDF is the right tradeoff: this word is frequent HERE and rare ELSEWHERE."*

**"Isn't cosine similarity just correlation?"**
> *"Very related! Pearson correlation measures linear relationship between variables.*
> *Cosine similarity measures the angle between vectors.*
> *For zero-mean vectors, they're mathematically identical.*
> *For TF-IDF vectors (always positive), cosine is the better choice."*

**"When should I NOT remove stopwords with TF-IDF?"**
> *"TF-IDF handles it automatically — stopwords get IDF ≈ 0.*
> *But for very small corpora where a stopword might not appear in every document,
> you might still want manual stopword removal as a safety net."*

## Energy Management

- **Best engagement point:** The cosine similarity matrix — seeing which documents
  are similar is satisfying and visually intuitive.
- **If they're getting lost in the math:** Say "here's the punchline — rare words
  in this doc get high scores, common words get zero. Everything else is mechanics."
- **Strongest real-world hook:** Google PageRank and TF-IDF are often mentioned together.
  Mention that BM25 (what modern search uses) is a direct descendant of TF-IDF.

---

# ─────────────────────────────────────────────
# QUICK REFERENCE — Session Timing
# ─────────────────────────────────────────────

```
SESSION 1  (90 min)
├── Opening hook — search engine       10 min
├── Section 1: BoW from scratch        25 min
├── Section 2: Why raw counts mislead  15 min
├── Section 3: TF-IDF derivation       25 min
└── Close + preview                    15 min

SESSION 2  (90 min)
├── Quick check                        10 min
├── Section 4: TF-IDF + sklearn verify 20 min
├── Section 5: Cosine similarity       20 min
├── Section 6: N-grams                 10 min
├── Section 7: Classification          15 min
└── Close + homework + road ahead      15 min
```

---

*Generated for MLForBeginners — Module 02 · Part 5: NLP*
