# MLForBeginners — Instructor Guide
## Part 5 · Module 06: Sentiment Analysis
### Two-Session Teaching Script

> **Prerequisites:** Module 05 complete (sklearn Pipeline, multi-class text
> classification). They also know TF-IDF, logistic regression, and have
> a basic intuition for word embeddings from Module 03.
> **Payoff today:** Three complete sentiment systems — rule-based, ML-based,
> and transformer-based — all compared live on sentences the class suggests.

---

# SESSION 1 (~90 min)
## "Reading emotion — rule-based VADER and TF-IDF + Logistic Regression"

## Before They Arrive
- Terminal open in `nlp/algorithms/`
- Whiteboard ready
- Post on the wall: "Is this positive or negative? → 'The food was great but the service was terrible'"

---

## OPENING (10 min)

> *"Before class I want everyone to think of one sentence — from a tweet,
> a review, a text message — that would be genuinely hard for a computer
> to classify as positive or negative. Hold on to it.*
>
> *By the end of today's session we will run all your sentences through three
> different systems and see which one gets it right."*

Write on board:
```
SENTIMENT ANALYSIS REAL APPLICATIONS:
  Brand monitoring:  "What are people saying about our product on Twitter?"
  Finance:           "Is this earnings call transcript positive or negative?"
  Customer support:  "Flag angry emails for escalation automatically"
  Movie studios:     "Predict box office from early review sentiment"
  HR:                "Analyse employee survey free-text responses at scale"
```

> *"Every company that has customers and public feedback uses this.
> Today you are going to understand exactly how it works — and where it fails."*

---

## SECTION 1: Rule-Based Sentiment — VADER (20 min)

> *"VADER stands for Valence Aware Dictionary and sEntiment Reasoner.
> It was published in 2014 and is still widely used because it is fast,
> requires no training data, and works especially well on social media text.*
>
> *It has a lexicon of about 7,500 words, each with a score from -4 to +4.
> And it has rules for things a bag-of-words model cannot capture."*

Write on board:
```
VADER SCORING RULES:

  1. Lexicon score:
     "excellent" = +3.5    "terrible" = -3.1

  2. CAPITALIZATION:
     "EXCELLENT" > "excellent"   (intensity boost)

  3. Punctuation:
     "excellent!!!" > "excellent"  (more exclamation = stronger)

  4. Degree modifiers (intensifiers / dampeners):
     "very good"     > "good"       (amplify by ~1.3x)
     "kind of good"  < "good"       (dampen by ~0.74x)

  5. Negation:
     "not good"      → flip polarity, dampen by ~0.74x
     "never good"    → same
     "not very good" → applies to 'very good' combined

  Output: { "pos": 0.62, "neg": 0.10, "neu": 0.28, "compound": 0.78 }
  compound ranges from -1.0 (most negative) to +1.0 (most positive)
```

**Live demo — negation handling:**

Write these on the board one at a time, ask the room to predict the score,
then show the rule-based calculation:
```
"The food is good"          → compound: +0.43
"The food is not good"      → compound: -0.31   (negation flips)
"The food is not very good" → compound: -0.37   (negation + intensifier)
"The food is AMAZING"       → compound: +0.63   (caps boost)
"The food is amazing!!!"    → compound: +0.68   (punctuation boost)
```

> *"Negation is tricky. VADER handles it with a simple rule: if 'not', 'never',
> 'no' appear in the three words before a sentiment word, flip the polarity.*
>
> *This works for simple cases. It fails for sarcasm:
> 'Oh great, another delayed flight' scores as positive. We will come back
> to that failure mode."*

**Ask the room:** *"Give me a sentence that would fool a purely lexicon-based
system."*

Collect 2-3 examples. Good answers include:
- "Not bad" (double negative = positive, VADER handles it; naive systems don't)
- "This movie is so bad it's good" (mixed signals)
- Sarcasm: "Oh brilliant, exactly what I needed on a Monday morning"

---

## SECTION 2: TF-IDF + Logistic Regression for Sentiment (25 min)

> *"Rule-based works but requires expert-crafted lexicons. The ML approach
> learns from labelled examples. You show it thousands of positive and negative
> reviews, and it figures out the patterns on its own.*
>
> *Same pipeline we built in Module 05 — but now binary: positive or negative."*

Write on board:
```
TRAINING DATA                         LEARNED PATTERNS
"This movie is excellent"  → POS      "excellent"    w = +2.1
"Terrible acting, hated it"→ NEG      "terrible"     w = -2.4
"Great story and visuals"  → POS      "great"        w = +1.9
"Worst film of the year"   → NEG      "worst"        w = -2.8
"Not bad at all"           → POS      "not bad"      w = +0.9
                                      "not" alone    w = -0.3
```

> *"Notice 'not bad' — with bigrams (ngram_range=(1,2)) the vectoriser
> creates a feature for the PAIR 'not bad', which is positive.
> Without bigrams, the model sees 'not' (negative weight) and 'bad'
> (negative weight) separately — and misclassifies it.*
>
> *This is why ngram_range=(1,2) matters so much for sentiment."*

**Live walkthrough — build and evaluate:**
```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assume X = list of review texts, y = list of 0/1 labels
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
    ("clf",   LogisticRegression(C=1.0, max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipe.fit(X_train, y_train)
print(classification_report(y_test, pipe.predict(X_test)))
```

**Ask the room:** *"If our model gets 90% on positive reviews but only 75%
on negative reviews, what might be causing the gap?"*

Discussion points:
- Positive reviews may use more consistent vocabulary
- Negative reviews often use hedged language ("could have been better")
- Training set might have more positive examples — check class balance

---

## SECTION 3: Aspect-Based Sentiment (15 min)

> *"So far we classify the whole document: positive or negative overall.
> But consider this sentence:*
>
> *'The food was absolutely fantastic but the service was incredibly rude and
> the wait time was unacceptable.'*
>
> *Overall: negative? Positive? Both? That depends on what you are analysing."*

Write on board:
```
ASPECT-BASED SENTIMENT ANALYSIS (ABSA)

  Sentence: "The food was fantastic but the service was rude"

  Aspects:  food    → POSITIVE (+0.85)
            service → NEGATIVE (-0.72)
            overall → MIXED

  Applications:
    Hotel reviews   → cleanliness / location / staff / price
    Product reviews → battery / camera / display / price
    Restaurant      → food / service / atmosphere / value
```

> *"The simplest ABSA approach: define aspect keywords, extract sentences
> containing each keyword, run your sentiment model per sentence.*
>
> *This is not perfect but it is surprisingly useful and completely interpretable."*

Draw the approach:
```
Full review
    |
    v
Split into sentences
    |
    v
For each sentence:
  Does it contain "food", "meal", "dish"?   → food aspect
  Does it contain "staff", "waiter", "service"? → service aspect
    |
    v
Run sentiment classifier on that sentence
    |
    v
Return: { "food": "positive", "service": "negative" }
```

---

## CLOSING SESSION 1 (10 min)

Board summary:
```
THREE APPROACHES SO FAR:
  Rule-based (VADER):
    + No training data needed
    + Handles caps, punctuation, intensifiers, negation
    - Fails on sarcasm, domain-specific language, novel slang

  TF-IDF + Logistic Regression:
    + Learns from your data
    + Fast, interpretable (top features)
    + Bigrams help with negation
    - Still bag-of-words — loses sentence structure

NEGATION IS THE HARD PROBLEM:
  "not good"     → VADER handles it
  "not bad"      → needs bigrams for TF-IDF
  "never been worse" → both handle it
  "so bad it's good" → nobody handles it well
```

**Homework:** Find three reviews online (movie, restaurant, product) that
you think would fool a sentiment classifier. Bring them to Session 2.

---

# SESSION 2 (~90 min)
## "The transformer approach — HuggingFace pipeline + live class demo"

## OPENING (10 min)

> *"You brought your tricky sentences. Perfect. We are going to run them through
> all three systems today — rule-based, ML, and a pretrained BERT model —
> and see which one wins.*
>
> *But first: why does BERT do better? That is what we are going to understand
> before the fun part."*

---

## SECTION 1: Why Transformers Win at Sentiment (20 min)

> *"BERT was pretrained on the entire English Wikipedia and Books Corpus —
> 3.3 billion words. During pretraining it learned to predict masked words
> in context.*
>
> *As a side effect, it learned what words mean in context.
> 'Bank' as in river bank vs bank account. 'Sick' as in ill vs slang for cool.*
>
> *Then we fine-tune it on sentiment-labelled reviews. The pretrained context
> understanding makes it dramatically better on edge cases."*

Write on board:
```
TF-IDF + LR SEES:                   BERT SEES:

"not bad at all"                    "not bad at all"
  → [not: -0.3] [bad: -1.1]         → Full sentence as context
  → score: -1.4  (WRONG)            → 'not bad' = informal positive
  → prediction: NEGATIVE             → prediction: POSITIVE  (correct)

"sick performance"
  → [sick: -0.9] [performance: 0.1]  → gaming/music context = "amazing"
  → prediction: NEGATIVE              → prediction: POSITIVE (correct)
```

> *"BERT reads the entire sentence bidirectionally — it uses the words BEFORE
> and AFTER each word to understand it. TF-IDF never sees word order at all."*

---

## SECTION 2: HuggingFace Sentiment Pipeline (15 min)

> *"HuggingFace's `pipeline()` function is a one-liner that downloads a
> pretrained fine-tuned BERT model and wraps it for inference."*

Write on board:
```python
from transformers import pipeline

# Downloads ~268MB model on first run
sentiment = pipeline("sentiment-analysis")

result = sentiment("This film was surprisingly moving despite the slow start")
# Returns: [{'label': 'POSITIVE', 'score': 0.997}]

result = sentiment("Not bad at all, actually quite enjoyed it")
# Returns: [{'label': 'POSITIVE', 'score': 0.984}]
```

> *"The score is a confidence. Above 0.95 is very confident. Below 0.6 means
> the model is genuinely unsure — treat that as a signal to review manually.*
>
> *The default model is 'distilbert-base-uncased-finetuned-sst-2-english'.
> DistilBERT is a smaller, faster version of BERT — 40% smaller, 60% faster,
> 97% of BERT's accuracy on sentiment."*

---

## SECTION 3: Live Class Demo — Run Your Sentences (20 min)

Collect sentences from the room. Run each through all three systems.

Suggested format — write this on the board as a table:

```
SENTENCE                              VADER     TF-IDF+LR   BERT
"Not bad at all"                      POS?      NEG?        POS?
"The movie was so bad it was funny"   ???       ???         ???
"I can't say I didn't enjoy it"       ???       ???         ???
"Meh"                                 ???       ???         ???
[Student sentence 1]                  ???       ???         ???
[Student sentence 2]                  ???       ???         ???
```

> *"Let's run each one and fill in the table.*
>
> *Pay attention to WHERE the systems disagree. Disagreement tells you
> exactly which linguistic phenomena the simpler models cannot handle."*

Run the module to show all three:
```bash
python3 nlp/algorithms/sentiment_analysis.py
```

After the run:
> *"BERT wins on almost every hard case. But it is 100x slower, requires
> more memory, and needs a GPU in production. For a system that needs to
> process millions of tweets per hour, TF-IDF + LR is the realistic choice.*
>
> *This is the trade-off you will face in every real NLP project."*

---

## SECTION 4: Evaluation — ROC Curve and Confidence Calibration (15 min)

> *"Accuracy alone does not tell the full story for sentiment.*
>
> *Suppose our model says 'positive' for 60% of reviews and gets 88% accuracy.
> But 87% of ALL reviews in our dataset are positive. The model might just be
> saying 'positive' most of the time and getting lucky."*

Write on board:
```
BETTER EVALUATION METRICS FOR SENTIMENT:

  ROC-AUC:
    Measures how well the model RANKS positive above negative
    AUC = 0.5 → random guess
    AUC = 1.0 → perfect
    AUC > 0.9 → very good

  Precision at high confidence:
    "Of the predictions where model is >90% confident, what % are correct?"
    This is what matters in production — you only act on high-confidence preds.

  Calibration:
    When model says 80% confidence, is it right ~80% of the time?
    Overconfident models need Platt scaling or temperature scaling.
```

**Ask the room:** *"A fraud detection model that catches 99% of fraud but
flags 40% of legitimate transactions as fraudulent — is it useful?"*

This leads into a discussion of precision vs recall trade-offs — the ROC
curve lets you choose your operating point.

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
ALL THREE APPROACHES:
  Rule-based (VADER):
    Use when: no training data, social media, real-time
    Fails at: sarcasm, novel slang, double negatives

  TF-IDF + Logistic Regression:
    Use when: enough labelled data (>1000), speed matters
    Fails at: word order dependent cases, context

  BERT / Transformer:
    Use when: accuracy is critical, hardware available
    Fails at: <100 training examples (needs fine-tuning data)

THE NEGATION LESSON:
  Bigrams in TF-IDF → partial fix
  VADER rules       → partial fix
  BERT context      → best fix (but still not perfect on sarcasm)
```

**Homework — from the module visualisations:**
Look at the precision-recall curves saved to `visuals/sentiment_analysis/`.
For a customer support escalation system (flag angry emails), should you
optimise for precision or recall? Write a one-paragraph answer with reasoning.

---

## INSTRUCTOR TIPS

**"Why does VADER do better than TF-IDF on short texts?"**
> *"Short texts like tweets don't have enough words for TF-IDF to work well.
> The vector is almost entirely zeros. VADER's lexicon scores each word directly
> and applies its rules — it works even on two-word phrases. TF-IDF needs
> enough words to find a meaningful signal."*

**"Can I use HuggingFace in production without a GPU?"**
> *"Yes, but it's slow. DistilBERT can process about 40-60 sentences per
> second on CPU. For a batch processing job (analyse last week's reviews
> overnight), that is fine. For a real-time API with high traffic, you need
> a GPU or to use the TF-IDF approach as a fast first pass."*

**"What is aspect-based sentiment used for in real products?"**
> *"Hotel booking sites break reviews into 'cleanliness', 'location', 'value',
> 'service'. Each star rating shown separately. Product review sites summarise
> 'battery', 'camera', 'screen'. All of that is ABSA running under the hood."*

**"A student's sentence confuses all three systems — what to say?"**
> *"That is the most valuable moment in the session. Ask: what linguistic
> feature makes this hard? Is it sarcasm? Irony? A cultural reference?
> Ambiguous polarity? The gap between human language and machine understanding
> is exactly where NLP research is happening right now."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                10 min
├── VADER rule-based system     20 min
├── TF-IDF + LR for sentiment   25 min
├── Aspect-based sentiment      15 min
└── Close + homework            10 min

SESSION 2  (90 min)
├── Opening bridge              10 min
├── Why transformers win        20 min
├── HuggingFace pipeline        15 min
├── Live class sentence demo    20 min
├── ROC + calibration           15 min
└── Close + homework            10 min
```

---
*MLForBeginners · Part 5: NLP · Module 06*
