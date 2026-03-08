# MLForBeginners — Instructor Guide
## Part 5 · Module 09: Movie Review Sentiment — End-to-End Project
### Single 120-Minute Session

> **Prerequisites:** All Part 5 algorithm modules complete (05-08). They can
> build TF-IDF pipelines, understand VADER, LSTM, and NER. They have seen
> every building block — now they wire them into a production-ready project.
> **Payoff today:** A complete NLP system that takes raw movie reviews and
> outputs sentiment labels with confidence — plus everyone submits their own
> review to the model live.

---

## Before They Arrive
- Terminal open in `nlp/projects/`
- Have three movie reviews ready to run as warm-up: one clearly positive,
  one clearly negative, one sarcastic or ambiguous
- Write on the whiteboard: "RULE-BASED vs ML vs DEEP LEARNING"
- Post this on the wall: "Which system will win? Vote now."

---

## OPENING (15 min)

> *"Quick vote — raise your hand: who thinks the rule-based system will have
> the best accuracy today? Who says ML? Who says deep learning?*
>
> *Keep those hands in mind. By the end of the session you might be surprised.*
>
> *You have spent the last four modules building individual NLP tools.
> Today is the first project — where you do what engineers do:
> take every tool available, apply it to a real problem, measure everything
> honestly, and pick a winner.*
>
> *The problem: given a movie review, is it positive or negative?
> Sounds trivial. It is not."*

Write on board:
```
WHAT MAKES MOVIE SENTIMENT HARD:

  "The director has absolutely no idea what he's doing — in the best
   possible way."                                 → POSITIVE (ironic praise)

  "I can't say this movie left me feeling nothing."  → POSITIVE (double neg)

  "It's exactly the kind of film that wins awards."  → Could be either!

  "Not since watching paint dry have I been so entertained."  → SARCASM
```

> *"Human language is ambiguous by design. We use irony, understatement,
> sarcasm, double negatives. A system that handles these correctly is doing
> something genuinely impressive — and today you will see exactly where
> each of your three systems breaks down."*

---

## SECTION 1: Dataset and Baseline Preprocessing (15 min)

Run the module and watch the dataset section:
```bash
python3 nlp/projects/movie_review_sentiment.py
```

While it loads, explain the dataset structure:
```
SYNTHETIC IMDb-STYLE DATASET:
  ~200 positive reviews
  ~200 negative reviews
  Mirrors IMDb Large Movie Review Dataset structure

PREPROCESSING PIPELINE:
  Raw text
   → lowercase
   → remove HTML tags  (real IMDb reviews contain <br /> tags)
   → remove special characters
   → remove stopwords  (optional — sometimes hurts sentiment)
   → tokenise

  Note: we do NOT remove stopwords like "not", "never", "no"
  These are CRITICAL for negation — removing them kills accuracy.
```

> *"This is a real-world lesson: the standard NLP advice 'remove stopwords'
> is WRONG for sentiment analysis. 'Not', 'never', 'no', 'hardly' —
> these words are the most important ones in a sentiment review.*
>
> *Always think about whether a preprocessing step makes sense for YOUR task.
> Never apply it blindly because a tutorial said to."*

**Ask the room:** *"Name three other preprocessing decisions that might be
right for topic classification but wrong for sentiment."*

Good answers:
- Stemming: "wonderful" → "wonder" (loses the -ful suffix sentiment signal)
- Lowercasing: "AMAZING" vs "amazing" — caps carry intensity information
- Removing punctuation: "!!!" signals strong positive sentiment

---

## SECTION 2: System 1 — Rule-Based Sentiment (15 min)

> *"System 1 uses a VADER-style lexicon. No training data. Just scores
> and rules from Module 06. Let's see where it stands on our 400 reviews."*

Point at the output as it runs. Look for the accuracy score and write it on
the whiteboard under "Rule-Based".

Then:
> *"Look at the errors. What do the misclassified reviews have in common?*
>
> *[Pause — let them look]*
>
> *Typically you will see: long reviews where the overall tone is buried
> in nuance, reviews with irony or sarcasm, reviews that use film-criticism
> vocabulary ('derivative', 'pedestrian', 'overwrought') that may not be
> in the lexicon."*

Draw on board — error analysis template:
```
FALSE POSITIVES (predicted POS, actually NEG):
  Common pattern: ___________________________

FALSE NEGATIVES (predicted NEG, actually POS):
  Common pattern: ___________________________

CONFIDENCE:
  High confidence correct:   ___  %
  High confidence wrong:     ___  %  ← dangerous
  Low confidence:            ___  %  ← the system "knows" it's unsure
```

---

## SECTION 3: System 2 — TF-IDF + Logistic Regression (15 min)

Write on board before running:
```
EXPECTED ADVANTAGES OVER RULE-BASED:
  1. Learns film-specific vocabulary from data
     ("masterpiece", "pretentious", "formulaic", "derivative")
  2. Learns relative importance of words
  3. Bigrams capture negation: "not good" as one feature

EXPECTED WEAKNESSES:
  Still bag-of-words — loses sentence structure
  Short reviews may not have enough signal
  Rare words may not have reliable weights
```

Run and observe. Write the accuracy on the board under "TF-IDF + LR".

> *"Most of the time, TF-IDF + LR beats the rule-based system by 5-10
> percentage points on movie reviews. The reason: film critics use
> domain-specific vocabulary — 'luminous cinematography', 'leaden pacing',
> 'overwrought', 'pedestrian' — that VADER's general lexicon does not score well.*
>
> *TF-IDF learned these from the training examples."*

**Ask the room:** *"What are the top 5 most positive-weighted words? What
are the top 5 most negative-weighted?"*

Have them inspect the feature importance output. Common answers:
- Positive: "masterpiece", "brilliant", "outstanding", "stunning", "exceptional"
- Negative: "awful", "terrible", "worst", "boring", "waste"

> *"Notice anything? The model learned real film criticism vocabulary.
> That is 400 training examples doing the work that would take weeks
> to hand-code into a lexicon."*

---

## SECTION 4: System 3 — Deep Learning LSTM (15 min)

> *"System 3 is our LSTM from Module 07 — Embedding + BiLSTM + Dense.*
>
> *This one reads the review as a sequence. It should handle negation
> and sentence structure better than bag-of-words."*

While it trains:
> *"Notice the training time versus the ML approach. The LSTM is taking
> roughly 20-30x longer to train. On 400 examples, that is still fast.
> On the full IMDb dataset (50,000 reviews), the LSTM would take minutes
> while TF-IDF takes seconds.*
>
> *At what accuracy improvement does the extra cost become worthwhile?
> That is a business decision, not a technical one."*

Write the result on the board. Compare all three:
```
System               Accuracy   Train time
Rule-Based (VADER)   ____%      0 sec (no training)
TF-IDF + LR          ____%      ~0.5s
LSTM / BiLSTM        ____%      ~30s
```

**Ask the room:** *"Who voted correctly? What is the pattern — does deep
learning always win?"*

Lead the discussion toward: on small datasets and binary sentiment, TF-IDF +
LR is a very competitive baseline. LSTM wins more clearly with 10,000+
examples.

---

## SECTION 5: Negation and Sarcasm — The Hard Cases (10 min)

> *"Let's stress test all three systems with the hard cases I had on the
> board at the start."*

Run the three difficult reviews through all three systems. Display the output
side by side.

Write on board:
```
CHALLENGE SENTENCE                 RULE  ML    LSTM
"Not bad at all"                   ___   ___   ___
"So bad it's entertaining"         ___   ___   ___
"An absolute must-miss"            ___   ___   ___
"I've seen worse"                  ___   ___   ___
```

> *"Sarcasm is the final frontier. Even BERT struggles with sarcasm because
> you often need EXTERNAL knowledge — knowing that the movie was universally
> panned, knowing the critic's track record, knowing cultural context.*
>
> *State-of-the-art sarcasm detection is still an open research problem.
> If your model confuses sarcasm, that is not a failure of your implementation —
> it is a limitation of current NLP technology."*

---

## SECTION 6: Live Class Demo — Submit Your Own Reviews (20 min)

> *"Now the fun part. I want everyone to type a movie review — real or made up.
> You can be honest, or you can try to fool the system.*
>
> *We will run each one through all three models and vote on which got
> the 'true' sentiment right."*

Collect reviews via chat or sticky notes. Run each through:
```python
# Quick inference function from the module
from movie_review_sentiment import predict_all_systems

review = "Your text here"
results = predict_all_systems(review)

for name, label, confidence in results:
    print(f"  {name:20s}: {label}  ({confidence:.2%} confidence)")
```

Track on board:
```
REVIEW                  RULE    ML    LSTM    TRUTH (class vote)
[Student 1]             ___     ___   ___     ___
[Student 2]             ___     ___   ___     ___
[Student 3]             ___     ___   ___     ___
...
```

> *"Every disagreement between the systems is a learning moment.
> When rule-based says positive and LSTM says negative — one of them is
> encoding something the other doesn't understand. What is it?"*

---

## SECTION 7: The Simple Sentiment API (10 min)

> *"The last thing the module builds is a deployable function —
> classify_sentiment() — that takes a string and returns a prediction.*
>
> *This is what you would wrap in a Flask route or FastAPI endpoint
> to serve this model over HTTP."*

Show the API structure from the module output:
```python
def classify_sentiment(text: str, model="tfidf") -> dict:
    """
    Parameters:
      text  : raw review text (any length)
      model : "rule_based" | "tfidf" | "lstm"

    Returns:
      {
        "label":      "POSITIVE" or "NEGATIVE",
        "confidence": 0.0 to 1.0,
        "model":      "tfidf"
      }
    """
    ...

# Usage:
result = classify_sentiment("Absolutely wonderful film", model="tfidf")
# {"label": "POSITIVE", "confidence": 0.97, "model": "tfidf"}
```

> *"This function is the boundary between your ML code and your application.
> Anyone can call it without knowing anything about TF-IDF or LSTMs.
> Designing clean interfaces like this is what separates a working notebook
> from production-ready code."*

---

## CLOSING (5 min)

Board summary:
```
PROJECT 1 — MOVIE REVIEW SENTIMENT:
  Three systems, same task, honest comparison

  LESSONS:
  1. Remove stopwords THOUGHTFULLY — "not" is critical for sentiment
  2. TF-IDF + LR is a hard baseline — often beats simple LSTM on small data
  3. Sarcasm and double negatives defeat all current systems
  4. Clean API design bridges ML and application code
  5. Confidence scores matter — "70% confident POSITIVE" should trigger review

NEXT SESSION:
  Part 5 Capstone — News Article Classifier
  Multi-class, class imbalance, macro/micro F1, deployable function
  We are one module away from completing NLP. Well done.
```

**Homework — error analysis:**
Look at the visualisations saved to `visuals/movie_review_sentiment/`.
Find the confusion matrices for all three models. For EACH model, write:
- One type of review it handles well
- One type of review it consistently gets wrong
- One idea for how you would fix that weakness

---

## INSTRUCTOR TIPS

**"My LSTM accuracy is lower than TF-IDF — did I do something wrong?"**
> *"No — this is expected on a small dataset. The LSTM has more parameters
> to train and needs more data to outperform the simpler baseline.
> On the full IMDb 50K dataset, BiLSTM typically beats TF-IDF by 3-5%.
> On 400 examples, results can go either way. This is an important lesson:
> model complexity requires data volume to pay off."*

**"Can we use this to classify tweets?"**
> *"Not without retraining. The model was trained on IMDb-style formal prose.
> Tweets have abbreviations ('luv', 'omg'), hashtags, @mentions, emojis,
> and very different vocabulary. You would need tweet-specific training data.
> VADER actually outperforms TF-IDF+LR on raw tweets for exactly this reason —
> it was designed for social media."*

**"A student's review fools all three systems — make it a teaching moment"**
> *"Ask: what linguistic feature made this hard? Write it on the board.
> The gap between human language understanding and machine understanding
> is exactly where research is happening. If you found a systematic failure
> mode, you just identified a research contribution."*

**"The confidence scores seem too high — everything is 90%+"**
> *"Logistic Regression is often overconfident — it does not naturally
> calibrate probabilities. Platt scaling or isotonic regression can
> post-hoc calibrate scores. In production, always validate that
> '90% confident' predictions are actually right ~90% of the time."*

---

## Quick Reference

```
SINGLE SESSION  (120 min)
├── Opening hook + vote         15 min
├── Dataset + preprocessing     15 min
├── System 1: Rule-Based        15 min
├── System 2: TF-IDF + LR       15 min
├── System 3: LSTM              15 min
├── Negation + sarcasm cases    10 min
├── Live class demo             20 min
├── Sentiment API design        10 min
└── Closing + homework           5 min
```

---
*MLForBeginners · Part 5: NLP · Module 09*
