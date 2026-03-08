# MLForBeginners — Instructor Guide
## Part 5 · Module 05: Text Classification Pipeline
### Two-Session Teaching Script

> **Prerequisites:** Modules 01–04 complete. They know tokenisation, TF-IDF,
> word embeddings, and RNN intuition. They are also comfortable with sklearn
> from Parts 1–2 and Keras from Parts 3–4.
> **Payoff today:** They will wire raw text all the way to a multi-class
> prediction using two complete pipelines — classical ML and a neural net.

---

# SESSION 1 (~90 min)
## "Building the machine — sklearn Pipeline + TF-IDF + multi-class classification"

## Before They Arrive
- Terminal open in `nlp/algorithms/`
- Whiteboard ready
- Four sticky notes on the board labelled: World / Sports / Business / Sci/Tech

---

## OPENING (10 min)

> *"You have all received a mis-labelled email — something that should have gone
> to spam but hit your inbox, or a newsletter filed under 'social' instead of
> 'updates'.*
>
> *That mis-labelling was done by a text classifier. A model that read the
> subject line and body, and decided 'this belongs in bucket X'.*
>
> *Today you're going to build one — from raw text all the way to a confident
> four-way prediction — and you'll understand every single step."*

Draw on board:
```
RAW TEXT
  "Scientists discover new black hole at galaxy centre"
         |
         v
   [Text Cleaning]       lowercase, remove punctuation
         |
         v
   [TF-IDF Vectors]      "scientists"=0.41, "galaxy"=0.37, ...
         |
         v
   [Classifier]          Logistic Regression / Naive Bayes / SVM
         |
         v
   PREDICTION: Sci/Tech  (confidence 91%)
```

> *"We've learned each of these steps in isolation over the last four modules.
> Today we bolt them together into a production pipeline object."*

---

## SECTION 1: What is Multi-Class Classification? (15 min)

Write on board:
```
BINARY classification:    2 outputs    (spam / not spam)
MULTI-CLASS classification: N outputs  (World / Sports / Business / Sci/Tech)

Two strategies for multi-class:
  OvR  = One vs Rest    Train N separate binary classifiers
  Multinomial  = One model, N output probabilities  (softmax)

Logistic Regression in sklearn defaults to OvR.
Multinomial Naive Bayes is natively multi-class.
```

> *"Logistic Regression handles multi-class via 'one vs rest' — it trains
> a separate binary model for each category. When you call predict(), it runs
> all four, and picks the one with the highest confidence.*
>
> *Naive Bayes has a multinomial variant purpose-built for text — it uses word
> counts directly to estimate 'how likely is this word given this category?'"*

**Ask the room:** *"If we have four categories, how many binary classifiers
does OvR train?"*

Expected answer: 4 — one for each class against all others.

---

## SECTION 2: The sklearn Pipeline Object (20 min)

> *"Here is the problem you have all hit before: you fit your vectoriser on
> training data, then you forget to apply the SAME fit to your test data.
> Or you apply different preprocessing steps in a different order.*
>
> *sklearn Pipeline solves this. It chains steps — and when you call
> .fit(), the whole chain fits. When you call .predict(), the whole chain
> transforms first, then predicts. No leakage. No forgetting steps."*

Write on board:
```
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("tfidf",  TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf",    LogisticRegression(max_iter=1000, C=1.0))
])

pipe.fit(X_train, y_train)
pipe.predict(["Scientists discover new black hole"])
```

Annotate each argument:
```
max_features=5000  → keep only the 5000 highest-scoring TF-IDF terms
ngram_range=(1,2)  → include both single words and two-word phrases
C=1.0              → regularisation strength (lower C = stronger penalty)
```

> *"The pipeline is one object. You can pickle it. You can pass it to
> GridSearchCV. You can ship it to production. This is how real NLP
> systems are built."*

**Live demo — build the pipeline:**
```python
# In a Python shell or notebook
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "Scientists launch new satellite into orbit",
    "Team wins championship in overtime",
    "Stock market falls on inflation fears",
    "President signs new trade agreement"
]
labels = [3, 1, 2, 0]  # Sci/Tech, Sports, Business, World

pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf",   LogisticRegression())
])
pipe.fit(texts, labels)
print(pipe.predict(["Astronomers spot distant galaxy"]))
```

---

## SECTION 3: Cross-Validation on Text Data (20 min)

> *"You all know train/test split from Part 1. But with text data there is an
> extra trap: if your dataset is sorted by category, a naive split might put
> all Sports articles in training and none in test.*
>
> *Cross-validation shuffles and folds the data, giving you a more reliable
> estimate of true performance."*

Write on board:
```
K-FOLD CROSS VALIDATION (K=5)

  Fold 1: [TEST ] [train] [train] [train] [train]
  Fold 2: [train] [TEST ] [train] [train] [train]
  Fold 3: [train] [train] [TEST ] [train] [train]
  Fold 4: [train] [train] [train] [TEST ] [train]
  Fold 5: [train] [train] [train] [train] [TEST ]

  Final score = mean of 5 test scores
  More reliable than a single train/test split
```

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipe, texts, labels, cv=5, scoring="accuracy")
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

> *"The (+/-) number is your confidence interval. A score of 0.89 +/- 0.02
> means you can trust this. A score of 0.89 +/- 0.15 means your model is
> very sensitive to which fold it saw — you may need more data."*

**Ask the room:** *"Why can't we just use the highest fold's accuracy as
our reported score?"*

Expected answer: that would be cherry-picking — the average is the honest number.

---

## SECTION 4: Live Demo — Run the Full Module (15 min)

```bash
python3 nlp/algorithms/text_classification_pipeline.py
```

Watch the output together. Point at:
- The per-class accuracy breakdown in the classification report
- The confusion matrix — which categories get confused with each other?
- The TF-IDF feature importance table per class

> *"Look at the top features for 'Sports' — words like 'championship',
> 'season', 'playoff' dominate. For 'Sci/Tech' it's 'researchers', 'launch',
> 'algorithm'. The model has learned the vocabulary of each domain.*
>
> *This is the interpretability advantage of TF-IDF over neural networks:
> you can always ask 'what words drove this decision?'"*

Open the confusion matrix visualisation:
```
            Predicted →
              World  Sports  Business  Sci/Tech
Actual  World   [42]   [ 2]     [ 3]     [ 1]
       Sports   [ 1]   [45]     [ 0]     [ 2]
     Business   [ 3]   [ 0]     [40]     [ 4]
     Sci/Tech   [ 2]   [ 1]     [ 3]     [43]
```

> *"Business and Sci/Tech confuse each other most — makes sense.
> A story about a tech IPO looks like both categories."*

---

## CLOSING SESSION 1 (10 min)

Board summary:
```
TODAY'S PIPELINE:
  Raw text
   → TfidfVectorizer (max_features, ngram_range)
   → Classifier (LogisticRegression, NaiveBayes, SVM)
   → pipe.fit(X_train, y_train)
   → pipe.predict(X_test)

EVALUATION:
  classification_report  → precision, recall, F1 per class
  cross_val_score        → honest performance estimate
  confusion_matrix       → where does the model get confused?
```

**Homework:** What would you change about the pipeline to handle a 20-class
problem (e.g., classify Stack Overflow questions into 20 programming topics)?
Write three ideas down — we will compare answers at the start of Session 2.

---

# SESSION 2 (~90 min)
## "Comparing approaches — TF-IDF vs neural, and reading the model"

## OPENING (10 min)

> *"Last session we built a classical pipeline. Today we ask the harder question:
> is it good enough?*
>
> *We will benchmark three classifiers on the same data — Logistic Regression,
> Naive Bayes, and a simple dense neural network — and then we will read what
> the model actually learned using top features per class."*

---

## SECTION 1: Comparing Classifiers Side by Side (20 min)

Write on board:
```
APPROACH          PROS                       CONS
TF-IDF + LR       Fast, interpretable        Ignores word order
TF-IDF + NB       Works with tiny datasets   Assumes feature independence
TF-IDF + SVM      Strong on sparse data      Hard to tune C and gamma
Neural (Dense)    Learns feature combos      Needs more data, slower
Neural (LSTM)     Captures word order        Much slower, needs GPU
```

> *"For most text classification tasks with more than ~500 training examples,
> TF-IDF + Logistic Regression is a hard baseline to beat.*
>
> *Neural networks win on tasks where word ORDER matters — sentiment with
> negation, sarcasm, very long documents. For topic classification,
> which is bag-of-words by nature, LR often matches or beats LSTM."*

Code together — run multiple classifiers and compare:
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "MultinomialNB":      MultinomialNB(),
    "LinearSVC":          LinearSVC(max_iter=2000),
}

tfidf = TfidfVectorizer(max_features=5000)
X_vec = tfidf.fit_transform(X_train)

for name, clf in classifiers.items():
    scores = cross_val_score(clf, X_vec, y_train, cv=5)
    print(f"{name:25s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## SECTION 2: Top TF-IDF Features Per Class (20 min)

> *"The most useful debugging tool for a text classifier is asking:
> 'What words most strongly push a document toward each class?'*
>
> *In Logistic Regression, each feature has a coefficient. The largest
> positive coefficients for class 'Sports' are the strongest Sports words."*

Write on board:
```
Class: Sports
  Top positive features:
    "championship" → 2.41
    "playoff"      → 2.38
    "quarterback"  → 2.29
    "tournament"   → 2.11
    "season"       → 1.98

  Top negative features (push AWAY from Sports):
    "algorithm"    → -1.87
    "treaty"       → -1.74
    "quarterly"    → -1.63
```

> *"Negative features are just as telling — the model learned that 'algorithm'
> is a strong anti-Sports signal.*
>
> *Use this during debugging. If a Sports article is being classified as
> Sci/Tech, check whether the article contains suspiciously many tech words."*

**Ask the room:** *"A news story about a professional esports tournament might
confuse the classifier. Which two classes would it pull toward? How could
you fix it?"*

Expected discussion: Sports (tournament) vs Sci/Tech (gaming, technology).
Fix: add more esports training examples labelled as Sports.

---

## SECTION 3: Simple Neural Classifier vs TF-IDF (20 min)

> *"Let's add a simple neural classifier — not LSTM yet, just dense layers
> on top of TF-IDF vectors. Think of it as logistic regression's smarter sibling."*

Write on board:
```
NEURAL TEXT CLASSIFIER (dense, no sequence modelling):

  TF-IDF vector (5000 dims)
       |
  Dense(256, relu)      ← learn combinations of features
       |
  Dropout(0.3)          ← regularise
       |
  Dense(128, relu)
       |
  Dense(4, softmax)     ← one output per class
       |
  argmax → class label
```

> *"The key question is always: does the added complexity of a neural network
> actually pay off? For topic classification, often the answer is no —
> TF-IDF captures the right signal already.*
>
> *When DOES the neural classifier win? When the vocabulary is ambiguous,
> when context matters, or when you have hundreds of thousands of examples."*

Show the comparison table from the module output:
```
Model                 Accuracy   Train time
TF-IDF + LogReg       0.91       0.3s
TF-IDF + NaiveBayes   0.88       0.1s
TF-IDF + Neural       0.90       12s
```

---

## SECTION 4: Hyperparameter Search on a Pipeline (15 min)

> *"One of the best features of sklearn Pipeline is that you can run
> GridSearchCV across the ENTIRE pipeline — including vectoriser parameters."*

Write on board:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "tfidf__max_features":  [3000, 5000, 10000],
    "tfidf__ngram_range":   [(1,1), (1,2)],
    "clf__C":               [0.1, 1.0, 10.0],
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
print(f"Best score:  {grid.best_score_:.3f}")
```

> *"The double underscore in 'tfidf__max_features' is how Pipeline passes
> parameters to named steps. 'tfidf' is the step name, 'max_features' is
> TfidfVectorizer's parameter."*

---

## CLOSING SESSION 2 (5 min)

Board summary:
```
THREE LEVELS OF TEXT CLASSIFICATION:
  1. TF-IDF + LR        → Fast baseline, usually competitive
  2. TF-IDF + Neural    → Marginal gains, interpretability lost
  3. Embeddings + LSTM  → Next module — word order matters

DEBUGGING TOOLKIT:
  classification_report → per-class F1
  confusion_matrix      → which classes look alike?
  top features per class → what did the model actually learn?
  cross_val_score       → honest accuracy estimate
```

**Homework — extend the pipeline:**
```python
# Try replacing the LogisticRegression with LinearSVC.
# Does it improve accuracy on your run?
# Which class improves the most? Which gets worse?
# Inspect the confusion matrix difference between the two models.
```

---

## INSTRUCTOR TIPS

**"TF-IDF gave me 0.95 accuracy — do I need neural networks at all?"**
> *"Probably not for this task. TF-IDF + LR is the right answer for topic
> classification. Save neural networks for tasks where word ORDER matters:
> sentiment with negation, question answering, generation. Always start simple."*

**"My confusion matrix diagonal isn't filling up — why?"**
> *"Check class balance first. If 80% of your data is 'World', the model can
> get 80% accuracy by predicting 'World' for everything. Look at per-class F1,
> not overall accuracy."*

**"What does C do in Logistic Regression?"**
> *"C is the inverse of regularisation strength. High C (e.g. 100) means the
> model fits training data tightly — potential overfit. Low C (e.g. 0.01)
> means the model is heavily penalised for large coefficients — potential
> underfit. Start with C=1, tune from there."*

**"Why does ngram_range=(1,2) help?"**
> *"Single words miss phrases. 'New York' is a place; 'new' and 'York' alone
> are not. Bigrams capture two-word signals. Trigrams rarely help — they're
> too specific and blow up the feature space."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                10 min
├── Multi-class classification  15 min
├── sklearn Pipeline object     20 min
├── Cross-validation on text    20 min
├── Live demo                   15 min
└── Close + homework            10 min

SESSION 2  (90 min)
├── Opening bridge              10 min
├── Comparing classifiers       20 min
├── Top features per class      20 min
├── Neural vs TF-IDF            20 min
├── GridSearchCV on Pipeline    15 min
└── Close + homework             5 min
```

---
*MLForBeginners · Part 5: NLP · Module 05*
