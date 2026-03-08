# 🎓 MLForBeginners — Instructor Guide
## Part 5 · Module 10: News Article Classifier (Project + Part 5 Capstone)
### Single 120-Minute Session

> **The Part 5 graduation project.**
> They've processed text, built TF-IDF features, trained sentiment models, used LSTMs,
> and extracted named entities. Today they put it ALL together:
> a multi-class text classifier that categorizes news into topics.
> Production-grade. End to end.

---

# SESSION (120 min)
## "Build a news topic classifier — end to end"

## Before They Arrive
- Terminal open in `nlp/projects/`
- Pull up a news site — BBC, Reuters, anything
- Think of 4-5 article topics as examples

---

## OPENING (10 min)

> *\"Open any news website.*
> *See how it's organized: Sports, Business, Technology, Politics, Entertainment?*
>
> *How does the site know which category each article belongs to?*
>
> *At small scale: humans label them.*
> *At big scale — thousands of articles per day across 50 categories —*
> *you need a classifier.*
>
> *Today you build that system.*
> *Raw news article → model → 'Technology' / 'Sports' / 'Politics'*
>
> *This is multi-class text classification.*
> *And it's one of the most common NLP tasks in production.\"*

Write on board:
```
MULTI-CLASS CLASSIFICATION:
  Binary:     spam / not spam  (2 classes)
  Multi-class: Business / Sports / Politics / Tech / Entertainment
               (5 classes — one correct answer per article)

  Output: probability distribution over all classes
  softmax([0.05, 0.02, 0.85, 0.06, 0.02]) → "Politics"
```

---

## SECTION 1: The Dataset (10 min)

```bash
python3 news_article_classifier.py
```

While it loads:

> *\"We're using a standard news categorization dataset.*
> *Thousands of real article headlines and summaries.*
> *Each labeled with one of several topic categories.*
>
> *Two key questions before training anything:*
> *1. How balanced are the classes?*
> *2. How long are the articles?*
>
> *These shape every decision we make.\"*

Show the class distribution plot from output:

> *\"Good news: this dataset is fairly balanced.*
> *If it weren't, we'd use the same techniques as churn prediction:*
> *class weights or oversampling.\"*

---

## SECTION 2: The Full NLP Pipeline (20 min)

Walk through each preprocessing step as it runs:

```
RAW ARTICLE:
"Apple unveils new iPhone with improved AI chip and camera system"

Step 1 — Tokenize:
  ["Apple", "unveils", "new", "iPhone", "with", "improved", "AI", ...]

Step 2 — Clean:
  lowercase, remove punctuation, remove stopwords
  ["apple", "unveils", "new", "iphone", "improved", "ai", "chip", "camera"]

Step 3 — Vectorize (TF-IDF):
  Each word gets a score: "iphone" = high (rare, informative)
                          "new" = lower (common)

Step 4 — Classify:
  Feature vector → Logistic Regression → P(Tech) = 0.91
```

> *\"Notice: 'apple' here means the company, not the fruit.*
> *TF-IDF doesn't know the difference.*
> *Word embeddings (from Module 03) do.*
> *That's why modern NLP uses embeddings — context matters.\"*

**Ask the room:** *\"What words would be the strongest signal for 'Sports'? For 'Finance'?\"*

---

## SECTION 3: Three Approaches, Three Results (25 min)

Run all three models and build the table together:

```
APPROACH            Accuracy   F1 (macro)   Training
────────────────────────────────────────────────────
TF-IDF + Logistic      ?           ?          fast
TF-IDF + SVM           ?           ?          fast
Word Embeddings + LSTM ?           ?          slow
```

Fill in from program output.

> *\"A pattern you'll see constantly:*
>
> *TF-IDF + Logistic Regression gets you 85-90% accuracy.*
> *Very fast, very interpretable.*
>
> *Adding LSTM with word embeddings might get 92-94%.*
> *But it's 10x slower to train.*
>
> *Is 3% worth 10x the compute cost?*
> *Depends on your use case.*
> *For a hobby project: not worth it.*
> *For 10 million articles/day: absolutely worth it.\"*

---

## SECTION 4: Error Analysis — What Does It Get Wrong? (20 min)

Open the confusion matrix. Focus on the off-diagonal cells.

> *\"The confusion matrix tells us exactly WHERE the model fails.*
>
> *Which two categories get confused most?*
>
> *Usually: Business and Technology.*
> *Why? Articles about tech companies mix both vocabularies.*
> *'Apple reports record revenue' — Tech? Business? Both.*
>
> *And Science/Technology often overlap.*
> *The model mirrors human ambiguity.\"*

Look at specific misclassified examples:

> *\"Read this headline: 'Tesla posts record profits despite supply chain issues.'*
> *Model says: Business. True label: Technology.*
>
> *Is the model wrong? Arguably not.*
> *Tesla IS a tech company. The article IS about business results.*
>
> *This is label ambiguity — a fundamental limit of classification.*
> *Some examples truly belong to multiple categories.*
> *Your classifier can only pick one.\"*

**Discuss:**
> *\"If this were a real product, how would you handle borderline cases?*
> *Predict the top-2 categories? Show a 'confidence' score?*
> *Flag for human review if confidence is below 80%?\"*

---

## SECTION 5: Production Considerations (10 min)

> *\"Shipping a classifier to production means thinking beyond accuracy.\"*

Write on board:
```
PRODUCTION CHECKLIST:
  ✅ Accuracy on held-out test set (not validation!)
  ✅ Confusion matrix — do errors make business sense?
  ✅ Prediction latency — how fast per article?
  ✅ Model size — can it fit on the server?
  ✅ Confidence calibration — does 90% confidence mean 90% accurate?
  ✅ Monitoring — does accuracy degrade over time as news topics shift?
  ✅ Retraining pipeline — when do we retrain?

THE LAST TWO ARE OFTEN FORGOTTEN.
News topics change. COVID wasn't a category in 2018.
Models drift. Build in a retraining schedule.
```

---

## PART 5 GRADUATION (5 min)

Write on board:

```
PART 5 COMPLETE — NLP MASTERED

You can now:
  ✅ Preprocess text: tokenize, clean, vectorize
  ✅ Build TF-IDF + classical ML pipelines
  ✅ Train word embeddings and understand their geometry
  ✅ Build LSTM models for sequence classification
  ✅ Extract named entities
  ✅ Build end-to-end sentiment and topic classifiers

WHAT COMES NEXT:
  Part 6: Transformers
  → What if instead of processing text left-to-right,
    the model could look at ALL words simultaneously?
  → Attention mechanism: "which words matter for understanding THIS word?"
  → That's how BERT and GPT work. And you're ready for it.
```

> *\"You just built the kind of system that runs inside every major*
> *content platform — YouTube, Reddit, Twitter, every news aggregator.*
>
> *Topic classification. Sentiment analysis. Entity extraction.*
> *You can now build all of them.*
>
> *Part 6 takes you to the state of the art.\"*

**Graduation moment:** Everyone types their own custom news headline into the live prediction. Print the result.

---

## INSTRUCTOR TIPS

**"Why use TF-IDF instead of just word counts?"**
> *"Word counts give 'the' and 'and' huge weights — they appear everywhere.*
> *TF-IDF divides by how common a word is across ALL documents.*
> *'Iphone' appears in 1% of articles → high IDF → high weight.*
> *'The' appears in 100% → IDF near 0 → low weight.*
> *TF-IDF automatically suppresses common words.\"*

**"What if we want to classify in real time?"**
> *"TF-IDF + Logistic Regression: microseconds per prediction.*
> *LSTM: milliseconds.*
> *BERT (Part 6): tens of milliseconds.*
> *For high-throughput systems, the fastest model that meets accuracy requirements wins.*
> *Don't use BERT when Logistic Regression works.\"*

**"What about multilingual classification?"**
> *"TF-IDF: must be trained per language.*
> *Multilingual BERT (Part 6): trained on 104 languages simultaneously.*
> *One model, all languages. That's the transformer revolution.\"*

---

## Quick Reference
```
Single Session (120 min)
├── Opening + motivation       10 min
├── Dataset exploration        10 min
├── NLP pipeline walkthrough   20 min
├── Three approaches           25 min
├── Error analysis             20 min
├── Production checklist       10 min
└── Part 5 graduation           5 min
```

---
*MLForBeginners · Part 5: NLP · Module 10 (Capstone)*
