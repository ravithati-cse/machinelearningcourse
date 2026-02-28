"""
💬 NLP — Algorithm 2: Sentiment Analysis
=========================================

Learning Objectives:
  1. Build a rule-based sentiment analyzer (VADER-style lexicon approach)
  2. Train TF-IDF + Logistic Regression for binary and fine-grained sentiment
  3. Use pretrained HuggingFace pipeline for state-of-the-art sentiment
  4. Handle negation ('not good') and intensifiers ('very bad')
  5. Evaluate with ROC curve, precision-recall, and confidence calibration
  6. Understand aspect-based sentiment (positive overall, negative shipping)
  7. Compare all three approaches on the same test examples

YouTube Resources:
  ⭐ Krish Naik — Sentiment Analysis https://www.youtube.com/watch?v=qU8ORYQH82E
  ⭐ HuggingFace — pipelines https://www.youtube.com/watch?v=QEaBAZQCtwE
  📚 VADER paper: https://ojs.aaai.org/index.php/ICWSM/article/view/14550

Time Estimate: 70 min
Difficulty: Intermediate
Prerequisites: text_classification_pipeline.py, 03_word_embeddings.py
Key Concepts: lexicon-based, ML-based, transformer-based, negation, aspect sentiment
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import os

os.makedirs("../visuals/sentiment_analysis", exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("💬 NLP ALGORITHM 2: SENTIMENT ANALYSIS")
print("=" * 70)
print()
print("Sentiment analysis: determine the EMOTIONAL TONE of text.")
print()
print("  Binary:        positive / negative")
print("  Fine-grained:  very positive / positive / neutral / negative / very negative")
print("  Aspect-based:  'The food was great but the service was terrible'")
print("                  food=positive, service=negative, overall=mixed")
print()
print("Three approaches (increasing complexity and accuracy):")
print("  1. Rule-based (VADER-style):  fast, no training data needed")
print("  2. ML-based (TF-IDF + LR):   needs labeled data, very competitive")
print("  3. Transformer-based (BERT):  best accuracy, needs GPU/patience")
print()


# ======================================================================
# SECTION 1: Rule-Based Sentiment (VADER-Style)
# ======================================================================
print("=" * 70)
print("SECTION 1: RULE-BASED SENTIMENT ANALYSIS (VADER-STYLE)")
print("=" * 70)
print()
print("VADER (Valence Aware Dictionary and sEntiment Reasoner):")
print("  • Lexicon of ~7,500 words with sentiment scores (-4 to +4)")
print("  • Rules for CAPITALIZATION, punctuation!!!, intensifiers, negation")
print("  • Returns: positive, negative, neutral, compound (-1 to +1)")
print()
print("Our simplified VADER-style implementation:")
print()

# Sentiment lexicon (abbreviated — real VADER has 7500+ entries)
SENTIMENT_LEXICON = {
    # Strong positive
    "excellent": 3.5, "outstanding": 3.5, "amazing": 3.4, "fantastic": 3.3,
    "superb": 3.3, "wonderful": 3.2, "brilliant": 3.2, "magnificent": 3.2,
    "perfect": 3.4, "exceptional": 3.3, "incredible": 3.1, "phenomenal": 3.2,
    # Moderate positive
    "good": 2.0, "great": 2.8, "nice": 1.8, "love": 2.8, "enjoy": 2.1,
    "happy": 2.4, "pleased": 2.0, "satisfied": 1.9, "recommend": 2.2,
    "beautiful": 2.5, "impressive": 2.4, "solid": 1.5, "decent": 1.2,
    # Mild positive
    "okay": 0.5, "fine": 0.7, "ok": 0.5, "alright": 0.6, "fair": 0.8,
    # Mild negative
    "mediocre": -1.0, "disappointing": -2.0, "average": -0.5, "bland": -1.2,
    # Moderate negative
    "bad": -2.0, "poor": -2.0, "boring": -1.8, "slow": -1.5, "annoying": -2.0,
    "frustrating": -2.1, "dislike": -1.9, "hate": -3.0, "waste": -2.3,
    "useless": -2.4, "terrible": -3.1, "horrible": -3.2, "awful": -3.2,
    # Strong negative
    "disgusting": -3.3, "atrocious": -3.4, "abysmal": -3.4, "dreadful": -3.0,
    "worst": -3.5, "catastrophic": -3.4,
}

# Intensifiers (multiply the following word's score)
INTENSIFIERS = {
    "very": 1.5, "extremely": 2.0, "really": 1.4, "absolutely": 1.8,
    "totally": 1.4, "incredibly": 1.7, "so": 1.3, "quite": 1.2,
    "pretty": 1.1, "especially": 1.3, "particularly": 1.2, "super": 1.5,
}

# Negation words (flip sign of following words for next 3 words)
NEGATIONS = {
    "not", "no", "never", "neither", "nor", "nobody", "nothing",
    "nowhere", "without", "don't", "doesn't", "didn't", "won't",
    "can't", "couldn't", "wouldn't", "shouldn't", "isn't", "aren't",
}


class VADERStyle:
    """
    Simplified VADER-style rule-based sentiment analyzer.
    Returns compound score (-1 to +1) and positive/negative/neutral proportions.
    """

    CAPS_BOOST = 0.733   # capitalized words get a boost

    def __init__(self, lexicon=None, intensifiers=None, negations=None):
        self.lex  = lexicon     or SENTIMENT_LEXICON
        self.ints = intensifiers or INTENSIFIERS
        self.negs = negations   or NEGATIONS

    def _tokenize(self, text):
        return re.findall(r"\b\w+(?:'\w+)?\b", text)

    def _word_score(self, word, orig_word):
        """Score a word, considering capitalization."""
        score = self.lex.get(word.lower(), 0.0)
        if score != 0 and orig_word.isupper() and len(orig_word) > 2:
            score += np.sign(score) * self.CAPS_BOOST
        return score

    def polarity_scores(self, text):
        """Main sentiment scoring function."""
        tokens    = self._tokenize(text)
        sentiments= []
        i = 0

        while i < len(tokens):
            tok  = tokens[i]
            word = tok.lower()

            # Punctuation boost: !!! adds 0.292 each (capped at 3 !!!)
            n_excl = text.count("!")
            excl_boost = min(n_excl, 3) * 0.292

            score = self._word_score(tok, tok)

            if score != 0:
                # Check for intensifier in previous position
                if i > 0 and tokens[i-1].lower() in self.ints:
                    mult  = self.ints[tokens[i-1].lower()]
                    score *= mult

                # Check for negation in previous 3 positions
                negated = any(tokens[max(0, i-k)].lower() in self.negs
                              for k in range(1, 4) if i-k >= 0)
                if negated:
                    score *= -0.74   # negation dampens (not just flips)

                sentiments.append(score)

            i += 1

        if not sentiments:
            return {"pos": 0.0, "neg": 0.0, "neu": 1.0, "compound": 0.0}

        sum_s   = sum(sentiments)
        sum_abs = sum(abs(s) for s in sentiments) + 15  # dampening constant (VADER's α)

        compound = sum_s / np.sqrt(sum_s**2 + 15)

        # Add punctuation boost
        n_excl = text.count("!")
        compound += min(n_excl, 3) * 0.038

        compound = max(-1.0, min(1.0, compound))

        pos = sum(s for s in sentiments if s > 0)
        neg = sum(abs(s) for s in sentiments if s < 0)
        neu = max(0, sum_abs - pos - neg)
        total = pos + neg + neu + 1e-9

        return {
            "pos": round(pos / total, 3),
            "neg": round(neg / total, 3),
            "neu": round(neu / total, 3),
            "compound": round(compound, 4),
            "label": "Positive" if compound >= 0.05 else
                     "Negative" if compound <= -0.05 else "Neutral"
        }


vader = VADERStyle()

test_sentences = [
    "This product is absolutely amazing! I love it so much!",
    "Terrible experience. The worst product I have ever bought.",
    "It's okay, nothing special.",
    "Not bad at all, quite impressed with the quality.",
    "Very disappointing. The food was not good at all.",
    "INCREDIBLE performance! FANTASTIC design! LOVE it!!!",
    "The phone is great but the battery life is terrible.",
    "I don't hate it, but I wouldn't recommend it.",
]

print("  Rule-based sentiment scores:")
print(f"  {'Text':<50} {'Compound':>10}  {'Label'}")
print(f"  {'─'*50} {'─'*10}  {'─'*10}")
for sent in test_sentences:
    scores = vader.polarity_scores(sent)
    bar    = "+" * int(scores["pos"] * 10) + "-" * int(scores["neg"] * 10)
    print(f"  {sent[:50]:<50} {scores['compound']:>10.4f}  {scores['label']}")
print()
print("  VADER thresholds:")
print("  compound >= +0.05 → Positive")
print("  compound <= -0.05 → Negative")
print("  else              → Neutral")
print()


# ======================================================================
# SECTION 2: ML-Based Sentiment (TF-IDF + LogReg)
# ======================================================================
print("=" * 70)
print("SECTION 2: ML-BASED SENTIMENT (TF-IDF + LOGISTIC REGRESSION)")
print("=" * 70)
print()
print("Labeled training data → learn which word patterns predict sentiment")
print()

# Larger synthetic sentiment dataset (binary)
positive_texts = [
    "This movie was absolutely wonderful and touching",
    "Great acting and a compelling story throughout",
    "I loved every minute of this masterpiece",
    "Highly recommend this to everyone I know",
    "One of the best films I have ever seen",
    "Brilliant direction and outstanding performances",
    "Fantastic storyline kept me hooked till the end",
    "Amazing film with incredible emotional depth",
    "Superb casting and beautiful cinematography",
    "Totally blown away by this incredible movie",
    "Excellent pacing and a satisfying conclusion",
    "A must-watch film that exceeded all expectations",
    "This product works perfectly and is great value",
    "Arrived quickly and exactly as described love it",
    "Best purchase I have made in years highly recommend",
    "Outstanding quality far exceeded my expectations",
    "Works flawlessly could not be happier with this",
    "Five stars absolutely perfect product service",
    "Great customer service fast shipping good product",
    "Really pleased with this exceeded all expectations",
]

negative_texts = [
    "This movie was absolutely terrible and boring",
    "Poor acting and a weak uninspiring story",
    "I hated every minute of this disaster",
    "Do not waste your time or money on this",
    "One of the worst films I have ever seen",
    "Dreadful direction and disappointing performances",
    "Awful storyline with no coherent plot whatsoever",
    "Terrible film with zero emotional impact",
    "Horrible casting and ugly cinematography choices",
    "Completely let down by this disaster of a film",
    "Worst pacing I have experienced deeply unsatisfying",
    "Avoid this film it will disappoint everyone",
    "This product broke after just two days of use",
    "Arrived damaged and nothing like the description",
    "Worst purchase I have made in years avoid it",
    "Disappointing quality far below my expectations",
    "Does not work at all extremely frustrated",
    "One star completely useless product terrible",
    "Appalling customer service slow shipping bad product",
    "Really angry with this did not meet any expectations",
]

X_raw_ml = positive_texts + negative_texts
y_raw_ml = [1] * len(positive_texts) + [0] * len(negative_texts)

# Shuffle
perm    = np.random.permutation(len(X_raw_ml))
X_raw_ml = [X_raw_ml[i] for i in perm]
y_raw_ml = [y_raw_ml[i]  for i in perm]

split_ml  = int(0.75 * len(X_raw_ml))
X_tr_ml, X_te_ml = X_raw_ml[:split_ml], X_raw_ml[split_ml:]
y_tr_ml, y_te_ml = y_raw_ml[:split_ml], y_raw_ml[split_ml:]

SK_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve,
                                  precision_recall_curve, average_precision_score)
    from sklearn.pipeline import Pipeline
    SK_AVAILABLE = True

    sentiment_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 3), max_features=5000,
                                  sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=500, C=5.0)),
    ])

    sentiment_pipeline.fit(X_tr_ml, y_tr_ml)
    proba   = sentiment_pipeline.predict_proba(X_te_ml)[:, 1]
    preds_ml= sentiment_pipeline.predict(X_te_ml)
    acc_ml  = accuracy_score(y_te_ml, preds_ml)
    auc_ml  = roc_auc_score(y_te_ml, proba)

    print(f"  TF-IDF + LogReg Sentiment:")
    print(f"  Test accuracy: {acc_ml:.4f} ({acc_ml*100:.1f}%)")
    print(f"  ROC-AUC:       {auc_ml:.4f}")
    print()

    # Predict on our VADER test sentences with ML model
    print("  ML model on the same test sentences:")
    ml_probs = sentiment_pipeline.predict_proba(test_sentences)[:, 1]
    for sent, prob in zip(test_sentences, ml_probs):
        label = "Positive" if prob >= 0.5 else "Negative"
        print(f"  {sent[:50]:<50} {prob:.3f}  {label}")
    print()

    # Top positive and negative words
    feat_names = sentiment_pipeline.named_steps["tfidf"].get_feature_names_out()
    coef_ml    = sentiment_pipeline.named_steps["clf"].coef_[0]

    print("  Top positive indicators:")
    for i in coef_ml.argsort()[-8:][::-1]:
        print(f"    {feat_names[i]:<20}: +{coef_ml[i]:.3f}")
    print()
    print("  Top negative indicators:")
    for i in coef_ml.argsort()[:8]:
        print(f"    {feat_names[i]:<20}: {coef_ml[i]:.3f}")
    print()

except ImportError:
    print("  scikit-learn not installed: pip install scikit-learn")
    print("  Expected: ~90-95% accuracy on this dataset")
    print()
    acc_ml = 0.93
    auc_ml = 0.97


# ======================================================================
# SECTION 3: HuggingFace Transformers (State-of-the-Art)
# ======================================================================
print("=" * 70)
print("SECTION 3: HUGGINGFACE TRANSFORMERS — BERT SENTIMENT")
print("=" * 70)
print()
print("HuggingFace pipeline() wraps pretrained BERT models in one line.")
print()
print("  from transformers import pipeline")
print("  classifier = pipeline('sentiment-analysis')")
print("  result = classifier('I love this movie!')")
print("  # [{'label': 'POSITIVE', 'score': 0.9998}]")
print()
print("Available sentiment models on HuggingFace Hub:")
print("  • distilbert-base-uncased-finetuned-sst-2-english  (fast, binary)")
print("  • cardiffnlp/twitter-roberta-base-sentiment-latest (social media)")
print("  • nlptown/bert-base-multilingual-uncased-sentiment (1-5 stars)")
print("  • ProsusAI/finbert                                 (finance)")
print()

HF_AVAILABLE = False
try:
    from transformers import pipeline as hf_pipeline
    HF_AVAILABLE = True

    print("  HuggingFace transformers available! Running...")
    print("  (First run downloads ~250MB model — cached afterwards)")
    print()

    clf_hf = hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        truncation=True, max_length=512
    )

    print("  HuggingFace BERT sentiment on test sentences:")
    hf_results = clf_hf(test_sentences)

    for sent, res in zip(test_sentences, hf_results):
        label = res["label"]
        score = res["score"]
        print(f"  {sent[:50]:<50} {score:.3f}  {label}")
    print()

except ImportError:
    print("  transformers not installed: pip install transformers")
    print()
    print("  Expected output (distilbert finetuned on SST-2):")
    print()
    simulated_hf = [
        ("POSITIVE", 0.9998), ("NEGATIVE", 0.9996), ("POSITIVE", 0.6823),
        ("POSITIVE", 0.9432), ("NEGATIVE", 0.9995), ("POSITIVE", 0.9999),
        ("NEGATIVE", 0.7234), ("NEGATIVE", 0.8412),
    ]
    for sent, (label, score) in zip(test_sentences, simulated_hf):
        print(f"  {sent[:50]:<50} {score:.3f}  {label}")
    print()
    print("  Note: BERT captures 'not good' (negative) correctly!")
    print("  Rule-based and TF-IDF+LR sometimes miss complex negation.")
    print()


# ======================================================================
# SECTION 4: Negation and Intensifier Handling
# ======================================================================
print("=" * 70)
print("SECTION 4: NEGATION AND INTENSIFIER HANDLING")
print("=" * 70)
print()
print("Negation and intensifiers are the hardest cases for BoW models:")
print()

negation_examples = [
    ("I love this",               "positive — baseline"),
    ("I do not love this",        "negative — negation"),
    ("I don't hate this",         "mild positive — double neg"),
    ("It's not bad at all",       "positive — negation + intensifier"),
    ("Not exactly what I wanted", "mild negative — soft negation"),
    ("I can't say I disliked it", "mild positive — can't + dislike"),
]

print("  Analyzing negation handling across approaches:")
print()
for text, expected in negation_examples:
    vader_s   = vader.polarity_scores(text)
    vader_lbl = vader_s["label"]
    vader_c   = vader_s["compound"]
    print(f"  Text:     {text!r}")
    print(f"  Expected: {expected}")
    print(f"  VADER:    {vader_lbl} (compound={vader_c:.3f})")

    if SK_AVAILABLE:
        ml_prob = sentiment_pipeline.predict_proba([text])[0, 1]
        ml_lbl  = "Positive" if ml_prob >= 0.5 else "Negative"
        print(f"  TF-IDF:   {ml_lbl} (prob={ml_prob:.3f})")
    print()


# ======================================================================
# SECTION 5: Aspect-Based Sentiment (Simple)
# ======================================================================
print("=" * 70)
print("SECTION 5: ASPECT-BASED SENTIMENT ANALYSIS")
print("=" * 70)
print()
print("Product reviews often have multiple aspects with different sentiments:")
print()
print("  'The camera is amazing but the battery life is terrible'")
print("  → camera: positive, battery: negative, overall: mixed")
print()

ASPECTS = {
    "food":    ["food", "meal", "dish", "taste", "flavor", "cuisine", "menu", "portion"],
    "service": ["service", "staff", "waiter", "waitress", "server", "host", "manager"],
    "price":   ["price", "cost", "value", "expensive", "cheap", "affordable", "overpriced"],
    "ambiance":["ambiance", "atmosphere", "decor", "music", "noise", "crowded", "cozy"],
    "delivery":["delivery", "shipping", "arrived", "packaging", "courier", "tracking"],
    "quality": ["quality", "material", "durable", "build", "construction", "solid", "flimsy"],
    "battery": ["battery", "charge", "charging", "power", "life", "drain", "last"],
    "camera":  ["camera", "photo", "picture", "lens", "zoom", "image", "shot"],
}


def aspect_sentiment(text, aspects, lexicon, window=4):
    """
    Detect aspect-level sentiment.
    For each aspect keyword found in text, score words within ±window positions.
    """
    tokens = text.lower().split()
    results = {}

    for aspect, keywords in aspects.items():
        aspect_scores = []
        for i, tok in enumerate(tokens):
            if tok in keywords:
                # Score words in a window around this aspect keyword
                window_tokens = tokens[max(0, i-window):min(len(tokens), i+window+1)]
                window_scores = [lexicon.get(t, 0) for t in window_tokens]
                aspect_score  = sum(window_scores)
                if aspect_score != 0:
                    aspect_scores.append(aspect_score)

        if aspect_scores:
            avg = np.mean(aspect_scores)
            results[aspect] = {
                "score": round(avg, 2),
                "label": "Positive" if avg > 0.5 else "Negative" if avg < -0.5 else "Mixed",
            }

    return results


review_examples = [
    "The food was absolutely delicious and the portions were generous, "
    "but the service was very slow and staff were rude. "
    "The price is fair for the quality of food. Great ambiance though.",

    "Camera quality is outstanding takes amazing photos, "
    "but the battery drains extremely fast and charging takes too long. "
    "Build quality is solid and the design is beautiful.",
]

for review in review_examples:
    print(f"  Review: {review[:80]}...")
    aspect_results = aspect_sentiment(review, ASPECTS, SENTIMENT_LEXICON)
    print(f"  Aspect Sentiments:")
    for aspect, result in aspect_results.items():
        bar = "+" * int(max(0, result["score"])) + "-" * int(abs(min(0, result["score"])))
        print(f"    {aspect:<12}: {result['label']:<10} (score={result['score']:.2f})  {bar}")
    print()


# ======================================================================
# SECTION 6: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 6: VISUALIZATIONS")
print("=" * 70)
print()


# --- PLOT 1: VADER compound scores visualization ---
print("Generating: VADER scores and sentiment distribution...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Sentiment Analysis: VADER, TF-IDF, and Score Distributions",
             fontsize=14, fontweight="bold")

# VADER scores for test sentences
vader_scores = [vader.polarity_scores(s)["compound"] for s in test_sentences]
short_labels = [s[:35] + "..." for s in test_sentences]
colors_v     = ["#2ECC71" if s >= 0.05 else "#E74C3C" if s <= -0.05 else "#95A5A6"
                for s in vader_scores]

bars_v = axes[0].barh(range(len(vader_scores)), vader_scores, color=colors_v,
                       edgecolor="white", linewidth=1.5)
axes[0].set_yticks(range(len(short_labels)))
axes[0].set_yticklabels(short_labels, fontsize=8)
axes[0].axvline(x=0.05,  color="#2ECC71", linestyle="--", linewidth=1.5, alpha=0.7)
axes[0].axvline(x=-0.05, color="#E74C3C", linestyle="--", linewidth=1.5, alpha=0.7)
axes[0].axvline(x=0,     color="black",   linewidth=1)
axes[0].set_xlabel("VADER Compound Score")
axes[0].set_title("VADER Sentiment Scores\n(green=positive, red=negative)",
                  fontsize=11, fontweight="bold")
axes[0].set_xlim(-1.2, 1.2)
axes[0].grid(axis="x", alpha=0.3)
for i, score in enumerate(vader_scores):
    axes[0].text(score + 0.02 * np.sign(score), i, f"{score:.2f}",
                 va="center", fontsize=8)

# ROC Curve
if SK_AVAILABLE:
    fpr, tpr, _ = roc_curve(y_te_ml, proba)
    axes[1].plot(fpr, tpr, "b-", linewidth=2.5, label=f"TF-IDF+LR (AUC={auc_ml:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random (AUC=0.5)")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="blue")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"ROC Curve — Sentiment Classifier\nAUC = {auc_ml:.4f}",
                      fontsize=11, fontweight="bold")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.02)
else:
    axes[1].text(0.5, 0.5, "Requires sklearn\n(ROC Curve)", ha="center", va="center",
                 fontsize=12, transform=axes[1].transAxes)
    axes[1].axis("off")

# Approach comparison bar chart
approaches  = ["Rule-Based\n(VADER)", "TF-IDF\n+LogReg", "BERT\n(distilbert)"]
typical_acc = [0.72, 0.88, 0.94]  # typical on SST-2 benchmark
typical_auc = [0.77, 0.93, 0.98]
colors_ap   = ["#F39C12", "#3498DB", "#E74C3C"]

x_ap = np.arange(len(approaches))
w_ap = 0.35
axes[2].bar(x_ap - w_ap/2, [a*100 for a in typical_acc], w_ap, label="Accuracy",
            color=[c for c in colors_ap], alpha=0.85, edgecolor="white")
axes[2].bar(x_ap + w_ap/2, [a*100 for a in typical_auc], w_ap, label="ROC-AUC",
            color=[c for c in colors_ap], alpha=0.55, edgecolor="white", hatch="//")
axes[2].set_xticks(x_ap)
axes[2].set_xticklabels(approaches, fontsize=10)
axes[2].set_ylim(0, 110)
axes[2].set_ylabel("Score (%)")
axes[2].set_title("Approach Comparison\n(typical on SST-2 benchmark)",
                  fontsize=11, fontweight="bold")
axes[2].legend(); axes[2].grid(axis="y", alpha=0.3)

# Speed/cost labels
costs = ["No training\nneeded", "Seconds\nto train", "GPU needed\n~10 min"]
for i, (acc, cost) in enumerate(zip(typical_acc, costs)):
    axes[2].text(i, acc * 100 + 2, cost, ha="center", fontsize=8, color=colors_ap[i],
                 fontweight="bold")

plt.tight_layout()
plt.savefig("../visuals/sentiment_analysis/sentiment_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: sentiment_comparison.png")


# --- PLOT 2: VADER gate breakdown + negation handling ---
print("Generating: VADER components and negation visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("VADER Components and Negation Handling",
             fontsize=14, fontweight="bold")

# VADER pos/neg/neu breakdown for test sentences
vader_all = [vader.polarity_scores(s) for s in test_sentences]
pos_v = [d["pos"] for d in vader_all]
neg_v = [d["neg"] for d in vader_all]
neu_v = [d["neu"] for d in vader_all]

x_v = np.arange(len(test_sentences))
axes[0].bar(x_v, pos_v, label="Positive", color="#2ECC71", alpha=0.85)
axes[0].bar(x_v, neg_v, bottom=pos_v, label="Negative", color="#E74C3C", alpha=0.85)
axes[0].bar(x_v, neu_v, bottom=[p+n for p,n in zip(pos_v, neg_v)], label="Neutral", color="#BDC3C7", alpha=0.85)
axes[0].set_xticks(x_v)
axes[0].set_xticklabels([f"s{i}" for i in range(len(test_sentences))], fontsize=9)
axes[0].set_ylabel("Proportion"); axes[0].set_ylim(0, 1.05)
axes[0].set_title("VADER Sentiment Breakdown\n(Stacked: pos / neg / neu)",
                  fontsize=11, fontweight="bold")
axes[0].legend(loc="lower right"); axes[0].grid(axis="y", alpha=0.3)

# Negation impact visualization
negation_pairs = [
    ("good",        "not good"),
    ("bad",         "not bad"),
    ("amazing",     "not amazing"),
    ("terrible",    "not terrible"),
    ("love this",   "don't love this"),
    ("hate this",   "don't hate this"),
]

bases_v = [vader.polarity_scores(b)["compound"] for b, _ in negation_pairs]
negated_v = [vader.polarity_scores(n)["compound"] for _, n in negation_pairs]
labels_np = [f"{b} →\n{n}" for b, n in negation_pairs]

x_np = np.arange(len(negation_pairs))
axes[1].bar(x_np - 0.2, bases_v, 0.35, label="Original", color="#3498DB", alpha=0.85, edgecolor="white")
axes[1].bar(x_np + 0.2, negated_v, 0.35, label="Negated", color="#E74C3C", alpha=0.85, edgecolor="white")
axes[1].axhline(y=0, color="black", linewidth=1.5)
axes[1].axhline(y=0.05, color="#2ECC71", linestyle="--", linewidth=1, alpha=0.6)
axes[1].axhline(y=-0.05, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.6)
axes[1].set_xticks(x_np)
axes[1].set_xticklabels(labels_np, fontsize=8)
axes[1].set_ylabel("VADER Compound Score")
axes[1].set_title("Negation Impact on VADER Score\n(negation dampens, partially flips)",
                  fontsize=11, fontweight="bold")
axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
axes[1].set_ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig("../visuals/sentiment_analysis/vader_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: vader_analysis.png")


# --- PLOT 3: Aspect-based sentiment visualization ---
print("Generating: Aspect-based sentiment visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Aspect-Based Sentiment Analysis",
             fontsize=14, fontweight="bold")

for ax_idx, (review, ax) in enumerate(zip(review_examples, axes)):
    aspect_results = aspect_sentiment(review, ASPECTS, SENTIMENT_LEXICON)
    if not aspect_results:
        ax.text(0.5, 0.5, "No aspects detected", ha="center", va="center",
                fontsize=12, transform=ax.transAxes)
        continue

    aspects_found  = list(aspect_results.keys())
    scores_found   = [aspect_results[a]["score"] for a in aspects_found]
    colors_a       = ["#2ECC71" if s > 0.5 else "#E74C3C" if s < -0.5 else "#F39C12"
                       for s in scores_found]

    bars_a = ax.barh(aspects_found, scores_found, color=colors_a,
                      edgecolor="white", linewidth=1.5, height=0.5)
    ax.axvline(x=0, color="black", linewidth=2)
    ax.axvline(x=0.5,  color="#2ECC71", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=-0.5, color="#E74C3C", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Aspect Sentiment Score")
    ax.set_title(f"Review {ax_idx+1} Aspect Sentiments\n{review[:50]}...",
                 fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(-5, 5)

    for bar, score in zip(bars_a, scores_found):
        ax.text(score + 0.1 * np.sign(score), bar.get_y() + bar.get_height()/2,
                f"{score:+.1f}", va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("../visuals/sentiment_analysis/aspect_sentiment.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: aspect_sentiment.png")


print()
print("=" * 70)
print("NLP ALGORITHM 2: SENTIMENT ANALYSIS COMPLETE!")
print("=" * 70)
print()
print("What you built:")
print("  ✓ VADER-style rule-based scorer with negation + intensifier handling")
print("  ✓ TF-IDF + LogReg binary sentiment (trigrams for negation capture)")
print("  ✓ HuggingFace pipeline (distilbert-base-uncased-finetuned-sst-2)")
print("  ✓ Aspect-based sentiment (multi-aspect scoring from lexicon)")
print()
print("Key takeaways:")
print("  Rule-based: ~72% accuracy — no training data, fast, interpretable")
print("  TF-IDF+LR:  ~88% accuracy — needs labels, very fast, interpretable")
print("  BERT:       ~94% accuracy — needs GPU, handles context perfectly")
print("  Negation is hard for BoW models; BERT handles it naturally")
print()
print("3 Visualizations saved to: ../visuals/sentiment_analysis/")
print("  1. sentiment_comparison.png — VADER scores + ROC curve + model comparison")
print("  2. vader_analysis.png       — pos/neg/neu breakdown + negation impact")
print("  3. aspect_sentiment.png     — aspect-level heatmaps")
print()
print("Next: Algorithm 3 → LSTM Text Classifier")
