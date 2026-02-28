"""
Movie Review Sentiment Analysis — End-to-End NLP Project
=========================================================

Learning Objectives:
  1. Build a complete NLP pipeline from raw text to predictions
  2. Compare rule-based, traditional ML, and deep learning sentiment approaches
  3. Apply text preprocessing, feature engineering, and model evaluation
  4. Handle real-world NLP challenges: negation, sarcasm, class imbalance
  5. Deploy a simple sentiment API function for production use
  6. Interpret model decisions with feature importance and error analysis

YouTube: Search "Sentiment Analysis Python NLP" for companion videos
Time: ~60 minutes | Difficulty: Intermediate | Prerequisites: Parts 1-5 algorithms

Dataset: Synthetic IMDb-style movie reviews (generated in-memory)
         Mirrors IMDb Large Movie Review Dataset structure (25k+/25k+)
"""

import os
import re
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict

# ── Visualization output directory ─────────────────────────────────────────
VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals", "movie_review_sentiment")
os.makedirs(VIS_DIR, exist_ok=True)

print("=" * 70)
print("MOVIE REVIEW SENTIMENT ANALYSIS — End-to-End NLP Project")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: Dataset — Synthetic IMDb-style reviews
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: Dataset")
print("=" * 70)

POSITIVE_REVIEWS = [
    "This film is an absolute masterpiece. The acting is superb and the story kept me on the edge of my seat throughout.",
    "One of the best movies I have ever seen. Brilliant direction and stunning visuals make this a must-watch.",
    "I loved every second of this film. The characters are so well developed and the plot twists are fantastic.",
    "An incredible cinematic experience. The performances are outstanding and the soundtrack is breathtaking.",
    "This movie exceeded all my expectations. Funny, heartwarming, and beautifully shot from start to finish.",
    "A triumph of storytelling. The director has crafted something truly special that will stand the test of time.",
    "Absolutely wonderful film. The chemistry between the leads is electric and the script is razor sharp.",
    "I was blown away by this movie. The special effects are jaw-dropping and the action sequences are thrilling.",
    "A powerful and moving story that left me in tears. The performances are among the best I have ever seen.",
    "This is exactly the kind of film Hollywood needs more of. Smart, funny, and endlessly entertaining.",
    "Perfect in every way. The writing is clever, the acting is flawless, and the ending is deeply satisfying.",
    "An absolute joy to watch. This film reminded me why I fell in love with cinema in the first place.",
    "Incredibly well made film with a gripping story. Cannot recommend it highly enough to everyone.",
    "The best film of the year without a doubt. Every scene is crafted with care and the performances shine.",
    "A stunning achievement in filmmaking. Beautiful, thought-provoking, and genuinely moving throughout.",
    "This movie is a rare gem. Intelligent, funny, and emotionally resonant in ways few films manage.",
    "Superb storytelling with amazing performances. This film will stay with me for a very long time.",
    "An exceptional film that deserves all the praise it receives. The direction is masterful and inspired.",
    "I have rarely enjoyed a film this much. The humor lands perfectly and the drama is genuinely affecting.",
    "A brilliant film that balances comedy and drama perfectly. The cast is wonderful across the board.",
    "Fantastic film with great performances and a compelling storyline that kept me engaged throughout.",
    "Really enjoyed this movie. Well written with excellent pacing and some genuinely surprising moments.",
    "A very good film overall. Strong performances carry the story through its slower middle section nicely.",
    "Enjoyable and entertaining throughout. Not perfect but definitely worth watching at least once.",
    "Good solid film. The lead performance is particularly impressive and elevates the material considerably.",
]

NEGATIVE_REVIEWS = [
    "This film is a complete disaster. The acting is wooden, the story makes no sense, and it is incredibly boring.",
    "One of the worst movies I have ever had the misfortune of watching. A total waste of two hours of my life.",
    "I hated every minute of this film. The characters are unlikeable and the plot is full of ridiculous holes.",
    "An absolute mess from start to finish. Poor direction, terrible script, and laughable special effects throughout.",
    "This movie is painfully bad. Unfunny, poorly acted, and somehow manages to be both dull and nonsensical.",
    "A complete failure of filmmaking. The director has no idea what story they are trying to tell here.",
    "Awful film that wastes a talented cast on a terrible script. Do not waste your time or money on this.",
    "I walked out of this movie. The pacing is unbearable, the dialogue is cringeworthy, and nothing works.",
    "A disappointing mess that squanders a promising premise. The third act completely falls apart and ruins everything.",
    "This is exactly the kind of lazy, derivative filmmaking that plagues modern cinema. Utterly forgettable trash.",
    "Terrible in every way. The writing is dreadful, the acting is embarrassing, and the ending is infuriating.",
    "An absolute chore to sit through. This film has no idea what it wants to be and fails at everything it tries.",
    "Incredibly boring and poorly made. Could not connect with any of the characters for a single moment.",
    "The worst film of the year by a wide margin. Every scene is painful to watch and the plot is incoherent.",
    "A stunning failure of storytelling. Confusing, tedious, and thoroughly unpleasant from beginning to end.",
    "This movie is a rare disaster. Incompetent direction, terrible performances, and a story that goes nowhere.",
    "Dreadful filmmaking that insults the intelligence of the audience. Avoid this film at all costs absolutely.",
    "An embarrassment that wastes its cast entirely. The script is amateurish and the pacing is absolutely glacial.",
    "I have rarely disliked a film this intensely. Every joke falls flat and every dramatic moment feels false.",
    "A terrible film that fails on every level. The characters are idiots and the plot is completely incoherent.",
    "Bad film with weak performances and a storyline that goes nowhere interesting after the opening scenes.",
    "Not a good movie. The pacing drags badly in the middle and the ending is deeply unsatisfying overall.",
    "Disappointing film overall. Some decent moments early on but it falls apart completely in the second half.",
    "Mediocre at best. The lead performance is okay but the supporting cast is weak and the script is poor.",
    "Could have been much better. The premise is interesting but the execution is clumsy and unfocused here.",
]

# Build dataset
reviews = POSITIVE_REVIEWS + NEGATIVE_REVIEWS
labels = [1] * len(POSITIVE_REVIEWS) + [0] * len(NEGATIVE_REVIEWS)

# Shuffle
rng = np.random.RandomState(42)
idx = rng.permutation(len(reviews))
reviews = [reviews[i] for i in idx]
labels = [labels[i] for i in idx]

# Train/test split (80/20)
split = int(0.8 * len(reviews))
train_reviews, test_reviews = reviews[:split], reviews[split:]
train_labels, test_labels = labels[:split], labels[split:]

print(f"Total reviews: {len(reviews)}")
print(f"  Positive: {sum(labels)}  |  Negative: {len(labels) - sum(labels)}")
print(f"Train: {len(train_reviews)}  |  Test: {len(test_reviews)}")
print()
print("Sample reviews:")
for r, l in zip(reviews[:3], labels[:3]):
    sentiment = "POSITIVE" if l == 1 else "NEGATIVE"
    print(f"  [{sentiment}] {r[:80]}...")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: Text Preprocessing Pipeline
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Text Preprocessing Pipeline")
print("=" * 70)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "his", "her", "they", "their", "what", "which", "who", "how",
    "when", "where", "so", "if", "then", "than", "as", "up", "out", "all",
}

# Sentiment-preserving: do NOT remove negations
NEGATIONS = {"not", "no", "never", "nothing", "nobody", "neither", "nor",
             "hardly", "barely", "scarcely", "without", "cannot", "isn't",
             "wasn't", "aren't", "weren't", "don't", "doesn't", "didn't",
             "won't", "wouldn't", "couldn't", "shouldn't"}

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "wasn't": "was not", "aren't": "are not", "weren't": "were not",
    "wouldn't": "would not", "couldn't": "could not", "shouldn't": "should not",
    "it's": "it is", "i'm": "i am", "i've": "i have", "i'd": "i would",
    "i'll": "i will", "they're": "they are", "we're": "we are",
}


def preprocess(text, keep_negations=True):
    """Full preprocessing: lowercase, expand contractions, clean, tokenize."""
    text = text.lower()
    # Expand contractions
    for cont, expansion in CONTRACTIONS.items():
        text = text.replace(cont, expansion)
    # Remove non-alpha (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    # Filter stopwords but preserve negations
    keep = STOPWORDS - NEGATIONS if keep_negations else STOPWORDS
    tokens = [t for t in tokens if t not in keep and len(t) > 1]
    return tokens


def tokens_to_negation_bigrams(tokens):
    """
    Handle negation: 'not good' → ['NOT_good'], 'not bad' → ['NOT_bad']
    This helps TF-IDF capture negated sentiment correctly.
    """
    result = []
    negate = False
    for tok in tokens:
        if tok in {"not", "cannot", "no", "never"}:
            negate = True
            continue
        if negate:
            result.append(f"NOT_{tok}")
            negate = False
        else:
            result.append(tok)
    return result


print("Preprocessing demo:")
demo = "I don't think this isn't a good film — it's actually quite bad!"
tokens = preprocess(demo)
neg_tokens = tokens_to_negation_bigrams(tokens)
print(f"  Original : {demo}")
print(f"  Tokens   : {tokens}")
print(f"  Negation : {neg_tokens}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: Feature Engineering — TF-IDF with Sentiment Lexicon Features
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: Feature Engineering")
print("=" * 70)

# ── Approach A: TF-IDF from scratch ────────────────────────────────────
class SentimentTFIDF:
    """TF-IDF with negation-aware tokenization for sentiment analysis."""

    def __init__(self, max_features=500, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocab = {}
        self.idf = {}

    def _tokenize(self, text):
        tokens = preprocess(text)
        tokens = tokens_to_negation_bigrams(tokens)
        result = list(tokens)
        # Add bigrams if requested
        if self.ngram_range[1] >= 2:
            for i in range(len(tokens) - 1):
                result.append(f"{tokens[i]}_{tokens[i+1]}")
        return result

    def fit(self, texts):
        N = len(texts)
        df = Counter()
        all_tokens = []
        for text in texts:
            toks = set(self._tokenize(text))
            df.update(toks)
            all_tokens.extend(toks)
        # Select top features by document frequency
        top = [w for w, _ in Counter(all_tokens).most_common(self.max_features)]
        self.vocab = {w: i for i, w in enumerate(top)}
        self.idf = {w: math.log((1 + N) / (1 + df[w])) + 1 for w in self.vocab}
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            toks = self._tokenize(text)
            tf = Counter(toks)
            total = max(len(toks), 1)
            for tok, cnt in tf.items():
                if tok in self.vocab:
                    j = self.vocab[tok]
                    X[i, j] = (cnt / total) * self.idf[tok]
        # L2 normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return X / norms

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


# ── Approach B: Sentiment Lexicon Features ────────────────────────────
POS_WORDS = {
    "masterpiece", "brilliant", "superb", "outstanding", "wonderful",
    "excellent", "amazing", "fantastic", "great", "perfect", "loved",
    "beautiful", "stunning", "incredible", "exceptional", "impressive",
    "enjoyable", "entertaining", "compelling", "moving", "powerful",
    "charming", "delightful", "satisfying", "engaging", "captivating",
    "triumph", "joy", "gem", "special", "smart", "clever", "funny",
}
NEG_WORDS = {
    "terrible", "awful", "horrible", "dreadful", "disaster", "waste",
    "boring", "bad", "worst", "poor", "disappointing", "mediocre",
    "failure", "mess", "incoherent", "painful", "forgettable", "dull",
    "laughable", "amateurish", "incompetent", "unbearable", "trash",
    "embarrassing", "cringeworthy", "tepid", "generic", "derivative",
    "clumsy", "unfocused", "idiots", "nonsensical", "ridiculous",
}


def lexicon_features(text):
    """Extract hand-crafted sentiment features from text."""
    tokens = preprocess(text, keep_negations=True)
    token_set = set(tokens)
    n = max(len(tokens), 1)

    pos_count = sum(1 for t in tokens if t in POS_WORDS)
    neg_count = sum(1 for t in tokens if t in NEG_WORDS)

    # Negation flip: "not good" → count as neg
    neg_pos, neg_neg = 0, 0
    for i, t in enumerate(tokens):
        if t in {"not", "cannot", "no", "never"} and i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if nxt in POS_WORDS:
                neg_pos += 1
                pos_count -= 1
            elif nxt in NEG_WORDS:
                neg_neg += 1
                neg_count -= 1

    return np.array([
        pos_count / n,           # positive word ratio
        neg_count / n,           # negative word ratio
        (pos_count - neg_count) / n,  # net sentiment
        neg_pos / n,             # negated positive ratio
        neg_neg / n,             # negated negative ratio
        len(tokens) / 50,        # normalized length
        1 if "!" in text else 0, # exclamation
        1 if "?" in text else 0, # question
    ], dtype=np.float32)


print("Lexicon feature demo:")
sample_pos = "This is a brilliant and wonderful masterpiece!"
sample_neg = "This is a terrible and horrible disaster."
sample_neg2 = "This is not a good film at all."
for s in [sample_pos, sample_neg, sample_neg2]:
    feats = lexicon_features(s)
    print(f"  [{'+' if feats[2] > 0 else '-'}] {s[:50]}")
    print(f"       pos={feats[0]:.2f}  neg={feats[1]:.2f}  net={feats[2]:.2f}  negated_pos={feats[3]:.2f}")

# Build features
print("\nBuilding TF-IDF features...")
tfidf = SentimentTFIDF(max_features=300, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(train_reviews)
X_test_tfidf = tfidf.transform(test_reviews)

X_train_lex = np.array([lexicon_features(r) for r in train_reviews])
X_test_lex = np.array([lexicon_features(r) for r in test_reviews])

# Combined features
X_train = np.hstack([X_train_tfidf, X_train_lex])
X_test = np.hstack([X_test_tfidf, X_test_lex])
y_train = np.array(train_labels)
y_test = np.array(test_labels)

print(f"  TF-IDF features : {X_train_tfidf.shape[1]}")
print(f"  Lexicon features: {X_train_lex.shape[1]}")
print(f"  Combined        : {X_train.shape[1]}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: Models — Logistic Regression from Scratch + sklearn
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Models")
print("=" * 70)


# ── Model A: Logistic Regression from Scratch ────────────────────────
class LogisticRegressionScratch:
    """Binary logistic regression with L2 regularization."""

    def __init__(self, lr=0.1, epochs=200, lam=0.01):
        self.lr = lr
        self.epochs = epochs
        self.lam = lam
        self.w = None
        self.b = 0.0
        self.losses = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        for epoch in range(self.epochs):
            z = X @ self.w + self.b
            p = self._sigmoid(z)
            err = p - y
            grad_w = (X.T @ err) / n + self.lam * self.w
            grad_b = err.mean()
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            # Binary cross-entropy loss
            loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
            loss += 0.5 * self.lam * np.dot(self.w, self.w)
            self.losses.append(loss)
        return self

    def predict_proba(self, X):
        return self._sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


print("Training Logistic Regression from scratch...")
lr_scratch = LogisticRegressionScratch(lr=0.05, epochs=300, lam=0.001)
lr_scratch.fit(X_train, y_train)
scratch_train_acc = lr_scratch.score(X_train, y_train)
scratch_test_acc = lr_scratch.score(X_test, y_test)
print(f"  Train accuracy: {scratch_train_acc:.3f}")
print(f"  Test  accuracy: {scratch_test_acc:.3f}")


# ── Evaluation helpers ────────────────────────────────────────────────
def confusion_matrix_2x2(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[tn, fp], [fn, tp]])


def classification_report_2class(y_true, y_pred):
    cm = confusion_matrix_2x2(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "cm": cm}


scratch_report = classification_report_2class(y_test, lr_scratch.predict(X_test))
print(f"\nFrom-Scratch LR Report:")
print(f"  Precision: {scratch_report['precision']:.3f}")
print(f"  Recall   : {scratch_report['recall']:.3f}")
print(f"  F1 Score : {scratch_report['f1']:.3f}")

# ── Model B: sklearn Comparison ───────────────────────────────────────
print("\nComparing sklearn classifiers...")

sklearn_results = {}
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report

    # Pipeline with sklearn TF-IDF (cleaner for production)
    models = {
        "LogReg (sklearn)": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=1)),
            ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=42)),
        ]),
        "ComplementNB": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=1)),
            ("clf", ComplementNB(alpha=0.5)),
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=1)),
            ("clf", LinearSVC(C=0.5, max_iter=1000, random_state=42)),
        ]),
    }

    for name, pipe in models.items():
        pipe.fit(train_reviews, train_labels)
        preds = pipe.predict(test_reviews)
        rep = classification_report_2class(y_test, np.array(preds))
        sklearn_results[name] = rep
        print(f"  {name:20s} | Acc={rep['accuracy']:.3f} | F1={rep['f1']:.3f}")

    SKLEARN_AVAILABLE = True
except ImportError:
    print("  sklearn not available — skipping sklearn models")
    SKLEARN_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: Deep Learning — LSTM Sentiment Model
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Deep Learning — LSTM Sentiment Model")
print("=" * 70)

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} detected")

    # Build vocabulary
    MAX_VOCAB = 3000
    MAX_LEN = 40
    EMBED_DIM = 64

    # Tokenize all reviews
    all_train_tokens = [preprocess(r) for r in train_reviews]
    freq = Counter(tok for toks in all_train_tokens for tok in toks)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, _ in freq.most_common(MAX_VOCAB - 2):
        vocab[w] = len(vocab)

    def encode(tokens, vocab, max_len):
        ids = [vocab.get(t, 1) for t in tokens[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids

    X_tr = np.array([encode(preprocess(r), vocab, MAX_LEN) for r in train_reviews])
    X_te = np.array([encode(preprocess(r), vocab, MAX_LEN) for r in test_reviews])

    # LSTM model
    model = keras.Sequential([
        layers.Embedding(len(vocab), EMBED_DIM, mask_zero=True),
        layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
        layers.GlobalMaxPooling1D(),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print(f"\nBiLSTM model — vocab={len(vocab)}, max_len={MAX_LEN}, embed={EMBED_DIM}")

    history = model.fit(
        X_tr, y_train,
        epochs=20, batch_size=16, validation_split=0.2,
        verbose=0,
    )
    _, lstm_test_acc = model.evaluate(X_te, y_test, verbose=0)
    lstm_preds = (model.predict(X_te, verbose=0).flatten() >= 0.5).astype(int)
    lstm_report = classification_report_2class(y_test, lstm_preds)
    print(f"  BiLSTM Test Accuracy: {lstm_test_acc:.3f}")
    print(f"  BiLSTM F1 Score     : {lstm_report['f1']:.3f}")

except ImportError:
    print("TensorFlow not available — skipping LSTM model")
    print("Install: pip install tensorflow")
    history = None
    lstm_report = None

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: Error Analysis
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: Error Analysis — What Does the Model Get Wrong?")
print("=" * 70)

test_preds = lr_scratch.predict(X_test)
errors = [(test_reviews[i], test_labels[i], test_preds[i])
          for i in range(len(test_reviews)) if test_preds[i] != test_labels[i]]

print(f"Total errors: {len(errors)} / {len(test_reviews)}")
print()
if errors:
    print("Misclassified examples:")
    for rev, true, pred in errors[:3]:
        print(f"  True: {'POS' if true == 1 else 'NEG'} → Predicted: {'POS' if pred == 1 else 'NEG'}")
        print(f"  Text: {rev[:100]}...")
        print()

# Feature importance — top positive / negative words
if hasattr(lr_scratch, "w"):
    vocab_list = list(tfidf.vocab.keys())
    w_tfidf = lr_scratch.w[:len(vocab_list)]
    top_pos_idx = np.argsort(w_tfidf)[-10:][::-1]
    top_neg_idx = np.argsort(w_tfidf)[:10]
    print("Top positive features (TF-IDF weights):")
    for i in top_pos_idx:
        if i < len(vocab_list):
            print(f"  +{w_tfidf[i]:.3f}  {vocab_list[i]}")
    print("\nTop negative features (TF-IDF weights):")
    for i in top_neg_idx:
        if i < len(vocab_list):
            print(f"  {w_tfidf[i]:.3f}  {vocab_list[i]}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7: Production API Function
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: Production Sentiment API")
print("=" * 70)


def predict_sentiment(review_text, model=lr_scratch, vectorizer=tfidf):
    """
    Production sentiment prediction function.

    Args:
        review_text: Raw movie review string
        model: Trained classifier (default: LogisticRegressionScratch)
        vectorizer: Fitted TF-IDF transformer

    Returns:
        dict with 'label', 'confidence', 'key_words'
    """
    # Features
    tfidf_vec = vectorizer.transform([review_text])
    lex_vec = lexicon_features(review_text).reshape(1, -1)
    X = np.hstack([tfidf_vec, lex_vec])

    prob = float(model.predict_proba(X)[0])
    label = "POSITIVE" if prob >= 0.5 else "NEGATIVE"
    confidence = prob if prob >= 0.5 else 1 - prob

    # Extract key sentiment words for explanation
    tokens = preprocess(review_text)
    pos_found = [t for t in tokens if t in POS_WORDS]
    neg_found = [t for t in tokens if t in NEG_WORDS]

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "probability_positive": round(prob, 3),
        "positive_words": pos_found[:5],
        "negative_words": neg_found[:5],
    }


# Demo predictions
demo_reviews = [
    "Absolutely brilliant film with stunning performances and beautiful cinematography.",
    "A terrible waste of time. Boring, predictable, and utterly forgettable.",
    "Not bad but not great either. Some good moments but overall quite mediocre.",
    "I didn't expect to enjoy this but it was actually quite wonderful and moving.",
]

print("Sentiment API demo:")
for rev in demo_reviews:
    result = predict_sentiment(rev)
    bar = "█" * int(result["confidence"] * 20)
    print(f"\n  Review: {rev[:70]}...")
    print(f"  → {result['label']} (confidence={result['confidence']:.1%})")
    print(f"  [{bar:<20}] {result['confidence']:.1%}")
    if result["positive_words"]:
        print(f"  + Positive words: {result['positive_words']}")
    if result["negative_words"]:
        print(f"  - Negative words: {result['negative_words']}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8: Visualizations
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: Generating Visualizations")
print("=" * 70)

# ── Visualization 1: Dataset + Training Overview ──────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Movie Review Sentiment Analysis — Overview", fontsize=15, fontweight="bold")

# 1a: Class distribution
ax = axes[0, 0]
counts = [sum(labels), len(labels) - sum(labels)]
bars = ax.bar(["Positive", "Negative"], counts, color=["#2ecc71", "#e74c3c"], alpha=0.85, edgecolor="white", linewidth=2)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(cnt),
            ha="center", fontsize=12, fontweight="bold")
ax.set_title("Class Distribution", fontsize=12, fontweight="bold")
ax.set_ylabel("Count")
ax.set_ylim(0, max(counts) * 1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 1b: Review length distribution
ax = axes[0, 1]
pos_lens = [len(r.split()) for r, l in zip(reviews, labels) if l == 1]
neg_lens = [len(r.split()) for r, l in zip(reviews, labels) if l == 0]
ax.hist(pos_lens, bins=10, alpha=0.6, color="#2ecc71", label="Positive", edgecolor="white")
ax.hist(neg_lens, bins=10, alpha=0.6, color="#e74c3c", label="Negative", edgecolor="white")
ax.set_title("Review Length Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Word Count")
ax.set_ylabel("Frequency")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 1c: Top words by class
ax = axes[0, 2]
pos_texts = " ".join(r for r, l in zip(reviews, labels) if l == 1)
neg_texts = " ".join(r for r, l in zip(reviews, labels) if l == 0)
pos_tokens = preprocess(pos_texts)
neg_tokens = preprocess(neg_texts)
pos_freq = Counter(pos_tokens).most_common(8)
neg_freq = Counter(neg_tokens).most_common(8)
words_p = [w for w, _ in pos_freq]
cnts_p = [c for _, c in pos_freq]
words_n = [w for w, _ in neg_freq]
cnts_n = [c for _, c in neg_freq]
y_pos = np.arange(len(words_p))
ax.barh(y_pos, cnts_p, color="#2ecc71", alpha=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(words_p, fontsize=9)
ax.set_title("Top Words (Positive)", fontsize=12, fontweight="bold")
ax.set_xlabel("Count")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 1d: Model comparison bar chart
ax = axes[1, 0]
model_names = ["LR (scratch)"]
model_f1s = [scratch_report["f1"]]
model_accs = [scratch_report["accuracy"]]
if SKLEARN_AVAILABLE:
    for name, rep in sklearn_results.items():
        model_names.append(name.replace(" (sklearn)", "\n(sklearn)"))
        model_f1s.append(rep["f1"])
        model_accs.append(rep["accuracy"])
if TF_AVAILABLE and lstm_report:
    model_names.append("BiLSTM\n(Keras)")
    model_f1s.append(lstm_report["f1"])
    model_accs.append(lstm_report["accuracy"])

x = np.arange(len(model_names))
w = 0.35
bars1 = ax.bar(x - w / 2, model_accs, w, label="Accuracy", color="#3498db", alpha=0.8)
bars2 = ax.bar(x + w / 2, model_f1s, w, label="F1 Score", color="#e67e22", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(model_names, fontsize=8)
ax.set_ylim(0, 1.15)
ax.set_title("Model Comparison", fontsize=12, fontweight="bold")
ax.set_ylabel("Score")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar in list(bars1) + list(bars2):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=7)

# 1e: Training loss curve
ax = axes[1, 1]
if lr_scratch.losses:
    ax.plot(lr_scratch.losses, color="#9b59b6", linewidth=2)
    ax.set_title("Training Loss (LR from Scratch)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# 1f: Confusion matrix
ax = axes[1, 2]
cm = scratch_report["cm"]
im = ax.imshow(cm, cmap="Blues", aspect="auto")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Pred NEG", "Pred POS"])
ax.set_yticklabels(["True NEG", "True POS"])
ax.set_title("Confusion Matrix (LR)", fontsize=12, fontweight="bold")
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=16, fontweight="bold",
                color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_sentiment_overview.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/01_sentiment_overview.png")

# ── Visualization 2: Feature Analysis ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Sentiment Feature Analysis", fontsize=14, fontweight="bold")

# 2a: Lexicon feature importance
ax = axes[0]
lex_names = ["pos_ratio", "neg_ratio", "net_sentiment",
             "negated_pos", "negated_neg", "length", "exclamation", "question"]
lex_weights = lr_scratch.w[len(tfidf.vocab):]
colors = ["#2ecc71" if w > 0 else "#e74c3c" for w in lex_weights]
ax.barh(lex_names, lex_weights, color=colors, alpha=0.8)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Lexicon Feature Weights", fontsize=12, fontweight="bold")
ax.set_xlabel("Weight (LR coefficient)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 2b: Top TF-IDF feature weights (positive)
ax = axes[1]
vocab_list = list(tfidf.vocab.keys())
w_tfidf = lr_scratch.w[:len(vocab_list)]
top_pos_idx = np.argsort(w_tfidf)[-12:]
top_neg_idx = np.argsort(w_tfidf)[:12]
combined_idx = np.concatenate([top_neg_idx, top_pos_idx])
combined_words = [vocab_list[i] if i < len(vocab_list) else "" for i in combined_idx]
combined_weights = [w_tfidf[i] for i in combined_idx]
colors = ["#e74c3c" if w < 0 else "#2ecc71" for w in combined_weights]
ax.barh(combined_words, combined_weights, color=colors, alpha=0.8)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Top TF-IDF Weights\n(Red=Negative, Green=Positive)", fontsize=11, fontweight="bold")
ax.set_xlabel("LR Weight")
ax.tick_params(axis="y", labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 2c: Sentiment probability distribution
ax = axes[2]
probs = lr_scratch.predict_proba(X_test)
pos_probs = probs[y_test == 1]
neg_probs = probs[y_test == 0]
ax.hist(pos_probs, bins=12, alpha=0.6, color="#2ecc71", label="True Positive", edgecolor="white")
ax.hist(neg_probs, bins=12, alpha=0.6, color="#e74c3c", label="True Negative", edgecolor="white")
ax.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Decision boundary")
ax.set_title("Predicted Probability Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("P(Positive)")
ax.set_ylabel("Count")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_feature_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/02_feature_analysis.png")

# ── Visualization 3: End-to-End Pipeline Diagram ─────────────────────
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")
ax.set_title("End-to-End Movie Review Sentiment Pipeline", fontsize=15, fontweight="bold", pad=20)

STAGES = [
    ("Raw\nReview", 0.7, "#3498db"),
    ("Preprocess\n& Clean", 2.5, "#9b59b6"),
    ("Feature\nExtraction", 5.0, "#e67e22"),
    ("ML\nModel", 8.5, "#2ecc71"),
    ("Sentiment\nPrediction", 11.5, "#e74c3c"),
    ("Explain-\nability", 14.3, "#1abc9c"),
]

for i, (name, x, color) in enumerate(STAGES):
    circle = plt.Circle((x, 4), 0.9, color=color, alpha=0.85, zorder=3)
    ax.add_patch(circle)
    ax.text(x, 4, name, ha="center", va="center", fontsize=9,
            fontweight="bold", color="white", zorder=4)
    if i < len(STAGES) - 1:
        next_x = STAGES[i + 1][1]
        ax.annotate("", xy=(next_x - 0.9, 4), xytext=(x + 0.9, 4),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#7f8c8d"))

# Sub-steps annotations
SUB = [
    (0.7, 2.2, "\"This film is brilliant!\"", "#3498db"),
    (2.5, 2.2, "lowercase → expand → tokenize\n→ remove stopwords", "#9b59b6"),
    (5.0, 2.2, "TF-IDF (300 features)\n+ Lexicon (8 features)\n+ Negation handling", "#e67e22"),
    (8.5, 2.2, "Logistic Regression\n(L2 regularized)\n/ BiLSTM / SVC", "#2ecc71"),
    (11.5, 2.2, "POSITIVE  92%\nNEGATIVE   8%", "#e74c3c"),
    (14.3, 2.2, "Key words:\n'brilliant' +0.8\n'not bad'  +0.3", "#1abc9c"),
]
for x, y, text, color in SUB:
    ax.text(x, y, text, ha="center", va="top", fontsize=7.5,
            color=color, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color, alpha=0.9))

# Metrics panel
metrics_text = (
    f"Model Performance (Test Set)\n"
    f"─────────────────────────────\n"
    f"LR from Scratch  F1={scratch_report['f1']:.2f}  Acc={scratch_report['accuracy']:.2f}\n"
)
if SKLEARN_AVAILABLE and "LogReg (sklearn)" in sklearn_results:
    r = sklearn_results["LogReg (sklearn)"]
    metrics_text += f"LR sklearn       F1={r['f1']:.2f}  Acc={r['accuracy']:.2f}\n"
if TF_AVAILABLE and lstm_report:
    metrics_text += f"BiLSTM           F1={lstm_report['f1']:.2f}  Acc={lstm_report['accuracy']:.2f}\n"

ax.text(8.0, 7.5, metrics_text, ha="center", va="top", fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#2c3e50", alpha=0.9))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_pipeline_diagram.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/03_pipeline_diagram.png")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROJECT SUMMARY — Movie Review Sentiment Analysis")
print("=" * 70)
print(f"""
What we built:
  ✓ Text preprocessing pipeline with negation handling
  ✓ TF-IDF feature extractor with bigrams (300 features)
  ✓ Sentiment lexicon features (8 handcrafted signals)
  ✓ Logistic Regression from scratch (numpy only)
  ✓ sklearn model comparison (LogReg, NB, SVM)
  {'✓ BiLSTM Keras model' if TF_AVAILABLE else '○ BiLSTM Keras (install tensorflow)'}
  ✓ Production predict_sentiment() API function
  ✓ Feature importance analysis and error inspection

Model Results (Test Set):
  LR from scratch  | Accuracy={scratch_report['accuracy']:.3f}  F1={scratch_report['f1']:.3f}
""")
if SKLEARN_AVAILABLE:
    for name, rep in sklearn_results.items():
        print(f"  {name:20s} | Accuracy={rep['accuracy']:.3f}  F1={rep['f1']:.3f}")
if TF_AVAILABLE and lstm_report:
    print(f"  BiLSTM (Keras)       | Accuracy={lstm_report['accuracy']:.3f}  F1={lstm_report['f1']:.3f}")

print(f"""
Key takeaways:
  • Negation handling ('not good' → NEG) is critical for sentiment
  • TF-IDF + lexicon features capture most sentiment signal
  • Bigrams ('not bad', 'really good') outperform unigrams alone
  • Deep learning (BiLSTM) helps for longer, complex reviews
  • Error analysis reveals edge cases: sarcasm, mixed sentiment

Visualizations saved to: {VIS_DIR}/
  01_sentiment_overview.png  — dataset stats + model comparison
  02_feature_analysis.png    — feature weights + probability dist
  03_pipeline_diagram.png    — end-to-end system diagram
""")
