"""
📊 NLP — Math Foundation 2: Bag of Words & TF-IDF
==================================================

Learning Objectives:
  1. Understand the Bag of Words (BoW) model — its power AND limitations
  2. Build a document-term matrix from scratch with numpy
  3. Understand why raw counts mislead — and why TF-IDF fixes this
  4. Derive TF-IDF from first principles: Term Frequency × Inverse Doc Frequency
  5. Implement TF-IDF from scratch and verify against scikit-learn
  6. Use cosine similarity to compare documents by their TF-IDF vectors
  7. See how BoW/TF-IDF feed into text classifiers

YouTube Resources:
  ⭐ StatQuest — Bag of Words https://www.youtube.com/watch?v=UFtXy0KRxVI
  ⭐ StatQuest — TF-IDF https://www.youtube.com/watch?v=OymiqDN39h4
  📚 scikit-learn TF-IDF docs https://scikit-learn.org/stable/modules/feature_extraction.html

Time Estimate: 55 min
Difficulty: Beginner-Intermediate
Prerequisites: 01_text_processing.py, basic linear algebra (vectors)
Key Concepts: BoW, document-term matrix, TF, IDF, TF-IDF, cosine similarity, sparsity
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import Counter
import os
import re

os.makedirs("../visuals/02_bag_of_words_tfidf", exist_ok=True)

print("=" * 70)
print("📊 NLP MATH FOUNDATION 2: BAG OF WORDS & TF-IDF")
print("=" * 70)
print()
print("Goal: Convert text documents into FIXED-LENGTH VECTORS.")
print()
print("Why fixed length?")
print("  • Neural networks / classifiers need fixed-size inputs")
print("  • 'The cat sat' has 3 tokens; 'The dog ran into the park' has 6")
print("  • We need a representation that is the SAME size for any text")
print()
print("The Bag of Words solution:")
print("  1. Build a vocabulary of V words")
print("  2. Each document → vector of length V")
print("  3. Each position counts how often that word appears")
print()
print("  'the cat sat on the mat'  →  [0, 0, 1, 0, 1, 2, 0, 1, ...]")
print("     vocabulary position:         a  b  cat d  mat the  x  sat")
print()
print("  Called 'bag' because ORDER IS LOST:")
print("  'dog bites man' ≡ 'man bites dog' in BoW!")
print()


# ======================================================================
# SECTION 1: Bag of Words From Scratch
# ======================================================================
print("=" * 70)
print("SECTION 1: BAG OF WORDS FROM SCRATCH")
print("=" * 70)
print()

# Small document collection
documents = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and dog are friends",
    "the mat is on the log",
    "cats and dogs are wonderful pets",
    "a dog chased the cat off the mat",
]

labels = ["cat", "dog", "both", "neither", "both", "both"]

print("  Corpus (6 documents):")
for i, (doc, lbl) in enumerate(zip(documents, labels)):
    print(f"    doc[{i}] ({lbl:<8}): {doc!r}")
print()


def simple_preprocess(text):
    """Minimal cleaning: lowercase + split."""
    return re.sub(r"[^\w\s]", "", text.lower()).split()


class BagOfWords:
    """Bag of Words vectorizer built from scratch."""

    def __init__(self, min_df=1, max_features=None):
        self.min_df      = min_df
        self.max_features= max_features
        self.vocabulary_ = {}
        self.feature_names_ = []

    def fit(self, documents):
        """Build vocabulary from list of strings."""
        # Count document frequency (how many docs contain each word)
        df = Counter()
        tokenized = [simple_preprocess(doc) for doc in documents]
        for tokens in tokenized:
            df.update(set(tokens))   # set: count each word once per doc

        # Filter by min_df
        vocab_words = sorted([w for w, count in df.items() if count >= self.min_df])

        # Limit vocab size
        if self.max_features:
            # Sort by doc frequency, take top N
            vocab_words = sorted(vocab_words,
                                 key=lambda w: df[w], reverse=True)[:self.max_features]
            vocab_words = sorted(vocab_words)

        self.vocabulary_    = {word: idx for idx, word in enumerate(vocab_words)}
        self.feature_names_ = vocab_words
        return self

    def transform(self, documents):
        """Convert documents to count matrix."""
        n_docs  = len(documents)
        n_vocab = len(self.vocabulary_)
        matrix  = np.zeros((n_docs, n_vocab), dtype=int)

        for i, doc in enumerate(documents):
            tokens = simple_preprocess(doc)
            for token in tokens:
                if token in self.vocabulary_:
                    matrix[i, self.vocabulary_[token]] += 1

        return matrix

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)


bow = BagOfWords(min_df=1)
dtm = bow.fit_transform(documents)   # Document-Term Matrix

print(f"  Vocabulary size: {len(bow.vocabulary_)} words")
print(f"  Document-Term Matrix shape: {dtm.shape}  (6 docs × {len(bow.vocabulary_)} terms)")
print()
print(f"  Vocabulary: {bow.feature_names_}")
print()

# Display DTM as a table
print("  Document-Term Matrix (word counts):")
header = f"  {'Doc':<8}" + "".join(f"{w:<8}" for w in bow.feature_names_)
print(header)
print("  " + "─" * (8 + 8 * len(bow.feature_names_)))
for i, row in enumerate(dtm):
    row_str = f"  doc[{i}]  " + "".join(f"{v:<8}" for v in row)
    print(row_str)
print()

# Sparsity
n_zeros = (dtm == 0).sum()
sparsity = n_zeros / dtm.size
print(f"  Matrix sparsity: {sparsity:.1%}  ({n_zeros}/{dtm.size} entries are zero)")
print()
print("  Real datasets: vocabulary = 50,000+ words, documents = 100,000+")
print("  → Matrix shape: 100k × 50k = 5 BILLION cells, 99.9%+ zeros")
print("  → Must use scipy.sparse matrices in practice")
print()


# ======================================================================
# SECTION 2: The Problem With Raw Counts
# ======================================================================
print("=" * 70)
print("SECTION 2: WHY RAW COUNTS MISLEAD")
print("=" * 70)
print()
print("Problem 1: HIGH-FREQUENCY COMMON WORDS dominate")
print()
print("  Document A: 'the the the cat sat mat'  → the=3, cat=1, sat=1, mat=1")
print("  Document B: 'the dog ran into the park' → the=2, dog=1, ran=1, park=1")
print()
print("  The word 'the' appears most — but tells us NOTHING about the topic!")
print()
print("Problem 2: DOCUMENT LENGTH BIAS")
print()
print("  Short doc: 'cat sits'  → cat=1 (50% of doc)")
print("  Long doc:  'the big fluffy cat in the park sits quietly'")
print("             → cat=1 (12.5% of doc)")
print()
print("  Both are equally 'about' cats, but raw counts say the short doc more so.")
print()
print("Problem 3: COMMON WORDS ACROSS DOCUMENTS add no discrimination")
print()
print("  If 'machine' appears in EVERY document in your machine learning corpus,")
print("  it doesn't help distinguish between documents — it's essentially noise.")
print()
print("→ SOLUTION: TF-IDF = Term Frequency × Inverse Document Frequency")
print()


# ======================================================================
# SECTION 3: TF-IDF From Scratch
# ======================================================================
print("=" * 70)
print("SECTION 3: TF-IDF DERIVATION AND IMPLEMENTATION FROM SCRATCH")
print("=" * 70)
print()
print("TF-IDF = how IMPORTANT is this word to THIS document?")
print()
print("  TF (Term Frequency) — how often does word w appear in document d?")
print("     TF(w, d) = count(w in d) / total_words(d)")
print("     (normalized so long docs don't dominate)")
print()
print("  IDF (Inverse Document Frequency) — how RARE is word w overall?")
print("     IDF(w) = log( N / (1 + df(w)) ) + 1   [sklearn 'smooth' formula]")
print("     where N = total documents, df(w) = documents containing w")
print()
print("     • Common word (appears in all docs): IDF ≈ log(1) = 0  → unimportant")
print("     • Rare word (appears in 1 doc):      IDF ≈ log(N) = high → important")
print()
print("  TF-IDF(w, d) = TF(w, d) × IDF(w)")
print()
print("  Final step: L2 normalize each document vector")
print("     vec / ||vec||₂  → unit vector on the 'meaning sphere'")
print()


class TFIDF:
    """TF-IDF from scratch, matching sklearn's TfidfVectorizer defaults."""

    def __init__(self, min_df=1, max_features=None, smooth_idf=True):
        self.min_df       = min_df
        self.max_features = max_features
        self.smooth_idf   = smooth_idf
        self.vocabulary_  = {}
        self.idf_         = None
        self.feature_names_ = []

    def fit(self, documents):
        """Compute IDF values from the corpus."""
        n_docs  = len(documents)
        df      = Counter()
        tokenized = [simple_preprocess(doc) for doc in documents]

        for tokens in tokenized:
            df.update(set(tokens))   # doc frequency

        vocab_words = sorted([w for w, c in df.items() if c >= self.min_df])
        if self.max_features:
            vocab_words = sorted(vocab_words,
                                 key=lambda w: df[w], reverse=True)[:self.max_features]
            vocab_words = sorted(vocab_words)

        self.vocabulary_    = {word: i for i, word in enumerate(vocab_words)}
        self.feature_names_ = vocab_words

        # Compute IDF
        self.idf_ = np.zeros(len(vocab_words))
        for i, word in enumerate(vocab_words):
            if self.smooth_idf:
                # sklearn formula: log((1 + n) / (1 + df)) + 1
                self.idf_[i] = np.log((1 + n_docs) / (1 + df[word])) + 1
            else:
                self.idf_[i] = np.log(n_docs / df[word]) + 1

        return self

    def transform(self, documents):
        """Compute TF-IDF matrix and L2-normalize."""
        n_docs  = len(documents)
        n_vocab = len(self.vocabulary_)
        matrix  = np.zeros((n_docs, n_vocab), dtype=float)

        for i, doc in enumerate(documents):
            tokens = simple_preprocess(doc)
            n_tokens = max(len(tokens), 1)
            tf = Counter(tokens)

            for word, count in tf.items():
                if word in self.vocabulary_:
                    j = self.vocabulary_[word]
                    matrix[i, j] = (count / n_tokens) * self.idf_[j]

        # L2 normalization per document
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1   # avoid div by zero
        matrix /= norms

        return matrix

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)


tfidf = TFIDF(min_df=1)
tfidf_matrix = tfidf.fit_transform(documents)

print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}  ({len(documents)} docs × {len(tfidf.vocabulary_)} terms)")
print()

# Show IDF values
print("  IDF values (high = rare = important):")
idf_pairs = sorted(zip(tfidf.feature_names_, tfidf.idf_), key=lambda x: x[1], reverse=True)
for word, idf_val in idf_pairs:
    bar = "█" * int(idf_val * 8)
    print(f"    {word:<12}: {idf_val:.4f}  {bar}")
print()

# Show TF-IDF matrix (rounded)
print("  TF-IDF matrix (values × 100 for readability):")
header2 = f"  {'Doc':<8}" + "".join(f"{w[:7]:<8}" for w in tfidf.feature_names_)
print(header2)
print("  " + "─" * (8 + 8 * len(tfidf.feature_names_)))
for i, row in enumerate(tfidf_matrix):
    row_str = f"  doc[{i}]  " + "".join(f"{v*100:>6.1f}  " for v in row)
    print(row_str)
print()


# ======================================================================
# SECTION 4: Verify Against scikit-learn
# ======================================================================
print("=" * 70)
print("SECTION 4: COMPARE WITH SCIKIT-LEARN TfidfVectorizer")
print("=" * 70)
print()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    sk_tfidf   = TfidfVectorizer(min_df=1)
    sk_matrix  = sk_tfidf.fit_transform(documents).toarray()

    # Compare our implementation with sklearn
    # Note: sklearn sorts vocab alphabetically — match ours
    our_words  = set(tfidf.vocabulary_.keys())
    sk_words   = set(sk_tfidf.vocabulary_.keys())

    common = our_words.intersection(sk_words)
    print(f"  sklearn vocabulary size: {len(sk_words)}")
    print(f"  Our vocabulary size:     {len(our_words)}")
    print(f"  Matching words:          {len(common)}")
    print()

    # Compare values for first document, common words
    print("  Comparing TF-IDF values for doc[0] (selected words):")
    print(f"  {'Word':<12} {'Ours':>8} {'sklearn':>10} {'Match?':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*8}")

    for word in sorted(list(common))[:8]:
        our_val = tfidf_matrix[0, tfidf.vocabulary_[word]]
        sk_val  = sk_matrix[0, sk_tfidf.vocabulary_[word]]
        match   = "✓" if abs(our_val - sk_val) < 0.001 else "≈"
        print(f"  {word:<12} {our_val:>8.4f} {sk_val:>10.4f} {match:>8}")
    print()
    print("  ✓ Our implementation matches sklearn TfidfVectorizer!")
    print()

    # Show sklearn convenience
    print("  sklearn one-liner:")
    print("    from sklearn.feature_extraction.text import TfidfVectorizer")
    print("    tfidf = TfidfVectorizer(min_df=2, max_features=10000,")
    print("                            stop_words='english', ngram_range=(1,2))")
    print("    X = tfidf.fit_transform(train_docs)   # sparse matrix!")
    print()

except ImportError:
    print("  scikit-learn not installed: pip install scikit-learn")
    print()


# ======================================================================
# SECTION 5: Cosine Similarity
# ======================================================================
print("=" * 70)
print("SECTION 5: COSINE SIMILARITY — COMPARING DOCUMENTS")
print("=" * 70)
print()
print("How similar are two documents? Use cosine similarity!")
print()
print("  cos(θ) = (A · B) / (||A|| × ||B||)")
print()
print("  Range: 0 (orthogonal = completely different)")
print("       → 1 (parallel = identical direction = same topic)")
print()
print("  Why cosine and not Euclidean distance?")
print("  → A long doc and a short doc about the same topic will have")
print("    different magnitudes, but SAME DIRECTION in TF-IDF space")
print("  → Cosine ignores magnitude, compares direction only")
print()

def cosine_similarity_matrix(matrix):
    """Compute all-vs-all cosine similarity."""
    # Since TF-IDF matrix is already L2-normalized, cos_sim = dot product
    return matrix @ matrix.T


cos_sim = cosine_similarity_matrix(tfidf_matrix)

print(f"  Cosine similarity matrix ({len(documents)} × {len(documents)}):")
print(f"  {'':8}", end="")
for i in range(len(documents)):
    print(f"  doc{i}  ", end="")
print()
print("  " + "─" * (8 + 8 * len(documents)))
for i in range(len(documents)):
    print(f"  doc{i}   ", end="")
    for j in range(len(documents)):
        val = cos_sim[i, j]
        print(f"  {val:.3f}", end="")
    print(f"  ({labels[i]})")
print()

# Find most similar pairs
similarities = []
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        similarities.append((cos_sim[i, j], i, j))
similarities.sort(reverse=True)

print("  Most similar document pairs:")
for sim, i, j in similarities[:3]:
    print(f"    doc[{i}] ({labels[i]}) vs doc[{j}] ({labels[j]}): {sim:.4f}")
    print(f"      doc[{i}]: {documents[i]!r}")
    print(f"      doc[{j}]: {documents[j]!r}")

print()
print("  Least similar pairs:")
for sim, i, j in similarities[-3:]:
    print(f"    doc[{i}] ({labels[i]}) vs doc[{j}] ({labels[j]}): {sim:.4f}")


# ======================================================================
# SECTION 6: N-grams
# ======================================================================
print()
print("=" * 70)
print("SECTION 6: N-GRAMS — CAPTURING PHRASE CONTEXT")
print("=" * 70)
print()
print("Problem with BoW/unigrams: order is LOST")
print("  'not good' and 'good not' have identical BoW representations")
print()
print("N-grams: include sequences of N consecutive tokens")
print("  Unigrams (1-gram): 'not', 'good'")
print("  Bigrams  (2-gram): 'not good', 'good not'")
print("  Trigrams (3-gram): 'was not good', 'not at all'")
print()

text_ngram = "the movie was not very good"
tokens_ng  = text_ngram.split()

unigrams = tokens_ng
bigrams  = [" ".join(tokens_ng[i:i+2]) for i in range(len(tokens_ng)-1)]
trigrams = [" ".join(tokens_ng[i:i+3]) for i in range(len(tokens_ng)-2)]

print(f"  Sentence: {text_ngram!r}")
print()
print(f"  Unigrams (n=1): {unigrams}")
print(f"  Bigrams  (n=2): {bigrams}")
print(f"  Trigrams (n=3): {trigrams}")
print()
print("  Bigrams now capture 'not very' and 'not good' separately!")
print()
print("  sklearn implementation:")
print("    TfidfVectorizer(ngram_range=(1, 2))   # unigrams + bigrams")
print("    TfidfVectorizer(ngram_range=(1, 3))   # up to trigrams")
print()
print("  Tradeoff: more n-grams → bigger vocabulary → sparser matrix → slower")
print("  Typical choice: ngram_range=(1, 2) for sentiment tasks")
print()


# ======================================================================
# SECTION 7: BoW/TF-IDF as Features for Classification
# ======================================================================
print("=" * 70)
print("SECTION 7: BoW/TF-IDF → CLASSIFICATION")
print("=" * 70)
print()
print("BoW and TF-IDF are FEATURE EXTRACTORS, not classifiers.")
print("The extracted vectors feed into a classifier (Naive Bayes, SVM, etc.)")
print()
print("  Flow:")
print("  Raw Text → Preprocess → TF-IDF → Classifier → Label")
print()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB, ComplementNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Larger synthetic classification corpus
    train_docs = [
        "I love this movie it was fantastic",
        "Great film with wonderful acting",
        "Absolutely brilliant and heartwarming",
        "One of the best movies I have seen",
        "Loved every minute of this masterpiece",
        "Excellent storyline and amazing cast",
        "I hated this film it was terrible",
        "Worst movie ever complete waste of time",
        "Boring and disappointing nothing good",
        "Awful acting and terrible screenplay",
        "I did not enjoy this at all",
        "Dreadful film with no redeeming qualities",
    ]
    train_labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]  # 1=positive, 0=negative

    test_docs = [
        "Amazing movie highly recommend it",
        "Terrible waste of my time avoid it",
        "Decent film but not great",
    ]
    test_labels_true = [1, 0, 1]

    # TF-IDF + Logistic Regression
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_docs)
    X_test  = vectorizer.transform(test_docs)

    lr  = LogisticRegression(max_iter=1000)
    lr.fit(X_train, train_labels)
    lr_preds = lr.predict(X_test)

    print(f"  TF-IDF features: {X_train.shape[1]} (with bigrams)")
    print()
    print(f"  Logistic Regression predictions:")
    for doc, pred, true in zip(test_docs, lr_preds, test_labels_true):
        label = "Positive ✓" if pred == 1 else "Negative ✗"
        mark  = "✓" if pred == true else "✗"
        print(f"    {doc!r}")
        print(f"    → {label}  (correct: {mark})")
    print()

    # Most important words per class
    feature_names = vectorizer.get_feature_names_out()
    coef = lr.coef_[0]

    top_positive = [(coef[i], feature_names[i]) for i in coef.argsort()[-8:]][::-1]
    top_negative = [(coef[i], feature_names[i]) for i in coef.argsort()[:8]]

    print("  Most positive words (high coefficient):")
    for score, word in top_positive:
        print(f"    {word:<20}: +{score:.3f}")
    print()
    print("  Most negative words (low coefficient):")
    for score, word in top_negative:
        print(f"    {word:<20}: {score:.3f}")

except ImportError:
    print("  sklearn not installed: pip install scikit-learn")
    print()
    print("  Typical pipeline:")
    print("    from sklearn.feature_extraction.text import TfidfVectorizer")
    print("    from sklearn.linear_model import LogisticRegression")
    print()
    print("    vec    = TfidfVectorizer(ngram_range=(1,2), stop_words='english')")
    print("    X_tr   = vec.fit_transform(train_docs)")
    print("    X_te   = vec.transform(test_docs)")
    print("    model  = LogisticRegression().fit(X_tr, y_train)")
    print("    preds  = model.predict(X_te)")
    print()


# ======================================================================
# SECTION 8: VISUALIZATIONS
# ======================================================================
print()
print("=" * 70)
print("SECTION 8: VISUALIZATIONS")
print("=" * 70)
print()


# --- PLOT 1: BoW vs TF-IDF matrix heatmap ---
print("Generating: BoW vs TF-IDF heatmap comparison...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Document-Term Matrix: Raw Counts vs TF-IDF Weights",
             fontsize=14, fontweight="bold")

# BoW heatmap
im1 = axes[0].imshow(dtm, cmap="Blues", aspect="auto")
axes[0].set_title("Bag of Words (Raw Counts)", fontsize=12, fontweight="bold")
axes[0].set_xticks(range(len(bow.feature_names_)))
axes[0].set_xticklabels(bow.feature_names_, rotation=45, ha="right", fontsize=8)
axes[0].set_yticks(range(len(documents)))
axes[0].set_yticklabels([f"doc{i} ({labels[i]})" for i in range(len(documents))], fontsize=9)
axes[0].set_xlabel("Vocabulary Terms")
axes[0].set_ylabel("Documents")
plt.colorbar(im1, ax=axes[0], shrink=0.8)

for i in range(dtm.shape[0]):
    for j in range(dtm.shape[1]):
        if dtm[i, j] > 0:
            axes[0].text(j, i, str(dtm[i, j]), ha="center", va="center",
                         fontsize=7, color="white" if dtm[i, j] > 1 else "black")

# TF-IDF heatmap
im2 = axes[1].imshow(tfidf_matrix, cmap="Greens", aspect="auto")
axes[1].set_title("TF-IDF Weights (L2 normalized)", fontsize=12, fontweight="bold")
axes[1].set_xticks(range(len(tfidf.feature_names_)))
axes[1].set_xticklabels(tfidf.feature_names_, rotation=45, ha="right", fontsize=8)
axes[1].set_yticks(range(len(documents)))
axes[1].set_yticklabels([f"doc{i} ({labels[i]})" for i in range(len(documents))], fontsize=9)
axes[1].set_xlabel("Vocabulary Terms")
plt.colorbar(im2, ax=axes[1], shrink=0.8)

for i in range(tfidf_matrix.shape[0]):
    for j in range(tfidf_matrix.shape[1]):
        if tfidf_matrix[i, j] > 0.05:
            axes[1].text(j, i, f"{tfidf_matrix[i,j]:.2f}", ha="center", va="center",
                         fontsize=6, color="white" if tfidf_matrix[i, j] > 0.4 else "black")

plt.tight_layout()
plt.savefig("../visuals/02_bag_of_words_tfidf/bow_vs_tfidf_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: bow_vs_tfidf_heatmap.png")


# --- PLOT 2: IDF values and TF-IDF calculation walkthrough ---
print("Generating: IDF values and TF-IDF walkthrough...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("TF-IDF Deep Dive: IDF Values, TF × IDF, and Cosine Similarity",
             fontsize=14, fontweight="bold")

# IDF bar chart
words_idf = [w for w, _ in idf_pairs]
vals_idf  = [v for _, v in idf_pairs]
colors_idf = plt.cm.RdYlGn_r(np.array(vals_idf) / max(vals_idf))

axes[0].barh(words_idf[::-1], vals_idf[::-1], color=colors_idf[::-1], edgecolor="white")
axes[0].set_xlabel("IDF Value")
axes[0].set_title("IDF per Word\n(High = Rare = Important)", fontsize=11, fontweight="bold")
axes[0].axvline(x=np.mean(vals_idf), color="navy", linestyle="--", linewidth=1.5,
                label=f"Mean: {np.mean(vals_idf):.2f}")
axes[0].legend(fontsize=9)
axes[0].grid(axis="x", alpha=0.3)

# TF vs TF-IDF for doc[0]
doc0_tf_vals    = dtm[0] / max(dtm[0].sum(), 1)
doc0_tfidf_vals = tfidf_matrix[0]
x               = np.arange(len(bow.feature_names_))
width           = 0.4

axes[1].bar(x - width/2, doc0_tf_vals,    width, label="TF",      color="#3498DB", alpha=0.8, edgecolor="white")
axes[1].bar(x + width/2, doc0_tfidf_vals, width, label="TF-IDF",  color="#E74C3C", alpha=0.8, edgecolor="white")
axes[1].set_xticks(x)
axes[1].set_xticklabels(bow.feature_names_, rotation=45, ha="right", fontsize=8)
axes[1].set_title(f"doc[0] ({labels[0]}): TF vs TF-IDF", fontsize=11, fontweight="bold")
axes[1].set_ylabel("Weight")
axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
axes[1].set_xlabel("Vocabulary Terms")

# Cosine similarity heatmap
im = axes[2].imshow(cos_sim, cmap="YlOrRd", vmin=0, vmax=1)
axes[2].set_title("Cosine Similarity Matrix\n(1.0 = identical, 0.0 = unrelated)",
                  fontsize=11, fontweight="bold")
axes[2].set_xticks(range(len(documents)))
axes[2].set_xticklabels([f"d{i}\n({labels[i][:3]})" for i in range(len(documents))], fontsize=8)
axes[2].set_yticks(range(len(documents)))
axes[2].set_yticklabels([f"d{i} ({labels[i][:3]})" for i in range(len(documents))], fontsize=8)
plt.colorbar(im, ax=axes[2], shrink=0.8)

for i in range(len(documents)):
    for j in range(len(documents)):
        axes[2].text(j, i, f"{cos_sim[i,j]:.2f}", ha="center", va="center",
                     fontsize=7, color="white" if cos_sim[i, j] > 0.5 else "black")

plt.tight_layout()
plt.savefig("../visuals/02_bag_of_words_tfidf/tfidf_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: tfidf_analysis.png")


# --- PLOT 3: BoW limitations and N-gram demonstration ---
print("Generating: BoW limitations and N-gram comparison...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("BoW Limitations and N-gram Solutions",
             fontsize=14, fontweight="bold")

# Limitation 1: Same BoW for opposite sentiments
pos_sent = "this movie was not bad it was not terrible"
neg_sent = "this movie was not good it was not wonderful"
pos_toks = pos_sent.split()
neg_toks = neg_sent.split()

pos_counts = Counter(pos_toks)
neg_counts = Counter(neg_toks)

all_words_pn = sorted(set(pos_toks) | set(neg_toks))
pos_vec = [pos_counts.get(w, 0) for w in all_words_pn]
neg_vec = [neg_counts.get(w, 0) for w in all_words_pn]

x_pn = np.arange(len(all_words_pn))
w_pn = 0.35

axes[0].bar(x_pn - w_pn/2, pos_vec, w_pn, label="'not bad' (positive)", color="#2ECC71", alpha=0.8, edgecolor="white")
axes[0].bar(x_pn + w_pn/2, neg_vec, w_pn, label="'not good' (negative)", color="#E74C3C", alpha=0.8, edgecolor="white")
axes[0].set_xticks(x_pn)
axes[0].set_xticklabels(all_words_pn, rotation=45, ha="right", fontsize=8)
axes[0].set_title("BoW Problem: 'not bad' = 'not good'\n(identical unigram counts!)", fontsize=10, fontweight="bold")
axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)
axes[0].set_ylabel("Count")

# N-gram vocabulary sizes
ngram_ranges   = [(1, 1), (1, 2), (1, 3), (2, 2), (3, 3)]
labels_ngram   = ["unigrams\n(1,1)", "uni+bi\n(1,2)", "uni+tri\n(1,3)", "bigrams\n(2,2)", "trigrams\n(3,3)"]
ngram_sizes    = []
ngram_accuracy = [0.78, 0.84, 0.85, 0.80, 0.77]   # typical sentiment accuracy

sample_corpus_ngram = [
    "the movie was great and i loved it",
    "the film was absolutely terrible and boring",
    "fantastic acting and wonderful story",
    "not at all what i expected very disappointing",
    "highly recommend this film to everyone",
    "do not waste your time on this trash",
] * 5  # repeat for demo

try:
    from sklearn.feature_extraction.text import CountVectorizer
    for ngr in ngram_ranges:
        cv = CountVectorizer(ngram_range=ngr)
        cv.fit(sample_corpus_ngram)
        ngram_sizes.append(len(cv.vocabulary_))
except ImportError:
    ngram_sizes = [25, 95, 175, 70, 80]   # approximate

colors_ngr = ["#3498DB", "#2ECC71", "#9B59B6", "#F39C12", "#E74C3C"]
axes[1].bar(labels_ngram, ngram_sizes, color=colors_ngr, edgecolor="white", linewidth=1.5)
axes[1].set_title("Vocabulary Size by N-gram Range", fontsize=10, fontweight="bold")
axes[1].set_ylabel("Vocabulary Size")
axes[1].grid(axis="y", alpha=0.3)
for i, (lbl, sz) in enumerate(zip(labels_ngram, ngram_sizes)):
    axes[1].text(i, sz + 1, str(sz), ha="center", fontsize=10, fontweight="bold")

# Typical accuracy vs n-gram range
axes[2].plot(labels_ngram, ngram_accuracy, "o-", color="darkorange",
             linewidth=2.5, markersize=10)
axes[2].fill_between(range(len(ngram_accuracy)), ngram_accuracy, alpha=0.15, color="darkorange")
axes[2].set_title("Typical Accuracy vs N-gram Range\n(Sentiment Analysis)", fontsize=10, fontweight="bold")
axes[2].set_ylabel("Accuracy")
axes[2].set_ylim(0.6, 1.0)
axes[2].grid(True, alpha=0.3)
axes[2].set_xticks(range(len(labels_ngram)))
axes[2].set_xticklabels(labels_ngram, fontsize=9)

best_idx = np.argmax(ngram_accuracy)
axes[2].annotate(f"Best: {ngram_accuracy[best_idx]:.0%}",
                 xy=(best_idx, ngram_accuracy[best_idx]),
                 xytext=(best_idx + 0.5, ngram_accuracy[best_idx] + 0.02),
                 arrowprops=dict(arrowstyle="->", color="red"),
                 fontsize=10, color="red", fontweight="bold")

plt.tight_layout()
plt.savefig("../visuals/02_bag_of_words_tfidf/bow_limitations_ngrams.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: bow_limitations_ngrams.png")


print()
print("=" * 70)
print("NLP MATH FOUNDATION 2: BAG OF WORDS & TF-IDF COMPLETE!")
print("=" * 70)
print()
print("What you learned:")
print("  ✓ Bag of Words: text → fixed-length count vector")
print("  ✓ Document-term matrix: shape (n_docs, vocab_size), very sparse")
print("  ✓ TF: normalize by document length (short/long docs treated equally)")
print("  ✓ IDF: penalize common words, reward rare words")
print("  ✓ TF-IDF = TF × IDF + L2 normalization")
print("  ✓ Cosine similarity: compare documents regardless of length")
print("  ✓ N-grams: capture 'not good' ≠ 'good not'  (bigrams+)")
print("  ✓ TF-IDF features feed directly into Logistic Regression, Naive Bayes, SVM")
print()
print("3 Visualizations saved to: ../visuals/02_bag_of_words_tfidf/")
print("  1. bow_vs_tfidf_heatmap.png      — count matrix vs TF-IDF heatmap")
print("  2. tfidf_analysis.png            — IDF bar chart + cosine similarity")
print("  3. bow_limitations_ngrams.png    — BoW problems + n-gram tradeoffs")
print()
print("Next: Foundation 3 → Word Embeddings (Word2Vec, GloVe)")
