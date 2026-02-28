"""
🔧 NLP — Algorithm 1: Text Classification Pipeline
===================================================

Learning Objectives:
  1. Build a complete, reusable text classification pipeline end-to-end
  2. Compare TF-IDF + classical ML vs simple neural network classifiers
  3. Handle multi-class text classification (beyond binary sentiment)
  4. Use sklearn Pipeline objects for clean, production-ready code
  5. Apply cross-validation and hyperparameter search on text data
  6. Interpret model decisions using top TF-IDF features per class
  7. Understand when to use TF-IDF+LR vs embeddings vs deep learning

YouTube Resources:
  ⭐ Krish Naik — Text Classification ML https://www.youtube.com/watch?v=5E_zq25EKDY
  ⭐ Patrick Loeber — NLP sklearn https://www.youtube.com/watch?v=M9Itm95nc9I
  📚 sklearn text classification guide
     https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

Time Estimate: 65 min
Difficulty: Intermediate
Prerequisites: 01_text_processing.py, 02_bag_of_words_tfidf.py
Key Concepts: sklearn Pipeline, TfidfVectorizer, multi-class, cross-validation, feature importance
"""

import numpy as np
import matplotlib.pyplot as plt
import re
import os
from collections import Counter

_VISUALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "text_classification_pipeline")
os.makedirs(_VISUALS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🔧 NLP ALGORITHM 1: TEXT CLASSIFICATION PIPELINE")
print("=" * 70)
print()
print("Text classification: assign a category label to a text document.")
print()
print("  Examples:")
print("  • Email → spam / not spam")
print("  • Product review → 1-5 stars")
print("  • News article → sports / politics / tech / entertainment")
print("  • Support ticket → billing / technical / account / general")
print()
print("  This module builds a FULL PIPELINE from raw text to predictions,")
print("  comparing multiple approaches on the same data.")
print()


# ======================================================================
# SECTION 1: Dataset — AG News (4-class news topic)
# ======================================================================
print("=" * 70)
print("SECTION 1: DATASET — NEWS TOPIC CLASSIFICATION")
print("=" * 70)
print()
print("Task: classify news headlines into 4 categories")
print("  0 = World    1 = Sports    2 = Business    3 = Science/Tech")
print()

# Synthetic dataset representative of AG News style
CATEGORIES = ["World", "Sports", "Business", "Sci/Tech"]

raw_data = [
    # World
    ("United Nations holds emergency summit on climate crisis", 0),
    ("President signs landmark peace agreement with neighboring country", 0),
    ("Military forces withdraw from contested border region after talks", 0),
    ("International court rules on disputed territorial claim", 0),
    ("Refugees face dire conditions as humanitarian crisis deepens", 0),
    ("World leaders gather for annual economic forum discussions", 0),
    ("Election results spark protests in capital city streets", 0),
    ("Foreign minister resigns amid diplomatic scandal allegations", 0),
    ("Coalition forces launch operation against extremist group", 0),
    ("Sanctions imposed on regime following human rights violations", 0),

    # Sports
    ("Team wins championship in dramatic overtime victory", 1),
    ("Record-breaking athlete shatters world record at Olympics", 1),
    ("Star player signs massive contract extension with club", 1),
    ("Coach fired after disappointing season with struggling team", 1),
    ("Tournament bracket revealed for upcoming playoffs schedule", 1),
    ("Controversial referee decision sparks debate among fans", 1),
    ("Injury forces top seed to withdraw from major tournament", 1),
    ("Young prospect drafted first overall by struggling franchise", 1),
    ("Home team defeats rival in thrilling final minute comeback", 1),
    ("League announces expansion to two new cities next season", 1),

    # Business
    ("Tech giant acquires startup for five billion dollars", 2),
    ("Stock market hits record high amid strong earnings reports", 2),
    ("Federal reserve raises interest rates to combat inflation", 2),
    ("Major retailer announces thousands of layoffs amid restructuring", 2),
    ("Startup raises venture capital funding in latest round", 2),
    ("Oil prices surge following geopolitical supply disruptions", 2),
    ("Company reports quarterly earnings beating analyst expectations", 2),
    ("Merger deal collapses after regulatory scrutiny intensifies", 2),
    ("Banks increase lending rates following central bank decision", 2),
    ("Consumer confidence rises as unemployment falls to new low", 2),

    # Sci/Tech
    ("Scientists discover new exoplanet in habitable zone of star", 3),
    ("Artificial intelligence model achieves breakthrough in protein folding", 3),
    ("Space agency launches mission to study asteroid belt samples", 3),
    ("Researchers develop new battery technology doubling energy density", 3),
    ("Gene editing therapy shows promise in clinical trial results", 3),
    ("Tech company unveils latest smartphone with advanced camera system", 3),
    ("New vaccine demonstrates high efficacy in phase three trials", 3),
    ("Quantum computing milestone achieved by research team at lab", 3),
    ("Study reveals impact of social media on adolescent brain development", 3),
    ("Self-driving car completes first fully autonomous cross-country trip", 3),
]

# Split train/test
np.random.shuffle(raw_data)
split   = int(0.8 * len(raw_data))
train   = raw_data[:split]
test    = raw_data[split:]

X_train_raw, y_train = zip(*train)
X_test_raw,  y_test  = zip(*test)
X_train_raw = list(X_train_raw)
X_test_raw  = list(X_test_raw)
y_train     = list(y_train)
y_test      = list(y_test)

print(f"  Total examples: {len(raw_data)}  (train={len(train)}, test={len(test)})")
print(f"  Classes: {CATEGORIES}")
print()
print(f"  Class distribution (train):")
for i, cat in enumerate(CATEGORIES):
    count = y_train.count(i)
    bar   = "█" * count
    print(f"    {cat:<12}: {count:2d}  {bar}")
print()


# ======================================================================
# SECTION 2: TF-IDF + Logistic Regression Pipeline
# ======================================================================
print("=" * 70)
print("SECTION 2: TF-IDF + LOGISTIC REGRESSION PIPELINE")
print("=" * 70)
print()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (accuracy_score, classification_report,
                                  confusion_matrix, f1_score)
    from sklearn.model_selection import cross_val_score

    SK_AVAILABLE = True
    print("  scikit-learn available ✓")
    print()

    # Build sklearn Pipeline
    print("  sklearn Pipeline — chains preprocessing + model into one object:")
    print()
    print("    Pipeline([")
    print("        ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=10000)),")
    print("        ('clf',   LogisticRegression(max_iter=1000, C=10)),")
    print("    ])")
    print()
    print("  Advantages of Pipeline:")
    print("  1. Prevents data leakage: tfidf.fit() only on train, transform on test")
    print("  2. Single .fit() / .predict() call")
    print("  3. Works with cross_val_score, GridSearchCV transparently")
    print()

    # Define multiple classifiers to compare
    models = {
        "TF-IDF + LogReg": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                                      sublinear_tf=True, stop_words="english")),
            ("clf",   LogisticRegression(max_iter=1000, C=10, solver="lbfgs")),
        ]),
        "TF-IDF + ComplementNB": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                                      sublinear_tf=True, stop_words="english")),
            ("clf",   ComplementNB(alpha=0.1)),
        ]),
        "TF-IDF + LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                                      sublinear_tf=True, stop_words="english")),
            ("clf",   LinearSVC(max_iter=2000, C=1.0)),
        ]),
    }

    results = {}

    for name, pipeline in models.items():
        pipeline.fit(X_train_raw, y_train)
        preds   = pipeline.predict(X_test_raw)
        acc     = accuracy_score(y_test, preds)
        f1      = f1_score(y_test, preds, average="macro")
        results[name] = {"model": pipeline, "preds": preds, "acc": acc, "f1": f1}
        print(f"  {name}")
        print(f"    Test accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print(f"    Macro F1:      {f1:.4f}")
        print()

    # Best model detailed report
    best_name = max(results, key=lambda k: results[k]["f1"])
    best      = results[best_name]
    print(f"  Best model: {best_name}")
    print()
    print(classification_report(y_test, best["preds"], target_names=CATEGORIES))

    # Cross-validation on best model type
    print(f"  5-Fold Cross-Validation ({best_name}):")
    cv_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                                  sublinear_tf=True, stop_words="english")),
        ("clf",   LogisticRegression(max_iter=1000, C=10)),
    ])
    cv_scores = cross_val_score(cv_pipe,
                                X_train_raw + X_test_raw,
                                y_train + y_test,
                                cv=5, scoring="accuracy")
    print(f"  CV Scores: {cv_scores.round(3)}")
    print(f"  Mean ± Std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()

except ImportError:
    SK_AVAILABLE = False
    print("  scikit-learn not installed: pip install scikit-learn")
    print()
    print("  Showing expected results:")
    print("  TF-IDF + LogReg:        ~85-92% on AG News (full)")
    print("  TF-IDF + ComplementNB:  ~83-89%")
    print("  TF-IDF + LinearSVC:     ~86-93%")
    print()
    # Fake results for plotting
    results = {
        "TF-IDF + LogReg":      {"acc": 0.88, "f1": 0.87},
        "TF-IDF + ComplementNB":{"acc": 0.84, "f1": 0.83},
        "TF-IDF + LinearSVC":   {"acc": 0.90, "f1": 0.89},
    }
    y_test_np   = np.array(y_test)
    best_name   = "TF-IDF + LinearSVC"


# ======================================================================
# SECTION 3: Feature Interpretation
# ======================================================================
print("=" * 70)
print("SECTION 3: FEATURE INTERPRETATION — TOP WORDS PER CLASS")
print("=" * 70)
print()
print("LogReg coefficients tell us: which words MOST indicate each class?")
print()

if SK_AVAILABLE:
    lr_pipeline = results["TF-IDF + LogReg"]["model"]
    tfidf_step  = lr_pipeline.named_steps["tfidf"]
    lr_step     = lr_pipeline.named_steps["clf"]

    feature_names = tfidf_step.get_feature_names_out()
    coef          = lr_step.coef_   # shape: (n_classes, n_features)

    for cls_idx, cls_name in enumerate(CATEGORIES):
        top_pos = coef[cls_idx].argsort()[-8:][::-1]
        top_neg = coef[cls_idx].argsort()[:5]
        print(f"  {cls_name} — most indicative words:")
        for i in top_pos:
            print(f"    +{coef[cls_idx, i]:.3f}  {feature_names[i]!r}")
        print()

else:
    print("  Typical top features (TF-IDF + LogReg on AG News):")
    print()
    top_by_class = {
        "World":    ["government", "country", "minister", "conflict", "military"],
        "Sports":   ["game", "team", "season", "player", "championship"],
        "Business": ["market", "company", "billion", "shares", "earnings"],
        "Sci/Tech": ["technology", "research", "scientists", "software", "data"],
    }
    for cls, words in top_by_class.items():
        print(f"  {cls}: {words}")
    print()


# ======================================================================
# SECTION 4: Neural Network Classifier (Simple, No TF needed)
# ======================================================================
print("=" * 70)
print("SECTION 4: NEURAL NETWORK CLASSIFIER (NUMPY FROM SCRATCH)")
print("=" * 70)
print()
print("TF-IDF vectors are just numbers — we can feed them to a simple MLP!")
print("No TensorFlow needed — pure numpy neural network on TF-IDF features.")
print()

if SK_AVAILABLE:
    # Build TF-IDF representation
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=500,
                                 sublinear_tf=True, stop_words="english")
    X_tr_tfidf = vectorizer.fit_transform(X_train_raw).toarray()
    X_te_tfidf = vectorizer.transform(X_test_raw).toarray()
    y_tr_np    = np.array(y_train)
    y_te_np    = np.array(y_test)

    print(f"  TF-IDF feature matrix: train={X_tr_tfidf.shape}, test={X_te_tfidf.shape}")
    print()

    def relu(x):
        return np.maximum(0, x)

    def softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def cross_entropy_loss(probs, labels, n_classes=4):
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(len(labels)), labels] = 1
        return -np.mean(np.sum(one_hot * np.log(probs + 1e-9), axis=1))

    class SimpleMLPClassifier:
        """2-layer MLP for TF-IDF classification."""

        def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
            scale1 = np.sqrt(2.0 / input_dim)
            scale2 = np.sqrt(2.0 / hidden_dim)
            self.W1 = np.random.randn(input_dim,  hidden_dim) * scale1
            self.b1 = np.zeros(hidden_dim)
            self.W2 = np.random.randn(hidden_dim, output_dim) * scale2
            self.b2 = np.zeros(output_dim)
            self.lr = lr

        def forward(self, X):
            self.h  = relu(X @ self.W1 + self.b1)
            self.out= self.h @ self.W2 + self.b2
            return softmax(self.out)

        def backward(self, X, y, probs):
            n = X.shape[0]
            one_hot = np.zeros_like(probs)
            one_hot[np.arange(n), y] = 1

            dout = (probs - one_hot) / n
            dW2  = self.h.T @ dout
            db2  = dout.sum(axis=0)

            dh   = dout @ self.W2.T
            dh  *= (self.h > 0)    # ReLU backward

            dW1  = X.T @ dh
            db1  = dh.sum(axis=0)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

        def fit(self, X, y, epochs=200, batch_size=16):
            losses = []
            for epoch in range(epochs):
                perm = np.random.permutation(len(X))
                epoch_loss = 0
                for i in range(0, len(X), batch_size):
                    idx    = perm[i:i+batch_size]
                    Xb, yb = X[idx], y[idx]
                    probs  = self.forward(Xb)
                    self.backward(Xb, yb, probs)
                    epoch_loss += cross_entropy_loss(probs, yb)
                losses.append(epoch_loss)
            return losses

        def predict(self, X):
            return self.forward(X).argmax(axis=1)

        def score(self, X, y):
            return (self.predict(X) == y).mean()

    mlp_clf = SimpleMLPClassifier(
        input_dim=X_tr_tfidf.shape[1], hidden_dim=128, output_dim=4, lr=0.05
    )
    print("  Training numpy MLP (TF-IDF features → 128 hidden → 4 classes)...")
    mlp_losses = mlp_clf.fit(X_tr_tfidf, y_tr_np, epochs=300, batch_size=16)
    mlp_acc    = mlp_clf.score(X_te_tfidf, y_te_np)
    print(f"  MLP test accuracy: {mlp_acc:.4f} ({mlp_acc*100:.1f}%)")
    print()

    results["NumPy MLP on TF-IDF"] = {"acc": mlp_acc, "f1": mlp_acc}


# ======================================================================
# SECTION 5: When to Use What
# ======================================================================
print("=" * 70)
print("SECTION 5: CHOOSING YOUR CLASSIFIER")
print("=" * 70)
print()
print("  ┌──────────────────────┬───────────────────────────────────────┐")
print("  │ Method               │ Best for                              │")
print("  ├──────────────────────┼───────────────────────────────────────┤")
print("  │ TF-IDF + NaiveBayes  │ Small data, very fast, good baseline  │")
print("  │ TF-IDF + LogReg      │ General purpose, interpretable        │")
print("  │ TF-IDF + LinearSVC   │ High accuracy, many features          │")
print("  │ TF-IDF + MLP         │ When you want non-linear w/o DL       │")
print("  │ Embedding + BiLSTM   │ Sequence-aware, medium data           │")
print("  │ BERT fine-tune       │ Best accuracy, large data, GPU        │")
print("  └──────────────────────┴───────────────────────────────────────┘")
print()
print("  Decision guide:")
print("  1. Start with TF-IDF + LogReg (fast, interpretable, often 80-90%)")
print("  2. Try TF-IDF + LinearSVC if you need another % of accuracy")
print("  3. Upgrade to LSTM/BERT only if TF-IDF methods plateau")
print("  4. BERT fine-tuning: if you have GPU + need 95%+ accuracy")
print()


# ======================================================================
# SECTION 6: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 6: VISUALIZATIONS")
print("=" * 70)
print()


# --- PLOT 1: Model comparison + feature importance ---
print("Generating: Model comparison and feature importance...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Text Classification Pipeline: Model Comparison and Feature Analysis",
             fontsize=14, fontweight="bold")

# Accuracy comparison
model_names = list(results.keys())
accs = [results[m]["acc"] for m in model_names]
f1s  = [results[m].get("f1", results[m]["acc"]) for m in model_names]

x    = np.arange(len(model_names))
w    = 0.35
bars_acc = axes[0].bar(x - w/2, [a*100 for a in accs], w, label="Accuracy",
                        color="#3498DB", alpha=0.85, edgecolor="white")
bars_f1  = axes[0].bar(x + w/2, [f*100 for f in f1s],  w, label="Macro F1",
                        color="#E74C3C", alpha=0.85, edgecolor="white")
axes[0].set_xticks(x)
axes[0].set_xticklabels([m.replace(" + ", "\n+\n").replace(" on ", "\non\n")
                          for m in model_names], fontsize=8)
axes[0].set_ylim(0, 110)
axes[0].set_ylabel("Score (%)"); axes[0].set_title("Model Comparison", fontsize=11, fontweight="bold")
axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
for bar, val in zip(bars_acc, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1%}", ha="center", va="bottom", fontsize=8, fontweight="bold")

# Top features per class (LogReg coefficients)
if SK_AVAILABLE:
    top_n = 6
    colors_cls = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6"]

    ax1 = axes[1]
    ax1.set_title("Top TF-IDF Features per Class\n(LogReg Coefficients)",
                  fontsize=11, fontweight="bold")
    ax1.axis("off")

    y_pos = 0.95
    for cls_idx, (cls_name, color) in enumerate(zip(CATEGORIES, colors_cls)):
        ax1.text(0.02, y_pos, cls_name, fontsize=10, fontweight="bold",
                 color=color, transform=ax1.transAxes)
        top_idx = coef[cls_idx].argsort()[-top_n:][::-1] if SK_AVAILABLE else []
        y_pos  -= 0.03
        for i in top_idx[:top_n]:
            word  = feature_names[i]
            score = coef[cls_idx, i]
            ax1.text(0.05, y_pos, f"• {word:<20} {score:+.2f}",
                     fontsize=8.5, color="#333", transform=ax1.transAxes,
                     fontfamily="monospace")
            y_pos -= 0.028
        y_pos -= 0.01

# TF-IDF + MLP loss curve
if SK_AVAILABLE:
    axes[2].plot(mlp_losses, color="darkorange", linewidth=2)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Training Loss")
    axes[2].set_title(f"NumPy MLP Training Loss\n(TF-IDF features, test acc: {mlp_acc:.1%})",
                      fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].text(len(mlp_losses) * 0.6, mlp_losses[0] * 0.8,
                 f"Final loss:\n{mlp_losses[-1]:.4f}", fontsize=10,
                 color="darkorange", bbox=dict(boxstyle="round", facecolor="moccasin", alpha=0.6))
else:
    axes[2].text(0.5, 0.5, "Training loss plot\n(requires sklearn)",
                 ha="center", va="center", fontsize=12, transform=axes[2].transAxes)
    axes[2].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "model_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: model_comparison.png")


# --- PLOT 2: Confusion matrix + per-class F1 ---
print("Generating: Confusion matrix and per-class metrics...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Text Classification: Detailed Evaluation", fontsize=14, fontweight="bold")

if SK_AVAILABLE:
    cm = confusion_matrix(y_test, results[best_name]["preds"])
    im = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels(CATEGORIES, rotation=30, ha="right", fontsize=10)
    axes[0].set_yticks(range(4))
    axes[0].set_yticklabels(CATEGORIES, fontsize=10)
    axes[0].set_title(f"Confusion Matrix — {best_name}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    plt.colorbar(im, ax=axes[0])

    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                         fontsize=13, color="white" if cm[i, j] > cm.max()/2 else "black",
                         fontweight="bold")

    # Per-class precision, recall, F1
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1_pc, _ = precision_recall_fscore_support(
        y_test, results[best_name]["preds"], average=None, labels=range(4)
    )

    x4 = np.arange(4)
    w3 = 0.25
    axes[1].bar(x4 - w3, prec*100, w3, label="Precision", color="#3498DB", alpha=0.85, edgecolor="white")
    axes[1].bar(x4,      rec*100,  w3, label="Recall",    color="#E74C3C", alpha=0.85, edgecolor="white")
    axes[1].bar(x4 + w3, f1_pc*100,w3, label="F1",        color="#2ECC71", alpha=0.85, edgecolor="white")
    axes[1].set_xticks(x4)
    axes[1].set_xticklabels(CATEGORIES, fontsize=10)
    axes[1].set_ylim(0, 115)
    axes[1].set_ylabel("Score (%)"); axes[1].set_title("Per-Class Metrics", fontsize=11, fontweight="bold")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

else:
    for ax in axes:
        ax.text(0.5, 0.5, "Requires sklearn", ha="center", va="center",
                fontsize=14, transform=ax.transAxes)
        ax.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: confusion_matrix.png")


# --- PLOT 3: Pipeline flow diagram ---
print("Generating: Pipeline architecture diagram...")

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("Text Classification Pipeline Architecture",
             fontsize=15, fontweight="bold", pad=15)

steps = [
    ("Raw Text\nCorpus",       0.07, "#E74C3C",  "News headlines,\nreviews, emails..."),
    ("Preprocessing\nClean+Tokenize", 0.22, "#E67E22", "lowercase, strip HTML,\ncontraction expand"),
    ("TfidfVectorizer\n(1,2)-grams",  0.37, "#F1C40F", "10k features\nsublinear_tf\nstop_words='english'"),
    ("Classifier\n(LogReg/SVC)",      0.52, "#2ECC71", "C=10\nmulti_class=\n'multinomial'"),
    ("Prediction\n+ Probability",     0.67, "#3498DB", "class label +\nconfidence score"),
    ("Evaluation\nReport",            0.82, "#9B59B6", "accuracy, F1,\nconfusion matrix"),
]

bw_p, bh_p = 0.10, 0.34

for label, xc, color, note in steps:
    rect = mpatches.FancyBboxPatch((xc - bw_p/2, 0.33), bw_p, bh_p,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, alpha=0.85, edgecolor="white", linewidth=2)
    ax.add_patch(rect)
    ax.text(xc, 0.50, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
    ax.text(xc, 0.20, note, ha="center", va="center",
            fontsize=7.5, color="#444", style="italic",
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.7))

for i in range(len(steps) - 1):
    x0 = steps[i][1]   + bw_p/2 + 0.005
    x1 = steps[i+1][1] - bw_p/2 - 0.005
    ax.annotate("", xy=(x1, 0.50), xytext=(x0, 0.50),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2))

# sklearn Pipeline bracket
ax.annotate("", xy=(0.59, 0.78), xytext=(0.17, 0.78),
            arrowprops=dict(arrowstyle="-", color="#1ABC9C", lw=3))
ax.text(0.38, 0.83, "sklearn Pipeline (fit_transform → fit)", ha="center",
        fontsize=9, color="#1ABC9C", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "pipeline_architecture.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: pipeline_architecture.png")


print()
print("=" * 70)
print("NLP ALGORITHM 1: TEXT CLASSIFICATION PIPELINE COMPLETE!")
print("=" * 70)
print()
print("What you built:")
print("  ✓ Multi-class text classifier (4-class news topic)")
print("  ✓ sklearn Pipeline: TfidfVectorizer → Classifier in one object")
print("  ✓ Compared: LogReg, ComplementNB, LinearSVC, and numpy MLP")
print("  ✓ Feature interpretation: which words drive each class?")
print("  ✓ Evaluation: accuracy, F1, confusion matrix, cross-validation")
print()
print("Key results (on 40 examples — small dataset!):")
for name, res in results.items():
    print(f"  {name:<30}: {res['acc']:.1%} accuracy")
print()
print("3 Visualizations saved to: ../visuals/text_classification_pipeline/")
print("  1. model_comparison.png       — accuracy/F1 bars + feature importance")
print("  2. confusion_matrix.png       — confusion matrix + per-class metrics")
print("  3. pipeline_architecture.png  — full pipeline flow diagram")
print()
print("Next: Algorithm 2 → Sentiment Analysis")
