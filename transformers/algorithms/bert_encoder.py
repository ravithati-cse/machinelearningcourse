"""
BERT Encoder — Bidirectional Representations with HuggingFace
=============================================================

Learning Objectives:
  1. Understand BERT's bidirectional pretraining: MLM + NSP objectives
  2. Use HuggingFace BertTokenizer and WordPiece subword tokenization
  3. Extract contextual embeddings via frozen BERT (feature extraction)
  4. Compute cosine similarity between sentence embeddings
  5. Compare feature-extraction vs full fine-tuning strategies
  6. Build and evaluate a lightweight classifier on top of BERT embeddings

Time:         ~45 minutes
Difficulty:   Intermediate-Advanced
Prerequisites: math_foundations 01-04 (attention, multi-head attention,
               positional encoding, encoder-decoder architecture)
YouTube:      https://www.youtube.com/watch?v=xI0HHN5XKDo  (BERT explained)
              https://www.youtube.com/watch?v=OR0wfP2FD3c  (HuggingFace walkthrough)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

VIS_DIR = os.path.join(os.path.dirname(__file__), "..", "visuals", "bert_encoder")
os.makedirs(VIS_DIR, exist_ok=True)

np.random.seed(42)

# =============================================================================
# SECTION 1: What is BERT?
# =============================================================================
print("=" * 70)
print("SECTION 1: What is BERT?")
print("=" * 70)

print("""
BERT — Bidirectional Encoder Representations from Transformers
Introduced by Devlin et al. (Google, 2019): https://arxiv.org/abs/1810.04805

Core Idea:
  Traditional language models read text LEFT-TO-RIGHT (like GPT) or
  RIGHT-TO-LEFT. BERT reads in BOTH directions simultaneously, giving
  each token full context from the entire sentence.

  Example:
    "The bank can guarantee deposits will eventually cover future costs."
    "I sat on the river bank and watched the water flow."
    
    The word "bank" means something completely different in each sentence.
    A unidirectional model misses context from the right. BERT sees all
    tokens at once and encodes the correct meaning.

Pre-training Tasks (what BERT learned before you fine-tune it):
  1. Masked Language Modeling (MLM):
       Input:  "The cat sat on the [MASK]."
       Target: "mat"
       15% of tokens are randomly masked and BERT predicts them.
       Forces the model to understand context from BOTH sides.

  2. Next Sentence Prediction (NSP):
       Input A: "The man went to the store."
       Input B: "He bought a gallon of milk."  [IsNext]
       or
       Input B: "Penguins live in Antarctica." [NotNext]
       BERT classifies whether B follows A. Teaches sentence relationships.

Fine-tuning:
  After pretraining on massive text (BooksCorpus + Wikipedia), BERT can
  be fine-tuned on downstream tasks with just a small classification head:
  - Text Classification
  - Named Entity Recognition (NER)
  - Question Answering (SQuAD)
  - Sentence Pair Similarity
""")

print("BERT Model Specifications:")
print("-" * 55)
print(f"{'Spec':<30} {'BERT-base':<15} {'BERT-large':<10}")
print("-" * 55)
specs = [
    ("Encoder Layers",     "12",    "24"),
    ("d_model (hidden)",   "768",   "1024"),
    ("Attention Heads",    "12",    "16"),
    ("FFN Inner dim",      "3072",  "4096"),
    ("Parameters",         "110M",  "340M"),
    ("Max Sequence Len",   "512",   "512"),
    ("Pretraining Data",   "16GB",  "16GB"),
]
for name, base, large in specs:
    print(f"  {name:<28} {base:<15} {large:<10}")
print("-" * 55)

# =============================================================================
# SECTION 2: BERT Tokenization
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: BERT Tokenization (WordPiece)")
print("=" * 70)

TRANSFORMERS_AVAILABLE = False
try:
    from transformers import BertTokenizer, BertModel
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("HuggingFace transformers detected — using real BERT tokenizer.\n")
except ImportError:
    print("transformers not installed. Simulating tokenizer output.")
    print("Install with:  pip install transformers torch\n")

if TRANSFORMERS_AVAILABLE:
    print("Loading bert-base-uncased tokenizer...")
    tok = BertTokenizer.from_pretrained("bert-base-uncased")

    sample_sentences = [
        "The cat sat on the mat.",
        "Machine learning is transforming the world.",
        "Unbelievable results from the new model.",
    ]

    print("\n--- Special Tokens ---")
    print(f"  [CLS] token id : {tok.cls_token_id}   (added at the START of every input)")
    print(f"  [SEP] token id : {tok.sep_token_id}   (added at the END of every input / between segments)")
    print(f"  [PAD] token id : {tok.pad_token_id}     (used for padding shorter sequences in a batch)")
    print(f"  [MASK] token id: {tok.mask_token_id}  (replaces tokens in MLM pretraining)")

    print("\n--- Tokenization Examples ---")
    for sent in sample_sentences:
        tokens = tok.tokenize(sent)
        ids = tok.convert_tokens_to_ids(tokens)
        print(f"\n  Input   : {sent!r}")
        print(f"  Tokens  : {tokens}")
        print(f"  Token IDs: {ids}")

    print("\n--- WordPiece Subword Tokenization ---")
    word = "unbelievable"
    tokens = tok.tokenize(word)
    print(f"  'unbelievable' → {tokens}")
    print("  Words not in vocabulary are split into subwords.")
    print("  '##' prefix means 'continuation of the previous token'.")

    print("\n--- Full Encoding (with special tokens + padding) ---")
    encoded = tok(
        sample_sentences,
        padding=True,
        truncation=True,
        max_length=20,
        return_tensors="pt",
    )
    for key, val in encoded.items():
        print(f"  {key:<20} shape: {tuple(val.shape)}")
    print("\n  input_ids row 0     :", encoded["input_ids"][0].tolist())
    print("  attention_mask row 0:", encoded["attention_mask"][0].tolist())
    print("  token_type_ids row 0:", encoded["token_type_ids"][0].tolist())
    print("  (token_type_ids = 0 for sentence A, 1 for sentence B in NSP pairs)")

else:
    # Simulated output
    print("--- Simulated Tokenizer Output ---")
    print("\n  Special Tokens:")
    print("    [CLS]  → id 101  (added at start of every input)")
    print("    [SEP]  → id 102  (added at end / between segments)")
    print("    [PAD]  → id 0    (padding for batching)")
    print("    [MASK] → id 103  (masked token for MLM pretraining)")

    print("\n  'The cat sat on the mat.' →")
    print("    tokens: ['the', 'cat', 'sat', 'on', 'the', 'mat', '.']")
    print("    ids:    [101, 1996, 4937, 4490, 2006, 1996, 13523, 1012, 102]")

    print("\n  'unbelievable' →")
    print("    tokens: ['un', '##beli', '##ev', '##able']")
    print("    (WordPiece splits unknown words into known subword pieces)")

    print("\n  Encoded batch (3 sentences, padded to length 20):")
    print("    input_ids       shape: (3, 20)")
    print("    attention_mask  shape: (3, 20)  — 1=real token, 0=padding")
    print("    token_type_ids  shape: (3, 20)  — 0=seg A, 1=seg B")

print("\n--- Subword Tokenization Approaches Comparison ---")
print("-" * 60)
print(f"  {'Method':<20} {'Used By':<20} {'Key Idea'}")
print("-" * 60)
approaches = [
    ("WordPiece", "BERT, DistilBERT", "Maximize likelihood of training data"),
    ("BPE",       "GPT-2, RoBERTa",  "Merge most frequent byte pairs"),
    ("Unigram",   "T5, ALBERT",      "Probabilistic subword model"),
    ("SentencePiece","XLNet, T5",    "Language-agnostic, raw text"),
]
for name, models, idea in approaches:
    print(f"  {name:<20} {models:<20} {idea}")
print("-" * 60)

# =============================================================================
# SECTION 3: Feature Extraction (Frozen BERT)
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: Feature Extraction (Frozen BERT)")
print("=" * 70)

SAMPLE_TEXTS = [
    "Astronomers discovered a new planet in a distant solar system.",
    "The spacecraft launched successfully and is heading to Mars.",
    "Scientists analyzed protein folding using deep learning.",
    "The stock market surged after the Federal Reserve announcement.",
    "Investors are optimistic about technology earnings this quarter.",
]

LABELS = ["science", "science", "biology", "finance", "finance"]

print("Sample sentences for embedding extraction:")
for i, (text, label) in enumerate(zip(SAMPLE_TEXTS, LABELS)):
    print(f"  [{i}] ({label:>8}) {text}")

if TRANSFORMERS_AVAILABLE:
    print("\nLoading bert-base-uncased model (this may take a moment)...")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()

    encoded = tok(
        SAMPLE_TEXTS,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    print("Running forward pass (no gradients)...")
    with torch.no_grad():
        outputs = model(**encoded)

    last_hidden = outputs.last_hidden_state   # (batch, seq_len, 768)
    pooler_out  = outputs.pooler_output       # (batch, 768)

    cls_emb  = last_hidden[:, 0, :]                                  # (batch, 768) — [CLS] token
    mean_emb = last_hidden.mean(dim=1)                               # (batch, 768) — mean pool
    max_emb  = last_hidden.max(dim=1).values                        # (batch, 768) — max pool

    print(f"\n  last_hidden_state shape : {tuple(last_hidden.shape)}  (batch, seq_len, d_model)")
    print(f"  pooler_output shape     : {tuple(pooler_out.shape)}        (batch, d_model)")
    print(f"  cls_embeddings shape    : {tuple(cls_emb.shape)}        ([CLS] token per sentence)")
    print(f"  mean_pool shape         : {tuple(mean_emb.shape)}        (average over all tokens)")
    print(f"  max_pool shape          : {tuple(max_emb.shape)}        (element-wise max over tokens)")

    print("\n  Embedding stats for sentence 0 (CLS token):")
    ce = cls_emb[0].numpy()
    print(f"    mean={ce.mean():.4f}  std={ce.std():.4f}  min={ce.min():.4f}  max={ce.max():.4f}")

    # Convert to numpy for downstream use
    embeddings = cls_emb.numpy()   # shape (5, 768)

else:
    # Simulate with structured random embeddings
    print("\nSimulating BERT embeddings with structured numpy arrays...")
    print("(Real BERT would produce contextual 768-dim vectors)")

    d_model = 768
    # Create class-structured embeddings: sentences in same class share a bias
    class_centroids = {
        "science":  np.random.randn(d_model) * 0.5,
        "biology":  np.random.randn(d_model) * 0.5,
        "finance":  np.random.randn(d_model) * 0.5,
    }
    embeddings = np.array([
        class_centroids[label] + np.random.randn(d_model) * 0.1
        for label in LABELS
    ])  # (5, 768)

    print(f"\n  Simulated last_hidden_state shape : (5, 64, 768)")
    print(f"  Simulated pooler_output shape     : (5, 768)")
    print(f"  Simulated cls_embeddings shape    : (5, 768)")
    print(f"  Simulated mean_pool shape         : (5, 768)")
    print(f"  Simulated max_pool shape          : (5, 768)")
    print(f"\n  Embedding stats for sentence 0 (simulated):")
    print(f"    mean={embeddings[0].mean():.4f}  std={embeddings[0].std():.4f}  "
          f"min={embeddings[0].min():.4f}  max={embeddings[0].max():.4f}")

print("""
Pooling Strategy Explanation:
  [CLS] token   : BERT's "summary" token. The pooler applies a tanh linear
                  layer on top of [CLS]. Best for classification tasks.
  Mean pooling  : Average all token embeddings. Often best for sentence
                  similarity (STS benchmarks). Recommended by sentence-transformers.
  Max pooling   : Takes the max across each dimension. Captures the most
                  prominent features. Less common but sometimes effective.
""")

# =============================================================================
# SECTION 4: Cosine Similarity of BERT Embeddings
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 4: Cosine Similarity of BERT Embeddings")
print("=" * 70)


def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def cosine_similarity_matrix(X):
    """Compute full cosine similarity matrix for rows of X."""
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
    X_norm = X / norms
    return X_norm @ X_norm.T


sim_matrix = cosine_similarity_matrix(embeddings)

SHORT_LABELS = ["Astron.", "Spacecr.", "Protein", "Stock Mkt", "Investors"]

print("Cosine Similarity Matrix (higher = more similar):")
print()
header = f"  {'':12}" + "".join(f"{l:>11}" for l in SHORT_LABELS)
print(header)
print("  " + "-" * (12 + 11 * len(SHORT_LABELS)))
for i, row_label in enumerate(SHORT_LABELS):
    row = f"  {row_label:<12}"
    for j in range(len(SHORT_LABELS)):
        val = sim_matrix[i, j]
        marker = " <--" if i != j and LABELS[i] == LABELS[j] else ""
        row += f"  {val:>7.4f} "
    print(row)

print()
print("Observations:")
print("  - Science sentences (0,1) should have high similarity with each other")
print("  - Finance sentences (3,4) should have high similarity with each other")
print("  - Cross-domain pairs (e.g. 0 vs 3) should have low similarity")

# Highlight pairs
pairs = [(0, 1), (3, 4), (0, 3), (1, 4)]
pair_names = [
    ("Astron. vs Spacecr.", "same-domain: science"),
    ("Stock Mkt vs Investors", "same-domain: finance"),
    ("Astron. vs Stock Mkt", "cross-domain"),
    ("Spacecr. vs Investors", "cross-domain"),
]
print()
for (i, j), (name, domain) in zip(pairs, pair_names):
    print(f"  {name:<30} ({domain:>20})  sim = {sim_matrix[i,j]:.4f}")

# =============================================================================
# SECTION 5: Fine-tuning Strategy
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: Fine-tuning Strategy")
print("=" * 70)

print("""
Two Approaches to Using BERT:

  ┌─────────────────────────────────────────────────────────────────┐
  │  Approach 1: Feature Extraction (Frozen BERT)                   │
  │                                                                 │
  │  BERT weights: FROZEN (not updated during training)             │
  │  Workflow:                                                       │
  │    1. Run BERT once → get [CLS] embeddings for all samples      │
  │    2. Train a lightweight classifier on top (LogReg, MLP, SVM)  │
  │  Pros: Fast, cheap, requires little labeled data                 │
  │  Cons: Cannot adapt BERT's representations to your domain        │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  Approach 2: Full Fine-tuning                                   │
  │                                                                 │
  │  BERT weights: UNFROZEN (updated with a very small LR)          │
  │  Workflow:                                                       │
  │    1. Add a linear classification head on top of [CLS]          │
  │    2. Train the entire model end-to-end                         │
  │  Pros: Best performance, adapts to domain-specific language      │
  │  Cons: Slower, needs ~1000+ labeled examples, GPU recommended   │
  └─────────────────────────────────────────────────────────────────┘

Recommended Hyperparameters for Fine-tuning BERT:
  Learning Rate    : 2e-5  (very small — BERT weights are pretrained)
  Batch Size       : 16 or 32
  Epochs           : 3–5   (more can cause catastrophic forgetting)
  Warmup Steps     : ~10% of total training steps
  Optimizer        : AdamW with weight decay 0.01
  Max Seq Length   : 128 (fast) or 512 (full, slower)
  Scheduler        : Linear decay after warmup
""")

print("--- From-Scratch Demo: BertClassificationHead ---")
print("Using simulated 768-dim [CLS] embeddings → classify into 5 categories")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class BertClassificationHead:
    """
    Minimal classification head over BERT [CLS] embeddings.
    Mimics BertForSequenceClassification's pooler + linear layer.
    """
    def __init__(self, d_model=768, num_classes=5):
        self.W = np.random.randn(d_model, num_classes) * 0.02
        self.b = np.zeros(num_classes)

    def forward(self, cls_emb):
        """cls_emb: (batch, d_model) → logits: (batch, num_classes)"""
        logits = cls_emb @ self.W + self.b
        # Softmax
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)


# Simulate a small dataset: 5 classes, 20 samples each
NUM_CLASSES = 5
SAMPLES_PER_CLASS = 20
CLASS_NAMES = ["science", "finance", "sports", "politics", "entertainment"]

# Create structured embeddings: each class has a distinct centroid
d_model = 768
class_centroids_synth = np.random.randn(NUM_CLASSES, d_model)
X_synth = []
y_synth = []
for cls_idx in range(NUM_CLASSES):
    noise = np.random.randn(SAMPLES_PER_CLASS, d_model) * 0.3
    X_synth.append(class_centroids_synth[cls_idx] + noise)
    y_synth.extend([cls_idx] * SAMPLES_PER_CLASS)

X_synth = np.vstack(X_synth)
y_synth = np.array(y_synth)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_synth)

clf = LogisticRegression(max_iter=1000, C=1.0)
scores = cross_val_score(clf, X_scaled, y_synth, cv=5, scoring="accuracy")

print(f"  Dataset   : {NUM_CLASSES} classes × {SAMPLES_PER_CLASS} samples = {NUM_CLASSES * SAMPLES_PER_CLASS} total")
print(f"  Embedding : 768-dim (simulated BERT [CLS] output)")
print(f"  Classifier: LogisticRegression (simulates linear classification head)")
print(f"  CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"  (High accuracy because structured synthetic embeddings; real BERT")
print(f"   performs similarly well on small labeled datasets via feature extraction)")

head = BertClassificationHead(d_model=768, num_classes=NUM_CLASSES)
sample_cls = X_synth[:3]
probs = head.forward(sample_cls)
print(f"\n  BertClassificationHead demo (random weights — not trained):")
print(f"  Input shape : {sample_cls.shape}")
print(f"  Output shape: {probs.shape}  (softmax probabilities)")
for i in range(3):
    pred = np.argmax(probs[i])
    print(f"  Sample {i}: predicted class '{CLASS_NAMES[pred]}' with prob {probs[i, pred]:.4f}")

# =============================================================================
# SECTION 6: Comparison — Feature Extraction vs Fine-tuning
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 6: Feature Extraction vs Fine-tuning Comparison")
print("=" * 70)

print()
print(f"  {'Aspect':<28} {'Feature Extraction':<25} {'Full Fine-tuning'}")
print("  " + "-" * 80)
comparison = [
    ("BERT weights",        "Frozen",                   "Updated (very small LR)"),
    ("Training speed",      "Very fast",                "Slow (full model)"),
    ("GPU requirement",     "Not required",             "Strongly recommended"),
    ("Labeled data needed", "~100–500 examples",        "~1000+ examples"),
    ("Performance ceiling", "Moderate",                 "High (SOTA on many tasks)"),
    ("Overfitting risk",    "Low",                      "Higher (needs regularization)"),
    ("Implementation",      "sklearn / simple numpy",   "PyTorch / HuggingFace Trainer"),
    ("Typical SST-2 Acc.",  "~88–90%",                  "~93–94%"),
    ("Typical MNLI Acc.",   "~74–76%",                  "~84–86%"),
    ("Typical SQuAD F1",    "~72–75%",                  "~88–91%"),
]
for aspect, feat, fine in comparison:
    print(f"  {aspect:<28} {feat:<25} {fine}")
print("  " + "-" * 80)

print("""
Key Takeaway:
  - Start with feature extraction when you have limited labeled data or compute.
  - Switch to fine-tuning when you need maximum performance and have 1000+ examples.
  - For production, consider DistilBERT (60% smaller, 97% of BERT performance).
""")

# =============================================================================
# SECTION 7: Visualizations
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 7: Generating Visualizations")
print("=" * 70)

# ── Plot 1: BERT Architecture + Parameter Count ────────────────────────────
fig, (ax_arch, ax_params) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle("BERT Architecture Overview", fontsize=15, fontweight="bold")

# Left: architecture diagram
ax_arch.set_xlim(0, 10)
ax_arch.set_ylim(0, 12)
ax_arch.axis("off")
ax_arch.set_title("BERT-base Architecture Flow", fontsize=12, fontweight="bold")

def draw_box(ax, x, y, w, h, text, color, fontsize=9, text_color="white"):
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="white", linewidth=1.5,
    )
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold",
            wrap=True, multialignment="center")

def draw_arrow(ax, x, y_start, y_end):
    ax.annotate("", xy=(x, y_end), xytext=(x, y_start),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.5))

cx = 5.0
layers = [
    (0.8,  "Raw Text Input",              "#607D8B", 1.0, 0.6),
    (1.9,  "WordPiece Tokenizer\n[CLS] + tokens + [SEP]", "#455A64", 1.8, 0.7),
    (3.1,  "Token + Positional\n+ Segment Embeddings",    "#1565C0", 2.0, 0.7),
    (4.5,  "Encoder Layer 1\n(Self-Attention + FFN)",     "#1976D2", 2.2, 0.7),
    (5.7,  "Encoder Layer 2…11\n(× 10 more layers)",      "#1E88E5", 2.2, 0.7),
    (6.9,  "Encoder Layer 12",                            "#2196F3", 2.2, 0.7),
    (8.1,  "[CLS] Hidden State\n(768-dim vector)",        "#0288D1", 2.0, 0.7),
    (9.4,  "Task-Specific Head\n(Linear + Softmax)",      "#00897B", 2.0, 0.7),
    (10.7, "Output: Class / NER\n/ Span / Similarity",   "#2E7D32", 2.0, 0.7),
]
prev_y = None
for y, text, color, w, h in layers:
    draw_box(ax_arch, cx, y, w, h, text, color, fontsize=8)
    if prev_y is not None:
        draw_arrow(ax_arch, cx, prev_y + h / 2, y - h / 2)
    prev_y = y

ax_arch.set_xlim(1, 9)
ax_arch.set_ylim(0, 11.5)

# Right: parameter comparison bar chart
ax_params.set_title("BERT Model Parameter Counts", fontsize=12, fontweight="bold")
models = ["BERT-tiny\n(4.4M)", "BERT-mini\n(11M)", "BERT-base\n(110M)",
          "BERT-large\n(340M)", "BERT-XL\n(~560M)"]
params = [4.4, 11, 110, 340, 560]
colors = ["#BBDEFB", "#90CAF9", "#1976D2", "#0D47A1", "#01579B"]
bars = ax_params.barh(models, params, color=colors, edgecolor="white", height=0.55)
for bar, val in zip(bars, params):
    ax_params.text(val + 8, bar.get_y() + bar.get_height() / 2,
                   f"{val}M", va="center", fontsize=9, fontweight="bold", color="#333333")
ax_params.set_xlabel("Parameters (Millions)", fontsize=10)
ax_params.set_xlim(0, 650)
ax_params.axvline(110, color="#E53935", linestyle="--", linewidth=1.5,
                  label="BERT-base (most common)")
ax_params.legend(fontsize=8)
ax_params.spines[["top", "right"]].set_visible(False)
ax_params.set_facecolor("#FAFAFA")

plt.tight_layout()
path1 = os.path.join(VIS_DIR, "01_bert_architecture.png")
plt.savefig(path1, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {path1}")

# ── Plot 2: Embeddings Analysis ────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle("BERT Embedding Analysis", fontsize=14, fontweight="bold")

# Left: PCA of embeddings (manual SVD-based PCA)
ax_pca = axes2[0]
ax_pca.set_title("PCA of Sentence Embeddings (2D)", fontsize=11, fontweight="bold")

X_centered = embeddings - embeddings.mean(axis=0)
_, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
pca_2d = X_centered @ Vt[:2].T   # project onto top 2 PCs

label_colors = {"science": "#1976D2", "biology": "#7B1FA2", "finance": "#E53935"}
for i, (text, label) in enumerate(zip(SAMPLE_TEXTS, LABELS)):
    color = label_colors[label]
    ax_pca.scatter(pca_2d[i, 0], pca_2d[i, 1], c=color, s=120, zorder=3,
                   edgecolors="white", linewidth=1.5)
    short = text[:30] + "…"
    ax_pca.annotate(short, (pca_2d[i, 0], pca_2d[i, 1]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=color)

legend_patches = [
    mpatches.Patch(color=c, label=l) for l, c in label_colors.items()
]
ax_pca.legend(handles=legend_patches, fontsize=8, loc="best")
ax_pca.set_xlabel("PC 1", fontsize=10)
ax_pca.set_ylabel("PC 2", fontsize=10)
ax_pca.spines[["top", "right"]].set_visible(False)
ax_pca.set_facecolor("#FAFAFA")

# Middle: Cosine similarity heatmap
ax_heat = axes2[1]
ax_heat.set_title("Cosine Similarity Heatmap", fontsize=11, fontweight="bold")
im = ax_heat.imshow(sim_matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
ax_heat.set_xticks(range(5))
ax_heat.set_yticks(range(5))
ax_heat.set_xticklabels(SHORT_LABELS, rotation=30, ha="right", fontsize=8)
ax_heat.set_yticklabels(SHORT_LABELS, fontsize=8)
for i in range(5):
    for j in range(5):
        ax_heat.text(j, i, f"{sim_matrix[i,j]:.2f}",
                     ha="center", va="center", fontsize=9,
                     color="white" if sim_matrix[i, j] > 0.6 else "black")
plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

# Right: CLS vs mean_pool vs max_pool
ax_pool = axes2[2]
ax_pool.set_title("Pooling Strategy Comparison\n(Cosine similarity for sentence pair 0 vs 1)",
                  fontsize=10, fontweight="bold")

if TRANSFORMERS_AVAILABLE:
    with torch.no_grad():
        enc2 = tok(SAMPLE_TEXTS[:2], padding=True, truncation=True,
                   max_length=64, return_tensors="pt")
        out2 = model(**enc2)
        lhs = out2.last_hidden_state.numpy()
    cls_sims  = cosine_similarity(lhs[0, 0], lhs[1, 0])
    mean_sims = cosine_similarity(lhs[0].mean(0), lhs[1].mean(0))
    max_sims  = cosine_similarity(lhs[0].max(0), lhs[1].max(0))
else:
    cls_sims  = cosine_similarity(embeddings[0], embeddings[1])
    # Simulate mean/max with slight variation
    mean_sims = float(np.clip(cls_sims + np.random.uniform(-0.05, 0.05), 0, 1))
    max_sims  = float(np.clip(cls_sims + np.random.uniform(-0.05, 0.05), 0, 1))

pool_names = ["[CLS]\nToken", "Mean\nPooling", "Max\nPooling"]
pool_vals  = [cls_sims, mean_sims, max_sims]
pool_colors = ["#1976D2", "#43A047", "#FB8C00"]
bars3 = ax_pool.bar(pool_names, pool_vals, color=pool_colors, edgecolor="white",
                    width=0.5)
for bar, val in zip(bars3, pool_vals):
    ax_pool.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")
ax_pool.set_ylim(0, 1.15)
ax_pool.set_ylabel("Cosine Similarity", fontsize=10)
ax_pool.spines[["top", "right"]].set_visible(False)
ax_pool.set_facecolor("#FAFAFA")

plt.tight_layout()
path2 = os.path.join(VIS_DIR, "02_embeddings_analysis.png")
plt.savefig(path2, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {path2}")

# ── Plot 3: Fine-tuning Guide ──────────────────────────────────────────────
fig3, (ax_lr, ax_perf) = plt.subplots(1, 2, figsize=(15, 6))
fig3.suptitle("BERT Fine-tuning Guide", fontsize=14, fontweight="bold")

# Left: LR schedule (warmup + linear decay)
ax_lr.set_title("Learning Rate Schedule\n(Warmup + Linear Decay)", fontsize=11,
                fontweight="bold")
total_steps = 1000
warmup_steps = 100
steps = np.arange(total_steps)

def lr_schedule(step, warmup=100, total=1000, peak_lr=2e-5):
    if step < warmup:
        return peak_lr * step / warmup
    else:
        return peak_lr * max(0.0, (total - step) / (total - warmup))

lr_vals = np.array([lr_schedule(s) for s in steps])

ax_lr.plot(steps, lr_vals * 1e5, color="#1976D2", linewidth=2.5, label="LR × 1e5")
ax_lr.axvline(warmup_steps, color="#E53935", linestyle="--", linewidth=1.5,
              label=f"Warmup ends (step {warmup_steps})")
ax_lr.fill_between(steps[:warmup_steps], lr_vals[:warmup_steps] * 1e5,
                   alpha=0.2, color="#43A047", label="Warmup phase")
ax_lr.fill_between(steps[warmup_steps:], lr_vals[warmup_steps:] * 1e5,
                   alpha=0.2, color="#1976D2", label="Decay phase")
ax_lr.set_xlabel("Training Step", fontsize=10)
ax_lr.set_ylabel("Learning Rate (×1e-5)", fontsize=10)
ax_lr.legend(fontsize=8)
ax_lr.spines[["top", "right"]].set_visible(False)
ax_lr.set_facecolor("#FAFAFA")

# Right: performance vs training data size
ax_perf.set_title("Performance vs Training Data Size\n(Feature Extraction vs Fine-tuning)",
                   fontsize=11, fontweight="bold")

data_sizes = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000])

# Feature extraction: good early, plateaus
feat_acc = 75 + 15 * (1 - np.exp(-data_sizes / 300)) + np.random.randn(len(data_sizes)) * 0.3
# Fine-tuning: needs more data, but reaches higher ceiling
fine_acc = 60 + 32 * (1 - np.exp(-data_sizes / 1500)) + np.random.randn(len(data_sizes)) * 0.3

ax_perf.semilogx(data_sizes, feat_acc, "o-", color="#1976D2", linewidth=2.5,
                  markersize=7, label="Feature Extraction (frozen BERT)")
ax_perf.semilogx(data_sizes, fine_acc, "s-", color="#E53935", linewidth=2.5,
                  markersize=7, label="Full Fine-tuning")
ax_perf.axhline(92, color="#E53935", linestyle=":", linewidth=1, alpha=0.6,
                label="Fine-tuning ceiling (~92%)")
ax_perf.axhline(88, color="#1976D2", linestyle=":", linewidth=1, alpha=0.6,
                label="Feature extraction ceiling (~88%)")
ax_perf.set_xlabel("Number of Labeled Training Examples (log scale)", fontsize=10)
ax_perf.set_ylabel("Accuracy (%)", fontsize=10)
ax_perf.set_ylim(55, 100)
ax_perf.legend(fontsize=8, loc="lower right")
ax_perf.spines[["top", "right"]].set_visible(False)
ax_perf.set_facecolor("#FAFAFA")

plt.tight_layout()
path3 = os.path.join(VIS_DIR, "03_finetuning_guide.png")
plt.savefig(path3, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {path3}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY — BERT Encoder")
print("=" * 70)
print("""
What You Learned:
  1. BERT is a bidirectional transformer encoder pretrained with MLM + NSP.
     It sees the FULL sentence context simultaneously — unlike GPT (left-to-right).

  2. WordPiece tokenization splits unknown words into subwords ('##' suffix).
     Special tokens: [CLS]=101, [SEP]=102, [PAD]=0, [MASK]=103.

  3. Feature Extraction: freeze BERT, run once to get [CLS] embeddings,
     train a lightweight head (LogReg / MLP). Fast, requires little labeled data.

  4. Full Fine-tuning: unfreeze all BERT weights, use LR ~2e-5, warmup + decay.
     Better performance ceiling. Needs ~1000+ labeled examples.

  5. Cosine similarity of [CLS] embeddings reflects semantic similarity.
     Same-domain sentences cluster together in the embedding space.

  6. Mean pooling often outperforms [CLS] for sentence-pair similarity tasks.
     [CLS] is preferred for classification tasks.

Next Steps:
  - Explore sentence-transformers library for optimized BERT embeddings
  - Try fine-tuning on SST-2 (sentiment) or MRPC (paraphrase) with Trainer API
  - See gpt_decoder.py for the generative / autoregressive counterpart

Visualizations saved to:
""")
for p in [path1, path2, path3]:
    print(f"  {p}")
