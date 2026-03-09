"""
🔁 NLP — Algorithm 3: LSTM Text Classifier
===========================================

Learning Objectives:
  1. Build a deep learning text classification pipeline end-to-end
  2. Tokenize and pad sequences for batch training in Keras
  3. Implement an Embedding layer (trainable from scratch and pretrained)
  4. Build LSTM / BiLSTM / GRU models with Keras for text classification
  5. Handle variable-length sequences with masking and global pooling
  6. Compare Embedding+LSTM vs TF-IDF+LR on the same classification task
  7. Visualize learned embeddings and attention-like LSTM activations

YouTube Resources:
  ⭐ Sentdex — LSTM text classification https://www.youtube.com/watch?v=fjmwkeSaB3Y
  ⭐ TF — Text classification RNN https://www.tensorflow.org/text/tutorials/text_classification_rnn
  📚 Keras LSTM docs https://keras.io/api/layers/recurrent_layers/lstm/

Time Estimate: 75 min
Difficulty: Intermediate-Advanced
Prerequisites: 04_rnn_intuition.py, cnn_with_keras.py (Keras patterns)
Key Concepts: Embedding layer, padding/masking, LSTM, BiLSTM, GlobalMaxPooling1D
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
import os

_VISUALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "lstm_text_classifier")
os.makedirs(_VISUALS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🔁 NLP ALGORITHM 3: LSTM TEXT CLASSIFIER")
print("=" * 70)
print()
print("TF-IDF + LogReg gets 85-92% on most tasks.")
print("LSTM gets another 2-5% by understanding WORD ORDER.")
print()
print("Pipeline:")
print("  Raw text")
print("   → Tokenize + build vocabulary")
print("   → Integer-encode: ['I', 'love', 'this'] → [3, 47, 12]")
print("   → Pad to fixed length: [3, 47, 12, 0, 0, 0, ...]")
print("   → Embedding layer: integers → dense vectors (100-300d)")
print("   → LSTM/BiLSTM: reads sequence, outputs hidden states")
print("   → Pooling: collapse sequence → single vector")
print("   → Dense + softmax → class probabilities")
print()


# ======================================================================
# SECTION 1: Dataset
# ======================================================================
print("=" * 70)
print("SECTION 1: DATASET PREPARATION")
print("=" * 70)
print()

# Synthetic IMDb-style review dataset
positive_reviews = [
    "This movie was absolutely wonderful and emotionally moving",
    "Great performances from all the actors especially the lead",
    "I loved the storyline it kept me hooked from beginning to end",
    "The director did a fantastic job with this masterpiece",
    "One of the best films I have seen in recent years",
    "Brilliant script with witty dialogue and a satisfying conclusion",
    "Amazing cinematography and a stellar soundtrack throughout",
    "A touching story that resonates long after the credits roll",
    "Superb acting and a compelling narrative I highly recommend",
    "This film exceeded all my expectations it was truly special",
    "An incredible journey that takes you through a range of emotions",
    "The character development was outstanding and felt very real",
    "A must watch film that everyone should experience at least once",
    "Perfectly paced with stunning visuals and a powerful message",
    "This is cinema at its finest absolutely loved every moment",
    "The plot twists were unexpected and very well executed",
    "Wonderful ensemble cast with chemistry that felt completely natural",
    "A beautiful film that deserves all the awards it has received",
    "Really moved me emotionally best film I have seen this year",
    "Exceptional filmmaking that showcases the power of storytelling",
]

negative_reviews = [
    "This movie was a complete waste of two hours of my life",
    "Terrible acting and a plot that made absolutely no sense",
    "I hated every single minute of this dreadful disaster",
    "The director clearly had no idea what they were doing here",
    "One of the worst films I have ever had the misfortune to watch",
    "Awful script with cringe dialogue and a deeply unsatisfying ending",
    "Boring and repetitive with nothing interesting happening at all",
    "A pointless story that goes nowhere and means nothing",
    "Horrendous acting and a narrative that was completely incoherent",
    "This film fell far short of all expectations it was truly bad",
    "A painful experience that drags on for far too long",
    "The characters were flat and the development was nonexistent",
    "Avoid this film at all costs it is not worth your time",
    "Terribly paced with ugly visuals and a muddled pointless message",
    "This is cinema at its worst absolutely hated every moment",
    "The plot holes were glaring and completely ruined the experience",
    "A miserable cast with zero chemistry it felt completely fake",
    "A dreadful film that deserves none of the praise it received",
    "Deeply disappointing worst film I have seen in years sadly",
    "Incompetent filmmaking that shows a complete lack of talent",
]

texts  = positive_reviews + negative_reviews
labels = [1] * len(positive_reviews) + [0] * len(negative_reviews)

# Shuffle
perm   = np.random.permutation(len(texts))
texts  = [texts[i]  for i in perm]
labels = [labels[i] for i in perm]

split = int(0.75 * len(texts))
X_train_raw, X_val_raw = texts[:split], texts[split:]
y_train, y_val         = labels[:split], labels[split:]

print(f"  Dataset: {len(texts)} reviews ({len(positive_reviews)} pos, {len(negative_reviews)} neg)")
print(f"  Train: {len(X_train_raw)}, Val: {len(X_val_raw)}")
print()


# ======================================================================
# SECTION 2: Text to Sequences
# ======================================================================
print("=" * 70)
print("SECTION 2: TEXT → INTEGER SEQUENCES → PADDED MATRIX")
print("=" * 70)
print()

MAX_VOCAB    = 2000
MAX_SEQ_LEN  = 20
EMBED_DIM    = 64


def simple_tokenize(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def build_vocab(texts, max_vocab=None):
    """Build word→idx mapping from training texts."""
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in counter.most_common(max_vocab):
        vocab[word] = len(vocab)
    return vocab


def texts_to_sequences(texts, vocab, max_len):
    """Convert list of strings to padded integer matrix."""
    sequences = []
    for text in texts:
        tokens = simple_tokenize(text)
        seq    = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
        seq    = seq[:max_len]
        # Pad right with zeros
        seq   += [0] * (max_len - len(seq))
        sequences.append(seq)
    return np.array(sequences)


vocab      = build_vocab(X_train_raw, max_vocab=MAX_VOCAB)
X_train_seq = texts_to_sequences(X_train_raw, vocab, MAX_SEQ_LEN)
X_val_seq   = texts_to_sequences(X_val_raw,   vocab, MAX_SEQ_LEN)
y_train_np  = np.array(y_train)
y_val_np    = np.array(y_val)

print(f"  Vocabulary size: {len(vocab)} (incl. PAD=0, UNK=1)")
print(f"  Max sequence length: {MAX_SEQ_LEN}")
print(f"  X_train_seq shape: {X_train_seq.shape}")
print()
print("  Example encoding:")
example_text = X_train_raw[0]
example_seq  = X_train_seq[0]
example_toks = simple_tokenize(example_text)
print(f"  Text:   {example_text!r}")
print(f"  Tokens: {example_toks}")
print(f"  IDs:    {[vocab.get(t, 1) for t in example_toks[:8]]}...")
print(f"  Padded: {example_seq.tolist()}")
print()

# Show padding effect
lengths = [len(simple_tokenize(t)) for t in X_train_raw]
print(f"  Sequence lengths: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}")
print(f"  Using MAX_SEQ_LEN={MAX_SEQ_LEN} → {sum(l > MAX_SEQ_LEN for l in lengths)} truncated, "
      f"{sum(l < MAX_SEQ_LEN for l in lengths)} padded")
print()


# ======================================================================
# SECTION 3: LSTM From Scratch (Numpy Demo)
# ======================================================================
print("=" * 70)
print("SECTION 3: LSTM TEXT CLASSIFIER FROM SCRATCH (CONCEPTUAL)")
print("=" * 70)
print()
print("Full LSTM in numpy (showing the architecture before Keras):")
print()
print("  Step 1: Embedding lookup")
print("    [3, 47, 12, 0] → embedding_matrix[[3,47,12,0]] → (4, 64) matrix")
print()
print("  Step 2: LSTM forward pass")
print("    for each timestep t:")
print("        h_t, c_t = lstm_cell(x_t, h_{t-1}, c_{t-1})")
print("    output = h_final  # (64,)")
print()
print("  Step 3: Classification head")
print("    logit  = Dense(1)(output)        # (1,)")
print("    prob   = sigmoid(logit)          # probability of positive")
print()

# Quick numpy demonstration (tiny dimensions)
V_demo, D_demo, H_demo = 10, 4, 6

emb_demo  = np.random.randn(V_demo, D_demo) * 0.1
W_demo    = np.random.randn(D_demo + H_demo, 4 * H_demo) * 0.1
b_demo    = np.zeros(4 * H_demo)
W_out_demo= np.random.randn(H_demo, 1) * 0.1

seq_demo  = [2, 5, 1, 3, 7]   # integer sequence

h, c = np.zeros(H_demo), np.zeros(H_demo)
for tok in seq_demo:
    x   = emb_demo[tok]
    combined = np.concatenate([x, h])
    gates    = combined @ W_demo + b_demo
    f = 1 / (1 + np.exp(-gates[:H_demo]))
    i = 1 / (1 + np.exp(-gates[H_demo:2*H_demo]))
    g = np.tanh(gates[2*H_demo:3*H_demo])
    o = 1 / (1 + np.exp(-gates[3*H_demo:]))
    c = f * c + i * g
    h = o * np.tanh(c)

logit = h @ W_out_demo
prob  = 1 / (1 + np.exp(-logit[0]))
print(f"  Demo LSTM forward pass: seq={seq_demo} → prob={prob:.4f}")
print()


# ======================================================================
# SECTION 4: Keras LSTM Models
# ======================================================================
print("=" * 70)
print("SECTION 4: KERAS LSTM — MULTIPLE ARCHITECTURES")
print("=" * 70)
print()

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
    tf.random.set_seed(42)
    print(f"  TensorFlow {tf.__version__} available ✓")
    print()
except ImportError:
    print("  TensorFlow not installed: pip install tensorflow")
    print("  Showing architectures and expected results below.")
    print()


def print_model_summary_manual(name, layers_desc, params_k):
    print(f"  ── {name} ──")
    for desc in layers_desc:
        print(f"    {desc}")
    print(f"    Total params: ~{params_k}K")
    print()


if TF_AVAILABLE:
    VOCAB_SIZE = len(vocab)

    # ---- Model 1: Simple LSTM ----
    print("  Model 1: Simple LSTM")
    model1 = keras.Sequential([
        layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True, name="embedding"),
        layers.LSTM(64, name="lstm"),
        layers.Dense(32, activation="relu", name="dense"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid", name="output"),
    ], name="SimpleLSTM")

    model1.compile(optimizer=keras.optimizers.Adam(0.001),
                   loss="binary_crossentropy", metrics=["accuracy"])
    model1.summary()
    print()

    hist1 = model1.fit(
        X_train_seq, y_train_np,
        epochs=20, batch_size=8,
        validation_data=(X_val_seq, y_val_np),
        verbose=0
    )
    _, acc1 = model1.evaluate(X_val_seq, y_val_np, verbose=0)
    print(f"  Simple LSTM val accuracy: {acc1:.4f} ({acc1*100:.1f}%)")
    print()

    # ---- Model 2: Bidirectional LSTM ----
    print("  Model 2: Bidirectional LSTM")
    model2 = keras.Sequential([
        layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True, name="embedding"),
        layers.Bidirectional(layers.LSTM(64), name="bilstm"),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="BiLSTM")

    model2.compile(optimizer=keras.optimizers.Adam(0.001),
                   loss="binary_crossentropy", metrics=["accuracy"])
    hist2 = model2.fit(
        X_train_seq, y_train_np,
        epochs=20, batch_size=8,
        validation_data=(X_val_seq, y_val_np),
        verbose=0
    )
    _, acc2 = model2.evaluate(X_val_seq, y_val_np, verbose=0)
    print(f"  BiLSTM val accuracy: {acc2:.4f} ({acc2*100:.1f}%)")
    print()

    # ---- Model 3: LSTM with GlobalMaxPooling ----
    print("  Model 3: LSTM + GlobalMaxPooling1D")
    inputs3 = keras.Input(shape=(MAX_SEQ_LEN,))
    x3      = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=False)(inputs3)
    x3      = layers.LSTM(64, return_sequences=True)(x3)  # return ALL hidden states
    x3      = layers.GlobalMaxPooling1D()(x3)              # max over timesteps
    x3      = layers.Dense(32, activation="relu")(x3)
    x3      = layers.Dropout(0.3)(x3)
    out3    = layers.Dense(1, activation="sigmoid")(x3)
    model3  = keras.Model(inputs3, out3, name="LSTM_GlobalMaxPool")

    model3.compile(optimizer=keras.optimizers.Adam(0.001),
                   loss="binary_crossentropy", metrics=["accuracy"])
    hist3 = model3.fit(
        X_train_seq, y_train_np,
        epochs=20, batch_size=8,
        validation_data=(X_val_seq, y_val_np),
        verbose=0
    )
    _, acc3 = model3.evaluate(X_val_seq, y_val_np, verbose=0)
    print(f"  LSTM+GlobalMaxPool val accuracy: {acc3:.4f} ({acc3*100:.1f}%)")
    print()

    # ---- Model 4: GRU (comparison) ----
    print("  Model 4: GRU (faster alternative)")
    model4 = keras.Sequential([
        layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
        layers.Bidirectional(layers.GRU(64)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name="BiGRU")

    model4.compile(optimizer=keras.optimizers.Adam(0.001),
                   loss="binary_crossentropy", metrics=["accuracy"])
    hist4 = model4.fit(
        X_train_seq, y_train_np,
        epochs=20, batch_size=8,
        validation_data=(X_val_seq, y_val_np),
        verbose=0
    )
    _, acc4 = model4.evaluate(X_val_seq, y_val_np, verbose=0)
    print(f"  BiGRU val accuracy: {acc4:.4f} ({acc4*100:.1f}%)")
    print()

    model_accs = {
        "Simple\nLSTM":   acc1,
        "BiLSTM":         acc2,
        "LSTM+\nGMaxPool": acc3,
        "BiGRU":          acc4,
    }
    histories = {
        "Simple LSTM":     hist1,
        "BiLSTM":          hist2,
        "LSTM+GlobalMax":  hist3,
        "BiGRU":           hist4,
    }

else:
    print("  Architecture overview (runs after pip install tensorflow):")
    print()
    print_model_summary_manual("Simple LSTM", [
        "Embedding(vocab, 64)",
        "LSTM(64)",
        "Dense(32, relu)",
        "Dropout(0.3)",
        "Dense(1, sigmoid)",
    ], 140)
    print_model_summary_manual("BiLSTM", [
        "Embedding(vocab, 64)",
        "Bidirectional(LSTM(64)) → 128d",
        "Dense(32, relu)",
        "Dropout(0.3)",
        "Dense(1, sigmoid)",
    ], 165)
    print_model_summary_manual("LSTM + GlobalMaxPool", [
        "Embedding(vocab, 64)",
        "LSTM(64, return_sequences=True) → (seq_len, 64)",
        "GlobalMaxPooling1D() → (64,)",
        "Dense(32, relu) + Dropout",
        "Dense(1, sigmoid)",
    ], 145)

    model_accs = {
        "Simple\nLSTM": 0.82, "BiLSTM": 0.85,
        "LSTM+\nGMaxPool": 0.83, "BiGRU": 0.84
    }


# ======================================================================
# SECTION 5: TF-IDF vs LSTM Comparison
# ======================================================================
print("=" * 70)
print("SECTION 5: TF-IDF + LOGREG vs LSTM COMPARISON")
print("=" * 70)
print()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import Pipeline

    tfidf_lr = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=500, C=5.0)),
    ])
    tfidf_lr.fit(X_train_raw, y_train)
    tfidf_acc = tfidf_lr.score(X_val_raw, y_val)

    print(f"  TF-IDF + LogReg val accuracy: {tfidf_acc:.4f} ({tfidf_acc*100:.1f}%)")
    if TF_AVAILABLE:
        best_lstm = max(model_accs, key=model_accs.get)
        best_acc  = model_accs[best_lstm]
        print(f"  Best LSTM ({best_lstm.strip()}): {best_acc:.4f} ({best_acc*100:.1f}%)")
        gain = best_acc - tfidf_acc
        if gain > 0:
            print(f"  LSTM gain: +{gain*100:.1f}%")
        else:
            print(f"  TF-IDF actually better by {-gain*100:.1f}% on this small dataset!")
    print()
    print("  Note: with only 40 training examples, TF-IDF often wins.")
    print("  LSTM excels when N > 5,000 examples (enough to learn embeddings)")
    print()

except ImportError:
    tfidf_acc = 0.85
    print("  sklearn not installed — skipping TF-IDF comparison")
    print()


# ======================================================================
# SECTION 6: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 6: VISUALIZATIONS")
print("=" * 70)
print()


# --- PLOT 1: Sequence length distribution and padding visualization ---
print("Generating: Sequence padding and vocabulary analysis...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("LSTM Text Classifier: Data Preparation Analysis",
             fontsize=14, fontweight="bold")

# Length distribution
all_lengths = [len(simple_tokenize(t)) for t in texts]
axes[0].hist(all_lengths, bins=15, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(x=MAX_SEQ_LEN, color="red", linewidth=2.5, linestyle="--",
                label=f"MAX_LEN={MAX_SEQ_LEN}")
axes[0].axvline(x=np.mean(all_lengths), color="orange", linewidth=2,
                label=f"Mean={np.mean(all_lengths):.1f}")
axes[0].set_xlabel("Sequence Length (tokens)"); axes[0].set_ylabel("Count")
axes[0].set_title("Token Sequence Length Distribution", fontsize=11, fontweight="bold")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Vocabulary frequency distribution (Zipf)
counter_all = Counter()
for t in texts:
    counter_all.update(simple_tokenize(t))

sorted_freqs = sorted(counter_all.values(), reverse=True)
axes[1].loglog(range(1, len(sorted_freqs)+1), sorted_freqs, "b-o",
               markersize=4, linewidth=2, label="Observed")
zipf = [sorted_freqs[0] / r for r in range(1, len(sorted_freqs)+1)]
axes[1].loglog(range(1, len(sorted_freqs)+1), zipf, "r--",
               linewidth=2, label="Zipf (1/rank)")
axes[1].set_xlabel("Word Rank (log)"); axes[1].set_ylabel("Frequency (log)")
axes[1].set_title("Vocabulary Frequency (Zipf's Law)", fontsize=11, fontweight="bold")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=MAX_VOCAB, color="orange", linestyle="--", linewidth=1.5,
                label=f"Vocab cutoff ({MAX_VOCAB})")

# Padded sequence visualization
fig_seq, ax_seq = plt.subplots(figsize=(1, 1))  # placeholder
# Show 10 sequences as a heatmap
sample_seqs = X_train_seq[:10].astype(float)
sample_seqs[sample_seqs == 0] = np.nan  # PAD tokens → grey

im = axes[2].imshow(sample_seqs, cmap="Blues", aspect="auto",
                    vmin=0, vmax=len(vocab))
axes[2].set_xlabel("Position (timestep)"); axes[2].set_ylabel("Review #")
axes[2].set_title(f"Padded Integer Sequences (10 examples)\nMax length={MAX_SEQ_LEN}",
                  fontsize=11, fontweight="bold")
plt.colorbar(im, ax=axes[2], shrink=0.8, label="Word ID")

# Mark padding
for i in range(10):
    seq = X_train_seq[i]
    pad_start = np.where(seq == 0)[0]
    if len(pad_start) > 0:
        axes[2].axvline(x=pad_start[0] - 0.5, color="red", linewidth=0.8, alpha=0.5)

plt.close(fig_seq)
plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "data_preparation.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: data_preparation.png")


# --- PLOT 2: Training history comparison ---
print("Generating: Model training histories...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("LSTM Architectures: Training History Comparison",
             fontsize=14, fontweight="bold")

if TF_AVAILABLE:
    colors_hist = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6"]
    for (name, hist), color in zip(histories.items(), colors_hist):
        axes[0].plot(hist.history["val_accuracy"], linewidth=2, label=name, color=color)
        axes[1].plot(hist.history["val_loss"],     linewidth=2, label=name, color=color)
else:
    # Synthetic training curves
    epochs_s = np.arange(1, 21)
    for name, final_acc, color in zip(
        ["Simple LSTM", "BiLSTM", "LSTM+GlobalMax", "BiGRU"],
        [0.82, 0.85, 0.83, 0.84],
        ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6"]
    ):
        # Simulate convergence curve
        acc_curve = final_acc * (1 - np.exp(-0.3 * epochs_s)) + np.random.randn(20) * 0.02
        acc_curve = np.clip(acc_curve, 0, 1)
        loss_curve = -np.log(np.clip(acc_curve, 0.01, 0.99))

        axes[0].plot(epochs_s, acc_curve,  linewidth=2, label=name, color=color)
        axes[1].plot(epochs_s, loss_curve, linewidth=2, label=name, color=color)

axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation Accuracy")
axes[0].set_title("Validation Accuracy over Training"); axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3); axes[0].set_ylim(0, 1.05)

axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss")
axes[1].set_title("Validation Loss over Training"); axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "training_histories.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: training_histories.png")


# --- PLOT 3: Model comparison + LSTM architecture diagram ---
print("Generating: Architecture diagram and model comparison...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("LSTM Text Classifier: Architecture and Performance",
             fontsize=14, fontweight="bold")

# Architecture diagram
ax = axes[0]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("BiLSTM Text Classifier Architecture", fontsize=12, fontweight="bold")

arch_steps = [
    ("Text Input\n\"I love this\"",                 0.5, 0.90, "#E74C3C",  0.40, 0.09),
    ("Integer Sequence\n[3, 47, 12, 0, ...]",       0.5, 0.76, "#E67E22",  0.40, 0.09),
    ("Embedding Layer\n(vocab, 64) → (seq, 64)",     0.5, 0.62, "#F1C40F",  0.40, 0.09),
    ("Bidirectional LSTM(64)\nforward + backward → 128d", 0.5, 0.47, "#2ECC71", 0.50, 0.09),
    ("Dense(32, ReLU)\n+ Dropout(0.3)",              0.5, 0.33, "#3498DB",  0.40, 0.09),
    ("Dense(1, sigmoid)\n→ prob",                    0.5, 0.19, "#9B59B6",  0.30, 0.09),
]

for label, xc, yc, color, bw, bh in arch_steps:
    rect = plt.Rectangle((xc - bw/2, yc - bh/2), bw, bh,
                          facecolor=color, alpha=0.80, edgecolor="white", linewidth=2)
    ax.add_patch(rect)
    ax.text(xc, yc, label, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white")

for i in range(len(arch_steps) - 1):
    yc_curr = arch_steps[i][2]   - arch_steps[i][5]/2
    yc_next = arch_steps[i+1][2] + arch_steps[i+1][5]/2
    ax.annotate("", xy=(0.5, yc_next + 0.002), xytext=(0.5, yc_curr - 0.002),
                arrowprops=dict(arrowstyle="->", color="#555", lw=2))

ax.text(0.5, 0.08, f"→ 'Positive' (prob ≥ 0.5)\n→ 'Negative' (prob < 0.5)",
        ha="center", fontsize=9, color="#555",
        bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

# Model comparison bar chart
model_names_plot = list(model_accs.keys())
acc_vals         = [model_accs[k] for k in model_names_plot]
colors_bar       = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6"]

bars = axes[1].bar(model_names_plot, [a*100 for a in acc_vals],
                   color=colors_bar, edgecolor="white", linewidth=2, width=0.55)
try:
    axes[1].axhline(y=tfidf_acc * 100, color="darkorange", linewidth=2.5, linestyle="--",
                    label=f"TF-IDF+LR: {tfidf_acc:.1%}")
    axes[1].legend(fontsize=10)
except NameError:
    pass

axes[1].set_ylim(0, 110)
axes[1].set_ylabel("Validation Accuracy (%)")
axes[1].set_title("LSTM Architecture Comparison\nvs TF-IDF Baseline", fontsize=11, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)
for bar, acc in zip(bars, acc_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{acc:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "architecture_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: architecture_comparison.png")


# ======================================================================
# SECTION 7: LSTM CELL ARCHITECTURE DIAGRAM
# ======================================================================
print("Generating: LSTM Cell Architecture diagram...")

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 9),
                                         gridspec_kw={"width_ratios": [3, 2]})
fig.patch.set_facecolor('#0f0f1a')
for ax in (ax_left, ax_right):
    ax.set_facecolor('#0f0f1a')
    ax.axis('off')

fig.suptitle("LSTM Cell Architecture & Text Classification Pipeline",
             fontsize=14, fontweight='bold', color='white', y=0.97)

# ── LEFT PANEL: LSTM cell internals ──────────────────────────────────
ax_left.set_xlim(0, 8.4)
ax_left.set_ylim(0, 9)
ax_left.set_title("LSTM Cell Internals", color='white', fontsize=12,
                  fontweight='bold', pad=10)

# Cell State horizontal line (top)
cell_y = 7.5
ax_left.plot([0.5, 7.9], [cell_y, cell_y], color='#F0E68C', linewidth=3, zorder=3)
ax_left.annotate('', xy=(7.9, cell_y), xytext=(7.5, cell_y),
                 arrowprops=dict(arrowstyle='->', color='#F0E68C', lw=2.5))

# Cell State label
ax_left.text(0.2, cell_y, 'C_t-1', color='#F0E68C', fontsize=9,
             fontweight='bold', va='center', ha='right')
ax_left.text(8.1, cell_y, 'C_t', color='#F0E68C', fontsize=9,
             fontweight='bold', va='center', ha='left')
ax_left.text(4.2, cell_y + 0.35, 'Cell State (long-term memory)',
             color='#F0E68C', fontsize=8, ha='center', style='italic')

# Multiply circle (⊗) on cell state — after forget gate
mul_x1, mul_y1 = 2.2, cell_y
circ_mul1 = Circle((mul_x1, mul_y1), 0.28, color='#888888', zorder=4)
ax_left.add_patch(circ_mul1)
ax_left.text(mul_x1, mul_y1, '×', color='white', fontsize=14, ha='center',
             va='center', fontweight='bold', zorder=5)

# Add circle (⊕) on cell state — after input gate
add_x, add_y = 5.0, cell_y
circ_add = Circle((add_x, add_y), 0.28, color='#888888', zorder=4)
ax_left.add_patch(circ_add)
ax_left.text(add_x, add_y, '+', color='white', fontsize=16, ha='center',
             va='center', fontweight='bold', zorder=5)

# Gate definitions: (label, role_text, x_center, y_center, color)
gates = [
    ('σ\nForget Gate',   'Forget Gate:\nwhat to erase',  1.5, 4.8, '#C0392B'),
    ('σ\nInput Gate',    'Input Gate:\nwhat to add',      3.5, 4.8, '#2980B9'),
    ('tanh\nCell Update','Cell Update:\nnew candidates',  5.2, 4.8, '#27AE60'),
    ('σ\nOutput Gate',   'Output Gate:\nwhat to expose',  6.8, 4.8, '#8E44AD'),
]

gate_box_w, gate_box_h = 1.15, 1.2
for (lbl, role, gx, gy, gcol) in gates:
    bbox = FancyBboxPatch((gx - gate_box_w / 2, gy - gate_box_h / 2),
                          gate_box_w, gate_box_h,
                          boxstyle="round,pad=0.08",
                          facecolor=gcol, edgecolor='white',
                          linewidth=1.8, zorder=4)
    ax_left.add_patch(bbox)
    ax_left.text(gx, gy + 0.1, lbl, color='white', fontsize=7.5,
                 ha='center', va='center', fontweight='bold', zorder=5)
    ax_left.text(gx, gy - 0.78, role, color=gcol, fontsize=6.5,
                 ha='center', va='top', style='italic', zorder=5)

# h_t-1 input arrow (from left, horizontal)
h_input_y = 3.5
ax_left.annotate('', xy=(1.0, h_input_y), xytext=(0.1, h_input_y),
                 arrowprops=dict(arrowstyle='->', color='#00BFFF', lw=2))
ax_left.text(0.05, h_input_y, 'h_t-1', color='#00BFFF', fontsize=8.5,
             fontweight='bold', va='center', ha='right')

# x_t input arrow (from bottom, vertical) for each gate
x_t_positions = [1.5, 3.5, 5.2, 6.8]
for xp in x_t_positions:
    ax_left.annotate('', xy=(xp, 4.2), xytext=(xp, 2.9),
                     arrowprops=dict(arrowstyle='->', color='#FF8C00', lw=1.8))

ax_left.text(4.2, 2.55, 'x_t  (current input token embedding)',
             color='#FF8C00', fontsize=8.5, ha='center', fontweight='bold')
ax_left.annotate('', xy=(4.2, 2.8), xytext=(4.2, 2.65),
                 arrowprops=dict(arrowstyle='->', color='#FF8C00', lw=1.8))

# h_t-1 fan-out to all gates
for xp in x_t_positions:
    ax_left.plot([1.0, xp], [h_input_y, h_input_y], color='#00BFFF',
                 linewidth=1.4, linestyle='--', alpha=0.7, zorder=2)
    ax_left.annotate('', xy=(xp, 4.2), xytext=(xp, h_input_y),
                     arrowprops=dict(arrowstyle='->', color='#00BFFF', lw=1.4))

# Forget gate → multiply circle on cell state
ax_left.annotate('', xy=(mul_x1, cell_y - 0.28), xytext=(1.5, 4.2 + 0.0),
                 arrowprops=dict(arrowstyle='->', color='#C0392B', lw=1.8))

# Input gate + cell update → add circle on cell state (tanh product first)
# draw a small tanh × input product dot
prod_x, prod_y = 4.2, 6.2
circ_prod = Circle((prod_x, prod_y), 0.22, color='#555577', zorder=4)
ax_left.add_patch(circ_prod)
ax_left.text(prod_x, prod_y, '×', color='white', fontsize=12, ha='center',
             va='center', fontweight='bold', zorder=5)
ax_left.annotate('', xy=(prod_x, prod_y - 0.22), xytext=(3.5, 4.2 + 0.0),
                 arrowprops=dict(arrowstyle='->', color='#2980B9', lw=1.5))
ax_left.annotate('', xy=(prod_x, prod_y - 0.22), xytext=(5.2, 4.2 + 0.0),
                 arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5))
ax_left.annotate('', xy=(add_x - 0.28, add_y), xytext=(prod_x + 0.22, prod_y),
                 arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=1.5))

# Add circle → C_t (continue along cell state line)
ax_left.plot([add_x + 0.28, 7.6], [add_y, add_y], color='#F0E68C',
             linewidth=3, zorder=3)

# Output gate → h_t (tanh of C_t)
tanh_x, tanh_y = 6.8, 6.2
circ_tanh = Circle((tanh_x, tanh_y), 0.28, color='#555566', zorder=4)
ax_left.add_patch(circ_tanh)
ax_left.text(tanh_x, tanh_y, 'tanh', color='white', fontsize=6.5, ha='center',
             va='center', fontweight='bold', zorder=5)
# C_t flows into tanh
ax_left.annotate('', xy=(tanh_x, tanh_y + 0.28), xytext=(tanh_x, cell_y - 0.28),
                 arrowprops=dict(arrowstyle='->', color='#F0E68C', lw=1.8))
# Output gate × tanh(C_t) → h_t
ax_left.annotate('', xy=(tanh_x, tanh_y - 0.22), xytext=(6.8, 5.4),
                 arrowprops=dict(arrowstyle='->', color='#8E44AD', lw=1.5))
# h_t output downward
ax_left.annotate('', xy=(tanh_x, 1.6), xytext=(tanh_x, tanh_y - 0.28),
                 arrowprops=dict(arrowstyle='->', color='#00BFFF', lw=2.5))
ax_left.text(tanh_x + 0.2, 1.4, 'h_t\n(new hidden state)', color='#00BFFF',
             fontsize=7.5, fontweight='bold', ha='left', va='top')

# Legend box
legend_items = [
    ('#C0392B', 'Forget Gate (σ): erases old memory'),
    ('#2980B9', 'Input Gate (σ): selects new info'),
    ('#27AE60', 'Cell Update (tanh): new candidates'),
    ('#8E44AD', 'Output Gate (σ): filters h_t output'),
    ('#F0E68C', 'Cell State C_t: long-term memory highway'),
    ('#00BFFF', 'Hidden State h_t: short-term / output'),
]
for k, (col, txt) in enumerate(legend_items):
    ax_left.plot(0.35, 2.0 - k * 0.28, 's', color=col, markersize=7)
    ax_left.text(0.55, 2.0 - k * 0.28, txt, color='#cccccc', fontsize=6.5,
                 va='center')

# ── RIGHT PANEL: LSTM sequence processing for text ───────────────────
ax_right.set_xlim(0, 5.6)
ax_right.set_ylim(0, 9)
ax_right.set_title("Text Sequence Processing", color='white', fontsize=12,
                   fontweight='bold', pad=10)

words = ['"The"', '"movie"', '"was"', '"great"']
word_colors = ['#1a6b8a', '#1a6b8a', '#1a6b8a', '#1a8a4a']
x_positions = [0.75, 1.95, 3.15, 4.35]
word_y = 1.2

# Draw word input boxes
for word, wx, wc in zip(words, x_positions, word_colors):
    wb = FancyBboxPatch((wx - 0.48, word_y - 0.28), 0.96, 0.56,
                        boxstyle="round,pad=0.06",
                        facecolor=wc, edgecolor='#4488aa',
                        linewidth=1.5, zorder=4)
    ax_right.add_patch(wb)
    ax_right.text(wx, word_y, word, color='white', fontsize=7.5,
                  ha='center', va='center', fontweight='bold', zorder=5)

ax_right.text(2.55, 0.65, 'Input token embeddings  (x_t)',
              color='#FF8C00', fontsize=7.5, ha='center')

# LSTM cells stacked horizontally
lstm_y = 3.5
lstm_w, lstm_h = 0.9, 0.7
for i, (wx, wc) in enumerate(zip(x_positions, ['#3b3b6b'] * 4)):
    # Arrow from word up into LSTM
    ax_right.annotate('', xy=(wx, lstm_y - lstm_h / 2),
                      xytext=(wx, word_y + 0.28),
                      arrowprops=dict(arrowstyle='->', color='#FF8C00', lw=1.5))
    lb = FancyBboxPatch((wx - lstm_w / 2, lstm_y - lstm_h / 2),
                        lstm_w, lstm_h,
                        boxstyle="round,pad=0.07",
                        facecolor='#3b3b7a', edgecolor='#7777cc',
                        linewidth=1.8, zorder=4)
    ax_right.add_patch(lb)
    ax_right.text(wx, lstm_y, 'LSTM', color='white', fontsize=7.5,
                  ha='center', va='center', fontweight='bold', zorder=5)
    ax_right.text(wx, lstm_y - 0.48, f't={i}', color='#aaaacc', fontsize=6.5,
                  ha='center', va='top', zorder=5)

# Hidden state arrows passing rightward between LSTM cells
for i in range(len(x_positions) - 1):
    ax_right.annotate('', xy=(x_positions[i + 1] - lstm_w / 2, lstm_y),
                      xytext=(x_positions[i] + lstm_w / 2, lstm_y),
                      arrowprops=dict(arrowstyle='->', color='#00BFFF', lw=2))
ax_right.text(2.55, lstm_y + 0.55, 'h_t passes right →',
              color='#00BFFF', fontsize=7, ha='center', style='italic')

# Final hidden state arrow upward
final_x = x_positions[-1]
ax_right.annotate('', xy=(final_x, lstm_y + lstm_h / 2 + 0.9),
                  xytext=(final_x, lstm_y + lstm_h / 2),
                  arrowprops=dict(arrowstyle='->', color='#00BFFF', lw=2))

# Dense layer box
dense_y = 5.5
db = FancyBboxPatch((final_x - 0.65, dense_y - 0.3), 1.3, 0.6,
                    boxstyle="round,pad=0.07",
                    facecolor='#2c6e49', edgecolor='#55bb88',
                    linewidth=1.8, zorder=4)
ax_right.add_patch(db)
ax_right.text(final_x, dense_y, 'Dense(32)\n+ Dropout', color='white',
              fontsize=7, ha='center', va='center', fontweight='bold', zorder=5)

ax_right.annotate('', xy=(final_x, dense_y + 0.55),
                  xytext=(final_x, dense_y + 0.3),
                  arrowprops=dict(arrowstyle='->', color='#55bb88', lw=2))

# Sigmoid output box
sig_y = 6.7
sb = FancyBboxPatch((final_x - 0.65, sig_y - 0.3), 1.3, 0.6,
                    boxstyle="round,pad=0.07",
                    facecolor='#7b2d8b', edgecolor='#cc66dd',
                    linewidth=1.8, zorder=4)
ax_right.add_patch(sb)
ax_right.text(final_x, sig_y, 'Dense(1)\nSigmoid σ', color='white',
              fontsize=7, ha='center', va='center', fontweight='bold', zorder=5)

ax_right.annotate('', xy=(final_x, sig_y + 0.55),
                  xytext=(final_x, sig_y + 0.3),
                  arrowprops=dict(arrowstyle='->', color='#cc66dd', lw=2))

# Sentiment output box
out_y = 8.0
ob = FancyBboxPatch((final_x - 0.8, out_y - 0.4), 1.6, 0.8,
                    boxstyle="round,pad=0.08",
                    facecolor='#1a4f1a', edgecolor='#44dd44',
                    linewidth=2, zorder=4)
ax_right.add_patch(ob)
ax_right.text(final_x, out_y, 'POSITIVE\n(prob = 0.93)', color='#88ff88',
              fontsize=8, ha='center', va='center', fontweight='bold', zorder=5)

ax_right.text(2.55, 8.55, 'Full Pipeline: Tokens \u2192 LSTM \u2192 Sentiment',
              color='#aaaaaa', fontsize=7.5, ha='center', style='italic')

plt.savefig(os.path.join(_VISUALS_DIR, '04_lstm_cell_architecture.png'),
            dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("   Saved: 04_lstm_cell_architecture.png")

print()
print("=" * 70)
print("NLP ALGORITHM 3: LSTM TEXT CLASSIFIER COMPLETE!")
print("=" * 70)
print()
print("What you built:")
print("  ✓ Text → integer sequence → padded matrix pipeline")
print("  ✓ Vocabulary built from training data (MAX_VOCAB most common words)")
print("  ✓ Keras Embedding layer (trainable word vectors from scratch)")
print("  ✓ Simple LSTM, BiLSTM, LSTM+GlobalMaxPool, BiGRU in Keras")
print("  ✓ Compared all models + TF-IDF baseline")
print()
print("Key design choices:")
print("  mask_zero=True     → Embedding ignores PAD tokens in LSTM")
print("  return_sequences=True → pass ALL hidden states to next layer")
print("  Bidirectional      → forward + backward context")
print("  GlobalMaxPooling1D → take max feature across all timesteps")
print()
print("4 Visualizations saved to: ../visuals/lstm_text_classifier/")
print("  1. data_preparation.png         — length dist + vocab + padded sequences")
print("  2. training_histories.png       — val accuracy/loss per architecture")
print("  3. architecture_comparison.png  — BiLSTM diagram + accuracy bar chart")
print("  4. lstm_cell_architecture.png   — LSTM cell internals + text pipeline")
print()
print("Next: Algorithm 4 → Named Entity Recognition (NER)")
