"""
🧠 NLP — Math Foundation 3: Word Embeddings
============================================

Learning Objectives:
  1. Understand why BoW/TF-IDF fails to capture word meaning
  2. Grasp the distributional hypothesis: "you shall know a word by its company"
  3. Implement a simplified Word2Vec Skip-gram training loop from scratch
  4. Explore embedding space: cosine similarity, analogies (king - man + woman = queen)
  5. Load and use pretrained GloVe embeddings
  6. Visualize embedding space with PCA and t-SNE dimensionality reduction
  7. Understand subword embeddings (FastText) and their advantage over word-level

YouTube Resources:
  ⭐ StatQuest — Word Embedding https://www.youtube.com/watch?v=viZrOnJclY0
  ⭐ Andrej Karpathy — makemore (character embeddings) https://www.youtube.com/watch?v=PaCmpygFfXo
  📚 Word2Vec paper (2013) https://arxiv.org/abs/1301.3781

Time Estimate: 65 min
Difficulty: Intermediate
Prerequisites: 01_text_processing.py, 02_bag_of_words_tfidf.py, numpy dot products
Key Concepts: distributional hypothesis, embedding, Word2Vec, Skip-gram, GloVe, t-SNE
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
import os

os.makedirs("../visuals/03_word_embeddings", exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🧠 NLP MATH FOUNDATION 3: WORD EMBEDDINGS")
print("=" * 70)
print()
print("TF-IDF problem: words are points in space with NO semantic relationship.")
print()
print("  Vocabulary of 50,000 words → 50,000-dimensional one-hot vectors")
print("  'cat'  = [1, 0, 0, 0, ..., 0]   (position 7 is 1)")
print("  'dog'  = [0, 0, 1, 0, ..., 0]   (position 9 is 1)")
print("  'car'  = [0, 0, 0, 1, ..., 0]   (position 12 is 1)")
print()
print("  cosine_similarity(cat, dog) = 0.0  ← they look identical!")
print("  cosine_similarity(cat, car) = 0.0  ← they also look identical!")
print()
print("  But WE know: cat ≈ dog (both animals), cat ≠ car (completely different)")
print()
print("Word Embeddings: map words to DENSE, LOW-DIMENSIONAL vectors")
print("that CAPTURE MEANING through geometric relationships.")
print()
print("  'cat'  →  [0.32, -0.14,  0.78, ...]   # 300-d vector")
print("  'dog'  →  [0.29, -0.11,  0.81, ...]   # similar to cat!")
print("  'car'  →  [-0.82, 0.65, -0.12, ...]   # different from cat")
print()
print("  cosine_similarity(cat, dog) ≈ 0.92  ✓")
print("  cosine_similarity(cat, car) ≈ 0.07  ✓")
print()
print("Even better — arithmetic in embedding space MEANS something:")
print("  king - man + woman ≈ queen")
print("  Paris - France + Italy ≈ Rome")
print()


# ======================================================================
# SECTION 1: The Distributional Hypothesis
# ======================================================================
print("=" * 70)
print("SECTION 1: THE DISTRIBUTIONAL HYPOTHESIS")
print("=" * 70)
print()
print("J.R. Firth (1957): 'You shall know a word by the company it keeps.'")
print()
print("Words that appear in SIMILAR CONTEXTS have SIMILAR MEANINGS.")
print()
print("  Contexts for 'cat':")
print("    'the ___ sat on the mat'")
print("    'feed the ___'")
print("    'the ___ chased the mouse'")
print("    'my ___ is sleeping'")
print()
print("  Contexts for 'dog':")
print("    'the ___ sat by the door'")
print("    'feed the ___'")
print("    'the ___ chased the ball'")
print("    'my ___ is sleeping'")
print()
print("  Almost identical contexts → very similar vectors")
print()
print("Word2Vec learns this automatically from raw text:")
print("  • Slide a context WINDOW over the text")
print("  • Train a model to PREDICT context words from center word (Skip-gram)")
print("  • OR predict center word from context words (CBOW)")
print("  • The WEIGHTS of the trained model = word embeddings!")
print()


# ======================================================================
# SECTION 2: Word2Vec Skip-gram From Scratch
# ======================================================================
print("=" * 70)
print("SECTION 2: WORD2VEC SKIP-GRAM FROM SCRATCH")
print("=" * 70)
print()
print("Skip-gram architecture:")
print("  Input:  one-hot vector of center word  (V × 1)")
print("  W_in:   embedding matrix               (V × D)  ← word embeddings")
print("  h:      hidden layer = W_in[word_idx]  (D,)     ← the embedding!")
print("  W_out:  output matrix                  (D × V)")
print("  output: softmax scores over vocabulary (V,)     ← context word probs")
print()
print("  Training: center word → predict surrounding words")
print("  Gradient updates move similar words' vectors closer together")
print()

# Small corpus for demonstration
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat chased the mouse",
    "the dog chased the ball",
    "cats and dogs are friends",
    "the mouse hid under the mat",
    "the ball rolled off the log",
    "cats like fish and dogs like bones",
    "the cat and the dog played together",
    "a happy cat is a sleepy cat",
]

# Tokenize
all_tokens = []
for sentence in corpus:
    all_tokens.extend(sentence.lower().split())

token_freq = Counter(all_tokens)
vocab_list = sorted(token_freq.keys())
vocab      = {word: i for i, word in enumerate(vocab_list)}
idx2word   = {i: w for w, i in vocab.items()}
V          = len(vocab)
D          = 8   # embedding dimension (small for demo)

print(f"  Corpus: {len(corpus)} sentences")
print(f"  Vocabulary: {V} words, embedding dim D={D}")
print(f"  Vocab: {vocab_list}")
print()

# Build skip-gram training pairs (center word, context word)
WINDOW = 2

def build_skipgram_pairs(tokenized_corpus, vocab, window=2):
    pairs = []
    for sentence in tokenized_corpus:
        tokens = sentence.lower().split()
        for i, word in enumerate(tokens):
            if word not in vocab:
                continue
            center_id = vocab[word]
            # Context window
            for delta in range(-window, window + 1):
                if delta == 0:
                    continue
                j = i + delta
                if 0 <= j < len(tokens) and tokens[j] in vocab:
                    pairs.append((center_id, vocab[tokens[j]]))
    return pairs


training_pairs = build_skipgram_pairs(corpus, vocab, window=WINDOW)
print(f"  Training pairs: {len(training_pairs)}")
print(f"  Sample pairs (center_word, context_word):")
for ctr, ctx in training_pairs[:6]:
    print(f"    ({idx2word[ctr]!r}, {idx2word[ctx]!r})")
print()


class Word2VecSkipGram:
    """
    Minimal Word2Vec Skip-gram with negative sampling (simplified).
    Uses binary cross-entropy: positive pair → high score, negative → low.
    """

    def __init__(self, vocab_size, embedding_dim, lr=0.05):
        self.V  = vocab_size
        self.D  = embedding_dim
        self.lr = lr
        # Xavier initialization
        self.W_in  = np.random.randn(vocab_size, embedding_dim) * 0.1  # input  embeddings
        self.W_out = np.random.randn(vocab_size, embedding_dim) * 0.1  # output embeddings

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -10, 10)))

    def forward(self, center_id, context_id, negative_ids):
        """Returns loss and stores gradients."""
        h = self.W_in[center_id]               # center embedding

        # Positive sample score
        pos_score = self.sigmoid(h @ self.W_out[context_id])
        pos_loss  = -np.log(pos_score + 1e-9)

        # Negative samples score
        neg_scores = self.sigmoid(-self.W_out[negative_ids] @ h)
        neg_loss   = -np.log(neg_scores + 1e-9).mean()

        loss = pos_loss + neg_loss

        # Gradients (for SGD update)
        grad_h = (pos_score - 1) * self.W_out[context_id]
        grad_h += (-neg_scores[:, np.newaxis] * self.W_out[negative_ids]).mean(axis=0)

        self.grad_center  = grad_h
        self.grad_context = (pos_score - 1) * h
        self.grad_neg     = -neg_scores[:, np.newaxis] * h[np.newaxis, :] / len(negative_ids)

        return loss

    def update(self, center_id, context_id, negative_ids):
        loss = self.forward(center_id, context_id, negative_ids)
        self.W_in[center_id]      -= self.lr * self.grad_center
        self.W_out[context_id]    -= self.lr * self.grad_context
        for ni, neg_id in enumerate(negative_ids):
            self.W_out[neg_id]    -= self.lr * self.grad_neg[ni]
        return loss

    def get_embedding(self, word_id):
        return self.W_in[word_id]

    def similarity(self, word1, word2, vocab):
        v1 = self.W_in[vocab[word1]]
        v2 = self.W_in[vocab[word2]]
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

    def most_similar(self, word, vocab, idx2word, top_n=5):
        query = self.W_in[vocab[word]]
        query = query / (np.linalg.norm(query) + 1e-9)
        sims  = self.W_in @ query / (np.linalg.norm(self.W_in, axis=1) + 1e-9)
        top   = sims.argsort()[::-1]
        return [(idx2word[i], sims[i]) for i in top if i != vocab[word]][:top_n]


# Train
model = Word2VecSkipGram(V, D, lr=0.08)
N_EPOCHS  = 500
N_NEG     = 5
all_word_ids = list(range(V))

losses = []
print(f"  Training Skip-gram (epochs={N_EPOCHS}, neg_samples={N_NEG})...")
for epoch in range(N_EPOCHS):
    epoch_loss = 0
    np.random.shuffle(training_pairs)
    for center_id, context_id in training_pairs:
        # Negative sampling — pick random words excluding center and context
        neg_ids = np.random.choice(
            [i for i in all_word_ids if i != center_id and i != context_id],
            size=N_NEG, replace=False
        )
        epoch_loss += model.update(center_id, context_id, neg_ids)
    losses.append(epoch_loss / len(training_pairs))

print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss:   {losses[-1]:.4f}")
print()

# Inspect learned similarities
print("  Learned word similarities:")
word_pairs = [
    ("cat",  "dog"),
    ("cat",  "mat"),
    ("cat",  "mouse"),
    ("dog",  "ball"),
    ("mat",  "log"),
    ("cat",  "ball"),
]
print(f"  {'Word 1':<12} {'Word 2':<12} {'Cosine Sim':>12}")
print(f"  {'─'*12} {'─'*12} {'─'*12}")
for w1, w2 in word_pairs:
    sim = model.similarity(w1, w2, vocab)
    bar = "█" * int((sim + 1) * 5)
    print(f"  {w1:<12} {w2:<12} {sim:>10.4f}  {bar}")
print()

# Most similar words
for target_word in ["cat", "dog", "mat"]:
    similar = model.most_similar(target_word, vocab, idx2word, top_n=4)
    print(f"  Words most similar to '{target_word}':")
    for word, sim in similar:
        print(f"    {word:<12}: {sim:.4f}")
    print()


# ======================================================================
# SECTION 3: Pretrained Embeddings (GloVe / Word2Vec)
# ======================================================================
print("=" * 70)
print("SECTION 3: PRETRAINED EMBEDDINGS (GLOVE / WORD2VEC)")
print("=" * 70)
print()
print("Training Word2Vec from scratch on a real corpus takes days.")
print("Instead, use PRETRAINED embeddings trained on billions of words.")
print()
print("Available pretrained embeddings:")
print("  • Word2Vec (Google): 3M words, 300d — trained on Google News")
print("  • GloVe (Stanford): 400k-2.2M words, 50/100/200/300d — trained on Wikipedia+Twitter")
print("  • FastText (Meta):  1M+ words — includes subword info")
print("  • GPT-2 embeddings: subword, contextual (changes with context!)")
print()
print("Key insight: GloVe/Word2Vec embeddings are STATIC")
print("  'bank' (financial) vs 'bank' (river) → SAME vector")
print("  BERT/GPT embeddings are CONTEXTUAL — different for each sentence")
print()

# Simulate pretrained embeddings for demonstration
# (Real GloVe would be loaded from file — too large to download here)

print("  Simulating GloVe-like embeddings for a demonstration vocabulary...")
print("  (In practice: gensim.downloader.load('glove-wiki-gigaword-100'))")
print()

# Manually crafted 4D embeddings encoding semantic dimensions
# Dims: [animal, vehicle, action, positive]
DEMO_EMBEDDINGS = {
    "cat":        np.array([ 0.8,  -0.7,  0.2,  0.4]),
    "dog":        np.array([ 0.9,  -0.6,  0.3,  0.5]),
    "kitten":     np.array([ 0.85, -0.65, 0.1,  0.6]),
    "puppy":      np.array([ 0.88, -0.62, 0.15, 0.7]),
    "horse":      np.array([ 0.7,  -0.5,  0.5,  0.3]),
    "car":        np.array([-0.5,   0.9, -0.2, -0.1]),
    "truck":      np.array([-0.4,   0.85, -0.1, -0.2]),
    "bicycle":    np.array([-0.3,   0.7,  0.3,  0.1]),
    "run":        np.array([ 0.1,  -0.1,  0.9, -0.1]),
    "walk":       np.array([ 0.0,   0.0,  0.8,  0.2]),
    "jump":       np.array([ 0.2,  -0.1,  0.85, 0.1]),
    "king":       np.array([ 0.0,   0.5,  0.1,  0.6]),
    "queen":      np.array([-0.1,   0.5,  0.0,  0.7]),
    "man":        np.array([ 0.2,   0.0,  0.2,  0.3]),
    "woman":      np.array([-0.2,   0.0,  0.1,  0.4]),
    "good":       np.array([ 0.0,   0.0,  0.0,  0.9]),
    "great":      np.array([ 0.0,   0.0,  0.0,  0.95]),
    "bad":        np.array([ 0.0,   0.0,  0.0, -0.9]),
    "terrible":   np.array([ 0.0,   0.0,  0.0, -0.95]),
    "paris":      np.array([-0.1,   0.2,  0.0,  0.5]),
    "france":     np.array([-0.1,   0.15, 0.0,  0.4]),
    "italy":      np.array([-0.15,  0.15, 0.0,  0.35]),
    "rome":       np.array([-0.15,  0.2,  0.0,  0.45]),
}

# L2 normalize all embeddings
for word in DEMO_EMBEDDINGS:
    v = DEMO_EMBEDDINGS[word]
    DEMO_EMBEDDINGS[word] = v / (np.linalg.norm(v) + 1e-9)

def cosine_sim(v1, v2):
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

def find_most_similar(query_vec, embeddings, exclude=None, top_n=5):
    sims = [(w, cosine_sim(query_vec, v)) for w, v in embeddings.items()
            if exclude is None or w not in exclude]
    return sorted(sims, key=lambda x: x[1], reverse=True)[:top_n]


print("  Semantic similarities in embedding space:")
print()
pairs_demo = [
    ("cat", "dog"), ("cat", "kitten"), ("cat", "car"),
    ("good", "great"), ("good", "bad"), ("king", "queen"),
]
print(f"  {'Word 1':<12} {'Word 2':<12} {'Similarity':>12}")
print(f"  {'─'*12} {'─'*12} {'─'*12}")
for w1, w2 in pairs_demo:
    sim = cosine_sim(DEMO_EMBEDDINGS[w1], DEMO_EMBEDDINGS[w2])
    print(f"  {w1:<12} {w2:<12} {sim:>12.4f}")

print()
print("  Word analogy: king - man + woman ≈ ?")
analogy_vec = DEMO_EMBEDDINGS["king"] - DEMO_EMBEDDINGS["man"] + DEMO_EMBEDDINGS["woman"]
analogy_vec /= np.linalg.norm(analogy_vec) + 1e-9
results = find_most_similar(analogy_vec, DEMO_EMBEDDINGS, exclude={"king", "man", "woman"})
print(f"  Top results:")
for word, sim in results[:5]:
    print(f"    {word:<12}: {sim:.4f}")

print()
print("  Word analogy: paris - france + italy ≈ ?")
analogy_vec2 = DEMO_EMBEDDINGS["paris"] - DEMO_EMBEDDINGS["france"] + DEMO_EMBEDDINGS["italy"]
analogy_vec2 /= np.linalg.norm(analogy_vec2) + 1e-9
results2 = find_most_similar(analogy_vec2, DEMO_EMBEDDINGS, exclude={"paris", "france", "italy"})
for word, sim in results2[:5]:
    print(f"    {word:<12}: {sim:.4f}")

print()


# ======================================================================
# SECTION 4: Loading GloVe (Production Code)
# ======================================================================
print("=" * 70)
print("SECTION 4: LOADING PRETRAINED EMBEDDINGS IN PRODUCTION")
print("=" * 70)
print()
print("  Method 1: gensim (easiest)")
print()
print("    import gensim.downloader as api")
print("    wv = api.load('glove-wiki-gigaword-100')")
print("    wv['cat']           # 100-d vector")
print("    wv.most_similar('cat')")
print("    wv.most_similar(positive=['king','woman'], negative=['man'])")
print()
print("  Method 2: Load GloVe .txt file directly")
print()
print("    def load_glove(path):")
print("        embeddings = {}")
print("        with open(path, 'r') as f:")
print("            for line in f:")
print("                parts = line.split()")
print("                word  = parts[0]")
print("                vec   = np.array(parts[1:], dtype='float32')")
print("                embeddings[word] = vec")
print("        return embeddings")
print()
print("    glove = load_glove('glove.6B.100d.txt')")
print()
print("  Method 3: Use Keras Embedding layer with pretrained weights")
print()
print("    embedding_matrix = np.zeros((vocab_size, EMBED_DIM))")
print("    for word, idx in word2idx.items():")
print("        vec = glove.get(word)")
print("        if vec is not None:")
print("            embedding_matrix[idx] = vec")
print()
print("    Embedding layer:")
print("    layers.Embedding(vocab_size, EMBED_DIM,")
print("                     weights=[embedding_matrix],")
print("                     trainable=False)   # frozen pretrained")
print()


# ======================================================================
# SECTION 5: FastText — Subword Embeddings
# ======================================================================
print("=" * 70)
print("SECTION 5: FASTTEXT — SUBWORD EMBEDDINGS")
print("=" * 70)
print()
print("Problem with Word2Vec / GloVe:")
print("  'running' is in vocab, but 'runing' (typo) is OOV → <UNK>")
print("  'unhappiness' may be OOV, even though 'happy' is in vocab")
print("  New words, rare words, morphological variants all get <UNK>")
print()
print("FastText solution: use CHARACTER N-GRAMS as the unit")
print()
print("  Word: 'playing'")
print("  Character 3-grams: <pl, pla, lay, ayi, yin, ing, ng>")
print("  (< and > mark word boundaries)")
print()
print("  Word vector = sum of all its character n-gram vectors")
print()
print("  Benefits:")
print("  1. OOV words can still be represented (sum their n-gram vectors)")
print("  2. Morphological variants share subword vectors ('play', 'playing', 'played')")
print("  3. Handles typos gracefully")
print("  4. Works well for morphologically rich languages (German, Finnish, Turkish)")
print()
print("  When to use:")
print("  • Noisy text (social media, reviews) → FastText")
print("  • Formal text, large vocab → GloVe/Word2Vec")
print("  • State-of-the-art → BERT/GPT (contextual subword embeddings)")
print()


# ======================================================================
# SECTION 6: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 6: VISUALIZATIONS")
print("=" * 70)
print()


# --- PLOT 1: Training loss curve + learned similarity matrix ---
print("Generating: Word2Vec training loss + similarity heatmap...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Word2Vec Skip-gram From Scratch: Training and Results",
             fontsize=14, fontweight="bold")

# Training loss curve
axes[0].plot(losses, color="darkorange", linewidth=2)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Average Loss")
axes[0].set_title("Skip-gram Training Loss")
axes[0].grid(True, alpha=0.3)
axes[0].text(len(losses) * 0.6, losses[0] * 0.85,
             f"Loss: {losses[0]:.2f} → {losses[-1]:.2f}\n"
             f"Reduction: {(1 - losses[-1]/losses[0])*100:.0f}%",
             fontsize=10, color="darkorange",
             bbox=dict(boxstyle="round", facecolor="moccasin", alpha=0.7))

# Similarity matrix of all vocab words
emb_matrix = model.W_in.copy()
norms       = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
norms[norms < 1e-9] = 1
emb_normalized = emb_matrix / norms
sim_matrix  = emb_normalized @ emb_normalized.T

im = axes[1].imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
axes[1].set_title("Word Similarity Matrix\n(Learned Embeddings)", fontsize=11, fontweight="bold")
axes[1].set_xticks(range(V))
axes[1].set_xticklabels(vocab_list, rotation=45, ha="right", fontsize=7)
axes[1].set_yticks(range(V))
axes[1].set_yticklabels(vocab_list, fontsize=7)
plt.colorbar(im, ax=axes[1], shrink=0.8, label="Cosine Similarity")

# PCA visualization of learned embeddings (2D projection)
try:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    emb_2d = pca.fit_transform(emb_normalized)
    var_explained = pca.explained_variance_ratio_.sum()

    # Color by semantic group
    animal_words  = {"cat", "cats", "dog", "dogs", "mouse"}
    place_words   = {"mat", "log", "fish", "bones", "ball"}
    action_words  = {"sat", "chased", "hid", "rolled", "played", "like"}
    function_words= {"the", "a", "and", "are", "on", "under", "off", "together"}

    colors_map = {"animal": "#E74C3C", "place": "#2ECC71",
                  "action": "#3498DB", "function": "#95A5A6", "other": "#9B59B6"}

    for i, word in enumerate(vocab_list):
        if word in animal_words:
            color, grp = colors_map["animal"], "animal"
        elif word in place_words:
            color, grp = colors_map["place"], "place"
        elif word in action_words:
            color, grp = colors_map["action"], "action"
        elif word in function_words:
            color, grp = colors_map["function"], "function"
        else:
            color, grp = colors_map["other"], "other"

        axes[2].scatter(emb_2d[i, 0], emb_2d[i, 1], c=color, s=80, zorder=3)
        axes[2].annotate(word, (emb_2d[i, 0], emb_2d[i, 1]),
                         textcoords="offset points", xytext=(4, 2), fontsize=7)

    legend_elements = [
        mpatches.Patch(color=colors_map["animal"],   label="Animals"),
        mpatches.Patch(color=colors_map["place"],    label="Places/Objects"),
        mpatches.Patch(color=colors_map["action"],   label="Actions"),
        mpatches.Patch(color=colors_map["function"], label="Function words"),
    ]
    axes[2].legend(handles=legend_elements, fontsize=8, loc="best")
    axes[2].set_title(f"Embedding Space: PCA 2D Projection\n(explains {var_explained:.0%} of variance)",
                      fontsize=11, fontweight="bold")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlabel("PC 1"); axes[2].set_ylabel("PC 2")

except ImportError:
    axes[2].text(0.5, 0.5, "sklearn not installed\n(needed for PCA)",
                 ha="center", va="center", fontsize=12, transform=axes[2].transAxes)

plt.tight_layout()
plt.savefig("../visuals/03_word_embeddings/word2vec_training_results.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: word2vec_training_results.png")


# --- PLOT 2: Semantic embedding space (demo embeddings) ---
print("Generating: Semantic embedding space visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Pretrained Embedding Space: Semantics as Geometry",
             fontsize=14, fontweight="bold")

# PCA on demo embeddings (4D → 2D)
demo_words = list(DEMO_EMBEDDINGS.keys())
demo_vecs  = np.array([DEMO_EMBEDDINGS[w] for w in demo_words])

try:
    from sklearn.decomposition import PCA
    pca_demo = PCA(n_components=2, random_state=42)
    demo_2d  = pca_demo.fit_transform(demo_vecs)

    # Color coding
    animals  = {"cat", "dog", "kitten", "puppy", "horse"}
    vehicles = {"car", "truck", "bicycle"}
    actions  = {"run", "walk", "jump"}
    royalty  = {"king", "queen", "man", "woman"}
    sentiment= {"good", "great", "bad", "terrible"}
    places   = {"paris", "france", "italy", "rome"}

    group_colors = {
        "animals": "#E74C3C", "vehicles": "#3498DB", "actions": "#2ECC71",
        "royalty": "#9B59B6", "sentiment": "#F39C12", "places": "#1ABC9C"
    }

    for i, word in enumerate(demo_words):
        if word in animals:    color, grp = group_colors["animals"],   "Animals"
        elif word in vehicles: color, grp = group_colors["vehicles"],  "Vehicles"
        elif word in actions:  color, grp = group_colors["actions"],   "Actions"
        elif word in royalty:  color, grp = group_colors["royalty"],   "Royalty/Gender"
        elif word in sentiment:color, grp = group_colors["sentiment"], "Sentiment"
        else:                  color, grp = group_colors["places"],    "Places"

        axes[0].scatter(demo_2d[i, 0], demo_2d[i, 1],
                        c=color, s=120, zorder=3, edgecolors="white", linewidth=1.5)
        axes[0].annotate(word, (demo_2d[i, 0], demo_2d[i, 1]),
                         textcoords="offset points", xytext=(6, 2), fontsize=9, fontweight="bold")

    legend_elements = [
        mpatches.Patch(color=v, label=k.title()) for k, v in group_colors.items()
    ]
    axes[0].legend(handles=legend_elements, fontsize=9, loc="lower right")
    axes[0].set_title("Embedding Space: Semantic Clusters (PCA 2D)",
                      fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel("PC 1 (semantic axis)"); axes[0].set_ylabel("PC 2")

    # Analogy arrows
    analogy_pairs = [
        ("king", "queen", "man", "woman", "Gender"),
        ("paris", "rome",  "france", "italy", "Capital-Country"),
    ]
    colors_arrow = ["#E67E22", "#16A085"]

    for (w1, w2, w3, w4, lbl), col in zip(analogy_pairs, colors_arrow):
        idx = {w: i for i, w in enumerate(demo_words)}
        p1, p2, p3, p4 = demo_2d[idx[w1]], demo_2d[idx[w2]], demo_2d[idx[w3]], demo_2d[idx[w4]]
        # Draw parallelogram
        axes[0].annotate("", xy=p2, xytext=p1,
                         arrowprops=dict(arrowstyle="->", color=col, lw=2))
        axes[0].annotate("", xy=p4, xytext=p3,
                         arrowprops=dict(arrowstyle="->", color=col, lw=2, linestyle="--"))
        mid = (p1 + p2) / 2
        axes[0].text(mid[0], mid[1] + 0.02, lbl, fontsize=8, color=col,
                     fontweight="bold", ha="center")

except ImportError:
    axes[0].text(0.5, 0.5, "sklearn not installed\n(needed for PCA)",
                 ha="center", va="center", fontsize=12, transform=axes[0].transAxes)

# Right plot: cosine similarity heatmap of demo embeddings
demo_sim = np.array([[cosine_sim(DEMO_EMBEDDINGS[w1], DEMO_EMBEDDINGS[w2])
                      for w2 in demo_words] for w1 in demo_words])

im = axes[1].imshow(demo_sim, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
axes[1].set_xticks(range(len(demo_words)))
axes[1].set_xticklabels(demo_words, rotation=45, ha="right", fontsize=8)
axes[1].set_yticks(range(len(demo_words)))
axes[1].set_yticklabels(demo_words, fontsize=8)
axes[1].set_title("Cosine Similarity Heatmap\n(Pretrained Demo Embeddings)",
                  fontsize=12, fontweight="bold")
plt.colorbar(im, ax=axes[1], shrink=0.8, label="Cosine Similarity")

plt.tight_layout()
plt.savefig("../visuals/03_word_embeddings/semantic_embedding_space.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: semantic_embedding_space.png")


# --- PLOT 3: BoW vs Embeddings comparison + static vs contextual ---
print("Generating: BoW vs Embeddings comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Word Representations: One-Hot → BoW → Embeddings → Contextual",
             fontsize=14, fontweight="bold")

# One-hot vs embedding size comparison
vocab_sizes = [1000, 5000, 10000, 30000, 50000, 100000]
onehot_dims = vocab_sizes
embed_dims  = [300] * len(vocab_sizes)

axes[0].plot(vocab_sizes, onehot_dims, "r-o", linewidth=2, markersize=8, label="One-hot dimension")
axes[0].axhline(y=300, color="#2ECC71", linewidth=2.5, linestyle="--", label="Embedding dim (300)")
axes[0].set_xlabel("Vocabulary Size"); axes[0].set_ylabel("Vector Dimension")
axes[0].set_title("Dimensionality: One-Hot vs Embeddings", fontsize=11, fontweight="bold")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].text(5000, 50, "Word2Vec/GloVe:\nalways 300-d\nregardless of vocab size!",
             fontsize=9, color="#2ECC71", fontweight="bold",
             bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.4))

# Sentence similarity with BoW vs embeddings
sentences = [
    "I love this movie",
    "I enjoy this film",
    "I hate this movie",
    "The weather is nice",
]
bow_short = {
    "I": [1, 1, 1, 0], "love": [1, 0, 0, 0], "enjoy": [0, 1, 0, 0],
    "hate": [0, 0, 1, 0], "this": [1, 1, 1, 0], "movie": [1, 0, 1, 0],
    "film": [0, 1, 0, 0], "The": [0, 0, 0, 1], "weather": [0, 0, 0, 1], "nice": [0, 0, 0, 1]
}

# Simulate BoW similarity matrix
bow_vecs = []
for sent in sentences:
    tokens = sent.lower().split()
    vec    = np.zeros(len(bow_short))
    for i, word in enumerate(bow_short):
        vec[i] = tokens.count(word.lower())
    norm = np.linalg.norm(vec)
    bow_vecs.append(vec / norm if norm > 0 else vec)

bow_sim_4x4 = np.array([[np.dot(a, b) for b in bow_vecs] for a in bow_vecs])

# Simulate semantic similarity matrix (embeddings would capture meaning)
embed_sim_4x4 = np.array([
    [1.00, 0.88, 0.35, 0.05],  # "I love this movie"
    [0.88, 1.00, 0.33, 0.06],  # "I enjoy this film"
    [0.35, 0.33, 1.00, 0.04],  # "I hate this movie"
    [0.05, 0.06, 0.04, 1.00],  # "The weather is nice"
])

labels_4 = ["love\nmovie", "enjoy\nfilm", "hate\nmovie", "weather\nnice"]

im1 = axes[1].imshow(bow_sim_4x4, cmap="Blues", vmin=0, vmax=1)
axes[1].set_xticks(range(4)); axes[1].set_xticklabels(labels_4, fontsize=9)
axes[1].set_yticks(range(4)); axes[1].set_yticklabels(labels_4, fontsize=9)
axes[1].set_title("Cosine Similarity: BoW\n('enjoy' ≠ 'love' to BoW)", fontsize=11, fontweight="bold")
plt.colorbar(im1, ax=axes[1], shrink=0.8)
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, f"{bow_sim_4x4[i,j]:.2f}", ha="center", va="center",
                     fontsize=10, color="white" if bow_sim_4x4[i,j] > 0.5 else "black")

im2 = axes[2].imshow(embed_sim_4x4, cmap="Greens", vmin=0, vmax=1)
axes[2].set_xticks(range(4)); axes[2].set_xticklabels(labels_4, fontsize=9)
axes[2].set_yticks(range(4)); axes[2].set_yticklabels(labels_4, fontsize=9)
axes[2].set_title("Cosine Similarity: Embeddings\n('enjoy' ≈ 'love' semantically!)", fontsize=11, fontweight="bold")
plt.colorbar(im2, ax=axes[2], shrink=0.8)
for i in range(4):
    for j in range(4):
        axes[2].text(j, i, f"{embed_sim_4x4[i,j]:.2f}", ha="center", va="center",
                     fontsize=10, color="white" if embed_sim_4x4[i,j] > 0.5 else "black")

plt.tight_layout()
plt.savefig("../visuals/03_word_embeddings/bow_vs_embeddings.png", dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: bow_vs_embeddings.png")


print()
print("=" * 70)
print("NLP MATH FOUNDATION 3: WORD EMBEDDINGS COMPLETE!")
print("=" * 70)
print()
print("What you learned:")
print("  ✓ TF-IDF one-hot fails: cat and dog look equally different")
print("  ✓ Distributional hypothesis: context defines meaning")
print("  ✓ Word2Vec Skip-gram: train to predict context from center word")
print("  ✓ Embeddings encode semantics: similar words have similar vectors")
print("  ✓ Analogy arithmetic: king - man + woman = queen")
print("  ✓ Pretrained embeddings (GloVe/Word2Vec): train once, use everywhere")
print("  ✓ FastText: subword n-grams fix OOV problem")
print("  ✓ Static (GloVe) vs contextual (BERT/GPT) embeddings")
print()
print("3 Visualizations saved to: ../visuals/03_word_embeddings/")
print("  1. word2vec_training_results.png  — training loss + PCA of scratch embeddings")
print("  2. semantic_embedding_space.png   — clusters + analogy arrows + heatmap")
print("  3. bow_vs_embeddings.png          — dimensionality + similarity comparison")
print()
print("Next: Foundation 4 → RNN Intuition (sequential data, hidden states)")
