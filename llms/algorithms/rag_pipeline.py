"""
RAG Pipeline: Retrieval-Augmented Generation from Scratch
==========================================================
Learning Objectives:
  1. Build a TF-IDF document retriever from scratch
  2. Implement 3 chunking strategies (fixed, sentence, paragraph)
  3. Construct a full RAG pipeline: index, retrieve, augment, generate
  4. Evaluate retrieval quality with precision@k and MRR
  5. Compare retrieval strategies: TF-IDF vs BM25 vs random
  6. Understand when RAG outperforms pure prompting
YouTube: Search "RAG from scratch LlamaIndex LangChain explained"
Time: ~45 min | Difficulty: Advanced
"""

import os
import re
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "rag_pipeline")
os.makedirs(VIS_DIR, exist_ok=True)

np.random.seed(42)

# ===========================================================================
print("\n" + "="*70)
print("SECTION 1: WHY RAG?")
print("="*70)
# ===========================================================================

print("""
The Limitations of Pure Language Models
----------------------------------------

Problem 1: Knowledge Cutoff
  - LLMs are trained on static snapshots of the internet
  - GPT-4 cutoff: ~early 2024. Cannot answer: "Who won the 2025 election?"
  - Fine-tuning to add new knowledge is expensive and causes forgetting

Problem 2: Hallucination
  - LLMs generate plausible-sounding text — not necessarily true text
  - Without retrieved context, the model "guesses" facts confidently
  - Example: "Who wrote the RAG paper?" → model might hallucinate an author

Problem 3: Context Window Limits
  - Even 128K-token context windows can't hold entire knowledge bases
  - Sending all documents is slow, expensive, and degrades attention quality
  - Solution: select only the most relevant documents at query time

RAG Solution: Retrieval-Augmented Generation
---------------------------------------------
1. Index knowledge base into a searchable vector/inverted index
2. At query time: retrieve top-K most relevant documents
3. Augment the LLM prompt with retrieved context
4. Generate grounded, verifiable answers

RAG Formula:
  P(answer | query) ≈ Σ_d P(answer | query, d) × P(d | query)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^
                      Language model component    Retrieval component

When RAG beats pure prompting:
  - Factual QA with up-to-date or private knowledge
  - Long-document QA (legal, medical, code)
  - Citation-required tasks (journalism, research)
  - Multi-hop reasoning across documents
""")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 2: KNOWLEDGE BASE")
print("="*70)
# ===========================================================================

KB = [
    {"id": 0, "title": "Linear Regression",
     "text": "Linear regression fits a line y = wx + b by minimizing mean squared error between predictions and targets using gradient descent or the normal equation."},
    {"id": 1, "title": "Backpropagation",
     "text": "Backpropagation computes gradients of the loss function with respect to every parameter using the chain rule, enabling neural networks to learn."},
    {"id": 2, "title": "Convolutional Networks",
     "text": "CNNs use learnable filters that slide over inputs to detect local patterns. Pooling reduces spatial dimensions while preserving important features."},
    {"id": 3, "title": "Attention Mechanism",
     "text": "Attention allows models to focus on relevant parts of the input. Scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V."},
    {"id": 4, "title": "BERT",
     "text": "BERT uses a bidirectional transformer encoder pre-trained on masked language modeling. The [CLS] token embedding represents the whole sequence."},
    {"id": 5, "title": "GPT",
     "text": "GPT uses a causal transformer decoder trained on next-token prediction. At inference, tokens are generated autoregressively one at a time."},
    {"id": 6, "title": "LoRA",
     "text": "LoRA decomposes weight updates as DeltaW = BA where rank r << d. Only A and B are trained, reducing trainable parameters by 99%+ while preserving quality."},
    {"id": 7, "title": "RAG",
     "text": "Retrieval-augmented generation retrieves relevant documents at inference time and includes them in the prompt, reducing hallucination and extending knowledge cutoff."},
    {"id": 8, "title": "Prompt Engineering",
     "text": "Prompt engineering crafts inputs to guide LLM behavior without gradient updates. Techniques: zero-shot, few-shot, chain-of-thought, structured output formatting."},
    {"id": 9, "title": "Scaling Laws",
     "text": "Neural scaling laws show loss decreases predictably with compute, data, and parameters. Chinchilla: optimal tokens = 20 x parameters for compute-efficient training."},
]

print(f"Knowledge base loaded: {len(KB)} documents")
print(f"\n{'ID':>3} | {'Title':<25} | Excerpt")
print("-" * 80)
for doc in KB:
    excerpt = doc['text'][:55] + "..."
    print(f"{doc['id']:>3} | {doc['title']:<25} | {excerpt}")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 3: CHUNKING STRATEGIES")
print("="*70)
# ===========================================================================

def chunk_fixed(text, chunk_size=50, overlap=10):
    """Split text into fixed-size token (word) chunks with overlap."""
    words  = text.split()
    chunks = []
    step   = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def chunk_sentences(text):
    """Split on sentence boundaries (. ? !)."""
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_paragraphs(text, min_words=5):
    """Split on double-newlines; fallback to whole text if no paragraphs."""
    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paras:
        paras = [text.strip()]
    # Filter tiny fragments
    return [p for p in paras if len(p.split()) >= min_words]


# Demo on combined text
demo_text = KB[0]['text'] + " " + KB[1]['text']

print(f"\nDemo text (combined docs 0+1):\n  '{demo_text[:100]}...'\n")

chunks_fixed   = chunk_fixed(demo_text, chunk_size=12, overlap=3)
chunks_sent    = chunk_sentences(demo_text)
chunks_para    = chunk_paragraphs(demo_text)

print(f"Fixed-size chunks (size=12, overlap=3): {len(chunks_fixed)} chunks")
for i, c in enumerate(chunks_fixed):
    print(f"  [{i}] '{c}'")

print(f"\nSentence chunks: {len(chunks_sent)} chunks")
for i, c in enumerate(chunks_sent):
    print(f"  [{i}] '{c}'")

print(f"\nParagraph chunks: {len(chunks_para)} chunks")
for i, c in enumerate(chunks_para):
    print(f"  [{i}] '{c[:80]}...' ")

print("""
Chunking Strategy Comparison:
  Fixed-size  : Simple, predictable length. Risk: cuts mid-sentence.
  Sentence    : Preserves meaning. Variable length. Good for QA tasks.
  Paragraph   : Best semantic coherence. Best for long documents.
  Recommended : Sentence or paragraph chunking with ~20% overlap.
""")

# ===========================================================================
print("\n" + "="*70)
print("SECTION 4: TF-IDF RETRIEVER (From Scratch)")
print("="*70)
# ===========================================================================

class TFIDFRetriever:
    """TF-IDF document retriever built entirely from scratch.

    TF  (term frequency)    : freq(term, doc) / len(doc)
    IDF (inverse doc freq)  : log(N / (1 + df(term)))
    TF-IDF                  : TF × IDF
    Similarity              : cosine(query_vec, doc_vec)
    """

    def __init__(self):
        self.vocab      = {}        # term → index
        self.idf        = None      # (vocab_size,)
        self.doc_vecs   = None      # (n_docs, vocab_size) — normalised
        self.n_docs     = 0

    def _tokenize(self, text):
        """Lowercase, remove punctuation, split on whitespace."""
        text   = text.lower()
        text   = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        # Remove single-char tokens and common stop words
        stops  = {'a', 'an', 'the', 'is', 'are', 'was', 'were',
                  'in', 'on', 'at', 'to', 'of', 'and', 'or', 'it',
                  'this', 'that', 'with', 'for', 'by', 'as', 'be'}
        return [t for t in tokens if len(t) > 1 and t not in stops]

    def fit(self, documents):
        """Build vocabulary, compute IDF, and index all documents."""
        n          = len(documents)
        self.n_docs = n

        # Build vocabulary from all documents
        tok_docs = [self._tokenize(d) for d in documents]
        all_terms = sorted(set(t for tokens in tok_docs for t in tokens))
        self.vocab = {term: idx for idx, term in enumerate(all_terms)}
        V = len(self.vocab)

        # Document frequency
        df = np.zeros(V)
        for tokens in tok_docs:
            for term in set(tokens):
                if term in self.vocab:
                    df[self.vocab[term]] += 1

        # IDF: log(N / (1 + df))  — smoothed
        self.idf = np.log((n + 1) / (df + 1)) + 1.0   # scikit-learn style smooth IDF

        # TF-IDF document vectors (normalised to unit length)
        doc_matrix = np.zeros((n, V))
        for i, tokens in enumerate(tok_docs):
            tf = np.zeros(V)
            for term in tokens:
                if term in self.vocab:
                    tf[self.vocab[term]] += 1
            if len(tokens) > 0:
                tf = tf / len(tokens)          # term frequency
            tfidf = tf * self.idf
            norm  = np.linalg.norm(tfidf)
            doc_matrix[i] = tfidf / (norm + 1e-10)

        self.doc_vecs = doc_matrix
        print(f"TF-IDF index built: {n} docs, vocabulary size {V}")

    def _cosine_sim(self, vec_a, vec_b):
        """Cosine similarity between two unit vectors."""
        return float(np.dot(vec_a, vec_b))

    def vectorize_query(self, query):
        """Convert query string to normalised TF-IDF vector."""
        tokens = self._tokenize(query)
        V      = len(self.vocab)
        tf     = np.zeros(V)
        for term in tokens:
            if term in self.vocab:
                tf[self.vocab[term]] += 1
        if len(tokens) > 0:
            tf = tf / len(tokens)
        tfidf = tf * self.idf
        norm  = np.linalg.norm(tfidf)
        return tfidf / (norm + 1e-10)

    def retrieve(self, query, top_k=3):
        """Return list of (doc_idx, score) sorted by descending similarity."""
        q_vec  = self.vectorize_query(query)
        scores = self.doc_vecs @ q_vec        # cosine sims via dot product on unit vectors
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked]

    def similarity_matrix(self, queries):
        """Return (n_queries, n_docs) similarity matrix for a list of query strings."""
        q_vecs = np.array([self.vectorize_query(q) for q in queries])
        return q_vecs @ self.doc_vecs.T


# ===========================================================================
print("\n" + "="*70)
print("SECTION 5: FULL RAG PIPELINE")
print("="*70)
# ===========================================================================

class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self, kb_docs):
        self.kb        = kb_docs
        self.retriever = TFIDFRetriever()
        self.retriever.fit([d['text'] for d in kb_docs])

    def retrieve(self, query, top_k=3):
        """Retrieve top-k (doc_idx, score) pairs for a query."""
        return self.retriever.retrieve(query, top_k)

    def build_prompt(self, query, top_k=3):
        """Construct the augmented prompt string with retrieved context."""
        results = self.retrieve(query, top_k)
        context = "\n\n".join(
            [f"[Source {i+1}: {self.kb[idx]['title']}]\n{self.kb[idx]['text']}"
             for i, (idx, score) in enumerate(results)]
        )
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        return prompt, results

    def answer(self, query, top_k=3, verbose=False):
        """Extract best answer sentence from top retrieved document."""
        prompt, results = self.build_prompt(query, top_k)
        top_idx, top_score = results[0]
        top_text  = self.kb[top_idx]['text']
        q_words   = set(self._clean_query(query))
        sentences = top_text.split('. ')
        best = max(sentences, key=lambda s: sum(1 for w in s.lower().split()
                                                if re.sub(r'[^\w]', '', w) in q_words))
        if verbose:
            print(f"\n  Query     : {query}")
            retrieved_titles = [self.kb[i]['title'] for i, _ in results]
            print(f"  Retrieved : {retrieved_titles}")
            print(f"  Score     : {top_score:.4f}")
            print(f"  Answer    : {best.strip()}")
        return best.strip(), results

    def _clean_query(self, query):
        return [re.sub(r'[^\w]', '', w.lower()) for w in query.split() if len(w) > 2]


# Build the pipeline
rag = RAGPipeline(KB)

DEMO_QUERIES = [
    "How does backpropagation compute gradients?",
    "What is attention mechanism in transformers?",
    "How does BERT differ from GPT architecture?",
    "What is LoRA and how does it reduce parameters?",
    "How does RAG reduce hallucination in language models?",
    "What are neural scaling laws and Chinchilla?",
]

print("\nRunning RAG pipeline on 6 demo queries:")
print("-" * 70)
for q in DEMO_QUERIES:
    rag.answer(q, top_k=3, verbose=True)

# ===========================================================================
print("\n" + "="*70)
print("SECTION 6: EVALUATION — PRECISION@K AND MRR")
print("="*70)
# ===========================================================================

# Ground truth: for each query, which KB doc IDs are relevant answers
GROUND_TRUTH = {
    "How does backpropagation compute gradients?":           [1],
    "What is attention mechanism in transformers?":          [3],
    "How does BERT differ from GPT architecture?":           [4, 5],
    "What is LoRA and how does it reduce parameters?":       [6],
    "How does RAG reduce hallucination in language models?": [7],
    "What are neural scaling laws and Chinchilla?":          [9],
}

def precision_at_k(retrieved_ids, relevant_ids, k):
    """P@k = |retrieved[:k] ∩ relevant| / k"""
    top_k_ids = [doc_id for doc_id, _ in retrieved_ids[:k]]
    hits = sum(1 for did in top_k_ids if did in relevant_ids)
    return hits / k

def reciprocal_rank(retrieved_ids, relevant_ids):
    """1/rank of first relevant document; 0 if not found in results."""
    for rank, (doc_id, _) in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0

def random_retrieval(n_docs, top_k, seed=0):
    """Random baseline: shuffle doc indices."""
    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(n_docs)[:top_k]
    return [(int(i), 0.0) for i in shuffled]


print("\nEvaluation Results (TF-IDF vs Random Baseline):")
print(f"\n{'Query':<55} | {'P@1':>5} | {'P@3':>5} | {'RR':>6}")
print("-" * 80)

tfidf_p1, tfidf_p3, tfidf_rr = [], [], []
rand_p1,  rand_p3,  rand_rr  = [], [], []
per_query_p3_tfidf = []
per_query_p3_rand  = []

for q, relevant in GROUND_TRUTH.items():
    results_tfidf  = rag.retrieve(q, top_k=len(KB))   # retrieve all to compute full MRR
    results_random = random_retrieval(len(KB), top_k=len(KB), seed=hash(q) % 1000)

    p1_t = precision_at_k(results_tfidf,  relevant, 1)
    p3_t = precision_at_k(results_tfidf,  relevant, 3)
    rr_t = reciprocal_rank(results_tfidf, relevant)

    p1_r = precision_at_k(results_random,  relevant, 1)
    p3_r = precision_at_k(results_random,  relevant, 3)
    rr_r = reciprocal_rank(results_random, relevant)

    tfidf_p1.append(p1_t); tfidf_p3.append(p3_t); tfidf_rr.append(rr_t)
    rand_p1.append(p1_r);  rand_p3.append(p3_r);  rand_rr.append(rr_r)
    per_query_p3_tfidf.append(p3_t)
    per_query_p3_rand.append(p3_r)

    q_short = q[:52] + "..." if len(q) > 52 else q
    print(f"  {q_short:<55} | {p3_t:>5.2f} | {rr_t:>6.3f}")

print("\nAggregate metrics:")
print(f"{'Metric':<20} | {'TF-IDF':>10} | {'Random':>10}")
print("-" * 44)
print(f"{'Precision@1':<20} | {np.mean(tfidf_p1):>10.3f} | {np.mean(rand_p1):>10.3f}")
print(f"{'Precision@3':<20} | {np.mean(tfidf_p3):>10.3f} | {np.mean(rand_p3):>10.3f}")
print(f"{'MRR':<20} | {np.mean(tfidf_rr):>10.3f} | {np.mean(rand_rr):>10.3f}")

print(f"\nTF-IDF lift over random:")
print(f"  P@1 improvement : {np.mean(tfidf_p1) - np.mean(rand_p1):+.3f}")
print(f"  P@3 improvement : {np.mean(tfidf_p3) - np.mean(rand_p3):+.3f}")
print(f"  MRR improvement : {np.mean(tfidf_rr) - np.mean(rand_rr):+.3f}")

# ===========================================================================
print("\n" + "="*70)
print("VISUALISATIONS")
print("="*70)
# ===========================================================================

# ------------------------------------------------------------------
# VIZ 1: RAG flow diagram
# ------------------------------------------------------------------
print("\nSaving 01_rag_flow.png ...")

fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis('off')
fig.patch.set_facecolor('#f8f9fa')
ax.set_facecolor('#f8f9fa')
ax.set_title("RAG Pipeline: Query → Retrieve → Augment → Answer",
             fontsize=14, fontweight='bold', pad=10)

flow_steps = [
    (1.1,  "User\nQuery",          '#264653', 'white'),
    (3.5,  "TF-IDF\nVectorize",    '#2a9d8f', 'white'),
    (5.9,  "Cosine\nSearch",       '#457b9d', 'white'),
    (8.3,  "Top-3\nDocuments",     '#e9c46a', '#111'),
    (10.7, "Augmented\nPrompt",    '#f4a261', '#111'),
    (13.1, "LLM\n(Generate)",      '#e76f51', 'white'),
    (15.5, "Answer",               '#264653', 'white'),
]

box_w, box_h = 2.0, 1.8
for (cx, label, facecolor, tc) in flow_steps:
    fancy = FancyBboxPatch((cx - box_w/2, 1.6), box_w, box_h,
                           boxstyle="round,pad=0.15",
                           facecolor=facecolor, edgecolor='white', lw=2)
    ax.add_patch(fancy)
    ax.text(cx, 2.5, label, ha='center', va='center',
            color=tc, fontsize=10, fontweight='bold')

# Arrows between boxes
for i in range(len(flow_steps) - 1):
    x_left  = flow_steps[i][0]   + box_w / 2 + 0.05
    x_right = flow_steps[i+1][0] - box_w / 2 - 0.05
    ax.annotate("", xy=(x_right, 2.5), xytext=(x_left, 2.5),
                arrowprops=dict(arrowstyle="->", color='#333', lw=2.0))

# Sub-labels below
sub_labels = [
    (1.1,  "e.g. 'How does\nattention work?'"),
    (3.5,  "word tokens →\ntfidf vector"),
    (5.9,  "dot product with\nall doc vectors"),
    (8.3,  "top-3 scored\ndocuments"),
    (10.7, "context +\nquestion string"),
    (13.1, "grounded\ngeneration"),
    (15.5, "cited,\nfactual"),
]
for (cx, slabel) in sub_labels:
    ax.text(cx, 1.35, slabel, ha='center', va='top', color='#555', fontsize=7.5,
            style='italic')

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_rag_flow.png", dpi=300, bbox_inches="tight", facecolor='#f8f9fa')
plt.close()
print("  Saved 01_rag_flow.png")

# ------------------------------------------------------------------
# VIZ 2: Similarity heatmap
# ------------------------------------------------------------------
print("Saving 02_similarity_heatmap.png ...")

queries_eval = list(GROUND_TRUTH.keys())
doc_texts    = [d['text'] for d in KB]
sim_matrix   = rag.retriever.similarity_matrix(queries_eval)    # (6, 10)

fig, ax = plt.subplots(figsize=(14, 5))
im = ax.imshow(sim_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=sim_matrix.max())

# Row labels: shortened queries
row_labels = [q[:38] + "..." if len(q) > 38 else q for q in queries_eval]
col_labels = [d['title'] for d in KB]

ax.set_yticks(range(len(queries_eval)))
ax.set_yticklabels(row_labels, fontsize=8.5)
ax.set_xticks(range(len(KB)))
ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=8.5)

ax.set_title("Query-Document Cosine Similarity (TF-IDF)", fontsize=13, fontweight='bold')
ax.set_xlabel("Knowledge Base Document", fontsize=11)
ax.set_ylabel("Query", fontsize=11)

# Annotate cells
for i in range(sim_matrix.shape[0]):
    for j in range(sim_matrix.shape[1]):
        val = sim_matrix[i, j]
        text_color = 'white' if val > 0.4 else '#333'
        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                fontsize=7.5, color=text_color, fontweight='bold')

plt.colorbar(im, ax=ax, shrink=0.8, label='Cosine similarity')
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_similarity_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved 02_similarity_heatmap.png")

# ------------------------------------------------------------------
# VIZ 3: Evaluation plots
# ------------------------------------------------------------------
print("Saving 03_evaluation.png ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RAG Retrieval Evaluation: TF-IDF vs Random Baseline", fontsize=13, fontweight='bold')

# (a) Aggregate metrics bar chart
ax = axes[0]
metrics       = ['Precision@1', 'Precision@3', 'MRR']
tfidf_scores  = [np.mean(tfidf_p1), np.mean(tfidf_p3), np.mean(tfidf_rr)]
rand_scores   = [np.mean(rand_p1),  np.mean(rand_p3),  np.mean(rand_rr)]

x_pos = np.arange(len(metrics))
bar_w = 0.35
bars1 = ax.bar(x_pos - bar_w/2, tfidf_scores, bar_w, label='TF-IDF', color='#2a9d8f', alpha=0.9)
bars2 = ax.bar(x_pos + bar_w/2, rand_scores,  bar_w, label='Random', color='#e63946', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("(a) Aggregate Retrieval Metrics", fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(list(bars1) + list(bars2), tfidf_scores + rand_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

# (b) Per-query Precision@3
ax2 = axes[1]
q_labels = [q[:35] + "..." if len(q) > 35 else q for q in GROUND_TRUTH.keys()]
x2 = np.arange(len(q_labels))
bars3 = ax2.bar(x2 - 0.18, per_query_p3_tfidf, 0.35, label='TF-IDF P@3', color='#2a9d8f', alpha=0.9)
bars4 = ax2.bar(x2 + 0.18, per_query_p3_rand,  0.35, label='Random P@3', color='#e63946', alpha=0.7)
ax2.set_xticks(x2)
ax2.set_xticklabels(q_labels, rotation=30, ha='right', fontsize=7.5)
ax2.set_ylabel("Precision@3", fontsize=11)
ax2.set_title("(b) Per-Query Precision@3", fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.2)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(list(bars3) + list(bars4), per_query_p3_tfidf + per_query_p3_rand):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_evaluation.png", dpi=300, bbox_inches="tight")
plt.close()
print("  Saved 03_evaluation.png")

print("\n" + "="*70)
print("COMPLETE: rag_pipeline.py")
print("="*70)
print(f"  Knowledge base      : {len(KB)} documents")
print(f"  Retrieval strategy  : TF-IDF cosine similarity (from scratch)")
print(f"  Chunking strategies : fixed-size, sentence, paragraph")
print(f"  Evaluation queries  : {len(GROUND_TRUTH)}")
print(f"  TF-IDF MRR          : {np.mean(tfidf_rr):.3f}  |  Random MRR: {np.mean(rand_rr):.3f}")
print(f"  Visuals saved       : {VIS_DIR}/")
print("  Files               : 01_rag_flow.png, 02_similarity_heatmap.png, 03_evaluation.png")
