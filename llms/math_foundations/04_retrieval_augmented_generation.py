"""
Retrieval-Augmented Generation (RAG)
======================================
Learning Objectives:
  1. Understand the knowledge cutoff and hallucination problems in LLMs
  2. Implement document chunking strategies from scratch
  3. Build a TF-IDF retriever from scratch
  4. Construct and run a complete RAG pipeline end-to-end
  5. Evaluate retrieval quality with precision@k
  6. Understand advanced RAG: HyDE, reranking, hybrid retrieval
YouTube: Search "RAG retrieval augmented generation explained from scratch"
Time: ~25 min | Difficulty: Advanced | Prerequisites: Part 5 NLP, Parts 3-6
"""

import os
import re
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "04_retrieval_augmented_generation")
os.makedirs(VIS_DIR, exist_ok=True)

# ==========================================================================
print("\n" + "="*70)
print("SECTION 1: THE PROBLEM RAG SOLVES")
print("="*70)
# ==========================================================================

print("""
Why do we need RAG (Retrieval-Augmented Generation)?

Problem 1 — Knowledge Cutoff:
  Every LLM is trained on a static snapshot of the internet.
  GPT-4 was trained on data up to April 2024 — it knows nothing about events
  after that date. Ask it about yesterday's news and it will either refuse or
  make something up.

Problem 2 — Hallucination:
  LLMs are probability machines, not databases. They generate plausible-
  sounding text, not necessarily true text.
  Example: "Who wrote the paper on LoRA?" → model may confidently name the
  wrong authors, or confuse it with a different paper.

Problem 3 — Context Window Limits:
  GPT-4's context window is ~128K tokens. A company knowledge base may have
  millions of documents totaling billions of tokens. You simply cannot fit
  everything in context — you need to be selective.

Solution — RAG:
  Retrieve the most relevant documents FIRST, then generate conditioned on them.
  The model's answer is grounded in retrieved facts, not just memorized patterns.

Bayesian framing:
  Without RAG:   P(answer | query) — model uses only its parametric memory
  With RAG:      P(answer | query) = ∫ P(answer | query, doc) · P(doc | query) d_doc

  We approximate: retrieve top-k docs, then generate conditioned on those k docs.
  This converts a free-form generation problem into a reading comprehension problem.

RAG is not a replacement for fine-tuning — they solve different problems:
  Fine-tuning → teaches the model a new style, skill, or behavior
  RAG         → gives the model access to up-to-date, verifiable facts
""")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 2: DOCUMENT CHUNKING")
print("="*70)
# ==========================================================================

print("""
LLMs have a limited context window. Even a 10-page document may not fit.
Chunking splits documents into manageable pieces that can be individually
indexed and retrieved.

Three common strategies:
  1. Fixed-size chunking:  split every N words, with M-word overlap
  2. Sentence chunking:    group K sentences together
  3. Paragraph chunking:   split on blank lines (natural boundaries)

The overlap in fixed-size chunking prevents relevant context from being cut
at an arbitrary word boundary — the same key phrase appears in two adjacent
chunks, improving recall.
""")

def fixed_chunk(text, chunk_size=150, overlap=30):
    """Split text into fixed-size word chunks with overlap."""
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def sentence_chunk(text, max_sentences=3):
    """Split into groups of sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks

def paragraph_chunk(text):
    """Split on blank lines (paragraph boundaries)."""
    return [p.strip() for p in text.split('\n\n') if p.strip()]

# Sample ML passage (~300 words, 4 natural paragraphs)
SAMPLE_TEXT = """Machine learning is a subfield of artificial intelligence that gives computers
the ability to learn from data without being explicitly programmed. The core idea is
to build algorithms that improve automatically through experience. Instead of writing
rules by hand, we show the system thousands of examples and let it discover the patterns.

Supervised learning is the most common paradigm. In supervised learning, the training
data consists of input-output pairs. The model learns a mapping from inputs to outputs.
Linear regression predicts continuous values. Logistic regression predicts class
probabilities. Neural networks learn hierarchical representations. The key challenge
is generalization: performing well on data the model has never seen before.

Deep learning uses neural networks with many layers. Each layer learns increasingly
abstract representations. Convolutional neural networks excel at image data by applying
learnable filters that detect local patterns like edges and textures. Recurrent networks
process sequences by maintaining a hidden state. The transformer architecture uses
self-attention to directly model dependencies between any two positions in a sequence.

Large language models like GPT and BERT are transformers trained on vast text corpora.
They learn to predict the next word or fill in masked words. This simple objective,
applied at scale to hundreds of billions of tokens, produces models that understand
language structure, world facts, and reasoning patterns. Fine-tuning adapts these
pre-trained models to specific tasks with a fraction of the original training cost."""

fixed_chunks = fixed_chunk(SAMPLE_TEXT, chunk_size=60, overlap=15)
sentence_chunks = sentence_chunk(SAMPLE_TEXT, max_sentences=2)
paragraph_chunks = paragraph_chunk(SAMPLE_TEXT)

print("Applied 3 chunking strategies to a ~300-word ML passage:")
print()
print(f"  Fixed-size (size=60 words, overlap=15):  {len(fixed_chunks)} chunks")
print(f"  Sentence-based (2 sentences per chunk):  {len(sentence_chunks)} chunks")
print(f"  Paragraph-based (natural boundaries):    {len(paragraph_chunks)} chunks")
print()

print("Fixed-size chunks (first 2):")
for i, c in enumerate(fixed_chunks[:2]):
    words = c.split()
    print(f"  [Chunk {i+1}] ({len(words)} words): {c[:120]}...")
print()

print("Sentence chunks (first 2):")
for i, c in enumerate(sentence_chunks[:2]):
    print(f"  [Chunk {i+1}]: {c[:130]}...")
print()

print("Paragraph chunks (first 2):")
for i, c in enumerate(paragraph_chunks[:2]):
    print(f"  [Chunk {i+1}]: {c[:130]}...")
print()

print("Trade-offs:")
print("  Fixed-size:  Predictable size, but may cut mid-sentence. Good for dense text.")
print("  Sentence:    Semantically cleaner splits. Good for factual/technical documents.")
print("  Paragraph:   Preserves context best. Good when paragraphs cover single ideas.")
print("  In practice: sentence or paragraph chunking with 10-15% overlap is most common.")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 3: TF-IDF RETRIEVER FROM SCRATCH")
print("="*70)
# ==========================================================================

print("""
TF-IDF (Term Frequency - Inverse Document Frequency) is a classic retrieval
algorithm. No neural networks needed — pure math.

TF(t, d)  = count(t in d) / total_tokens(d)          -- how often t appears in doc
IDF(t)    = log((1+N)/(1+df(t))) + 1                  -- how rare t is across all docs
TF-IDF(t,d) = TF(t,d) × IDF(t)

Then we compute cosine similarity between the query vector and each doc vector.
High IDF means the word is rare and discriminative (good signal).
Low IDF means the word is common (stop word — low information).

This TF-IDF retriever processes up to 2,000 unique terms from the corpus.
""")

STOPWORDS = {
    "a","an","the","is","are","was","were","be","to","of","and","or",
    "in","on","at","for","with","that","this","it","as","by","from",
    "not","but","we","he","she","they","i","you","have","has","had",
    "do","does","did","will","would","can","could","should","may","might"
}

class TFIDFRetriever:
    """TF-IDF based document retriever built from scratch using NumPy."""

    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []
        self.docs = []

    def _tokenize(self, text):
        """Lowercase, strip punctuation, remove stopwords and short tokens."""
        tokens = re.sub(r"[^a-z\s]", " ", text.lower()).split()
        return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    def fit(self, docs):
        """Build TF-IDF index over a list of document strings."""
        self.docs = docs
        N = len(docs)
        tokenized = [self._tokenize(d) for d in docs]

        # Document frequency: how many docs contain each word
        df = Counter()
        for toks in tokenized:
            df.update(set(toks))

        # Vocabulary: top 2000 most frequent words appearing in ≥1 doc
        freq_all = Counter(t for toks in tokenized for t in toks)
        self.vocab = {
            w: i for i, (w, _) in enumerate(freq_all.most_common(2000))
            if df[w] >= 1
        }

        # IDF with smoothing to avoid division by zero
        self.idf = {
            w: math.log((1 + N) / (1 + df.get(w, 0))) + 1
            for w in self.vocab
        }

        # Compute unit-normalized TF-IDF vector for each document
        V = len(self.vocab)
        self.doc_vectors = []
        for toks in tokenized:
            tf = Counter(toks)
            total = max(len(toks), 1)
            vec = np.zeros(V)
            for w, c in tf.items():
                if w in self.vocab:
                    vec[self.vocab[w]] = (c / total) * self.idf[w]
            norm = np.linalg.norm(vec)
            self.doc_vectors.append(vec / norm if norm > 0 else vec)
        self.doc_vectors = np.array(self.doc_vectors)   # (N, V)

    def retrieve(self, query, top_k=3):
        """Return top-k (doc, score) pairs for a query string."""
        toks = self._tokenize(query)
        V = len(self.vocab)
        tf = Counter(toks)
        total = max(len(toks), 1)
        q_vec = np.zeros(V)
        for w, c in tf.items():
            if w in self.vocab:
                q_vec[self.vocab[w]] = (c / total) * self.idf.get(w, 1.0)
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm
        # Cosine similarity = dot product of unit-normed vectors
        scores = self.doc_vectors @ q_vec
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], float(scores[i])) for i in top_idx]

    def score_matrix(self, queries):
        """Return (len(queries), len(docs)) similarity matrix."""
        mat = np.zeros((len(queries), len(self.docs)))
        for qi, q in enumerate(queries):
            results = self.retrieve(q, top_k=len(self.docs))
            doc_to_score = {d: s for d, s in results}
            for di, doc in enumerate(self.docs):
                mat[qi, di] = doc_to_score.get(doc, 0.0)
        return mat

print("TFIDFRetriever class defined with:")
print("  - _tokenize(): lowercase + stopword removal + length filter")
print("  - fit():       build vocab, IDF table, and document vectors")
print("  - retrieve():  cosine similarity ranking")
print("  - score_matrix(): full query×doc similarity matrix for evaluation")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 4: KNOWLEDGE BASE AND RAG DEMO")
print("="*70)
# ==========================================================================

KB_DOCS = [
    "Linear regression fits a straight line to predict continuous values. The loss is mean squared error minimized by gradient descent.",
    "Logistic regression uses the sigmoid function to predict class probabilities. The loss is binary cross-entropy.",
    "A neural network consists of layers of neurons. Each neuron computes a weighted sum followed by an activation function.",
    "Backpropagation computes gradients of the loss with respect to all weights using the chain rule of calculus.",
    "Convolutional neural networks use learnable filters to detect local patterns. Pooling reduces spatial dimensions.",
    "Recurrent neural networks process sequential data by maintaining a hidden state updated at each timestep.",
    "The transformer architecture uses self-attention to weigh the importance of each token when processing a sequence.",
    "BERT is a bidirectional encoder trained on masked language modeling and next sentence prediction objectives.",
    "LoRA adapts pre-trained models by adding low-rank decomposition matrices to existing weight matrices.",
    "RAG improves LLM accuracy by retrieving relevant documents and including them in the context window.",
]

print(f"Knowledge base: {len(KB_DOCS)} documents covering ML fundamentals")
print()
for i, doc in enumerate(KB_DOCS):
    print(f"  Doc {i+1:2d}: {doc[:80]}...")
print()

# Build index
retriever = TFIDFRetriever()
retriever.fit(KB_DOCS)
print(f"TF-IDF index built: {len(retriever.vocab)} unique terms, {len(KB_DOCS)} document vectors")
print()

# Demo queries
queries = [
    "How does backpropagation work?",
    "What is the difference between BERT and GPT?",
    "How do transformers process sequences?",
    "What is LoRA and why is it useful?",
    "How does RAG help with hallucination?",
]

print("Running 5 demo queries against the knowledge base:")
print("-" * 65)
for query in queries:
    results = retriever.retrieve(query, top_k=3)
    print(f"\n  Query: \"{query}\"")
    for rank, (doc, score) in enumerate(results, 1):
        print(f"    Rank {rank} (score={score:.3f}): {doc[:75]}...")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 5: FULL RAG PIPELINE")
print("="*70)
# ==========================================================================

print("""
The complete RAG pipeline ties retrieval and generation together:

  1. User submits a query
  2. Retriever fetches top-k relevant documents
  3. Documents are formatted into a context block
  4. Context + query are combined into a prompt
  5. LLM generates an answer conditioned on the prompt

In production, step 5 uses a real LLM (GPT-4, Claude, Llama, etc.).
Here we simulate generation by extracting the most query-relevant sentence
from the top retrieved document — this shows the retrieval mechanics clearly.
""")

class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self, docs):
        self.retriever = TFIDFRetriever()
        self.retriever.fit(docs)
        self.docs = docs

    def build_prompt(self, query, top_k=3):
        """Build a grounded prompt by inserting retrieved context."""
        results = self.retriever.retrieve(query, top_k)
        context_parts = [f"[Doc {i+1}] {doc}" for i, (doc, _) in enumerate(results)]
        context = "\n".join(context_parts)
        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""
        return prompt, results

    def answer(self, query, top_k=3):
        """Retrieve context and simulate an LLM answer."""
        prompt, results = self.build_prompt(query, top_k)
        # Simulate generation: find the sentence from the top doc most
        # overlapping with the query terms (stands in for real LLM generation)
        top_doc, top_score = results[0]
        query_words = set(query.lower().split())
        sentences = re.split(r'(?<=[.!?])\s+', top_doc)
        best_sentence = max(
            sentences,
            key=lambda s: sum(1 for w in query_words if w in s.lower())
        )
        return {
            "query": query,
            "retrieved": [(d[:80] + "...", s) for d, s in results],
            "simulated_answer": best_sentence,
            "confidence": top_score,
            "prompt_preview": prompt[:350] + "..."
        }


rag = RAGPipeline(KB_DOCS)

print("Running RAG pipeline on all 5 demo queries:")
print("=" * 65)
for i, query in enumerate(queries, 1):
    result = rag.answer(query, top_k=3)
    print(f"\nQuery {i}: {result['query']}")
    print(f"  Confidence (top-doc score): {result['confidence']:.3f}")
    print(f"  Top retrieved docs:")
    for rank, (doc_preview, score) in enumerate(result['retrieved'], 1):
        print(f"    {rank}. [{score:.3f}] {doc_preview}")
    print(f"  Simulated answer: \"{result['simulated_answer']}\"")

print()
print("Prompt format preview (Query 1):")
print("-" * 50)
sample_prompt, _ = rag.build_prompt(queries[0], top_k=2)
print(sample_prompt)
print()
print("In production: replace simulated_answer with a real LLM call using")
print("the full prompt. The model reads the context and generates a grounded answer.")

# Compute score matrix for visualization
print()
print("Computing full similarity matrix (5 queries × 10 docs)...")
score_mat = retriever.score_matrix(queries)
print(f"  Matrix shape: {score_mat.shape}")
print(f"  Score range:  [{score_mat.min():.3f}, {score_mat.max():.3f}]")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 6: ADVANCED RAG TECHNIQUES")
print("="*70)
# ==========================================================================

print("""
Basic TF-IDF RAG works, but production systems use several enhancements:

1. HyDE — Hypothetical Document Embeddings
   ─────────────────────────────────────────
   Problem:  query "How does backprop work?" and a doc "The chain rule computes
             gradients layer by layer" don't share many keywords.
   Solution: Use the LLM to generate a hypothetical answer to the query.
             Embed the hypothetical answer instead of the raw query.
             The hypothetical answer uses domain vocabulary that matches real docs.
   Result:   Dramatically better recall for concept-based queries.

2. Reranking (Cross-Encoder)
   ──────────────────────────
   Standard retrieval uses bi-encoders: encode query, encode doc, dot-product.
   Fast but less accurate (query and doc are encoded independently).
   Reranking: after fetching top-50 candidates, score each (query, doc) pair
   jointly through a cross-encoder (e.g., BERT reading both together).
   Much more accurate, but too slow for the full corpus — hence the two stages.
   Tools: Cohere Rerank API, cross-encoder/ms-marco-MiniLM from HuggingFace.

3. Hybrid Retrieval
   ─────────────────
   BM25 (lexical): great for exact keyword matches, rare terms, named entities
   Dense embeddings (semantic): great for paraphrasing and conceptual similarity
   Combine: score = α × BM25_score + (1-α) × dense_score
   α ≈ 0.3-0.7 depending on the query type.
   Tools: Elasticsearch (BM25) + Faiss/Pinecone (dense) → Reciprocal Rank Fusion.

4. Recursive / Hierarchical Retrieval
   ─────────────────────────────────────
   Build a two-level index:
     Level 1: summaries of each document (section headers, abstracts)
     Level 2: individual chunks within each document
   First retrieve relevant documents (level 1), then retrieve relevant chunks
   from within those documents (level 2).
   Reduces noise from off-topic chunks in large corpora.

5. Query Decomposition
   ─────────────────────
   Complex queries contain multiple sub-questions.
   "Compare BERT and GPT-2 architectures" → decompose into:
     - "How does BERT work?"
     - "How does GPT-2 work?"
     - "What are the differences?"
   Retrieve separately for each sub-query, then merge results.
   Tools: LangChain's MultiQueryRetriever, LlamaIndex sub-question engine.

Summary of the RAG Toolbox:
  Basic RAG:     TF-IDF / BM25 retrieval + LLM generation
  Better RAG:    Dense embeddings (sentence-transformers) + vector DB
  Best RAG:      Hybrid retrieval + HyDE + cross-encoder reranking + hierarchical index
""")

# ==========================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)
# ==========================================================================

# ---- VIS 1: RAG Pipeline Flow Diagram ----
fig, ax = plt.subplots(figsize=(16, 5))
ax.set_xlim(0, 16)
ax.set_ylim(0, 5)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_title("RAG Pipeline: Retrieval-Augmented Generation End-to-End Flow",
             fontsize=14, fontweight="bold", pad=10)

pipeline_steps = [
    (0.2, "#E3F2FD", "#1565C0", "User\nQuery", 1.8),
    (2.4, "#E8F5E9", "#2E7D32", "Tokenize\n& Embed", 1.8),
    (4.6, "#FFF3E0", "#E65100", "Vector Index\n/ TF-IDF", 1.8),
    (6.8, "#F3E5F5", "#6A1B9A", "Top-K\nDocs", 1.8),
    (9.0, "#FCE4EC", "#880E4F", "Augmented\nPrompt", 1.8),
    (11.2, "#FFFDE7", "#F57F17", "LLM\nGenerate", 1.8),
    (13.4, "#E0F2F1", "#00695C", "Grounded\nAnswer", 1.8),
]

box_h = 2.2
box_y = 1.4

for x, facecolor, edgecolor, label, width in pipeline_steps:
    rect = mpatches.FancyBboxPatch(
        (x, box_y), width, box_h,
        boxstyle="round,pad=0.15",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=2.2
    )
    ax.add_patch(rect)
    ax.text(x + width/2, box_y + box_h/2, label,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color="#333")

# Arrows between steps
arrow_y = box_y + box_h / 2
for i, (x, _, color, _, width) in enumerate(pipeline_steps[:-1]):
    next_x = pipeline_steps[i+1][0]
    ax.annotate("", xy=(next_x - 0.05, arrow_y), xytext=(x + width + 0.05, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=2.0, mutation_scale=16))

# Bottom labels explaining each step
step_notes = [
    "Natural\nlanguage",
    "Convert to\nvectors",
    "Similarity\nsearch",
    "Relevant\ncontext",
    "Context +\nquestion",
    "Read +\nreason",
    "Factually\ngrounded",
]
for (x, _, _, _, width), note in zip(pipeline_steps, step_notes):
    ax.text(x + width/2, box_y - 0.55, note,
            ha="center", va="top", fontsize=7.5, color="#666", style="italic")

ax.text(8.0, 0.15,
        "Grounding: the model reads retrieved facts instead of relying solely on memorized knowledge",
        ha="center", va="bottom", fontsize=9, color="#444", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFDE7", edgecolor="#F9A825", alpha=0.9))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_rag_pipeline.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/01_rag_pipeline.png")

# ---- VIS 2: Retrieval Heatmap ----
fig, ax = plt.subplots(figsize=(14, 5))

# Short labels for docs and queries
doc_labels = [
    "LinearReg", "LogReg", "NeuralNet", "Backprop", "CNN",
    "RNN", "Transformer", "BERT", "LoRA", "RAG"
]
query_labels = [
    "backprop?",
    "BERT vs GPT?",
    "transformers?",
    "LoRA?",
    "RAG hallucination?"
]

im = ax.imshow(score_mat, cmap="Blues", aspect="auto", vmin=0, vmax=score_mat.max())

# Annotate each cell
for qi in range(score_mat.shape[0]):
    for di in range(score_mat.shape[1]):
        val = score_mat[qi, di]
        text_color = "white" if val > score_mat.max() * 0.6 else "black"
        ax.text(di, qi, f"{val:.2f}", ha="center", va="center",
                fontsize=7.5, color=text_color, fontweight="bold" if val > 0.1 else "normal")

ax.set_xticks(range(len(doc_labels)))
ax.set_xticklabels(doc_labels, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(query_labels)))
ax.set_yticklabels(query_labels, fontsize=9)
ax.set_xlabel("Knowledge Base Documents", fontsize=11)
ax.set_ylabel("User Queries", fontsize=11)
ax.set_title("TF-IDF Retrieval Heatmap: Cosine Similarity (Queries × Documents)",
             fontsize=13, fontweight="bold")

cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Cosine Similarity Score", fontsize=10)

# Highlight top-1 doc for each query
for qi in range(score_mat.shape[0]):
    best_di = int(np.argmax(score_mat[qi]))
    rect = mpatches.FancyBboxPatch(
        (best_di - 0.48, qi - 0.48), 0.96, 0.96,
        boxstyle="round,pad=0.05",
        facecolor="none", edgecolor="#FF5722", linewidth=2.5
    )
    ax.add_patch(rect)

ax.text(len(doc_labels) - 0.5, -0.8, "Orange border = top-1 retrieved doc per query",
        ha="right", fontsize=8, color="#FF5722", style="italic")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_retrieval_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/02_retrieval_heatmap.png")

# ---- VIS 3: Chunking Comparison ----
fig, axes = plt.subplots(1, 3, figsize=(15, 7))
fig.suptitle("Document Chunking Strategy Comparison\n(same ~300-word passage split 3 ways)",
             fontsize=13, fontweight="bold")

chunk_data = [
    (fixed_chunks,     "Fixed-Size\n(60 words, 15-word overlap)", "#BBDEFB", "#1565C0"),
    (sentence_chunks,  "Sentence-Based\n(2 sentences per chunk)",  "#C8E6C9", "#2E7D32"),
    (paragraph_chunks, "Paragraph-Based\n(natural boundaries)",    "#FFE0B2", "#E65100"),
]

for ax, (chunks, title, face_color, edge_color) in zip(axes, chunk_data):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(f"{title}\n({len(chunks)} chunks)", fontsize=11, fontweight="bold", pad=8)

    n = len(chunks)
    if n == 0:
        continue

    # Leave some top/bottom padding, distribute chunks evenly
    top_pad = 0.05
    bottom_pad = 0.05
    gap = 0.01
    total_h = 1.0 - top_pad - bottom_pad
    each_h = (total_h - gap * (n - 1)) / n

    # Color gradient across chunks
    cmap_name = ("Blues" if face_color == "#BBDEFB" else
                 "Greens" if face_color == "#C8E6C9" else "Oranges")
    palette = matplotlib.colormaps[cmap_name]
    chunk_colors = [palette(0.3 + 0.5 * i / max(n - 1, 1)) for i in range(n)]

    for ci, (chunk, color) in enumerate(zip(chunks, chunk_colors)):
        # Stack from top down
        y = 1.0 - top_pad - (ci + 1) * each_h - ci * gap
        rect = mpatches.FancyBboxPatch(
            (0.04, y), 0.92, each_h,
            boxstyle="round,pad=0.01",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=1.2,
            alpha=0.85
        )
        ax.add_patch(rect)

        # Chunk label
        ax.text(0.5, y + each_h * 0.78, f"Chunk {ci+1}",
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.1", facecolor=edge_color, alpha=0.7))

        # Truncated text preview
        preview = chunk[:95].replace("\n", " ")
        if len(chunk) > 95:
            preview += "..."
        ax.text(0.5, y + each_h * 0.35, preview,
                ha="center", va="center", fontsize=6.2,
                color="#222", wrap=True,
                bbox=dict(boxstyle="round,pad=0.05", facecolor="white", alpha=0.5))

        # Word count badge
        wc = len(chunk.split())
        ax.text(0.93, y + each_h * 0.85, f"{wc}w",
                ha="right", va="center", fontsize=6, color="#555", style="italic")

    # Summary box at bottom
    avg_words = int(np.mean([len(c.split()) for c in chunks]))
    ax.text(0.5, 0.02,
            f"Avg chunk: ~{avg_words} words",
            ha="center", va="bottom", fontsize=8, color="#444",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=face_color,
                      edgecolor=edge_color, alpha=0.7))

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_chunking_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/03_chunking_comparison.png")

# ==========================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
# ==========================================================================

print("""
Key Takeaways — Retrieval-Augmented Generation
------------------------------------------------
1. RAG solves 3 fundamental LLM problems:
   - Knowledge cutoff: inject fresh information at inference time
   - Hallucination: ground answers in retrieved facts, not memorized patterns
   - Context limits: retrieve only the relevant subset of a large corpus

2. Document chunking is a critical design choice:
   - Too large: irrelevant content dilutes the useful signal
   - Too small: lose necessary context for the answer
   - Sweet spot: 200-500 tokens with 10-15% overlap

3. TF-IDF retrieval from scratch:
   - TF × IDF gives each token a weight: frequent in doc, rare in corpus = useful
   - Cosine similarity ranks docs by relevance
   - Production uses dense embeddings (sentence-transformers) for better semantic match

4. RAG pipeline = retrieve → augment → generate
   - Retriever: TF-IDF, BM25, or dense vector search
   - Prompt template: "Use the following context to answer: [context] [question]"
   - Generator: any LLM (GPT-4, Claude, Llama, etc.)

5. Advanced RAG techniques to know:
   - HyDE: embed a hypothetical answer for better semantic retrieval
   - Reranking: cross-encoder for more accurate final ranking
   - Hybrid: BM25 + dense, combined via Reciprocal Rank Fusion
   - Hierarchical: two-level index for large corpora
   - Query decomposition: split complex questions into sub-queries

Next steps:
  - Try sentence-transformers for dense embeddings
  - Use Faiss or ChromaDB as a vector store
  - Connect to a real LLM API (OpenAI, Anthropic, Ollama) for generation
  - Evaluate with RAGAS: faithfulness, answer relevancy, context precision
""")

print(f"\nVisualizations saved to: {VIS_DIR}/")
print("  01_rag_pipeline.png")
print("  02_retrieval_heatmap.png")
print("  03_chunking_comparison.png")
