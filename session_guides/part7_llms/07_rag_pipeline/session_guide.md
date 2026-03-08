# 🎓 MLForBeginners — Instructor Guide
## Part 7 · Module 07: RAG Pipeline
### Two-Session Teaching Script

> **Prerequisites:** Modules 01–06 of Part 7 (how LLMs work, prompt engineering,
> fine-tuning basics, RAG theory, LLM from scratch, LoRA fine-tuning).
> **Payoff:** Build a working Retrieval-Augmented Generation pipeline from scratch.
> This is the production pattern behind every enterprise AI assistant.

---

# SESSION 1 (~90 min)
## "Why LLMs hallucinate — and how RAG fixes it"

## Before They Arrive
- Terminal open in `llms/algorithms/`
- Prepare a small text document they'll use as the knowledge base (e.g., a product FAQ)
- Write "Knowledge Cutoff" on the board with today's date

---

## OPENING (10 min)

> *\"Ask me anything about events from last week.*
> *(Pause.)*
> *I don't know. My knowledge ends at my training cutoff.*
> *And even before that cutoff, I might confuse details.*
>
> *This is the fundamental problem with language models:*
> *they're frozen in time, and they make things up when they don't know.*
>
> *RAG — Retrieval-Augmented Generation — is the solution.*
> *Instead of asking the model to remember, we GIVE it the relevant information.*
> *In real time. Every query.*
>
> *This is how every production AI assistant works.*
> *ChatGPT with browsing. Perplexity AI. Notion AI. GitHub Copilot.*
> *All RAG under the hood.\"*

Write on board:
```
WITHOUT RAG:                    WITH RAG:
  User: "What's our             User: "What's our
  return policy?"               return policy?"
  LLM: hallucinates             System:
  a plausible-sounding            1. Search docs for "return policy"
  but incorrect answer.           2. Find the actual policy text
                                  3. Pass text + question to LLM
                                  4. LLM answers using REAL text
                                  → Accurate, cited, up-to-date
```

---

## SECTION 1: The RAG Architecture (25 min)

Write the full pipeline on board:

```
RAG PIPELINE:

OFFLINE (build once):
  Your documents (PDFs, docs, web pages)
       ↓
  Chunk into paragraphs (~200-500 words each)
       ↓
  Embed each chunk → dense vector (768d or 1536d)
       ↓
  Store in vector database (FAISS, Pinecone, Chroma)

ONLINE (every query):
  User question
       ↓
  Embed question → query vector
       ↓
  Vector similarity search → top-k most relevant chunks
       ↓
  Build prompt: question + retrieved context
       ↓
  LLM generates answer using retrieved context
       ↓
  Return answer (+ source citations)
```

> *\"Two separate systems working together:*
> *a retriever (finds relevant text) and a generator (writes the answer).*
>
> *The LLM's job is NOT to know the answer.*
> *Its job is to SYNTHESIZE the answer from text we provide.*
> *Reading comprehension, not memorization.\"*

---

## SECTION 2: Vector Embeddings for Search (20 min)

> *\"The magical ingredient: embedding models.*
> *They convert text to dense vectors.*
> *Similar meaning = nearby vectors.*
>
> *'What is the return period?' and 'How many days can I send it back?'*
> *Sound completely different. But their embeddings are close.*
> *Traditional keyword search would miss this.*
> *Vector search finds it.\"*

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

q1 = "What is the return period?"
q2 = "How many days can I send it back?"
q3 = "What is machine learning?"

e1 = model.encode(q1)
e2 = model.encode(q2)
e3 = model.encode(q3)

from numpy import dot
from numpy.linalg import norm

sim_12 = dot(e1, e2) / (norm(e1) * norm(e2))  # ~0.85 (similar)
sim_13 = dot(e1, e3) / (norm(e1) * norm(e3))  # ~0.15 (different)
```

> *\"0.85 similarity between two questions about the same topic.*
> *0.15 similarity between unrelated topics.*
> *The vector knows they mean the same thing, even with different words.*
> *This is semantic search — beyond keywords.\"*

---

## SECTION 3: FAISS — Fast Approximate Nearest Neighbor Search (20 min)

> *\"Vector databases solve one problem: given a query vector,*
> *find the most similar vectors from millions stored.*
>
> *FAISS (Facebook AI Similarity Search) is the most popular library.*
> *It can search 1 million vectors in milliseconds on a laptop.\"*

```python
import faiss
import numpy as np

# Build an index
dimension = 384  # all-MiniLM-L6-v2 output size
index = faiss.IndexFlatL2(dimension)

# Add document embeddings
doc_embeddings = np.array([...]).astype('float32')
index.add(doc_embeddings)

# Search at query time
query_embedding = np.array([...]).astype('float32').reshape(1, -1)
distances, indices = index.search(query_embedding, k=3)  # top 3
```

> *\"The entire knowledge base is indexed once.*
> *Each search: milliseconds.*
> *You can index thousands of documents and still respond in under a second.\"*

---

## CLOSING SESSION 1 (5 min)

```
SESSION 1 SUMMARY:
  RAG = retrieval + generation
  Offline: embed documents → vector database
  Online: embed query → search → retrieve → generate
  Vector search: semantic similarity, not keywords
  FAISS: fast nearest-neighbor search at scale
```

**Homework:** *\"Think of a use case where RAG would be most valuable.*
*A company FAQ? A legal document assistant? A textbook Q&A system?*
*What documents would you use? How would you chunk them?\"*

---

# SESSION 2 (~90 min)
## "Build a complete RAG pipeline — end to end"

## OPENING (5 min)

> *\"Session 1: the theory and the components.*
> *Today: we build the whole thing from scratch.*
> *By the end of this session, you'll have a working Q&A system*
> *that answers questions from YOUR documents.\"*

---

## SECTION 1: Running the Full Pipeline (30 min)

```bash
python3 rag_pipeline.py
```

Walk through each stage as it runs:

```
Stage 1: Load and chunk documents
  → Print: "Loaded 47 chunks from 3 documents"

Stage 2: Embed all chunks
  → Print: "Embedding 47 chunks with all-MiniLM-L6-v2..."
  → Takes 5-10 seconds

Stage 3: Build FAISS index
  → Print: "Index built: 47 vectors at 384 dimensions"

Stage 4: Interactive Q&A
  → "Enter your question: "
```

Try sample questions together:
```
Questions to ask:
  "What is the return policy?"
  "How do I contact customer support?"
  "What payment methods are accepted?"
  "Can I get a refund after 60 days?"  ← should say "no" from policy
```

After each:
> *\"Which chunks were retrieved? (show the top-3)*
> *Did the retrieved chunks contain the answer?*
> *Did the LLM synthesize correctly from those chunks?*
> *Where could this go wrong?\"*

---

## SECTION 2: Chunking Strategies (20 min)

> *\"Chunking is the underappreciated art of RAG.*
> *The best embedding model in the world can't help*
> *if your chunks split the answer across two chunks.\"*

Write the strategies:
```
CHUNKING STRATEGIES:

1. FIXED SIZE (simplest):
   Split every N words/characters
   Pro: simple, predictable
   Con: can cut mid-sentence, mid-thought

2. SENTENCE-BASED:
   Split on sentence boundaries (.!?)
   Pro: coherent chunks
   Con: sentences vary wildly in length

3. PARAGRAPH-BASED (usually best):
   Split on blank lines / section headers
   Pro: natural topic units
   Con: some paragraphs very long/short

4. RECURSIVE (LangChain default):
   Try paragraph → sentence → word until right size
   Pro: adaptive, rarely cuts ideas mid-stream
   Con: more complex

5. SEMANTIC CHUNKING (advanced):
   Split where embedding similarity drops
   Pro: chunks always contain one coherent topic
   Con: expensive, needs embeddings to chunk
```

> *\"For most projects: paragraph-based with ~300-500 word max.*
> *Add 20% overlap between chunks so answers near boundaries aren't missed.\"*

---

## SECTION 3: When RAG Fails — and How to Fix It (20 min)

> *\"RAG isn't magic. Here's where it goes wrong.\"*

```
RAG FAILURE MODES:

1. RETRIEVAL FAILURE: wrong chunks returned
   Fix: better embeddings, smaller chunks,
        re-ranker model (cross-encoder)

2. CONTEXT OVERFLOW: retrieved text too long for LLM
   Fix: limit to top-3 chunks, trim to token limit

3. SYNTHESIS FAILURE: LLM ignores retrieved text
   Fix: better prompt template:
        "Answer ONLY using the context provided.
         If the context doesn't contain the answer, say 'I don't know.'"

4. OUTDATED INDEX: documents updated but index isn't
   Fix: incremental indexing pipeline,
        timestamp-based invalidation

5. QUESTION MISMATCH: user asks in way embedder doesn't recognize
   Fix: query rewriting with LLM before retrieval
        ("turn 'refund?' into 'What is the refund policy?'")
```

---

## CLOSING SESSION 2 (10 min)

```
RAG PIPELINE — COMPLETE:
  Build once: chunk → embed → index (FAISS)
  Query time: embed → search → retrieve → generate

  Key insight: LLM as reader, not memorizer

  Production checklist:
  ✅ Chunking strategy (paragraph-based, ~400 words, 20% overlap)
  ✅ Embedding model (all-MiniLM or text-embedding-ada-002)
  ✅ Vector store (FAISS local, Pinecone/Chroma for production)
  ✅ Prompt template (force LLM to use context)
  ✅ "I don't know" fallback (important for trust)
  ✅ Source citations in response
  ✅ Reranking (optional but helpful)
```

**Next:** *\"Module 08: Q&A System with RAG — we take this pipeline and build*
*a polished, end-to-end Q&A product with a proper interface.\"*

---

## INSTRUCTOR TIPS

**"Why not just put all documents in the context window?"**
> *"GPT-4 has a 128K token context. At 750 words/1000 tokens,*
> *that's ~96,000 words — about 100 pages.*
> *Sounds like a lot, but: it's expensive ($0.01-0.03 per 1K tokens),*
> *LLMs lose accuracy in very long contexts ('lost in the middle' problem),*
> *and you often have MORE than 100 pages.*
> *RAG retrieves only what's needed — faster and cheaper.\"*

**"What's LangChain? Should I use it?"**
> *"LangChain is a framework that wraps all of this:*
> *loaders, splitters, embedders, vector stores, LLM calls.*
> *We built it from scratch so you understand every piece.*
> *In production: LangChain or LlamaIndex are the standard choices.*
> *Never use a framework you don't understand. Now you do.\"*

---

## Quick Reference
```
SESSION 1  (90 min)
├── Opening — why RAG?          10 min
├── Architecture diagram        25 min
├── Vector embeddings           20 min
├── FAISS similarity search     20 min
└── Close + homework             5 min  (+ 10 min buffer)

SESSION 2  (90 min)
├── Opening                      5 min
├── Full pipeline live          30 min
├── Chunking strategies         20 min
├── Failure modes + fixes       20 min
└── Close + next module         10 min  (+ 5 min buffer)
```

---
*MLForBeginners · Part 7: LLMs · Module 07*
