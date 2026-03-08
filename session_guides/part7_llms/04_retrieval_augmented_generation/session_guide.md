# MLForBeginners — Instructor Guide
## Part 7 · Module 04: Retrieval-Augmented Generation (RAG)
### Two-Session Teaching Script

> **Prerequisites:** Modules 01–03 complete. They know how LLMs work, understand
> prompt engineering, and have experimented with fine-tuning basics. They also
> know Transformers, BERT, and GPT from Part 6.
> **Payoff today:** They will build a working RAG pipeline from scratch — no
> LLM API required — and understand why every production AI system uses retrieval.

---

# SESSION 1 (~90 min)
## "The problem LLMs can't solve alone — and the cheat sheet that fixes it"

## Before They Arrive
- Terminal open in `llms/math_foundations/`
- Whiteboard ready — draw two columns: "What the LLM Knows" vs "What It Doesn't"
- Have a sticky note or index card visible on the desk (the cheat sheet prop)

---

## OPENING (10 min)

> *"Raise your hand if you've ever asked ChatGPT something and it confidently
> gave you a completely wrong answer. Yeah. We all have.*
>
> *That's not a bug — it's a fundamental limitation. LLMs are probability
> machines trained on a static snapshot of the internet. They don't know
> anything that happened after their training cutoff. And even for things
> they 'know', they can hallucinate — generating plausible-sounding text
> that is simply false.*
>
> *Today we're going to fix that. The fix has a name: RAG.*
>
> *Here's my analogy for the whole session:*
> *Imagine you're taking a closed-book exam. You studied hard, but you might
> misremember details. Now imagine you're allowed to bring a cheat sheet —
> a single page with the most relevant facts. You're still doing the reasoning,
> but now you're grounded in verified information.*
>
> *RAG gives the LLM that cheat sheet. We retrieve the relevant documents
> first, then ask the model to generate conditioned on those facts."*

Hold up the index card on your desk.

> *"This card is the cheat sheet. By the end of session 1, you'll know how
> to build the system that fills it in automatically for any query."*

---

## SECTION 1: The Three Problems RAG Solves (15 min)

Draw on board:
```
PROBLEM 1: KNOWLEDGE CUTOFF
  Training → Fixed snapshot of internet
  GPT-4 cutoff: ~April 2024
  Ask: "Who won the 2025 World Cup?" → Refuses or makes it up

PROBLEM 2: HALLUCINATION
  LLMs are not databases — they generate plausible text
  "Who wrote the LoRA paper?" → might say the wrong authors
  The model doesn't know what it doesn't know

PROBLEM 3: CONTEXT WINDOW LIMITS
  128K tokens sounds like a lot
  A company knowledge base: millions of docs, billions of tokens
  You must be SELECTIVE — you can't stuff everything in
```

> *"Fine-tuning solves problem 1 partially — you can teach the model new
> knowledge by training on it. But it's expensive ($100K+ for large models),
> slow, and causes catastrophic forgetting — the model may lose old knowledge
> as it learns new stuff.*
>
> *RAG solves all three problems with a retrieval step that costs almost nothing."*

Write the RAG formula:
```
Without RAG:  P(answer | query)
              ← model guesses from memory alone

With RAG:     P(answer | query, retrieved_docs)
              ← model reasons over verified retrieved context

Approximation: retrieve top-k docs, generate conditioned on them
               converts free-form generation → reading comprehension
```

> *"Reading comprehension is a much easier task for LLMs than pure recall.
> The retrieved docs act as working memory — facts the model can point to."*

---

## SECTION 2: Document Chunking Strategies (20 min)

> *"Before we can retrieve anything, we need to index our knowledge base.
> Documents are too long to retrieve whole — we split them into chunks.*
>
> *The chunking strategy matters enormously. Get it wrong and your retriever
> will return pieces that are missing crucial context."*

Draw on board:
```
ORIGINAL DOCUMENT (1000 words)
┌─────────────────────────────────────────────────────────┐
│ Introduction ... body ... conclusion ...                 │
└─────────────────────────────────────────────────────────┘

STRATEGY 1: FIXED-SIZE CHUNKS (naive)
  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  │ 200 w  │ │ 200 w  │ │ 200 w  │ │ 200 w  │
  └────────┘ └────────┘ └────────┘ └────────┘
  Pro: Simple. Con: May split mid-sentence.

STRATEGY 2: SENTENCE-BOUNDARY CHUNKS
  ┌──────────────┐ ┌────────────┐ ┌──────────────┐
  │ 2-3 sentences│ │ 2-3 sent.  │ │ 2-3 sentences│
  └──────────────┘ └────────────┘ └──────────────┘
  Pro: Coherent units. Con: Variable length.

STRATEGY 3: PARAGRAPH CHUNKS
  ┌────────────────────────┐ ┌─────────────────────────┐
  │ Full paragraph          │ │ Full paragraph           │
  └────────────────────────┘ └─────────────────────────┘
  Pro: Natural semantic units. Con: Paragraphs vary hugely.
```

> *"There's also overlapping chunks — slide a window so each chunk overlaps
> the next by N words. This helps when an answer spans a chunk boundary.*
>
> *In production systems like LlamaIndex, you tune chunk size just like
> you tune hyperparameters. For this module, we'll implement all three
> strategies from scratch."*

**Ask the room:** *"If you're building a RAG system over a legal document,
which chunking strategy would you pick, and why?"*

Let a few people answer. Look for answers about preserving legal sentence
structure (sentence-boundary) or paragraph-level clauses.

---

## SECTION 3: TF-IDF Retriever from Scratch (30 min)

> *"Now the retriever. We need a way to score how relevant each chunk is
> to a query. We're going to build a TF-IDF retriever — you've seen TF-IDF
> in Part 5 NLP. Now we're weaponizing it as a search engine."*

Write the formulas on the board:
```
TF(t, d)  = count of term t in document d / total terms in d
IDF(t)    = log(N / df(t))
            N  = total number of documents
            df = number of docs containing t

TF-IDF(t, d) = TF(t, d) × IDF(t)

RETRIEVAL SCORE = cosine_similarity(query_vector, doc_vector)
                = (q · d) / (||q|| × ||d||)
```

> *"IDF rewards terms that are rare across the corpus — they are more
> discriminative. TF rewards terms that appear frequently in a specific
> document. Together they identify documents that are specifically about
> your query topic.*
>
> *Common words like 'the', 'is', 'and' get a near-zero IDF score and
> barely influence retrieval. That's the desired behavior."*

Live demo:
```bash
python3 04_retrieval_augmented_generation.py
```

Watch the output for:
- The document chunks being created
- The TF-IDF scores for a sample query
- Top-k retrieved chunks per query

Point at the output:

> *"Look at the scores. The retriever found the most relevant chunk in
> under a millisecond. No GPU, no API call, no embedding model needed.
> This is the baseline — and it performs surprisingly well."*

---

## SECTION 4: Evaluating Retrieval Quality — Precision@k (10 min)

Draw on board:
```
QUERY: "What is gradient descent?"
RELEVANT DOCS: [doc_3, doc_7, doc_12]

Retriever returns: [doc_3, doc_9, doc_7, doc_1, doc_12]
                    ↑ hit    miss   ↑ hit   miss   ↑ hit

Precision@1 = hits in top 1 / 1 = 1/1 = 1.00
Precision@3 = hits in top 3 / 3 = 2/3 = 0.67
Precision@5 = hits in top 5 / 5 = 3/5 = 0.60
```

> *"Precision@k tells you: of the top k documents I retrieved, what fraction
> were actually relevant? It's the primary metric for retrieval evaluation.*
>
> *A random retriever on 20 documents would have P@1 ≈ 0.05 on average.
> Our TF-IDF achieves P@1 ≈ 0.80+. That's the difference retrieval makes."*

---

## CLOSING SESSION 1 (5 min)

Board summary:
```
RAG = CHEAT SHEET FOR THE MODEL
  Problem:    LLMs hallucinate, have cutoffs, can't hold all context
  Solution:   Retrieve relevant docs first, generate conditioned on them
  Chunking:   Split docs into fixed / sentence / paragraph chunks
  Retriever:  TF-IDF cosine similarity = fast, no GPU needed
  Measure:    Precision@k = fraction of top-k that are truly relevant
```

**Homework:** Write a two-sentence answer to: "Why does IDF penalize common
words, and why is that the right behavior for document retrieval?"

---

# SESSION 2 (~90 min)
## "End-to-end RAG pipeline and advanced techniques"

## OPENING (10 min)

> *"Last session we built the retriever — the component that finds relevant
> chunks for any query. Today we wire everything together into a complete
> RAG pipeline.*
>
> *Then we'll look at what happens when TF-IDF isn't enough — and preview
> the advanced techniques used in production systems: HyDE, reranking,
> and hybrid retrieval."*

---

## SECTION 1: The Full RAG Pipeline (25 min)

Draw on board:
```
INDEXING (offline, done once):
  Knowledge Base → Chunker → TF-IDF Index
                                    ↓
                             [doc1_vec, doc2_vec, ...]

QUERYING (online, per request):
  Query → TF-IDF vectorize → cosine_sim(query, all_docs) → top-k chunks
        → Build prompt: "Context:\n{chunks}\nAnswer: {query}"
        → LLM (or our simulated generator) → Answer
```

> *"The indexing step is offline — you do it once when your knowledge base
> changes. The querying step is fast — milliseconds. That's the architecture
> of every production RAG system: LlamaIndex, LangChain, Haystack.*
>
> *We're building the same pipeline, just without the paid LLM API at the end.
> Our 'generator' will be a template-based response — the retrieval logic
> is identical."*

Code walkthrough — trace through the module together:
```python
# Step 1: Index documents
retriever.index(chunks)        # builds TF-IDF matrix

# Step 2: Retrieve for a query
query = "How does attention work?"
results = retriever.retrieve(query, top_k=3)

# Step 3: Augment prompt
context = "\n\n".join([r["text"] for r in results])
prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

# Step 4: Generate (simulated)
answer = generate(prompt, results)
```

**Ask the room:** *"What would happen if we skip step 3 and just send the raw
query to the LLM? What are we losing?"*

Expected answer: The LLM would rely on parametric memory, could hallucinate,
and wouldn't have access to private or up-to-date documents.

---

## SECTION 2: Advanced RAG — HyDE and Reranking (20 min)

> *"TF-IDF has a core weakness: it matches exact words, not meaning.
> Query: 'How do neural networks learn?' — will miss documents that say
> 'backpropagation updates weights to minimize loss' because the words
> don't overlap.*
>
> *Advanced RAG addresses this. Let me show you three techniques."*

Draw on board:
```
TECHNIQUE 1: HyDE (Hypothetical Document Embeddings)
  Problem:  Query and relevant docs may use different vocabulary
  Solution: Ask LLM to generate a HYPOTHETICAL answer to the query
            Retrieve docs similar to that hypothetical answer
            Query:       "How do neural networks learn?"
            Hypothetical: "Neural networks learn by adjusting weights using
                           gradient descent and backpropagation..."
            Now retrieve based on the hypothetical — much better word overlap!

TECHNIQUE 2: RERANKING (cross-encoder)
  Two-stage:
    Stage 1: TF-IDF retrieves top-50 quickly (recall)
    Stage 2: Cross-encoder scores query-doc pairs carefully (precision)
  Cross-encoder reads query + doc together → much better relevance score
  Slower per document, but only applied to the top-50 candidates

TECHNIQUE 3: HYBRID RETRIEVAL
  Combine TF-IDF (keyword) + dense embeddings (semantic similarity)
  Score = alpha × sparse_score + (1-alpha) × dense_score
  Best of both worlds: exact match + semantic understanding
```

> *"In production, you almost always use at least technique 2 — retrieval then
> reranking. The open-source Cohere Rerank, CrossEncoder from sentence-transformers,
> and Vertex AI Search all implement this pattern."*

**Ask the room:** *"If HyDE generates a hypothetical answer using the LLM,
and then retrieves based on that — what's the risk? When could this go wrong?"*

Expected answer: If the LLM generates a hallucinated hypothetical, the retrieval
will find irrelevant documents that match the hallucination.

---

## SECTION 3: Viewing the Visualizations (15 min)

```bash
# Visuals are in: llms/visuals/04_retrieval_augmented_generation/
open llms/visuals/04_retrieval_augmented_generation/
```

Walk through each visualization together. For each one, ask students what they
see before explaining.

Expected visuals:
- RAG pipeline architecture diagram
- Chunk size vs retrieval quality curve
- TF-IDF score heatmap (query × document)
- Precision@k bar chart

> *"This precision@k chart is the bottom line. TF-IDF with good chunking
> achieves precision@1 around 0.80 on well-structured knowledge bases.
> Adding a reranker typically pushes that to 0.90+. Adding dense embeddings
> (semantic search) can reach 0.95+ for general queries."*

---

## SECTION 4: When to Use RAG vs Fine-tuning (15 min)

Draw the decision table:
```
USE CASE                        RAG     FINE-TUNING
----------------------------------------------------
Private company knowledge       YES     No
Up-to-date information          YES     No (expensive to retrain)
New writing style / tone        No      YES
Domain-specific behavior        Partial YES
Long-document QA                YES     Partial
Reducing hallucination          YES     Partial
Low latency required            Partial YES (no retrieval overhead)
Interpretability / citations    YES     No
```

> *"The key insight: RAG and fine-tuning are complementary, not competing.
> Production systems often use BOTH:*
> *1. Fine-tune the model on domain-specific writing style*
> *2. Use RAG to ground its answers in up-to-date facts*
>
> *Think of it as: fine-tuning teaches the model HOW to talk,
> RAG gives it the FACTS to talk about."*

---

## CLOSING SESSION 2 (5 min)

Board summary:
```
COMPLETE RAG PIPELINE:
  1. Chunk documents          (fixed / sentence / paragraph)
  2. Build TF-IDF index       (offline, once)
  3. Retrieve top-k chunks    (cosine similarity, milliseconds)
  4. Augment prompt           ("Context: {chunks}\nQuestion: {q}")
  5. Generate answer          (grounded, verifiable)

ADVANCED RAG:
  HyDE         → hypothetical doc to bridge vocab mismatch
  Reranking    → two-stage: recall then precision
  Hybrid       → sparse + dense retrieval combined

WHEN TO USE:
  RAG          → fresh facts, private knowledge, citations
  Fine-tuning  → style, behavior, domain-specific skills
  Both         → production systems
```

**Homework:** Pick a topic you know well. Write 5 short "documents" about it
(3-4 sentences each). Build the TF-IDF retriever from this module and query it.
Report back: what queries work well? Where does it fail?

---

## INSTRUCTOR TIPS

**"How is RAG different from just searching Google?"**
> *"Great question. The difference is the generation step. Google gives you
> links; RAG synthesizes an answer conditioned on the retrieved text.
> RAG is: search + comprehension + synthesis, all in one step."*

**"Do we always need an LLM for the generation step?"**
> *"No! For structured QA tasks, you can extract the answer directly from
> the retrieved chunks using a simple extractive reader — no generation needed.
> This is cheaper, faster, and more reliable for factual questions."*

**"What's the difference between RAG and a search engine with snippets?"**
> *"A search engine shows you snippets; a RAG system READS those snippets
> and ANSWERS your question in natural language, potentially combining
> information from multiple retrieved documents into a single coherent answer."*

**"Isn't TF-IDF too simple for real production use?"**
> *"TF-IDF is the baseline. ElasticSearch uses BM25 (a TF-IDF variant) in
> production for billions of documents. For small knowledge bases (<10K docs),
> TF-IDF with good chunking often matches fancy embedding models.
> Always start simple, then upgrade when you have evidence it's needed."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                    10 min
├── Three problems RAG solves       15 min
├── Document chunking strategies    20 min
├── TF-IDF retriever from scratch   30 min
├── Precision@k evaluation          10 min
└── Close + homework                 5 min

SESSION 2  (90 min)
├── Opening bridge                  10 min
├── Full RAG pipeline end-to-end    25 min
├── Advanced RAG: HyDE + reranking  20 min
├── Viewing visualizations          15 min
├── RAG vs fine-tuning              15 min
└── Close + homework                 5 min
```

---
*MLForBeginners · Part 7: LLMs · Module 04*
