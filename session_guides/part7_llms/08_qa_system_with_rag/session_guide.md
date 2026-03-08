# 🎓 MLForBeginners — Instructor Guide
## Part 7 · Module 08: Q&A System with RAG (Project)
### Single 120-Minute Session

> **Prerequisites:** All of Part 7 Modules 01–07.
> They understand LLMs, prompt engineering, RAG pipeline architecture.
> **Payoff:** Build a polished, end-to-end Q&A product using their own documents.
> This is the most immediately practical system in the course — ready to deploy.

---

# SESSION (120 min)
## "Build a Q&A assistant for your own documents"

## Before They Arrive
- Terminal open in `llms/projects/`
- Have 2-3 PDF or text documents ready as the knowledge base
- Great options: course syllabus, a product manual, a recipe collection, company FAQ

---

## OPENING (10 min)

> *\"Imagine this scenario.*
>
> *You're a new employee. You've been handed 200 pages of company documentation.*
> *Policies, procedures, product specs, customer FAQs.*
> *Your first week: 90% of your time is searching those documents.*
>
> *Now imagine: ask any question in plain English, get the answer in 3 seconds,*
> *with the exact page and paragraph cited.*
>
> *That's what we're building today.*
> *Not a toy. A real, production-quality Q&A system*
> *that you can point at ANY document collection.\"*

Write on board:
```
TODAY'S OUTPUT:
  Input:  Any question in plain English
  Output: Accurate answer + source citation

  Powered by:
    Your documents (any PDFs, text, web pages)
    + Embedding model (all-MiniLM-L6-v2)
    + FAISS vector database
    + LLM for synthesis (GPT-4 / Llama 2 / Mistral)
```

---

## SECTION 1: Choosing Your Knowledge Base (10 min)

Each person picks their document set:
```
OPTIONS BY AUDIENCE:
  Students:    Textbook chapters, lecture notes, syllabus
  Developers:  API documentation, code comments, README files
  Business:    Company policies, product specs, customer FAQs
  Personal:    Recipe book, travel notes, research papers
  Fun:         Wikipedia articles on a topic you love

RULE: If you'd normally Ctrl+F through it, RAG can do it better.
```

> *\"The beauty of RAG: the system is completely domain-agnostic.*
> *Medical? Legal? Financial? Engineering?*
> *Same pipeline. Different documents.*
> *You're building infrastructure, not a specific tool.\"*

---

## SECTION 2: The System End to End (30 min)

```bash
python3 qa_system_with_rag.py
```

While it indexes:

> *\"Watch the pipeline run in sequence.*
>
> *Document loading: reading files into memory.*
> *Chunking: splitting into 400-word passages with 50-word overlap.*
> *Embedding: converting each passage to a 384-dimensional vector.*
> *Indexing: loading all vectors into FAISS for fast retrieval.*
>
> *This happens ONCE. Then we can answer unlimited questions instantly.\"*

Once the Q&A loop starts, try questions live:

**Round 1** — Questions with clear answers in the documents:
> *\"Ask something you KNOW is answered in your document.*
> *Watch which passages get retrieved. Is the answer there?\"*

**Round 2** — Questions that require combining information:
> *\"Ask something that requires reading TWO different sections.*
> *The retriever finds relevant chunks from both.*
> *The LLM synthesizes them into one answer.*
> *This is where RAG beats keyword search completely.\"*

**Round 3** — Questions NOT in the documents:
> *\"Ask something that isn't answered anywhere in the knowledge base.*
>
> *A good RAG system says: 'I don't have information about that.'*
> *A bad one hallucinates an answer.*
> *Check: does ours handle 'I don't know' correctly?\"*

---

## SECTION 3: Prompt Template Engineering (20 min)

> *\"The prompt template is where good RAG and bad RAG diverge.*
> *Look at what we're actually sending to the LLM.\"*

Write the template:
```python
PROMPT_TEMPLATE = """
You are a helpful assistant that answers questions
based ONLY on the provided context.

Context:
{retrieved_context}

Question: {user_question}

Instructions:
- Answer ONLY using information from the context above
- If the context doesn't contain enough information, say:
  "I don't have enough information to answer this question."
- Cite specific parts of the context in your answer
- Be concise and direct

Answer:
"""
```

> *\"Three critical phrases:*
> *'Based ONLY on the provided context' — prevents hallucination.*
> *'If the context doesn't contain enough information' — the I don't know path.*
> *'Cite specific parts' — gives the user a way to verify.*
>
> *Remove any of these and the system degrades.*
> *The prompt is as important as the model.\"*

**Ask the room:** *\"What would happen if we removed 'ONLY'?*
*Try it. Change the prompt and ask a question not in the docs.*
*See the difference.\"*

---

## SECTION 4: Evaluation — Is It Actually Working? (20 min)

> *\"How do you know if your RAG system is good?*
> *'It seems fine' isn't good enough for production.\"*

Write the evaluation framework:
```
RAG EVALUATION METRICS:

1. RETRIEVAL PRECISION:
   Of the chunks retrieved, how many are actually relevant?
   (Hand-label 20 queries, check retrieved chunks)

2. RETRIEVAL RECALL:
   Of all relevant chunks, how many were retrieved?
   (Did we miss important paragraphs?)

3. ANSWER FAITHFULNESS:
   Does the answer contain claims NOT in the retrieved context?
   (Hallucination check — LLM judges this)

4. ANSWER RELEVANCE:
   Does the answer actually address the question?
   (Sometimes it answers a related but different question)

5. END-TO-END ACCURACY:
   On 50 known Q&A pairs, what % does the system get right?
   (Requires a labeled eval dataset — expensive but worth it)

TOOLS: RAGAS (open source), LangSmith, Arize Phoenix
```

Run the built-in evaluation on sample questions:

> *\"Faithfulness score: how much is grounded in retrieved text.*
> *Relevance score: how directly the answer addresses the question.*
> *These two numbers tell you most of what you need to know.\"*

---

## CLOSING + PREVIEW (10 min)

> *\"You've built a system that most companies pay $10K-$100K+*
> *to consultants to build for them.*
>
> *You built it this afternoon.*
>
> *The last module — Module 09 — is the final project:*
> *combining an LLM with a classifier to build an intelligent routing system.*
> *Then: the graduation.\"*

Write on board:
```
Q&A WITH RAG — COMPLETE:
  ✅ Custom knowledge base from any documents
  ✅ Semantic search with FAISS
  ✅ Prompt templates that prevent hallucination
  ✅ "I don't know" fallback
  ✅ Source citations in answers
  ✅ Evaluation framework (faithfulness + relevance)

WHERE TO GO FROM HERE:
  → Add a web interface (Streamlit, Gradio)
  → Use GPT-4 or Llama 3 as the generator
  → Add re-ranking for better retrieval precision
  → Deploy to the cloud (Heroku, AWS, GCP)
  → Add document update pipeline (re-index on change)
```

---

## INSTRUCTOR TIPS

**"Can I use this with my company's internal documents?"**
> *"Yes — and this is the most common enterprise AI use case.*
> *For sensitive data: run everything locally (Ollama + Llama 3 + FAISS).*
> *No data ever leaves your machine.*
> *For less sensitive data: OpenAI API for the generation step is fine.*
> *The retrieval (FAISS) is always local.\"*

**"How do I handle PDFs?"**
> *"Use PyMuPDF (fitz) or pdfplumber to extract text.*
> *Then chunk it like any other text.*
> *Challenge: scanned PDFs need OCR first (pytesseract).*
> *Tables and figures don't extract well — known RAG limitation.\"*

**"What's the difference between this and just using ChatGPT?"**
> *"ChatGPT has no access to your specific documents.*
> *If you paste them in, you're limited by context window and paying per token.*
> *RAG: unlimited documents, retrieves only what's needed, always cited, cheaper.*
> *For document Q&A: RAG is almost always the right architecture.\"*

---

## Quick Reference
```
Single Session (120 min)
├── Opening story               10 min
├── Knowledge base selection    10 min
├── System end to end           30 min
├── Prompt template engineering 20 min
├── Evaluation framework        20 min
└── Closing + preview           10 min
```

---
*MLForBeginners · Part 7: LLMs · Module 08*
