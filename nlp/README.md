# Part 5: Natural Language Processing (NLP)

Learn how machines read, understand, and classify human language — from raw text all the way to production NLP systems. Every concept is built from scratch in NumPy first, then shown with production-grade libraries.

---

## Module Map

| # | File | Topic | Time | Difficulty |
|---|------|--------|------|-----------|
| **Math Foundations** | | | | |
| 1 | `math_foundations/01_text_processing.py` | Tokenization, cleaning, stopwords, stemming/lemmatization, vocabulary | 25 min | Beginner |
| 2 | `math_foundations/02_bag_of_words_tfidf.py` | BoW, TF-IDF, cosine similarity, n-grams | 30 min | Beginner |
| 3 | `math_foundations/03_word_embeddings.py` | Word2Vec Skip-gram, GloVe, analogy arithmetic | 35 min | Intermediate |
| 4 | `math_foundations/04_rnn_intuition.py` | Vanilla RNN, LSTM gates, GRU, vanishing gradient | 40 min | Intermediate |
| **Algorithms** | | | | |
| 5 | `algorithms/text_classification_pipeline.py` | sklearn Pipeline, TF-IDF+LogReg/NB/SVC, cross-validation | 35 min | Intermediate |
| 6 | `algorithms/sentiment_analysis.py` | VADER-style rule-based, TF-IDF+LR, HuggingFace | 40 min | Intermediate |
| 7 | `algorithms/lstm_text_classifier.py` | Embedding→LSTM/BiLSTM/GRU, 4 architecture variants | 45 min | Advanced |
| 8 | `algorithms/named_entity_recognition.py` | BIO tagging, gazetteer NER, CRF features, spaCy | 45 min | Advanced |
| **Projects** | | | | |
| 9 | `projects/movie_review_sentiment.py` | End-to-end sentiment pipeline, error analysis, production API | 60 min | Intermediate |
| 10 | `projects/news_article_classifier.py` | Multi-class classification, TextCNN, macro/micro F1, inference API | 60 min | Intermediate |

---

## Quick Start

```bash
# Install NLP dependencies
pip install scikit-learn tensorflow spacy nltk transformers
python -m spacy download en_core_web_sm

# Run all math foundations (no deep learning required)
cd nlp/math_foundations
python3 01_text_processing.py
python3 02_bag_of_words_tfidf.py
python3 03_word_embeddings.py
python3 04_rnn_intuition.py

# Run algorithm modules
cd ../algorithms
python3 text_classification_pipeline.py
python3 sentiment_analysis.py
python3 lstm_text_classifier.py
python3 named_entity_recognition.py

# Run end-to-end projects
cd ../projects
python3 movie_review_sentiment.py
python3 news_article_classifier.py

# View generated visualizations
open ../visuals/
```

> **No GPU required.** All modules run on CPU. TensorFlow modules take 1–3 minutes on a modern laptop.

---

## What You'll Learn

### Math Foundations

**01 — Text Processing Pipeline**
- Tokenization (word, character, subword)
- Text cleaning: HTML removal, contraction expansion, punctuation
- Stopword filtering, stemming (Porter), lemmatization
- Vocabulary building with special tokens `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
- Zipf's Law: why word frequency follows a power law

**02 — Bag of Words & TF-IDF**
- Document-Term Matrix (DTM) from scratch
- Term Frequency (TF) and Inverse Document Frequency (IDF) — the math
- TF-IDF formula: `tfidf(t,d) = tf(t,d) × (log((1+N)/(1+df(t))) + 1)`
- L2 normalization for cosine similarity
- N-grams: capturing word order in a bag-of-words world

**03 — Word Embeddings**
- Distributional hypothesis: "words that appear in similar contexts have similar meanings"
- Word2Vec Skip-gram from scratch with negative sampling
- Analogy arithmetic: king − man + woman ≈ queen
- Pre-trained embeddings: GloVe, FastText, and why they matter
- PCA visualization of semantic space

**04 — RNN Intuition**
- Vanilla RNN: `h_t = tanh(W_xh × x_t + W_hh × h_{t-1} + b)`
- Vanishing gradient problem: `0.9^50 ≈ 0.005`
- LSTM gates from scratch: forget (f), input (i), output (o), cell state (c)
- GRU: simpler gating with reset and update gates
- Sequence-to-sequence tasks: classification, generation, translation

### Algorithms

**Text Classification Pipeline**
- sklearn `Pipeline` prevents data leakage during cross-validation
- Comparing: Logistic Regression, Complement Naive Bayes, LinearSVC
- When to use each: NB (fast, sparse), LR (interpretable), SVM (high-dimensional)
- Feature importance via coefficient inspection

**Sentiment Analysis**
- Rule-based VADER: lexicon + intensifiers + negation dampening (×−0.74)
- Compound score normalization: `sum / sqrt(sum² + 15)`
- ML approach: TF-IDF + Logistic Regression binary classifier
- HuggingFace `pipeline("sentiment-analysis")` for production
- Aspect-based sentiment: "The acting is great but the story is weak"

**LSTM Text Classifier**
- Text → integers → padded sequences → Embedding layer
- 4 architectures compared: LSTM, BiLSTM, LSTM+GlobalMaxPool, BiGRU
- `mask_zero=True` in Embedding ignores padding during LSTM computation
- `return_sequences=True` enables pooling over all hidden states

**Named Entity Recognition**
- BIO tagging scheme: B-ORG, I-ORG, O
- Rule-based NER: gazetteer matching (3-gram → 2-gram → 1-gram)
- CRF feature engineering: token shape, prefix/suffix, context window
- spaCy production NER with `en_core_web_sm`
- Span-level F1 evaluation (not token accuracy — spans must match exactly)

### Projects

**Movie Review Sentiment** — IMDb-style end-to-end pipeline
- Negation-aware tokenization: "not good" → `NOT_good`
- Combined features: TF-IDF (300d) + lexicon (8d)
- From-scratch Logistic Regression vs sklearn vs BiLSTM
- Production `predict_sentiment()` API with confidence + explanation
- Error analysis: what does the model get wrong and why?

**News Article Classifier** — 5-class production system
- Domain keyword features for each category
- Sublinear TF scaling: `1 + log(tf)` reduces impact of repeated terms
- Softmax classifier from scratch vs OvR/Multinomial LogReg/SVM
- TextCNN: 1D convolutions with filter sizes [2,3,4] for n-gram detection
- Production `classify_article()` API with top-K predictions
- Macro vs Micro F1: when they differ and why it matters

---

## Key Concepts Cheat Sheet

| Concept | Formula / Key Idea |
|---------|-------------------|
| TF-IDF | `tf(t,d) × (log((1+N)/(1+df)) + 1)` |
| Cosine Similarity | `dot(A,B) / (‖A‖ × ‖B‖)` |
| Softmax | `exp(z_k) / Σ exp(z_j)` |
| LSTM Forget Gate | `f_t = σ(W_f × [h_{t-1}, x_t] + b_f)` |
| Vanishing Gradient | `W^T → 0` exponentially over time steps |
| Macro F1 | Mean of per-class F1 (treats classes equally) |
| Micro F1 | F1 from total TP/FP/FN (weighted by class size) |
| BIO Tagging | B=Begin entity, I=Inside entity, O=Outside |

---

## Generated Visualizations

Each module saves 3 plots to `visuals/<module_name>/` at 300 dpi:

```
visuals/
├── 01_text_processing/
│   ├── 01_pipeline_flow.png          — preprocessing steps + word frequency
│   ├── 02_text_statistics.png        — length dist + Zipf's law
│   └── 03_tokenization_comparison.png
├── 02_bag_of_words_tfidf/
│   ├── 01_bow_vs_tfidf_heatmap.png
│   ├── 02_tfidf_analysis.png         — IDF bars + cosine similarity matrix
│   └── 03_ngram_analysis.png
├── 03_word_embeddings/
│   ├── 01_training_loss_pca.png
│   ├── 02_semantic_space.png         — PCA + analogy arrows
│   └── 03_bow_vs_embeddings.png
├── 04_rnn_intuition/
│   ├── 01_vanishing_gradient_lstm.png
│   ├── 02_rnn_architectures.png
│   └── 03_hidden_state_heatmap.png
├── text_classification_pipeline/
│   ├── 01_model_comparison.png
│   ├── 02_confusion_matrix.png
│   └── 03_pipeline_diagram.png
├── sentiment_analysis/
│   ├── 01_scores_roc_comparison.png
│   ├── 02_vader_components.png
│   └── 03_aspect_sentiment.png
├── lstm_text_classifier/
│   ├── 01_data_preparation.png
│   ├── 02_training_histories.png
│   └── 03_architecture_comparison.png
├── named_entity_recognition/
│   ├── 01_bio_tagging.png
│   ├── 02_crf_features.png
│   └── 03_entity_predictions.png
├── movie_review_sentiment/
│   ├── 01_sentiment_overview.png
│   ├── 02_feature_analysis.png
│   └── 03_pipeline_diagram.png
└── news_article_classifier/
    ├── 01_classifier_overview.png
    ├── 02_feature_analysis.png
    └── 03_pipeline_diagram.png
```

---

## Prerequisites

- **Parts 1–3** (Regression, Classification, DNNs) — especially backpropagation and gradient descent
- Python 3.8+, NumPy, Matplotlib

**Optional (graceful fallback if missing):**
- `scikit-learn` — for sklearn comparison models
- `tensorflow` — for LSTM/BiLSTM/TextCNN modules
- `spacy` + `en_core_web_sm` — for production NER
- `transformers` — for HuggingFace sentiment pipeline

---

## Next: Part 6 — Transformers

- Attention mechanism (the "attention is all you need" paper)
- Multi-head attention from scratch
- Positional encoding
- BERT (encoder) and GPT (decoder) with HuggingFace
