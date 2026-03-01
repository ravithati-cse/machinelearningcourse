"""
Q&A System with RAG — End-to-End Project
=========================================
Learning Objectives:
  1. Build a production-ready Q&A system using RAG
  2. Create and index a multi-topic ML knowledge base
  3. Implement query preprocessing and intent classification
  4. Add answer confidence scoring and fallback handling
  5. Evaluate system performance with held-out test questions
  6. Deploy a clean answer() API function
YouTube: Search "RAG Q&A system build production LangChain LlamaIndex"
Time: ~60 min | Difficulty: Advanced | Prerequisites: LLMs math_foundations, rag_pipeline
"""

import os
import math
import numpy as np
from collections import defaultdict

# --------------------------------------------------------------------------
# Visualization setup — Agg backend BEFORE pyplot import
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "qa_system_with_rag")
os.makedirs(VIS_DIR, exist_ok=True)

# ==========================================================================
print("\n" + "="*70)
print("SECTION 1: KNOWLEDGE BASE (20 ML Articles, 5 Topics)")
print("="*70)
# ==========================================================================

KNOWLEDGE_BASE = [
    # ── Topic: fundamentals ──────────────────────────────────────────────
    {
        "id": 0,
        "title": "Gradient Descent",
        "topic": "fundamentals",
        "text": (
            "Gradient descent is an iterative optimization algorithm used to minimize a "
            "loss function by updating model parameters in the direction of the negative "
            "gradient. At each step, parameters are updated as theta = theta - lr * "
            "grad(L). The learning rate controls step size: too large and the optimizer "
            "overshoots; too small and convergence is slow. Variants include batch "
            "gradient descent, stochastic gradient descent (SGD), and mini-batch SGD. "
            "Momentum, RMSProp, and Adam extend plain SGD by accumulating gradient "
            "history to accelerate convergence and reduce oscillation in narrow valleys."
        ),
    },
    {
        "id": 1,
        "title": "Loss Functions",
        "topic": "fundamentals",
        "text": (
            "A loss function measures how far a model's predictions are from the true "
            "labels. Mean Squared Error (MSE) is used for regression: L = (1/n) sum "
            "(y_pred - y_true)^2. Cross-entropy loss is used for classification: "
            "L = -sum y_true * log(y_pred). Huber loss is robust to outliers by "
            "combining MSE and MAE. The choice of loss function determines what the "
            "model optimizes: using the wrong loss can cause training to diverge or "
            "produce poorly calibrated probability estimates."
        ),
    },
    {
        "id": 2,
        "title": "Overfitting",
        "topic": "fundamentals",
        "text": (
            "Overfitting occurs when a model learns the training data too well, "
            "capturing noise alongside the true signal, so it generalizes poorly to "
            "unseen data. Signs include very low training loss but high validation loss. "
            "Common remedies are: collecting more data, applying L1/L2 regularization, "
            "using dropout in neural networks, reducing model complexity, and early "
            "stopping. Overfitting is more likely with high-capacity models trained on "
            "small datasets. The bias-variance tradeoff formalizes this: overfit models "
            "have low bias but high variance."
        ),
    },
    {
        "id": 3,
        "title": "Cross-Validation",
        "topic": "fundamentals",
        "text": (
            "Cross-validation is a technique for estimating generalization performance "
            "by splitting the dataset into multiple train/validation folds. In k-fold "
            "cross-validation, data is split into k equal parts; the model trains on "
            "k-1 folds and validates on the remaining fold, rotating until every fold "
            "has been the validation set. The final score is the average across folds. "
            "Stratified k-fold preserves class proportions in each fold, important for "
            "imbalanced datasets. Leave-one-out cross-validation (LOOCV) uses a single "
            "sample as the validation set each time."
        ),
    },
    {
        "id": 4,
        "title": "Bias-Variance Tradeoff",
        "topic": "fundamentals",
        "text": (
            "The bias-variance tradeoff describes the tension between two sources of "
            "prediction error. Bias is error from incorrect assumptions: a high-bias "
            "model underfits, missing relevant patterns. Variance is error from "
            "sensitivity to fluctuations in training data: a high-variance model "
            "overfits. Total expected error = Bias^2 + Variance + irreducible noise. "
            "Increasing model complexity typically decreases bias but increases variance. "
            "Regularization, ensemble methods, and cross-validation help find the "
            "sweet spot that minimizes total error on unseen data."
        ),
    },
    # ── Topic: neural_networks ───────────────────────────────────────────
    {
        "id": 5,
        "title": "Perceptrons",
        "topic": "neural_networks",
        "text": (
            "A perceptron is the simplest form of a neural network — a single "
            "computational unit that computes a weighted sum of inputs and passes it "
            "through a step activation function: output = 1 if (w · x + b) > 0 else 0. "
            "Perceptrons can learn linearly separable functions (AND, OR) but fail on "
            "XOR because it is not linearly separable. The perceptron learning rule "
            "updates weights only when a prediction error occurs: w += lr * (y - y_pred) * x. "
            "Multilayer perceptrons (MLPs) stack multiple layers to overcome the linear "
            "limitation."
        ),
    },
    {
        "id": 6,
        "title": "Activation Functions",
        "topic": "neural_networks",
        "text": (
            "Activation functions introduce non-linearity into neural networks, enabling "
            "them to learn complex mappings. ReLU (Rectified Linear Unit) outputs "
            "max(0, x) and is computationally efficient but suffers from the dying ReLU "
            "problem. Sigmoid squashes output to (0, 1) and is used for binary "
            "classification outputs. Tanh outputs (-1, 1) and is zero-centered. GELU "
            "and Swish are smooth approximations used in transformers and modern "
            "architectures. Softmax converts a vector of logits into a probability "
            "distribution for multi-class classification outputs."
        ),
    },
    {
        "id": 7,
        "title": "Backpropagation",
        "topic": "neural_networks",
        "text": (
            "Backpropagation is the algorithm that computes gradients of the loss with "
            "respect to every parameter in a neural network by applying the chain rule "
            "of calculus. The forward pass computes predictions and caches intermediate "
            "activations. The backward pass propagates the error gradient layer by layer "
            "from output to input. For a weight w_ij: dL/dw_ij = dL/da_j * da_j/dz_j * "
            "dz_j/dw_ij. Vanishing gradients occur when gradients shrink exponentially "
            "in deep networks; exploding gradients occur when they grow. Solutions "
            "include residual connections, batch normalization, and gradient clipping."
        ),
    },
    {
        "id": 8,
        "title": "Batch Normalization",
        "topic": "neural_networks",
        "text": (
            "Batch normalization normalizes the pre-activation values within a mini-batch "
            "to have zero mean and unit variance, then applies learnable scale (gamma) "
            "and shift (beta) parameters: y = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta. "
            "This stabilizes training by reducing internal covariate shift, allows "
            "higher learning rates, and acts as a mild regularizer. During inference, "
            "running statistics computed during training replace the batch statistics. "
            "Layer normalization (LayerNorm) normalizes across feature dimensions instead "
            "of batch dimension, making it suitable for recurrent and transformer models."
        ),
    },
    {
        "id": 9,
        "title": "Dropout",
        "topic": "neural_networks",
        "text": (
            "Dropout is a regularization technique that randomly sets a fraction p of "
            "neuron activations to zero during each training step. This prevents "
            "neurons from co-adapting and forces the network to learn redundant "
            "representations, reducing overfitting. During inference, dropout is "
            "disabled and all activations are scaled by (1 - p) to compensate. "
            "Typical dropout rates are 0.1 to 0.5. Dropout can be interpreted as "
            "training an ensemble of 2^n sub-networks that share weights. Spatial "
            "dropout drops entire feature maps in CNNs rather than individual neurons."
        ),
    },
    # ── Topic: transformers ──────────────────────────────────────────────
    {
        "id": 10,
        "title": "Attention Mechanism",
        "topic": "transformers",
        "text": (
            "The attention mechanism allows a model to focus on relevant parts of the "
            "input when producing each output. Scaled dot-product attention computes "
            "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V, where Q, K, V are "
            "query, key, and value matrices. Dividing by sqrt(d_k) prevents the dot "
            "products from growing too large and saturating the softmax. Self-attention "
            "lets each token attend to all other tokens in the same sequence. Cross-"
            "attention lets the decoder attend to encoder outputs. Causal masking "
            "sets future positions to negative infinity, enabling autoregressive generation."
        ),
    },
    {
        "id": 11,
        "title": "BERT",
        "topic": "transformers",
        "text": (
            "BERT (Bidirectional Encoder Representations from Transformers) is a "
            "transformer encoder pre-trained with two objectives: masked language "
            "modeling (MLM) and next sentence prediction (NSP). MLM randomly masks 15% "
            "of input tokens and trains the model to predict them using full bidirectional "
            "context. The [CLS] token prepended to every input serves as a sequence-level "
            "representation. BERT is fine-tuned by adding task-specific heads. BERT-base "
            "has 12 layers, 12 attention heads, and 110M parameters. It set state-of-the-art "
            "on 11 NLP benchmarks when released in 2018."
        ),
    },
    {
        "id": 12,
        "title": "GPT",
        "topic": "transformers",
        "text": (
            "GPT (Generative Pre-trained Transformer) uses a decoder-only transformer "
            "trained autoregressively to predict the next token given all preceding tokens. "
            "Unlike BERT, GPT uses causal (left-to-right) self-attention so each position "
            "only attends to previous positions. Pre-training on large text corpora gives "
            "GPT strong generative capability. GPT-3 scaled to 175 billion parameters and "
            "demonstrated impressive few-shot and zero-shot performance. InstructGPT and "
            "ChatGPT further aligned the model with human preferences using RLHF (Reinforcement "
            "Learning from Human Feedback)."
        ),
    },
    {
        "id": 13,
        "title": "Positional Encoding",
        "topic": "transformers",
        "text": (
            "Transformers process all input tokens in parallel and have no inherent notion "
            "of word order, so positional encodings are added to token embeddings to inject "
            "sequence position information. Sinusoidal positional encoding uses sine and "
            "cosine functions of different frequencies: PE(pos, 2i) = sin(pos / 10000^(2i/d)), "
            "PE(pos, 2i+1) = cos(pos / 10000^(2i/d)). Learned positional embeddings are "
            "trained end-to-end and used in BERT and GPT. Rotary Position Embedding (RoPE) "
            "encodes position by rotating query and key vectors, enabling better length "
            "extrapolation."
        ),
    },
    {
        "id": 14,
        "title": "Tokenization",
        "topic": "transformers",
        "text": (
            "Tokenization converts raw text into discrete tokens that the model can process. "
            "Word-level tokenization suffers from large vocabularies and out-of-vocabulary "
            "issues. Character-level tokenization has tiny vocabularies but long sequences. "
            "Byte-Pair Encoding (BPE) iteratively merges the most frequent character pairs "
            "into subword tokens, balancing vocabulary size and sequence length. WordPiece "
            "(used in BERT) and SentencePiece (used in T5, Llama) are similar subword "
            "algorithms. GPT-2 uses BPE on raw bytes, handling any Unicode text without "
            "special unknown tokens."
        ),
    },
    # ── Topic: training ──────────────────────────────────────────────────
    {
        "id": 15,
        "title": "Learning Rate Schedules",
        "topic": "training",
        "text": (
            "A learning rate schedule adjusts the learning rate during training to improve "
            "convergence. Constant learning rate is simplest but rarely optimal. Step decay "
            "reduces the learning rate by a factor at fixed epochs. Cosine annealing smoothly "
            "decays the learning rate following a cosine curve, often reaching near-zero before "
            "resetting (warm restarts). Linear warmup then cosine decay is the standard schedule "
            "for transformer pre-training: the rate ramps from 0 to peak over a warmup period, "
            "then decays. Cyclical learning rates alternate between low and high values to "
            "escape sharp minima."
        ),
    },
    {
        "id": 16,
        "title": "Batch Size",
        "topic": "training",
        "text": (
            "Batch size is the number of training examples processed before updating model "
            "weights. Small batches (8-32) introduce gradient noise that can act as regularization "
            "and help escape sharp minima, but are slower per epoch. Large batches (512-8192) "
            "compute more accurate gradients and are faster on modern hardware, but can converge "
            "to sharp, poorly generalizing minima (the large-batch generalization gap). Linear "
            "scaling rule: when multiplying batch size by k, multiply the learning rate by k. "
            "Gradient accumulation simulates large batches on memory-limited hardware."
        ),
    },
    {
        "id": 17,
        "title": "Weight Decay",
        "topic": "training",
        "text": (
            "Weight decay is an L2 regularization technique that adds a penalty proportional "
            "to the squared magnitude of weights to the loss: L_reg = L + lambda * ||w||^2. "
            "This discourages large weights and reduces overfitting. In SGD, weight decay is "
            "equivalent to L2 regularization. In Adam, they are not equivalent: AdamW "
            "decouples weight decay from the adaptive learning rate scaling, which improves "
            "regularization. Typical weight decay values range from 1e-4 to 0.1. Biases "
            "and LayerNorm parameters are usually excluded from weight decay."
        ),
    },
    {
        "id": 18,
        "title": "Early Stopping",
        "topic": "training",
        "text": (
            "Early stopping halts training when validation performance stops improving, "
            "preventing overfitting. A patience parameter specifies how many epochs to wait "
            "after the last improvement before stopping. The best model checkpoint is saved "
            "and restored at the end. Early stopping effectively regularizes the model by "
            "limiting training iterations. It is commonly combined with learning rate "
            "scheduling. The optimal stopping point balances the tradeoff between underfitting "
            "(stopping too early) and overfitting (stopping too late). It is one of the most "
            "practical regularization tools available."
        ),
    },
    {
        "id": 19,
        "title": "Mixed Precision Training",
        "topic": "training",
        "text": (
            "Mixed precision training uses 16-bit floating point (FP16 or BF16) for most "
            "computations and 32-bit (FP32) for critical accumulations, reducing memory usage "
            "and increasing throughput on modern GPUs with tensor cores. A master copy of "
            "weights is kept in FP32 to preserve numerical precision during updates. Loss "
            "scaling prevents FP16 underflow by multiplying the loss before the backward pass "
            "and dividing gradients afterward. BF16 has the same range as FP32 and is more "
            "numerically stable than FP16. Mixed precision can cut memory usage by ~2x and "
            "training time by 2-3x."
        ),
    },
    # ── Topic: deployment ────────────────────────────────────────────────
    {
        "id": 20,
        "title": "Model Serving",
        "topic": "deployment",
        "text": (
            "Model serving deploys a trained model behind an API so applications can request "
            "predictions in real time. Common frameworks include TensorFlow Serving, TorchServe, "
            "FastAPI, and Triton Inference Server. Key metrics are latency (time per request), "
            "throughput (requests per second), and availability. Batching multiple requests "
            "together improves GPU utilization at the cost of slightly higher latency. Caching "
            "frequent inputs avoids redundant computation. Horizontal scaling adds more serving "
            "replicas to handle traffic spikes. Model versioning ensures safe rollouts and "
            "rollbacks."
        ),
    },
    {
        "id": 21,
        "title": "ONNX Export",
        "topic": "deployment",
        "text": (
            "ONNX (Open Neural Network Exchange) is an open format for representing machine "
            "learning models, enabling interoperability between frameworks. A PyTorch or "
            "TensorFlow model can be exported to ONNX and then run with ONNX Runtime, which "
            "applies graph optimizations (operator fusion, constant folding) that reduce "
            "inference latency by 2-4x. ONNX Runtime supports CPU, GPU, and edge devices. "
            "Dynamic axes allow the model to handle variable-length inputs. The export process "
            "traces the model's computation graph using a sample input."
        ),
    },
    {
        "id": 22,
        "title": "Quantization",
        "topic": "deployment",
        "text": (
            "Quantization reduces model size and inference latency by representing weights "
            "and activations in lower-bit formats. Post-training quantization (PTQ) converts "
            "a trained FP32 model to INT8 without retraining, using a small calibration "
            "dataset to determine scale factors. Quantization-aware training (QAT) simulates "
            "quantization during training, producing more accurate quantized models. INT8 "
            "models are typically 4x smaller than FP32 and run 2-4x faster on hardware "
            "with INT8 support. Dynamic range quantization only quantizes weights, not "
            "activations, and is the simplest to apply."
        ),
    },
    {
        "id": 23,
        "title": "A/B Testing",
        "topic": "deployment",
        "text": (
            "A/B testing in ML deployment routes a fraction of live traffic to a new model "
            "candidate while the remainder goes to the current production model. Statistical "
            "significance tests (t-test, chi-squared) determine whether observed differences "
            "in metrics are real. Key metrics to track include accuracy, latency, click-through "
            "rate, and business KPIs. Shadow mode runs the new model on live traffic but "
            "ignores its outputs, allowing safe testing without user impact. Canary deployments "
            "gradually increase traffic to the new model as confidence grows."
        ),
    },
    {
        "id": 24,
        "title": "Model Monitoring",
        "topic": "deployment",
        "text": (
            "Model monitoring tracks model performance in production to detect degradation "
            "over time. Data drift occurs when the statistical distribution of incoming data "
            "shifts from the training distribution, causing accuracy to drop. Concept drift "
            "occurs when the relationship between features and the target changes. Monitoring "
            "tools track prediction distributions, input feature statistics, and outcome "
            "metrics. Alerts trigger retraining pipelines when drift exceeds thresholds. "
            "Logging inputs and predictions enables offline analysis and debugging. Responsible "
            "monitoring also checks for fairness metrics across demographic groups."
        ),
    },
]

print(f"Knowledge base: {len(KNOWLEDGE_BASE)} documents across 5 topics")
topic_counts = defaultdict(int)
for doc in KNOWLEDGE_BASE:
    topic_counts[doc["topic"]] += 1
for topic, count in sorted(topic_counts.items()):
    print(f"  {topic:20s}: {count} documents")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 2: TF-IDF INDEX WITH METADATA FILTERING")
print("="*70)
# ==========================================================================

import re

def _tokenize(text):
    """Simple whitespace + punctuation tokenizer."""
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "to", "of", "in", "for", "on",
        "with", "at", "by", "from", "as", "it", "its", "this", "that",
        "and", "or", "but", "so", "if", "not", "no", "nor", "yet",
    }
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    return [t for t in tokens if t not in stopwords and len(t) > 1]


class TFIDFRetriever:
    """TF-IDF retriever with optional topic metadata filtering."""

    def __init__(self):
        self.docs = []
        self.vocab = {}
        self.idf = {}
        self.doc_vectors = []

    def fit(self, docs):
        """Build TF-IDF index over a list of dicts or strings."""
        self.docs = docs
        texts = [d["text"] if isinstance(d, dict) else d for d in docs]
        # Build vocabulary
        all_tokens = set()
        tok_lists = [_tokenize(t) for t in texts]
        for tl in tok_lists:
            all_tokens.update(tl)
        self.vocab = {w: i for i, w in enumerate(sorted(all_tokens))}
        V = len(self.vocab)
        N = len(texts)
        # Document frequency
        df = np.zeros(V)
        for tl in tok_lists:
            unique = set(tl)
            for tok in unique:
                if tok in self.vocab:
                    df[self.vocab[tok]] += 1
        # IDF with smoothing
        self.idf = np.log((N + 1) / (df + 1)) + 1.0
        # TF-IDF vectors
        self.doc_vectors = []
        for tl in tok_lists:
            tf = np.zeros(V)
            for tok in tl:
                if tok in self.vocab:
                    tf[self.vocab[tok]] += 1
            if len(tl) > 0:
                tf = tf / len(tl)
            vec = tf * self.idf
            norm = np.linalg.norm(vec)
            self.doc_vectors.append(vec / norm if norm > 0 else vec)
        print(f"  TF-IDF index built: {N} docs, {V} vocab terms")

    def _query_vector(self, query):
        V = len(self.vocab)
        tl = _tokenize(query)
        tf = np.zeros(V)
        for tok in tl:
            if tok in self.vocab:
                tf[self.vocab[tok]] += 1
        if len(tl) > 0:
            tf = tf / len(tl)
        vec = tf * self.idf
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def retrieve(self, query, top_k=3, filter_topic=None):
        """
        Retrieve top_k documents matching query.
        Optionally restrict to a single topic.
        Returns list of (doc_index, cosine_score) tuples.
        """
        q_vec = self._query_vector(query)
        scores = []
        for i, dv in enumerate(self.doc_vectors):
            if filter_topic is not None:
                doc = self.docs[i]
                if isinstance(doc, dict) and doc.get("topic") != filter_topic:
                    continue
            score = float(q_vec @ dv)
            scores.append((i, score))
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


# Build index on full knowledge base
retriever = TFIDFRetriever()
retriever.fit(KNOWLEDGE_BASE)

# Demonstrate filtered retrieval
query_demo = "How does attention work?"
print(f"\nFiltered retrieval: '{query_demo}' (topic='transformers')")
results_filtered = retriever.retrieve(query_demo, top_k=3, filter_topic="transformers")
for idx, score in results_filtered:
    print(f"  [{score:.3f}] {KNOWLEDGE_BASE[idx]['title']} — {KNOWLEDGE_BASE[idx]['topic']}")

print(f"\nUnfiltered retrieval: '{query_demo}'")
results_all = retriever.retrieve(query_demo, top_k=3)
for idx, score in results_all:
    print(f"  [{score:.3f}] {KNOWLEDGE_BASE[idx]['title']} — {KNOWLEDGE_BASE[idx]['topic']}")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 3: QUERY PREPROCESSING")
print("="*70)
# ==========================================================================

def preprocess_query(query):
    """Clean, lowercase, and expand common ML abbreviations."""
    abbrevs = {
        "nn":   "neural network",
        "lr":   "learning rate",
        "sgd":  "stochastic gradient descent",
        "llm":  "large language model",
        "bert": "BERT",
        "gpt":  "GPT",
        "mlp":  "multilayer perceptron",
        "cnn":  "convolutional neural network",
        "rnn":  "recurrent neural network",
        "rag":  "retrieval augmented generation",
    }
    query = query.strip().lower()
    for abbrev, expansion in abbrevs.items():
        query = re.sub(r'\b' + abbrev + r'\b', expansion, query)
    return query


def classify_intent(query):
    """Classify query intent into one of four categories."""
    q = query.lower().strip()
    if any(q.startswith(w) for w in ["what is", "define", "what are", "what does"]):
        return "definition"
    elif any(w in q for w in ["difference", "compare", "vs", "versus", "better", "versus"]):
        return "comparison"
    elif any(q.startswith(w) for w in ["how to", "how do", "how does", "explain how", "steps to"]):
        return "howto"
    elif any(q.startswith(w) for w in ["why", "when should", "when to", "when do"]):
        return "whywhen"
    return "general"


test_queries = [
    "What is gradient descent?",
    "how does attention work?",
    "why use dropout?",
    "difference between BERT and GPT",
    "when should I use early stopping?",
    "tell me about quantization",
]
print("Query preprocessing and intent classification:")
for q in test_queries:
    cleaned = preprocess_query(q)
    intent  = classify_intent(cleaned)
    print(f"  Original : {q}")
    print(f"  Cleaned  : {cleaned}")
    print(f"  Intent   : {intent}\n")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 4: FULL Q&A SYSTEM CLASS")
print("="*70)
# ==========================================================================

class MLQASystem:
    """
    Production-ready Q&A system for ML knowledge using TF-IDF RAG.
    Main API: answer(query) → dict with answer, confidence, sources, intent.
    """

    def __init__(self, knowledge_base):
        self.kb = knowledge_base
        self.retriever = TFIDFRetriever()
        self.retriever.fit(knowledge_base)

    def answer(self, query, top_k=3, filter_topic=None, verbose=True):
        """
        Answer a natural language question about ML.

        Parameters
        ----------
        query        : str   — the user's question
        top_k        : int   — number of documents to retrieve
        filter_topic : str   — optional topic filter
        verbose      : bool  — print debug trace

        Returns
        -------
        dict with keys: query, answer, confidence, intent, sources, topic
        """
        query_clean = preprocess_query(query)
        intent = classify_intent(query_clean)

        # Retrieve relevant documents
        results = self.retriever.retrieve(query_clean, top_k=top_k,
                                          filter_topic=filter_topic)
        if not results:
            return {
                "query":      query,
                "answer":     "I don't have information about that topic.",
                "confidence": 0.0,
                "intent":     intent,
                "sources":    [],
                "topic":      None,
            }

        # Extract answer from the top document
        top_idx, top_score = results[0]
        top_doc = self.kb[top_idx]

        # Sentence-level selection: pick the sentence most lexically similar to query
        sentences = [s.strip() for s in top_doc["text"].split(". ") if s.strip()]
        q_words = set(query_clean.split()) - {
            "what", "is", "how", "does", "why", "the", "a", "an",
            "do", "are", "can", "should", "when", "will",
        }
        best_sent = max(
            sentences,
            key=lambda s: sum(1 for w in s.lower().split() if w in q_words),
        )

        # Confidence from retrieval cosine score (clip to [0, 1])
        confidence = float(np.clip(top_score * 2, 0.0, 1.0))

        if confidence > 0.1:
            answer = best_sent + ("." if not best_sent.endswith(".") else "")
        else:
            answer = ("I'm not confident about this answer. "
                      "Please consult additional sources.")

        if verbose:
            print(f"\n{'='*60}")
            print(f"Query      : {query}")
            print(f"Intent     : {intent}")
            print(f"Top sources: {[self.kb[i]['title'] for i, _ in results[:3]]}")
            print(f"Confidence : {confidence:.1%}")
            print(f"Answer     : {answer}")

        return {
            "query":      query,
            "answer":     answer,
            "confidence": confidence,
            "intent":     intent,
            "sources":    [self.kb[i]["title"] for i, _ in results],
            "topic":      self.kb[results[0][0]]["topic"],
        }


qa = MLQASystem(KNOWLEDGE_BASE)
print("\nQ&A System initialized successfully.")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 5: DEMO — 10 TEST QUESTIONS")
print("="*70)
# ==========================================================================

demo_questions = [
    "What is gradient descent?",
    "How does backpropagation work?",
    "What is the attention mechanism in transformers?",
    "Why use dropout regularization?",
    "How does batch normalization work?",
    "What is early stopping?",
    "How does BERT differ from GPT?",
    "What is quantization in model deployment?",
    "Why do we use learning rate schedules?",
    "How does cross-validation work?",
]

demo_results = []
for q in demo_questions:
    result = qa.answer(q, verbose=True)
    demo_results.append(result)

# ==========================================================================
print("\n" + "="*70)
print("SECTION 6: EVALUATION")
print("="*70)
# ==========================================================================

# 15 test questions with ground-truth document IDs (the most relevant doc)
TEST_SET = [
    # fundamentals (3)
    {"query": "What is gradient descent?",          "relevant_ids": [0]},
    {"query": "What is a loss function?",            "relevant_ids": [1]},
    {"query": "How do you prevent overfitting?",     "relevant_ids": [2]},
    # neural_networks (3)
    {"query": "What is a perceptron?",               "relevant_ids": [5]},
    {"query": "What activation functions exist?",    "relevant_ids": [6]},
    {"query": "How does backpropagation work?",      "relevant_ids": [7]},
    # transformers (3)
    {"query": "How does self-attention work?",       "relevant_ids": [10]},
    {"query": "What is BERT and how is it trained?", "relevant_ids": [11]},
    {"query": "How does positional encoding work?",  "relevant_ids": [13]},
    # training (3)
    {"query": "What is a learning rate schedule?",   "relevant_ids": [15]},
    {"query": "How does early stopping work?",       "relevant_ids": [18]},
    {"query": "What is mixed precision training?",   "relevant_ids": [19]},
    # deployment (3)
    {"query": "How do you serve a model in production?", "relevant_ids": [20]},
    {"query": "What is model quantization?",         "relevant_ids": [22]},
    {"query": "How does A/B testing work for ML?",   "relevant_ids": [23]},
]

TOPICS = ["fundamentals", "neural_networks", "transformers", "training", "deployment"]

def evaluate(qa_system, test_set, top_k=3):
    """Compute P@1, P@3, MRR, and answer coverage metrics."""
    metrics = {
        "p_at_1": [], "p_at_3": [], "mrr": [],
        "confidence": [], "correct_at_1": [],
    }
    per_topic = defaultdict(lambda: {"p_at_1": [], "p_at_3": [], "mrr": []})

    for item in test_set:
        query   = item["query"]
        rel_ids = set(item["relevant_ids"])

        result = qa_system.retriever.retrieve(query, top_k=top_k)
        retrieved_ids = [idx for idx, _ in result]

        # P@1
        p1 = 1.0 if retrieved_ids and retrieved_ids[0] in rel_ids else 0.0
        # P@3
        hits_3 = sum(1 for rid in retrieved_ids[:3] if rid in rel_ids)
        p3 = hits_3 / min(3, len(rel_ids))
        # MRR
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in rel_ids:
                rr = 1.0 / rank
                break

        metrics["p_at_1"].append(p1)
        metrics["p_at_3"].append(p3)
        metrics["mrr"].append(rr)

        # Confidence & correctness for answer coverage
        ans = qa_system.answer(query, top_k=top_k, verbose=False)
        metrics["confidence"].append(ans["confidence"])
        metrics["correct_at_1"].append(p1)

        # Per-topic
        topic = None
        for doc in qa_system.kb:
            if doc["id"] in rel_ids:
                topic = doc["topic"]
                break
        if topic:
            per_topic[topic]["p_at_1"].append(p1)
            per_topic[topic]["p_at_3"].append(p3)
            per_topic[topic]["mrr"].append(rr)

    overall = {
        "P@1":               np.mean(metrics["p_at_1"]),
        "P@3":               np.mean(metrics["p_at_3"]),
        "MRR":               np.mean(metrics["mrr"]),
        "Answer Coverage":   np.mean([c > 0.3 for c in metrics["confidence"]]),
    }
    return overall, per_topic, metrics


overall, per_topic, raw_metrics = evaluate(qa, TEST_SET, top_k=3)

print("\n--- Overall Evaluation Results ---")
for metric, val in overall.items():
    print(f"  {metric:20s}: {val:.3f}")

print("\n--- Per-Topic Results ---")
print(f"  {'Topic':20s} {'P@1':>6} {'P@3':>6} {'MRR':>6}")
print("  " + "-"*42)
for topic in TOPICS:
    tm = per_topic.get(topic, {})
    p1  = np.mean(tm.get("p_at_1", [0]))
    p3  = np.mean(tm.get("p_at_3", [0]))
    mrr = np.mean(tm.get("mrr",    [0]))
    print(f"  {topic:20s} {p1:6.3f} {p3:6.3f} {mrr:6.3f}")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 7: VISUALIZATIONS")
print("="*70)
# ==========================================================================

# ── Visualization 1: System Architecture Diagram ─────────────────────────
fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis("off")
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("#F8F9FA")
ax.set_title("RAG Q&A System Architecture", fontsize=16, fontweight="bold", pad=20)

# Box definitions: (x_center, y_center, width, height, label, color)
boxes = [
    (1.5,  6.5, 2.4, 0.9, "User Query",           "#4A90D9"),
    (4.5,  6.5, 2.4, 0.9, "Query Preprocessor",   "#5CB85C"),
    (7.5,  6.5, 2.4, 0.9, "Intent Classifier",    "#F0AD4E"),
    (10.5, 6.5, 2.4, 0.9, "TF-IDF Retriever",     "#D9534F"),
    (10.5, 4.5, 2.4, 0.9, "Knowledge Base Index", "#9B59B6"),
    (7.5,  2.5, 2.4, 0.9, "Top-K Documents",      "#1ABC9C"),
    (4.5,  2.5, 2.4, 0.9, "Answer Extractor",     "#E67E22"),
    (1.5,  2.5, 2.4, 0.9, "Confidence Scorer",    "#3498DB"),
    (1.5,  0.5, 2.4, 0.9, "Response",             "#2ECC71"),
]

box_centers = {}
for (xc, yc, w, h, label, color) in boxes:
    rect = mpatches.FancyBboxPatch(
        (xc - w/2, yc - h/2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="white", linewidth=2, alpha=0.9,
    )
    ax.add_patch(rect)
    ax.text(xc, yc, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", wrap=True)
    box_centers[label] = (xc, yc)

# Arrows
arrow_pairs = [
    ("User Query",           "Query Preprocessor",   "Expand abbrevs"),
    ("Query Preprocessor",   "Intent Classifier",    "Classify intent"),
    ("Intent Classifier",    "TF-IDF Retriever",     "Route query"),
    ("Knowledge Base Index", "TF-IDF Retriever",     "Cosine sim"),
    ("TF-IDF Retriever",     "Top-K Documents",      "Top-K results"),
    ("Top-K Documents",      "Answer Extractor",     "Extract best sent"),
    ("Answer Extractor",     "Confidence Scorer",    "Score answer"),
    ("Confidence Scorer",    "Response",             "Return dict"),
]

def draw_arrow(ax, p1, p2, label="", color="#555555"):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=1.5, connectionstyle="arc3,rad=0.0"),
    )
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.18, label, ha="center", va="center",
                fontsize=7, color="#333333",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1))


# Horizontal arrows (same y-level)
draw_arrow(ax, (2.7,  6.5), (3.3,  6.5), "")
draw_arrow(ax, (5.7,  6.5), (6.3,  6.5), "")
draw_arrow(ax, (8.7,  6.5), (9.3,  6.5), "")
# Vertical KB → Retriever
draw_arrow(ax, (10.5, 5.0), (10.5, 6.0), "")
# Retriever → Top-K (diagonal down)
draw_arrow(ax, (10.5, 6.0), (8.7,  2.5), "")
# Top-K → Answer Extractor
draw_arrow(ax, (6.3,  2.5), (5.7,  2.5), "")
# Answer Extractor → Confidence Scorer
draw_arrow(ax, (3.3,  2.5), (2.7,  2.5), "")
# Confidence Scorer → Response
draw_arrow(ax, (1.5,  2.0), (1.5,  1.0), "")

# Labels on key arrows
ax.text(4.5, 6.8, "abbrev expand", ha="center", fontsize=7, color="#555")
ax.text(7.5, 6.8, "intent label",  ha="center", fontsize=7, color="#555")
ax.text(10.5, 3.5, "cosine sim",   ha="center", fontsize=7, color="#555")

plt.tight_layout()
out1 = os.path.join(VIS_DIR, "01_system_architecture.png")
plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {out1}")

# ── Visualization 2: Evaluation Results ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Q&A System Evaluation", fontsize=14, fontweight="bold")

# (a) Per-topic metric bar chart
ax = axes[0]
metric_names = ["P@1", "P@3", "MRR"]
n_topics = len(TOPICS)
n_metrics = len(metric_names)
x = np.arange(n_topics)
width = 0.22
colors_m = ["#4A90D9", "#5CB85C", "#F0AD4E"]

for mi, (mname, color) in enumerate(zip(metric_names, colors_m)):
    vals = []
    for topic in TOPICS:
        tm = per_topic.get(topic, {})
        key = {"P@1": "p_at_1", "P@3": "p_at_3", "MRR": "mrr"}[mname]
        vals.append(np.mean(tm.get(key, [0])))
    bars = ax.bar(x + mi * width, vals, width, label=mname, color=color,
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + width)
ax.set_xticklabels([t.replace("_", "\n") for t in TOPICS], fontsize=8)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("Per-Topic Retrieval Metrics")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)

# (b) Confidence distribution histogram (correct vs incorrect)
ax = axes[1]
confs  = raw_metrics["confidence"]
corr   = raw_metrics["correct_at_1"]
c_corr = [c for c, ok in zip(confs, corr) if ok]
c_incr = [c for c, ok in zip(confs, corr) if not ok]

bins = np.linspace(0, 1, 12)
ax.hist(c_corr, bins=bins, color="#5CB85C", alpha=0.7, label="Correct @1", edgecolor="white")
ax.hist(c_incr, bins=bins, color="#D9534F", alpha=0.7, label="Incorrect @1", edgecolor="white")
ax.axvline(0.3, color="#333", linestyle="--", lw=1.2, label="Coverage threshold (0.3)")
ax.set_xlabel("Confidence Score")
ax.set_ylabel("Number of Queries")
ax.set_title("Confidence Distribution by Correctness")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out2 = os.path.join(VIS_DIR, "02_evaluation.png")
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {out2}")

# ── Visualization 3: Knowledge Base Analysis ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Knowledge Base Analysis", fontsize=14, fontweight="bold")

# (a) Document count per topic
ax = axes[0]
topic_labels = list(topic_counts.keys())
topic_vals   = [topic_counts[t] for t in topic_labels]
colors_t = ["#4A90D9", "#5CB85C", "#F0AD4E", "#D9534F", "#9B59B6"]
bars = ax.bar(topic_labels, topic_vals, color=colors_t, edgecolor="white", linewidth=1)
for bar, val in zip(bars, topic_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            str(val), ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_xticks(range(len(topic_labels)))
ax.set_xticklabels([t.replace("_", "\n") for t in topic_labels], fontsize=9)
ax.set_ylim(0, max(topic_vals) + 1.5)
ax.set_ylabel("Number of Documents")
ax.set_title("Documents per Topic")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

# (b) Topic centroid similarity heatmap (5×5)
ax = axes[1]
# Build per-topic average TF-IDF vectors
V = len(retriever.vocab)
topic_centroids = {}
for topic in TOPICS:
    idxs = [i for i, doc in enumerate(KNOWLEDGE_BASE) if doc["topic"] == topic]
    if idxs:
        vecs = np.array([retriever.doc_vectors[i] for i in idxs])
        centroid = vecs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        topic_centroids[topic] = centroid / norm if norm > 0 else centroid

sim_matrix = np.zeros((len(TOPICS), len(TOPICS)))
for i, t1 in enumerate(TOPICS):
    for j, t2 in enumerate(TOPICS):
        sim_matrix[i, j] = float(topic_centroids[t1] @ topic_centroids[t2])

sns.heatmap(
    sim_matrix,
    annot=True, fmt=".2f",
    xticklabels=[t.replace("_", "\n") for t in TOPICS],
    yticklabels=[t.replace("_", "\n") for t in TOPICS],
    cmap="Blues", vmin=0, vmax=1,
    linewidths=0.5, linecolor="white",
    ax=ax,
)
ax.set_title("Topic Centroid Cosine Similarities\n(lower off-diagonal = better topic separation)")
ax.tick_params(labelsize=8)

plt.tight_layout()
out3 = os.path.join(VIS_DIR, "03_knowledge_base.png")
plt.savefig(out3, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {out3}")

# ==========================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
# ==========================================================================
print(f"\nQ&A System with RAG — Complete")
print(f"  Knowledge base   : {len(KNOWLEDGE_BASE)} documents, 5 topics")
print(f"  Index vocab size : {len(retriever.vocab)} terms")
print(f"  Overall P@1      : {overall['P@1']:.3f}")
print(f"  Overall MRR      : {overall['MRR']:.3f}")
print(f"  Answer coverage  : {overall['Answer Coverage']:.1%}")
print(f"\nVisualizations saved to: {VIS_DIR}")
print(f"  01_system_architecture.png")
print(f"  02_evaluation.png")
print(f"  03_knowledge_base.png")
