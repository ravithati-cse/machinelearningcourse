"""
LLM-Powered Text Classifier — Zero-Shot and Few-Shot
======================================================
Learning Objectives:
  1. Use LLM-style reasoning for text classification without labeled training data
  2. Implement zero-shot classification using template-based prompting
  3. Build few-shot classification with in-context examples
  4. Compare prompting strategies: zero-shot vs few-shot vs fine-tuned baseline
  5. Handle multi-class classification and confidence calibration
  6. Build a production classify() API that works without a live LLM
YouTube: Search "zero-shot few-shot classification LLM prompting NLP"
Time: ~60 min | Difficulty: Advanced | Prerequisites: LLMs math_foundations
"""

import os
import re
import numpy as np
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Visualization setup — Agg backend BEFORE pyplot import
# --------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

VIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "llm_powered_classifier")
os.makedirs(VIS_DIR, exist_ok=True)

CLASSES = ["Technology", "Sports", "Politics", "Business", "Entertainment"]

# ==========================================================================
print("\n" + "="*70)
print("SECTION 1: THE ZERO-SHOT CLASSIFICATION APPROACH")
print("="*70)
# ==========================================================================

print("""
Traditional ML Pipeline
-----------------------
  1. Collect thousands of labeled text examples
  2. Extract features (TF-IDF, embeddings)
  3. Train a classifier (logistic regression, SVM, neural net)
  4. Evaluate on a held-out test set
  5. Deploy the trained model

  Drawback: requires labeled data — expensive, time-consuming.

Zero-Shot Classification (LLM-style)
--------------------------------------
  1. Describe each class in natural language
  2. For a new document, measure its similarity to each class description
  3. Assign the class with the highest similarity score
  4. No labeled training examples required!

  How this works without a live LLM
  -----------------------------------
  We simulate LLM behavior with TF-IDF similarity between the document
  and per-class template descriptions. Real LLMs (GPT-4, Llama 3) do
  this implicitly via pre-trained contextual embeddings — the core idea
  is identical: describe the class and measure semantic alignment.

Few-Shot Classification
------------------------
  1. Provide a small number of labeled examples (2-10 per class)
  2. Classify new documents by similarity to those examples
  3. More examples → better accuracy, approaching fully supervised methods

  In real LLMs, examples are placed directly in the prompt context window.
  Here we implement it as a Jaccard-similarity k-NN over the few-shot set.

Classes: {classes}
""".format(classes=", ".join(CLASSES)))

# ==========================================================================
print("\n" + "="*70)
print("SECTION 2: DATASET (50 TRAIN + 20 TEST ARTICLES)")
print("="*70)
# ==========================================================================

ARTICLES = {
    "Technology": [
        # Training set (10)
        "Scientists unveiled a new microprocessor architecture that achieves record GPU performance for deep learning inference at significantly lower power consumption.",
        "The open-source machine learning framework released version 3.0, adding native support for distributed training across thousands of GPU nodes.",
        "Researchers at a leading university developed a neural network capable of diagnosing rare diseases from medical imaging with 94 percent accuracy.",
        "A semiconductor company announced its next-generation chip fabricated using a 2-nanometer process, doubling transistor density compared to previous designs.",
        "Engineers demonstrated a quantum computing system that solved a logistics optimization problem in minutes that would take classical computers years.",
        "The AI startup released a code generation model trained on 500 billion tokens that autonomously writes and debugs software across 20 programming languages.",
        "Cybersecurity researchers discovered a critical vulnerability in widely used encryption libraries affecting millions of cloud servers worldwide.",
        "A new robotics platform integrates computer vision and reinforcement learning to enable warehouse robots to handle previously unseen object shapes.",
        "The browser company deployed a privacy-preserving federated learning system that trains ad relevance models without transmitting user data to servers.",
        "Autonomous vehicle software achieved a one-million-mile safety milestone in urban environments without a human safety driver intervention.",
        # Test set (4)
        "The generative AI model produces photorealistic images from text descriptions in under two seconds using a diffusion-based architecture.",
        "Developers released an open-weight large language model with 70 billion parameters that rivals proprietary systems on standard benchmarks.",
        "A tech giant announced plans to invest 100 billion dollars in domestic AI data center construction over the next five years.",
        "Researchers published a method for reducing LLM hallucination rates by 40 percent using retrieval-augmented generation with fact verification.",
    ],
    "Sports": [
        # Training set (10)
        "The national soccer team advanced to the World Cup final after a penalty shootout victory, the first time the country has reached the final in 20 years.",
        "A marathon runner broke the world record by 12 seconds, completing the 26-mile course in under two hours and one minute at altitude.",
        "The basketball franchise signed a five-year contract extension with its star point guard, keeping the league MVP in the city through 2030.",
        "Torrential rain forced the suspension of the championship tennis match, which will resume tomorrow with the fifth set tied at three games each.",
        "The cycling team revealed its roster for the upcoming Tour de France, featuring three stage winners and last year's overall champion.",
        "An Olympic gymnastics committee updated the scoring system to increase difficulty credits for release moves on the uneven bars.",
        "The rugby union announced expanded playoffs beginning next season, adding four additional teams to the postseason bracket.",
        "A teenage swimmer qualified for the national championships after setting a new age-group record in the 200-meter butterfly event.",
        "The coaching staff attributed the team's seven-match winning streak to improved set-piece execution and defensive organization.",
        "International cricket officials approved the use of a new ball-tracking technology to review boundary decisions in test matches.",
        # Test set (4)
        "The hockey franchise captured the championship trophy after a seven-game series that featured three overtime finishes.",
        "A professional golfer withdrew from the tournament after aggravating a wrist injury sustained during the second round.",
        "The athletics governing body imposed a two-year ban on a sprinter who tested positive for a prohibited performance-enhancing substance.",
        "Ticket demand for the boxing heavyweight title fight broke the venue's online sales record within 90 minutes of going on sale.",
    ],
    "Politics": [
        # Training set (10)
        "The senate passed landmark climate legislation with bipartisan support, allocating 400 billion dollars for renewable energy infrastructure over a decade.",
        "Diplomatic talks between the two nations collapsed after disagreements over border demarcation in a disputed region, raising regional tensions.",
        "The prime minister called a snap election following a vote of no confidence, citing the need for a fresh mandate on economic policy.",
        "International observers certified the presidential election as free and fair, with the incumbent winning by a narrow margin of 2.3 percent.",
        "Congress approved the annual defense authorization bill, including funding for three new naval destroyers and expanded cyber command capabilities.",
        "The foreign minister met with counterparts from five allied nations to coordinate sanctions targeting the rogue state's nuclear program.",
        "A federal appeals court upheld the immigration reform regulation, allowing temporary protected status extensions for 500,000 residents.",
        "The municipal government approved a comprehensive zoning reform to increase housing density near public transit corridors across the city.",
        "Senior party officials announced primary election dates and delegate allocation rules for the upcoming presidential nomination contest.",
        "The trade delegation returned without a signed agreement after negotiations over agricultural tariffs reached an impasse on the final day.",
        # Test set (4)
        "Lawmakers introduced a bill to establish independent oversight of artificial intelligence systems deployed in federal agencies.",
        "The ambassador was recalled for consultations after the host government expelled three embassy staff over espionage allegations.",
        "The constitutional court invalidated a voter identification law, ruling that it created undue burdens on minority communities.",
        "Regional governors convened an emergency summit to coordinate disaster response after flooding displaced 200,000 residents.",
    ],
    "Business": [
        # Training set (10)
        "The e-commerce giant reported quarterly revenues of 180 billion dollars, beating analyst consensus estimates by 8 percent on strong cloud growth.",
        "A fintech startup raised 500 million dollars in Series D funding, bringing its valuation to 12 billion dollars ahead of a planned IPO next year.",
        "The central bank held benchmark interest rates steady at 5.25 percent, signaling that inflation has not yet reached its 2 percent target.",
        "Merger talks between the two pharmaceutical companies collapsed over disagreements on post-merger leadership structure and asset valuations.",
        "The automaker announced plans to close two legacy factories and redirect 8 billion dollars toward electric vehicle battery production facilities.",
        "Supply chain disruptions pushed freight rates to a three-year high, squeezing profit margins for retail companies dependent on overseas manufacturing.",
        "The hedge fund disclosed a 6 percent stake in the media conglomerate, fueling speculation about a potential leveraged buyout.",
        "Consumer confidence rose to its highest level in 18 months, driven by falling fuel prices and a robust labor market adding 250,000 jobs monthly.",
        "The retailer launched a subscription loyalty program offering free same-day delivery, aiming to compete directly with the dominant marketplace.",
        "An accounting standards board finalized new rules requiring companies to disclose climate-related financial risks starting in fiscal year 2027.",
        # Test set (4)
        "The airline industry association forecasted a record 4.7 billion passenger journeys this year, fully recovering from pandemic-era declines.",
        "The streaming platform raised its monthly subscription price by three dollars, citing increased content production and licensing costs.",
        "Venture capital investment in generative AI companies exceeded 30 billion dollars in the first half of the year, up 140 percent year-over-year.",
        "The logistics company acquired a last-mile delivery network for 2.1 billion dollars to reduce dependence on third-party carriers.",
    ],
    "Entertainment": [
        # Training set (10)
        "The superhero sequel broke the global opening weekend record with 450 million dollars, driven by record attendances across 72 international markets.",
        "A streaming platform's prestige drama series swept the awards ceremony, winning best series, directing, writing, and four acting categories.",
        "The pop star announced a 52-date world tour, with tickets selling out in under four minutes and resale prices reaching 3,000 dollars for floor seats.",
        "A veteran filmmaker's documentary about deep-sea ecosystems earned standing ovations at three consecutive international film festivals.",
        "The animated feature studio unveiled its first trailer for the upcoming sequel, generating 120 million views in 24 hours across social media.",
        "A record label reported that music streaming revenues exceeded physical sales globally for the seventh consecutive year, rising 9 percent.",
        "The video game adaptation scored the highest Metacritic rating of the year, praised for its faithful portrayal of the source material's lore.",
        "An independent film produced on a budget of 4 million dollars became a surprise hit, grossing 90 million dollars domestically in its first month.",
        "The comedian's sold-out arena show was simultaneously released as a streaming special, attracting the platform's largest debut audience for comedy.",
        "Award nominations announced today include five recognitions for the biographical drama that traces a musician's rise to global fame.",
        # Test set (4)
        "The action franchise announced it will conclude with a two-part finale, with the first installment releasing next summer.",
        "A music producer dropped a surprise album collaboration featuring ten Grammy-winning artists across hip-hop, R&B, and pop genres.",
        "The reality competition series returned for its 25th season with the highest premiere ratings the network has seen in six years.",
        "A major theme park resort unveiled plans for a new immersive entertainment district inspired by a beloved fantasy film series.",
    ],
}

# Separate train and test sets
TRAIN_ARTICLES = {cls: texts[:10] for cls, texts in ARTICLES.items()}
TEST_ARTICLES  = {cls: texts[10:] for cls, texts in ARTICLES.items()}

# Flatten to lists
train_texts  = [t for cls in CLASSES for t in TRAIN_ARTICLES[cls]]
train_labels = [cls for cls in CLASSES for _ in TRAIN_ARTICLES[cls]]
test_texts   = [t for cls in CLASSES for t in TEST_ARTICLES[cls]]
test_labels  = [cls for cls in CLASSES for _ in TEST_ARTICLES[cls]]

print(f"Training articles : {len(train_texts)} ({len(CLASSES)} classes × 10)")
print(f"Test articles     : {len(test_texts)} ({len(CLASSES)} classes × 4)")
for cls in CLASSES:
    print(f"  {cls:15s}: {len(TRAIN_ARTICLES[cls])} train, {len(TEST_ARTICLES[cls])} test")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 3: CLASS DESCRIPTION TEMPLATES")
print("="*70)
# ==========================================================================

CLASS_TEMPLATES = {
    "Technology": [
        "artificial intelligence machine learning software hardware computing technology innovation",
        "programming code algorithm data science robotics automation digital",
        "semiconductor chip processor GPU neural network computer science engineering",
    ],
    "Sports": [
        "game match tournament championship team athlete player coach score",
        "football basketball soccer tennis Olympics medal competition",
        "training fitness workout performance record victory defeat",
    ],
    "Politics": [
        "government election policy law vote congress parliament legislation",
        "president senator diplomat foreign affairs geopolitics democracy",
        "treaty international relations sanctions embassy summit diplomacy",
    ],
    "Business": [
        "market economy finance investment stock revenue profit company merger",
        "startup venture capital IPO earnings quarterly revenue sales growth",
        "CEO leadership strategy acquisition supply chain retail consumer",
    ],
    "Entertainment": [
        "movie film music album actor director award Oscar Grammy",
        "streaming Netflix theater television celebrity star show",
        "concert tour box office release premiere audience fans",
    ],
}

print("Class description templates (3 templates per class):")
for cls, templates in CLASS_TEMPLATES.items():
    print(f"\n  {cls}:")
    for i, t in enumerate(templates, 1):
        words = t.split()
        print(f"    [{i}] {t[:70]}{'...' if len(t) > 70 else ''}")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 4: ZERO-SHOT CLASSIFIER")
print("="*70)
# ==========================================================================

class ZeroShotClassifier:
    """Classify text using cosine similarity to class description templates."""

    def __init__(self, class_templates, stopwords=None):
        self.classes   = list(class_templates.keys())
        self.templates = class_templates
        self.stopwords = stopwords or {
            "a", "an", "the", "is", "are", "was", "were",
            "be", "been", "to", "of", "in", "for", "on",
            "with", "at", "by", "from", "as", "it",
        }
        self._build_class_vectors()

    def _tokenize(self, text):
        return [t for t in re.sub(r"[^a-z\s]", " ", text.lower()).split()
                if t not in self.stopwords and len(t) > 2]

    def _build_class_vectors(self):
        # Collect vocabulary from all templates
        all_tokens = []
        for templates in self.templates.values():
            for t in templates:
                all_tokens.extend(self._tokenize(t))
        self.vocab = {w: i for i, w in enumerate(sorted(set(all_tokens)))}
        V = len(self.vocab)

        # Build per-class average template vector
        self.class_vectors = {}
        for cls, templates in self.templates.items():
            vecs = []
            for t in templates:
                toks = self._tokenize(t)
                v = np.zeros(V)
                for tok in toks:
                    if tok in self.vocab:
                        v[self.vocab[tok]] += 1
                norm = np.linalg.norm(v)
                vecs.append(v / norm if norm > 0 else v)
            self.class_vectors[cls] = np.mean(vecs, axis=0)

        print(f"  Vocabulary size: {V} terms")
        print(f"  Classes: {self.classes}")

    def _vectorize(self, text):
        toks = self._tokenize(text)
        v = np.zeros(len(self.vocab))
        for tok in toks:
            if tok in self.vocab:
                v[self.vocab[tok]] += 1
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def classify(self, text, return_probs=False):
        vec = self._vectorize(text)
        scores = {cls: float(vec @ self.class_vectors[cls])
                  for cls in self.classes}
        # Sharpen with scaled softmax for more decisive probabilities
        vals = np.array(list(scores.values()))
        vals_shifted = vals - vals.max()
        exp_vals = np.exp(vals_shifted * 5)
        probs_arr = exp_vals / exp_vals.sum()
        probs = {cls: float(p) for cls, p in zip(self.classes, probs_arr)}
        pred  = max(scores, key=scores.get)
        if return_probs:
            return pred, probs
        return pred


zero_shot_clf = ZeroShotClassifier(CLASS_TEMPLATES)
print("\nZero-shot classification examples:")
for cls in CLASSES:
    sample = TRAIN_ARTICLES[cls][0]
    pred, probs = zero_shot_clf.classify(sample, return_probs=True)
    top_prob = max(probs.values())
    print(f"  True: {cls:15s} | Predicted: {pred:15s} | Confidence: {top_prob:.1%}")
    print(f"    Text: {sample[:80]}...")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 5: FEW-SHOT CLASSIFIER")
print("="*70)
# ==========================================================================

class FewShotClassifier:
    """Classify using Jaccard similarity to labeled few-shot examples (k-NN style)."""

    def __init__(self, k=3, stopwords=None):
        self.k = k
        self.examples = []   # list of (text, label) pairs
        self.stopwords = stopwords or {
            "a", "an", "the", "is", "are", "was", "were",
            "be", "been", "to", "of", "in", "for", "on",
            "with", "at", "by", "from", "as", "it",
        }

    def add_examples(self, texts, labels):
        self.examples.extend(zip(texts, labels))

    def _tokenize(self, text):
        return [t for t in re.sub(r"[^a-z\s]", " ", text.lower()).split()
                if t not in self.stopwords]

    def _jaccard_sim(self, a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / max(len(sa | sb), 1)

    def classify(self, text, return_probs=False):
        toks = self._tokenize(text)
        sims = [
            (self._jaccard_sim(toks, self._tokenize(ex_text)), label)
            for ex_text, label in self.examples
        ]
        top_k = sorted(sims, reverse=True)[: self.k]
        votes = Counter(label for _, label in top_k)
        pred  = votes.most_common(1)[0][0]
        if return_probs:
            all_labels = list(set(l for _, l in self.examples))
            total = sum(votes.values())
            probs = {c: votes.get(c, 0) / total for c in all_labels}
            return pred, probs
        return pred


# Build few-shot classifiers with different k values
def build_few_shot(k, n_examples_per_class=5):
    """Create and fit a few-shot classifier using n_examples_per_class per class."""
    clf = FewShotClassifier(k=k)
    for cls in CLASSES:
        texts  = TRAIN_ARTICLES[cls][:n_examples_per_class]
        labels = [cls] * len(texts)
        clf.add_examples(texts, labels)
    return clf

few_shot_clf = build_few_shot(k=5, n_examples_per_class=5)  # default for production API
print("Few-shot classification examples (k=5, 5 examples/class):")
for cls in CLASSES:
    sample = TEST_ARTICLES[cls][0]
    pred, probs = few_shot_clf.classify(sample, return_probs=True)
    top_prob = max(probs.values())
    print(f"  True: {cls:15s} | Predicted: {pred:15s} | Confidence: {top_prob:.1%}")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 6: COMPARISON — ZERO-SHOT vs FEW-SHOT vs BASELINE")
print("="*70)
# ==========================================================================

def accuracy(preds, labels):
    return np.mean([p == l for p, l in zip(preds, labels)])

def per_class_accuracy(preds, labels, classes):
    acc = {}
    for cls in classes:
        cls_preds  = [p for p, l in zip(preds, labels) if l == cls]
        cls_labels = [l for p, l in zip(preds, labels) if l == cls]
        if cls_labels:
            acc[cls] = np.mean([p == l for p, l in zip(cls_preds, cls_labels)])
        else:
            acc[cls] = 0.0
    return acc

# Zero-shot evaluation
zs_preds = [zero_shot_clf.classify(t) for t in test_texts]
zs_acc   = accuracy(zs_preds, test_labels)
zs_per   = per_class_accuracy(zs_preds, test_labels, CLASSES)

# Few-shot with k = 3, 5, 7
fs_results = {}
for k in [3, 5, 7]:
    clf = build_few_shot(k=k, n_examples_per_class=5)
    preds = [clf.classify(t) for t in test_texts]
    fs_results[k] = {
        "acc": accuracy(preds, test_labels),
        "per": per_class_accuracy(preds, test_labels, CLASSES),
        "preds": preds,
    }

# TF-IDF + Logistic Regression baseline
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    tfidf  = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    le     = LabelEncoder()
    X_tr   = tfidf.fit_transform(train_texts)
    y_tr   = le.fit_transform(train_labels)
    X_te   = tfidf.transform(test_texts)
    y_te   = le.transform(test_labels)

    lr = LogisticRegression(max_iter=500, C=1.0, random_state=42)
    lr.fit(X_tr, y_tr)
    lr_preds_enc = lr.predict(X_te)
    lr_preds     = le.inverse_transform(lr_preds_enc)
    lr_acc       = accuracy(lr_preds, test_labels)
    lr_per       = per_class_accuracy(lr_preds, test_labels, CLASSES)
    lr_available = True
    print("  scikit-learn baseline: TF-IDF + Logistic Regression")
except ImportError:
    lr_acc  = 0.0
    lr_per  = {cls: 0.0 for cls in CLASSES}
    lr_preds = ["N/A"] * len(test_texts)
    lr_available = False
    print("  scikit-learn not available — baseline will show 0.0")

# Print accuracy table
print(f"\n{'Method':25s} {'Accuracy':>10}")
print("-" * 38)
print(f"{'Zero-Shot':25s} {zs_acc:>10.1%}")
for k in [3, 5, 7]:
    print(f"{'Few-Shot (k='+str(k)+')':25s} {fs_results[k]['acc']:>10.1%}")
print(f"{'TF-IDF + LR Baseline':25s} {lr_acc:>10.1%}")

# Per-class accuracy table
print(f"\nPer-Class Accuracy:")
header = f"{'Method':20s}" + "".join(f"{cls:>15s}" for cls in CLASSES)
print(header)
print("-" * (20 + 15 * len(CLASSES)))
methods = [
    ("Zero-Shot",      zs_per),
    ("Few-Shot(k=3)", fs_results[3]["per"]),
    ("Few-Shot(k=5)", fs_results[5]["per"]),
    ("Few-Shot(k=7)", fs_results[7]["per"]),
    ("TF-IDF+LR",     lr_per),
]
for mname, per in methods:
    row = f"{mname:20s}" + "".join(f"{per[cls]:>15.1%}" for cls in CLASSES)
    print(row)

# ==========================================================================
print("\n" + "="*70)
print("SECTION 7: PRODUCTION classify() API")
print("="*70)
# ==========================================================================

def classify(text, method="few_shot", top_k=3):
    """
    Production text classification API.

    Parameters
    ----------
    text   : str  — article or document to classify
    method : str  — "zero_shot" | "few_shot"
    top_k  : int  — number of top predictions to return

    Returns
    -------
    dict with prediction, confidence, top_k predictions, and method
    """
    if method == "zero_shot":
        pred, probs = zero_shot_clf.classify(text, return_probs=True)
    else:
        pred, probs = few_shot_clf.classify(text, return_probs=True)

    sorted_preds = sorted(probs.items(), key=lambda x: -x[1])[:top_k]
    return {
        "prediction": pred,
        "confidence": round(probs.get(pred, 0.0), 3),
        "top_k":      [{"class": c, "prob": round(p, 3)} for c, p in sorted_preds],
        "method":     method,
    }


# Demo on 3 new articles
new_articles = [
    "A software company announced a new programming language designed for safe, "
    "concurrent systems programming that compiles to WebAssembly.",
    "The underdog team came back from three goals down to win the championship "
    "in the final minutes of extra time.",
    "The central bank raised interest rates by 50 basis points to combat "
    "persistent inflation in the services sector.",
]

print("\nProduction API demos:")
for article in new_articles:
    print(f"\n  Article: {article[:80]}...")
    for method in ["zero_shot", "few_shot"]:
        result = classify(article, method=method)
        print(f"  [{result['method']:10s}] Prediction: {result['prediction']:15s} "
              f"| Confidence: {result['confidence']:.1%}")
        top_str = ", ".join(f"{d['class']}({d['prob']:.0%})" for d in result["top_k"])
        print(f"              Top-{len(result['top_k'])}: {top_str}")

# ==========================================================================
print("\n" + "="*70)
print("SECTION 8: VISUALIZATIONS")
print("="*70)
# ==========================================================================

# ── Visualization 1: Classification Approaches Diagram ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 7))
fig.patch.set_facecolor("#F5F5F5")
fig.suptitle("Text Classification Approaches", fontsize=15, fontweight="bold", y=1.02)

approach_data = [
    {
        "ax": axes[0],
        "title": "Traditional ML",
        "color": "#4A90D9",
        "steps": [
            ("Labeled\nDataset", "#4A90D9"),
            ("Feature\nExtraction\n(TF-IDF)", "#5B9FD4"),
            ("Train\nClassifier\n(LR / SVM)", "#6CAED0"),
            ("Evaluate\non Test Set", "#7DBDCC"),
            ("Deploy\nModel", "#5CB85C"),
        ],
        "label": "Requires thousands\nof labeled examples",
    },
    {
        "ax": axes[1],
        "title": "Zero-Shot (LLM-style)",
        "color": "#F0AD4E",
        "steps": [
            ("Class\nDescriptions\n(Templates)", "#F0AD4E"),
            ("Encode Text\n+ Descriptions\n(TF-IDF / embed)", "#E8A040"),
            ("Compute\nCosine\nSimilarity", "#E09030"),
            ("Assign\nHighest-Score\nClass", "#D88020"),
            ("Classify!", "#5CB85C"),
        ],
        "label": "No labeled data\nrequired!",
    },
    {
        "ax": axes[2],
        "title": "Few-Shot (In-Context)",
        "color": "#D9534F",
        "steps": [
            ("Few Labeled\nExamples\n(2-10/class)", "#D9534F"),
            ("Encode Query\n+ Examples\n(tokenize)", "#CC4844"),
            ("Jaccard /\nk-NN\nSimilarity", "#BF3D40"),
            ("Vote from\nTop-K\nNeighbors", "#B2333B"),
            ("Classify!", "#5CB85C"),
        ],
        "label": "Minimal labeled data;\nimproves with more examples",
    },
]

for ap in approach_data:
    ax = ap["ax"]
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(ap["steps"]) * 2 + 1)
    ax.axis("off")
    ax.set_facecolor("#F5F5F5")
    ax.set_title(ap["title"], fontsize=12, fontweight="bold", color=ap["color"])

    n = len(ap["steps"])
    for i, (step_text, color) in enumerate(reversed(ap["steps"])):
        y = i * 2 + 1
        rect = mpatches.FancyBboxPatch(
            (0.4, y - 0.55), 3.2, 1.1,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(2.0, y, step_text, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="white")
        # Arrow between boxes
        if i < n - 1:
            ax.annotate("", xy=(2.0, y + 0.55), xytext=(2.0, y + 1.45),
                        arrowprops=dict(arrowstyle="-|>", color="#666",
                                        lw=1.3, mutation_scale=12))

    # Bottom label
    ax.text(2.0, 0.2, ap["label"], ha="center", va="center",
            fontsize=8, style="italic", color="#444",
            bbox=dict(facecolor="white", edgecolor=ap["color"],
                      linewidth=1, alpha=0.8, boxstyle="round,pad=0.3"))

plt.tight_layout()
out1 = os.path.join(VIS_DIR, "01_classification_approaches.png")
plt.savefig(out1, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {out1}")

# ── Visualization 2: Accuracy Comparison ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Classifier Performance Comparison", fontsize=14, fontweight="bold")

# (a) Overall accuracy bar chart
ax = axes[0]
method_names = ["Zero-Shot", "Few-Shot\n(k=3)", "Few-Shot\n(k=5)", "Few-Shot\n(k=7)", "TF-IDF+LR\nBaseline"]
accs = [
    zs_acc,
    fs_results[3]["acc"],
    fs_results[5]["acc"],
    fs_results[7]["acc"],
    lr_acc,
]
colors_bar = ["#F0AD4E", "#D9534F", "#C0392B", "#A93226", "#4A90D9"]
bars = ax.bar(method_names, accs, color=colors_bar, edgecolor="white", linewidth=1, width=0.6)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.set_ylabel("Accuracy")
ax.set_title("Overall Test Accuracy")
ax.axhline(zs_acc, color="#F0AD4E", linestyle="--", lw=1, alpha=0.5, label="Zero-shot baseline")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

# (b) Per-class accuracy heatmap
ax = axes[1]
per_class_data = np.array([
    [zs_per[cls] for cls in CLASSES],
    [fs_results[3]["per"][cls] for cls in CLASSES],
    [fs_results[5]["per"][cls] for cls in CLASSES],
    [fs_results[7]["per"][cls] for cls in CLASSES],
    [lr_per[cls] for cls in CLASSES],
])
row_labels = ["Zero-Shot", "Few-Shot(k=3)", "Few-Shot(k=5)", "Few-Shot(k=7)", "TF-IDF+LR"]
sns.heatmap(
    per_class_data,
    annot=True, fmt=".0%",
    xticklabels=CLASSES,
    yticklabels=row_labels,
    cmap="RdYlGn", vmin=0, vmax=1,
    linewidths=0.5, linecolor="white",
    ax=ax,
)
ax.set_title("Per-Class Accuracy (methods × classes)")
ax.tick_params(axis="x", labelsize=8, rotation=20)
ax.tick_params(axis="y", labelsize=8, rotation=0)

plt.tight_layout()
out2 = os.path.join(VIS_DIR, "02_accuracy_comparison.png")
plt.savefig(out2, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {out2}")

# ── Visualization 3: Confidence Analysis ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Confidence Analysis", fontsize=14, fontweight="bold")

# Collect confidence data
zs_confs_corr, zs_confs_incr = [], []
fs_confs_corr, fs_confs_incr = [], []

for text, label in zip(test_texts, test_labels):
    _, zs_probs = zero_shot_clf.classify(text, return_probs=True)
    zs_conf = max(zs_probs.values())
    if zero_shot_clf.classify(text) == label:
        zs_confs_corr.append(zs_conf)
    else:
        zs_confs_incr.append(zs_conf)

    _, fs_probs = few_shot_clf.classify(text, return_probs=True)
    fs_conf = max(fs_probs.values())
    if few_shot_clf.classify(text) == label:
        fs_confs_corr.append(fs_conf)
    else:
        fs_confs_incr.append(fs_conf)

# (a) Box plot: confidence distributions
ax = axes[0]
box_data  = [zs_confs_corr, zs_confs_incr, fs_confs_corr, fs_confs_incr]
box_labels = ["ZS Correct", "ZS Incorrect", "FS Correct", "FS Incorrect"]
bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True, notch=False,
                medianprops=dict(color="black", lw=2))
box_colors = ["#5CB85C", "#D9534F", "#5CB85C", "#D9534F"]
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Confidence Score")
ax.set_title("Confidence by Correctness\n(Zero-Shot vs Few-Shot)")
ax.tick_params(axis="x", labelsize=8)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(0.5, color="#888", linestyle="--", lw=1, alpha=0.5, label="0.5 threshold")
ax.legend(fontsize=8)

# (b) Scatter: confidence vs correctness colored by true class
ax = axes[1]
color_map = {cls: c for cls, c in zip(
    CLASSES, ["#4A90D9", "#5CB85C", "#F0AD4E", "#D9534F", "#9B59B6"]
)}
# Jitter correctness axis slightly for visibility
rng = np.random.default_rng(42)
for text, label in zip(test_texts, test_labels):
    _, probs = few_shot_clf.classify(text, return_probs=True)
    conf = max(probs.values())
    correct = 1 if few_shot_clf.classify(text) == label else 0
    jittered_y = correct + rng.uniform(-0.08, 0.08)
    ax.scatter(conf, jittered_y, color=color_map[label], alpha=0.75,
               s=60, edgecolors="white", linewidth=0.5)

# Legend
handles = [mpatches.Patch(color=color_map[cls], label=cls) for cls in CLASSES]
ax.legend(handles=handles, fontsize=7, loc="center left",
          bbox_to_anchor=(1.01, 0.5))
ax.set_xlabel("Confidence Score")
ax.set_yticks([0, 1])
ax.set_yticklabels(["Incorrect", "Correct"])
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.3, 1.3)
ax.set_title("Few-Shot: Confidence vs Correctness\n(colored by true class)")
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out3 = os.path.join(VIS_DIR, "03_confidence_analysis.png")
plt.savefig(out3, dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {out3}")

# ==========================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
# ==========================================================================
print(f"\nLLM-Powered Classifier — Complete")
print(f"  Training articles : {len(train_texts)}")
print(f"  Test articles     : {len(test_texts)}")
print(f"  Zero-shot accuracy: {zs_acc:.1%}")
print(f"  Few-shot accuracy (k=5): {fs_results[5]['acc']:.1%}")
print(f"  TF-IDF+LR baseline    : {lr_acc:.1%}")
print(f"\nKey insight: Few-shot classification approaches supervised performance")
print(f"with just 5 labeled examples per class — no model training required.")
print(f"\nVisualizations saved to: {VIS_DIR}")
print(f"  01_classification_approaches.png")
print(f"  02_accuracy_comparison.png")
print(f"  03_confidence_analysis.png")
