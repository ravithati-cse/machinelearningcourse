"""
News Article Classifier — End-to-End Multi-Class NLP Project
=============================================================

Learning Objectives:
  1. Build a production-grade multi-class text classification system
  2. Apply advanced feature engineering: TF-IDF, entity features, n-grams
  3. Implement multi-class metrics: macro/micro F1, per-class accuracy
  4. Handle real-world challenges: class imbalance, ambiguous categories
  5. Build a full training pipeline with cross-validation
  6. Create a deployable classify_article() inference function

YouTube: Search "Multi-class Text Classification NLP Python" for companion videos
Time: ~60 minutes | Difficulty: Intermediate | Prerequisites: Parts 1-5 algorithms

Dataset: Synthetic news articles across 5 categories
         Mirrors AG News / BBC News Dataset structure
"""

import os
import re
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict

# ── Visualization output directory ─────────────────────────────────────────
VIS_DIR = "../visuals/news_article_classifier"
os.makedirs(VIS_DIR, exist_ok=True)

print("=" * 70)
print("NEWS ARTICLE CLASSIFIER — End-to-End Multi-Class NLP Project")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: Dataset — Synthetic Multi-Category News Articles
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 1: Dataset — 5-Category News")
print("=" * 70)

# Categories: Technology, Sports, Politics, Business, Entertainment
CATEGORIES = ["Technology", "Sports", "Politics", "Business", "Entertainment"]

ARTICLES = {
    "Technology": [
        "Apple unveiled its latest iPhone model featuring advanced AI capabilities and improved battery life. The new chip delivers 40 percent faster performance than the previous generation according to company benchmarks.",
        "Google DeepMind researchers have developed a new machine learning algorithm that achieves state-of-the-art results on protein folding prediction tasks with significantly reduced computational requirements.",
        "Microsoft announced a major update to its cloud computing platform Azure introducing new tools for artificial intelligence development and automated machine learning pipeline management.",
        "SpaceX successfully launched its reusable Starship rocket on a full orbital test flight demonstrating rapid iteration in commercial space technology development and reusability.",
        "Meta released a new open-source large language model that outperforms competing systems on standard benchmarks while requiring less computational resources to run effectively.",
        "Tesla unveiled its next generation autonomous driving software update claiming significant improvements in safety metrics and reducing driver intervention frequency across all weather conditions.",
        "NVIDIA announced record quarterly revenue driven by strong demand for its GPU chips used in artificial intelligence training and inference workloads across hyperscale data centers.",
        "Amazon Web Services launched a new suite of generative AI developer tools designed to help enterprise customers integrate large language models into their existing application infrastructure.",
        "OpenAI introduced a new multimodal model capable of processing images video and audio alongside text enabling complex reasoning across different types of input data simultaneously.",
        "Intel unveiled its next generation processor architecture claiming major improvements in energy efficiency and performance per watt compared to the previous chip generation.",
        "Samsung announced breakthrough developments in next generation semiconductor memory technology that could dramatically increase storage density while reducing power consumption significantly.",
        "A team of researchers published findings on a quantum computing breakthrough achieving stable qubit coherence times that bring practical quantum advantage significantly closer to reality.",
        "Cybersecurity researchers discovered a critical vulnerability affecting millions of IoT devices worldwide prompting urgent patches from manufacturers and warnings from government agencies.",
        "The latest smartphone chipset benchmarks show that mobile AI processing capabilities now rival dedicated workstation hardware from just three years ago at a fraction of the cost.",
        "New open source robotics platform enables researchers to develop and test autonomous navigation algorithms using standardized simulation environments before deploying to physical hardware.",
    ],
    "Sports": [
        "The World Cup final drew a record television audience of over one billion viewers as the host nation claimed the championship trophy in dramatic penalty shootout fashion.",
        "NBA playoffs continue as the defending champions overcame a 20-point deficit to defeat their rivals in overtime securing their spot in the conference finals series.",
        "The Olympic Committee announced new eligibility rules for the upcoming summer games that will allow professional athletes to compete in sports previously restricted to amateurs.",
        "A star quarterback signed the largest contract in NFL history totaling 350 million dollars over five years with the team that selected him in the first round.",
        "Tennis world number one dominated the grand slam final winning in straight sets to claim a record breaking 24th major title and cement legendary status in the sport.",
        "The Premier League title race remains wide open with three teams separated by just two points heading into the final five matches of the season.",
        "American swimmer broke the world record in the 200-meter freestyle event at the world championships improving the previous mark by nearly a full second.",
        "Formula One champion extended his lead in the drivers standings with a dominant victory in the Monaco Grand Prix demonstrating superior race strategy and tire management.",
        "College basketball tournament produced major upsets in the first round with four top-seeded teams eliminated by lower-ranked opponents in stunning fashion.",
        "The Boston Marathon was completed by a record number of finishers this year with the women's course record shattered by almost three minutes in ideal conditions.",
        "International cricket board announced a new tournament format that will see all major test playing nations compete in a round-robin series over the next three years.",
        "Hockey trade deadline saw several blockbuster deals as playoff contenders bolstered their rosters and rebuilding teams accumulated draft picks for future development.",
        "Youth sports participation rates dropped significantly during the pandemic and have yet to fully recover according to a comprehensive new national survey of athletic associations.",
        "Golf major championship saw a dramatic final round comeback as the leader collapsed under pressure allowing a late charge from the veteran who claimed a career first title.",
        "New sports science research suggests that high-intensity interval training combined with adequate recovery periods produces superior athletic performance compared to traditional methods.",
    ],
    "Politics": [
        "Congress passed a landmark infrastructure bill allocating two trillion dollars for road bridge and broadband improvements across the nation over the next decade.",
        "The president announced new executive orders targeting climate change including stricter emissions standards for automobiles and new renewable energy requirements for federal buildings.",
        "Senate confirmation hearings began for the supreme court nominee as lawmakers questioned the candidate extensively on constitutional interpretation and judicial philosophy.",
        "European Union leaders reached agreement on new digital regulation framework requiring large technology platforms to comply with stricter data privacy and competition rules.",
        "Midterm election results showed significant gains for the opposition party as voter turnout reached record highs in competitive suburban districts across multiple states.",
        "United Nations security council failed to reach consensus on a resolution addressing the ongoing humanitarian crisis as two permanent members exercised their veto power.",
        "The Federal Reserve raised interest rates for the sixth consecutive time citing persistent inflation that remains well above the central bank's two percent annual target.",
        "Congressional budget office released its latest projections showing the federal deficit is expected to grow substantially over the next decade without major policy changes.",
        "State governors from both parties issued a joint statement calling on Congress to address border security and immigration reform through comprehensive bipartisan legislation.",
        "The administration unveiled a new foreign policy doctrine prioritizing democratic alliances and multilateral institutions over bilateral agreements with authoritarian governments.",
        "Local election results across the country showed shifting political allegiances with several historically safe seats changing party control for the first time in decades.",
        "Campaign finance reports revealed that total spending on this election cycle has exceeded all previous records with over fifteen billion dollars raised by all candidates.",
        "International summit on climate change concluded with 190 countries signing a non-binding agreement to reduce carbon emissions while disputes over funding mechanisms remain unresolved.",
        "Whistleblower testimony before congressional committee revealed new details about surveillance programs conducted by intelligence agencies that may have exceeded legal authority.",
        "New polling data shows public approval ratings for the legislature at historic lows with voters expressing frustration over partisan gridlock and lack of policy progress.",
    ],
    "Business": [
        "Stock market indices reached record highs driven by strong corporate earnings reports and investor optimism about economic growth prospects in the coming quarters.",
        "Federal Reserve minutes indicated policymakers are divided on the pace of future interest rate adjustments as inflation shows mixed signals across different economic sectors.",
        "Merger between two of the largest retail chains valued at 25 billion dollars awaits regulatory approval from antitrust authorities who have expressed preliminary concerns.",
        "Startup unicorn completed its initial public offering raising eight billion dollars making it the largest technology IPO of the calendar year by total funds raised.",
        "Global supply chain disruptions continue to affect multiple industries as shipping costs remain elevated and delivery times extend beyond pre-pandemic baseline levels.",
        "Central bank announced quantitative tightening measures will accelerate as policymakers seek to reduce the expanded balance sheet accumulated during the stimulus period.",
        "Consumer confidence index fell unexpectedly last month as households cited concerns about inflation employment security and the overall direction of the national economy.",
        "Commercial real estate market faces headwinds as remote work policies reduce demand for office space in major metropolitan areas forcing landlords to offer significant concessions.",
        "Quarterly earnings season exceeded analyst expectations with more than 75 percent of major companies reporting results above forecasts driven by strong consumer spending.",
        "Venture capital investment declined for the third consecutive quarter as rising interest rates made earlier stage technology investments less attractive relative to fixed income.",
        "New economic data showed unemployment claims fell to their lowest level in six months suggesting continued resilience in the labor market despite tighter monetary conditions.",
        "Oil prices surged after OPEC announced production cuts exceeding market expectations sending energy sector stocks sharply higher while airline and transportation shares fell.",
        "Cryptocurrency market experienced significant volatility as regulatory uncertainty in major markets combined with macroeconomic concerns triggered substantial price swings.",
        "Retail sales data for the holiday quarter came in stronger than expected indicating consumer spending remains robust despite persistent inflation eroding purchasing power.",
        "Corporate debt refinancing activity surged as companies rushed to lock in fixed rate financing before further interest rate increases made borrowing costs prohibitively expensive.",
    ],
    "Entertainment": [
        "Box office revenues rebounded strongly this summer with multiple blockbuster sequels and original films driving the biggest weekend receipts since the pandemic began.",
        "Streaming platform announced it would release its entire film slate simultaneously on streaming and in theaters ending an exclusive theatrical window that studios had maintained for decades.",
        "Award season kicked off with nominations announced that surprised many observers including several first-time nominees and notable snubs of critically acclaimed performances.",
        "Music streaming data revealed that the most popular artist of the decade has amassed over 100 billion streams across all platforms representing an unprecedented achievement.",
        "New biographical drama series about the legendary musician shattered viewership records on its debut weekend attracting more simultaneous viewers than any previous streaming premiere.",
        "Hollywood studios are increasingly investing in video game adaptations following several high-profile successes that demonstrated the commercial potential of gaming intellectual property.",
        "International film festival awarded its top prize to a debut feature film from a previously unknown director whose low-budget production generated significant critical acclaim.",
        "Reality television franchise announced its highest-rated season in five years with finale viewership surpassing the previous record by a substantial margin among key demographics.",
        "Concert ticket prices for major touring artists have reached unprecedented levels with some premium seats selling for over ten thousand dollars on secondary market platforms.",
        "Animated feature film broke opening weekend records for the genre earning more than 200 million dollars domestically on its debut far exceeding studio projections.",
        "Independent film distribution landscape transformed significantly as streaming platforms aggressively acquire festival hits offering filmmakers wider audiences than traditional theatrical releases.",
        "Video game industry revenues now exceed the combined revenues of the film and music industries making it the dominant entertainment medium by total consumer spending.",
        "Celebrity couple announced their engagement generating massive social media engagement and driving record traffic to entertainment news websites within minutes of the announcement.",
        "Podcast advertising market grew by 35 percent last year as brands shifted budgets from traditional radio to on-demand audio content targeting younger demographic audiences.",
        "Broadway productions are experimenting with dynamic pricing models similar to airline tickets leading to controversy about affordability and accessibility of live theater performances.",
    ],
}

# Build flat dataset
all_articles = []
all_labels = []
for cat_idx, cat in enumerate(CATEGORIES):
    for article in ARTICLES[cat]:
        all_articles.append(article)
        all_labels.append(cat_idx)

# Shuffle
rng = np.random.RandomState(42)
idx = rng.permutation(len(all_articles))
all_articles = [all_articles[i] for i in idx]
all_labels = [all_labels[i] for i in idx]

# Train/test split (80/20)
split = int(0.8 * len(all_articles))
train_articles, test_articles = all_articles[:split], all_articles[split:]
train_labels, test_labels = all_labels[:split], all_labels[split:]

print(f"Total articles: {len(all_articles)}")
print(f"Categories: {CATEGORIES}")
print(f"Train: {len(train_articles)}  |  Test: {len(test_articles)}")
per_class = Counter(all_labels)
for cat_idx, cat in enumerate(CATEGORIES):
    print(f"  {cat:15s}: {per_class[cat_idx]} articles")
print()
print("Sample articles:")
for art, lbl in zip(all_articles[:2], all_labels[:2]):
    print(f"  [{CATEGORIES[lbl]}] {art[:90]}...")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: Text Preprocessing
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 2: Text Preprocessing")
print("=" * 70)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "it", "its", "this", "that",
    "these", "those", "i", "me", "my", "we", "our", "you", "your", "he",
    "she", "his", "her", "they", "their", "what", "which", "who", "how",
    "when", "where", "so", "if", "then", "than", "as", "up", "out", "all",
    "more", "also", "after", "over", "new", "just", "said", "say", "its",
}

# Domain keyword signals per category
DOMAIN_KEYWORDS = {
    "Technology": {"ai", "machine", "learning", "software", "hardware", "chip",
                   "algorithm", "data", "cloud", "cyber", "robot", "quantum",
                   "processor", "semiconductor", "neural", "model", "gpu"},
    "Sports": {"game", "team", "player", "season", "championship", "league",
               "tournament", "coach", "athlete", "match", "win", "score",
               "final", "olympic", "medal", "record"},
    "Politics": {"congress", "president", "senate", "election", "vote", "policy",
                 "government", "bill", "law", "democrat", "republican", "party",
                 "administration", "legislation", "federal", "regulation"},
    "Business": {"market", "stock", "economy", "company", "investment", "billion",
                 "revenue", "profit", "trade", "financial", "bank", "growth",
                 "inflation", "interest", "rate", "earnings"},
    "Entertainment": {"film", "movie", "music", "streaming", "celebrity", "award",
                      "actor", "singer", "director", "box", "office", "series",
                      "concert", "festival", "release", "show"},
}


def preprocess_news(text):
    """Preprocess news article text."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens


def domain_features(text):
    """Extract per-category keyword match features (5-dim vector)."""
    tokens = set(preprocess_news(text))
    feats = []
    for cat in CATEGORIES:
        keywords = DOMAIN_KEYWORDS[cat]
        overlap = len(tokens & keywords) / max(len(tokens), 1)
        feats.append(overlap)
    return np.array(feats, dtype=np.float32)


def length_features(text):
    """Text length and structural features."""
    words = text.split()
    sentences = text.split(".")
    return np.array([
        len(words) / 100,                        # normalized word count
        len(sentences) / 5,                      # sentence count
        np.mean([len(w) for w in words]) / 8 if words else 0,  # avg word length
    ], dtype=np.float32)


demo_text = all_articles[0]
dom_feat = domain_features(demo_text)
print("Domain feature demo:")
print(f"  Article: {demo_text[:80]}...")
print(f"  Label: {CATEGORIES[all_labels[0]]}")
for cat, score in zip(CATEGORIES, dom_feat):
    bar = "█" * int(score * 200)
    print(f"  {cat:15s} {score:.4f}  {bar}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: Multi-Class TF-IDF Vectorizer
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 3: TF-IDF Feature Extraction")
print("=" * 70)


class MultiClassTFIDF:
    """TF-IDF vectorizer with bigram support and sublinear TF scaling."""

    def __init__(self, max_features=500, ngram=(1, 2), sublinear_tf=True, min_df=2):
        self.max_features = max_features
        self.ngram = ngram
        self.sublinear_tf = sublinear_tf
        self.min_df = min_df
        self.vocab = {}
        self.idf = {}

    def _tokenize(self, text):
        tokens = preprocess_news(text)
        result = list(tokens)
        if self.ngram[1] >= 2:
            for i in range(len(tokens) - 1):
                result.append(f"{tokens[i]}_{tokens[i+1]}")
        return result

    def fit(self, texts):
        N = len(texts)
        df = Counter()
        term_freq = Counter()
        tokenized = []
        for text in texts:
            toks = self._tokenize(text)
            tokenized.append(toks)
            term_freq.update(toks)
            df.update(set(toks))

        # Filter by min_df and select top features
        eligible = {w for w, c in df.items() if c >= self.min_df}
        top = [w for w, _ in term_freq.most_common() if w in eligible][:self.max_features]
        self.vocab = {w: i for i, w in enumerate(top)}
        self.idf = {w: math.log((1 + N) / (1 + df[w])) + 1 for w in self.vocab}
        return self

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            toks = self._tokenize(text)
            tf = Counter(toks)
            total = max(len(toks), 1)
            for tok, cnt in tf.items():
                if tok in self.vocab:
                    j = self.vocab[tok]
                    tf_val = (1 + math.log(cnt)) if self.sublinear_tf else cnt / total
                    X[i, j] = tf_val * self.idf[tok]
        # L2 normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return X / norms

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)


print("Building TF-IDF features (sublinear TF, bigrams)...")
tfidf = MultiClassTFIDF(max_features=400, ngram=(1, 2), sublinear_tf=True, min_df=1)
X_train_tfidf = tfidf.fit_transform(train_articles)
X_test_tfidf = tfidf.transform(test_articles)

X_train_dom = np.array([domain_features(a) for a in train_articles])
X_test_dom = np.array([domain_features(a) for a in test_articles])

X_train_len = np.array([length_features(a) for a in train_articles])
X_test_len = np.array([length_features(a) for a in test_articles])

X_train = np.hstack([X_train_tfidf, X_train_dom, X_train_len])
X_test = np.hstack([X_test_tfidf, X_test_dom, X_test_len])
y_train = np.array(train_labels)
y_test = np.array(test_labels)

print(f"  TF-IDF   : {X_train_tfidf.shape[1]} features")
print(f"  Domain   : {X_train_dom.shape[1]} features")
print(f"  Length   : {X_train_len.shape[1]} features")
print(f"  Combined : {X_train.shape[1]} features")
print(f"  X_train  : {X_train.shape}  |  X_test: {X_test.shape}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: Softmax Classifier from Scratch
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 4: Softmax Classifier from Scratch")
print("=" * 70)

print("""
Multi-class classification uses SOFTMAX instead of sigmoid:

  For K classes:
    z_k = x · w_k + b_k          (one score per class)
    p_k = exp(z_k) / Σ exp(z_j)  (normalize to probabilities)

  Loss: Cross-Entropy = -Σ y_k * log(p_k)
  Gradient: ∂L/∂w_k = x * (p_k - y_k)

  This is equivalent to K binary classifiers sharing a normalization.
""")


class SoftmaxClassifier:
    """Multi-class logistic regression (softmax) with L2 regularization."""

    def __init__(self, n_classes, lr=0.1, epochs=300, lam=0.01):
        self.n_classes = n_classes
        self.lr = lr
        self.epochs = epochs
        self.lam = lam
        self.W = None   # shape: (n_features, n_classes)
        self.b = None   # shape: (n_classes,)
        self.losses = []
        self.train_accs = []

    def _softmax(self, Z):
        Z = Z - Z.max(axis=1, keepdims=True)  # numerical stability
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def _one_hot(self, y):
        Y = np.zeros((len(y), self.n_classes))
        Y[np.arange(len(y)), y] = 1
        return Y

    def fit(self, X, y):
        n, d = X.shape
        self.W = np.zeros((d, self.n_classes))
        self.b = np.zeros(self.n_classes)
        Y = self._one_hot(y)

        for epoch in range(self.epochs):
            # Forward pass
            Z = X @ self.W + self.b       # (n, K)
            P = self._softmax(Z)          # (n, K)
            # Loss: cross-entropy + L2
            loss = -np.mean(np.sum(Y * np.log(P + 1e-9), axis=1))
            loss += 0.5 * self.lam * np.sum(self.W ** 2)
            # Gradients
            dZ = (P - Y) / n             # (n, K)
            dW = X.T @ dZ + self.lam * self.W
            db = dZ.sum(axis=0)
            # Update
            self.W -= self.lr * dW
            self.b -= self.lr * db
            self.losses.append(loss)
            if epoch % 50 == 0:
                acc = self.score(X, y)
                self.train_accs.append(acc)
        return self

    def predict_proba(self, X):
        Z = X @ self.W + self.b
        return self._softmax(Z)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


print("Training Softmax Classifier from scratch...")
softmax_clf = SoftmaxClassifier(
    n_classes=len(CATEGORIES), lr=0.08, epochs=400, lam=0.001
)
softmax_clf.fit(X_train, y_train)
train_acc = softmax_clf.score(X_train, y_train)
test_acc = softmax_clf.score(X_test, y_test)
print(f"  Train accuracy: {train_acc:.3f}")
print(f"  Test  accuracy: {test_acc:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 5: Multi-Class Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 5: Multi-Class Evaluation Metrics")
print("=" * 70)

print("""
For K-class problems we need class-aware metrics:

  Per-class Precision_k = TP_k / (TP_k + FP_k)
  Per-class Recall_k    = TP_k / (TP_k + FN_k)
  Per-class F1_k        = 2 * P_k * R_k / (P_k + R_k)

  Macro-F1  = mean of all per-class F1 scores (treats classes equally)
  Micro-F1  = F1 computed from total TP/FP/FN (weights by class size)

  Confusion Matrix: C[i,j] = # samples of true class i predicted as class j
""")


def multiclass_metrics(y_true, y_pred, n_classes):
    """Compute confusion matrix and per-class + macro metrics."""
    # Confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    # Per-class metrics
    per_class = {}
    for k in range(n_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[k] = {"precision": prec, "recall": rec, "f1": f1,
                         "support": cm[k, :].sum()}

    accuracy = np.trace(cm) / cm.sum()
    macro_f1 = np.mean([per_class[k]["f1"] for k in range(n_classes)])

    # Micro F1
    total_tp = sum(cm[k, k] for k in range(n_classes))
    total_fp = sum(cm[:, k].sum() - cm[k, k] for k in range(n_classes))
    total_fn = sum(cm[k, :].sum() - cm[k, k] for k in range(n_classes))
    micro_prec = total_tp / max(total_tp + total_fp, 1)
    micro_rec = total_tp / max(total_tp + total_fn, 1)
    micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-9)

    return {
        "cm": cm,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_class": per_class,
    }


test_preds = softmax_clf.predict(X_test)
metrics = multiclass_metrics(y_test, test_preds, len(CATEGORIES))

print(f"Overall Accuracy : {metrics['accuracy']:.3f}")
print(f"Macro  F1        : {metrics['macro_f1']:.3f}")
print(f"Micro  F1        : {metrics['micro_f1']:.3f}")
print()
print(f"{'Category':15s} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
print("-" * 55)
for k, cat in enumerate(CATEGORIES):
    pc = metrics["per_class"][k]
    print(f"{cat:15s} {pc['precision']:>10.3f} {pc['recall']:>8.3f} {pc['f1']:>8.3f} {pc['support']:>8}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 6: sklearn Comparison + Cross-Validation
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 6: sklearn Model Comparison + Cross-Validation")
print("=" * 70)

sklearn_results = {}
SKLEARN_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import f1_score

    SKLEARN_AVAILABLE = True
    print("sklearn available — running comparison")

    models = {
        "LogReg (OvR)": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=1)),
            ("clf", LogisticRegression(C=1.0, multi_class="ovr",
                                       max_iter=1000, random_state=42)),
        ]),
        "LogReg (Multinomial)": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=1)),
            ("clf", LogisticRegression(C=1.0, multi_class="multinomial",
                                       solver="lbfgs", max_iter=1000, random_state=42)),
        ]),
        "ComplementNB": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=1)),
            ("clf", ComplementNB(alpha=0.5)),
        ]),
        "LinearSVC": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=1)),
            ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42)),
        ]),
    }

    print(f"\n{'Model':25s} {'Test Acc':>10} {'Macro F1':>10} {'CV F1 (5-fold)':>15}")
    print("-" * 65)

    # Scratch model first
    scratch_macro = metrics["macro_f1"]
    print(f"{'Softmax (scratch)':25s} {metrics['accuracy']:>10.3f} {scratch_macro:>10.3f} {'N/A':>15}")

    for name, pipe in models.items():
        pipe.fit(train_articles, train_labels)
        preds = np.array(pipe.predict(test_articles))
        m = multiclass_metrics(y_test, preds, len(CATEGORIES))
        # 5-fold CV on training set
        cv_scores = cross_val_score(pipe, train_articles, train_labels,
                                    cv=5, scoring="f1_macro")
        sklearn_results[name] = {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "metrics": m,
        }
        print(f"{name:25s} {m['accuracy']:>10.3f} {m['macro_f1']:>10.3f} "
              f"{cv_scores.mean():.3f}±{cv_scores.std():.3f}")

    # Best model per-class breakdown
    best_name = max(sklearn_results, key=lambda k: sklearn_results[k]["macro_f1"])
    best = sklearn_results[best_name]
    print(f"\nBest model: {best_name} (Macro F1={best['macro_f1']:.3f})")
    print(f"\n{'Category':15s} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("-" * 45)
    for k, cat in enumerate(CATEGORIES):
        pc = best["metrics"]["per_class"][k]
        print(f"{cat:15s} {pc['precision']:>10.3f} {pc['recall']:>8.3f} {pc['f1']:>8.3f}")

except ImportError:
    print("sklearn not available — skipping comparison")
    print("Install: pip install scikit-learn")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 7: Deep Learning — CNN Text Classifier
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 7: Deep Learning — CNN Text Classifier")
print("=" * 70)

print("""
TextCNN (Kim 2014) uses 1D convolutions over word embeddings:

  Embed → Conv1D (filter sizes 2,3,4) → GlobalMaxPool → Dense → Softmax

  Each filter detects different n-gram patterns.
  GlobalMaxPool captures the strongest activation across the sequence.
  Much faster than LSTM while achieving competitive accuracy.
""")

TF_AVAILABLE = False
keras_history = None
keras_test_acc = None
keras_metrics = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    TF_AVAILABLE = True
    print(f"TensorFlow {tf.__version__} detected — building TextCNN")

    MAX_VOCAB = 5000
    MAX_LEN = 60
    EMBED_DIM = 64

    # Build vocabulary from training set
    all_train_tokens = [preprocess_news(a) for a in train_articles]
    freq = Counter(tok for toks in all_train_tokens for tok in toks)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, _ in freq.most_common(MAX_VOCAB - 2):
        vocab[w] = len(vocab)

    def encode(tokens, vocab, max_len):
        ids = [vocab.get(t, 1) for t in tokens[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids

    X_tr_cnn = np.array([encode(preprocess_news(a), vocab, MAX_LEN) for a in train_articles])
    X_te_cnn = np.array([encode(preprocess_news(a), vocab, MAX_LEN) for a in test_articles])

    # TextCNN model
    inputs = keras.Input(shape=(MAX_LEN,))
    embed = layers.Embedding(len(vocab), EMBED_DIM, mask_zero=False)(inputs)

    # Multiple filter sizes to capture different n-gram windows
    convs = []
    for fs in [2, 3, 4]:
        c = layers.Conv1D(64, fs, activation="relu", padding="valid")(embed)
        c = layers.GlobalMaxPooling1D()(c)
        convs.append(c)

    x = layers.Concatenate()(convs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(len(CATEGORIES), activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"  TextCNN — vocab={len(vocab)}, max_len={MAX_LEN}, filters=[2,3,4]×64")

    keras_history = model.fit(
        X_tr_cnn, np.array(train_labels),
        epochs=25, batch_size=16, validation_split=0.15,
        verbose=0,
    )
    _, keras_test_acc = model.evaluate(X_te_cnn, np.array(test_labels), verbose=0)
    keras_preds = np.argmax(model.predict(X_te_cnn, verbose=0), axis=1)
    keras_metrics = multiclass_metrics(y_test, keras_preds, len(CATEGORIES))
    print(f"  TextCNN Test Accuracy : {keras_test_acc:.3f}")
    print(f"  TextCNN Macro F1      : {keras_metrics['macro_f1']:.3f}")

except ImportError:
    print("TensorFlow not available — skipping TextCNN")
    print("Install: pip install tensorflow")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 8: Production Inference API
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 8: Production classify_article() API")
print("=" * 70)


def classify_article(text, model=softmax_clf, vectorizer=tfidf,
                     categories=CATEGORIES, top_k=3):
    """
    Classify a news article into one of N categories.

    Args:
        text       : Raw article text
        model      : Trained SoftmaxClassifier
        vectorizer : Fitted MultiClassTFIDF
        categories : List of category names
        top_k      : Number of top predictions to return

    Returns:
        dict with 'category', 'confidence', 'top_k_predictions', 'key_signals'
    """
    # Build feature vector
    tfidf_vec = vectorizer.transform([text])
    dom_vec = domain_features(text).reshape(1, -1)
    len_vec = length_features(text).reshape(1, -1)
    X = np.hstack([tfidf_vec, dom_vec, len_vec])

    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))

    # Top-K predictions
    top_k_idx = np.argsort(probs)[::-1][:top_k]
    top_k_preds = [{"category": categories[i], "confidence": round(float(probs[i]), 3)}
                   for i in top_k_idx]

    # Key domain signals
    tokens = set(preprocess_news(text))
    key_signals = []
    for cat in CATEGORIES:
        matched = tokens & DOMAIN_KEYWORDS[cat]
        if matched:
            key_signals.append(f"{cat}: {', '.join(list(matched)[:3])}")

    return {
        "category": categories[pred_idx],
        "confidence": round(float(probs[pred_idx]), 3),
        "top_k_predictions": top_k_preds,
        "key_signals": key_signals[:3],
    }


# Demo articles (held-out examples not in training set)
demo_articles = [
    ("New quantum processor from IBM achieves breakthrough in computational speed "
     "for machine learning workloads, outperforming classical chips by 1000x on "
     "specific optimization tasks."),
    ("The championship final saw the home team triumph in extra time, with the "
     "star striker scoring a hat-trick to secure the title in a thrilling match."),
    ("Federal Reserve chair signaled potential interest rate cuts as inflation data "
     "shows sustained decline toward the central bank's two percent target."),
    ("The blockbuster sequel shattered opening weekend records at the global box "
     "office, earning over 300 million dollars in its first three days of release."),
    ("Congressional leaders reached a bipartisan deal on the spending bill after "
     "weeks of negotiations, narrowly avoiding a government shutdown deadline."),
]

print("classify_article() API demo:")
for article in demo_articles:
    result = classify_article(article)
    print(f"\n  Article: {article[:80]}...")
    print(f"  → {result['category']:15s} (confidence={result['confidence']:.1%})")
    print(f"  Top-3: {', '.join(f\"{p['category']}({p['confidence']:.2f})\" for p in result['top_k_predictions'])}")
    if result["key_signals"]:
        print(f"  Signals: {' | '.join(result['key_signals'][:2])}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 9: Visualizations
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SECTION 9: Generating Visualizations")
print("=" * 70)

COLORS = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

# ── Visualization 1: Dataset & Model Overview ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle("News Article Classifier — Overview", fontsize=15, fontweight="bold")

# 1a: Class distribution
ax = axes[0, 0]
cat_counts = [per_class[k] for k in range(len(CATEGORIES))]
bars = ax.bar(CATEGORIES, cat_counts, color=COLORS, alpha=0.85, edgecolor="white", linewidth=2)
for bar, cnt in zip(bars, cat_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(cnt),
            ha="center", fontsize=11, fontweight="bold")
ax.set_title("Articles per Category", fontsize=12, fontweight="bold")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=15)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 1b: Article word count distribution
ax = axes[0, 1]
for k, (cat, color) in enumerate(zip(CATEGORIES, COLORS)):
    lens = [len(a.split()) for a, l in zip(all_articles, all_labels) if l == k]
    ax.hist(lens, bins=8, alpha=0.5, color=color, label=cat, edgecolor="white")
ax.set_title("Article Length Distribution", fontsize=12, fontweight="bold")
ax.set_xlabel("Word Count")
ax.set_ylabel("Frequency")
ax.legend(fontsize=7)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 1c: Domain keyword heatmap
ax = axes[0, 2]
dom_matrix = np.array([domain_features(a) for a in all_articles])
# Average by category
avg_dom = np.zeros((len(CATEGORIES), len(CATEGORIES)))
for k in range(len(CATEGORIES)):
    mask = np.array(all_labels) == k
    if mask.sum() > 0:
        avg_dom[k] = dom_matrix[mask].mean(axis=0)
im = ax.imshow(avg_dom, cmap="YlOrRd", aspect="auto", vmin=0)
ax.set_xticks(range(len(CATEGORIES)))
ax.set_yticks(range(len(CATEGORIES)))
ax.set_xticklabels([c[:5] for c in CATEGORIES], fontsize=9)
ax.set_yticklabels(CATEGORIES, fontsize=9)
ax.set_title("Domain Keyword Overlap\n(True Category × Domain)", fontsize=11, fontweight="bold")
ax.set_xlabel("Keyword Domain →")
ax.set_ylabel("True Category →")
for i in range(len(CATEGORIES)):
    for j in range(len(CATEGORIES)):
        ax.text(j, i, f"{avg_dom[i, j]:.3f}", ha="center", va="center",
                fontsize=8, color="black" if avg_dom[i, j] < avg_dom.max() * 0.6 else "white")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 1d: Confusion matrix
ax = axes[1, 0]
cm = metrics["cm"]
im = ax.imshow(cm, cmap="Blues", aspect="auto")
ax.set_xticks(range(len(CATEGORIES)))
ax.set_yticks(range(len(CATEGORIES)))
ax.set_xticklabels([c[:5] for c in CATEGORIES], fontsize=9, rotation=15)
ax.set_yticklabels([c[:5] for c in CATEGORIES], fontsize=9)
ax.set_title(f"Confusion Matrix (Softmax)\nAcc={metrics['accuracy']:.2f}  MacroF1={metrics['macro_f1']:.2f}",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
for i in range(len(CATEGORIES)):
    for j in range(len(CATEGORIES)):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black")

# 1e: Per-class F1 comparison
ax = axes[1, 1]
x = np.arange(len(CATEGORIES))
w = 0.25
models_to_plot = [
    ("Softmax\n(scratch)", [metrics["per_class"][k]["f1"] for k in range(len(CATEGORIES))], "#9b59b6"),
]
if SKLEARN_AVAILABLE and sklearn_results:
    best_n = max(sklearn_results, key=lambda k: sklearn_results[k]["macro_f1"])
    best_m = sklearn_results[best_n]["metrics"]
    models_to_plot.append(
        (best_n.replace(" (", "\n("),
         [best_m["per_class"][k]["f1"] for k in range(len(CATEGORIES))],
         "#3498db")
    )
if TF_AVAILABLE and keras_metrics:
    models_to_plot.append(
        ("TextCNN\n(Keras)",
         [keras_metrics["per_class"][k]["f1"] for k in range(len(CATEGORIES))],
         "#2ecc71")
    )

n_models = len(models_to_plot)
offsets = np.linspace(-w * (n_models - 1) / 2, w * (n_models - 1) / 2, n_models)
for (name, f1s, color), offset in zip(models_to_plot, offsets):
    ax.bar(x + offset, f1s, w * 0.9, label=name, color=color, alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels([c[:5] for c in CATEGORIES], fontsize=9)
ax.set_ylim(0, 1.2)
ax.set_title("Per-Class F1 Score by Model", fontsize=12, fontweight="bold")
ax.set_ylabel("F1 Score")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 1f: Training loss
ax = axes[1, 2]
ax.plot(softmax_clf.losses, color="#9b59b6", linewidth=2, label="Train Loss")
ax.set_title("Training Loss — Softmax Classifier", fontsize=12, fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
if TF_AVAILABLE and keras_history:
    ax2 = ax.twinx()
    ax2.plot(keras_history.history["val_accuracy"], color="#2ecc71",
             linewidth=2, linestyle="--", label="TextCNN Val Acc")
    ax2.set_ylabel("Validation Accuracy", color="#2ecc71")
    ax2.tick_params(axis="y", labelcolor="#2ecc71")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_classifier_overview.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/01_classifier_overview.png")

# ── Visualization 2: Feature Analysis + Model Weights ─────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 7))
fig.suptitle("Feature Analysis — News Article Classifier", fontsize=14, fontweight="bold")

# 2a: Top TF-IDF features per class (from softmax weight matrix)
ax = axes[0]
top_features_per_class = {}
vocab_list = list(tfidf.vocab.keys())
n_vocab = len(vocab_list)
for k in range(len(CATEGORIES)):
    w_k = softmax_clf.W[:n_vocab, k]
    top_idx = np.argsort(w_k)[-5:][::-1]
    top_features_per_class[k] = [(vocab_list[i] if i < len(vocab_list) else "", w_k[i])
                                  for i in top_idx if i < len(vocab_list)]

y_pos = 0
yticks_pos = []
ytick_labels = []
for k, (cat, color) in enumerate(zip(CATEGORIES, COLORS)):
    feats = top_features_per_class[k]
    for word, weight in reversed(feats):
        ax.barh(y_pos, weight, color=color, alpha=0.8)
        ytick_labels.append(f"{word}")
        yticks_pos.append(y_pos)
        y_pos += 1
    y_pos += 0.5  # gap between categories

ax.set_yticks(yticks_pos)
ax.set_yticklabels(ytick_labels, fontsize=8)
ax.axvline(0, color="black", linewidth=1)
ax.set_title("Top Discriminative Words\nper Category (Softmax W)", fontsize=11, fontweight="bold")
ax.set_xlabel("Weight")
patches = [mpatches.Patch(color=c, label=cat) for cat, c in zip(CATEGORIES, COLORS)]
ax.legend(handles=patches, fontsize=8, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 2b: Model accuracy comparison (overall)
ax = axes[1]
model_names_all = ["Softmax\n(scratch)"]
model_accs_all = [metrics["accuracy"]]
model_f1s_all = [metrics["macro_f1"]]
if SKLEARN_AVAILABLE:
    for name, res in sklearn_results.items():
        model_names_all.append(name.replace(" (", "\n("))
        model_accs_all.append(res["accuracy"])
        model_f1s_all.append(res["macro_f1"])
if TF_AVAILABLE and keras_metrics:
    model_names_all.append("TextCNN\n(Keras)")
    model_accs_all.append(keras_test_acc)
    model_f1s_all.append(keras_metrics["macro_f1"])

xv = np.arange(len(model_names_all))
bw = 0.35
bars_acc = ax.bar(xv - bw / 2, model_accs_all, bw, label="Accuracy", color="#3498db", alpha=0.85)
bars_f1 = ax.bar(xv + bw / 2, model_f1s_all, bw, label="Macro F1", color="#e67e22", alpha=0.85)
ax.set_xticks(xv)
ax.set_xticklabels(model_names_all, fontsize=8)
ax.set_ylim(0, 1.2)
ax.set_title("Model Comparison\n(Test Set)", fontsize=12, fontweight="bold")
ax.set_ylabel("Score")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar in list(bars_acc) + list(bars_f1):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{bar.get_height():.2f}", ha="center", fontsize=7)

# 2c: Prediction confidence for test samples
ax = axes[2]
all_probs = softmax_clf.predict_proba(X_test)
correct_conf = all_probs.max(axis=1)[test_preds == y_test]
wrong_conf = all_probs.max(axis=1)[test_preds != y_test]
ax.hist(correct_conf, bins=10, alpha=0.6, color="#2ecc71", label=f"Correct (n={len(correct_conf)})", edgecolor="white")
if len(wrong_conf) > 0:
    ax.hist(wrong_conf, bins=5, alpha=0.6, color="#e74c3c", label=f"Wrong (n={len(wrong_conf)})", edgecolor="white")
ax.set_title("Prediction Confidence\n(Correct vs Wrong)", fontsize=12, fontweight="bold")
ax.set_xlabel("Max Softmax Probability")
ax.set_ylabel("Count")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_feature_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/02_feature_analysis.png")

# ── Visualization 3: Pipeline Architecture Diagram ────────────────────
fig, ax = plt.subplots(figsize=(17, 9))
ax.set_xlim(0, 17)
ax.set_ylim(0, 9)
ax.axis("off")
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")
ax.set_title("News Article Classification — Production Pipeline", fontsize=15, fontweight="bold", pad=20)

# Pipeline stages
STAGES = [
    ("Raw\nArticle", 1.0, "#3498db"),
    ("Preprocess\n& Tokenize", 3.2, "#9b59b6"),
    ("Feature\nExtraction", 6.0, "#e67e22"),
    ("Softmax /\nTextCNN", 9.5, "#2ecc71"),
    ("Category\nPrediction", 13.0, "#e74c3c"),
    ("Top-K &\nExplain", 15.8, "#1abc9c"),
]

for i, (name, x, color) in enumerate(STAGES):
    rect = mpatches.FancyBboxPatch((x - 0.9, 3.3), 1.8, 1.4,
                                    boxstyle="round,pad=0.1",
                                    facecolor=color, alpha=0.85, zorder=3)
    ax.add_patch(rect)
    ax.text(x, 4.0, name, ha="center", va="center", fontsize=9,
            fontweight="bold", color="white", zorder=4)
    if i < len(STAGES) - 1:
        next_x = STAGES[i + 1][1]
        ax.annotate("", xy=(next_x - 0.9, 4.0), xytext=(x + 0.9, 4.0),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color="#7f8c8d"))

# Detail boxes below each stage
DETAILS = [
    (1.0, 2.5, "\"Apple unveiled its\nnew AI processor...\""),
    (3.2, 2.5, "lowercase\nstopwords removed\nbigrams formed"),
    (6.0, 2.5, "TF-IDF (400d)\n+ Domain (5d)\n+ Length (3d)"),
    (9.5, 2.5, "Softmax: W (408×5)\nTextCNN: [2,3,4]×64\nfilters → Dense(5)"),
    (13.0, 2.5, "Technology: 0.91\nBusiness:    0.05\nSports:      0.02"),
    (15.8, 2.5, "Key signals:\nai, chip, processor\nConf: 91%"),
]
for x, y, text in DETAILS:
    ax.text(x, y, text, ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#bdc3c7", alpha=0.9))
    ax.annotate("", xy=(x, 3.3), xytext=(x, y + 0.05),
                arrowprops=dict(arrowstyle="-", lw=1, color="#bdc3c7", linestyle="dashed"))

# Performance summary panel
perf_lines = ["Performance Summary (Test Set)", "─" * 34,
              f"Softmax (scratch)  Acc={metrics['accuracy']:.2f}  MacroF1={metrics['macro_f1']:.2f}"]
if SKLEARN_AVAILABLE and sklearn_results:
    best_n2 = max(sklearn_results, key=lambda k: sklearn_results[k]["macro_f1"])
    r2 = sklearn_results[best_n2]
    perf_lines.append(f"{best_n2:18s} Acc={r2['accuracy']:.2f}  MacroF1={r2['macro_f1']:.2f}")
if TF_AVAILABLE and keras_metrics:
    perf_lines.append(f"TextCNN (Keras)    Acc={keras_test_acc:.2f}  MacroF1={keras_metrics['macro_f1']:.2f}")

ax.text(8.5, 8.7, "\n".join(perf_lines), ha="center", va="top", fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor="#2c3e50", alpha=0.95))

# Category legend
legend_x = 0.5
legend_y = 1.7
ax.text(legend_x, legend_y + 0.3, "Categories:", fontsize=10, fontweight="bold")
for k, (cat, color) in enumerate(zip(CATEGORIES, COLORS)):
    rx = legend_x + k * 3.2
    rect = mpatches.FancyBboxPatch((rx, legend_y - 0.35), 2.9, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor=color, alpha=0.75, zorder=3)
    ax.add_patch(rect)
    ax.text(rx + 1.45, legend_y - 0.1, cat, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", zorder=4)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_pipeline_diagram.png", dpi=300, bbox_inches="tight")
plt.close()
print(f"  Saved: {VIS_DIR}/03_pipeline_diagram.png")

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PROJECT SUMMARY — News Article Classifier")
print("=" * 70)
print(f"""
What we built:
  ✓ Synthetic 5-class news dataset (75 training articles)
  ✓ TF-IDF vectorizer with sublinear TF + bigrams (400 features)
  ✓ Domain keyword features (5 per-category signals)
  ✓ Text length/structure features (3 signals)
  ✓ Softmax classifier from scratch (numpy, L2 regularized)
  ✓ Multi-class evaluation: per-class P/R/F1, macro/micro F1
  ✓ sklearn comparison: LogReg OvR/Multinomial, NaiveBayes, SVM
  {'✓ TextCNN (Keras): 1D conv with filter sizes [2,3,4]' if TF_AVAILABLE else '○ TextCNN (install tensorflow)'}
  ✓ Production classify_article() API with top-K + explanation

Model Performance (Test Set):
  Softmax (scratch)    | Acc={metrics['accuracy']:.3f}  MacroF1={metrics['macro_f1']:.3f}
""")
if SKLEARN_AVAILABLE:
    for name, res in sklearn_results.items():
        print(f"  {name:24s} | Acc={res['accuracy']:.3f}  MacroF1={res['macro_f1']:.3f}")
if TF_AVAILABLE and keras_metrics:
    print(f"  TextCNN (Keras)          | Acc={keras_test_acc:.3f}  MacroF1={keras_metrics['macro_f1']:.3f}")

print(f"""
Key takeaways:
  • Multi-class problems use softmax (not sigmoid) — one score per class
  • Macro F1 is more informative than accuracy for imbalanced classes
  • Domain keyword features provide strong category-specific signals
  • TextCNN captures local n-gram patterns without sequential recurrence
  • OvR vs Multinomial LogReg — both work; multinomial is more principled

Visualizations saved to: {VIS_DIR}/
  01_classifier_overview.png  — dataset + confusion matrix + per-class F1
  02_feature_analysis.png     — feature weights + model comparison
  03_pipeline_diagram.png     — end-to-end system diagram
""")
