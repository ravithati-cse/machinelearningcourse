"""
🏷️ NLP — Algorithm 4: Named Entity Recognition (NER)
======================================================

Learning Objectives:
  1. Understand NER: label each token with its entity type
  2. Master BIO / BIOES tagging schemes from scratch
  3. Build a rule-based NER with regex and gazetteers
  4. Train a CRF-based NER using sklearn_crfsuite (feature engineering)
  5. Use spaCy for production NER with pretrained models
  6. Evaluate with span-level F1 (seqeval) — the standard NER metric
  7. Visualize entity distributions and span predictions

YouTube Resources:
  ⭐ HuggingFace — token classification https://www.youtube.com/watch?v=wVHdVlPScxA
  ⭐ Explosion.ai — spaCy NER https://www.youtube.com/watch?v=THduWAnG97k
  📚 spaCy NER docs https://spacy.io/usage/linguistic-features#named-entities

Time Estimate: 70 min
Difficulty: Intermediate-Advanced
Prerequisites: lstm_text_classifier.py, 04_rnn_intuition.py
Key Concepts: BIO tagging, token classification, CRF, seqeval F1, span extraction
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from collections import defaultdict, Counter
import os

_VISUALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "named_entity_recognition")
os.makedirs(_VISUALS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("🏷️ NLP ALGORITHM 4: NAMED ENTITY RECOGNITION (NER)")
print("=" * 70)
print()
print("NER = label each word in a sentence with its entity type.")
print()
print("  Input:  'Apple Inc. was founded by Steve Jobs in Cupertino, California'")
print("  Output: 'Apple Inc.' → ORG")
print("          'Steve Jobs' → PER")
print("          'Cupertino'  → LOC")
print("          'California' → LOC")
print()
print("Common entity types (CoNLL-2003 standard):")
print("  PER  = Person name (John Smith, Marie Curie)")
print("  ORG  = Organization (Apple, WHO, United Nations)")
print("  LOC  = Location (Paris, Mount Everest, Pacific Ocean)")
print("  MISC = Miscellaneous (English, the Olympics, iPhone 15)")
print("  O    = No entity")
print()


# ======================================================================
# SECTION 1: BIO Tagging Scheme
# ======================================================================
print("=" * 70)
print("SECTION 1: BIO TAGGING SCHEME")
print("=" * 70)
print()
print("NER is a SEQUENCE LABELING task — we label each token, not each document.")
print()
print("  BIO scheme: B = Begin, I = Inside, O = Outside")
print()
print("  Text:  'Steve  Jobs  founded  Apple  Inc.  in  California'")
print("  Tags:  B-PER  I-PER  O        B-ORG  I-ORG  O  B-LOC")
print()
print("  Why not just label each word with the entity type?")
print("  → 'New York City' would be: [LOC, LOC, LOC]")
print("     Ambiguous: is it one entity or three?")
print("  → BIO: [B-LOC, I-LOC, I-LOC] — clearly ONE entity spanning 3 words")
print()
print("  BIOES (more expressive, used by spaCy):")
print("  B = Begin, I = Inside, O = Outside, E = End, S = Single")
print()
print("  'Steve Jobs is at the White House'")
print("  B-PER I-PER O  O  O   B-LOC I-LOC → B is 2 tokens (BIO)")
print("  B-PER E-PER O  O  O   B-LOC E-LOC → E marks the last token explicitly")
print("  S-PER                               → S for single-token entities")
print()


def bio_tag(text, entities):
    """
    Apply BIO tagging to text given a list of (start_char, end_char, label) entities.
    Returns: list of (token, bio_tag) pairs
    """
    tokens = text.split()
    tags   = ["O"] * len(tokens)

    # Build char-to-token mapping
    char_pos = 0
    tok_starts = []
    tok_ends   = []
    for tok in tokens:
        idx = text.find(tok, char_pos)
        tok_starts.append(idx)
        tok_ends.append(idx + len(tok))
        char_pos = idx + len(tok)

    for ent_start, ent_end, label in entities:
        for ti, (ts, te) in enumerate(zip(tok_starts, tok_ends)):
            if ts >= ent_start and te <= ent_end:
                if ts == ent_start:
                    tags[ti] = f"B-{label}"
                else:
                    tags[ti] = f"I-{label}"

    return list(zip(tokens, tags))


# Demonstration
demo_text     = "Steve Jobs founded Apple Inc in Cupertino California"
demo_entities = [
    (0,  10, "PER"),   # Steve Jobs
    (19, 28, "ORG"),   # Apple Inc
    (32, 41, "LOC"),   # Cupertino
    (42, 52, "LOC"),   # California
]

tagged = bio_tag(demo_text, demo_entities)
print("  BIO tagging demonstration:")
print(f"  Text: {demo_text!r}")
print()
print(f"  {'Token':<14} {'BIO Tag'}")
print(f"  {'─'*14} {'─'*12}")
for token, tag in tagged:
    color_sym = "●" if tag != "O" else "○"
    print(f"  {token:<14} {tag:<12} {color_sym}")
print()


# ======================================================================
# SECTION 2: Rule-Based NER (Gazetteer + Regex)
# ======================================================================
print("=" * 70)
print("SECTION 2: RULE-BASED NER (GAZETTEER + REGEX)")
print("=" * 70)
print()
print("Gazetteer = curated list of entity names.")
print("Fast, transparent, but requires manual curation and misses new entities.")
print()

# Simplified gazetteers
GAZETTEERS = {
    "PER": {
        "steve jobs", "elon musk", "tim cook", "sundar pichai", "jeff bezos",
        "bill gates", "mark zuckerberg", "satya nadella", "warren buffett",
        "jensen huang", "sam altman", "demis hassabis", "yann lecun",
    },
    "ORG": {
        "apple", "google", "microsoft", "amazon", "meta", "tesla", "nvidia",
        "openai", "deepmind", "anthropic", "ibm", "samsung", "intel",
        "alphabet", "facebook", "twitter", "netflix", "spotify",
    },
    "LOC": {
        "california", "new york", "london", "paris", "tokyo", "beijing",
        "seattle", "san francisco", "cupertino", "mountain view", "redmond",
        "silicon valley", "hong kong", "berlin", "sydney", "toronto",
    },
}

# Regex patterns for MISC / special entities
MISC_PATTERNS = [
    (r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", "DATE"),
    (r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", "DATE"),
    (r"\b\$[\d,]+(?:\.\d+)?\b", "MONEY"),
    (r"\b\d+\s*(?:million|billion|trillion)\s*(?:dollars?)?\b", "MONEY"),
    (r"\b(?:Q[1-4]|quarter [1-4])\s*\d{4}\b", "DATE"),
]


class RuleBasedNER:
    """Simple rule-based NER using gazetteers and regex."""

    def __init__(self, gazetteers, misc_patterns):
        self.gazetteers      = {k: v for k, v in gazetteers.items()}
        self.misc_patterns   = misc_patterns

    def predict(self, text):
        """Returns list of (entity_text, label, start, end) tuples."""
        entities = []
        text_lower = text.lower()
        tokens = text.split()

        # 1. Multi-word gazetteer matching (try 3-gram, 2-gram, 1-gram)
        for n in range(3, 0, -1):
            for i in range(len(tokens) - n + 1):
                span = " ".join(tokens[i:i+n]).lower()
                for label, names in self.gazetteers.items():
                    if span in names:
                        # Check not already covered
                        orig_span = " ".join(tokens[i:i+n])
                        start     = text.find(orig_span)
                        end       = start + len(orig_span)
                        # Don't double-add
                        if not any(s <= start < e for _, _, s, e in entities):
                            entities.append((orig_span, label, start, end))

        # 2. Regex patterns
        for pattern, label in self.misc_patterns:
            for m in re.finditer(pattern, text_lower):
                if not any(s <= m.start() < e for _, _, s, e in entities):
                    entities.append((text[m.start():m.end()], label, m.start(), m.end()))

        return sorted(entities, key=lambda x: x[2])

    def to_bio(self, text, entities):
        """Convert to BIO-tagged tokens."""
        tokens = text.split()
        tags   = ["O"] * len(tokens)
        char_pos = 0
        tok_starts, tok_ends = [], []
        for tok in tokens:
            idx = text.find(tok, char_pos)
            tok_starts.append(idx)
            tok_ends.append(idx + len(tok))
            char_pos = idx + len(tok)

        for ent_text, label, ent_start, ent_end in entities:
            is_first = True
            for ti, (ts, te) in enumerate(zip(tok_starts, tok_ends)):
                if ts >= ent_start and te <= ent_end:
                    tags[ti] = f"B-{label}" if is_first else f"I-{label}"
                    is_first = False

        return list(zip(tokens, tags))


ner = RuleBasedNER(GAZETTEERS, MISC_PATTERNS)

test_sents = [
    "Apple CEO Tim Cook announced that Nvidia and Microsoft will collaborate",
    "Elon Musk said Tesla will expand to London and Paris next quarter",
    "Google acquired DeepMind for $500 million back in 2014",
    "Anthropic was founded in San Francisco and focuses on AI safety",
    "Meta and Google are both headquartered in California",
]

print("  Rule-based NER predictions:")
print()
for sent in test_sents:
    entities = ner.predict(sent)
    bio      = ner.to_bio(sent, entities)
    print(f"  Text: {sent}")
    if entities:
        print(f"  Entities:")
        for ent_text, label, start, end in entities:
            print(f"    [{label}] {ent_text!r}")
    print()


# ======================================================================
# SECTION 3: Feature-Based NER (CRF-style Features)
# ======================================================================
print("=" * 70)
print("SECTION 3: FEATURE ENGINEERING FOR NER")
print("=" * 70)
print()
print("CRF (Conditional Random Field) NER uses hand-crafted FEATURES per token.")
print("Each token gets a feature dictionary; CRF learns which features predict entity type.")
print()
print("Rich features capture:")
print("  • Token shape: 'Apple' → starts_upper, has_alpha, not_all_lower")
print("  • Prefix/suffix: 'tion', 'ing', 'Inc', '.com'")
print("  • Context window: previous and next 2 tokens")
print("  • POS-like patterns: follows 'Mr.', 'the', 'in'")
print("  • Gazetteer match: 'is_known_org', 'is_known_person'")
print()


def token_features(tokens, i, gazetteers):
    """
    Extract feature dictionary for token at position i.
    This is the feature vector CRF uses per token.
    """
    tok = tokens[i]
    word_lower = tok.lower()

    features = {
        # Token-level features
        "bias":           1.0,
        "word":           word_lower,
        "word[:3]":       word_lower[:3],
        "word[-3:]":      word_lower[-3:],
        "word_upper":     tok.isupper(),
        "word_title":     tok.istitle(),
        "word_digit":     tok.isdigit(),
        "word_has_digit": any(c.isdigit() for c in tok),
        "word_has_upper": any(c.isupper() for c in tok),
        "word_has_dash":  "-" in tok,
        "word_len":       len(tok),
        "is_short":       len(tok) <= 3,
        "starts_upper":   tok[0].isupper() if tok else False,

        # Gazetteer features
        "in_per_gaz":   word_lower in gazetteers.get("PER", set()),
        "in_org_gaz":   word_lower in gazetteers.get("ORG", set()),
        "in_loc_gaz":   word_lower in gazetteers.get("LOC", set()),

        # Position
        "is_first":     i == 0,
        "is_last":      i == len(tokens) - 1,
    }

    # Context: previous token
    if i > 0:
        prev = tokens[i-1]
        features.update({
            "prev_word":       prev.lower(),
            "prev_word_upper": prev.isupper(),
            "prev_word_title": prev.istitle(),
        })
    else:
        features["BOS"] = True

    # Context: next token
    if i < len(tokens) - 1:
        nxt = tokens[i+1]
        features.update({
            "next_word":       nxt.lower(),
            "next_word_upper": nxt.isupper(),
            "next_word_title": nxt.istitle(),
        })
    else:
        features["EOS"] = True

    # 2 tokens ahead
    if i < len(tokens) - 2:
        features["next2_word"] = tokens[i+2].lower()

    return features


# Show features for a sample token
sample_sentence = "Steve Jobs founded Apple Inc in Silicon Valley".split()
sample_idx      = 0  # "Steve"
feat_dict       = token_features(sample_sentence, sample_idx, GAZETTEERS)

print(f"  Features for token {sample_idx} ({sample_sentence[sample_idx]!r}) in:")
print(f"  '{' '.join(sample_sentence)}'")
print()
for feat_name, feat_val in list(feat_dict.items())[:20]:
    print(f"    {feat_name:<25}: {feat_val}")
print(f"  ... ({len(feat_dict)} total features)")
print()


# ======================================================================
# SECTION 4: spaCy NER (Production)
# ======================================================================
print("=" * 70)
print("SECTION 4: SPACY NER — PRODUCTION-READY")
print("=" * 70)
print()
print("spaCy provides pretrained NER models for 60+ languages.")
print()
print("  import spacy")
print("  nlp  = spacy.load('en_core_web_sm')     # 12MB, fast")
print("  nlp  = spacy.load('en_core_web_trf')    # 438MB, transformer-based, best")
print()
print("  doc  = nlp('Apple CEO Tim Cook was in London')")
print("  for ent in doc.ents:")
print("      print(ent.text, ent.label_, ent.start_char, ent.end_char)")
print()
print("  Output:")
print("    Apple    ORG     0   5")
print("    Tim Cook PERSON  10  18")
print("    London   GPE     26  32")
print()

SPACY_AVAILABLE = False
try:
    import spacy
    SPACY_AVAILABLE = True
    print("  spaCy available! Loading model...")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("  Using en_core_web_sm")
        print()

        for sent in test_sents[:3]:
            doc = nlp(sent)
            print(f"  Text: {sent}")
            if doc.ents:
                for ent in doc.ents:
                    print(f"    [{ent.label_}] {ent.text!r}  (chars {ent.start_char}-{ent.end_char})")
            else:
                print("    No entities detected")
            print()

    except OSError:
        print("  en_core_web_sm not found. Run: python -m spacy download en_core_web_sm")
        print()

except ImportError:
    print("  spaCy not installed: pip install spacy")
    print("  Then: python -m spacy download en_core_web_sm")
    print()
    print("  Expected output on test sentences:")
    simulated_spacy = [
        [("Apple", "ORG"), ("Tim Cook", "PERSON"), ("Nvidia", "ORG"), ("Microsoft", "ORG")],
        [("Elon Musk", "PERSON"), ("Tesla", "ORG"), ("London", "GPE"), ("Paris", "GPE")],
        [("Google", "ORG"), ("DeepMind", "ORG"), ("$500 million", "MONEY")],
    ]
    for sent, ents in zip(test_sents[:3], simulated_spacy):
        print(f"  Text: {sent}")
        for ent_text, label in ents:
            print(f"    [{label}] {ent_text!r}")
        print()


# ======================================================================
# SECTION 5: NER Evaluation — Span-Level F1
# ======================================================================
print("=" * 70)
print("SECTION 5: NER EVALUATION — SPAN-LEVEL F1 (seqeval)")
print("=" * 70)
print()
print("Important: NER is evaluated at SPAN LEVEL, not token level!")
print()
print("  Why not token-level accuracy?")
print("  → If ground truth is [B-PER, I-PER, I-PER] and we predict [B-PER, B-PER, O]")
print("    Token accuracy = 1/3 = 33%")
print("    But we got 'Steve' right and 'Jobs Smith' wrong — span-level = 0/1 = 0%")
print()
print("  Span-level metrics (seqeval standard):")
print("  • TRUE POSITIVE:  predicted span matches gold span exactly (text AND label)")
print("  • FALSE POSITIVE: predicted span not in gold (hallucinated entity)")
print("  • FALSE NEGATIVE: gold span not predicted (missed entity)")
print()
print("  Precision = TP / (TP + FP)")
print("  Recall    = TP / (TP + FN)")
print("  F1        = 2 × Precision × Recall / (Precision + Recall)")
print()


def extract_spans(bio_tags):
    """
    Extract (label, start_idx, end_idx) spans from BIO sequence.
    This is how seqeval computes span-level metrics.
    """
    spans   = []
    current = None

    for i, tag in enumerate(bio_tags):
        if tag.startswith("B-"):
            if current:
                spans.append(current)
            current = [tag[2:], i, i]  # [label, start, end]
        elif tag.startswith("I-") and current and tag[2:] == current[0]:
            current[2] = i  # extend current span
        else:
            if current:
                spans.append(current)
            current = None

    if current:
        spans.append(current)

    return [tuple(s) for s in spans]


def span_f1(gold_bio, pred_bio):
    """Compute span-level precision, recall, F1."""
    gold_spans = set(map(tuple, extract_spans(gold_bio)))
    pred_spans = set(map(tuple, extract_spans(pred_bio)))

    tp = len(gold_spans & pred_spans)
    fp = len(pred_spans - gold_spans)
    fn = len(gold_spans - pred_spans)

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# Simulated NER predictions
gold_examples = [
    ["B-PER", "I-PER", "O", "B-ORG", "O", "O", "B-LOC"],    # Steve Jobs founded Apple O O California
    ["B-ORG", "O", "B-PER", "I-PER", "O", "O", "B-LOC"],    # Google O Tim Cook O O O London
]
pred_perfect  = gold_examples
pred_partial  = [
    ["B-PER", "O",    "O", "B-ORG", "O", "O", "B-LOC"],      # missed 'Jobs'
    ["B-ORG", "O", "B-PER", "O",    "O", "O", "O"],           # missed 'Cook' and 'London'
]
pred_fp       = [
    ["B-PER", "I-PER","B-ORG","B-ORG","O","O","B-LOC"],       # extra FP
    ["B-ORG", "O",    "B-PER","I-PER","O","O","B-LOC"],
]

scenarios = [
    ("Perfect predictions", gold_examples, pred_perfect),
    ("Partial matches",     gold_examples, pred_partial),
    ("False positives",     gold_examples, pred_fp),
]

print("  Span-F1 evaluation scenarios:")
for name, golds, preds in scenarios:
    all_gold = [tag for seq in golds for tag in seq]
    all_pred = [tag for seq in preds for tag in seq]
    metrics  = span_f1(all_gold, all_pred)
    print(f"  {name}:")
    print(f"    Precision: {metrics['precision']:.3f}")
    print(f"    Recall:    {metrics['recall']:.3f}")
    print(f"    F1:        {metrics['f1']:.3f}")
    print(f"    TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print()


# ======================================================================
# SECTION 6: VISUALIZATIONS
# ======================================================================
print("=" * 70)
print("SECTION 6: VISUALIZATIONS")
print("=" * 70)
print()


# --- PLOT 1: BIO tagging visualization ---
print("Generating: BIO tagging visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Named Entity Recognition: BIO Tagging and Entity Types",
             fontsize=14, fontweight="bold")

# BIO sequence visualization
ax = axes[0]
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("BIO Tagging: Span Extraction", fontsize=12, fontweight="bold")

example_tokens = ["Steve", "Jobs", "founded", "Apple", "Inc.", "in", "California"]
example_tags   = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O", "B-LOC"]
tag_colors     = {
    "B-PER": "#3498DB", "I-PER": "#85C1E9",
    "B-ORG": "#E74C3C", "I-ORG": "#F1948A",
    "B-LOC": "#2ECC71", "I-LOC": "#82E0AA",
    "O":     "#BDC3C7",
}

n_toks = len(example_tokens)
xs     = np.linspace(0.08, 0.92, n_toks)
bw3, bh3 = 0.10, 0.12

for i, (tok, tag) in enumerate(zip(example_tokens, example_tags)):
    color = tag_colors.get(tag, "#BDC3C7")
    rect  = mpatches.FancyBboxPatch((xs[i] - bw3/2, 0.60), bw3, bh3,
                                     boxstyle="round,pad=0.02",
                                     facecolor=color, edgecolor="white", linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(xs[i], 0.66, tok, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white" if tag != "O" else "black")
    ax.text(xs[i], 0.53, tag, ha="center", va="center",
            fontsize=7.5, color=color if tag != "O" else "#555")

# Draw entity brackets
entity_spans = [
    (0, 1, "PER", "#3498DB"),
    (3, 4, "ORG", "#E74C3C"),
    (6, 6, "LOC", "#2ECC71"),
]
for start, end, label, color in entity_spans:
    x_start = xs[start] - bw3/2
    x_end   = xs[end] + bw3/2
    width   = x_end - x_start
    rect2   = mpatches.FancyBboxPatch((x_start, 0.38), width, 0.10,
                                       boxstyle="round,pad=0.01",
                                       facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
    ax.add_patch(rect2)
    mid = (x_start + x_end) / 2
    ax.text(mid, 0.43, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=color)

# Legend
legend_items = [
    mpatches.Patch(color=tag_colors["B-PER"], label="B- (Begin entity)"),
    mpatches.Patch(color=tag_colors["I-PER"], label="I- (Inside entity)"),
    mpatches.Patch(color=tag_colors["O"],     label="O (No entity)"),
]
ax.legend(handles=legend_items, loc="lower center", fontsize=9, ncol=3,
          bbox_to_anchor=(0.5, 0.02))
ax.text(0.5, 0.28, "→ Spans extracted: (Steve Jobs, PER), (Apple Inc., ORG), (California, LOC)",
        ha="center", fontsize=9, style="italic", color="#333",
        bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))

# Entity type distribution (simulated news corpus)
entity_type_counts = {
    "PER":  320, "ORG":  285, "LOC":  210, "MISC": 95, "DATE": 140, "MONEY": 75
}
colors_ent = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12", "#1ABC9C"]

types   = list(entity_type_counts.keys())
counts  = list(entity_type_counts.values())
wedges, texts, autotexts = axes[1].pie(
    counts, labels=types, colors=colors_ent, autopct="%1.1f%%",
    startangle=90, pctdistance=0.75
)
for at in autotexts:
    at.set_fontsize(9)
axes[1].set_title("Typical Entity Distribution in News Corpora\n(CoNLL-2003 style)",
                   fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "bio_tagging.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: bio_tagging.png")


# --- PLOT 2: Feature engineering + NER pipeline ---
print("Generating: NER feature engineering visualization...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("NER Feature Engineering and Pipeline Architecture",
             fontsize=14, fontweight="bold")

# Feature categories and their importance (simulated CRF weights)
feature_categories = [
    "Token shape\n(upper, digit, dash)",
    "Prefix/Suffix\n(3-char n-grams)",
    "Context window\n(prev/next 2 words)",
    "Gazetteer match\n(known names)",
    "POS-like\n(follows 'Mr.', 'in')",
    "Word itself",
]
feature_importance = [0.78, 0.65, 0.72, 0.88, 0.61, 0.55]
colors_fi = plt.cm.RdYlGn([v for v in feature_importance])

bars_fi = axes[0].barh(feature_categories, feature_importance,
                        color=colors_fi, edgecolor="white", linewidth=1.5)
axes[0].set_xlim(0, 1.1)
axes[0].set_xlabel("Relative Feature Importance (simulated CRF weights)")
axes[0].set_title("CRF NER: Feature Importance\n(Typical relative importance)",
                  fontsize=11, fontweight="bold")
axes[0].grid(axis="x", alpha=0.3)
for bar, val in zip(bars_fi, feature_importance):
    axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=10, fontweight="bold")

# NER pipeline comparison
approaches_ner = ["Regex +\nGazetteer", "CRF with\nFeatures", "BiLSTM-CRF\n(DL)", "BERT-NER\n(transformer)"]
per_f1  = [0.60, 0.78, 0.88, 0.93]
org_f1  = [0.55, 0.75, 0.85, 0.91]
loc_f1  = [0.65, 0.82, 0.90, 0.94]

x_n  = np.arange(len(approaches_ner))
w_n  = 0.25
axes[1].bar(x_n - w_n, per_f1, w_n, label="PER", color="#3498DB", alpha=0.85, edgecolor="white")
axes[1].bar(x_n,       org_f1, w_n, label="ORG", color="#E74C3C", alpha=0.85, edgecolor="white")
axes[1].bar(x_n + w_n, loc_f1, w_n, label="LOC", color="#2ECC71", alpha=0.85, edgecolor="white")
axes[1].set_xticks(x_n)
axes[1].set_xticklabels(approaches_ner, fontsize=9)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel("Entity-Level F1 Score")
axes[1].set_title("NER Approach Comparison\n(typical CoNLL-2003 F1 scores)",
                  fontsize=11, fontweight="bold")
axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "features_and_comparison.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: features_and_comparison.png")


# --- PLOT 3: Predictions visualization ---
print("Generating: Entity predictions visualization...")

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
ax.set_title("NER Predictions: Rule-Based vs Expected (spaCy/BERT)",
             fontsize=14, fontweight="bold")

label_colors = {
    "PER": "#3498DB", "ORG": "#E74C3C", "LOC": "#2ECC71",
    "MISC": "#9B59B6", "MONEY": "#F39C12", "DATE": "#1ABC9C", "PERSON": "#3498DB", "GPE": "#2ECC71"
}

display_sents = test_sents[:4]
y_positions   = [0.82, 0.62, 0.42, 0.22]

for sent, y_pos in zip(display_sents, y_positions):
    entities_rb = ner.predict(sent)

    # Display sentence
    ax.text(0.02, y_pos + 0.06, f"Text: {sent}", fontsize=9, color="#333",
            transform=ax.transAxes, clip_on=False)

    # Rule-based entities
    if entities_rb:
        ax.text(0.02, y_pos + 0.02, "Rules:", fontsize=8, color="#555",
                transform=ax.transAxes)
        x_ent = 0.10
        for ent_text, label, _, _ in entities_rb:
            color = label_colors.get(label, "#95A5A6")
            ax.text(x_ent, y_pos + 0.02, f"[{label}] {ent_text}",
                    fontsize=8.5, color=color, fontweight="bold",
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.15,
                              edgecolor=color, linewidth=1))
            x_ent += len(f"[{label}] {ent_text}") * 0.012 + 0.02

    ax.axhline(y=y_pos - 0.02, xmin=0.01, xmax=0.99, color="#ddd", linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "entity_predictions.png"),
            dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: entity_predictions.png")


print()
print("=" * 70)
print("NLP ALGORITHM 4: NAMED ENTITY RECOGNITION COMPLETE!")
print("=" * 70)
print()
print("What you built:")
print("  ✓ BIO tagging scheme — representing multi-word spans")
print("  ✓ Rule-based NER: gazetteer lookup + regex patterns")
print("  ✓ CRF feature engineering: shape, prefix/suffix, context window, gazetteer")
print("  ✓ spaCy production NER (en_core_web_sm pretrained model)")
print("  ✓ Span-level F1 evaluation (the correct NER metric)")
print()
print("NER performance benchmarks (CoNLL-2003):")
print("  Regex + Gazetteer: ~55-65% F1  (misses unknown names)")
print("  CRF features:      ~75-83% F1  (fast, interpretable)")
print("  BiLSTM-CRF:        ~85-90% F1  (learns representations)")
print("  BERT-NER:          ~90-93% F1  (state-of-the-art as of 2019)")
print()
print("3 Visualizations saved to: ../visuals/named_entity_recognition/")
print("  1. bio_tagging.png              — BIO span extraction + entity distribution")
print("  2. features_and_comparison.png — CRF feature importance + approach F1 bars")
print("  3. entity_predictions.png      — side-by-side predictions visualization")
print()
print("🎉 All 4 NLP Algorithms Complete!")
print()
print("Next: Project 1 → Movie Review Sentiment Analysis")
