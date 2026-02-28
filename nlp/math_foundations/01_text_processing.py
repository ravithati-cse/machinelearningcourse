"""
📝 NLP — Math Foundation 1: Text Processing
============================================

Learning Objectives:
  1. Understand why text is hard for machines (raw strings → numbers)
  2. Master the text preprocessing pipeline: tokenize → clean → normalize
  3. Implement tokenization from scratch (word, character, subword intuition)
  4. Apply stopword removal, stemming, and lemmatization
  5. Understand vocabulary building and out-of-vocabulary (OOV) handling
  6. Compute basic text statistics: token frequency, sentence length distributions
  7. Compare preprocessing choices and their downstream impact

YouTube Resources:
  ⭐ Krish Naik — NLP Preprocessing https://www.youtube.com/watch?v=6ZVf1jnEKGI
  ⭐ StatQuest — NLP basics https://www.youtube.com/watch?v=viZrOnJclY0
  📚 NLTK book (free) https://www.nltk.org/book/

Time Estimate: 50 min
Difficulty: Beginner
Prerequisites: Basic Python (strings, lists, dicts)
Key Concepts: tokenization, vocabulary, stopwords, stemming, lemmatization, OOV
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from collections import Counter
import os

_VISUALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "visuals", "01_text_processing")
os.makedirs(_VISUALS_DIR, exist_ok=True)

print("=" * 70)
print("📝 NLP MATH FOUNDATION 1: TEXT PROCESSING")
print("=" * 70)
print()
print("The core NLP challenge: machines work with NUMBERS, not words.")
print()
print("  'The cat sat on the mat'  ← what humans read")
print("  [2, 5, 9, 4, 2, 7]       ← what the model needs")
print()
print("Getting from raw text to numbers requires a PIPELINE:")
print()
print("  Raw Text")
print("     ↓  1. Cleaning   (remove HTML, punctuation, lowercase)")
print("     ↓  2. Tokenize   (split into words/characters/subwords)")
print("     ↓  3. Normalize  (stemming / lemmatization)")
print("     ↓  4. Filter     (remove stopwords)")
print("     ↓  5. Vocabulary (assign integer IDs)")
print("     ↓  6. Encode     (text → integer sequence)")
print("  Integer Sequence")
print("     ↓  (next module: TF-IDF / word embeddings)")
print("  Vector Representation  → ML model")
print()


# ======================================================================
# SECTION 1: Why Raw Text Is Hard
# ======================================================================
print("=" * 70)
print("SECTION 1: WHY RAW TEXT IS HARD")
print("=" * 70)
print()

examples = [
    ("The dog barked loudly.",      "Normal sentence"),
    ("the dog barked loudly",       "Lowercase, no punct"),
    ("THE DOG BARKED LOUDLY!!!",    "Uppercase + extra punct"),
    ("d0g b4rked l0udly",           "Leetspeak noise"),
    ("<p>The dog barked.</p>",      "HTML markup"),
    ("The dog (a Labrador) barked.","Parenthetical info"),
    ("It's the dog's bark!",        "Contractions + possessives"),
    ("barking → bark → barked",    "Morphological variants"),
]

print("  Same concept, wildly different string representations:")
print()
for text, label in examples:
    print(f"  [{label:<28}]  {text!r}")
print()
print("  A model treating these as different 'words' will fail badly.")
print("  Preprocessing standardizes the representation before modelling.")
print()


# ======================================================================
# SECTION 2: Tokenization From Scratch
# ======================================================================
print("=" * 70)
print("SECTION 2: TOKENIZATION FROM SCRATCH")
print("=" * 70)
print()
print("Tokenization = split text into atomic units (tokens).")
print("Tokens can be: characters, words, subwords, sentences, n-grams.")
print()

sample = "It's a dog-eat-dog world, isn't it? Run! Run as fast as you can."
print(f"  Input: {sample!r}")
print()


def word_tokenize_simple(text):
    """Naive whitespace tokenizer."""
    return text.split()


def word_tokenize_regex(text):
    """Regex-based tokenizer that handles contractions and punctuation."""
    # Split on word boundaries, keep apostrophes within contractions
    tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
    return tokens


def char_tokenize(text):
    """Character-level tokenizer."""
    return list(text)


def ngram_tokenize(tokens, n=2):
    """Create n-gram tokens from a word list."""
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


t_whitespace = word_tokenize_simple(sample)
t_regex      = word_tokenize_regex(sample)
t_char       = char_tokenize(sample)[:20]  # first 20 chars
t_bigrams    = ngram_tokenize(t_regex, n=2)

print("  1. Whitespace tokenizer (naive):")
print(f"     {t_whitespace}")
print(f"     Count: {len(t_whitespace)} tokens")
print()
print("  2. Regex tokenizer (smarter):")
print(f"     {t_regex}")
print(f"     Count: {len(t_regex)} tokens")
print()
print("  3. Character tokenizer (first 20 chars):")
print(f"     {t_char}")
print("     Advantages: tiny vocabulary, handles ANY word")
print("     Disadvantages: very long sequences, loses word-level meaning")
print()
print("  4. Bigram tokenizer:")
print(f"     {t_bigrams[:6]}...")
print("     Captures some multi-word meaning ('dog eat', 'eat dog')")
print()
print("  Key question: which granularity?")
print("  • Words:     standard for most tasks (vocab: 50k-200k)")
print("  • Characters: great for noisy text, spelling correction")
print("  • Subwords:  BEST in modern NLP (BPE / WordPiece — used by BERT, GPT)")
print("               'unhappiness' → ['un', '##happ', '##iness'] (vocab: 30k)")
print()


# ======================================================================
# SECTION 3: Text Cleaning
# ======================================================================
print("=" * 70)
print("SECTION 3: TEXT CLEANING")
print("=" * 70)
print()

def clean_text(text, lowercase=True, remove_html=True, remove_punct=True,
               remove_numbers=False, expand_contractions=True):
    """
    Complete text cleaning pipeline.

    Steps (applied in order):
      1. HTML tag removal
      2. Contraction expansion
      3. Lowercase
      4. Number removal (optional)
      5. Punctuation removal
      6. Whitespace normalization
    """
    # Step 1: Remove HTML tags
    if remove_html:
        text = re.sub(r"<[^>]+>", " ", text)

    # Step 2: Expand common contractions
    if expand_contractions:
        contractions = {
            "it's": "it is", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not", "don't": "do not",
            "doesn't": "does not", "didn't": "did not", "won't": "will not",
            "can't": "cannot", "i'm": "i am", "i've": "i have",
            "i'll": "i will", "i'd": "i would", "you're": "you are",
            "you've": "you have", "that's": "that is", "there's": "there is",
            "they're": "they are", "we're": "we are", "we've": "we have",
        }
        text_lower = text.lower()
        for contraction, expansion in contractions.items():
            text_lower = text_lower.replace(contraction, expansion)
        text = text_lower if lowercase else text_lower  # lowercase applied here

    # Step 3: Lowercase (if not already done)
    if lowercase and not expand_contractions:
        text = text.lower()

    # Step 4: Remove numbers
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)

    # Step 5: Remove punctuation (keep spaces)
    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text)

    # Step 6: Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


dirty_texts = [
    "<p>The dog's <b>bark</b> wasn't loud!</p>",
    "I'm SO excited! Can't wait 4 tomorrow :)",
    "Run! He's been running since 5AM... isn't he tired?",
    "The price is $49.99 (50% off). Don't miss it!!!",
]

print("  Cleaning pipeline demonstration:")
print()
for text in dirty_texts:
    cleaned = clean_text(text)
    print(f"  Raw:     {text!r}")
    print(f"  Cleaned: {cleaned!r}")
    print()


# ======================================================================
# SECTION 4: Stopword Removal
# ======================================================================
print("=" * 70)
print("SECTION 4: STOPWORD REMOVAL")
print("=" * 70)
print()
print("Stopwords: common words that appear in nearly every sentence")
print("and carry very little meaning for classification tasks.")
print()

# Common English stopwords (abbreviated from NLTK's full list)
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "up", "about", "into", "through",
    "is", "was", "are", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "dare",
    "it", "its", "this", "that", "these", "those", "i", "me", "my",
    "you", "your", "he", "him", "his", "she", "her", "we", "us",
    "our", "they", "them", "their", "what", "which", "who", "whom",
    "not", "no", "so", "if", "as", "than", "then", "just", "also",
}

print(f"  Using {len(STOPWORDS)} common English stopwords")
print()

sentence = "the movie was really not that great and i did not enjoy it at all"
tokens   = sentence.split()
filtered = [t for t in tokens if t not in STOPWORDS]

print(f"  Original ({len(tokens)} tokens):  {tokens}")
print(f"  Filtered ({len(filtered)} tokens): {filtered}")
print()
print(f"  Removed {len(tokens) - len(filtered)} stopwords — saved {(len(tokens)-len(filtered))/len(tokens)*100:.0f}% of tokens")
print()
print("  When to KEEP stopwords:")
print("  • Sentiment analysis: 'not bad' vs 'bad' — 'not' matters!")
print("  • Language modeling: stopwords ARE the grammar")
print("  • Question answering: 'who', 'what', 'where' are key")
print()
print("  When to REMOVE stopwords:")
print("  • Topic classification: 'science', 'physics', 'experiment' matter")
print("  • TF-IDF retrieval: stopwords would dominate the IDF scores")
print("  • Keyword extraction")
print()


# ======================================================================
# SECTION 5: Stemming and Lemmatization
# ======================================================================
print("=" * 70)
print("SECTION 5: STEMMING VS LEMMATIZATION")
print("=" * 70)
print()
print("Both reduce words to a common base form.")
print()
print("  'running', 'runs', 'ran' → all about 'run'")
print("  'better', 'best'         → both about 'good'")
print("  'dogs', 'dog'            → same concept")
print()

# Stemming: rule-based suffix stripping (fast, approximate)
def porter_stem(word):
    """
    Simplified Porter Stemmer (subset of rules).
    The real Porter Stemmer has 5 phases — this shows the principle.
    """
    word = word.lower()
    suffixes_step1 = [
        ("ational", "ate"), ("tional",  "tion"), ("enci", "ence"),
        ("anci",   "ance"), ("izer",    "ize"),  ("ising", "ise"),
        ("izing",  "ize"),  ("alism",   "al"),   ("ness", ""),
        ("ment",   ""),     ("ments",   ""),     ("ment", ""),
        ("ing",    ""),     ("ings",    ""),     ("ingly", ""),
        ("ies",    "i"),    ("ied",     "i"),    ("ess",  ""),
        ("er",     ""),     ("ers",     ""),     ("ed",   ""),
        ("ation",  "ate"),  ("ations",  "ate"),  ("ness", ""),
        ("ly",     ""),     ("ful",     ""),     ("less", ""),
        ("ness",   ""),     ("ous",     ""),     ("ment", ""),
        ("ment",   ""),
    ]

    for suffix, replacement in suffixes_step1:
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)] + replacement

    # Remove trailing 's' for plurals
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]

    return word


# Lemmatization: dictionary lookup for true base form
LEMMA_DICT = {
    "running": "run",  "runs": "run",    "ran": "run",
    "better":  "good", "best": "good",   "worse": "bad",    "worst": "bad",
    "dogs":    "dog",  "cats": "cat",    "mice": "mouse",   "geese": "goose",
    "having":  "have", "had": "have",    "has": "have",
    "went":    "go",   "going": "go",    "goes": "go",
    "was":     "be",   "were": "be",     "is": "be",        "are": "be",
    "happily": "happy","happiness": "happy",
    "studies": "study","studying": "study","studied": "study",
    "bought":  "buy",  "buying": "buy",  "buys": "buy",
}

def simple_lemmatize(word):
    return LEMMA_DICT.get(word.lower(), word.lower())


test_words = [
    "running", "runs", "ran",
    "dogs", "cats", "mice",
    "studies", "studying", "studied",
    "happily", "happiness",
    "better", "best",
    "generalization", "generalizing",
]

print(f"  {'Word':<18} {'Stemmed':<18} {'Lemmatized':<18}")
print(f"  {'─'*18} {'─'*18} {'─'*18}")
for word in test_words:
    stemmed   = porter_stem(word)
    lemmatized = simple_lemmatize(word)
    print(f"  {word:<18} {stemmed:<18} {lemmatized:<18}")

print()
print("  Key differences:")
print("  • Stemming: fast, rule-based, may produce non-words ('studi', 'happi')")
print("  • Lemmatization: slower, dictionary-based, always real words")
print("  • Use stemming for: search engines, large-scale retrieval")
print("  • Use lemmatization for: NLP tasks requiring real words")
print("  • Modern transformers (BERT/GPT): use NEITHER — subword tokenizer handles it")
print()


# ======================================================================
# SECTION 6: Vocabulary Building
# ======================================================================
print("=" * 70)
print("SECTION 6: VOCABULARY BUILDING")
print("=" * 70)
print()
print("After preprocessing, build a VOCABULARY: {word → integer ID}")
print()
print("  Special tokens to always include:")
print("  <PAD>  (0) — padding to make sequences equal length")
print("  <UNK>  (1) — out-of-vocabulary (OOV) token")
print("  <BOS>  (2) — beginning of sequence (some models)")
print("  <EOS>  (3) — end of sequence (some models)")
print()

# Build vocabulary from a mini corpus
mini_corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "a cat and a dog played in the park",
    "the park had a big log and a small mat",
    "cats and dogs are common pets",
    "my dog loves to run in the park",
]

all_tokens = []
for sentence in mini_corpus:
    cleaned = clean_text(sentence, remove_punct=True)
    tokens  = [t for t in cleaned.split() if t not in STOPWORDS]
    all_tokens.extend(tokens)

token_freq = Counter(all_tokens)
print(f"  Mini corpus: {len(mini_corpus)} sentences")
print(f"  Total tokens (after stopword removal): {len(all_tokens)}")
print(f"  Unique tokens: {len(token_freq)}")
print()
print(f"  Token frequencies:")
for tok, count in token_freq.most_common(15):
    bar = "█" * count
    print(f"    {tok:<12}: {count:2d}  {bar}")
print()


class Vocabulary:
    """Build and use a word → integer vocabulary."""

    SPECIAL = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def __init__(self, min_freq=1, max_vocab=None):
        self.min_freq  = min_freq
        self.max_vocab = max_vocab
        self.word2idx  = {}
        self.idx2word  = {}

    def build(self, token_freq: Counter):
        self.word2idx = dict(self.SPECIAL)

        # Sort by frequency (descending)
        sorted_words = [w for w, c in token_freq.most_common() if c >= self.min_freq]
        if self.max_vocab:
            sorted_words = sorted_words[:self.max_vocab - len(self.SPECIAL)]

        for i, word in enumerate(sorted_words, start=len(self.SPECIAL)):
            self.word2idx[word] = i

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        return self

    def encode(self, tokens):
        return [self.word2idx.get(t, self.SPECIAL["<UNK>"]) for t in tokens]

    def decode(self, ids):
        return [self.idx2word.get(i, "<UNK>") for i in ids]

    def __len__(self):
        return len(self.word2idx)


vocab = Vocabulary(min_freq=1).build(token_freq)

print(f"  Vocabulary size: {len(vocab)}")
print()
print(f"  word2idx (sample):")
for word, idx in list(vocab.word2idx.items())[:10]:
    print(f"    {word!r:<14} → {idx}")
print()

# Encode and decode
test_sentence = "the cat and dog played in the park"
test_tokens   = [t for t in clean_text(test_sentence).split() if t not in STOPWORDS]
encoded       = vocab.encode(test_tokens)
decoded       = vocab.decode(encoded)

print(f"  Encoding: {test_sentence!r}")
print(f"  Tokens:   {test_tokens}")
print(f"  Encoded:  {encoded}")
print(f"  Decoded:  {decoded}")
print()

# OOV demo
oov_sentence = "the hamster played with a ferret near the pond"
oov_tokens   = [t for t in clean_text(oov_sentence).split() if t not in STOPWORDS]
oov_encoded  = vocab.encode(oov_tokens)
print(f"  OOV example: {oov_sentence!r}")
print(f"  Tokens:      {oov_tokens}")
print(f"  Encoded:     {oov_encoded}  (1 = <UNK>)")
print()


# ======================================================================
# SECTION 7: Full Pipeline (Put It All Together)
# ======================================================================
print("=" * 70)
print("SECTION 7: FULL PREPROCESSING PIPELINE")
print("=" * 70)
print()


def preprocess_pipeline(text, vocab=None, remove_stops=True, do_stem=False):
    """
    Full text preprocessing pipeline.
    Returns: cleaned string, tokens, encoded integer sequence
    """
    # Step 1: Clean
    cleaned = clean_text(text)

    # Step 2: Tokenize
    tokens = cleaned.split()

    # Step 3: Stopword removal
    if remove_stops:
        tokens = [t for t in tokens if t not in STOPWORDS]

    # Step 4: Stemming (optional)
    if do_stem:
        tokens = [porter_stem(t) for t in tokens]

    # Step 5: Encode
    encoded = vocab.encode(tokens) if vocab else None

    return cleaned, tokens, encoded


raw_texts = [
    "The dog's bark wasn't loud! It's just a puppy.",
    "<p>Cats and dogs are <b>amazing</b> pets.</p>",
    "Running in the park is great — the cat ran too!",
]

print(f"  {'─'*68}")
for text in raw_texts:
    cleaned, tokens, encoded = preprocess_pipeline(text, vocab=vocab)
    print(f"  Raw:     {text!r}")
    print(f"  Cleaned: {cleaned!r}")
    print(f"  Tokens:  {tokens}")
    print(f"  Encoded: {encoded}")
    print(f"  {'─'*68}")
print()


# ======================================================================
# SECTION 8: Using NLTK (Production Library)
# ======================================================================
print("=" * 70)
print("SECTION 8: NLTK — PRODUCTION TEXT PROCESSING")
print("=" * 70)
print()
print("NLTK (Natural Language Toolkit) is the standard Python NLP library.")
print("It has: tokenizers, stemmers, lemmatizers, POS taggers, and more.")
print()

NLTK_AVAILABLE = False
try:
    import nltk
    # Download needed resources quietly
    for resource in ["punkt", "stopwords", "wordnet", "averaged_perceptron_tagger",
                     "punkt_tab"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass
    NLTK_AVAILABLE = True
    print("  NLTK installed and ready!")
    print()
except ImportError:
    print("  NLTK not installed. Run: pip install nltk")
    print("  Showing code and expected output below.")
    print()

if NLTK_AVAILABLE:
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    test_text = "The dogs aren't running. They've been playing all day. It's wonderful!"

    # Sentence tokenization
    sentences = sent_tokenize(test_text)
    print(f"  Sentence tokenization:")
    for s in sentences:
        print(f"    {s!r}")
    print()

    # Word tokenization
    tokens_nltk = word_tokenize(test_text)
    print(f"  Word tokenization: {tokens_nltk}")
    print()

    # Stopwords
    stop_words_nltk = set(nltk_stopwords.words("english"))
    print(f"  NLTK has {len(stop_words_nltk)} English stopwords")
    filtered_nltk = [t for t in tokens_nltk if t.lower() not in stop_words_nltk
                     and t.isalpha()]
    print(f"  After filtering: {filtered_nltk}")
    print()

    # Stemming
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(t) for t in filtered_nltk]
    print(f"  Porter stemmed: {stemmed}")
    print()

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(t.lower()) for t in filtered_nltk]
    print(f"  WordNet lemmatized: {lemmatized}")
    print()

else:
    print("  NLTK code (run after 'pip install nltk'):")
    print()
    print("    import nltk")
    print("    nltk.download('punkt')")
    print("    nltk.download('stopwords')")
    print("    nltk.download('wordnet')")
    print()
    print("    from nltk.tokenize import word_tokenize, sent_tokenize")
    print("    from nltk.corpus import stopwords")
    print("    from nltk.stem import PorterStemmer, WordNetLemmatizer")
    print()
    print("    text = 'The dogs aren't running.'")
    print("    tokens    = word_tokenize(text)                    # ['The', 'dogs', ...]")
    print("    sentences = sent_tokenize(text)                    # ['The dogs ...']")
    print("    stop_set  = set(stopwords.words('english'))        # 179 words")
    print("    stemmed   = [PorterStemmer().stem(t) for t in tokens]")
    print("    lemmatized= [WordNetLemmatizer().lemmatize(t) for t in tokens]")
    print()


# ======================================================================
# SECTION 9: Visualizations
# ======================================================================
print("=" * 70)
print("SECTION 9: VISUALIZATIONS")
print("=" * 70)
print()

# Larger synthetic corpus for visualization
corpus_large = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Deep neural networks can learn complex patterns from data",
    "Natural language processing enables computers to understand text",
    "The cat sat on the mat and watched the dog",
    "Convolutional neural networks are great for image recognition",
    "Transfer learning allows models to apply knowledge across domains",
    "Recurrent neural networks handle sequential data like text and audio",
    "Attention mechanisms revolutionized natural language processing tasks",
    "Word embeddings capture semantic relationships between words",
    "Tokenization splits text into meaningful units for processing",
    "Stopword removal filters common words with little semantic value",
    "The vocabulary maps words to integers for model consumption",
    "Text classification assigns categories to documents automatically",
    "Sentiment analysis determines positive or negative tone in text",
]

# Process all text
all_words_raw   = []
all_words_clean = []
sentence_lengths_raw   = []
sentence_lengths_clean = []

for sent in corpus_large:
    raw_tokens   = sent.lower().split()
    clean_tokens = [t for t in clean_text(sent).split() if t not in STOPWORDS]
    all_words_raw.extend(raw_tokens)
    all_words_clean.extend(clean_tokens)
    sentence_lengths_raw.append(len(raw_tokens))
    sentence_lengths_clean.append(len(clean_tokens))

freq_raw   = Counter(all_words_raw)
freq_clean = Counter(all_words_clean)


# --- PLOT 1: Preprocessing pipeline flow + token frequency comparison ---
print("Generating: Preprocessing pipeline + frequency comparison...")

fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
ax_pipe  = fig.add_subplot(gs[0, :])    # top row: full width pipeline
ax_freq1 = fig.add_subplot(gs[1, 0])
ax_freq2 = fig.add_subplot(gs[1, 1])

fig.suptitle("Text Processing: Pipeline and Token Analysis",
             fontsize=15, fontweight="bold")

# Pipeline diagram
ax_pipe.set_xlim(0, 1)
ax_pipe.set_ylim(0, 1)
ax_pipe.axis("off")
ax_pipe.set_title("Full Text Preprocessing Pipeline", fontsize=12, fontweight="bold")

pipeline_steps = [
    ("Raw\nText",       0.07, "#E74C3C"),
    ("HTML\nClean",     0.19, "#E67E22"),
    ("Expand\nContrac.", 0.31, "#F1C40F"),
    ("Lower\ncase",     0.43, "#2ECC71"),
    ("Tokenize",        0.55, "#3498DB"),
    ("Remove\nStopwords", 0.67, "#9B59B6"),
    ("Stem /\nLemmatize", 0.79, "#1ABC9C"),
    ("Integer\nIDs",    0.91, "#E74C3C"),
]

bw, bh = 0.09, 0.44
for text, xc, color in pipeline_steps:
    rect = mpatches.FancyBboxPatch((xc - bw/2, 0.28), bw, bh,
                                   boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor="white",
                                   linewidth=2, alpha=0.85)
    ax_pipe.add_patch(rect)
    ax_pipe.text(xc, 0.50, text, ha="center", va="center",
                 fontsize=9, fontweight="bold", color="white")

for i in range(len(pipeline_steps) - 1):
    x0 = pipeline_steps[i][1]   + bw / 2 + 0.005
    x1 = pipeline_steps[i+1][1] - bw / 2 - 0.005
    ax_pipe.annotate("", xy=(x1, 0.50), xytext=(x0, 0.50),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=2),
                     xycoords="axes fraction", textcoords="axes fraction")

# Bottom labels
examples_labels = [
    "\"<p>It's\\ngreat!</p>\"",
    "\"It's\\ngreat!\"",
    "\"it is\\ngreat!\"",
    "\"it is\\ngreat\"",
    "['it','is','great']",
    "['great']",
    "['great']",
    "[7]",
]
for (_, xc, _), lbl in zip(pipeline_steps, examples_labels):
    ax_pipe.text(xc, 0.15, lbl, ha="center", va="top",
                 fontsize=7, color="#444", style="italic")

# Token frequency before cleaning
top_raw   = freq_raw.most_common(12)
top_clean = freq_clean.most_common(12)

words_r, counts_r = zip(*top_raw)
words_c, counts_c = zip(*top_clean)

ax_freq1.barh(list(reversed(words_r)), list(reversed(counts_r)),
              color="#E74C3C", alpha=0.8, edgecolor="white")
ax_freq1.set_title("Top Tokens — BEFORE Processing", fontsize=11, fontweight="bold")
ax_freq1.set_xlabel("Frequency")
ax_freq1.grid(axis="x", alpha=0.3)

ax_freq2.barh(list(reversed(words_c)), list(reversed(counts_c)),
              color="#2ECC71", alpha=0.8, edgecolor="white")
ax_freq2.set_title("Top Tokens — AFTER Processing\n(lowercase, stopwords removed)", fontsize=11, fontweight="bold")
ax_freq2.set_xlabel("Frequency")
ax_freq2.grid(axis="x", alpha=0.3)

plt.savefig(os.path.join(_VISUALS_DIR, "pipeline_and_frequency.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: pipeline_and_frequency.png")


# --- PLOT 2: Sentence length distributions + vocabulary coverage ---
print("Generating: Sentence length distribution and vocabulary coverage...")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Text Statistics: Length Distributions and Vocabulary Coverage",
             fontsize=14, fontweight="bold")

# Sentence lengths before/after
axes[0].hist(sentence_lengths_raw,   bins=range(3, 18), alpha=0.7, color="#E74C3C",
             label="Before processing", edgecolor="white")
axes[0].hist(sentence_lengths_clean, bins=range(3, 18), alpha=0.7, color="#2ECC71",
             label="After processing",  edgecolor="white")
axes[0].set_xlabel("Tokens per Sentence"); axes[0].set_ylabel("Count")
axes[0].set_title("Sentence Length Distribution")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].axvline(np.mean(sentence_lengths_raw),   color="#E74C3C", linestyle="--", linewidth=2,
                label=f"Mean (raw): {np.mean(sentence_lengths_raw):.1f}")
axes[0].axvline(np.mean(sentence_lengths_clean), color="#2ECC71", linestyle="--", linewidth=2,
                label=f"Mean (clean): {np.mean(sentence_lengths_clean):.1f}")

# Vocabulary growth (type-token ratio)
cumulative_types = []
seen = set()
for token in all_words_clean:
    seen.add(token)
    cumulative_types.append(len(seen))

axes[1].plot(range(1, len(cumulative_types)+1), cumulative_types,
             color="steelblue", linewidth=2)
axes[1].fill_between(range(1, len(cumulative_types)+1), cumulative_types,
                     alpha=0.2, color="steelblue")
axes[1].set_xlabel("Tokens Seen"); axes[1].set_ylabel("Unique Vocabulary Size")
axes[1].set_title("Vocabulary Growth (Type-Token Curve)")
axes[1].grid(True, alpha=0.3)
axes[1].text(len(cumulative_types) * 0.5, cumulative_types[-1] * 0.5,
             f"Final vocab: {cumulative_types[-1]} words",
             fontsize=10, color="steelblue", fontweight="bold")

# Zipf's law demonstration
sorted_counts = sorted(freq_raw.values(), reverse=True)
ranks = range(1, len(sorted_counts) + 1)
axes[2].loglog(ranks, sorted_counts, "o-", color="#9B59B6",
               markersize=5, linewidth=2, label="Observed")
# Theoretical Zipf
zipf_curve = [sorted_counts[0] / r for r in ranks]
axes[2].loglog(ranks, zipf_curve, "--", color="darkorange",
               linewidth=2, label="Zipf's Law (1/rank)")
axes[2].set_xlabel("Word Rank (log scale)"); axes[2].set_ylabel("Frequency (log scale)")
axes[2].set_title("Zipf's Law — Word Frequency Distribution")
axes[2].legend(); axes[2].grid(True, alpha=0.3)
axes[2].text(1.5, sorted_counts[0] * 0.4,
             "Top words are\nexponentially\nmore common!",
             fontsize=9, color="#9B59B6")

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "text_statistics.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: text_statistics.png")


# --- PLOT 3: Tokenization comparison diagram ---
print("Generating: Tokenization method comparison...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Tokenization Strategies Compared",
             fontsize=14, fontweight="bold")

example_word = "unhappiness"
colors_tok = ["#3498DB", "#E74C3C", "#2ECC71"]

# Word-level
word_tokens = ["un", "happiness"]  # split at morpheme (showing subword intuition)
word_tokens_actual = [example_word]

# Character-level
char_tokens = list(example_word)

# Subword (simulating BPE-like split)
subword_tokens = ["un", "##happ", "##iness"]  # BERT-style

tokenizations = [
    ("Word-level\n(keeps whole word)",     word_tokens_actual, "#3498DB",
     "Vocab: 100k-500k words\n+ many OOV words"),
    ("Character-level\n(every character)", char_tokens, "#E74C3C",
     "Vocab: ~100 chars\nLong sequences (×4-6)"),
    ("Subword / BPE\n(BERT, GPT style)",   subword_tokens, "#2ECC71",
     "Vocab: ~30k pieces\nNo OOV — best of both!"),
]

for ax, (title, tokens, color, note) in zip(axes, tokenizations):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Show input word
    ax.text(0.5, 0.88, f"Input: \"{example_word}\"", ha="center",
            fontsize=11, fontweight="bold", color="#333",
            bbox=dict(boxstyle="round", facecolor="#f8f9fa", edgecolor="#ccc"))
    ax.text(0.5, 0.76, "↓", ha="center", fontsize=16, color="#555")

    # Show tokens as colored boxes
    n = len(tokens)
    xs = np.linspace(0.1, 0.9, n)
    bwidth = min(0.18, 0.72 / (n + 1))

    for x, tok in zip(xs, tokens):
        rect = mpatches.FancyBboxPatch((x - bwidth/2, 0.55), bwidth, 0.13,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, alpha=0.85, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x, 0.615, tok, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    ax.text(0.5, 0.42, f"{n} token{'s' if n > 1 else ''}", ha="center",
            fontsize=11, color=color, fontweight="bold")
    ax.text(0.5, 0.20, note, ha="center", va="center",
            fontsize=9, color="#555", style="italic",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(_VISUALS_DIR, "tokenization_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()
print("   Saved: tokenization_comparison.png")


print()
print("=" * 70)
print("NLP MATH FOUNDATION 1: TEXT PROCESSING COMPLETE!")
print("=" * 70)
print()
print("What you learned:")
print("  ✓ Why text needs preprocessing: raw strings → numbers")
print("  ✓ Full pipeline: clean → tokenize → normalize → filter → encode")
print("  ✓ Tokenization: word, character, subword (BPE/WordPiece)")
print("  ✓ Stopword removal: when to keep vs remove common words")
print("  ✓ Stemming (fast, approximate) vs lemmatization (accurate)")
print("  ✓ Vocabulary building and OOV handling with <UNK>")
print("  ✓ Zipf's law: word frequency follows power-law distribution")
print()
print("3 Visualizations saved to: ../visuals/01_text_processing/")
print("  1. pipeline_and_frequency.png    — pipeline flow + token frequency bars")
print("  2. text_statistics.png           — length distributions + Zipf's law")
print("  3. tokenization_comparison.png   — word vs char vs subword side-by-side")
print()
print("Next: Foundation 2 → Bag of Words & TF-IDF")
