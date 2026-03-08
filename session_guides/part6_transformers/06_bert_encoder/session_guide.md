# MLForBeginners — Instructor Guide
## Part 6 · Module 06: BERT Encoder
### Two-Session Teaching Script

> **Prerequisites:** Modules 01–05 complete. They have built a Transformer
> encoder from scratch in NumPy and understand every layer. They know
> multi-head attention, positional encoding, Pre-LN blocks, and mean pooling.
> **Payoff today:** They will use a real, pretrained BERT model for the first
> time. They will extract contextual embeddings, compare sentence similarities,
> and train a classifier on top of frozen BERT — in under 30 lines of code.

---

# SESSION 1 (~90 min)
## "BERT's pretraining — bidirectionality, MLM, and WordPiece tokenization"

## Before They Arrive
- Terminal open in `transformers/algorithms/`
- `pip install transformers` confirmed working
- Whiteboard ready — draw two arrows, one pointing right (→), one bidirectional (↔)
- Print or open the original BERT paper abstract: arxiv.org/abs/1810.04805

---

## OPENING (10 min)

> *"Every language model we've discussed reads text one direction at a time.
> RNNs go left to right. GPT goes left to right.*
>
> *But humans don't read that way. When you see the word 'bank' in a sentence,
> you automatically look BOTH ways — what came before, what comes after —
> to understand which meaning is correct.*
>
> *BERT was Google's 2019 answer to that problem.*
> *Bidirectional Encoder Representations from Transformers.*
> *Every token sees every other token simultaneously.*
>
> *The result was a model that destroyed the state of the art on 11 NLP
> benchmarks the day it was published. Fine-tuned BERT is still widely
> deployed in production search, question answering, and classification.*
>
> *You're going to use it today."*

Draw on board:
```
GPT (left-to-right):
  "The  bank  can  guarantee  deposits  ..."
    →     →    →      →           →

BERT (bidirectional):
  "The  bank  can  guarantee  deposits  ..."
    ↔     ↔    ↔      ↔           ↔
  (every token attends to ALL others simultaneously)
```

> *"For the word 'bank' specifically:*
> *GPT only uses 'The' as context — misses 'guarantee deposits' which clarifies meaning.*
> *BERT uses ALL tokens. It sees 'guarantee deposits' and encodes 'bank' as a financial institution."*

---

## SECTION 1: BERT's Two Pretraining Tasks (25 min)

> *"BERT learned from BooksCorpus + English Wikipedia — about 3 billion words.
> It never saw a labeled example. It learned from two self-supervised tasks."*

**Task 1: Masked Language Modeling (MLM)**

Write on board:
```
MLM OBJECTIVE

Input:   "The cat sat on the [MASK]."
Target:  "mat"

15% of tokens are randomly replaced:
  • 80%: replaced with [MASK]
  • 10%: replaced with a random token
  • 10%: kept unchanged (but still predicted)

BERT must predict the masked token from context
using information from BOTH directions.
```

> *"The 80/10/10 split is intentional. If BERT only ever saw [MASK]
> during pretraining but never during fine-tuning, it would behave
> differently on real text. The 10% random and 10% unchanged keep BERT
> honest — it has to represent every token correctly, not just [MASK]."*

**Ask the room:** *"Why does MLM force bidirectionality? Could you do MLM
with a left-to-right language model?"*

Pause. Let them think.

> *"A left-to-right model predicts token 5 using tokens 1–4. If token 5 is
> masked, it can only look left. That's fine but weak.*
>
> *MLM specifically masks positions throughout the sequence and asks BERT
> to fill them in. To predict a masked word in the middle of a sentence,
> BERT MUST look right as well as left. That forces bidirectionality."*

**Task 2: Next Sentence Prediction (NSP)**

```
NSP OBJECTIVE

Input: [CLS] Sentence A [SEP] Sentence B [SEP]

Label IsNext:
  A: "The man went to the store."
  B: "He bought a gallon of milk."
  → Sentences are consecutive. Answer: IsNext.

Label NotNext:
  A: "The man went to the store."
  B: "Penguins live in Antarctica."
  → Sentences are random. Answer: NotNext.

BERT predicts using the [CLS] token representation.
Teaches understanding of sentence relationships.
```

> *"The [CLS] token is special. It's prepended to every input.
> Because it has no inherent meaning, attention training pushes it
> to gather global information about the whole sequence.*
>
> *For NSP, the [CLS] vector after all encoder layers represents
> the whole sentence pair. A small linear classifier on top predicts
> IsNext vs NotNext.*
>
> *We'll use this [CLS] trick for our own classifier in Session 2."*

---

## SECTION 2: WordPiece Subword Tokenization (20 min)

> *"Before BERT can process text, it must tokenize it.
> But BERT doesn't tokenize by word. It uses WordPiece.*
>
> *The problem with word tokenization: vocabulary explodes.*
> *English has hundreds of thousands of word forms.*
> *'run', 'running', 'runner', 'runs', 'ran' — five separate tokens.*
> *Rare words or typos become unknown tokens: [UNK]."*

Draw on board:

```
WordPiece TOKENIZATION

"unbelievable"  →  ["un", "##believ", "##able"]
"embeddings"    →  ["em", "##bed", "##dings"]
"transformer"   →  ["transform", "##er"]
"hello"         →  ["hello"]

## prefix means: this piece continues a word (not a new word start)

Benefits:
1. Fixed vocabulary (30,522 for BERT-base)
2. Handles any word — just split into known sub-pieces
3. Rare words share sub-pieces with common words
   ("cardiovascular" shares "##ar" with "particular")
4. No unknown tokens for most inputs
```

**Live Demo — see WordPiece in action:**
```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = [
    "The cat sat on the mat.",
    "Transformers use self-attention.",
    "unbelievable antidisestablishmentarianism"
]

for text in texts:
    tokens = tokenizer.tokenize(text)
    ids    = tokenizer.encode(text)
    print(f"\nText:   {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs:    {ids}")
```

> *"See [CLS]=101 at the start and [SEP]=102 at the end — always added.*
>
> *'##' pieces continue the previous word. The tokenizer handles
> capitalization ('bert-base-uncased' lowercases everything).*
>
> *A real sentence like 'The cat sat on the mat.' becomes maybe 9-10 tokens.
> A long article becomes up to 512 tokens — BERT's maximum.*
> *Longer text needs chunking or a long-context model."*

**Ask the room:** *"What would happen to the token ID list for a word
BERT has never seen, like a made-up word 'gronderflib'?"*

> *"WordPiece would split it into sub-pieces it does know.
> It might become ['gron', '##der', '##f', '##lib'] or similar.
> There's essentially no UNK for in-language text —
> at worst each character becomes its own piece."*

---

## CLOSING SESSION 1 (5 min)

```
TODAY:
  BERT = Bidirectional Encoder from Transformers (Google, 2019)

  Pretraining:
    MLM  → predict 15% masked tokens from both directions
    NSP  → predict if sentence B follows sentence A

  Special tokens: [CLS] [SEP] [MASK] [PAD]

  WordPiece: subword tokenization → 30,522 vocab, no [UNK]

  Architecture: 12 layers, 768 d_model, 12 heads, 110M parameters
```

**Homework:** Tokenize three sentences of your own. Look for any
##-prefixed tokens. What determines whether a word gets split or not?

---

# SESSION 2 (~90 min)
## "BERT as a feature extractor — embeddings, similarity, and a classifier"

## OPENING (10 min)

> *"Last session we learned what BERT knows and how it was trained.*
>
> *Today we use it. We're going to treat BERT as a frozen feature extractor:
> load the pretrained model, pass text through it, and pull out the
> contextual embeddings it produces.*
>
> *Then we'll see something remarkable: two sentences about the same topic
> will have similar embeddings even if they use completely different words.
> That's BERT's deep understanding of meaning.*
>
> *And we'll train a lightweight classifier on top of those embeddings
> without updating a single BERT weight."*

---

## SECTION 1: Loading BERT and Extracting [CLS] Embeddings (25 min)

> *"The [CLS] token sits at position 0 of the output.*
> *After passing through 12 encoder layers, it has attended to every
> other token in the sentence. Its 768-dimensional vector summarizes
> the whole sequence.*
>
> *That's our sentence embedding."*

```python
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()  # no dropout, no gradient updates

def get_bert_embedding(text):
    """Return the [CLS] embedding for a sentence."""
    encoded = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        output = model(**encoded)
    # output.last_hidden_state: [batch, seq, 768]
    cls_embedding = output.last_hidden_state[:, 0, :]   # position 0 = [CLS]
    return cls_embedding.numpy().squeeze()  # [768]

emb = get_bert_embedding("The cat sat on the mat.")
print(f"Embedding shape: {emb.shape}")   # (768,)
print(f"Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")
```

> *"768 numbers. That's BERT's understanding of 'The cat sat on the mat.'
> Every number was learned from 3 billion words of text.*
>
> *The embedding is dense — most values are non-zero.
> Compare to TF-IDF which was sparse — mostly zeros.*
> Dense vectors encode meaning. Sparse vectors encode word counts."*

---

## SECTION 2: Cosine Similarity — Measuring Semantic Distance (20 min)

Write on board:
```
COSINE SIMILARITY

cos(A, B) = (A · B) / (||A|| × ||B||)

Range: -1 to +1
  1.0  = identical direction (same meaning)
  0.0  = orthogonal (unrelated)
 -1.0  = opposite direction

Why NOT Euclidean distance for embeddings?
  Two sentences can have the same MEANING but different LENGTHS.
  Longer text → larger magnitude vector.
  Cosine ignores magnitude → compares direction (meaning) only.
```

**Live Demo — semantic similarity:**
```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sentence_pairs = [
    ("The bank can guarantee deposits will cover costs.",
     "I sat on the river bank and watched the water."),
    ("Machine learning requires lots of data.",
     "AI systems need large datasets to learn effectively."),
    ("The stock market crashed yesterday.",
     "Equity markets experienced significant decline on Thursday."),
    ("The cat sat on the mat.",
     "Quantum physics describes the behavior of subatomic particles."),
]

for s1, s2 in sentence_pairs:
    e1 = get_bert_embedding(s1)
    e2 = get_bert_embedding(s2)
    sim = cosine_similarity(e1, e2)
    print(f"\nSim: {sim:.3f}")
    print(f"  A: {s1[:50]}...")
    print(f"  B: {s2[:50]}...")
```

> *"Look at the numbers. The machine learning / AI pair should be very high —
> they mean the same thing in different words.*
>
> *The 'bank' financial vs 'bank' river pair should be noticeably lower —
> same word, completely different context.*
>
> *That's BERT's bidirectional context at work. 'Bank' near 'deposits' and
> 'guarantee' becomes a financial embedding.*
> 'Bank' near 'river' and 'water' becomes a geography embedding.*
>
> *No other model we've built this course could distinguish those."*

**Ask the room:** *"TF-IDF also represents sentences as vectors.
What would TF-IDF say about 'I need a doctor' vs 'I require medical assistance'?
What would BERT say?"*

> *"TF-IDF similarity: very low. Zero shared important words.*
> *BERT similarity: very high. The semantics are identical.*
> *That's the gap between word-counting and meaning-understanding."*

---

## SECTION 3: BERT as Feature Extractor — Lightweight Classifier (20 min)

> *"Here's the power of transfer learning: BERT does all the hard work.
> We don't update its 110 million parameters at all.*
> We just extract its [CLS] embeddings and train a simple classifier on top.*
>
> *This is 'frozen BERT' or 'feature extraction mode'.*
> *Fine-tuning (which we'll see in the project) updates BERT's weights.
> Feature extraction keeps them frozen. Much cheaper, often almost as good."*

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

# Small labeled dataset
texts = [
    "The stock market rose sharply on strong earnings reports.",   # Finance
    "Scientists discover new species of deep-sea fish.",           # Science
    "The team won the championship in overtime.",                   # Sports
    "AI model achieves human-level performance on language tasks.", # Technology
    "Inflation data shows consumer prices rising faster than expected.", # Finance
    "Astronomers detect radio signal from distant galaxy.",         # Science
    "Quarterback throws record-setting five touchdowns.",           # Sports
    "New chip architecture doubles neural network inference speed.", # Technology
]
labels = [0, 1, 2, 3, 0, 1, 2, 3]  # Finance=0, Science=1, Sports=2, Tech=3

# Extract BERT features
print("Extracting BERT embeddings...")
X = np.array([get_bert_embedding(t) for t in texts])  # [8, 768]
y = np.array(labels)

# Train/test split — tiny dataset, demonstration only
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds) * 100:.0f}%")
```

> *"A logistic regression on top of BERT. We haven't updated a single
> BERT parameter. Just extracted 768 numbers per sentence and trained a
> linear classifier on those numbers.*
>
> *Compare to Part 5 where we needed TF-IDF, stemming, stop word removal,
> and a custom pipeline. Here it's: tokenize → BERT → [CLS] → classify.*
>
> *The full project (Module 08) does this with 50 labeled examples per class
> and reaches high accuracy on 5-class news classification."*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
BERT WORKFLOW:
  Text  →  BertTokenizer  →  token IDs + attention mask
        →  BertModel      →  [batch, seq, 768] contextual embeddings
        →  [:, 0, :]      →  [CLS] vector (global representation)

FEATURE EXTRACTION vs FINE-TUNING:
  Feature extraction:  BERT frozen, classifier trained
                       Fast, cheap, good baseline
  Fine-tuning:         BERT + classifier trained together
                       Better accuracy, needs more data/compute

BERT ARCHITECTURE (bert-base-uncased):
  12 layers | 768 d_model | 12 heads | 110M parameters
  Vocabulary: 30,522 WordPiece tokens
  Max length: 512 tokens
```

**Homework:** Use `get_bert_embedding` on 5 sentences from different topics.
Compute all pairwise cosine similarities. Do the within-topic pairs have
higher similarity than across-topic pairs?

---

## INSTRUCTOR TIPS

**"Why do we use [CLS] and not mean pool like we did in Module 05?"**
> *"Both work. BERT was specifically trained with [CLS] gathering global
> information through NSP. The [CLS] token has 12 layers of training
> pressure to summarize the sentence.*
>
> *Mean pooling works well too, especially for semantic similarity tasks.
> SentenceBERT (a later paper) actually mean-pools for better sentence vectors.*
> *For classification, [CLS] is the standard BERT convention."*

**"The model takes forever to download"**
> *"Bert-base-uncased is about 440MB. It downloads once and caches.*
> *If internet is slow, we can mock the embeddings with random vectors
> for the cosine similarity demo and return to real BERT later.*
> *The point is the API pattern: tokenize → forward → extract position 0."*

**"What's the difference between bert-base and bert-large?"**
> *"bert-base: 12 layers, 768 d_model, 12 heads, 110M params.*
> *bert-large: 24 layers, 1024 d_model, 16 heads, 340M params.*
> *Large is more accurate but 3x slower and 3x more memory.*
> *For most tasks, base is the right starting point."*

**"Can we use BERT for sequence labeling, not just classification?"**
> *"Yes. For NER (which we built in Part 5), you use the full output —
> [batch, seq, 768] — and apply a classifier to each token position
> independently. That's how BERT does NER and POS tagging.*
> *[CLS] is only for sentence-level tasks."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening — bidirectionality hook    10 min
├── BERT pretraining: MLM + NSP        25 min
├── WordPiece tokenization             20 min
├── Live tokenizer demo                20 min
└── Close + homework                    5 min  (approx.)

SESSION 2  (90 min)
├── Opening bridge                     10 min
├── [CLS] embedding extraction         25 min
├── Cosine similarity demo             20 min
├── Lightweight classifier             20 min
└── Close + board summary              10 min  (approx.)
```

---
*MLForBeginners · Part 6: Transformers · Module 06*
