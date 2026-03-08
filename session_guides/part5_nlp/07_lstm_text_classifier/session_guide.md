# MLForBeginners — Instructor Guide
## Part 5 · Module 07: LSTM Text Classifier
### Two-Session Teaching Script

> **Prerequisites:** Modules 05-06 complete. They know TF-IDF pipelines,
> logistic regression for text, and the RNN intuition from Module 04.
> They are comfortable with Keras Sequential models from Parts 3-4.
> **Payoff today:** They build a full deep learning text pipeline —
> tokenise, pad, embed, LSTM — and understand when it beats TF-IDF and when
> it does not.

---

# SESSION 1 (~90 min)
## "Giving the model memory — from bag-of-words to sequences"

## Before They Arrive
- Terminal open in `nlp/algorithms/`
- Whiteboard split in two halves: "TF-IDF sees:" on left, "LSTM sees:" on right
- Keras import tested: `python3 -c "import tensorflow; print('ok')"`

---

## OPENING (10 min)

> *"Imagine you are a detective. You get a case file — 200 words. You need to
> decide: is this witness telling the truth or is this a cover story?*
>
> *Option 1: Someone gives you a bag containing 200 individual words,
> scrambled. You look at the bag. You count how many times 'alibi' appears,
> how many times 'nervous' appears. You make your call.*
>
> *Option 2: You read the file yourself, one word at a time. As you read,
> you remember what came before. 'He said he was at home' — then later
> 'He wasn't sure which home.' The contradiction is only visible if you
> read in order.*
>
> *TF-IDF is Option 1. LSTM is Option 2. Today we build Option 2."*

Draw on board:
```
TF-IDF                          LSTM

  "I love this movie"           "I love this movie"
       |                              |
  count words, ignore order     read word by word, left to right
       |                              |
  { love:1, this:1, ... }       hidden state updates at each step
       |                              |
  flat vector                   remembers "I love" when reading "movie"
       |                              |
  classifier                    richer, ordered representation
```

---

## SECTION 1: Tokenisation and Vocabulary Building (20 min)

> *"Before we can feed text to an LSTM, we need to convert words to integers.
> LSTM cannot read letters — it reads numbers.*
>
> *Step 1: Build a vocabulary — a lookup table from word to integer index."*

Write on board:
```
TEXT:  "I love this film"
       "This film is great"
       "I hate bad films"

VOCABULARY (sorted by frequency):
  <PAD>  = 0    (padding token — fills short sequences)
  <UNK>  = 1    (unknown token — words not in vocabulary)
  "this" = 2    (appears 2 times)
  "film" = 3    (appears 2 times)
  "i"    = 4
  "love" = 5
  "is"   = 6
  "great"= 7
  "hate" = 8
  "bad"  = 9
  "films"= 10

ENCODED:
  "I love this film"   → [4, 5, 2, 3]
  "This film is great" → [2, 3, 6, 7]
  "I hate bad films"   → [4, 8, 9, 10]
```

> *"The vocabulary size is a hyperparameter — 'max_features'. If you set
> max_features=10000, you keep the 10,000 most frequent words. Everything
> else becomes <UNK>.*
>
> *Rare words often don't help anyway — they appear too infrequently to
> learn reliable weights."*

**Ask the room:** *"What happens if a word appears in the test set that was
never in the training set vocabulary?"*

Answer: it maps to `<UNK>` token. The model has learned a generic representation
for "I don't recognise this word" — not ideal but acceptable.

---

## SECTION 2: Padding Sequences (20 min)

> *"Here is a problem: reviews have different lengths.*
>
> *'Great film' → [7, 3]  — length 2*
> *'This was the most beautifully crafted...' → length 47*
>
> *Neural networks need fixed-size inputs for batch training.
> Solution: pad all sequences to the same length."*

Write on board:
```
MAX_LEN = 10   (we pad/truncate everything to 10 tokens)

Short sequence (pad with zeros at the end):
  [4, 5, 2, 3]   → [4, 5, 2, 3, 0, 0, 0, 0, 0, 0]
                                   ^^^^^^^^^^^^^^^^^^^
                                   zero-padding

Long sequence (truncate):
  [4, 8, 9, 10, 2, 3, 6, 7, 5, 4, 2, 1, 9]
   → [4, 8, 9, 10, 2, 3, 6, 7, 5, 4]  (truncate at MAX_LEN)

In Keras:
  from tensorflow.keras.preprocessing.sequence import pad_sequences
  X_padded = pad_sequences(sequences, maxlen=100, padding="post", truncating="post")
```

> *"The zeros from padding are a problem: the LSTM will spend time processing
> empty padding tokens. Solution: masking.*
>
> *Masking tells the LSTM 'ignore any timestep with value 0'. Keras Embedding
> layer has a mask_zero=True argument that propagates a mask through all
> subsequent layers automatically."*

**Ask the room:** *"Should you pad at the beginning of the sequence or the
end? Does it matter for an LSTM?"*

Answer: technically it matters — LSTM processes left to right, so padding
at the end means the final hidden state sees real content. Padding at the
beginning buries content later. `padding='post'` (end padding) is generally
preferred.

---

## SECTION 3: The Embedding Layer (15 min)

> *"Once words are integers, we could feed them directly to the LSTM.
> But integer 4 and integer 5 are not more 'similar' than integer 4 and
> integer 400. Numbers have no semantic meaning.*
>
> *The Embedding layer is a lookup table: integer → dense vector.
> It converts index 4 to a 128-dimensional vector, index 5 to another,
> and LEARNS those vectors during training."*

Write on board:
```
EMBEDDING LAYER:

  Vocab size: 10,000    Embedding dim: 128

  word "love" → index 5 → [ 0.23, -0.41, 0.17, ..., 0.09 ]  (128 numbers)
  word "like" → index 18→ [ 0.21, -0.38, 0.19, ..., 0.11 ]  (128 numbers)
  word "hate" → index 8 → [-0.24,  0.39, -0.18, ..., -0.08] (128 numbers)

  "love" and "like" are close in embedding space
  "hate" is far away — the model learned semantic relationships

Keras:
  Embedding(input_dim=10000, output_dim=128, mask_zero=True)

  input:  batch of integer sequences    shape (batch, max_len)
  output: batch of embedding sequences  shape (batch, max_len, 128)
```

> *"The Embedding layer has 10,000 × 128 = 1.28 million parameters that
> get trained by backpropagation. Alternatively, you can initialise with
> pretrained GloVe or Word2Vec vectors — which we saw in Module 03 —
> and optionally freeze them (trainable=False)."*

---

## CLOSING SESSION 1 (5 min)

Board summary:
```
DEEP LEARNING TEXT PIPELINE (so far):

  Raw text
   → Tokenize + build vocabulary
   → pad_sequences (maxlen, padding='post')
   → Embedding(vocab_size, embed_dim, mask_zero=True)

TOMORROW:
   → LSTM reads the embedded sequence
   → GlobalMaxPooling1D collapses to one vector
   → Dense + softmax → prediction
```

**Homework:** What is the total parameter count of an Embedding layer with
vocabulary size 20,000 and embedding dimension 200? What about 300 dimensions?

---

# SESSION 2 (~90 min)
## "LSTM, BiLSTM, GRU — adding the sequence model and benchmarking"

## OPENING (10 min)

> *"We have the pipeline built up to the Embedding layer. Today we add the
> recurrent layer — LSTM — and then we benchmark it against the TF-IDF
> baseline from Module 05.*
>
> *I'll also show you two variants: Bidirectional LSTM and GRU.
> By the end you will know which one to reach for and when."*

---

## SECTION 1: The LSTM Layer in Keras (20 min)

> *"In Module 04 we built the intuition for how LSTM works: it has gates that
> control what to remember, what to forget, and what to output.*
>
> *In Keras, all of that complexity is one line. Let's build the full model."*

Write the architecture on board:
```
FULL MODEL ARCHITECTURE:

  Input: integer sequences, shape (batch, 100)
       |
  Embedding(10000, 128, mask_zero=True)
       |  shape: (batch, 100, 128)
       |
  LSTM(64, return_sequences=False)
       |  shape: (batch, 64)
       |  [returns only the final hidden state]
       |
  Dropout(0.3)
       |
  Dense(32, activation='relu')
       |
  Dense(2, activation='softmax')   [or Dense(1, activation='sigmoid')]
       |
  prediction
```

Code together:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, mask_zero=True),
    LSTM(64),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

> *"Look at the parameter count in model.summary(). The Embedding layer
> dominates — 10,000 × 128 = 1.28M parameters. The LSTM itself is only
> about 50K. This is why pretrained embeddings save so much training time:
> you skip learning 1.28M of those parameters from scratch."*

**Ask the room:** *"What does return_sequences=False mean? When would you
use return_sequences=True?"*

Answer: `return_sequences=False` returns only the final timestep's hidden
state — a single vector. `return_sequences=True` returns the full sequence
of hidden states — needed when stacking two LSTM layers on top of each other.

---

## SECTION 2: BiLSTM and GRU — Variants Explained (20 min)

Write on board:
```
THREE RECURRENT VARIANTS:

LSTM (Long Short-Term Memory):
  Reads LEFT → RIGHT
  Has 4 gates: forget, input, cell, output
  ~4x parameters of equivalent GRU
  Best when: long documents, fine-grained sentiment

BiLSTM (Bidirectional LSTM):
  Reads LEFT → RIGHT  AND  RIGHT → LEFT simultaneously
  Concatenates both directions: output dim doubles
  Best when: context from both directions matters
             (e.g. "she kicked the bucket" — idiom needs right context)

GRU (Gated Recurrent Unit):
  Reads LEFT → RIGHT
  Has 2 gates: reset, update  (simpler than LSTM)
  ~25% fewer parameters, trains ~25% faster
  Usually matches LSTM quality on text classification
  Best when: limited compute, shorter sequences
```

Code the BiLSTM variant:
```python
from tensorflow.keras.layers import Bidirectional

model_bilstm = Sequential([
    Embedding(10000, 128, mask_zero=True),
    Bidirectional(LSTM(64)),   # output dim = 64*2 = 128
    Dropout(0.3),
    Dense(2, activation='softmax')
])
```

> *"For sentiment analysis, BiLSTM often wins over regular LSTM because
> negation context can come AFTER the sentiment word.*
>
> *'The food was — for once — not bad at all.'*
>
> *The negation 'not bad' is the key signal. A left-to-right LSTM processes
> 'bad' before it sees 'not'. A BiLSTM processes from both directions and
> connects them."*

---

## SECTION 3: GlobalMaxPooling vs Last Hidden State (15 min)

> *"There is an alternative to using the final hidden state.*
>
> *If we use return_sequences=True, we get the full sequence of hidden states.
> We can then apply GlobalMaxPooling1D — which takes the maximum value at
> each dimension across all timesteps.*
>
> *This is like asking: what is the most 'activated' the model ever got,
> for each feature, at any point in the sequence?"*

Write on board:
```
Hidden states (one per word):
  t=1: [ 0.1,  0.8, -0.3]
  t=2: [ 0.5,  0.2,  0.9]
  t=3: [-0.1,  0.6,  0.4]
  t=4: [ 0.9, -0.2,  0.1]

GlobalMaxPooling1D → take max over time dimension:
  [ max(0.1,0.5,-0.1,0.9), max(0.8,0.2,0.6,-0.2), max(-0.3,0.9,0.4,0.1) ]
  [       0.9,                    0.8,                      0.9           ]
```

> *"GlobalMaxPooling is especially useful for long documents where the
> key signal might appear anywhere — not just at the end.
> For short texts, the final hidden state usually works just as well."*

---

## SECTION 4: Benchmark — LSTM vs TF-IDF Baseline (15 min)

Show the comparison from running the module:
```
python3 nlp/algorithms/lstm_text_classifier.py
```

Expected results table — draw on board:
```
MODEL                    ACCURACY   PARAMS     TRAIN TIME
TF-IDF + LogReg          0.88       ~50K       0.5s
TF-IDF + Dense           0.89       ~1.3M      8s
LSTM                     0.91       ~1.4M      45s
BiLSTM                   0.92       ~1.5M      70s
GRU                      0.91       ~1.4M      38s
```

> *"Four percent accuracy improvement from TF-IDF to BiLSTM.
> Is that worth 140x longer training time and 30x more parameters?*
>
> *That depends entirely on your application.*
>
> *Spam filter on 10 million emails per day? Go TF-IDF.*
> *Detecting suicidal ideation in social media posts? That 4% matters enormously.*
>
> *Always ask: what is the COST of the mistake we are trying to prevent?"*

**Ask the room:** *"What kinds of reviews in the IMDb dataset do you think
the LSTM gets right that TF-IDF gets wrong? What would you look for in the
errors?"*

Encourage them to look at the error analysis printed by the module — reviews
with negation, long complex sentences, sarcasm.

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
FULL DEEP LEARNING TEXT PIPELINE:
  Raw text
   → Tokenize + Vocabulary
   → pad_sequences(maxlen=100, padding='post')
   → Embedding(vocab_size, 128, mask_zero=True)
   → LSTM / BiLSTM / GRU
   → GlobalMaxPooling1D or last hidden state
   → Dense + softmax → prediction

WHEN TO CHOOSE WHAT:
  TF-IDF + LR     → default first choice, fast, interpretable
  LSTM            → word order matters, medium datasets (>5K)
  BiLSTM          → context from both directions needed
  GRU             → faster version of LSTM, similar accuracy
  Transformer     → best accuracy (next section of the course)
```

**Homework — tune one hyperparameter:**
Change the LSTM units from 64 to 128. Does accuracy improve? Does training
time increase proportionally? Check the parameter count in model.summary()
before and after.

---

## INSTRUCTOR TIPS

**"My LSTM is not converging — loss stays high"**
> *"Check three things: (1) Is padding='post' and mask_zero=True? Without
> masking, the model wastes capacity on zeros. (2) Is the learning rate
> too high? Try 1e-4 instead of Adam's default 1e-3. (3) Are sequences
> too long? Try maxlen=50 first."*

**"When should I use pretrained embeddings vs training from scratch?"**
> *"Pretrained GloVe or Word2Vec when: your training dataset is small (<10K
> examples), your domain is general English. Train from scratch when: you
> have a specialised domain (medical, legal) where pretrained vectors encode
> the wrong semantic relationships, or when you have >100K examples."*

**"BiLSTM is slower but barely more accurate — why use it?"**
> *"On sentiment specifically, the gain is small because most sentiment words
> appear near the end or are self-contained. BiLSTM shines on tasks where
> the label depends on the full context — NER, question answering, parsing.
> For topic classification, the gain over LSTM is negligible."*

**"Student asks: why not just use Transformers instead of LSTM?"**
> *"You will — Module 06 is Transformers, Module 07 is LLMs. LSTM is still
> taught because: (1) it is faster and lighter for embedded systems,
> (2) understanding LSTM gates gives you the conceptual foundation
> for attention mechanisms, (3) for small datasets, LSTM often beats
> a transformer fine-tuned without enough data."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                10 min
├── Tokenisation + vocabulary   20 min
├── Padding sequences           20 min
├── Embedding layer             15 min
└── Close + homework             5 min  (wait — 10 min total missing?)

Wait — re-check:
├── Opening hook                10 min
├── Tokenisation + vocabulary   20 min
├── Padding sequences           20 min
├── Embedding layer             15 min
├── Running partial demo        15 min
└── Close + homework            10 min

SESSION 2  (90 min)
├── Opening bridge              10 min
├── LSTM layer in Keras         20 min
├── BiLSTM and GRU variants     20 min
├── GlobalMaxPooling vs hidden  15 min
├── Benchmark vs TF-IDF         15 min
└── Close + homework            10 min
```

---
*MLForBeginners · Part 5: NLP · Module 07*
