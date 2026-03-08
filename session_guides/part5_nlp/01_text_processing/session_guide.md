# MLForBeginners — Instructor Guide
## Part 5, Module 1: Text Processing  ·  Two-Session Teaching Script

> **Who this is for:** You, teaching close friends who finished Parts 1-4.
> **They already know:** DNNs, Keras, backpropagation, gradient descent.
> **Tone:** Casual and conversational — like explaining over coffee.
> **Goal:** Everyone leaves understanding why raw text is hard for machines
> and can build a full preprocessing pipeline from scratch.

---

# ─────────────────────────────────────────────
# SESSION 1  (~90 min)
# "Why your computer can't read — and how we fix that"
# ─────────────────────────────────────────────

## Before They Arrive (10 min setup)

**On your laptop, have ready:**
- Terminal open in `MLForBeginners/nlp/math_foundations/`
- The file `01_text_processing.py` open in your editor
- Visuals folder `nlp/visuals/01_text_processing/` open in Finder
- A whiteboard or large notepad

**Room vibe:** NLP is surprisingly intuitive — most people already "get it"
because they use language every day. Lean into that familiarity.

---

## OPENING  (10 min)

### Hook — The spam email question

Say this out loud, naturally:

> *"Quick question. How does Gmail know that 'CONGRATULATIONS! You have WON a
> FREE iPhone!!!!' is spam, but 'Congratulations on your promotion, looking
> forward to celebrating' is not?*
>
> *Both start with 'Congratulations.' The computer can't read. It has no idea
> what 'congratulations' means. So what is it actually doing?*
>
> *That's NLP. Natural Language Processing. And for the next few weeks,
> that's what we're learning."*

**Then ask the room:**

> *"What's the fundamental problem? Why can't we just feed text straight into
> a neural network like we feed pixel values into a CNN?"*

Let them think for 30 seconds. Someone will probably say "it's not numbers."

> *"Exactly. Neural networks eat numbers. Text is strings. So the entire
> first half of NLP is solving one problem: turning words into numbers
> that actually capture meaning.*
>
> *Today we do the first step: cleaning and tokenizing raw text.*
> *Everything else in this course builds on getting this right."*

---

## SECTION 1: Why Raw Text Is Hard  (15 min)

### The "dog" problem

**Write on the whiteboard:**

```
"the dog barked loudly"
"The Dog Barked Loudly"
"THE DOG BARKED LOUDLY!!!"
"the dog (a Labrador) barked"
"<p>The dog barked.</p>"
"barking → bark → barked"
"d0g b4rked l0udly"
```

> *"Seven different strings. All say the same thing. A human reads all of them
> and thinks 'dog barking.' A naive model treats every single one as completely
> different inputs.*
>
> *This is the raw text problem. And the solution is a preprocessing pipeline —
> a series of cleaning steps you run BEFORE your model ever sees the text.*
>
> *Think of it like mise en place in cooking. You don't throw raw vegetables
> into a pot. You chop, wash, peel first. Text preprocessing is the mise en place
> of NLP."*

**Ask the room:**

> *"Which of these variations do you think trips models up the most in the real
> world? Typos? HTML tags? Case differences?"*

Let them argue. There's no single right answer. Then:

> *"In practice — HTML tags and case differences are the most common at scale.
> Every blog post, product review, tweet comes with invisible junk. The pipeline
> strips all of it before the model ever touches the text."*

---

## SECTION 2: Tokenization  (20 min)

### The Lego brick analogy

**Draw on the board:**

```
Raw sentence:
"It's a dog-eat-dog world, isn't it?"

Step 1 — TOKENIZE (chop into Lego bricks):
["It's", "a", "dog", "eat", "dog", "world", "isn't", "it"]

Each token = one brick. The model works with bricks, not the whole sentence.
```

> *"Tokenization is chopping a sentence into Lego bricks. Each brick is a token.
> The model never sees 'It's a dog-eat-dog world' — it sees individual pieces.*
>
> *Now here's where it gets interesting. HOW you chop matters enormously."*

**Write three approaches on the board:**

```
WORD-LEVEL:       ["It's", "a", "dog-eat-dog", "world"]
WORD (regex):     ["It's", "a", "dog", "eat", "dog", "world"]
CHARACTER-LEVEL:  ['I', 't', "'", 's', ' ', 'a', ' ', 'd', 'o', 'g', ...]
SUBWORD:          ["It", "'s", "a", "dog", "-", "eat", "-", "dog", "world"]
```

> *"Word-level is simple but fails on typos and new words.*
> *Character-level never has unknown words but loses all word-level meaning.*
> *Subword — that's what GPT and BERT actually use. It's the sweet spot.*
>
> *For now we work at the word level. Subword tokenization comes in Part 6.*"

**Interactive — ask the room:**

> *"What happens if you use whitespace splitting on 'I don't love this movie'?
> What's in your vocabulary? Is 'don't' the same as 'dont'? Same as 'do not'?"*

Let them figure out the problem. Then:

> *"This is why regex tokenization is better than naive splitting.
> You want contractions handled consistently."*

---

## SECTION 3: Text Cleaning  (15 min)

### The cleaning pipeline

**Write on the board:**

```
CLEANING STEPS (in order):
1. Lowercase:     "The DOG" → "the dog"
2. Remove HTML:   "<b>dog</b>" → "dog"
3. Remove URLs:   "visit http://dogs.com" → "visit"
4. Remove punct:  "dog!!!" → "dog"
5. Normalize ws:  "dog  cat" → "dog cat"
```

> *"Each step is simple on its own. Together they standardize every input
> so that 'The DOG!!' and 'the dog' produce identical tokens.*
>
> *One thing to watch: don't remove ALL punctuation blindly. Question marks
> can carry sentiment. Ellipses carry uncertainty. You decide what to keep
> based on your task."*

**Common confusion to pre-empt:**

> *"Some of you might be thinking — if I lowercase everything, doesn't
> 'Apple' the company become the same as 'apple' the fruit?*
>
> *Yes. That's a real problem. Which is why for Named Entity Recognition —
> identifying people, places, companies — you often DON'T lowercase.*
> *You tune the pipeline to your task, not the other way around."*

---

## SECTION 4: Stopwords, Stemming, Lemmatization  (15 min)

### Stopwords — signal vs noise

> *"In any English text, the most common words are: the, is, a, an, of, in, to, it.*
>
> *These words appear in EVERY document. They carry almost no information about
> what the document is ABOUT. We call them stopwords, and we often remove them.*
>
> *But — and this matters — 'not' is sometimes treated as a stopword.
> Remove 'not' from 'I did not love this movie' and you get 'I love this movie.'
> Completely flipped sentiment. Context matters."*

**Write on the board:**

```
BEFORE stopword removal:
["the", "cat", "sat", "on", "the", "mat"]

AFTER (removing the, on):
["cat", "sat", "mat"]
```

### Stemming vs Lemmatization

**Draw a comparison:**

```
STEMMING (chop the suffix — fast, rough):
  "running" → "run"
  "studies" → "studi"       ← "studi" is not a real word!
  "better"  → "better"      ← misses that better = good

LEMMATIZATION (look up the dictionary form — slower, accurate):
  "running" → "run"
  "studies" → "study"       ← real word
  "better"  → "good"        ← understands the relationship
```

> *"Stemming is fast. Lemmatization is accurate. For most NLP pipelines today,
> lemmatization is preferred because storage is cheap and accuracy matters.*
>
> *For search engines where speed is everything, stemming still wins."*

---

## SECTION 5: Live Demo  (15 min)

> *"Let's see all of this in action."*

**Open terminal, navigate to the math_foundations folder:**

```bash
python3 01_text_processing.py
```

Walk through the output as it prints. Point at key moments:

> *"See — Section 1 shows the exact same dog sentence in 8 different forms.
> Section 2 shows the three tokenizers on the same input — notice how
> word-level vs regex gives different results on 'It's'.*
>
> *Section 5 — look at the stemming vs lemmatization comparison.
> 'studies' stems to 'studi' which is garbage. Lemmatized correctly to 'study'.*
>
> *Section 7 is the full pipeline. That one function call handles everything
> we talked about today in the right order."*

**Open the generated visuals** (in `nlp/visuals/01_text_processing/`):

> *"These plots show token frequency distributions — the classic power law shape.
> A few words appear constantly. Most words appear rarely.*
> *This is called Zipf's Law. It holds for EVERY language ever studied.
> It's one of the most reliable patterns in all of linguistics."*

---

## CLOSING SESSION 1  (5 min)

Write on board, have them read it back:

```
THE PREPROCESSING PIPELINE:
1. Clean   → lowercase, remove HTML/URLs/punctuation
2. Tokenize → chop into word or subword tokens
3. Normalize → stem or lemmatize
4. Filter  → remove stopwords (carefully!)
5. Encode  → assign integer IDs (next module)

Tokenization = chopping a sentence into Lego bricks.
Each brick = one token. Model works with bricks.
```

---
---

# ─────────────────────────────────────────────
# SESSION 2  (~90 min)
# "Vocabulary building and the full pipeline"
# ─────────────────────────────────────────────

## Opening  (10 min)

### Homework debrief

> *"Before we continue — who ran the script on their own? What was the
> weirdest or most surprising thing you saw in the output?"*

Let them share. Common observations: the frequency plot, "studi" from stemming.

> *"Good. Today we finish the pipeline: how do we turn those clean tokens
> into the integer IDs a model can actually eat? That's vocabulary building.*
> *And we'll see what happens when a word appears at test time that the
> model has never seen during training — the OOV problem."*

---

## SECTION 6: Vocabulary Building  (25 min)

### From tokens to integers

**Write on the board:**

```
CORPUS (training documents):
  doc1: "the cat sat on the mat"
  doc2: "the dog barked loudly"
  doc3: "my cat is better than your dog"

VOCABULARY (sorted, unique):
  {"barked": 0, "better": 1, "cat": 2, "dog": 3,
   "is": 4, "loudly": 5, "mat": 6, "my": 7,
   "on": 8, "sat": 9, "than": 10, "the": 11,
   "your": 12}

SPECIAL TOKENS:
  <PAD> = 0   (pad short sequences to fixed length)
  <UNK> = 1   (unknown words not in vocabulary)
  <BOS> = 2   (beginning of sequence — used in generation)
  <EOS> = 3   (end of sequence)

ENCODE doc1:
  "the cat sat on the mat"
  →  [11, 2, 9, 8, 11, 6]
```

> *"That's it. Your model never sees text. It sees integer sequences.*
> *Vocabulary = the dictionary that maps words to numbers.*
>
> *Now the question: what vocabulary size do you pick?*
> *Too small: you miss rare but important words.*
> *Too large: memory blows up, model learns noise.*
> *In practice: 20,000 to 50,000 for English. BPE / WordPiece for modern models."*

### The OOV problem

**Ask the room:**

> *"You build your vocabulary on the training set. At test time, a user types
> 'cryptocurrency' — and that word was never in your training data.*
> *What does your model do?"*

Let them think. Then:

> *"Without handling this: the model crashes or ignores the word entirely.*
> *With an UNK token: the word maps to UNK and the model at least knows
> 'something unknown appeared here.'*
>
> *Modern models use subword tokenization — they break 'cryptocurrency' into
> ['crypto', 'currency'] or even ['crypt', 'o', 'currency']. You can always
> tokenize known subwords, so OOV essentially disappears.*
> *That's why GPT and BERT never see an unknown word.*"

**Write on the board:**

```
OOV STRATEGIES:
1. UNK token           → simplest, loses information
2. Character fallback  → spell out unknown words as characters
3. Subword (BPE/WP)    → break into known subwords (GPT/BERT approach)
4. Hash trick          → hash the word to a fixed-size integer bucket
```

---

## SECTION 7: Full Pipeline Demo  (20 min)

> *"Let's wire it all together. I want you to see the pipeline go from
> a raw tweet to a padded integer sequence ready for a model."*

**Live code on screen (or have them type along):**

```python
import re
from collections import Counter

# --- Raw text ---
tweets = [
    "OMG I LOVE this product!!! <3 #amazing https://t.co/xyz",
    "Terrible. Don't buy this. Total waste of money!!!",
    "Not bad at all — actually pretty decent for the price",
    "Running runs ran — morphological variants test",
]

# --- Step 1: Clean ---
def clean(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # URLs
    text = re.sub(r"<[^>]+>", "", text)           # HTML
    text = re.sub(r"[^\w\s']", " ", text)          # punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Step 2: Tokenize ---
def tokenize(text):
    return re.findall(r"\b\w+(?:'\w+)?\b", text)

# --- Step 3: Vocabulary ---
all_tokens = [t for tweet in tweets for t in tokenize(clean(tweet))]
freq = Counter(all_tokens)
vocab = {"<PAD>": 0, "<UNK>": 1}
vocab.update({w: i+2 for i, (w, _) in enumerate(freq.most_common(50))})

# --- Step 4: Encode ---
def encode(text, max_len=10):
    tokens = tokenize(clean(text))
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    ids = ids[:max_len]               # truncate
    ids += [vocab["<PAD>"]] * (max_len - len(ids))   # pad
    return ids

for tweet in tweets:
    print(f"IN:  {tweet[:50]}")
    print(f"OUT: {encode(tweet)}")
    print()
```

**Ask the room before running:**

> *"What do you expect to see for 'OMG'? Is it in the vocabulary?
> What token ID does an unknown word get?"*

Run it. Point at the UNK tokens.

> *"This is the exact same logic scikit-learn's CountVectorizer and
> Keras's Tokenizer use under the hood. We've just made it transparent."*

---

## SECTION 8: NLTK — Production Text Processing  (10 min)

> *"In the real world, you don't write your own tokenizer.
> NLTK and spaCy are the standard libraries.*
> *Let's see what NLTK adds on top of what we built."*

**Back to the script — show the NLTK section output:**

> *"NLTK's word_tokenize correctly handles contractions: 'don't' → ['do', "n't'].*
> *Its lemmatizer uses WordNet — an actual English dictionary with 150k+ entries.*
> *And it knows about part of speech: 'better' as an adjective lemmatizes to 'good',
> but 'better' as a verb lemmatizes to 'better'. Context matters."*

---

## CLOSING SESSION 2  (10 min)

### Recap board

Write on board as you say each item:

```
FULL PIPELINE TODAY:
─────────────────────────────────────────────
Raw Text
  ↓  1. Clean    (lower, strip HTML/URLs/punct)
  ↓  2. Tokenize (Lego bricks — word or regex)
  ↓  3. Normalize (lemmatize > stem)
  ↓  4. Filter   (stopwords — carefully)
  ↓  5. Vocab    (build word→integer map)
  ↓  6. Encode   (tokens → integer IDs)
  ↓  7. Pad/Trunc (fixed length for batching)
Integer Sequence  →  ready for the model
─────────────────────────────────────────────

KEY DECISIONS:
  lowercase?         yes, almost always
  remove stopwords?  depends on task
  stem or lemmatize? lemmatize preferred
  vocab size?        20k-50k typical
  OOV strategy?      UNK token or subword
```

### The road ahead

```
WHERE WE ARE:
  ✅ Module 1: Text Preprocessing (this module)

NEXT UP:
  → Module 2: Bag of Words + TF-IDF
              (how do we turn integer sequences into meaningful vectors?)
  → Module 3: Word Embeddings
              (how do we make the vectors capture meaning?)
  → Module 4: RNNs
              (how do we handle word ORDER?)
```

---

## Homework

No lab file for this module, but assign this exercise:

```python
# Take any 5 tweets or product reviews you can find online.
# Run them through the pipeline we built today.
# Questions to answer:
#
# 1. Which stopwords would you KEEP for a sentiment task?
#    (Hint: think about "not", "never", "but")
#
# 2. Run stemming vs lemmatization on your tweets.
#    Find one example where stemming gives a clearly wrong result.
#
# 3. What vocab size would you choose if your training set has
#    100 reviews vs 100,000 reviews? Why?
```

---

# ─────────────────────────────────────────────
# INSTRUCTOR TIPS & SURVIVAL GUIDE
# ─────────────────────────────────────────────

## When People Get Confused

**"Why do we lowercase? What if case matters?"**
> *"Great question. For most classification tasks — spam, sentiment, topic —
> case adds noise without adding signal. But for NER (finding people/places/companies),
> capitalization is often the strongest signal. You decide per task.
> The pipeline is tunable, not fixed."*

**"Why not just use the whole sentence as one feature?"**
> *"Because two sentences with the same words in different order look completely
> different as strings, but convey very similar meaning. Tokenization decomposes
> the sentence into units the model can reason about independently.*
> *Later we'll use positional encoding and attention to recover the order."*

**"What's the difference between stemming and lemmatization in practice?"**
> *"Try running 'studies', 'universal', 'better' through both.*
> *Stemmer: 'studi', 'univers', 'better'.*
> *Lemmatizer: 'study', 'universal', 'good'.*
> *The stemmer is wrong on all three. Use lemmatization unless you're in a
> speed-critical pipeline."*

**"Does any of this matter if we're using BERT or GPT?"**
> *"Great observation — modern transformer models use their own built-in
> tokenizers (BPE / WordPiece) and often don't need manual cleaning.*
> *But understanding THIS pipeline is what lets you debug when BERT fails,
> and it's essential when you DON'T have a GPU for a giant model."*

## Energy Management

- **30-min mark:** Natural break. Get coffee. NLP is vocabulary-heavy — let it settle.
- **If they're bored:** Skip straight to the live coding demo. Hands-on beats slides every time.
- **If someone asks about transformers:** "That's Part 6. We're building the foundation."
- **Best engagement moment:** The spam email opening and the OOV tweet demo.

## The Golden Rule

> Every preprocessing decision is a choice with tradeoffs.
> Never present a step as "always do this" — always say "here's when and why."

---

# ─────────────────────────────────────────────
# QUICK REFERENCE — Session Timing
# ─────────────────────────────────────────────

```
SESSION 1  (90 min)
├── Opening hook — spam email         10 min
├── Section 1: Why raw text is hard   15 min
├── Section 2: Tokenization           20 min
├── Section 3: Text cleaning          15 min
├── Section 4: Stopwords/Stemming     15 min
├── Section 5: Live demo              10 min
└── Close + preview                    5 min

SESSION 2  (90 min)
├── Homework debrief                  10 min
├── Section 6: Vocabulary building    25 min
├── Section 7: Full pipeline coding   20 min
├── Section 8: NLTK production        10 min
├── Closing recap board               10 min
└── Homework assignment + road ahead  15 min
```

---

*Generated for MLForBeginners — Module 01 · Part 5: NLP*
