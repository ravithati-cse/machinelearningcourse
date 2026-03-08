# MLForBeginners — Instructor Guide
## Part 5 · Module 08: Named Entity Recognition (NER)
### Two-Session Teaching Script

> **Prerequisites:** Module 07 complete (LSTM, tokenisation, Keras text
> pipelines). They understand sequence modelling and token-level thinking.
> Basic regex from Module 01 (text processing) is also assumed.
> **Payoff today:** Three NER systems — rule-based regex, statistical CRF,
> and spaCy production model — and the skill to extract entities from any
> text.

---

# SESSION 1 (~90 min)
## "Labelling tokens — BIO tagging, rule-based NER, and CRF features"

## Before They Arrive
- Terminal open in `nlp/algorithms/`
- Whiteboard ready
- Write on the board: *"Apple Inc. was founded by Steve Jobs in Cupertino, California"*

---

## OPENING (10 min)

> *"Look at that sentence on the board. You can read it in two seconds and
> immediately know: 'Apple Inc.' is a company, 'Steve Jobs' is a person,
> 'Cupertino' and 'California' are places.*
>
> *Every search engine, every news aggregator, every financial data terminal
> in the world does this automatically, for billions of documents.*
>
> *That is Named Entity Recognition. Today you build it from scratch — and
> then you use the production-grade version.*
>
> *But first — the thing that trips everyone up — how do we label sequences?"*

Point at the sentence on the board. Ask the class:
> *"If I want to label every single word with its entity type, what problem
> do I immediately run into?"*

Expected answer: multi-word entities like "Steve Jobs" and "Apple Inc." —
how do I know "Steve" and "Jobs" together are ONE person, not two?

> *"Exactly. That is why we need a tagging scheme."*

---

## SECTION 1: BIO Tagging Scheme (20 min)

Write on board:
```
NAIVE APPROACH (wrong):
  Apple  Inc.  founded  Steve  Jobs  Cupertino  California
   ORG    ORG    O       PER    PER    LOC         LOC

  Problem: Is [Apple, Inc.] one ORG span or two?
           Is [New, York, City] one LOC span or three?

BIO TAGGING (correct):
  B = Beginning of an entity span
  I = Inside (continuing) an entity span
  O = Outside (not an entity)

  Apple  Inc.  was  founded  by  Steve  Jobs  in  Cupertino  California
  B-ORG  I-ORG  O    O       O   B-PER  I-PER  O   B-LOC      B-LOC

  Now we KNOW: [Apple Inc.] is one ORG (B then I)
               [Steve Jobs] is one PER (B then I)
               [Cupertino] is one LOC (single B)
               [California] is one LOC (single B, different span)
```

> *"The B-tag marks the START of a new entity. The I-tag continues it.
> Once you see a B-tag of the SAME type or a different token entirely,
> the previous span is closed.*
>
> *This is the CoNLL-2003 standard, used in almost all NER benchmarks."*

**Ask the room:** *"How would you tag 'New York City' as a single location?"*

Answer:
```
New    York   City
B-LOC  I-LOC  I-LOC
```

Then ask: *"What about 'New York Yankees'?"*
```
New    York   Yankees
B-ORG  I-ORG  I-ORG
```

> *"The B-tag tells us it's a new entity. The type (LOC vs ORG) changes
> based on what it refers to in context. 'New York' alone is a city.
> 'New York Yankees' is an organisation. Context matters."*

---

## SECTION 2: BIOES — The Extended Scheme (10 min)

Write on board:
```
BIOES = B / I / O / E / S
  B = Begin     (first token of multi-token entity)
  I = Inside    (middle tokens)
  O = Outside   (not an entity)
  E = End       (last token of multi-token entity)
  S = Single    (entire entity is one token)

  "Steve Jobs founded Apple"
   B-PER I-PER   O     S-ORG

vs BIO:
   B-PER I-PER   O     B-ORG

  BIOES advantage: S-tag unambiguously marks a single-token entity.
                   Models trained with BIOES often outperform BIO.
  Used by: spaCy, Flair, AllenNLP
```

> *"You will rarely implement BIOES yourself — but you will see it in
> spaCy's output and in research papers. When you see 'S-LOC', that is
> a single-word location entity."*

---

## SECTION 3: Rule-Based NER with Regex and Gazetteers (25 min)

> *"The simplest NER: write rules. Use regular expressions for patterns
> (phone numbers, dates, email addresses). Use gazetteers for names —
> a gazetteer is just a lookup list.*
>
> *This is still widely used in production for structured entities:
> dates, monetary values, phone numbers, product IDs."*

Write on board:
```
REGEX PATTERNS FOR ENTITIES:

  Date:    \b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b
           matches: 03/15/2024, 15-03-24

  Time:    \b\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?\b
           matches: 3:45PM, 14:30:00

  Money:   \$[\d,]+(?:\.\d{2})?|\b[\d,]+\s(?:dollars|USD|EUR)\b
           matches: $1,200.50, 500 USD

  Email:   \b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b

GAZETTEERS (lookup lists):
  person_names.txt:  ["Steve Jobs", "Tim Cook", "Elon Musk", ...]
  companies.txt:     ["Apple", "Google", "Microsoft", ...]
  cities.txt:        ["New York", "London", "Tokyo", ...]
```

**Live demo — rule-based NER:**
```python
import re

MONEY_PATTERN = re.compile(r'\$[\d,]+(?:\.\d{2})?')
DATE_PATTERN  = re.compile(r'\b\d{1,2}/\d{1,2}/\d{4}\b')

text = "Apple paid $2,400,000.00 on 03/15/2024 for the acquisition."

money_matches = [(m.start(), m.end(), m.group(), "MONEY")
                 for m in MONEY_PATTERN.finditer(text)]
date_matches  = [(m.start(), m.end(), m.group(), "DATE")
                 for m in DATE_PATTERN.finditer(text)]

all_entities = sorted(money_matches + date_matches)
for start, end, text_span, label in all_entities:
    print(f"  [{label}] '{text_span}'  (chars {start}-{end})")
```

> *"This is simple but powerful for structured entities. Dates, prices,
> phone numbers — regex catches them more reliably than any ML model
> in most production systems."*

**Ask the room:** *"What entities CANNOT be caught by regex or gazetteer lookup?"*

Expected answers:
- New companies not in the gazetteer
- Ambiguous names ("Apple" = fruit or company?)
- Context-dependent entities

> *"Exactly. For those, we need a statistical model that understands context."*

---

## CLOSING SESSION 1 (5 min)

Board summary:
```
TODAY:
  BIO tagging    → label each token: B-PER, I-PER, B-ORG, O...
  BIOES          → adds End and Single tags (used by spaCy)

  Rule-based NER:
    Regex         → structured patterns (dates, money, emails)
    Gazetteers    → lookup lists (known names, companies, places)
    + Fast, deterministic, interpretable
    - Misses ambiguous, context-dependent, novel entities
```

**Homework:** Find one sentence in a news article that contains at least
three different entity types (PERSON, ORG, LOC, DATE). Write out the BIO
tags for each token manually. Bring it to Session 2.

---

# SESSION 2 (~90 min)
## "Statistical NER — CRF features, spaCy in production, span-level F1"

## OPENING (10 min)

> *"You manually tagged a sentence. You probably found it harder than
> expected — especially for ambiguous cases. Statistical models do this
> automatically by learning from thousands of tagged examples.*
>
> *Today we look at the two most important statistical approaches:
> CRF (the classic) and spaCy (the production workhorse).*
>
> *We will also learn the right way to MEASURE NER performance —
> which is trickier than you might expect."*

---

## SECTION 1: CRF — Conditional Random Fields (20 min)

> *"CRF is a sequence labelling model. It does not use neural networks —
> it uses engineered features and learns transition probabilities between tags.*
>
> *For example: after a B-PER tag, I-PER is very likely. O is less likely.
> B-LOC is unlikely. The CRF learns these transition costs from data."*

Write on board:
```
CRF FEATURE ENGINEERING FOR NER:

For each token at position i, we extract features:
  word.lower()            "steve" or "apple"
  word.isupper()          False (Apple) or True (NASA)
  word.istitle()          True (Steve) — Capitalised → likely entity
  word.isdigit()          False
  word[-2:]               "le", "bs"  — suffix often indicates proper noun
  word[:2]                "St", "Ap"  — prefix
  prev_word.lower()       previous word
  next_word.lower()       next word
  BOS / EOS               beginning/end of sentence flags

TRANSITION SCORES (learned):
  B-PER → I-PER : +2.3   (likely: continue person span)
  B-PER → B-LOC : -0.8   (unlikely: switch type mid-span)
  I-ORG → O     : +0.5   (possible: end org span)
  O     → I-PER : -3.1   (very unlikely: I without B)
```

> *"The CRF learns that capitalised words following 'Mr.', 'Dr.', 'CEO of'
> are very likely person names — without any of us telling it that explicitly.*
> It infers these patterns from thousands of labelled examples."*

**Ask the room:** *"Why is it important to include features about the PREVIOUS
and NEXT word, not just the current word?"*

Expected answer: Context resolves ambiguity. "Washington" can be a person
(Washington crossed the Delaware) or a place (Washington D.C.) — the
surrounding words disambiguate.

---

## SECTION 2: spaCy — Production NER (20 min)

> *"spaCy is the industry standard for production NLP. Its NER model is
> a transition-based parser — more sophisticated than CRF — trained on
> large corpora like OntoNotes (news, telephone, web text).*
>
> *It is also fast: processes millions of tokens per second on CPU."*

Write on board:
```python
import spacy

nlp = spacy.load("en_core_web_sm")   # small model (~12MB)
# or  spacy.load("en_core_web_lg")   # large model (~750MB, better accuracy)

doc = nlp("Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.")

for ent in doc.ents:
    print(f"  {ent.text:20s} {ent.label_:8s} [{ent.start_char}:{ent.end_char}]")

# Output:
#   Apple Inc.           ORG      [0:10]
#   Steve Jobs           PERSON   [23:33]
#   Cupertino            GPE      [37:46]
#   California           GPE      [48:58]
#   1976                 DATE     [62:66]
```

> *"GPE = Geo-Political Entity — a location that also has political meaning
> (country, city, state). spaCy uses OntoNotes entity types: ORG, PERSON,
> GPE, LOC, DATE, MONEY, EVENT, PRODUCT, and 11 more.*
>
> *Note it found '1976' as a DATE automatically — no regex needed."*

**Show the displacy visualisation:**
```python
from spacy import displacy
displacy.render(doc, style="ent", jupyter=True)
```

> *"displacy renders colour-coded entity spans inline. This is the fastest
> way to sanity-check NER output on new data."*

**Ask the room:** *"spaCy says 'Apple' is an ORG. But 'I ate an apple' —
how would you expect spaCy to handle that?"*

Expected answer: spaCy should NOT tag it as an entity because context shows
it is lowercase and preceded by "an". Run it live to confirm.

---

## SECTION 3: Span-Level F1 — The Right NER Metric (15 min)

> *"You cannot use token-level accuracy to evaluate NER. Here is why."*

Write on board:
```
PREDICTED:  Apple   Inc.  was   founded   by  Steve  Jobs
GOLD:       B-ORG  I-ORG   O      O       O  B-PER  I-PER

Case 1 — CORRECT prediction:
  pred = B-ORG I-ORG → span "Apple Inc." [ORG]
  gold = B-ORG I-ORG → span "Apple Inc." [ORG]
  → TRUE POSITIVE (full span match)

Case 2 — PARTIAL prediction (wrong):
  pred = B-ORG O     → span "Apple" [ORG]  (missed "Inc.")
  gold = B-ORG I-ORG → span "Apple Inc." [ORG]
  → FALSE NEGATIVE (gold not found) + FALSE POSITIVE (wrong span predicted)

SPAN-LEVEL F1 SCORING:
  A predicted span is a TRUE POSITIVE only if:
    1. Exact start AND end position match
    2. Entity type matches
  Partial matches count as errors.

  precision = TP / (TP + FP)
  recall    = TP / (TP + FN)
  F1        = 2 × precision × recall / (precision + recall)
```

> *"This is harsh — get one token boundary wrong and the entire span is wrong.
> State-of-the-art NER models score around F1=0.92 on CoNLL-2003.
> spaCy's large model gets about 0.86-0.88 on general English text.*
>
> *Domain-specific text (medical, legal) often requires fine-tuning."*

---

## SECTION 4: Run the Module + Visualise (15 min)

```bash
python3 nlp/algorithms/named_entity_recognition.py
```

Watch together:
- Rule-based NER output on the sample sentences
- CRF feature extraction printout
- spaCy predictions with entity spans
- Entity type distribution chart

> *"Look at the entity distribution chart: PERSON and ORG dominate in most
> news text. DATE and MONEY appear frequently in financial and business text.
> LOC is everywhere.*
>
> *When you fine-tune a NER model for a specific domain — medical records,
> legal contracts — the distribution shifts dramatically."*

Open the span prediction visualisation. Ask:
> *"Which spans did the rule-based system miss that spaCy caught?
> Which did rule-based catch correctly that spaCy got wrong?"*

---

## CLOSING SESSION 2 (10 min)

Board summary:
```
THREE NER APPROACHES:
  Rule-based (Regex + Gazetteer):
    + Deterministic, fast, interpretable
    + Perfect for structured patterns (dates, money, IDs)
    - Misses contextual, ambiguous, novel entities

  CRF (Feature Engineering):
    + Learns transition probabilities
    + Good on small labelled datasets
    - Requires careful feature engineering

  spaCy (Neural + Production):
    + Best accuracy on general English
    + Multi-entity types, fast, displacy visualisation
    - Needs fine-tuning for specialised domains

EVALUATION:
  Span-level F1 (seqeval) — the ONLY honest NER metric
  Partial span matches count as errors
```

**Homework — entity extraction exercise:**
Take any news article. Run it through spaCy. Collect all unique PERSON,
ORG, and GPE entities. Which two organisations appear most often?
Which person is mentioned most? Write the code to answer these questions
automatically.

---

## INSTRUCTOR TIPS

**"Why does spaCy miss some entities I can clearly see?"**
> *"Three common causes: (1) The entity was not in the training distribution —
> a new company or a niche proper noun. (2) The text is grammatically unusual —
> headlines, tweets, code. (3) Spelling errors — 'Micorsoft' is not in the
> vocabulary. The fix for (1) is fine-tuning. For (2) and (3), preprocess first."*

**"CRF vs spaCy — when would I use CRF in production?"**
> *"If you have a small labelled dataset (500-2000 sentences), CRF often
> outperforms a poorly fine-tuned spaCy model. CRF also trains in seconds
> while fine-tuning spaCy takes minutes. For very resource-constrained
> environments (IoT, mobile), CRF is smaller. But for general production
> use with adequate data, spaCy wins."*

**"What if I need to add a custom entity type — like DRUG or SYMPTOM?"**
> *"That is NER fine-tuning. You annotate examples with your custom entity type,
> then update spaCy's model using spacy train. 200-500 annotated sentences
> per entity type is usually enough to get reasonable performance.
> Prodigy (from Explosion.ai, spaCy's creators) is the standard tool for
> annotation."*

**"The span-level F1 seems brutal — is there a partial credit version?"**
> *"Yes — there are four overlap variants: exact, partial, type-only, and any.
> The CoNLL standard is exact match. For exploratory analysis or when exact
> boundaries are hard to define (e.g., 'Secretary of State John Kerry' —
> is 'John Kerry' the entity or 'Secretary of State John Kerry'?), partial
> match scoring can be more informative. seqeval computes exact by default."*

---

## Quick Reference

```
SESSION 1  (90 min)
├── Opening hook                10 min
├── BIO tagging scheme          20 min
├── BIOES extended scheme       10 min
├── Rule-based NER demo         25 min
└── Close + homework             5 min (adjust: 10 min open → 5 min rule)

Actually:
├── Opening hook                10 min
├── BIO tagging scheme          20 min
├── BIOES extended scheme       10 min
├── Rule-based regex + gazetteer 25 min
└── Close + homework            15 min (adjust demo + close into slot)

SESSION 2  (90 min)
├── Opening bridge              10 min
├── CRF feature engineering     20 min
├── spaCy production NER        20 min
├── Span-level F1 metric        15 min
├── Run module + visualise      15 min
└── Close + homework            10 min
```

---
*MLForBeginners · Part 5: NLP · Module 08*
