# 🎓 MLForBeginners — Instructor Guide
## Part 2 · Module 11: Spam Classifier (Project)
### Single 120-Minute Session

> **Prerequisites:** All Part 2 modules 01–10. They know logistic regression,
> KNN, decision trees, random forests, and all classification metrics.
> **Payoff:** Build a real spam detector — the kind in your email today.

---

# SESSION (120 min)
## "Build a spam filter from scratch — end to end"

## Before They Arrive
- Terminal open in `classification_algorithms/projects/`
- Pull up a few real spam emails from your own inbox to show (anonymized)
- Whiteboard ready
- Optional: show a Gmail spam folder

---

## OPENING (10 min)

> *"Quick question — raise your hand if you've used email today.*
> *Now keep your hand up if you got a spam email.*
>
> *Every email service in the world runs a spam filter.
> Gmail, Outlook, Yahoo — all of them.*
>
> *Today you're going to build one. From raw text to working classifier.
> The same pipeline the big companies use, just smaller."*

Write on board:
```
RAW EMAIL TEXT
     ↓
Clean & Tokenize ("FREE MONEY!!!" → ["free", "money"])
     ↓
Extract Features (TF-IDF: how unusual is each word?)
     ↓
Train Classifier (logistic regression, Naive Bayes, etc.)
     ↓
Predict: SPAM or HAM?
```

> *"Ham = not spam. That's the actual industry term."*

---

## SECTION 1: The Dataset (10 min)

```bash
python3 spam_classifier.py
```

Let it start loading. While it does:

> *"We're using a classic dataset — thousands of real SMS messages,
> labeled as spam or ham by humans.*
>
> *What do you think the class balance looks like?
> What fraction of messages are spam?"*

Let them guess. Answer: roughly 13% spam, 87% ham.

> *"That imbalance matters. If our model just says 'ham' every time,
> it's 87% accurate — but useless. We need better metrics.*
>
> *This is exactly why we learned precision, recall, and F1."*

---

## SECTION 2: Text Preprocessing (15 min)

Write on board:
```
Raw: "WINNER! You've been selected for a FREE prize! Call NOW!"

Step 1 — Lowercase:    "winner! you've been selected for a free prize! call now!"
Step 2 — Remove punct: "winner youve been selected for a free prize call now"
Step 3 — Tokenize:     ["winner", "youve", "been", "selected", ...]
Step 4 — Remove stops: ["winner", "selected", "free", "prize", "call"]
Step 5 — TF-IDF:       each word gets a score based on rarity
```

> *"Notice the spam language: 'winner', 'free', 'prize', 'call now'.*
> *These words appear in spam but rarely in normal conversation.*
> *TF-IDF captures exactly that signal."*

**Ask the room:**
> *"What words do you think have the highest TF-IDF score in spam messages?*
> *And what would you expect in legitimate messages?"*

---

## SECTION 3: Training Multiple Classifiers (20 min)

> *"We're not going to commit to one algorithm.*
> *We'll train several and compare — that's how professionals work."*

Draw comparison table on board as results come in:
```
Algorithm           Accuracy   Precision  Recall    F1
─────────────────────────────────────────────────────
Naive Bayes         ?          ?          ?         ?
Logistic Regression ?          ?          ?         ?
Random Forest       ?          ?          ?         ?
SVM                 ?          ?          ?         ?
```

Fill in the blanks from the program output together.

> *"Which metric matters most for spam filtering?*
>
> *Think about it:
> False Positive = mark legit email as spam (user misses important email)
> False Negative = let spam through (user gets annoyed)*
>
> *Most services optimize PRECISION — don't wrongly block real emails.
> Better to let some spam through than destroy trust by blocking legitimate email."*

---

## SECTION 4: ROC Curve & Choosing a Threshold (15 min)

Open the generated ROC curve visualization.

> *"The ROC curve shows every possible tradeoff.*
> *Right now the model uses threshold = 0.5 by default.*
>
> *But we can adjust it:
> Threshold = 0.3 → catch more spam (higher recall, more false positives)
> Threshold = 0.7 → only flag very confident spam (higher precision, miss some)"*

```python
# Ask: what threshold would YOU use for a work email filter?
# vs a promotional email folder?
```

> *"AUC = 0.98 means: if you pick one spam and one ham email at random,
> the model ranks the spam higher 98% of the time.*
> *That's excellent."*

---

## SECTION 5: Live Prediction (15 min)

The best part — test custom messages:

```python
test_messages = [
    "You've won a FREE iPhone! Click now to claim!",
    "Hey, are we still on for lunch tomorrow?",
    "URGENT: Your account will be suspended. Verify now.",
    "The meeting notes are attached, let me know your thoughts.",
    "Congratulations! You've been pre-approved for a loan!"
]
```

> *"Before we run — everyone predict: spam or ham for each one.*
> *Shout it out."*

Run it. The satisfying moment when the model gets them all right.

---

## SECTION 6: What Makes a Word 'Spammy'? (15 min)

Show the top features visualization — the words with highest coefficients.

> *"Look at these words. 'Free', 'prize', 'urgent', 'winner', 'call'.*
> *These are literally the spam playbook.*
>
> *And on the ham side: 'meeting', 'attached', 'team', 'report'.*
>
> *This is what interpretability looks like.
> We don't just get predictions — we understand WHY.*
> A lawyer asking 'why was this email marked spam?' can get an answer."*

---

## CLOSING (15 min)

Write on board:
```
WHAT WE BUILT TODAY:
  Raw text → TF-IDF features → trained classifier → spam/ham

WHAT WE LEARNED:
  • Class imbalance requires careful metric selection
  • Precision vs Recall tradeoff is a business decision
  • Interpretable features build trust in production
  • AUC lets you compare classifiers independently of threshold

REAL WORLD:
  Gmail spam filter: billions of emails/day, same basic pipeline
  just at massive scale with additional signals (sender reputation, etc.)
```

**Homework:**
> *"Try modifying the test_messages list with your own examples.*
> *Can you fool the classifier? What kinds of messages trip it up?*
> *That's called adversarial testing — real ML teams do this constantly."*

---

## INSTRUCTOR TIPS

**"Why Naive Bayes for spam specifically?"**
> *"Naive Bayes assumes all words are independent — obviously false
> (words relate to each other). But for spam it works surprisingly well.
> It's fast, interpretable, and the 'naive' assumption doesn't hurt much
> because spammy words do appear independently of context."*

**"How do spam filters avoid being tricked?"**
> *"Real filters use many more signals: sender reputation, IP address,
> email headers, HTML patterns, link destinations.
> ML is just one layer. Adversarial robustness is a whole research field."*

---

## Quick Reference
```
Single Session (120 min)
├── Opening hook           10 min
├── Dataset + class balance 10 min
├── Text preprocessing     15 min
├── Multiple classifiers   20 min
├── ROC + threshold        15 min
├── Live predictions       15 min
├── Feature importance     15 min
└── Close + homework       20 min
```

---
*MLForBeginners · Part 2: Classification · Module 11*
