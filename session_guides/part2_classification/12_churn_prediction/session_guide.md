# 🎓 MLForBeginners — Instructor Guide
## Part 2 · Module 12: Churn Prediction (Project)
### Single 120-Minute Session

> **Prerequisites:** Module 11 (Spam Classifier). They've now built a full
> text classification pipeline and understand imbalanced classes.
> **Payoff:** Predict which customers will leave — and take action before they do.

---

# SESSION (120 min)
## "Predict customer churn — and save the business"

## Before They Arrive
- Terminal open in `classification_algorithms/projects/`
- Think of a subscription service you've cancelled (Netflix, gym, etc.)

---

## OPENING (10 min)

> *"Has anyone ever cancelled a subscription — Netflix, Spotify, a gym?*
> *What made you cancel?*
>
> *Now imagine you're the CEO of that company.*
> *You can't call every customer every day.*
> *But what if you could PREDICT who's about to leave —
> with enough time to offer them a discount or fix their problem?*
>
> *That's churn prediction. And it's one of the highest-ROI
> ML applications in business. Every subscription company uses it."*

Write on board:
```
CHURN = customer cancels / stops using your service

COST OF LOSING A CUSTOMER:
  Acquisition cost: $200
  Lifetime value:   $800
  Loss if they leave: $600

If we can prevent 100 churns/month → $60,000/month saved
That pays for the entire ML team.
```

---

## SECTION 1: The Dataset & Business Context (10 min)

```bash
python3 churn_prediction.py
```

While it loads:

> *"Our dataset has telecom customers. For each customer we know:
> tenure (months), contract type, monthly charges, services used, etc.*
>
> *The target: did they churn? Yes or No.*
>
> *Class balance guess — what % of customers churn?"*

Typical answer: ~15-25%.

> *"Same imbalance problem as spam. But now the stakes are business dollars,
> not just annoying emails. The cost of errors is real money."*

---

## SECTION 2: Imbalanced Classes — The Right Strategy (20 min)

Write on board:
```
NAIVE APPROACH:
  Train on imbalanced data → model just predicts "no churn" always
  Accuracy = 80%+  ← looks great! But catches 0% of churners

BETTER APPROACHES:
  1. Class weights: penalize missing a churner MORE
     class_weight = {0: 1, 1: 5}  ← churner errors cost 5×

  2. SMOTE: oversample minority class
     Synthesize new "churn" examples

  3. Threshold tuning: lower decision threshold
     Predict churn if P(churn) > 0.3 instead of 0.5
```

> *"Which one you use depends on business context.*
>
> *If calling churners costs $10/call, but saving one is worth $500,
> you can afford many false positives.*
>
> *If you only have 10 retention agents and 10,000 customers,
> precision matters more — only call the likeliest churners."*

**Ask the room:** *"What would YOU do if you were the CEO — cast a wide net or be precise?"*

---

## SECTION 3: Feature Importance — What Drives Churn? (20 min)

Open the feature importance visualization.

> *"This is where ML becomes business insight.*
>
> *What's the #1 predictor of churn?*
> *(Usually: contract type — month-to-month customers churn 5× more than annual contract customers)*
>
> *What else matters? Tenure — new customers are most vulnerable.*
> *Monthly charges — customers paying above average feel the pinch.*
> *No tech support — customers who can't get help leave faster."*

Write the business actions:
```
HIGH CHURN RISK INDICATORS → BUSINESS RESPONSE
─────────────────────────────────────────────────
Month-to-month contract   → Offer annual discount
Low tenure (< 6 months)   → Assign onboarding agent
High charges              → Proactively offer plan review
No tech support add-on    → Free 3-month support trial
```

> *"This is the real power of ML in business.*
> *Not just 'this customer will churn' — but WHY,
> so you can take targeted action."*

---

## SECTION 4: Cost-Sensitive Evaluation (20 min)

```python
# Define costs
cost_false_negative = 600  # missed churner → lose $600 lifetime value
cost_false_positive =  10  # unnecessary retention call → costs $10

# At default threshold 0.5:
# At optimized threshold:
```

> *"Standard accuracy doesn't capture this.*
> *A model that's 95% accurate but misses every churner is worthless.*
>
> *We define a business cost matrix:
> Missing a churner = $600 loss
> Unnecessary retention call = $10 cost*
>
> *The optimal threshold minimizes TOTAL BUSINESS COST, not accuracy."*

Run the threshold optimization. Show the cost curve.

> *"See how the total cost drops as we lower the threshold to ~0.3?
> We catch more churners (fewer $600 losses)
> even though we make more unnecessary calls (more $10 costs).*
>
> *At scale: lowering threshold from 0.5 to 0.3 might save $200,000/month.
> That's a business decision driven by your ML model."*

---

## SECTION 5: Actionable Output (15 min)

Show the final output — the ranked customer list with churn probability:

> *"The output your retention team actually uses:
> Customer ID, churn probability, top risk factors, recommended action.*
>
> *This is what it means to deploy ML:
> not just a model in a notebook,
> but something that changes what the business does tomorrow."*

---

## CLOSING (10 min)

Board summary:
```
CHURN PREDICTION PIPELINE:
  Data → Features → Imbalance handling → Train → Threshold tune → Action

KEY LESSONS:
  • Accuracy is a misleading metric for imbalanced problems
  • Cost matrices translate ML metrics into business dollars
  • Feature importance drives targeted interventions
  • The output must be actionable by humans, not just numbers
```

**Homework:**
> *"Think of another business where churn prediction matters.*
> *What features would you collect? What action would you take?*
> *Healthcare: patient no-shows. SaaS: trial-to-paid conversion.*
> *Write down 5 features and 3 business actions."*

---

## INSTRUCTOR TIPS

**"How is this different from spam?"**
> *"Spam: binary signal in text data. Fast predictions on new items.*
> *Churn: structured/tabular data, predictions made periodically.*
> *Very different domains, but the same classification framework."*

**"Can we prevent all churn?"**
> *"No — some churn is natural (people move, circumstances change).*
> *The goal is to identify 'saveable' churn — customers who leave*
> *due to a problem you CAN fix. That's the addressable market."*

---

## Quick Reference
```
Single Session (120 min)
├── Opening + business context  10 min
├── Dataset + imbalance         10 min
├── Class balance strategies    20 min
├── Feature importance          20 min
├── Cost-sensitive evaluation   20 min
├── Actionable output           15 min
└── Close + homework            25 min
```

---
*MLForBeginners · Part 2: Classification · Module 12*
