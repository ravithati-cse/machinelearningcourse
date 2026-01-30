# 🧪 Practice Lab: Confusion Matrix

**The foundation of ALL classification metrics!** Master this!

---

## 🎯 Quick Win: Build the Matrix (10 min)

```python
import numpy as np

# Medical test results
actual =    np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])  # 1=sick, 0=healthy
predicted = np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0])

# Count each category!
TP = np.sum((actual == 1) & (predicted == 1))  # Sick, predicted sick
TN = np.sum((actual == 0) & (predicted == 0))  # Healthy, predicted healthy
FP = np.sum((actual == 0) & (predicted == 1))  # Healthy, predicted sick (false alarm)
FN = np.sum((actual == 1) & (predicted == 0))  # Sick, predicted healthy (MISSED!)

print("🎯 CONFUSION MATRIX")
print("=" * 35)
print(f"              Predicted")
print(f"              Neg   Pos")
print(f"Actual  Neg   {TN}     {FP}")
print(f"        Pos   {FN}     {TP}")
print("=" * 35)

# 🤔 In medical terms:
# TP = Correctly caught __ sick patients
# FN = MISSED __ sick patients (dangerous!)
# FP = Gave false alarms to __ healthy patients
# TN = Correctly cleared __ healthy patients
```

---

## 📊 Calculate ALL Metrics (15 min)

```python
# Using your TP, TN, FP, FN from above:
TP, TN, FP, FN = 2, 5, 1, 2  # Fill in your values!

# Calculate each metric BY HAND first!

# Accuracy = (TP + TN) / Total
accuracy = ???

# Precision = TP / (TP + FP) - "Of predicted positive, how many correct?"
precision = ???

# Recall = TP / (TP + FN) - "Of actual positive, how many caught?"
recall = ???

# F1 = 2 × (Precision × Recall) / (Precision + Recall)
f1 = ???

print("📊 METRICS:")
print(f"  Accuracy:  {accuracy:.1%}")
print(f"  Precision: {precision:.1%}")
print(f"  Recall:    {recall:.1%}")
print(f"  F1 Score:  {f1:.3f}")

# Verify with sklearn!
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
actual = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
predicted = [1, 1, 0, 0, 0, 0, 0, 1, 0, 0]

print("\n✅ Sklearn verification:")
print(f"  Accuracy: {accuracy_score(actual, predicted):.1%}")
print(f"  Precision: {precision_score(actual, predicted):.1%}")
print(f"  Recall: {recall_score(actual, predicted):.1%}")
```

---

## 🏆 Boss Challenge: Which Metric Matters? (10 min)

Different problems need different metrics!

```python
# Scenario A: Cancer Detection
# Missing cancer (FN) could be fatal!
# False alarm (FP) just means more tests
# → Which metric matters most? ___________

# Scenario B: Email Spam Filter
# Missing spam (FN) is annoying
# Blocking good email (FP) could lose important messages!
# → Which metric matters most? ___________

# Scenario C: Credit Card Fraud
# Missing fraud (FN) = customer loses money
# False alarm (FP) = annoying card decline
# → Which metric matters most? ___________

# ANSWERS:
# A: RECALL (catch all cancers, even if some false positives)
# B: PRECISION (don't block good emails!)
# C: RECALL (catch all fraud, but also consider cost)
```

---

## 🎯 The Accuracy Trap

```python
# Imbalanced dataset: 95 healthy, 5 sick
actual = [0]*95 + [1]*5  # 95 healthy, 5 sick

# Model A: Predicts everyone as healthy (lazy!)
pred_a = [0]*100

# Model B: Actually tries to find sick patients
pred_b = [0]*93 + [1]*7  # Says 7 are sick

from sklearn.metrics import accuracy_score, recall_score

print("Model A (predicts all healthy):")
print(f"  Accuracy: {accuracy_score(actual, pred_a):.0%}")
print(f"  Recall: {recall_score(actual, pred_a):.0%}")

print("\nModel B (tries to find sick):")
print(f"  Accuracy: {accuracy_score(actual, pred_b):.0%}")
print(f"  Recall: {recall_score(actual, pred_b):.1%}")

# 🤔 Model A has 95% accuracy but catches 0 sick patients!
# NEVER trust accuracy alone with imbalanced data!
```

---

## ✅ You're Ready When...

- [ ] You can build a confusion matrix from scratch
- [ ] You know TP, TN, FP, FN by heart
- [ ] You can calculate precision, recall, accuracy, F1
- [ ] You know WHICH metric to use for different problems

**This is CRITICAL knowledge for ML interviews!** 💼

**Next up:** Decision Boundaries! 🎨
