"""
📊 CUSTOMER CHURN PREDICTION - Business ML Application
===============================================================

PROJECT OVERVIEW:
----------------
Predict which customers are likely to leave (churn) a service!
A critical business application of classification with real-world impact.

LEARNING OBJECTIVES:
-------------------
1. Understanding churn and business metrics
2. Handling imbalanced datasets (churn is usually rare)
3. Feature importance for business insights
4. Optimizing for business value (cost-sensitive learning)
5. Threshold tuning for different business scenarios
6. Building actionable recommendations

YOUTUBE RESOURCES:
-----------------
⭐ Krish Naik: "Customer Churn Prediction End to End"
   https://www.youtube.com/watch?v=O3TvFzD7uPw
   Complete project walkthrough

📚 Data Professor: "Churn Prediction Machine Learning Project"
   Business-focused ML application

📚 Ken Jee: "Data Science for Business"
   Translating ML to business value

TIME: 2-3 hours
DIFFICULTY: Intermediate
PREREQUISITES: All classification algorithm modules

BUSINESS CONTEXT:
----------------
Losing customers (churn) is expensive!
- Acquiring new customer: $500
- Retention campaign: $50
- Lifetime value of retained customer: $2000

Goal: Identify high-risk customers for retention campaigns
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup directories
PROJECT_DIR = Path(__file__).parent.parent
VISUAL_DIR = PROJECT_DIR / 'visuals' / 'churn_prediction'
DATA_DIR = PROJECT_DIR / 'data'

VISUAL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("📊 CUSTOMER CHURN PREDICTION PROJECT")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: BUSINESS CONTEXT
# ============================================================================

print("=" * 80)
print("SECTION 1: Understanding the Business Problem")
print("=" * 80)
print()

print("WHAT IS CHURN?")
print("-" * 70)
print("   Churn = Customer leaves/cancels service")
print()
print("   Examples:")
print("      • Cancels subscription (Netflix, gym)")
print("      • Switches to competitor (telecom, bank)")
print("      • Stops using product (app uninstall)")
print()

print("WHY PREDICT CHURN?")
print("-" * 70)
print("   Cost of acquiring new customer >> Cost of retention")
print()
print("   Example Economics:")
print("      • Customer Acquisition Cost: $500")
print("      • Retention Campaign Cost: $50")
print("      • Customer Lifetime Value: $2,000")
print()
print("   If we can predict churn, we can:")
print("      1. Target high-risk customers with retention offers")
print("      2. Save money (retention cheaper than acquisition)")
print("      3. Improve customer satisfaction")
print()

print("BUSINESS METRICS:")
print("-" * 70)
print("   • Churn Rate: % of customers who left")
print("   • Retention Rate: % of customers who stayed")
print("   • Customer Lifetime Value (CLV): Total revenue per customer")
print("   • Cost per Save: Cost of retention / customers retained")
print()

# ============================================================================
# SECTION 2: DATASET CREATION
# ============================================================================

print("=" * 80)
print("SECTION 2: Dataset and Features")
print("=" * 80)
print()

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (confusion_matrix, classification_report,
                                  roc_curve, auc, precision_recall_curve,
                                  accuracy_score, precision_score, recall_score, f1_score)

    print("Creating sample telecom churn dataset...")
    print()

    # Create realistic customer data
    np.random.seed(42)
    n_customers = 1000

    # Features
    tenure = np.random.randint(1, 73, n_customers)  # months with company
    monthly_charges = np.random.uniform(20, 120, n_customers)
    total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_customers)
    contract_type = np.random.choice([0, 1, 2], n_customers, p=[0.5, 0.3, 0.2])  # 0=month-to-month, 1=1yr, 2=2yr
    has_internet = np.random.choice([0, 1], n_customers, p=[0.2, 0.8])
    num_services = np.random.randint(0, 6, n_customers)  # number of services subscribed
    customer_service_calls = np.random.poisson(2, n_customers)  # support calls

    # Create target (churn) with realistic relationships
    churn_prob = (
        0.5 * (tenure < 12) +  # New customers more likely to churn
        0.3 * (contract_type == 0) +  # Month-to-month more likely
        0.2 * (monthly_charges > 80) +  # Higher charges increase churn
        0.15 * (customer_service_calls > 3) +  # Many support calls = unhappy
        -0.2 * (num_services > 3)  # More services = more sticky
    )

    churn_prob = 1 / (1 + np.exp(-churn_prob + 2))  # Sigmoid transformation
    churned = (np.random.random(n_customers) < churn_prob).astype(int)

    # Create DataFrame
    df = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_type,
        'has_internet': has_internet,
        'num_services': num_services,
        'customer_service_calls': customer_service_calls,
        'churned': churned
    })

    print("DATASET OVERVIEW:")
    print("-" * 70)
    print(f"Total customers: {len(df):,}")
    print(f"   Churned: {(df['churned']==1).sum()} ({(df['churned']==1).sum()/len(df)*100:.1f}%)")
    print(f"   Stayed:  {(df['churned']==0).sum()} ({(df['churned']==0).sum()/len(df)*100:.1f}%)")
    print()

    print("FEATURES:")
    print("-" * 70)
    print("   • tenure: Months with company")
    print("   • monthly_charges: Monthly bill amount")
    print("   • total_charges: Total amount paid")
    print("   • contract_type: 0=month-to-month, 1=1yr, 2=2yr")
    print("   • has_internet: 0=No, 1=Yes")
    print("   • num_services: Number of services subscribed")
    print("   • customer_service_calls: Number of support calls")
    print()

    print("SAMPLE DATA (first 5 customers):")
    print("-" * 70)
    print(df.head().to_string())
    print()

    sklearn_available = True

except ImportError:
    print("⚠ Scikit-learn/pandas not available")
    sklearn_available = False
    exit(1)

# ============================================================================
# SECTION 3: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("=" * 80)
print("SECTION 3: Exploratory Data Analysis")
print("=" * 80)
print()

print("CHURN RATE BY TENURE:")
print("-" * 70)
new_customers = df[df['tenure'] < 12]
old_customers = df[df['tenure'] >= 24]

print(f"New customers (< 12 months):")
print(f"   Churn rate: {(new_customers['churned']==1).sum()/len(new_customers)*100:.1f}%")
print()
print(f"Loyal customers (≥ 24 months):")
print(f"   Churn rate: {(old_customers['churned']==1).sum()/len(old_customers)*100:.1f}%")
print()

print("CHURN RATE BY CONTRACT TYPE:")
print("-" * 70)
for contract in [0, 1, 2]:
    subset = df[df['contract_type'] == contract]
    churn_rate = (subset['churned']==1).sum() / len(subset) * 100
    contract_name = ['Month-to-Month', '1-Year', '2-Year'][contract]
    print(f"   {contract_name:<15}: {churn_rate:>5.1f}% churn rate")
print()

print("CHURN RATE BY SERVICE USAGE:")
print("-" * 70)
high_service = df[df['num_services'] >= 4]
low_service = df[df['num_services'] < 2]

print(f"High engagement (≥4 services):")
print(f"   Churn rate: {(high_service['churned']==1).sum()/len(high_service)*100:.1f}%")
print()
print(f"Low engagement (<2 services):")
print(f"   Churn rate: {(low_service['churned']==1).sum()/len(low_service)*100:.1f}%")
print()

# ============================================================================
# SECTION 4: FEATURE ENGINEERING
# ============================================================================

print("=" * 80)
print("SECTION 4: Feature Engineering")
print("=" * 80)
print()

print("Creating new features for better predictions...")
print()

# Create new features
df['avg_charges_per_month'] = df['total_charges'] / df['tenure']
df['is_new_customer'] = (df['tenure'] < 12).astype(int)
df['has_issues'] = (df['customer_service_calls'] > 3).astype(int)
df['service_engagement'] = df['num_services'] / 6  # Normalize to 0-1

print("NEW FEATURES:")
print("-" * 70)
print("   1. avg_charges_per_month = total_charges / tenure")
print("   2. is_new_customer = 1 if tenure < 12 months")
print("   3. has_issues = 1 if customer_service_calls > 3")
print("   4. service_engagement = num_services / 6 (normalized)")
print()

# ============================================================================
# SECTION 5: TRAIN-TEST SPLIT
# ============================================================================

print("=" * 80)
print("SECTION 5: Train-Test Split")
print("=" * 80)
print()

# Separate features and target
X = df.drop('churned', axis=1)
y = df['churned']

# Split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} customers")
print(f"   Churned: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
print(f"   Stayed:  {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%)")
print()

print(f"Test set: {len(X_test)} customers")
print(f"   Churned: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.1f}%)")
print(f"   Stayed:  {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.1f}%)")
print()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ Features scaled (StandardScaler)")
print()

# ============================================================================
# SECTION 6: TRAIN MODELS
# ============================================================================

print("=" * 80)
print("SECTION 6: Training Classification Models")
print("=" * 80)
print()

print("Training 2 models for churn prediction...")
print()

# Train models
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_train)

# Predictions
lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]

rf_pred = rf.predict(X_test_scaled)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]

# Metrics
print("-" * 80)
print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 80)

for name, y_pred in [('Logistic Regression', lr_pred), ('Random Forest', rf_pred)]:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name:<20} {acc:<12.3f} {prec:<12.3f} {rec:<12.3f} {f1:<12.3f}")

print()

# ============================================================================
# SECTION 7: BUSINESS-FOCUSED EVALUATION
# ============================================================================

print("=" * 80)
print("SECTION 7: Business-Focused Evaluation")
print("=" * 80)
print()

print("CONFUSION MATRIX (Random Forest):")
print("-" * 70)
cm = confusion_matrix(y_test, rf_pred)
print(f"                Predicted Stay    Predicted Churn")
print(f"Actually Stayed {cm[0,0]:<16}  {cm[0,1]:<16}")
print(f"Actually Churned {cm[1,0]:<16}  {cm[1,1]:<16}")
print()

tn, fp, fn, tp = cm.ravel()

print("BUSINESS INTERPRETATION:")
print("-" * 70)
print(f"   True Negatives (TN): {tn}")
print(f"      → Correctly predicted staying - no action needed ✓")
print()
print(f"   False Positives (FP): {fp}")
print(f"      → Predicted churn but stayed - wasted retention offer")
print(f"      → Cost: {fp} × $50 = ${fp * 50}")
print()
print(f"   False Negatives (FN): {fn}")
print(f"      → Predicted stay but churned - MISSED OPPORTUNITY!")
print(f"      → Cost: {fn} × $2000 (lost CLV) = ${fn * 2000}")
print()
print(f"   True Positives (TP): {tp}")
print(f"      → Correctly predicted churn - can intervene ✓")
print(f"      → Savings: {tp} × ($2000 - $50) = ${tp * 1950}")
print()

# Calculate ROI
retention_cost = (tp + fp) * 50  # Cost of targeting
saved_revenue = tp * 2000  # Customers saved
lost_revenue = fn * 2000  # Customers lost
net_benefit = saved_revenue - retention_cost - lost_revenue

print("FINANCIAL IMPACT:")
print("-" * 70)
print(f"   Retention campaigns sent: {tp + fp}")
print(f"   Total campaign cost: ${retention_cost}")
print(f"   Revenue saved: ${saved_revenue}")
print(f"   Revenue lost (missed): ${lost_revenue}")
print(f"   Net benefit: ${net_benefit}")
print()

if net_benefit > 0:
    print(f"   ✓ POSITIVE ROI: Model saves ${net_benefit}")
else:
    print(f"   ✗ NEGATIVE ROI: Model costs ${abs(net_benefit)}")
print()

# ============================================================================
# SECTION 8: THRESHOLD OPTIMIZATION
# ============================================================================

print("=" * 80)
print("SECTION 8: Optimizing for Business Value")
print("=" * 80)
print()

print("Default threshold = 0.5, but we can optimize for business value!")
print()

# Try different thresholds
thresholds = np.linspace(0.1, 0.9, 9)

print("-" * 80)
print(f"{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'Cost':<12} {'Saved':<12} {'Net Benefit'}")
print("-" * 80)

best_net_benefit = -float('inf')
best_threshold = 0.5

for threshold in thresholds:
    y_pred_custom = (rf_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_custom)
    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    cost = (tp + fp) * 50
    saved = tp * 2000
    net = saved - cost - (fn * 2000)

    print(f"{threshold:<12.2f} {recall:<10.3f} {precision:<12.3f} ${cost:<11} ${saved:<11} ${net}")

    if net > best_net_benefit:
        best_net_benefit = net
        best_threshold = threshold

print()
print(f"🏆 OPTIMAL THRESHOLD: {best_threshold:.2f}")
print(f"   Net benefit: ${best_net_benefit}")
print()

print("BUSINESS RECOMMENDATION:")
print("-" * 70)
if best_threshold < 0.5:
    print(f"   Use LOWER threshold ({best_threshold:.2f}):")
    print("   → Cast wider net (higher recall)")
    print("   → Better to save more customers, even with false positives")
    print("   → False negatives are very expensive!")
elif best_threshold > 0.5:
    print(f"   Use HIGHER threshold ({best_threshold:.2f}):")
    print("   → Be more selective (higher precision)")
    print("   → Focus retention efforts on high-confidence cases")
else:
    print("   Use default threshold (0.5)")
print()

# ============================================================================
# SECTION 9: FEATURE IMPORTANCE
# ============================================================================

print("=" * 80)
print("SECTION 9: What Drives Churn?")
print("=" * 80)
print()

# Feature importance from Random Forest
importances = rf.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

print("FEATURE IMPORTANCE:")
print("-" * 70)
print(f"{'Rank':<6} {'Feature':<25} {'Importance':<12} {'Business Insight'}")
print("-" * 70)

insights = {
    'tenure': 'Tenure matters - retain early!',
    'contract_type': 'Lock in with contracts',
    'monthly_charges': 'Price sensitivity',
    'is_new_customer': 'First year is critical',
    'customer_service_calls': 'Service quality impact',
    'num_services': 'Cross-sell to increase stickiness',
    'service_engagement': 'Engagement drives retention'
}

for i, idx in enumerate(indices[:7], 1):
    feature = feature_names[idx]
    importance = importances[idx]
    insight = insights.get(feature, 'Review this factor')
    print(f"{i:<6} {feature:<25} {importance:<11.4f} {insight}")

print()

print("ACTIONABLE RECOMMENDATIONS:")
print("-" * 70)
print("   1. EARLY INTERVENTION:")
print("      → Target new customers (<12 months) with onboarding")
print()
print("   2. CONTRACT INCENTIVES:")
print("      → Offer discounts for annual contracts")
print("      → Convert month-to-month to committed plans")
print()
print("   3. SERVICE BUNDLING:")
print("      → Cross-sell additional services")
print("      → Customers with more services less likely to churn")
print()
print("   4. PROACTIVE SUPPORT:")
print("      → Reach out after multiple support calls")
print("      → High support calls indicate dissatisfaction")
print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: Comprehensive analysis
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Customer Churn Prediction: Complete Analysis', fontsize=16, fontweight='bold')

# Plot 1: ROC Curve
ax1 = axes[0, 0]
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_rf = auc(fpr_rf, tpr_rf)

ax1.plot(fpr_lr, tpr_lr, 'b-', linewidth=2, label=f'Logistic Reg (AUC={roc_auc_lr:.3f})')
ax1.plot(fpr_rf, tpr_rf, 'g-', linewidth=2, label=f'Random Forest (AUC={roc_auc_rf:.3f})')
ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
ax1.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax1.set_ylabel('True Positive Rate (Recall)', fontsize=11, fontweight='bold')
ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Feature Importance
ax2 = axes[0, 1]
top_n = 7
top_indices = indices[:top_n]
ax2.barh(range(top_n), importances[top_indices], color='steelblue', edgecolor='black')
ax2.set_yticks(range(top_n))
ax2.set_yticklabels([feature_names[i] for i in top_indices], fontsize=10)
ax2.invert_yaxis()
ax2.set_xlabel('Importance', fontsize=11, fontweight='bold')
ax2.set_title('Top Features Driving Churn', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Threshold vs Net Benefit
ax3 = axes[1, 0]
net_benefits = []
for threshold in np.linspace(0.1, 0.9, 50):
    y_pred_custom = (rf_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_custom)
    tn, fp, fn, tp = cm.ravel()
    net = (tp * 2000) - ((tp + fp) * 50) - (fn * 2000)
    net_benefits.append(net)

ax3.plot(np.linspace(0.1, 0.9, 50), net_benefits, 'b-', linewidth=2)
ax3.axvline(best_threshold, color='r', linestyle='--', linewidth=2,
           label=f'Optimal: {best_threshold:.2f}')
ax3.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
ax3.set_xlabel('Classification Threshold', fontsize=11, fontweight='bold')
ax3.set_ylabel('Net Benefit ($)', fontsize=11, fontweight='bold')
ax3.set_title('Optimizing for Business Value', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Confusion Matrix Heatmap
ax4 = axes[1, 1]
y_pred_optimal = (rf_proba >= best_threshold).astype(int)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
im = ax4.imshow(cm_optimal, cmap='RdYlGn', interpolation='nearest')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax4.text(j, i, cm_optimal[i, j],
                       ha="center", va="center", color="black",
                       fontsize=20, fontweight='bold')

ax4.set_xticks([0, 1])
ax4.set_yticks([0, 1])
ax4.set_xticklabels(['Predicted Stay', 'Predicted Churn'], fontsize=10)
ax4.set_yticklabels(['Actually Stayed', 'Actually Churned'], fontsize=10)
ax4.set_xlabel('Predicted', fontsize=11, fontweight='bold')
ax4.set_ylabel('Actual', fontsize=11, fontweight='bold')
ax4.set_title(f'Confusion Matrix (Threshold={best_threshold:.2f})', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{VISUAL_DIR}/01_churn_analysis.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VISUAL_DIR}/01_churn_analysis.png")
plt.close()

print()

# ============================================================================
# PROJECT SUMMARY
# ============================================================================

print()
print("=" * 80)
print("📊 PROJECT COMPLETE: Key Takeaways")
print("=" * 80)
print()

print("✓ BUSINESS PROBLEM SOLVED:")
print(f"   • Trained models to predict customer churn")
print(f"   • Optimized for business value, not just accuracy")
print(f"   • Net benefit: ${best_net_benefit}")
print()

print("✓ KEY INSIGHTS:")
print()
print("   1. EARLY CUSTOMERS AT RISK:")
print("      • First 12 months are critical")
print("      • Implement strong onboarding programs")
print()

print("   2. CONTRACT TYPE MATTERS:")
print("      • Month-to-month contracts have highest churn")
print("      • Incentivize annual/multi-year commitments")
print()

print("   3. ENGAGEMENT IS PROTECTIVE:")
print("      • More services = lower churn")
print("      • Cross-selling improves retention")
print()

print("   4. SERVICE QUALITY SIGNALS:")
print("      • Multiple support calls predict churn")
print("      • Proactive outreach after 3+ calls")
print()

print("✓ MODEL PERFORMANCE:")
print(f"   • Best model: Random Forest")
print(f"   • Optimal threshold: {best_threshold:.2f} (not default 0.5!)")
print(f"   • ROC AUC: {roc_auc_rf:.3f}")
print()

print("✓ ACTIONABLE RECOMMENDATIONS:")
print()
print("   1. Deploy model to score customers monthly")
print("   2. Target high-risk customers (score > threshold)")
print("   3. Personalize retention based on risk factors")
print("   4. Monitor model performance and retrain quarterly")
print()

print("=" * 80)
print("📊 Churn Prediction Project Complete!")
print(f"   Visualizations: {VISUAL_DIR}/")
print("=" * 80)
