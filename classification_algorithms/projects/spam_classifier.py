"""
📧 SPAM EMAIL CLASSIFIER - Complete End-to-End Project
===============================================================

PROJECT OVERVIEW:
----------------
Build a real spam email classifier using multiple algorithms!
Apply everything you've learned: preprocessing, feature engineering,
model training, evaluation, and comparison.

LEARNING OBJECTIVES:
-------------------
1. Text preprocessing and feature extraction
2. Handling imbalanced classes (spam is rare)
3. Training multiple classification algorithms
4. Comparing model performance with ROC/AUC and Precision-Recall
5. Selecting optimal threshold for production
6. Building a complete ML pipeline

YOUTUBE RESOURCES:
-----------------
⭐ StatQuest: "Text Mining and Sentiment Analysis"
   Understanding text as data

📚 Krish Naik: "End-to-End Text Classification Project"
   https://www.youtube.com/watch?v=fiz1ORTBGpY
   Complete text classification pipeline

📚 sentdex: "Natural Language Processing"
   Text preprocessing techniques

TIME: 2-3 hours (comprehensive project!)
DIFFICULTY: Intermediate
PREREQUISITES: All classification algorithm modules

WHAT WE'LL BUILD:
----------------
1. Load and explore spam email dataset
2. Text preprocessing (cleaning, tokenization)
3. Feature extraction (Bag of Words, TF-IDF)
4. Train 4 models: Logistic Regression, KNN, Decision Tree, Random Forest
5. Compare using ROC curves, Precision-Recall curves
6. Select best model and optimal threshold
7. Final production-ready classifier
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import re

# Setup directories
PROJECT_DIR = Path(__file__).parent.parent
VISUAL_DIR = PROJECT_DIR / 'visuals' / 'spam_classifier'
DATA_DIR = PROJECT_DIR / 'data'

VISUAL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("📧 SPAM EMAIL CLASSIFIER - End-to-End Project")
print("=" * 80)
print()

# ============================================================================
# SECTION 1: DATASET CREATION AND EXPLORATION
# ============================================================================

print("=" * 80)
print("SECTION 1: Dataset and Exploration")
print("=" * 80)
print()

print("For this project, we'll create a sample spam dataset.")
print("In real projects, you'd use datasets like:")
print("   • Enron Email Dataset")
print("   • SpamAssassin Public Corpus")
print("   • SMS Spam Collection")
print()

# Create sample spam and ham (non-spam) emails
spam_emails = [
    "FREE! Click here to win a million dollars now! Limited time offer!!!",
    "Congratulations! You've won a free iPhone. Claim your prize today!",
    "Make money fast! Work from home. No experience needed. Apply now!",
    "URGENT: Your account has been compromised. Click here immediately.",
    "Get rich quick with this one simple trick! Doctors hate him!",
    "Lowest prices on medications! No prescription needed. Order now!",
    "You have been selected for a special promotion. Act fast!",
    "Hot singles in your area want to meet you! Click here!",
    "Increase your income by 500%! This is not a scam!",
    "Free credit report! Check your score now. Limited time!",
    "Winner! You've been selected for cash prize. Respond immediately.",
    "Lose weight fast with these pills! Amazing results guaranteed!",
    "Make thousands working from home! No investment required!",
    "Your package is waiting. Click here to claim your delivery.",
    "Urgent action required for your bank account. Update now!",
    "Exclusive offer just for you! Don't miss out on this deal!",
    "Get free samples of our amazing product! Order today!",
    "This is your last chance to win big! Enter now!",
    "Refinance your mortgage at unbeatable rates! Apply now!",
    "Work from home and earn big! Join thousands of successful people!",
    "Free trial! No credit card required! Sign up today!",
    "Amazing discount! 90% off! Limited stock available!",
    "You qualify for a large loan! Bad credit OK! Apply now!",
    "Miracle cure for all diseases! Buy now and save!",
    "Get paid to take surveys! Easy money! Start today!",
]

ham_emails = [
    "Hi John, can we schedule a meeting for next Tuesday?",
    "The project deadline has been moved to next Friday.",
    "Thanks for your email. I'll review the document and get back to you.",
    "Reminder: Team lunch tomorrow at noon in the conference room.",
    "Please find attached the quarterly report as requested.",
    "I'll be out of office next week. Please contact Sarah for urgent matters.",
    "Great work on the presentation! Let's discuss the feedback tomorrow.",
    "The client approved our proposal. Let's start planning the next phase.",
    "Can you send me the latest version of the budget spreadsheet?",
    "Happy birthday! Hope you have a wonderful day!",
    "Meeting notes from yesterday are now available on the shared drive.",
    "Please review the attached contract and let me know if you have questions.",
    "Your order has been shipped and will arrive in 3-5 business days.",
    "Thank you for your recent purchase. We hope you enjoy your product.",
    "Your subscription has been successfully renewed for another year.",
    "Here's the agenda for next week's conference. See you there!",
    "I've updated the project timeline based on our discussion.",
    "The new policy will take effect starting next month.",
    "Could you help me with this technical issue when you have time?",
    "Congratulations on your promotion! Well deserved!",
    "The training session has been rescheduled to Wednesday afternoon.",
    "Please complete the feedback survey before Friday.",
    "I've forwarded your question to the appropriate department.",
    "Your report was very thorough. Great attention to detail!",
    "Let me know when you're available for a quick call.",
]

# Combine and create labels
emails = spam_emails + ham_emails
labels = np.array([1] * len(spam_emails) + [0] * len(ham_emails))  # 1=spam, 0=ham

print(f"Total emails: {len(emails)}")
print(f"   Spam emails: {(labels == 1).sum()} ({(labels == 1).sum()/len(labels)*100:.1f}%)")
print(f"   Ham emails:  {(labels == 0).sum()} ({(labels == 0).sum()/len(labels)*100:.1f}%)")
print()

print("Sample spam email:")
print("-" * 70)
print(f'"{spam_emails[0]}"')
print()

print("Sample ham email:")
print("-" * 70)
print(f'"{ham_emails[0]}"')
print()

# ============================================================================
# SECTION 2: TEXT PREPROCESSING
# ============================================================================

print("=" * 80)
print("SECTION 2: Text Preprocessing")
print("=" * 80)
print()

print("TEXT PREPROCESSING STEPS:")
print()
print("1. LOWERCASE: Convert all text to lowercase")
print("   'FREE Money' → 'free money'")
print()
print("2. REMOVE PUNCTUATION: Keep only letters and spaces")
print("   'Click here!!!' → 'click here'")
print()
print("3. TOKENIZATION: Split into words")
print("   'free money fast' → ['free', 'money', 'fast']")
print()
print("4. REMOVE STOP WORDS: Remove common words")
print("   ['this', 'is', 'spam'] → ['spam']")
print()

# Simple stop words list
stop_words = set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
    'to', 'was', 'will', 'with', 'this', 'but', 'they', 'have', 'had',
    'what', 'when', 'where', 'who', 'which', 'why', 'how', 'all', 'each',
    'other', 'some', 'such', 'than', 'too', 'very', 'can', 'just', 'should'
])

def preprocess_text(text):
    """Preprocess a single email"""
    # Lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    words = text.split()

    # Remove stop words
    words = [w for w in words if w not in stop_words and len(w) > 2]

    return words

# Preprocess all emails
print("Preprocessing all emails...")
processed_emails = [preprocess_text(email) for email in emails]

print()
print("Example preprocessing:")
print("-" * 70)
original = emails[0]
processed = ' '.join(processed_emails[0])
print(f"Original:  {original[:60]}...")
print(f"Processed: {processed[:60]}...")
print()

# ============================================================================
# SECTION 3: FEATURE EXTRACTION - BAG OF WORDS
# ============================================================================

print("=" * 80)
print("SECTION 3: Feature Extraction - Bag of Words")
print("=" * 80)
print()

print("BAG OF WORDS MODEL:")
print("-" * 70)
print("Convert text to numbers!")
print()
print("1. Create VOCABULARY: All unique words across all emails")
print("2. For each email, COUNT how many times each word appears")
print("3. Each email becomes a VECTOR of word counts")
print()

# Build vocabulary
all_words = []
for email_words in processed_emails:
    all_words.extend(email_words)

word_counts = Counter(all_words)
print(f"Total words (with duplicates): {len(all_words)}")
print(f"Unique words (vocabulary): {len(word_counts)}")
print()

# Get top words for spam and ham
spam_words = []
ham_words = []

for i, email_words in enumerate(processed_emails):
    if labels[i] == 1:  # spam
        spam_words.extend(email_words)
    else:  # ham
        ham_words.extend(email_words)

spam_word_counts = Counter(spam_words)
ham_word_counts = Counter(ham_words)

print("Top 10 words in SPAM emails:")
print("-" * 70)
for word, count in spam_word_counts.most_common(10):
    print(f"   {word:<15} appears {count:>3} times")
print()

print("Top 10 words in HAM emails:")
print("-" * 70)
for word, count in ham_word_counts.most_common(10):
    print(f"   {word:<15} appears {count:>3} times")
print()

# Create vocabulary (limit to top words for simplicity)
vocab_size = 200
vocabulary = [word for word, count in word_counts.most_common(vocab_size)]
word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

print(f"Using vocabulary of top {vocab_size} words")
print()

# Convert emails to feature vectors
def email_to_vector(email_words, vocabulary, word_to_idx):
    """Convert email to bag-of-words vector"""
    vector = np.zeros(len(vocabulary))
    for word in email_words:
        if word in word_to_idx:
            vector[word_to_idx[word]] += 1
    return vector

print("Converting emails to feature vectors...")
X = np.array([email_to_vector(email_words, vocabulary, word_to_idx)
              for email_words in processed_emails])

print(f"Feature matrix shape: {X.shape}")
print(f"   {X.shape[0]} emails × {X.shape[1]} features")
print()

print("Example feature vector (first 10 features):")
print("-" * 70)
print(f"Email: {emails[0][:50]}...")
print(f"Vector: {X[0, :10]}")
print()

# ============================================================================
# SECTION 4: TRAIN-TEST SPLIT
# ============================================================================

print("=" * 80)
print("SECTION 4: Train-Test Split")
print("=" * 80)
print()

# Simple train-test split (70-30)
np.random.seed(42)
indices = np.random.permutation(len(X))
split_idx = int(0.7 * len(X))

train_indices = indices[:split_idx]
test_indices = indices[split_idx:]

X_train = X[train_indices]
y_train = labels[train_indices]
X_test = X[test_indices]
y_test = labels[test_indices]

print(f"Training set: {len(X_train)} emails")
print(f"   Spam: {(y_train == 1).sum()} ({(y_train == 1).sum()/len(y_train)*100:.1f}%)")
print(f"   Ham:  {(y_train == 0).sum()} ({(y_train == 0).sum()/len(y_train)*100:.1f}%)")
print()

print(f"Test set: {len(X_test)} emails")
print(f"   Spam: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
print(f"   Ham:  {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
print()

# ============================================================================
# SECTION 5: TRAIN MULTIPLE MODELS
# ============================================================================

print("=" * 80)
print("SECTION 5: Training Multiple Classification Models")
print("=" * 80)
print()

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, confusion_matrix, roc_curve, auc,
                                  precision_recall_curve, classification_report)

    print("Training 4 different models...")
    print()

    # Dictionary to store models and results
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    }

    results = {}

    print("-" * 80)
    print(f"{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)

    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Probabilities (for ROC curve)
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = y_test_pred  # KNN doesn't have predict_proba by default

        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)
        f1 = f1_score(y_test, y_test_pred)

        # Store results
        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba
        }

        print(f"{name:<20} {train_acc*100:<12.1f}% {test_acc*100:<12.1f}% {precision:<12.3f} {recall:<12.3f} {f1:<12.3f}")

    print()

    sklearn_available = True

except ImportError:
    print("⚠ Scikit-learn not available. Using manual implementation...")
    sklearn_available = False

# ============================================================================
# SECTION 6: MODEL COMPARISON
# ============================================================================

if sklearn_available:
    print("=" * 80)
    print("SECTION 6: Detailed Model Comparison")
    print("=" * 80)
    print()

    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['f1'])
    best_result = results[best_model_name]

    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   F1 Score: {best_result['f1']:.3f}")
    print(f"   Test Accuracy: {best_result['test_acc']*100:.1f}%")
    print(f"   Precision: {best_result['precision']:.3f}")
    print(f"   Recall: {best_result['recall']:.3f}")
    print()

    # Confusion matrix for best model
    cm = confusion_matrix(y_test, best_result['y_test_pred'])
    print("Confusion Matrix (Best Model):")
    print("-" * 70)
    print(f"                Predicted Ham    Predicted Spam")
    print(f"Actual Ham      {cm[0,0]:<15}  {cm[0,1]:<15}")
    print(f"Actual Spam     {cm[1,0]:<15}  {cm[1,1]:<15}")
    print()

    tn, fp, fn, tp = cm.ravel()
    print("Interpretation:")
    print(f"   ✓ True Negatives (TN):  {tn} ham emails correctly identified")
    print(f"   ✗ False Positives (FP): {fp} ham emails wrongly flagged as spam")
    print(f"   ✗ False Negatives (FN): {fn} spam emails missed")
    print(f"   ✓ True Positives (TP):  {tp} spam emails correctly caught")
    print()

    # Classification report
    print("Classification Report (Best Model):")
    print("-" * 70)
    print(classification_report(y_test, best_result['y_test_pred'],
                                target_names=['Ham', 'Spam']))

# ============================================================================
# SECTION 7: PRODUCTION DEPLOYMENT
# ============================================================================

if sklearn_available:
    print("=" * 80)
    print("SECTION 7: Production-Ready Classifier")
    print("=" * 80)
    print()

    print("Creating production classifier with optimal threshold...")
    print()

    # Use best model
    final_model = results[best_model_name]['model']

    def classify_email(email_text, model, vocabulary, word_to_idx, threshold=0.5):
        """
        Production-ready email classifier

        Args:
            email_text: Raw email text
            model: Trained model
            vocabulary: Word vocabulary
            word_to_idx: Word to index mapping
            threshold: Classification threshold (default 0.5)

        Returns:
            prediction: 0 (ham) or 1 (spam)
            confidence: Probability of spam
        """
        # Preprocess
        words = preprocess_text(email_text)

        # Convert to vector
        vector = email_to_vector(words, vocabulary, word_to_idx)
        vector = vector.reshape(1, -1)

        # Predict
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vector)[0, 1]
        else:
            proba = model.predict(vector)[0]

        prediction = 1 if proba >= threshold else 0

        return prediction, proba

    print("Testing production classifier on new emails...")
    print()

    # Test on some new examples
    test_emails = [
        "Hi, can we meet for lunch tomorrow?",
        "WINNER! You've won $1,000,000! Click here now!",
        "Please review the attached document and provide feedback.",
        "Make money fast! Work from home! No experience needed!"
    ]

    print("-" * 80)
    print(f"{'Email':<50} {'Prediction':<15} {'Confidence'}")
    print("-" * 80)

    for test_email in test_emails:
        pred, conf = classify_email(test_email, final_model, vocabulary, word_to_idx)
        pred_label = "SPAM 🚨" if pred == 1 else "HAM ✓"
        email_short = test_email[:47] + "..." if len(test_email) > 50 else test_email
        print(f"{email_short:<50} {pred_label:<15} {conf:.1%}")

    print()

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("=" * 80)
print("Creating Visualizations...")
print("=" * 80)
print()

# Visualization 1: Word frequency comparison
if sklearn_available:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Spam Classifier: Complete Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Top words in spam vs ham
    ax1 = axes[0, 0]

    spam_top = spam_word_counts.most_common(10)
    ham_top = ham_word_counts.most_common(10)

    y_pos = np.arange(10)

    ax1.barh(y_pos, [count for word, count in spam_top], alpha=0.7, color='red', label='Spam')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([word for word, count in spam_top])
    ax1.invert_yaxis()
    ax1.set_xlabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Top 10 Words in Spam Emails', fontsize=12, fontweight='bold', color='red')
    ax1.grid(axis='x', alpha=0.3)

    # Plot 2: Model comparison
    ax2 = axes[0, 1]

    model_names = list(results.keys())
    f1_scores = [results[name]['f1'] for name in model_names]
    colors = ['green' if name == best_model_name else 'lightblue' for name in model_names]

    bars = ax2.bar(range(len(model_names)), f1_scores, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.set_ylabel('F1 Score', fontsize=11, fontweight='bold')
    ax2.set_title('Model Performance Comparison (F1 Score)', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, 1.0])
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: ROC curves for all models
    ax3 = axes[1, 0]

    for name in results.keys():
        y_proba = results[name]['y_test_proba']
        if len(np.unique(y_proba)) > 2:  # Has probabilities
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            linestyle = '-' if name == best_model_name else '--'
            linewidth = 3 if name == best_model_name else 2
            ax3.plot(fpr, tpr, linestyle=linestyle, linewidth=linewidth,
                    label=f'{name} (AUC={roc_auc:.3f})')

    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax3.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax3.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax3.set_title('ROC Curves: All Models', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Confusion matrix heatmap for best model
    ax4 = axes[1, 1]

    cm = confusion_matrix(y_test, best_result['y_test_pred'])
    im = ax4.imshow(cm, interpolation='nearest', cmap='RdYlGn')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax4.text(j, i, cm[i, j],
                          ha="center", va="center", color="black",
                          fontsize=20, fontweight='bold')

    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Ham', 'Spam'], fontsize=11)
    ax4.set_yticklabels(['Ham', 'Spam'], fontsize=11)
    ax4.set_xlabel('Predicted', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Actual', fontsize=11, fontweight='bold')
    ax4.set_title(f'Confusion Matrix: {best_model_name}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/01_spam_classifier_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/01_spam_classifier_analysis.png")
    plt.close()

    # Visualization 2: Precision-Recall curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle('Precision-Recall Curves: Spam Detection', fontsize=14, fontweight='bold')

    for name in results.keys():
        y_proba = results[name]['y_test_proba']
        if len(np.unique(y_proba)) > 2:
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
            linestyle = '-' if name == best_model_name else '--'
            linewidth = 3 if name == best_model_name else 2
            ax.plot(recall_vals, precision_vals, linestyle=linestyle, linewidth=linewidth,
                   label=f'{name}')

    baseline = (y_test == 1).sum() / len(y_test)
    ax.axhline(baseline, color='r', linestyle='--', linewidth=2,
              label=f'Baseline ({baseline:.3f})', alpha=0.7)

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Important for Imbalanced Spam Detection', fontsize=12, pad=20)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])

    plt.tight_layout()
    plt.savefig(f'{VISUAL_DIR}/02_precision_recall_spam.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VISUAL_DIR}/02_precision_recall_spam.png")
    plt.close()

print()

# ============================================================================
# SUMMARY AND KEY TAKEAWAYS
# ============================================================================

print()
print("=" * 80)
print("📧 PROJECT SUMMARY: What You Built")
print("=" * 80)
print()

if sklearn_available:
    print("✓ COMPLETE SPAM CLASSIFIER PIPELINE:")
    print()
    print(f"   1. Dataset: {len(emails)} emails ({(labels==1).sum()} spam, {(labels==0).sum()} ham)")
    print(f"   2. Preprocessing: Cleaned, tokenized, removed stop words")
    print(f"   3. Features: Bag-of-Words with {vocab_size} word vocabulary")
    print(f"   4. Models Trained: 4 different algorithms")
    print(f"   5. Best Model: {best_model_name}")
    print(f"      • F1 Score: {best_result['f1']:.3f}")
    print(f"      • Precision: {best_result['precision']:.3f} (few false alarms)")
    print(f"      • Recall: {best_result['recall']:.3f} (catches spam)")
    print(f"   6. Production-ready classifier function created")
    print()

print("✓ KEY LEARNINGS:")
print()
print("   1. TEXT PREPROCESSING:")
print("      • Lowercase, remove punctuation")
print("      • Tokenization (split into words)")
print("      • Remove stop words")
print()
print("   2. FEATURE EXTRACTION:")
print("      • Bag-of-Words: Count word occurrences")
print("      • Creates numerical features from text")
print("      • Vocabulary limits feature dimensionality")
print()
print("   3. IMBALANCED CLASSES:")
print("      • Spam is often rare (like fraud, diseases)")
print("      • Use Precision-Recall curve, not just accuracy")
print("      • Consider cost of false positives vs false negatives")
print()
print("   4. MODEL SELECTION:")
print("      • Compare multiple models")
print("      • Use appropriate metrics (F1, Precision, Recall)")
print("      • Consider speed vs accuracy tradeoffs")
print()
print("   5. PRODUCTION CONSIDERATIONS:")
print("      • Threshold tuning (balance precision vs recall)")
print("      • False positives (blocking real emails) are costly")
print("      • False negatives (letting spam through) are annoying")
print()

print("✓ NEXT STEPS FOR IMPROVEMENT:")
print()
print("   1. TF-IDF instead of Bag-of-Words")
print("   2. N-grams (bigrams, trigrams)")
print("   3. More sophisticated preprocessing (stemming, lemmatization)")
print("   4. Larger, more diverse dataset")
print("   5. Deep learning (LSTM, BERT) for better performance")
print("   6. Feature engineering (email metadata, sender info)")
print("   7. Active learning (retrain with user feedback)")
print()

print("=" * 80)
print("📧 Spam Classifier Project Complete!")
print(f"   Check visualizations: {VISUAL_DIR}/")
print("=" * 80)
