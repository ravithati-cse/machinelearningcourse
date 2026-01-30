# 🤖 Machine Learning Course - From Zero to Hero

A comprehensive, visual-first machine learning course that teaches you from absolute basics (algebra) to building production-ready ML models. Perfect for beginners with no ML background!

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success)]()

## 🌟 What Makes This Course Special

### 1. **Visual-First Learning** 🎨
- **80+ auto-generated visualizations** - Every concept illustrated
- Infographics, mind maps, 3D plots, animations
- See it before you read it!

### 2. **Math from Absolute Scratch** 📐
- Start with basic algebra (y = mx + b)
- Build up to gradient descent and matrix operations
- No prerequisites - we teach everything!

### 3. **YouTube Integration** 📺
- Curated StatQuest videos embedded throughout
- 3Blue1Brown visual explanations
- Multiple teaching styles for different learners

### 4. **Complete Implementations** 💻
- Build algorithms from scratch (understand the math)
- Then use scikit-learn (production-ready)
- Full pipeline: data → model → evaluation → deployment

### 5. **Project-Based Learning** 🚀
- House price prediction
- Spam email detection
- Customer churn prediction
- Real datasets, real problems

---

## 📚 Course Structure

### 🔹 Part 1: Regression Algorithms (75% Complete)

**Learn to predict continuous values (prices, temperatures, sales)**

#### Week 1: Math Foundations (5/5 modules) ✅
- [Algebra Basics](regression_algorithms/math_foundations/01_algebra_basics.py) - Variables, slopes, y=mx+b
- [Statistics Fundamentals](regression_algorithms/math_foundations/02_statistics_fundamentals.py) - Mean, variance, correlation
- [Intro to Derivatives](regression_algorithms/math_foundations/03_intro_to_derivatives.py) - Gradient descent!
- [Linear Algebra Basics](regression_algorithms/math_foundations/04_linear_algebra_basics.py) - Vectors, matrices, dot products
- [Probability Basics](regression_algorithms/math_foundations/05_probability_basics.py) - Normal distribution, randomness

#### Week 2: Regression Algorithms (5/5 modules) ✅
- [Linear Regression Intro](regression_algorithms/algorithms/linear_regression_intro.py) - The foundation
- [Multiple Regression](regression_algorithms/algorithms/multiple_regression.py) - Using multiple features
- [Simple Examples](regression_algorithms/examples/simple_examples.py) - Practice problems
- [Data Exploration](regression_algorithms/examples/data_exploration.py) - EDA workflow
- Model Evaluation - Metrics deep dive (coming soon)

#### Week 3: Capstone Project (in progress)
- Housing Analysis - Complete EDA
- **House Price Prediction** - End-to-end ML pipeline

**Progress: 9/12 modules (75%)**

---

### 🔹 Part 2: Classification Algorithms (56% Complete)

**Learn to predict categories (spam/not spam, disease/healthy)**

#### Week 1: Math Foundations (5/5 modules) ✅
- [Sigmoid Function](classification_algorithms/math_foundations/01_sigmoid_function.py) - The key transformation
- [Probability for Classification](classification_algorithms/math_foundations/02_probability_for_classification.py) - Thresholds, odds, log-odds
- [Log Loss](classification_algorithms/math_foundations/03_log_loss.py) - Classification cost function
- [**Confusion Matrix**](classification_algorithms/math_foundations/04_confusion_matrix.py) - TP, FP, TN, FN - CRITICAL! 🌟
- [Decision Boundaries](classification_algorithms/math_foundations/05_decision_boundaries.py) - Visualizing classifiers

#### Week 2: Classification Algorithms (1/5 modules)
- [**Logistic Regression**](classification_algorithms/algorithms/logistic_regression_intro.py) - The main algorithm ✅
- KNN Classifier - Distance-based (coming soon)
- Decision Trees - Tree-based (coming soon)
- Random Forests - Ensemble methods (coming soon)
- Metrics Deep Dive - ROC, AUC (coming soon)

#### Week 3: Projects (in progress)
- Spam Email Classifier
- Customer Churn Prediction
- Model Comparison Project

**Progress: 9/16 modules (56%)**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Basic command line knowledge
- No ML or math background required!

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ravithati-cse/machinelearningcourse.git
cd machinelearningcourse
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run your first module**
```bash
cd regression_algorithms/math_foundations
python3 01_algebra_basics.py
```

4. **View the visualizations**
```bash
open ../visuals/01_algebra/
```

That's it! You're learning ML! 🎉

---

## 📖 Learning Paths

### Path 1: Complete Beginner (Recommended)
Start here if you're new to ML or need to refresh math concepts.

```bash
# Week 1: Math foundations (regression)
cd regression_algorithms/math_foundations
python3 01_algebra_basics.py
python3 02_statistics_fundamentals.py
python3 03_intro_to_derivatives.py
python3 04_linear_algebra_basics.py
python3 05_probability_basics.py

# Week 2: Linear regression
cd ../algorithms
python3 linear_regression_intro.py
python3 multiple_regression.py

# Week 3: Move to classification
cd ../../classification_algorithms/math_foundations
python3 01_sigmoid_function.py
# ... continue through all modules
```

### Path 2: Math-Comfortable Learner
Skip math foundations if you know calculus and linear algebra.

```bash
# Start with algorithms
cd regression_algorithms/algorithms
python3 linear_regression_intro.py

# Then classification
cd ../../classification_algorithms/algorithms
python3 logistic_regression_intro.py
```

### Path 3: Quick Overview
Just want to see what ML looks like?

```bash
# Run the main algorithms
python3 regression_algorithms/algorithms/linear_regression_intro.py
python3 classification_algorithms/algorithms/logistic_regression_intro.py
```

---

## 🎯 What You'll Learn

### Regression (Predicting Numbers)
- ✅ How linear regression works (from scratch!)
- ✅ Gradient descent optimization
- ✅ Multiple features and coefficients
- ✅ Model evaluation (R², RMSE, MAE)
- ✅ Real prediction on housing data

### Classification (Predicting Categories)
- ✅ Sigmoid function and probabilities
- ✅ Logistic regression (complete pipeline)
- ✅ Confusion matrix - foundation of all metrics
- ✅ Precision, Recall, F1 Score, Accuracy
- ✅ Decision boundaries and visualization
- 🚧 Tree-based methods (coming soon)
- 🚧 Ensemble methods (coming soon)

### Essential Math (No Prerequisites!)
- ✅ Algebra: y = mx + b
- ✅ Statistics: mean, variance, correlation
- ✅ Calculus: derivatives, gradient descent
- ✅ Linear Algebra: vectors, matrices, dot products
- ✅ Probability: distributions, randomness

---

## 📊 Course Features

| Feature | Description |
|---------|-------------|
| **Modules** | 18+ complete Python modules |
| **Visualizations** | 80+ auto-generated PNG files |
| **YouTube Videos** | 50+ curated video links |
| **Implementations** | From-scratch + scikit-learn |
| **Projects** | 3 real-world capstone projects |
| **Code Comments** | Extensive explanations |
| **Learning Time** | ~40-50 hours total |

---

## 🛠️ Technologies Used

- **Python 3.8+** - Programming language
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Visualizations (2D)
- **Seaborn** - Statistical visualizations
- **Scikit-learn** - ML library
- **SciPy** - Scientific computing

---

## 📂 Repository Structure

```
AdvancedMLCourse/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
│
├── regression_algorithms/             # Part 1: Regression
│   ├── README.md                      # Regression course guide
│   ├── math_foundations/              # 5 math modules
│   ├── algorithms/                    # 2 algorithm modules
│   ├── examples/                      # 3 example modules
│   ├── projects/                      # Capstone projects
│   ├── data/                          # Datasets
│   └── visuals/                       # Auto-generated plots
│
├── classification_algorithms/         # Part 2: Classification
│   ├── README.md                      # Classification course guide
│   ├── QUICK_START_GUIDE.md          # Quick start
│   ├── CLASSIFICATION_STATUS.md       # Progress tracking
│   ├── math_foundations/              # 5 math modules
│   ├── algorithms/                    # Algorithm modules
│   ├── projects/                      # Projects
│   └── visuals/                       # Auto-generated plots
│
└── COURSE_COMPLETION_GUIDE.md        # Full completion guide
```

---

## 🎓 Prerequisites and Requirements

### Required
- Python 3.8 or higher
- 4GB RAM minimum
- Text editor or IDE (VS Code recommended)

### Optional but Recommended
- Jupyter Notebook (for interactive exploration)
- Basic programming knowledge (any language)
- High school math (we teach from scratch, but it helps!)

### Not Required
- ❌ Machine Learning background
- ❌ Advanced math (calculus, linear algebra)
- ❌ Statistics knowledge
- ❌ GPU or powerful hardware

---

## 💡 Learning Tips

### 1. Follow the Order
Modules build on each other. Don't skip ahead!

### 2. Run Every Module
Don't just read - execute the code and see the visualizations.

### 3. Watch the Videos
The embedded YouTube links provide alternative explanations.

### 4. Take Notes
The visualizations are saved - use them for review!

### 5. Do the Math Manually
Before running code, try calculations by hand.

### 6. Experiment
Change parameters, try different data, break things!

### 7. Ask Questions
Open issues on GitHub if you're stuck.

---

## 📈 Success Stories

*Course launched January 2026 - success stories coming soon!*

Share your progress:
- Tag [@ravithati-cse](https://github.com/ravithati-cse)
- Use hashtag #MLFromZero
- Open a discussion to share what you built!

---

## 🤝 Contributing

This course is actively developed! Contributions welcome:

- 🐛 **Bug reports** - Found an error? Open an issue!
- 💡 **Suggestions** - Ideas for new modules? Let us know!
- 📝 **Documentation** - Improve explanations
- 🎨 **Visualizations** - Better plots and diagrams
- 🌍 **Translations** - Help make this global!

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines (coming soon).

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

**TL;DR**: You can use this course for anything - personal learning, teaching, commercial use. Just keep the license notice!

---

## 🙏 Acknowledgments

### Inspiration
- **StatQuest** by Josh Starmer - Best ML explanations on YouTube
- **3Blue1Brown** - Beautiful visual math
- **Andrew Ng** - ML course structure inspiration
- **scikit-learn** documentation and tutorials

### Educational Resources Referenced
- Khan Academy
- MIT OpenCourseWare
- Stanford CS229
- Fast.ai

---

## 📧 Contact

**Created by:** Ravi Thati
**GitHub:** [@ravithati-cse](https://github.com/ravithati-cse)
**Repository:** [machinelearningcourse](https://github.com/ravithati-cse/machinelearningcourse)

Questions? Open an issue or start a discussion!

---

## 🗺️ Roadmap

### Current Status (January 2026)
- ✅ Regression math foundations (100%)
- ✅ Regression algorithms (83%)
- ✅ Classification math foundations (100%)
- ✅ Logistic regression (100%)
- 🚧 Additional classification algorithms (in progress)
- 🚧 Capstone projects (in progress)

### Coming Soon
- 🔜 KNN Classifier
- 🔜 Decision Trees
- 🔜 Random Forests
- 🔜 Complete project templates
- 🔜 Jupyter notebook versions
- 🔜 Video walkthroughs

### Future Plans
- Neural Networks course
- Deep Learning fundamentals
- Computer Vision basics
- NLP introduction
- MLOps and deployment

---

## ⭐ Star This Repository!

If this course helps you learn ML, please star ⭐ the repository!

It helps others discover this resource and motivates continued development.

---

## 📊 Course Statistics

```
Total Modules: 18 complete, 7 in progress
Total Lines of Code: ~15,000+
Total Visualizations: 80+
YouTube Videos Curated: 50+
Math Concepts Covered: 30+
ML Algorithms: 3 complete, 4 in progress
Learning Hours: 40-50 hours
```

---

## 🎉 Get Started Now!

```bash
# Clone and start learning
git clone https://github.com/ravithati-cse/machinelearningcourse.git
cd machinelearningcourse
pip install -r requirements.txt
cd regression_algorithms/math_foundations
python3 01_algebra_basics.py
```

**Welcome to your Machine Learning journey!** 🚀

---

*Built with ❤️ for aspiring ML engineers and data scientists*

*Last updated: January 30, 2026*
