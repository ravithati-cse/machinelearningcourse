# 🧠 Part 3: Deep Neural Networks

**Learn to build neural networks from a single neuron all the way to production-ready Keras models.**

---

## 📚 Module Map

### Week 1: Math Foundations (5/5 modules) ✅

| # | File | Topic | Time |
|---|------|--------|------|
| 1 | [01_neurons_and_activations.py](math_foundations/01_neurons_and_activations.py) | Neurons, weights, biases, 6 activation functions | 45 min |
| 2 | [02_forward_propagation.py](math_foundations/02_forward_propagation.py) | Layers, matrix math, shape tracking, XOR proof | 50 min |
| 3 | [03_backpropagation.py](math_foundations/03_backpropagation.py) | Chain rule, gradients, full from-scratch training | 60 min |
| 4 | [04_loss_functions_and_optimizers.py](math_foundations/04_loss_functions_and_optimizers.py) | MSE/BCE/CCE, SGD, Momentum, RMSprop, Adam | 55 min |
| 5 | [05_regularization.py](math_foundations/05_regularization.py) | L1/L2, Dropout, Batch Norm, Early Stopping | 50 min |

### Week 2: Algorithms (4/4 modules) ✅

| # | File | Topic | Time |
|---|------|--------|------|
| 6 | [perceptron_from_scratch.py](algorithms/perceptron_from_scratch.py) | 1957 Perceptron, AND/OR, XOR failure | 45 min |
| 7 | [multilayer_perceptron.py](algorithms/multilayer_perceptron.py) | Full MLP + Adam from scratch, solves XOR | 65 min |
| 8 | [mlp_with_keras.py](algorithms/mlp_with_keras.py) | Sequential API, compile/fit/evaluate/save | 55 min |
| 9 | [hyperparameter_tuning.py](algorithms/hyperparameter_tuning.py) | Grid search, LR schedules, depth vs width | 65 min |

### Week 3: Projects (2/2 modules) ✅

| # | File | Topic | Time |
|---|------|--------|------|
| 10 | [mnist_digit_classifier.py](projects/mnist_digit_classifier.py) | MNIST 10-class, 98%+ accuracy, confusion matrix | 75 min |
| 11 | [tabular_deep_learning.py](projects/tabular_deep_learning.py) | DNN vs LogReg vs RandomForest on medical data | 70 min |

**Progress: 11/11 modules (100%)** 🎉

---

## 🚀 Quick Start

```bash
pip install numpy matplotlib scikit-learn tensorflow

# Math foundations
cd deep_neural_networks/math_foundations
python3 01_neurons_and_activations.py
python3 02_forward_propagation.py
python3 03_backpropagation.py
python3 04_loss_functions_and_optimizers.py
python3 05_regularization.py

# Algorithms
cd ../algorithms
python3 perceptron_from_scratch.py
python3 multilayer_perceptron.py
python3 mlp_with_keras.py
python3 hyperparameter_tuning.py

# Projects
cd ../projects
python3 mnist_digit_classifier.py
python3 tabular_deep_learning.py

# View visualizations
open ../visuals/
```

---

## 🎯 What You'll Build

- A Perceptron that solves AND/OR but fails on XOR
- A full MLP with backprop written in pure numpy
- A Keras model that classifies MNIST digits at ~98% accuracy
- A 3-way model comparison (LogReg vs RF vs DNN) on real medical data

---

## 📦 Dependencies

```
numpy >= 1.21
matplotlib >= 3.4
scikit-learn >= 1.0
tensorflow >= 2.10   (for Keras modules — graceful fallback if not installed)
```
