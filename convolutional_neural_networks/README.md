# 🖼️ Part 4: Convolutional Neural Networks (CNNs)

**Learn how computers see — from raw pixel arrays to state-of-the-art image classifiers.**

---

## 📚 Module Map

### Week 1: Math Foundations (3/3 modules) ✅

| # | File | Topic | Time |
|---|------|--------|------|
| 1 | [01_image_basics.py](math_foundations/01_image_basics.py) | Pixels, channels, (N,H,W,C) format, MLP vs CNN param count | 45 min |
| 2 | [02_convolution_operation.py](math_foundations/02_convolution_operation.py) | Filters, stride, padding, output size formula from scratch | 50 min |
| 3 | [03_pooling_and_depth.py](math_foundations/03_pooling_and_depth.py) | MaxPool, AvgPool, GAP, receptive field, feature hierarchy | 45 min |

### Week 2: Algorithms (4/4 modules) ✅

| # | File | Topic | Time |
|---|------|--------|------|
| 4 | [conv_layer_from_scratch.py](algorithms/conv_layer_from_scratch.py) | Conv2D, MaxPool2D, Dense — all in pure numpy | 60 min |
| 5 | [cnn_with_keras.py](algorithms/cnn_with_keras.py) | 3-block Keras CNN, data augmentation, feature map inspection | 60 min |
| 6 | [classic_architectures.py](algorithms/classic_architectures.py) | LeNet, AlexNet, VGG, ResNet + skip connections in Keras | 65 min |
| 7 | [transfer_learning.py](algorithms/transfer_learning.py) | MobileNetV2 + ResNet50: feature extract → fine-tuning | 70 min |

### Week 3: Projects (2/2 modules) ✅

| # | File | Topic | Time |
|---|------|--------|------|
| 8 | [cifar10_classifier.py](projects/cifar10_classifier.py) | MLP vs CNN vs TL on CIFAR-10, confusion matrix, per-class analysis | 80 min |
| 9 | [custom_image_classifier.py](projects/custom_image_classifier.py) | End-to-end pipeline for YOUR dataset, model export, inference | 85 min |

**Progress: 9/9 modules (100%)** 🎉

---

## 🚀 Quick Start

```bash
pip install numpy matplotlib scikit-learn tensorflow

# Math foundations
cd convolutional_neural_networks/math_foundations
python3 01_image_basics.py
python3 02_convolution_operation.py
python3 03_pooling_and_depth.py

# Algorithms
cd ../algorithms
python3 conv_layer_from_scratch.py
python3 cnn_with_keras.py
python3 classic_architectures.py
python3 transfer_learning.py

# Projects
cd ../projects
python3 cifar10_classifier.py
python3 custom_image_classifier.py

# View visualizations
open ../visuals/
```

---

## 🎯 What You'll Build

- A `Conv2D` layer written in pure numpy — forward pass, filters, feature maps
- A 3-block Keras CNN that trains on CIFAR-10 with data augmentation
- Classic architectures: LeNet-5, AlexNet-Mini, VGG-Mini, ResNet with skip connections
- MobileNetV2 and ResNet50 feature extractors + fine-tuned classifiers on flowers dataset
- A CIFAR-10 benchmark comparing MLP vs CNN vs Transfer Learning head-to-head
- A complete reusable pipeline for YOUR custom image dataset (any classes!)

---

## 🧠 Key Concepts Covered

| Concept | Where |
|---------|-------|
| Image as numpy array, pixel values, channels | 01_image_basics |
| Convolution: filter, stride, padding, output size formula | 02_convolution_operation |
| MaxPool, AvgPool, Global Average Pooling | 03_pooling_and_depth |
| Weight sharing — why CNNs have fewer params than MLPs | conv_layer_from_scratch |
| BatchNormalization + Dropout in CNN context | cnn_with_keras |
| Vanishing gradients, skip connections, residual blocks | classic_architectures |
| Feature extraction vs fine-tuning, catastrophic forgetting | transfer_learning |
| Per-class accuracy, confusion matrix, failure mode analysis | cifar10_classifier |
| Class imbalance, two-phase training, model export + inference | custom_image_classifier |

---

## 📦 Dependencies

```
numpy >= 1.21
matplotlib >= 3.4
scikit-learn >= 1.0
tensorflow >= 2.10   (required for CNN modules — graceful fallback if not installed)
```

---

## 🖼️ Generated Visualizations

Each module saves 3–7 PNG plots at 300 dpi to its own `visuals/` subdirectory:

| Module | Visuals |
|--------|---------|
| 01_image_basics | Grayscale augmentations, RGB channels, param comparison |
| 02_convolution_operation | 6 filter feature maps, stride comparison, step-by-step diagram |
| 03_pooling_and_depth | Pooling comparison, CNN pipeline, feature hierarchy |
| conv_layer_from_scratch | Filter bank + feature maps, shape trace, feature map grid |
| cnn_with_keras | CIFAR samples, training history, confusion matrix, feature maps |
| classic_architectures | Architecture timeline, residual block diagram, ResNet training |
| transfer_learning | Strategies diagram, accuracy comparison, fine-tuning history |
| cifar10_classifier | Sample images, augmentation, training, confusion, per-class, predictions |
| custom_image_classifier | Dataset samples, pipeline diagram, training history, confidence |

---

## 📈 Expected Performance

| Task | Model | Expected Accuracy |
|------|-------|-------------------|
| CIFAR-10 | MLP baseline | ~50% |
| CIFAR-10 | 3-block CNN | ~85-88% |
| CIFAR-10 | MobileNetV2 TL | ~88-91% |
| Flowers (3.6k) | From scratch | ~65-75% |
| Flowers (3.6k) | MobileNetV2 FE | ~85-90% |
| Flowers (3.6k) | MobileNetV2 FT | ~88-93% |
| Custom dataset | MobileNetV2 TL | Depends on data |
