# CLAUDE.md - MLForBeginners Project

## Project Overview

A comprehensive, visual-first machine learning course for absolute beginners. Teaches from algebra basics through production-ready ML models — including Deep Neural Networks, CNNs, NLP, Transformers, and LLMs. No ML or math background required.

## Repository Structure

```
MLForBeginners/
├── regression_algorithms/          # Part 1: Regression (100% complete)
│   ├── math_foundations/           # 5 modules: algebra → probability
│   ├── algorithms/                 # linear regression, multiple regression
│   ├── examples/                   # simple examples, data exploration, model eval
│   ├── projects/                   # housing analysis, house price prediction
│   ├── visuals/                    # auto-generated visualizations
│   ├── data/                       # datasets
│   └── requirements.txt
├── classification_algorithms/      # Part 2: Classification (100% complete)
│   ├── math_foundations/           # 5 modules: sigmoid → decision boundaries
│   ├── algorithms/                 # logistic regression, KNN, decision trees, random forests, metrics
│   ├── examples/                   # practice examples
│   ├── projects/                   # spam classifier, churn prediction, model comparison
│   └── visuals/                    # auto-generated visualizations
├── deep_neural_networks/           # Part 3: DNNs (100% complete)
│   ├── math_foundations/           # 5 modules: neurons → regularization
│   ├── algorithms/                 # perceptron, MLP from scratch, Keras MLP, HP tuning
│   ├── projects/                   # MNIST classifier, tabular deep learning
│   └── visuals/                    # auto-generated visualizations
├── convolutional_neural_networks/  # Part 4: CNNs (100% complete)
│   ├── math_foundations/           # 3 modules: image basics, convolution, pooling & depth
│   ├── algorithms/                 # conv from scratch, Keras CNN, classic archs, transfer learning
│   ├── projects/                   # CIFAR-10 classifier, custom image classifier
│   └── visuals/                    # auto-generated visualizations
├── nlp/                            # Part 5: NLP (100% complete)
│   ├── math_foundations/           # 4 modules: text processing → RNN intuition
│   ├── algorithms/                 # text classification, sentiment, LSTM, NER
│   ├── projects/                   # movie review sentiment, news article classifier
│   ├── visuals/                    # auto-generated visualizations
│   └── README.md
├── transformers/                   # Part 6: Transformers (100% complete)
│   ├── math_foundations/           # 4 modules: attention → encoder-decoder arch
│   ├── algorithms/                 # transformer from scratch, BERT, GPT decoder
│   ├── projects/                   # BERT text classifier, GPT-2 text generator
│   ├── visuals/                    # auto-generated visualizations
│   └── README.md
├── README.md
├── CONTRIBUTING.md
├── SETUP_GUIDE.md
└── COURSE_COMPLETION_GUIDE.md
```

## Course Status

- **Part 1 - Regression**: 12/12 modules (100%) ✅
- **Part 2 - Classification**: 16/16 modules (100%) ✅
- **Part 3 - Deep Neural Networks**: 11/11 modules (100%) ✅
- **Part 4 - CNNs**: 9/9 modules (100%) ✅
- **Part 5 - NLP**: 10/10 modules (100%) ✅
- **Part 6 - Transformers**: 9/9 modules (100%) ✅
- **Part 7 - LLMs**: 0/9 modules (planned)

## Completed Curriculum (Parts 1–6) / Planned (Part 7)

### Part 4: Convolutional Neural Networks (CNNs) ✅
- Math Foundations (3): 01_image_basics, 02_convolution_operation, 03_pooling_and_depth
- Algorithms (4): conv_layer_from_scratch, cnn_with_keras, classic_architectures, transfer_learning
- Projects (2): cifar10_classifier, custom_image_classifier

### Part 5: Natural Language Processing (NLP) ✅
- Math Foundations (4): 01_text_processing, 02_bag_of_words_tfidf, 03_word_embeddings, 04_rnn_intuition
- Algorithms (4): text_classification_pipeline, sentiment_analysis, lstm_text_classifier, named_entity_recognition
- Projects (2): movie_review_sentiment, news_article_classifier

### Part 6: Transformers ✅
- Math Foundations (4): 01_attention_mechanism, 02_multi_head_attention, 03_positional_encoding, 04_encoder_decoder_arch
- Algorithms (3): transformer_from_scratch, bert_encoder, gpt_decoder
- Projects (2): bert_text_classifier, gpt2_text_generator

### Part 7: LLMs
- Concepts (4): how LLMs work, prompt engineering, fine-tuning basics, RAG
- Practical (3): using LLM APIs, build a chatbot, LLM evaluation
- Projects (2): Q&A system with RAG, LLM-powered classifier

## Tech Stack

- **Python 3.8+**
- **Core**: numpy, pandas, scikit-learn
- **Visualization**: matplotlib, seaborn, plotly
- **Deep Learning**: tensorflow/keras (Parts 3-7)
- **NLP** (Part 5+): spacy, nltk, transformers (HuggingFace)
- **Animation**: imageio, imageio-ffmpeg
- **Interactive**: jupyter, ipywidgets

Install base: `pip install -r regression_algorithms/requirements.txt`
Install DNN:  `pip install tensorflow`
Install NLP:  `pip install transformers spacy nltk`

## Development Conventions

### Module Structure
Each Python module follows this pattern:
1. Header docstring: title, 5-7 learning objectives, YouTube links, time/difficulty/prerequisites
2. Sections divided by `"=" * 70` separators with print() headers
3. Heavy print statements explaining concepts as they happen (educational output)
4. "From scratch" (numpy) implementation FIRST
5. Production implementation second (scikit-learn / Keras / HuggingFace)
6. 2-3 auto-generated visualizations saved to `visuals/` subdirectory (300 dpi PNG)
7. Real or standard dataset usage where applicable

### Naming Conventions
- Math foundation files: `NN_descriptive_name.py` (e.g., `01_neurons_and_activations.py`)
- Algorithm files: `descriptive_name.py` (e.g., `perceptron_from_scratch.py`)
- Lab files: `module_name_lab.md` (e.g., `linear_regression_intro_lab.md`)
- Visualization directories: named after the concept (e.g., `visuals/01_neurons_activations/`)

### Visualization Pattern
Each module saves plots to its own `visuals/` subfolder. Always use `plt.savefig()` at 300 dpi. Never `plt.show()` — visualizations are auto-generated on run.

### Framework Convention (Parts 3+)
- Always show "from scratch" numpy version first
- Then show the Keras/HuggingFace equivalent
- Use `try/except ImportError` for tensorflow/transformers with a helpful install message
- Never crash if optional libraries are missing — degrade gracefully

### Practice Labs
Lab files (`.md`) accompany key algorithm modules. They provide hands-on exercises that build on the module content. Target: 4-5 progressive tasks per lab.

## Running Modules

```bash
# Part 3: Deep Neural Networks
cd deep_neural_networks/math_foundations
python3 01_neurons_and_activations.py
python3 02_forward_propagation.py

cd ../algorithms
python3 perceptron_from_scratch.py
python3 mlp_with_keras.py

cd ../projects
python3 mnist_digit_classifier.py

# Part 4: Convolutional Neural Networks
cd convolutional_neural_networks/math_foundations
python3 01_image_basics.py
python3 02_convolution_operation.py
python3 03_pooling_and_depth.py

cd ../algorithms
python3 conv_layer_from_scratch.py
python3 cnn_with_keras.py
python3 classic_architectures.py
python3 transfer_learning.py

cd ../projects
python3 cifar10_classifier.py
python3 custom_image_classifier.py

# View generated visuals
open visuals/

# Part 5: Natural Language Processing
cd nlp/math_foundations
python3 01_text_processing.py
python3 02_bag_of_words_tfidf.py
python3 03_word_embeddings.py
python3 04_rnn_intuition.py

cd ../algorithms
python3 text_classification_pipeline.py
python3 sentiment_analysis.py
python3 lstm_text_classifier.py
python3 named_entity_recognition.py

cd ../projects
python3 movie_review_sentiment.py
python3 news_article_classifier.py

# Part 6: Transformers
cd transformers/math_foundations
python3 01_attention_mechanism.py
python3 02_multi_head_attention.py
python3 03_positional_encoding.py
python3 04_encoder_decoder_arch.py

cd ../algorithms
python3 transformer_from_scratch.py
python3 bert_encoder.py
python3 gpt_decoder.py

cd ../projects
python3 bert_text_classifier.py
python3 gpt2_text_generator.py
```

## Git Workflow

- Main branch: `main`
- Commit style: descriptive, present-tense summary (e.g., `Add Deep Neural Networks module`)
- Worktrees used for parallel development (current worktree: `bold-edison`)
