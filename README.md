# Advanced Spam Detection with NLP and Transformers

---

## Overview

This project presents a **comprehensive comparison between classical machine learning approaches and transformer-based deep learning models** for the task of email spam detection. The study evolves from traditional models using Bag-of-Words and TF-IDF into hybrid embeddings and culminates with a fine-tuned **DistilBERT** classifier. Through careful preprocessing, balancing, and evaluation, the project explores the trade-offs between explainability, accuracy, and computational efficiency.

The goal is not only high performance but to gain deep, transferable understanding of how text-based models behave across a spectrum of features, architectures, and preprocessing techniques.

---

## Objectives

- Benchmark traditional classifiers (Naïve Bayes, Logistic Regression, SVM) on BoW/TF-IDF/Word2Vec features.
- Implement and fine-tune a transformer model (DistilBERT) using HuggingFace.
- Address dataset imbalance using upsampling techniques.
- Conduct detailed error analysis to understand model blind spots.
- Reflect on the interpretability vs. performance trade-offs of deep models.

---

## What I Learned

Working on this project deepened my technical understanding and provided practical exposure to many core NLP and ML principles:

### Technical Concepts Mastered
- **Pipeline design** for text classification: from cleaning to vectorization to modeling
- **Vectorization trade-offs**: When BoW is more stable, TF-IDF more nuanced, and Word2Vec adds semantic value
- **Tokenization for Transformers**: How HuggingFace's tokenizer handles subwords and context
- **Regularization & early stopping** in transformer fine-tuning
- **Upsampling** for minority classes without introducing overfitting
- Building a **hybrid feature space** using both TF-IDF and Word2Vec

### Strategic Insights
- Even **basic models perform surprisingly well** on short text, due to high signal-to-noise ratio
- **Transformers offer semantic understanding** unmatched by traditional models, especially for informal or camouflaged spam
- Transformer-based models are **resource-intensive** and should be balanced against performance gains in production
- **Explainability trade-offs** are real: BERT is a black box, while Naïve Bayes is interpretable but naive

---

## Dataset Details

- **Source**: [Kaggle - Spam Email Classification](https://www.kaggle.com)
- **Samples**: 5,572 email messages (cleaned)
- **Target variable**: `label` ∈ {ham, spam}
- **Class imbalance**: ~87% ham, 13% spam

---

## Preprocessing Pipeline

| Step                     | Purpose                                           |
|--------------------------|---------------------------------------------------|
| HTML tag & link removal  | Reduce noise, strip web-related artifacts         |
| Lowercasing              | Normalize vocabulary                              |
| Tokenization             | Split into units for modeling                     |
| Stopword removal         | Remove non-informative words                      |
| Stemming (Snowball)      | Normalize word forms                              |
| Upsampling (for spam)    | Address severe imbalance in training set          |

---

## Models & Evaluation

### Naïve Bayes & Logistic Regression

| Vectorizer | Model               | Accuracy | Precision | Recall | F1 Score |
|------------|---------------------|----------|-----------|--------|----------|
| BoW        | Naïve Bayes         | 96.7%    | 97.1%     | 90.6%  | 93.8%    |
| BoW        | Logistic Regression | 97.1%    | 96.8%     | 91.3%  | 93.9%    |
| TF-IDF     | Naïve Bayes         | 95.5%    | 94.2%     | 88.5%  | 91.3%    |
| TF-IDF     | Logistic Regression | 96.2%    | 95.9%     | 89.0%  | 92.3%    |

>  Conclusion: BoW often outperforms TF-IDF on short, noisy text due to overpenalization of rare informative words in TF-IDF.

---

### SVM + Hybrid Vectorization

Combined TF-IDF + Word2Vec to provide frequency + semantic information.

**Best Results** (BoW + linear SVM):  
- Accuracy: **98.2%**  
- Precision: **97.8%**  
- Recall: **88.6%**  
- F1 Score: **93.0%**

>  Still missed some cleverly disguised spam due to limited context-awareness.

---

### DistilBERT Transformer

Used HuggingFace's `Trainer` and `DistilBertTokenizer` for fine-tuning.

**Configuration:**
- 1 epoch (due to convergence)
- Learning rate: 5e-5
- Weight decay (L2 regularization)
- Training set upsampled
- Early stopping based on validation loss

**Results**:

| Metric     | Score   |
|------------|---------|
| Accuracy   | 99.3%   |
| Precision  | 97.8%   |
| Recall     | 97.3%   |
| F1 Score   | 97.6%   |

> BERT recognized subtle context, slang, and structure in spam—even where traditional models failed. However, some borderline cases still confused the model.

---

## Error Analysis

**False negatives** (missed spam) included:
- Informal spam disguised as friendly notes
- Text without traditional spam features

**False positives** (misclassified ham):
- Contained promotional language
- Included links or numeric codes

> Insight: Spam evolves; models must learn beyond keyword spotting and include tone, semantics, and sender context.

---

## Tools Used

- `pandas`, `numpy`: Data handling  
- `scikit-learn`: ML models, metrics, resampling  
- `nltk`, `re`: NLP preprocessing  
- `gensim`: Word2Vec embedding  
- `transformers`, `torch`: DistilBERT training  
- `matplotlib`, `seaborn`: Visualizations

---

## Future Work

- Evaluate **zero-shot learning** with models like T5 or GPT-Neo
- Add **explainability tools** (e.g. SHAP, LIME) for audits
- Implement **ensemble model** combining transformer and traditional models
- Collect newer, multilingual spam samples
- Deploy with Flask for a real-time API filter

---
