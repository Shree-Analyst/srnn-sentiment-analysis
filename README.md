# City Cinema Movie Review Sentiment Analysis

## Overview

| Summary Component | Description |
| :-------------: | ------ |
| **Dataset Size** | 50,000 movie reviews (25k train / 25k test) |
| **Problem Type** | Binary Classification (Positive / Negative) |
| **Tools Used** | Python, TensorFlow/Keras, Streamlit |
| **Key Metric** | Validation Accuracy ~81.2% |
| **Outcome** | Faciliated accurate, automatic sentiment prediction through a trained SimpleRNN model deployed to Streamlit. |

Jump to:
- [Executive Summary](#executive-summary)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Model Architecture & Training](#model-architecture--training)
- [Results](#results)
- [How to Run](#how-to-run)
- [Caveats & Next Steps](#caveats--next-steps)

# Project Background

City Cinema wants to automatically gauge audience sentiment from movie reviews. This project trains a Simple Recurrent Neural Network (SimpleRNN) on 50,000 labelled reviews to classify each as positive or negative, and serves predictions through a Streamlit app.

**Business Question:** Can we automatically classify the sentiment of a movie review as positive or negative?

# Executive Summary

A SimpleRNN with an Embedding layer was trained on City Cinema's 50,000-review dataset. Best performance was achieved at **epoch 3**, before training diverged to `NaN` loss:

- **Validation accuracy:** ~81.2%
- **Validation loss:** ~0.4161
- **Training accuracy:** ~83.5%

Early stopping (`patience=5`, `restore_best_weights=True`) restored the epoch-3 weights. The trained model is saved as `simple_rnn_imdb.keras` and served via a Streamlit app.

# Dataset & Preprocessing

- **Source:** Movie reviews via `tensorflow.keras.datasets.imdb`
- **Rows:** 25,000 training + 25,000 test
- **Labels:** `1` = positive, `0` = negative
- **Vocabulary:** Top 10,000 words
- **Sequence length:** Padded/truncated to 500 tokens using `pad_sequences`

# Model Architecture & Training

| Layer | Details |
| :--- | :--- |
| Embedding | Vocab 10,000, dim 128, input_length 500 |
| SimpleRNN | 128 units, ReLU |
| Dense | 1 unit, Sigmoid |
| Optimizer | Adam |
| Loss | Binary Crossentropy |
| Metric | Accuracy |
| Callbacks | EarlyStopping (monitor=val_loss, patience=5, restore_best_weights=True) |

**Total parameters:** 1,313,025

# Results

| Epoch | Train Acc | Val Acc | Val Loss |
| :---: | :---: | :---: | :---: |
| 1 | 0.5895 | 0.6758 | 0.5826 |
| **3** | **0.8354** | **0.8120** | **0.4161** |
| 8 | 0.6510 | 0.5062 | NaN |

Loss became `NaN` at epoch 8, triggering early stopping. Best weights (epoch 3) were restored.

# How to Run

## Streamlit App
[https://city-cinema-srnn-sentiment-analysis.streamlit.app/](https://city-cinema-srnn-sentiment-analysis.streamlit.app/)
