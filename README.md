# Twitter Sentiment Analysis

This project analyzes the sentiment of entities mentioned in tweets. The goal is to classify sentiments as positive, negative, or neutral for specific entities extracted from Twitter data.

---

## Project Overview

- **Task**: Entity-level sentiment classification from tweets
- **Dataset**: Twitter Sentiment Dataset (from Kaggle)
- **Approach**: Text preprocessing → Feature extraction (TF-IDF) → ML classification
- **Tools Used**: Python, pandas, scikit-learn, matplotlib, seaborn, NLP (spaCy)

---

## Dataset Details

The dataset contains labeled tweets in the following structure:

| Column        | Description                                 |
|---------------|---------------------------------------------|
| `tweet_id`    | Unique tweet ID                             |
| `entity`      | Entity mentioned (e.g., product, company)   |
| `sentiment`   | Sentiment label (Positive, Negative, Neutral) |
| `content`     | Actual tweet content                        |

### Files Used

- `twitter_training.csv` – Training data
- `twitter_validation.csv` – Validation/test data

---

## Preprocessing

- Removed punctuation, special characters
- Tokenization using **spaCy**
- Lowercasing and stopword removal
- Lemmatization
- Feature extraction using **TF-IDF**

---

## Models Used

- **Multinomial Naive Bayes**
- **Logistic Regression**
- **Random Forest Classifier**

### Evaluation Metrics

- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrix**

---

## Results

| Model                  | Accuracy | F1 Score |
|------------------------|----------|----------|
| Multinomial Naive Bayes| ~89%     | 0.89     |
| Random Forest          | ~91%    | 0.9148    |

Best Model: **Random Forest** with **F1 Score ≈ 0.9148**

---

## Visualizations

- Sentiment distribution bar plot
- Confusion matrices for model comparison
- Word cloud for positive/negative tweets

---

## Tech Stack

- Python
- NLP
- ML
- pandas, numpy
- scikit-learn
- spaCy
- matplotlib, seaborn
- Jupyter Notebook
