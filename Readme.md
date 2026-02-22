# Movie Recommendation System

This project applies concepts like Linear Regression, Logistic Regression, Gradient Descent, and Regularization to a real-world movie dataset.  
The goal is to predict how a user would rate a movie and whether or not they will "Like" it.

## Project Structure

```
Movie-Recommendation-System/
│
├── data/
│   ├── ratings.csv
│   ├── movies.csv
│   └── tags.csv
│
├── analysis.ipynb
├── Requirements.txt
└── README.md
```
---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yWissem-Sahli-Engineer/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Acquisition & Initialization

**Data Source:**  
Downloaded from the [MovieLens Latest Small Dataset](https://grouplens.org/datasets/movielens/latest/)

**Action:**
- Extracted the zip file containing `ratings.csv`, `movies.csv`, and `tags.csv`.
- Loaded the `.csv` files into Pandas DataFrames for manipulation.

**Datasets overview:**

| Dataset      | Rows     | Columns                              |
|--------------|----------|--------------------------------------|
| `movies.csv` | 9,742    | `movieId`, `title`, `genres`         |
| `ratings.csv`| 100,836  | `userId`, `movieId`, `rating`, `timestamp` |
| `tags.csv`   | 3,683    | `userId`, `movieId`, `tag`, `timestamp`    |

---

## Joining Datasets

- Merged `ratings` and `movies` DataFrames on the `movieId` column using `pd.merge()`.
- The resulting DataFrame (`df`) has **100,836 rows** and **6 columns**: `userId`, `movieId`, `rating`, `timestamp`, `title`, `genres`.
- Key statistics on the merged dataset:
  - Average rating: **3.50**
  - Rating range: **0.5 – 5.0**
  - Number of unique users: **610**

---

## Exploratory Data Analysis (EDA)

### Rating Distribution
- Plotted a histogram of the `rating` column to visualize how ratings are distributed across the dataset.

### Genre Frequency
- Exploded the pipe-separated `genres` column using `str.get_dummies(sep='|')` to count genre occurrences.
- Top genres by number of ratings:
  - **Drama** — 41,928
  - **Comedy** — 39,053
  - **Action** — 30,635
  - **Thriller** — 26,452
  - **Adventure** — 24,161
- Visualized genre frequencies with a bar chart.

---

## Preprocessing

- **Target variable:** `rating`
- **Feature used:** `genres` (one-hot encoded via `CountVectorizer` with a pipe `|` tokenizer)
- **Train/Test Split:** 80/20 split (`random_state=42`)
- **Pipeline:**
  - Used `sklearn.pipeline.Pipeline` with `CountVectorizer` for genre tokenization.
  - Wrapped inside a `ColumnTransformer` for clean integration.
- **Result:** 20 unique genre features extracted, training data shape: `(80,668, 20)`.

---

## Part 1: Regression — Predicting the Rating

### 1. Linear Regression
- Trained a `LinearRegression` model on the genre features.
- **Training RMSE: ~1.021**
- Extracted feature weights (coefficients) sorted by importance:
  - Top positive: `Documentary (+0.37)`, `Animation (+0.34)`, `War (+0.27)`
  - Top negative: `Children (−0.27)`, `Horror (−0.18)`, `Comedy (−0.13)`

### 2. Learning Curves
- Plotted learning curves for Linear Regression using 5-fold cross-validation.
- Both training and validation errors converge close to each other (~1.04), indicating **high bias (underfitting)** — the model is too simple to capture the complexity of user preferences from genres alone.

---

## Regularized Models

### 3. Ridge Regression (L2 Regularization)
- Trained a `Ridge` model with `alpha=41`.
- **Training RMSE: ~1.021** (nearly identical to plain Linear Regression, confirming the model is not overfitting).

### 4. Lasso Regression (L1 Regularization)
- Trained a `Lasso` model with `alpha=1`.
- **Training RMSE: ~1.041** (slightly worse — Lasso's tendency to zero-out features may discard useful genre signals).

### 5. ElasticNet Regression (L1 + L2)
- Trained an `ElasticNet` model with `alpha=0.01` and `l1_ratio=0.1`.
- **Training RMSE: ~1.022** (comparable to Ridge and Linear Regression).

### 6. Hyperparameter Tuning

#### GridSearchCV (ElasticNet)
- Searched over a grid of `alpha` and `l1_ratio` values.
- **Best parameters found:** `alpha=0.01`, `l1_ratio=0.1`.

#### RandomizedSearchCV (Ridge)
- Searched over a log-uniform distribution for `alpha` with 100 iterations.
- **Best alpha found: 41.25**

### 7. Evaluation on the Test Set
- Applied the preprocessing pipeline to the test set.
- Used the best model (Linear Regression) for final predictions.
- **Final Test RMSE: ~1.026**

### Regression Conclusion

A Linear Regression model using only genres is a **Baseline Model**. It's like a weather app that only tells you the season (e.g., "It's Summer") but can't tell you the temperature for today. It is useful as a **starting point**, but not for a final product.

---

## Part 2: Classification — Predicting "Like" or "Not Like"

### Problem Setup
- Converted the regression target into a **binary classification** problem:
  - **1 ("Liked"):** rating ≥ 4.0
  - **0 ("Not Liked"):** rating < 4.0
- Class distribution is roughly balanced: ~51.8% Not Liked vs. ~48.2% Liked.

### Logistic Regression
- Trained a `LogisticRegression` model (`random_state=42`) on the same genre features used for regression.

### Confusion Matrix

|                    | Predicted: Not Liked | Predicted: Liked |
|--------------------|----------------------|------------------|
| **Actual: Not Liked** | 27,757 (TN)          | 14,022 (FP)      |
| **Actual: Liked**     | 20,520 (FN)          | 18,369 (TP)      |

### Precision, Recall & F1-Score

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| Not Liked   | 0.57      | 0.66   | 0.62     | 41,779  |
| Liked       | 0.57      | 0.47   | 0.52     | 38,889  |
| **Accuracy**|           |        | **0.57** | 80,668  |

### Classification Conclusion

The Logistic Regression model achieves an accuracy of **~57%**, which is only marginally better than random guessing on a balanced dataset. This further confirms that **genres alone are insufficient** to accurately predict user preferences — additional features such as user history, collaborative filtering signals, or content embeddings would be needed to build a more effective recommendation system.
