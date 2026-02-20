# Movie Recommendation System

This project applies concepts like Linear Regression, Logistic Regression, Gradient Descent, and Regularization to a real-world movie dataset.  
The goal is to predict how a user would rate a movie and whether or not they will "Like" it.

## Project Structure

```
Movie-Recommendation-System/
│
├── data/
│ ├── ratings.csv
│ ├── movies.csv
│ └── tags.csv
│
├── analysis.ipynb
├──Requirements.txt
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

## Data Acquisition & Initialization

**Data Source:** 
Downloaded from the [MovieLens Latest Small Dataset](https://grouplens.org/datasets/movielens/latest/)

**Action:**
- Extracted the zip file containing `ratings.csv`, `movies.csv`, and `tags.csv`.
- Loaded the `.csv` files into Pandas DataFrames for manipulation.

---
