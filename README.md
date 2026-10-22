# Sentiment Analysis of Amazon Fine Food Reviews

A supervised machine learning project that classifies product reviews as
positive or negative using TF-IDF features and Logistic Regression.
Built as an AI/ML mini project.

## Results

| Model | Accuracy |
|-------|----------|
| Baseline (TF-IDF, 20,000 features) | 93.43% |
| Optimized (TF-IDF, 30,000 features, sublinear TF) | 93.47% |

Baseline per-class performance on the test set:

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Negative | 0.87 | 0.68 | 0.77 |
| Positive | 0.94 | 0.98 | 0.96 |

## Dataset

- Source: [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) on Kaggle
- Original size: 568,454 reviews
- After cleaning: 363,903 reviews (306,819 positive, 57,084 negative)
- The dataset file (`Reviews.csv`) is not included because of its size.
  Download it from Kaggle and place it in the same folder as the notebook.

## Approach

1. **Data cleaning:** removed missing values, duplicate reviews and
   neutral 3-star reviews
2. **Labels:** ratings of 4 and 5 are positive, ratings of 1 and 2 are negative
3. **Train/test split:** 80/20, stratified (291,122 training and
   72,781 test reviews)
4. **Feature extraction:** TF-IDF with English stopword removal and
   unigrams and bigrams
5. **Model:** Logistic Regression
6. **Optimization:** increased TF-IDF features from 20,000 to 30,000
   and enabled sublinear term frequency
7. **Evaluation:** accuracy, precision, recall, F1-score and a confusion
   matrix (plotted with matplotlib and seaborn)

## Limitations

- The data is imbalanced (about 84% of reviews are positive), so accuracy
  looks higher than the model's performance on negative reviews.
  Negative recall is 0.68.
- The model only predicts positive or negative, so neutral reviews
  are forced into one of the two classes.
- Possible improvements: class weighting, other algorithms,
  and further hyperparameter tuning.

## Tech Stack

Python, Pandas, scikit-learn, matplotlib, seaborn

## How to Run

1. Clone this repository
2. Install the libraries:
   `pip install pandas scikit-learn matplotlib seaborn`
3. Download `Reviews.csv` from the Kaggle link above and place it
   next to `sentiment_analysis.ipynb`
4. Open the notebook in Jupyter or VS Code and run all cells

## Author

Aditi Anand Guddeti
[LinkedIn](https://www.linkedin.com/in/aditi-guddeti-73592b310/) |
[GitHub](https://github.com/Aditiii222)
