# E-Commerce Review Sentiment Analysis

A machine learning project that performs sentiment analysis on women's clothing e-commerce reviews using various NLP techniques and ML models.

![](https://github.com/tenzinmigmar/review-sentiment-analysis/blob/main/banner.png)

## Overview
This project analyzes customer reviews from a women's clothing e-commerce platform to extract insights and predict sentiment. It implements multiple ML models (Logistic Regression, SVM, Naive Bayes) with Natural Language Processing techniques for text analysis.

## Features
- Text preprocessing with NLTK (tokenization, lemmatization, stopword removal)
- Sentiment classification using multiple ML models
- Comprehensive visualizations of review patterns and model performance
- Cross-validation and model comparison
- N-gram analysis of positive and negative reviews

## Project Structure
```
.
├── logistic_regression.py  # Logistic Regression model implementation
├── naive_bayes.py         # Naive Bayes model implementation
├── preprocessing.py       # Text cleaning and data preparation
├── svm_model.py          # Support Vector Machine implementation
├── visualization.py      # Data visualization functions
└── README.md
```

## Installation
```bash
# clone repository
git clone https://github.com/yourusername/ecommerce-review-analysis.git

# install required packages
pip install -r requirements.txt
```

## Usage
```python
from preprocessing import prepare_data
from logistic_regression import LogisticRegressionModel
from visualization import plot_basic_stats

# load and preprocess data
df = prepare_data('your_data.csv')

# train model
model = LogisticRegressionModel()
accuracy = model.train_and_evaluate(df)

# generate visualizations
plot_basic_stats(df)
```

## Dependencies
- Python 3.7+
- scikit-learn
- NLTK
- pandas
- numpy
- matplotlib
- seaborn

## Contributing
Feel free to open issues and pull requests for any improvements.

## License
MIT License

## Acknowledgments
- Dataset sourced from Kaggle: [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
