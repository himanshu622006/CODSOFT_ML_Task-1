# CODSOFT
ML(MACHINE LEARNING) TASK 1
 Movie Genre Classification 🎬

This project predicts the genre of a movie based on its plot summary using a machine learning model with TF-IDF and Logistic Regression.

## Steps:
1. Data Cleaning & Preprocessing
2. Feature Extraction using TF-IDF
3. Model Training using Logistic Regression
4. Evaluation using accuracy and classification report

## Libraries Required
- scikit-learn
- pandas

## How to Run
```bash
pip install -r requirements.txt
python genre_classifier.py
```
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

# Load the dataset
df = pd.read_csv('movies.csv')  # Make sure this CSV has 'plot' and 'genre' columns
X = df['plot']
y = df['genre']

# Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
