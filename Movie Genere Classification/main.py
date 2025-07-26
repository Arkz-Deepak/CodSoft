import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# This code reads a dataset, processes it, and trains a logistic regression model to classify movie genres based on their names and descriptions.
# Load the dataset
df = pd.read_csv('data/train_data.txt',sep=':::',engine='python',names=['No.','MovieName','Genere', 'Description'])
y= df['Genere']
x= df['MovieName'] + " " + df['Description']
# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=9000)
X_tfidf = vectorizer.fit_transform(x)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=100)
# Logistic regression model
model= LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# Report
print(classification_report(y_test, y_pred))