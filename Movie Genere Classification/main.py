import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
df = pd.read_csv('train_data.txt',sep=':::',engine='python',names=['No.','MovieName','Genere', 'Description'])
y= df['Genere']
x= df['MovieName'] + " " + df['Description']
vectorizer = TfidfVectorizer(stop_words='english', max_features=9000)
X_tfidf = vectorizer.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=100)
model= LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))