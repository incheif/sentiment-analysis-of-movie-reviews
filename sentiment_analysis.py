import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
# Load the IMDb movie review dataset
data = pd.read_csv('imdb_reviews.csv')

# Remove HTML tags from the text
data['review'] = data['review'].apply(lambda x: re.sub('<.*?>', '', x))

# Remove non-alphabetic characters from the text
data['review'] = data['review'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

# Convert text to lowercase
data['review'] = data['review'].apply(lambda x: x.lower())

# Remove stop words from the text
stop_words = set(stopwords.words('english'))
data['review'] = data['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Lemmatize the text
lemmatizer = WordNetLemmatizer()
data['review'] = data['review'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

# Convert the text into a numerical format
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['review'])
y = data['sentiment']
print(data.head(5))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing set
accuracy = model.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))
from sklearn.metrics import classification_report

# Evaluate the model on the testing set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
