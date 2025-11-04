import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

df = pd.read_csv("data/IFND.csv")
df.drop("Unnamed: 0",axis=1,inplace=True)



stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#','', text)
    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    # Remove words containing numbers
    text = re.sub(r'\w*\d\w*', '', text)
    # Tokenize and remove stopwords
    tokens = text.split()
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

# Apply cleaning function to the news_headings column
df['cleaned_text'] = df['Statement'].apply(clean_text)


from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the cleaned text
X = vectorizer.fit_transform(df['cleaned_text']).toarray()

# Save the fitted vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Encode labels: fake as 1, true as 0
y = df['Label'].apply(lambda x: 1 if x.lower() == 'fake' else 0)



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialize and train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# Save the trained model
with open("logistic_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

    