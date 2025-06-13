from flask import Flask, request, render_template
import pickle
import re
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

app = Flask(__name__)

# Load the vectorizer and model once when app starts
with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

print("Vocabulary size:", len(vectorizer.vocabulary_))

with open("model/logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    tokens = text.split()
    filtered_tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    confidence = ''
    if request.method == 'POST':
        text = request.form['text']
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        pred = model.predict(vectorized)[0]
        prob = model.predict_proba(vectorized)[0]
        label = 'FAKE' if pred == 1 else 'REAL'
        score = f"{max(prob) * 100:.2f}%"
        return render_template('frontpage.html', prediction=label, confidence=score)

    return render_template('frontpage.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
