from flask import Flask,render_template,request
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import spacy  
from nltk.stem import WordNetLemmatizer
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords')
nltk.download('wordnet')
vectorizer = joblib.load('tfidf_vectorizer.joblib')
model = joblib.load('news_classifier.joblib')
data = pd.read_csv('merged_news.csv')
data.dropna(inplace=True)
unique_subjects = sorted(data['subject'].unique())




def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(f'[{string.punctuation}]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
    return ' '.join(words)

def predict_news(text):
    cleaned_text = clean_text(text)  # Clean the text
    transformed_text = vectorizer.transform([cleaned_text])  # Convert to TF-IDF
    probabilities = model.predict_proba(transformed_text)  # Get probability scores
    fake_prob = probabilities[0][1]  # Probability of being Fake
    real_prob = probabilities[0][0]  # Probability of being Real
    if fake_prob > real_prob:
        classification = "FAKE NEWS"
        confidence = fake_prob * 100  
    else:
        classification = "REAL NEWS"
        confidence = real_prob * 100  
    return classification,confidence
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html', subjects=unique_subjects)

@app.route('/get_titles', methods=['POST'])
def get_titles():
    subject = request.form['subject']
    titles = data[data['subject'] == subject][['title', 'text']].head(10).to_dict(orient='records')  # Fetch only 10 titles
    return {'titles': titles}


@app.route('/news/<title>')
def show_news(title):
    news_item = data[data['title'] == title].iloc[0]
    return render_template('news.html', title=news_item['title'], text=news_item['text'])

@app.route("/check")
def check():
    a = request.args.get("text")
    blob = TextBlob(a)
    doc = nlp(a)

    # Sentiment Analysis
    sentiment_polarity = blob.sentiment.polarity
    if sentiment_polarity > 0:
        sentiment = "Positive Statement"
    elif sentiment_polarity < 0:
        sentiment = "Negative Statement"
    else:
        sentiment = "Neutral Statement"

    classification, confidence = predict_news(a)

    # Named Entity Recognition (NER)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    entity_labels = [ent[1] for ent in entities]
    entity_counts = Counter(entity_labels)

    # Generate NER Bar Chart
    plt.figure(figsize=(8, 5))
    plt.bar(entity_counts.keys(), entity_counts.values(), color='skyblue')
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.title("Named Entity Recognition (NER) Analysis")
    plt.xticks(rotation=45)
    plt.savefig("static/ner_chart.png")  # Save chart
    plt.close()

    return render_template("result.html", sentiment=sentiment, classification=classification, confidence=confidence, entities=entities)

app.run(debug=False)