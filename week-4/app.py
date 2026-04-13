# =========================
# FASTAPI APP
# =========================
from fastapi import FastAPI
import pickle
import re
from nltk.corpus import stopwords
from textblob import TextBlob

# load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# stopwords
stop_words = set(stopwords.words('english'))

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Complaint Analysis API Running"}

# =========================
# HELPER FUNCTIONS
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

def get_priority(sentiment):
    if sentiment == "Negative":
        return "High"
    elif sentiment == "Neutral":
        return "Medium"
    else:
        return "Low"

def predict_complaint(text):
    text_clean = clean_text(text)
    text_vec = vectorizer.transform([text_clean])

    complaint = model.predict(text_vec)[0]
    sentiment = get_sentiment(text_clean)
    priority = get_priority(sentiment)

    return complaint, sentiment, priority

# =========================
# API ENDPOINT
# =========================
@app.get("/predict")
def predict(text: str):
    complaint, sentiment, priority = predict_complaint(text)

    return {
        "Complaint Type": complaint,
        "Sentiment": sentiment,
        "Priority": priority
    }