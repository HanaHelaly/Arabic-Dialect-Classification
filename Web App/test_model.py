from flask import Flask, request, jsonify
import pickle
import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the trained model, TF-IDF vectorizer, and label encoder
model_path = "finalnb_path.pkl"
vectorizer_path = "vectorizer_path.pkl"
label_encoder_path = "label_encoder_path.pkl"

with open(model_path, 'rb') as model_file:
    nb_model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

with open(label_encoder_path, 'rb') as le_file:
    label_encoder = pickle.load(le_file)

def preprocess_text(text):
    text = re.sub(r'\d+', '', text)  
    text = re.sub(r'[^ا-ي\s]', '', text, re.I|re.A)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Arabic unicode range

    text = text.strip()  
    stop_words = set(stopwords.words('arabic'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
# Define a function to predict the dialect
def predict_dialect(text):
    text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction_encoded = nb_model.predict(text_tfidf)
    prediction = label_encoder.inverse_transform(prediction_encoded)
    return prediction[0]

# Test the model with some sample texts
sample_texts = [
    "يا استاذي الفاضل احنا عاصرنا ده محدش حكاهولنا",
    "هوينه هانيبال ولد العقيد متزوج لبنانيه عارضه ازياء",
    "السعاده حضرتك المنشن المحترمين",
    "يلّي باعك ببصلة، بيعو بقشرتها! يلّي بيعملك حمار، البُطة! إذا قلتلَّك تقبرني بتجيب المجرفة وبتطمني"
]

for text in sample_texts:
    prediction = predict_dialect(text)
    print(f"Text: {text}\nPredicted Dialect: {prediction}\n")

