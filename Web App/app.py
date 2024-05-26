from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

app = Flask(__name__)

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

classes = {
    "EG": "Egypt",
    "LB": "Lebanon",
    "LY" : "Libyan",
    "SD" : "Sudanese",
    "MA" : "Moroccan"
    }
# Define preprocessing function
def preprocess_text(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^ا-ي\s]', '', text, re.I | re.A)
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
    prediction = label_encoder.inverse_transform(prediction_encoded)[0]
    prediction_full_name = classes.get(prediction, prediction)
    return prediction_full_name

# Route for the home page and form submission
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_dialect(text)
        return render_template('index.html', text=text, prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
