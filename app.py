from flask import Flask, request, render_template
import joblib
import os

# Load model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news']
    cleaned_text = input_text.strip().lower()
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]
    result = "FAKE" if prediction == 1 else "REAL"
    return render_template('index.html', prediction=result, input_text=input_text)

# Run the app with proper port binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
