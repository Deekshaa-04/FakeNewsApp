from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

# Load trained model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['news']
    if not input_text.strip():
        return render_template('index.html', prediction="⚠️ Please enter some news text!", input_text=input_text)

    transformed_text = vectorizer.transform([input_text])
    prediction = model.predict(transformed_text)[0]
    result = "🟥 FAKE" if prediction == 1 else "🟩 REAL"
    return render_template('index.html', prediction=result, input_text=input_text)

if __name__ == "__main__":
    app.run(debug=True)
