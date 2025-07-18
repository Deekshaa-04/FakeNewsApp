from flask import Flask, request, render_template
import joblib
import os

# Load the model and vectorizer
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_text = request.form['news']
        if not input_text.strip():
            return render_template('index.html', error="Please enter news content to classify.")

        transformed_text = vectorizer.transform([input_text])
        prediction = model.predict(transformed_text)[0]
        result = "FAKE" if prediction == 1 else "REAL"
        return render_template('index.html', prediction=result, input_text=input_text)
    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")

# Ensure it runs on Render's public IP
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # use PORT env if set by Render
    app.run(host='0.0.0.0', port=port, debug=False)
