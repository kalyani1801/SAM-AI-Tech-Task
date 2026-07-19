from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)[0]
    print("=" * 50)
    print("Prediction:", prediction)
    print("News:", news[:150])
    print("=" * 50)

    if prediction == 1:
        result = "✅ REAL NEWS"
    else:
        result = "❌ FAKE NEWS"

    return render_template("index.html", prediction=result, news=news)


if __name__ == "__main__":
    app.run(debug=True)