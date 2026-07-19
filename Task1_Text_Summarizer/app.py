from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""

    if request.method == "POST":
        text = request.form["text"]

        if len(text.strip()) > 0:
            result = summarizer(
                text,
                max_length=130,
                min_length=30,
                do_sample=False
            )

            summary = result[0]["summary_text"]

    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)