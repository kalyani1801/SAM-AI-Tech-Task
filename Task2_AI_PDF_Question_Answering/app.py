from flask import Flask, render_template, request
from pypdf import PdfReader
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
import pytesseract
import faiss
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# CHANGE THIS PATH IF TESSERACT IS INSTALLED SOMEWHERE ELSE
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# CHANGE THIS TO YOUR POPPLER PATH
POPPLER_PATH = r"C:\poppler\Library\bin"

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = []
index = None


def read_pdf(path):
    text = ""

    try:
        reader = PdfReader(path)

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    except:
        pass

    # If no text found, use OCR
    if text.strip() == "":
        images = convert_from_path(
            path,
            poppler_path=POPPLER_PATH
        )

        for img in images:
            text += pytesseract.image_to_string(img)

    return text


def split_text(text, chunk_size=500):
    return [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
    ]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global chunks, index

    file = request.files["pdf"]

    if file.filename == "":
        return render_template(
            "index.html",
            message="Please select a PDF."
        )

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(filepath)

    text = read_pdf(filepath)

    if text.strip() == "":
        return render_template(
            "index.html",
            message="No readable text found in PDF."
        )

    chunks = split_text(text)

    embeddings = model.encode(chunks)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    return render_template(
        "index.html",
        message="PDF uploaded successfully!"
    )


@app.route("/ask", methods=["POST"])
def ask():
    global chunks, index

    if index is None:
        return render_template(
            "index.html",
            answer="Please upload a PDF first."
        )

    question = request.form["question"]

    q_embedding = model.encode([question])

    q_embedding = np.array(q_embedding).astype("float32")

    _, I = index.search(q_embedding, 1)

    answer = chunks[I[0][0]]

    return render_template(
        "index.html",
        answer=answer,
        message="Ask another question."
    )


if __name__ == "__main__":
    app.run(debug=True)