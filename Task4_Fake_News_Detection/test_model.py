import joblib

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

news = """
WASHINGTON (Reuters) - U.S. military will accept transgender recruits on Monday,
the Pentagon said on Friday, after President Donald Trump's administration
decided not to appeal a court ruling.
"""

vector = vectorizer.transform([news])

prediction = model.predict(vector)[0]

print("Prediction:", prediction)
