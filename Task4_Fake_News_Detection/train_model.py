import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

# Combine title and text
fake["content"] = fake["title"] + " " + fake["text"]
true["content"] = true["title"] + " " + true["text"]

# Labels
fake["label"] = 0
true["label"] = 1

# Merge
data = pd.concat([fake, true], ignore_index=True)

# Shuffle
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

X = data["content"]
y = data["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.75,
    min_df=2
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print(f"Accuracy: {accuracy*100:.2f}%")

# Save
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("Model saved successfully.")