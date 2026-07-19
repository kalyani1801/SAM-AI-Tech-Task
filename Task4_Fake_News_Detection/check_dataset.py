import pandas as pd

fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print("Fake columns:", fake.columns.tolist())
print("True columns:", true.columns.tolist())

print(fake.head())
print(true.head())