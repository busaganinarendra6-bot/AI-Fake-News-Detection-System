import pandas as pd
import re

# Load datasets
fake = pd.read_csv("dataset/Fake.csv")
real = pd.read_csv("dataset/True.csv")

# Add labels
fake["label"] = 0
real["label"] = 1

# Merge datasets
data = pd.concat([fake, real])

print("Total Records:", data.shape)

# Clean text
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

data["text"] = data["text"].apply(clean_text)

print(data["text"].head())

# Convert text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(data["text"])
y = data["label"]

print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy * 100, "%")
import pickle
import os

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
pickle.dump(model, open("model/model.pkl", "wb"))

# Save vectorizer
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("Model Saved Successfully!")