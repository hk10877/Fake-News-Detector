import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import LinearSVC

# Load datasets
true_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")

true_news["label"] = 0  # 0 = Real
fake_news["label"] = 1  # 1 = Fake

# Combine
data = pd.concat([true_news, fake_news], axis=0).sample(frac=1, random_state=42)

X = data["text"]
y = data["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7)),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "fake_news_model.pkl")
print("✅ Model saved as fake_news_model.pkl")
