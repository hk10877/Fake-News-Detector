from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI()

# Enable CORS (so React frontend at localhost:3000 can talk to backend at localhost:8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("fake_news_model.pkl")

class Article(BaseModel):
    text: str

@app.get("/")
def root():
    return {"message": "Fake News Detector API is running!"}

@app.post("/predict")
def predict(article: Article):
    proba = model.predict_proba([article.text])[0]
    is_fake_score = proba[1]
    return {
        "fake_probability": float(is_fake_score),
        "verdict": "Fake" if is_fake_score > 0.6 else "Real"
    }
