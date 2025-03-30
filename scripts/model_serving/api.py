from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import SentimentPredictor
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text using DistilBERT",
    version="1.0.0"
)

# Initialize the model
sentiment_predictor = SentimentPredictor()

# Define request model
class SentimentRequest(BaseModel):
    text: str

# Define response model
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict

# Define API endpoints
@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment of input text.
    Returns sentiment label, confidence, and probability scores.
    """
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    result = sentiment_predictor.predict(request.text)
    return result

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "healthy"}

# Run the API server
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 