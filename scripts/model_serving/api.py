from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from predict import SentimentPredictor
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text using DistilBERT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods  
    allow_headers=["*"],  # Allows all headers
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Get model and tokenizer paths from environment variables
MODEL_PATH = os.getenv("MODEL_PATH")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH")

# Initialize the model with environment variables
sentiment_predictor = SentimentPredictor(
    model_path=MODEL_PATH,
    tokenizer_path=TOKENIZER_PATH
)

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
@app.get("/")
async def read_root():
    """
    Serve the index.html file
    """
    return FileResponse('index.html')

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