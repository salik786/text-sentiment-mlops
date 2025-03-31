from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from pydantic import BaseModel
import torch
from predict import SentimentPredictor
from scripts.monitoring.logger import api_logger, monitor
import os
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import logging
import json
import numpy as np
import uvicorn

# Get the project root directory (2 levels up from this script)
project_root = Path(__file__).parent.parent.parent

# Get model paths from environment variables or use defaults
model_path = os.environ.get('MODEL_PATH', str(project_root / 'models/saved/sentiment_model'))
tokenizer_path = os.environ.get('TOKENIZER_PATH', str(project_root / 'models/saved/sentiment_tokenizer'))

# Check if we're in test mode
test_mode = os.environ.get('TEST_MODE', 'false').lower() == 'true'
skip_model_load = os.environ.get('SKIP_MODEL_LOAD', 'false').lower() == 'true'

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for predicting sentiment of text using DistilBERT",
    version="1.0.0"
)

# Add CORS middleware with more specific configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly specify allowed methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["*"],  # Expose all headers
)

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

@app.on_event("startup")
async def startup_event():
    """Initialize application state on startup"""
    app.state.start_time = datetime.now()
    api_logger.info(f"Application started at {app.state.start_time}")
    
    # Initialize the model only if not in test mode or if model loading is not skipped
    if not skip_model_load:
        try:
            api_logger.info(f"Loading model from {model_path}")
            app.state.sentiment_predictor = SentimentPredictor(model_path=model_path, tokenizer_path=tokenizer_path)
            api_logger.info("Model loaded successfully")
        except Exception as e:
            api_logger.error(f"Error loading model: {str(e)}")
            if not test_mode:
                raise
    else:
        api_logger.info("Skipping model loading in test mode")

# Define request model
class SentimentRequest(BaseModel):
    text: str

# Define response model
class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict
    response_time_ms: float

# Define API endpoints
@app.get("/")
async def read_root():
    """
    Serve the index.html file
    """
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))

@app.get("/dashboard")
async def read_dashboard():
    """
    Serve the dashboard.html file
    """
    return FileResponse(str(Path(__file__).parent / "static" / "dashboard.html"))

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment of input text.
    Returns sentiment label, confidence, and probability scores.
    """
    api_logger.info(f"Received prediction request: {request.text[:50]}...")
    
    if not request.text or len(request.text.strip()) == 0:
        api_logger.warning("Empty text provided in request")
        raise HTTPException(status_code=400, detail="Empty text provided")
    
    if skip_model_load or test_mode:
        # Return mock response in test mode
        return SentimentResponse(
            text=request.text,
            sentiment="NEUTRAL",
            confidence=0.5,
            probabilities={"POSITIVE": 0.5, "NEGATIVE": 0.5, "NEUTRAL": 0.5},
            response_time_ms=0.0
        )
    
    start_time = time.time()
    try:
        # Get prediction
        prediction = app.state.sentiment_predictor.predict(request.text)
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return SentimentResponse(
            text=request.text,
            sentiment=prediction['sentiment'],
            confidence=prediction['confidence'],
            probabilities=prediction['probabilities'],
            response_time_ms=response_time
        )
    except Exception as e:
        api_logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/stats")
async def get_stats():
    """
    Get API statistics
    """
    if not hasattr(app.state, 'start_time'):
        return {"error": "Application not started"}
    
    uptime = datetime.now() - app.state.start_time
    return {
        "uptime_seconds": uptime.total_seconds(),
        "start_time": app.state.start_time.isoformat()
    }

# Run the API server
if __name__ == "__main__":
    api_logger.info("Starting API server")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    api_logger.info("API server stopped") 