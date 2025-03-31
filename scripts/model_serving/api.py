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
    
    # Initialize the model
    api_logger.info(f"Loading model from {model_path}")
    app.state.sentiment_predictor = SentimentPredictor(model_path=model_path, tokenizer_path=tokenizer_path)
    api_logger.info("Model loaded successfully")

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
    
    # Measure prediction time
    start_time = time.time()
    result = app.state.sentiment_predictor.predict(request.text)
    end_time = time.time()
    response_time = end_time - start_time
    
    # Add response time to result
    result["response_time_ms"] = round(response_time * 1000, 2)
    
    # Log prediction
    monitor.log_prediction(
        text=request.text,
        prediction=result["sentiment"],
        confidence=result["confidence"] / 100,  # Convert to 0-1 scale
        response_time=response_time
    )
    
    return result

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {
        "status": "healthy",
        "uptime_seconds": (datetime.now() - app.state.start_time).total_seconds()
    }

@app.get("/stats")
async def get_stats():
    """
    Get detailed statistics about the model's performance and system status.
    """
    # Get basic statistics from monitor
    stats = monitor.get_statistics()
    
    # Add system information
    stats["system_info"] = {
        "start_time": app.state.start_time.isoformat(),
        "uptime_seconds": (datetime.now() - app.state.start_time).total_seconds(),
        "model_version": "DistilBERT-sentiment-v1",
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "total_requests": monitor.total_requests,
        "successful_requests": monitor.successful_requests,
        "failed_requests": monitor.failed_requests,
        "average_response_time": monitor.average_response_time,
        "p95_response_time": monitor.p95_response_time,
        "p99_response_time": monitor.p99_response_time,
    }
    
    # Add model performance metrics
    stats["model_performance"] = {
        "average_confidence": monitor.average_confidence,
        "sentiment_distribution": monitor.sentiment_distribution,
        "average_text_length": monitor.average_text_length,
        "drift_score": monitor.drift_score,
        "last_drift_check": monitor.last_drift_check.isoformat() if monitor.last_drift_check else None,
    }
    
    return stats

# Run the API server
if __name__ == "__main__":
    api_logger.info("Starting API server")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    api_logger.info("API server stopped") 