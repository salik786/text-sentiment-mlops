import logging
import time
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict

# Create logs directory if it doesn't exist
logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)

# Configure logger
def setup_logger(name):
    """Set up logger with proper formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    log_filename = logs_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create specific loggers
api_logger = setup_logger("api")
model_logger = setup_logger("model")
performance_logger = setup_logger("performance")

class ModelMonitor:
    """Class to monitor model predictions and performance"""
    
    def __init__(self):
        self.predictions = []
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        self.confidences = []
        self.text_lengths = []
        self.sentiment_distribution = {'positive': 0, 'negative': 0}  # Initialize with both sentiments
        self.drift_score = 0.0
        self.last_drift_check = None
        self.average_response_time = 0.0
        self.p95_response_time = 0.0
        self.p99_response_time = 0.0
        self.average_confidence = 0.0
        self.average_text_length = 0.0
        
    def log_prediction(self, text: str, prediction: str, confidence: float, response_time: float):
        """Log a prediction with its metadata"""
        self.total_requests += 1
        self.successful_requests += 1
        
        # Store prediction data
        self.predictions.append({
            'text': text,
            'prediction': prediction,
            'confidence': confidence,
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update metrics
        self.response_times.append(response_time)
        self.confidences.append(confidence)
        self.text_lengths.append(len(text))
        
        # Update sentiment distribution
        if prediction in self.sentiment_distribution:
            self.sentiment_distribution[prediction] += 1
        
        # Calculate averages
        if self.response_times:
            self.average_response_time = np.mean(self.response_times)
            self.p95_response_time = np.percentile(self.response_times, 95)
            self.p99_response_time = np.percentile(self.response_times, 99)
        
        if self.confidences:
            self.average_confidence = np.mean(self.confidences)
        
        if self.text_lengths:
            self.average_text_length = np.mean(self.text_lengths)
        
        # Normalize sentiment distribution
        total = sum(self.sentiment_distribution.values())
        if total > 0:
            self.sentiment_distribution = {
                k: v/total for k, v in self.sentiment_distribution.items()
            }
        
        # Check for drift every 100 predictions
        if len(self.predictions) % 100 == 0:
            self.check_drift()
        
        # Log to file
        api_logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}, Response time: {response_time:.3f}s")
    
    def check_drift(self):
        """Check for potential data drift in predictions"""
        if len(self.predictions) < 100:
            return
        
        # Calculate drift score based on recent predictions
        recent_predictions = self.predictions[-100:]
        recent_confidences = [p['confidence'] for p in recent_predictions]
        
        # Compare with historical data
        if len(self.confidences) > 100:
            historical_confidences = self.confidences[:-100]
            self.drift_score = abs(np.mean(recent_confidences) - np.mean(historical_confidences))
            self.last_drift_check = datetime.now()
            
            if self.drift_score > 0.1:  # Threshold for drift detection
                api_logger.warning(f"Potential drift detected! Score: {self.drift_score:.3f}")
    
    def get_statistics(self):
        """Get current statistics about model performance"""
        return {
            'total_predictions': len(self.predictions),
            'average_response_time': self.average_response_time,
            'p95_response_time': self.p95_response_time,
            'p99_response_time': self.p99_response_time,
            'average_confidence': self.average_confidence,
            'average_text_length': self.average_text_length,
            'sentiment_distribution': self.sentiment_distribution,
            'drift_score': self.drift_score,
            'last_drift_check': self.last_drift_check
        }

# Create global monitor instance
monitor = ModelMonitor()
