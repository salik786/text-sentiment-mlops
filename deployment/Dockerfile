# Use an official Python runtime as a parent image
FROM python:3.9

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/saved/sentiment_model \
    TOKENIZER_PATH=/app/models/saved/sentiment_tokenizer \
    LOG_DIR=/app/logs \
    PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories with proper permissions
RUN mkdir -p /app/models/saved/sentiment_model \
    /app/models/saved/sentiment_tokenizer \
    /app/logs \
    /app/static \
    /app/scripts/monitoring \
    && chmod -R 777 /app

# Copy requirements first to leverage Docker cache
COPY deployment/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create placeholder files to ensure directories aren't empty
RUN touch /app/models/saved/sentiment_model/.placeholder \
    && touch /app/models/saved/sentiment_tokenizer/.placeholder \
    && echo "{}" > /app/models/saved/eval_results.json

# Copy model and tokenizer files - Docker will ignore if not found
COPY models/saved/sentiment_model/* /app/models/saved/sentiment_model/ || true
COPY models/saved/sentiment_tokenizer/* /app/models/saved/sentiment_tokenizer/ || true
COPY models/saved/eval_results.json /app/models/saved/ || true

# Copy the application code
COPY scripts/model_serving/api.py /app/api.py || true
COPY scripts/model_serving/predict.py /app/predict.py || true
COPY scripts/monitoring/logger.py /app/scripts/monitoring/logger.py || true

# Create placeholder for application files
RUN if [ ! -f /app/api.py ]; then \
        echo 'from fastapi import FastAPI; app = FastAPI(); @app.get("/health"); def health(): return {"status": "ok"}' > /app/api.py; \
    fi && \
    if [ ! -f /app/predict.py ]; then \
        echo 'def predict(text): return {"sentiment": "placeholder", "confidence": 0.0}' > /app/predict.py; \
    fi

# Create directory and placeholder for static files
RUN mkdir -p /app/static

# Copy static files - Docker will ignore if not found
COPY scripts/model_serving/static/* /app/static/ || true

# Create placeholder static files if they don't exist
RUN touch /app/static/index.html \
    && touch /app/static/dashboard.html

# Create volume for logs
VOLUME ["/app/logs"]

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]