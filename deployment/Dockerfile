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

# Copy model and tokenizer files if they exist
# Use COPY with --chown option and set conditional execution
COPY models/saved/sentiment_model/ /app/models/saved/sentiment_model/ 2>/dev/null || true
COPY models/saved/sentiment_tokenizer/ /app/models/saved/sentiment_tokenizer/ 2>/dev/null || true
COPY models/saved/eval_results.json /app/models/saved/eval_results.json 2>/dev/null || true

# Add a placeholder file to ensure the directories aren't empty
RUN touch /app/models/saved/sentiment_model/.placeholder \
    && touch /app/models/saved/sentiment_tokenizer/.placeholder \
    && echo "{}" > /app/models/saved/eval_results.json

# Copy the application code
COPY scripts/model_serving/api.py /app/api.py
COPY scripts/model_serving/predict.py /app/predict.py
COPY scripts/monitoring/logger.py /app/scripts/monitoring/logger.py || true

# Copy static files and dashboard
COPY scripts/model_serving/static /app/static 2>/dev/null || true
COPY scripts/model_serving/static/index.html /app/static/index.html 2>/dev/null || true
COPY scripts/model_serving/static/dashboard.html /app/static/dashboard.html 2>/dev/null || true

# Create placeholder static files if they don't exist
RUN mkdir -p /app/static \
    && touch /app/static/index.html \
    && touch /app/static/dashboard.html

# Create volume for logs
VOLUME ["/app/logs"]

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]