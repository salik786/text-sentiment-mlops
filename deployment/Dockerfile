# Use an official Python runtime as a parent image
FROM python:3.11

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

# Create a basic requirements.txt if it doesn't exist
RUN echo "fastapi\nuvicorn\npydantic\nnumpy\npandas\nscikit-learn\ntransformers\ntorch" > requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create placeholder files for the models
RUN touch /app/models/saved/sentiment_model/.placeholder \
    && touch /app/models/saved/sentiment_tokenizer/.placeholder \
    && echo "{}" > /app/models/saved/eval_results.json

# Copy model files if they exist
COPY models/saved/sentiment_model/ /app/models/saved/sentiment_model/
COPY models/saved/sentiment_tokenizer/ /app/models/saved/sentiment_tokenizer/
COPY models/saved/eval_results.json /app/models/saved/

# Copy application code
COPY scripts/model_serving/api.py /app/api.py
COPY scripts/model_serving/predict.py /app/predict.py
COPY scripts/monitoring/logger.py /app/scripts/monitoring/logger.py

# Copy static files if they exist
COPY scripts/model_serving/static/ /app/static/

# Create volume for logs
VOLUME ["/app/logs"]

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]