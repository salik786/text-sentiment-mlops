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

# Create a basic requirements.txt with essential packages
RUN echo "fastapi\nuvicorn\npydantic\nnumpy\npandas\nscikit-learn\ntransformers\ntorch\npython-dotenv\nrequests" > requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create placeholder files for the models
RUN touch /app/models/saved/sentiment_model/.placeholder \
    && touch /app/models/saved/sentiment_tokenizer/.placeholder \
    && echo "{}" > /app/models/saved/eval_results.json

# Copy application code first
COPY scripts/model_serving/api.py /app/api.py
COPY scripts/model_serving/predict.py /app/predict.py
COPY scripts/monitoring/logger.py /app/scripts/monitoring/logger.py

# Copy static files if they exist
COPY scripts/model_serving/static/ /app/static/

# Create a script to handle model file copying
RUN echo '#!/bin/bash\n\
if [ -d "models/saved/sentiment_model" ]; then\n\
    cp -r models/saved/sentiment_model/* /app/models/saved/sentiment_model/\n\
fi\n\
if [ -d "models/saved/sentiment_tokenizer" ]; then\n\
    cp -r models/saved/sentiment_tokenizer/* /app/models/saved/sentiment_tokenizer/\n\
fi\n\
if [ -f "models/saved/eval_results.json" ]; then\n\
    cp models/saved/eval_results.json /app/models/saved/\n\
fi' > /app/copy_models.sh && chmod +x /app/copy_models.sh

# Create volume for logs
VOLUME ["/app/logs"]

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]