# 1. Use a lightweight Python base image
FROM python:3.9-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Install build dependencies (if needed)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# 4. Copy and install Python dependencies
# Adjust path since Dockerfile is in deployment/
COPY scripts/API/requirements.txt ./
COPY scripts/API/ ./
RUN pip install --no-cache-dir -r requirements.txt


# 6. Expose the port your FastAPI uses
EXPOSE 8080

# 7. Default environment variables (overrideable at run-time)
ENV MLFLOW_TRACKING_URI=https://dagshub.com/saleemsalik786/my-first-repo.mlflow \
    REGISTERED_MODEL_NAME=top_perform_model \
    MODEL_STAGE=latest

# 8. Default command to start the application
CMD ["uvicorn", "scripts.API.main:app", "--host", "0.0.0.0", "--port", "8080"]