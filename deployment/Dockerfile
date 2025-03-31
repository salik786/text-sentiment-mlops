# Use Python 3.9 as base image
FROM python:3.9-slim

# Set working directory
WORKDIR /project

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model serving code and index.html
COPY scripts/model_serving /project/scripts/model_serving

# Set Python path
ENV PYTHONPATH="${PYTHONPATH}:/project"

# Set the entry point
WORKDIR /project/scripts/model_serving
CMD ["python", "api.py"]

# Expose the port
EXPOSE 8000 