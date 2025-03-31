# Text Sentiment Analysis with MLOps

A machine learning project that performs sentiment analysis on text using DistilBERT, with a modern web interface and MLOps practices.

## Project Overview

This project implements a sentiment analysis system that:
- Uses DistilBERT for text classification
- Provides a modern web interface for user interaction
- Implements MLOps best practices
- Includes data processing, model training, and serving pipelines

## Project Structure

```
text-sentiment-mlops/
├── data/
│   └── processed/
│       ├── imdb_train_processed.csv
│       ├── imdb_test_processed.csv
│       └── metadata.json
├── models/
│   └── saved/
│       ├── sentiment_model/
│       └── sentiment_tokenizer/
├── scripts/
│   ├── data_processing/
│   │   └── prepare_data.py
│   ├── model_training/
│   │   ├── preprocess_data.py
│   │   └── train_model.py
│   └── model_serving/
│       ├── api.py
│       ├── predict.py
│       ├── test_api.py
│       └── index.html
├── deployment/            # Docker configuration
│   ├── Dockerfile        # Container definition
│   ├── docker-compose.yml # Container orchestration
│   └── .dockerignore     # Files to exclude from build
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.8+
- PyTorch
- Transformers
- FastAPI
- Pandas
- scikit-learn
- uvicorn
- Docker (for containerized deployment)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd text-sentiment-mlops
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Processing

Process the raw data:
```bash
python scripts/data_processing/prepare_data.py
```

### 2. Model Training

Train the sentiment analysis model:
```bash
python scripts/model_training/train_model.py
```

### 3. Running the Application

#### Local Development

You need to run two servers:

1. Start the FastAPI server (in one terminal):
```bash
python scripts/model_serving/api.py
```
The API will be available at `http://127.0.0.1:8000`

2. Start the web interface server (in another terminal):
```bash
cd scripts/model_serving
python -m http.server 8080
```
The web interface will be available at `http://localhost:8080/index.html`

#### Docker Deployment

1. Build and start the container:
```bash
cd deployment
docker-compose up --build
```

2. Access the application:
- Web interface: `http://localhost:8000`
- API endpoints: `http://localhost:8000/predict` and `http://localhost:8000/health`

The Docker setup includes:
- Automatic model file mounting
- Health checks
- Container restart policy
- Environment variable configuration

### 4. Testing the API

You can test the API using the provided test script:
```bash
python scripts/model_serving/test_api.py
```

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /predict`: Sentiment analysis endpoint
  ```json
  {
    "text": "Your text here"
  }
  ```

## Web Interface Features

The web interface provides:
- Modern, responsive design
- Real-time sentiment analysis
- Visual confidence indicators
- Detailed probability scores
- Loading animations
- Error handling

## Model Details

- Architecture: DistilBERT
- Task: Binary sentiment classification (positive/negative)
- Input: Text
- Output: Sentiment label, confidence score, and probability distribution

## Development

### Adding New Features

1. Data Processing:
   - Modify `scripts/data_processing/prepare_data.py`
   - Add new preprocessing steps as needed

2. Model Training:
   - Modify `scripts/model_training/train_model.py`
   - Adjust hyperparameters or model architecture

3. API:
   - Modify `scripts/model_serving/api.py`
   - Add new endpoints or modify existing ones

4. Web Interface:
   - Modify `scripts/model_serving/index.html`
   - Update the UI/UX as needed

### Testing

Run the test script to verify the API functionality:
```bash
python scripts/model_serving/test_api.py
```

## Troubleshooting

1. Port Already in Use:
   ```bash
   # Find process using port 8080
   lsof -i :8080
   # Kill the process
   kill <PID>
   ```

2. Model Loading Issues:
   - Ensure model files exist in `models/saved/`
   - Check model and tokenizer paths in `predict.py`

3. API Connection Issues:
   - Verify both servers are running
   - Check CORS settings in `api.py`
   - Ensure correct ports are being used

4. Docker Issues:
   - Ensure Docker daemon is running
   - Check container logs: `docker-compose logs`
   - Verify model files are properly mounted
   - Check environment variables in docker-compose.yml

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.