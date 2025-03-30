# Text Sentiment Analysis MLOps Project

This project implements a machine learning pipeline for sentiment analysis using the IMDB movie reviews dataset. It follows MLOps best practices and includes components for data processing, model training, and model serving.

## Project Structure

```
text-sentiment-mlops/
├── data/
│   ├── raw/               # Raw IMDB dataset files
│   └── processed/         # Processed and cleaned data
├── models/
│   └── saved/            # Saved trained models and tokenizers
├── scripts/
│   ├── data_processing/  # Data preprocessing scripts
│   └── model_training/   # Model training scripts
└── requirements.txt      # Project dependencies
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/salik786/text-sentiment-mlops.git
   cd text-sentiment-mlops
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. Run the data processing script to download and prepare the IMDB dataset:
   ```bash
   python scripts/data_processing/preprocess_data.py
   ```

This will:
- Download the IMDB dataset
- Clean and preprocess the text data
- Convert labels to sentiment values
- Save processed data in CSV format

## Project Components

### 1. Data Processing
- Text cleaning and preprocessing
- Label conversion
- Dataset statistics generation

### 2. Model Training
The project uses DistilBERT for sentiment classification with the following features:
- Fine-tuning on IMDB dataset
- Training metrics: accuracy, F1-score, precision, recall
- Model checkpointing and evaluation
- Training progress logging

Current model performance:
- Accuracy: 80%
- F1-score: 0.74
- Precision: 0.95
- Recall: 0.61

### 3. Model Serving (Coming Soon)
- FastAPI-based REST API
- Model inference endpoints
- Input validation and error handling

## Dependencies

Core libraries:
- numpy
- pandas
- scikit-learn
- torch
- transformers
- datasets

Development tools:
- jupyter (for notebook experimentation)
- fastapi (for model serving)
- uvicorn (for API server)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.