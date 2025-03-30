# Text Sentiment Analysis MLOps Project

This project implements a machine learning pipeline for sentiment analysis using the IMDB movie reviews dataset. It follows MLOps best practices with proper data processing, model training, and deployment pipelines.

## Project Structure

```
text-sentiment-mlops/
├── data/
│   ├── raw/           # Raw input data
│   └── processed/     # Processed/cleaned data
├── models/            # Saved model files
├── scripts/
│   └── data_processing/  # Data processing scripts
└── venv/              # Python virtual environment
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/salik786/text-sentiment-mlops.git
cd text-sentiment-mlops
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The project uses the IMDB movie reviews dataset. To prepare the data:

1. Run the data processing script:
```bash
python scripts/data_processing/prepare_data.py
```

This will:
- Download the IMDB dataset
- Save raw data in `data/raw/`
- Process and save cleaned data in `data/processed/`
- Generate metadata in `data/processed/metadata.json`

## Project Components

### Data Processing
- `scripts/data_processing/prepare_data.py`: Downloads and prepares the IMDB dataset
- Handles text preprocessing and feature engineering
- Creates train/test splits
- Generates metadata

### Model Training (Coming Soon)
- Will include model training scripts
- Model evaluation and metrics
- Model saving and versioning

### Model Serving (Coming Soon)
- FastAPI service for model inference
- API documentation
- Docker containerization

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