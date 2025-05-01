#!/usr/bin/env python3
"""
Train a model with real IMDB data using a specified fraction of the dataset.
This version avoids TensorFlow dependencies.
"""

import os
import sys
import json
import argparse
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import datetime

# Add parent directory to path to import model_registry
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_training.model_registry import ModelRegistry

def compute_metrics(pred):
    """
    Compute metrics for model evaluation.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def load_parameters(param_file):
    """
    Load parameters from a JSON file.
    
    Args:
        param_file: Path to the parameter file
        
    Returns:
        Dictionary of parameters
    """
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Parameter file {param_file} not found")
    
    with open(param_file, 'r') as f:
        try:
            params = json.load(f)
            return params
        except json.JSONDecodeError:
            raise ValueError(f"Error parsing parameter file {param_file}. File must be valid JSON.")

def train_with_real_data(hyperparameters, version):
    """
    Train and evaluate a model with real data.
    
    Args:
        hyperparameters: Dictionary of hyperparameters
        version: Version string for the model
        
    Returns:
        Dictionary with model metrics
    """
    print(f"\n{'='*80}")
    print(f"Training model version {version} with hyperparameters:")
    for k, v in hyperparameters.items():
        print(f"  {k}: {v}")
    print(f"{'='*80}\n")
    
    # Create model directory based on version
    model_dir = f"models/versions/{version}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Load processed data
    print("Loading processed IMDB data...")
    train_df = pd.read_csv("data/processed/imdb_train_processed.csv")
    test_df = pd.read_csv("data/processed/imdb_test_processed.csv")
    
    # Use only a fraction of the data if specified
    data_fraction = hyperparameters.get("data_fraction", 1.0)
    if data_fraction < 1.0:
        print(f"Using {data_fraction*100:.1f}% of the training data")
        train_size = int(len(train_df) * data_fraction)
        test_size = int(len(test_df) * data_fraction)
        train_df = train_df.sample(train_size, random_state=42)
        test_df = test_df.sample(test_size, random_state=42)
    
    print(f"Training with {len(train_df)} examples, evaluating with {len(test_df)} examples")
    
    # Convert sentiment to numeric labels
    label_map = {'positive': 1, 'negative': 0}
    train_df['labels'] = train_df['sentiment'].map(label_map).astype('int64')
    test_df['labels'] = test_df['sentiment'].map(label_map).astype('int64')
    
    # Keep only necessary columns
    train_df = train_df[['text', 'labels']]
    test_df = test_df[['text', 'labels']]
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load model
    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=hyperparameters.get("max_length", 512)
        )
    
    # Tokenize and format datasets
    print("Tokenizing data...")
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Set format for pytorch
    train_dataset.set_format('torch')
    test_dataset.set_format('torch')
    
    # Define training arguments
    epochs = hyperparameters.get("epochs", 3)
    batch_size = hyperparameters.get("batch_size", 16)
    learning_rate = hyperparameters.get("learning_rate", 5e-5)
    
    print(f"Setting up training with {epochs} epochs, batch size {batch_size}, learning rate {learning_rate}")
    
    training_args = TrainingArguments(
        output_dir=f"{model_dir}/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=hyperparameters.get("warmup_steps", 500),
        weight_decay=hyperparameters.get("weight_decay", 0.01),
        logging_dir=f"{model_dir}/logs",
        logging_steps=hyperparameters.get("logging_steps", 50),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        metric_for_best_model=hyperparameters.get("metric_for_best_model", "accuracy"),
        greater_is_better=True,
    )
    
    # Create Trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train model
    print("Training model...")
    trainer.train()
    
    # Evaluate model
    print("Evaluating model...")
    eval_result = trainer.evaluate()
    
    # Save model
    print(f"Saving model to {model_dir}...")
    model.save_pretrained(f"{model_dir}/model")
    tokenizer.save_pretrained(f"{model_dir}/tokenizer")
    
    # Save hyperparameters and evaluation results
    with open(f"{model_dir}/hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)
        
    with open(f"{model_dir}/eval_results.json", "w") as f:
        json.dump(eval_result, f, indent=2)
    
    print(f"Model {version} training complete.")
    print(f"Evaluation results: {eval_result}")
    
    return eval_result

def train_with_params(param_file, version=None):
    """
    Train a model using parameters from a file and register it.
    
    Args:
        param_file: Path to the parameter file
        version: Version identifier (optional)
        
    Returns:
        The registered model info
    """
    # Load parameters
    print(f"Loading parameters from {param_file}...")
    params = load_parameters(param_file)
    
    # Generate version if not provided
    if version is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = os.path.basename(param_file).split('.')[0]
        version = f"{basename}_{timestamp}"
    
    print(f"Training model version: {version}")
    
    # Train and evaluate model with real data
    metrics = train_with_real_data(params, version)
    
    # Register model
    registry = ModelRegistry()
    model_info = registry.register_model(
        model_path=f"models/versions/{version}",
        version=version,
        metrics=metrics,
        hyperparameters=params,
        description=f"Model trained with real data using parameters from {os.path.basename(param_file)}",
        set_as_active=True
    )
    
    print(f"\nModel {version} registered successfully:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
    print(f"  Model path: {model_info['path']}")
    
    return model_info

def main():
    """Parse arguments and train model with parameters"""
    parser = argparse.ArgumentParser(description='Train a model using real data with parameters from a file')
    parser.add_argument('param_file', help='Path to the parameter JSON file')
    parser.add_argument('--version', help='Version identifier (optional)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()
    
    # Set CUDA device if requested and available
    if args.gpu and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Using CPU for training")
    
    try:
        model_info = train_with_params(args.param_file, args.version)
        print("\nTraining completed successfully.")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 