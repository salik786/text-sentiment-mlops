import os
import json
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from pathlib import Path
import itertools
import datetime
import time

# Import our model registry
from model_registry import ModelRegistry

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

def train_and_evaluate(hyperparameters, version):
    """
    Train and evaluate a model with specific hyperparameters.
    
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
    train_df = pd.read_csv("data/processed/imdb_train_processed.csv")
    test_df = pd.read_csv("data/processed/imdb_test_processed.csv")
    
    # For quick testing, limit the dataset size
    # Remove these lines for full training
    if hyperparameters.get("sample_size"):
        sample_size = hyperparameters["sample_size"]
        train_df = train_df.sample(sample_size, random_state=42)
        test_df = test_df.sample(int(sample_size * 0.2), random_state=42)
    
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
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load model
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
    train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    
    # Set format for pytorch
    train_dataset.set_format('torch')
    test_dataset.set_format('torch')
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"{model_dir}/checkpoints",
        num_train_epochs=hyperparameters.get("epochs", 3),
        per_device_train_batch_size=hyperparameters.get("batch_size", 8),
        per_device_eval_batch_size=hyperparameters.get("batch_size", 8),
        warmup_steps=hyperparameters.get("warmup_steps", 500),
        weight_decay=hyperparameters.get("weight_decay", 0.01),
        logging_dir=f"{model_dir}/logs",
        logging_steps=hyperparameters.get("logging_steps", 50),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=hyperparameters.get("metric_for_best_model", "accuracy"),
        greater_is_better=True,
    )
    
    # Create Trainer
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

def test_hyperparameters(param_grid, base_version="exp"):
    """
    Test different combinations of hyperparameters.
    
    Args:
        param_grid: Dictionary where keys are parameter names and values are lists of values to try
        base_version: Base string for version identifiers
    """
    # Initialize model registry
    registry = ModelRegistry()
    
    # Generate all combinations of parameters
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Use current timestamp as experiment group identifier
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Track best model
    best_model = None
    best_accuracy = 0.0
    
    # Loop through all combinations
    for i, values in enumerate(itertools.product(*param_values)):
        # Create parameters dictionary
        params = dict(zip(param_names, values))
        
        # Create version identifier
        version = f"{base_version}_{timestamp}_{i+1}"
        
        try:
            # Train and evaluate model
            metrics = train_and_evaluate(params, version)
            
            # Register model
            registry.register_model(
                model_path=f"models/versions/{version}",
                version=version,
                metrics=metrics,
                hyperparameters=params,
                description=f"Hyperparameter test {i+1}"
            )
            
            # Check if this is the best model so far
            if metrics.get('accuracy', 0) > best_accuracy:
                best_accuracy = metrics.get('accuracy', 0)
                best_model = version
                
            print(f"\nModel {version} registered with accuracy: {metrics.get('accuracy', 0):.4f}")
            
        except Exception as e:
            print(f"Error training model {version}: {str(e)}")
    
    # Set the best model as active
    if best_model:
        registry.set_active_model(best_model)
        print(f"\nSet best model {best_model} as active (accuracy: {best_accuracy:.4f})")
    
    # Generate comparison report
    print("\nHyperparameter Comparison Report:")
    print(f"{'='*80}")
    print(f"{'Version':<20} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Hyperparameters'}")
    print(f"{'-'*80}")
    
    for model in registry.list_models():
        if model['version'].startswith(f"{base_version}_{timestamp}"):
            v = model['version']
            m = model['metrics']
            h = ", ".join([f"{k}={v}" for k, v in model['hyperparameters'].items()])
            print(f"{v[-5:]:<20} {m.get('accuracy', 0):<10.4f} {m.get('f1', 0):<10.4f} "
                  f"{m.get('precision', 0):<10.4f} {m.get('recall', 0):<10.4f} {h}")
    
    print(f"{'='*80}")
    
    return registry.get_best_model()

if __name__ == "__main__":
    # Define hyperparameter grid
    param_grid = {
        "batch_size": [8, 16],
        "epochs": [2, 3],
        "learning_rate": [2e-5, 5e-5],
        "max_length": [128, 256],
        # Limit dataset size for quick testing
        "sample_size": [500]  # Remove this for full training
    }
    
    # Test hyperparameters
    best_model = test_hyperparameters(param_grid, "distilbert") 