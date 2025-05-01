#!/usr/bin/env python3
"""
A simplified training script that loads parameters but mocks the actual training
for demo purposes.
"""

import os
import sys
import json
import argparse
import random
import time
from pathlib import Path
import datetime

# Add parent directory to path to import model_registry
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_training.model_registry import ModelRegistry

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

def mock_training(hyperparameters, version):
    """
    Mock training a model with specific hyperparameters.
    For demo purposes only - doesn't actually train a model.
    
    Args:
        hyperparameters: Dictionary of hyperparameters
        version: Version string for the model
        
    Returns:
        Dictionary with model metrics (mock values)
    """
    print(f"\n{'='*80}")
    print(f"Training model version {version} with hyperparameters:")
    for k, v in hyperparameters.items():
        print(f"  {k}: {v}")
    print(f"{'='*80}\n")
    
    # Create model directory based on version
    model_dir = f"models/versions/{version}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Simulate training time
    total_epochs = hyperparameters.get("epochs", 3)
    for epoch in range(1, total_epochs + 1):
        print(f"Epoch {epoch}/{total_epochs}")
        time.sleep(1)  # Simulating training time
        
        # Simulate epoch metrics
        train_loss = 0.5 - 0.1 * epoch + random.random() * 0.1
        print(f"  Training loss: {train_loss:.4f}")
        
    # Simulate final metrics based on hyperparameters
    # Higher batch size slightly improves metrics
    batch_size_factor = min(hyperparameters.get("batch_size", 16) / 16, 1.2)
    
    # More epochs improves metrics but with diminishing returns
    epochs_factor = min(1.0 + (hyperparameters.get("epochs", 3) - 3) * 0.03, 1.15)
    
    # Learning rate effect (3e-5 is "optimal" in this mock)
    lr = hyperparameters.get("learning_rate", 5e-5)
    lr_factor = 1.0 - abs(lr - 3e-5) * 1000
    
    # Weight decay effect (0.05 is "optimal" in this mock)
    wd = hyperparameters.get("weight_decay", 0.01)
    wd_factor = 1.0 - abs(wd - 0.05) * 0.5
    
    # Base metrics with some randomness
    base_accuracy = 0.82 + random.random() * 0.02
    
    # Combine factors
    accuracy = min(base_accuracy * batch_size_factor * epochs_factor * lr_factor * wd_factor, 0.95)
    f1 = accuracy - 0.02 + random.random() * 0.04
    precision = accuracy - 0.03 + random.random() * 0.06
    recall = accuracy + 0.01 + random.random() * 0.04
    
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'loss': 0.2 + (1.0 - accuracy) * 0.5
    }
    
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save mock model files
    with open(f"{model_dir}/model.bin", "w") as f:
        f.write(f"Mock model file for version {version}")
    
    # Save hyperparameters and metrics
    with open(f"{model_dir}/hyperparameters.json", "w") as f:
        json.dump(hyperparameters, f, indent=2)
    
    with open(f"{model_dir}/eval_results.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nModel {version} training complete.")
    print(f"Model saved to {model_dir}")
    
    return metrics

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
    
    # Train and evaluate model (mocked)
    metrics = mock_training(params, version)
    
    # Register model
    registry = ModelRegistry()
    model_info = registry.register_model(
        model_path=f"models/versions/{version}",
        version=version,
        metrics=metrics,
        hyperparameters=params,
        description=f"Model trained with parameters from {os.path.basename(param_file)}",
        set_as_active=True
    )
    
    print(f"\nModel {version} registered successfully:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  F1 Score: {metrics.get('f1', 0):.4f}")
    print(f"  Model path: {model_info['path']}")
    
    return model_info

def main():
    """Parse arguments and train model with parameters"""
    parser = argparse.ArgumentParser(description='Train a model using parameters from a file')
    parser.add_argument('param_file', help='Path to the parameter JSON file')
    parser.add_argument('--version', help='Version identifier (optional)')
    args = parser.parse_args()
    
    try:
        model_info = train_with_params(args.param_file, args.version)
        print("\nTraining completed successfully.")
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 