#!/usr/bin/env python3
"""
Train a model using parameters from a file and register the results.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import datetime

# Add parent directory to path to import model_registry
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_training.model_registry import ModelRegistry
from model_training.hyperparameter_tester import train_and_evaluate

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
    
    # Train and evaluate model
    metrics = train_and_evaluate(params, version)
    
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