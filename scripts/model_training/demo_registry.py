#!/usr/bin/env python3
"""
Demo script to test the model registry.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path to import model_registry
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_training.model_registry import ModelRegistry

def create_demo_model_dir(version, accuracy, f1):
    """Create a mock model directory for demo purposes"""
    model_dir = f"models/versions/{version}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create mock model files
    with open(f"{model_dir}/model.bin", "w") as f:
        f.write(f"Mock model file for version {version}")
    
    # Create mock evaluation results
    eval_results = {
        "eval_accuracy": accuracy,
        "eval_f1": f1,
        "eval_loss": 0.1 + (1.0 - accuracy) / 2,
        "eval_precision": accuracy * 0.95,
        "eval_recall": accuracy * 1.05,
    }
    
    with open(f"{model_dir}/eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)
    
    # Create mock hyperparameters
    hyperparams = {
        "batch_size": 16,
        "epochs": 3,
        "learning_rate": 5e-5,
        "max_length": 512,
        "model_type": "distilbert"
    }
    
    with open(f"{model_dir}/hyperparameters.json", "w") as f:
        json.dump(hyperparams, f, indent=2)
    
    return model_dir, eval_results, hyperparams

def main():
    """Run the model registry demo"""
    print("Running Model Registry Demo\n")
    
    # Initialize registry
    registry = ModelRegistry()
    
    # Create a few model versions with different metrics
    print("Creating mock model directories and files...")
    
    # Model 1 - baseline model
    base_date = datetime.now() - timedelta(days=5)
    v1_dir, v1_metrics, v1_params = create_demo_model_dir("v1", 0.82, 0.80)
    
    # Model 2 - improved model with better accuracy
    v2_dir, v2_metrics, v2_params = create_demo_model_dir("v2", 0.85, 0.83)
    
    # Model 3 - better f1 score but similar accuracy
    v3_dir, v3_metrics, v3_params = create_demo_model_dir("v3", 0.84, 0.86)
    
    print("\nRegistering models in registry...")
    
    # Register the models
    registry.register_model(
        model_path=v1_dir,
        version="v1",
        metrics=v1_metrics,
        hyperparameters=v1_params,
        description="Baseline DistilBERT model"
    )
    
    registry.register_model(
        model_path=v2_dir,
        version="v2",
        metrics=v2_metrics,
        hyperparameters=v2_params,
        description="Improved DistilBERT model with higher accuracy",
        set_as_active=True
    )
    
    registry.register_model(
        model_path=v3_dir,
        version="v3",
        metrics=v3_metrics,
        hyperparameters={**v2_params, "weight_decay": 0.1},
        description="Variant with better F1 score"
    )
    
    # List all models
    print("\nListing all registered models:")
    print(f"{'='*80}")
    print(f"{'Version':<10} {'Accuracy':<10} {'F1':<10} {'Path':<30} {'Description'}")
    print(f"{'-'*80}")
    
    for model in registry.list_models():
        print(f"{model['version']:<10} "
              f"{model['metrics'].get('eval_accuracy', 0):<10.4f} "
              f"{model['metrics'].get('eval_f1', 0):<10.4f} "
              f"{model['path']:<30} "
              f"{model['description'][:40]}")
    
    # Get the active model
    active = registry.get_active_model()
    print(f"\nActive model: {active['version'] if active else 'None'}")
    
    # Get the best model by accuracy
    best_acc = registry.get_best_model(metric="eval_accuracy")
    print(f"Best model by accuracy: {best_acc['version']} (accuracy: {best_acc['metrics'].get('eval_accuracy'):.4f})")
    
    # Get the best model by F1
    best_f1 = registry.get_best_model(metric="eval_f1")
    print(f"Best model by F1: {best_f1['version']} (F1: {best_f1['metrics'].get('eval_f1'):.4f})")
    
    # Compare models
    print("\nComparing all models:")
    comparison = registry.compare_models(metrics=["eval_accuracy", "eval_f1", "eval_precision", "eval_recall"])
    
    for metric, values in comparison["metrics"].items():
        print(f"\n{metric}:")
        for version, value in values.items():
            print(f"  {version}: {value:.4f}")
    
    print("\nDemo completed successfully!")
    print(f"Registry saved at: {registry.registry_path}")
    
if __name__ == "__main__":
    main() 