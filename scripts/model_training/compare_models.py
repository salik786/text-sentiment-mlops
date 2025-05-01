#!/usr/bin/env python3
"""
Script to compare models in the registry.
"""

import os
import sys
import json
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path to import model_registry
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_training.model_registry import ModelRegistry

def format_value(value):
    """Format a numeric value as a percentage with 2 decimal places"""
    if isinstance(value, (int, float)):
        return f"{value*100:.2f}%"
    return str(value)

def print_comparison_table(models, metrics):
    """
    Print a comparison table of models.
    
    Args:
        models: List of model info dictionaries
        metrics: List of metrics to compare
    """
    # Create the table header
    headers = ["Version", "Description"]
    headers.extend([m.capitalize() for m in metrics])
    headers.append("Hyperparameters")
    
    # Create table rows
    rows = []
    for model in models:
        row = [
            model["version"],
            model.get("description", "")[:30]
        ]
        
        # Add metrics
        for metric in metrics:
            value = model["metrics"].get(metric, model["metrics"].get(f"eval_{metric}", "-"))
            row.append(format_value(value))
        
        # Add hyperparameters summary
        hyper_summary = ", ".join([
            f"{k}={v}" for k, v in model["hyperparameters"].items()
            if k in ["batch_size", "epochs", "learning_rate"]
        ])
        row.append(hyper_summary)
        
        rows.append(row)
    
    # Print table
    print(tabulate(rows, headers=headers, tablefmt="grid"))

def create_comparison_chart(models, metrics, output_file=None):
    """
    Create a bar chart comparing models on specified metrics.
    
    Args:
        models: List of model info dictionaries
        metrics: List of metrics to compare
        output_file: Path to save the chart image
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib not installed. Skipping chart generation.")
        return
    
    # Extract model versions and metric values
    versions = [m["version"] for m in models]
    
    # Create a figure
    plt.figure(figsize=(12, 6))
    
    # Position the bars
    x = np.arange(len(versions))
    width = 0.8 / len(metrics)
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        values = []
        for model in models:
            # Get the metric value (support both raw and eval_ prefixed metrics)
            value = model["metrics"].get(metric, model["metrics"].get(f"eval_{metric}", 0))
            values.append(value)
        
        # Plot the bars
        plt.bar(x + i*width - width*len(metrics)/2 + width/2, 
                values, 
                width, 
                label=metric.capitalize())
    
    # Add labels and legend
    plt.xlabel('Model Version')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, versions, rotation=45)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file)
        print(f"Chart saved to {output_file}")
    else:
        plt.show()

def compare_models(versions=None, metrics=None, chart=False, chart_output=None):
    """
    Compare models from the registry.
    
    Args:
        versions: List of model versions to compare (if None, compare all)
        metrics: List of metrics to compare (if None, use common metrics)
        chart: Whether to generate a chart
        chart_output: Path to save the chart
    """
    # Initialize registry
    registry = ModelRegistry()
    
    # Get models to compare
    models = registry.list_models()
    
    if not models:
        print("No models found in registry.")
        return
    
    # Filter by versions if specified
    if versions:
        models = [m for m in models if m["version"] in versions]
        if not models:
            print(f"No models found matching versions: {versions}")
            return
    
    # Sort models by version
    models = sorted(models, key=lambda m: m["version"])
    
    # Determine metrics to compare
    if not metrics:
        # Get all metrics from all models
        all_metrics = set()
        for model in models:
            all_metrics.update(model.get("metrics", {}).keys())
        
        # Prefer common evaluation metrics
        preferred_metrics = ["accuracy", "f1", "precision", "recall", 
                           "eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]
        
        metrics = []
        # First add preferred metrics in order if they exist
        for metric in preferred_metrics:
            if metric in all_metrics:
                # Remove "eval_" prefix for consistency in display
                clean_metric = metric.replace("eval_", "")
                if clean_metric not in metrics:
                    metrics.append(clean_metric)
        
        # Then add any other metrics
        for metric in all_metrics:
            clean_metric = metric.replace("eval_", "")
            if clean_metric not in metrics and clean_metric != "loss":
                metrics.append(clean_metric)
    
    # Print comparison
    print("\nModel Comparison:")
    print_comparison_table(models, metrics)
    
    # Get active model
    active_model = registry.get_active_model()
    if active_model:
        print(f"\nActive Model: {active_model['version']}")
    
    # Get best model for each metric
    print("\nBest Models by Metric:")
    for metric in metrics:
        best_model = registry.get_best_model(metric)
        if best_model:
            print(f"  {metric.capitalize()}: {best_model['version']} "
                  f"({format_value(best_model['metrics'].get(metric, best_model['metrics'].get(f'eval_{metric}', 0)))})")
    
    # Create chart if requested
    if chart:
        create_comparison_chart(models, metrics, chart_output)

def main():
    """Parse arguments and compare models"""
    parser = argparse.ArgumentParser(description='Compare models in the registry')
    parser.add_argument('--versions', nargs='+', help='Specific model versions to compare')
    parser.add_argument('--metrics', nargs='+', help='Specific metrics to compare')
    parser.add_argument('--chart', action='store_true', help='Generate a comparison chart')
    parser.add_argument('--output', help='Output file for the chart')
    args = parser.parse_args()
    
    try:
        compare_models(args.versions, args.metrics, args.chart, args.output)
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 