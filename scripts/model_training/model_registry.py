import os
import json
from datetime import datetime
from pathlib import Path

class ModelRegistry:
    """
    A registry for tracking model versions, metrics, and hyperparameters.
    """
    
    def __init__(self, registry_path=None):
        """
        Initialize the model registry.
        
        Args:
            registry_path: Path to the registry JSON file.
                           If None, defaults to 'models/registry.json'.
        """
        self.registry_path = registry_path or "models/registry.json"
        self.registry = self._load_registry()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
    
    def _load_registry(self):
        """
        Load the registry from disk.
        
        Returns:
            The registry as a dictionary.
        """
        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize with an empty registry
            return {
                "models": {},
                "active_model": None,
                "last_updated": datetime.now().isoformat()
            }
    
    def _save_registry(self):
        """
        Save the registry to disk.
        """
        # Update last updated timestamp
        self.registry["last_updated"] = datetime.now().isoformat()
        
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)
        
        print(f"Registry saved to {self.registry_path}")
    
    def register_model(self, model_path, version, metrics=None, hyperparameters=None, 
                      description=None, set_as_active=False):
        """
        Register a model in the registry.
        
        Args:
            model_path: Path to the model directory
            version: Model version identifier
            metrics: Dictionary of model metrics (accuracy, f1, etc.)
            hyperparameters: Dictionary of hyperparameters
            description: Optional description of the model
            set_as_active: Whether to set this model as the active model
        
        Returns:
            The registered model info dictionary
        """
        # Ensure model directory exists
        if not os.path.exists(model_path):
            raise ValueError(f"Model path {model_path} does not exist")
            
        # Create model entry
        model_info = {
            "version": version,
            "path": model_path,
            "metrics": metrics or {},
            "hyperparameters": hyperparameters or {},
            "description": description or f"Model version {version}",
            "registered_at": datetime.now().isoformat()
        }
        
        # Add to registry
        self.registry["models"][version] = model_info
        
        # Set as active if requested or if it's the first model
        if set_as_active or self.registry["active_model"] is None:
            self.registry["active_model"] = version
        
        # Save registry
        self._save_registry()
        
        return model_info
    
    def get_model_info(self, version):
        """
        Get information about a specific model.
        
        Args:
            version: Model version identifier
            
        Returns:
            Model info dictionary or None if not found
        """
        return self.registry["models"].get(version)
    
    def list_models(self):
        """
        List all registered models.
        
        Returns:
            List of model info dictionaries
        """
        return list(self.registry["models"].values())
    
    def set_active_model(self, version):
        """
        Set the active model version.
        
        Args:
            version: Model version identifier
            
        Returns:
            True if successful, False otherwise
        """
        if version not in self.registry["models"]:
            print(f"Model version {version} not found in registry")
            return False
        
        self.registry["active_model"] = version
        self._save_registry()
        return True
    
    def get_active_model(self):
        """
        Get the active model.
        
        Returns:
            The active model info dictionary or None if no active model
        """
        active_version = self.registry["active_model"]
        if active_version:
            return self.registry["models"].get(active_version)
        return None
    
    def delete_model(self, version):
        """
        Delete a model from the registry.
        
        Args:
            version: Model version identifier
            
        Returns:
            True if successful, False otherwise
        """
        if version not in self.registry["models"]:
            print(f"Model version {version} not found in registry")
            return False
        
        # Remove from registry
        del self.registry["models"][version]
        
        # Update active model if needed
        if self.registry["active_model"] == version:
            # Set to None or most recent model
            if self.registry["models"]:
                # Get most recent model by registration time
                recent_model = max(
                    self.registry["models"].values(),
                    key=lambda m: m["registered_at"]
                )
                self.registry["active_model"] = recent_model["version"]
            else:
                self.registry["active_model"] = None
        
        self._save_registry()
        return True
    
    def get_best_model(self, metric="accuracy"):
        """
        Get the best model according to a specific metric.
        
        Args:
            metric: Metric to use for comparison (default: 'accuracy')
            
        Returns:
            The best model info dictionary or None if no models
        """
        if not self.registry["models"]:
            return None
        
        # Find model with highest metric value
        try:
            best_model = max(
                self.registry["models"].values(),
                key=lambda m: m["metrics"].get(metric, 0)
            )
            return best_model
        except ValueError:
            return None
    
    def compare_models(self, versions=None, metrics=None):
        """
        Compare models based on specific metrics.
        
        Args:
            versions: List of model versions to compare. If None, use all models.
            metrics: List of metrics to compare. If None, use all metrics.
            
        Returns:
            Dictionary of comparison results
        """
        # Get models to compare
        if versions is None:
            models = list(self.registry["models"].values())
        else:
            models = [
                self.registry["models"][v] for v in versions 
                if v in self.registry["models"]
            ]
        
        if not models:
            return {}
        
        # Get metrics to compare
        if metrics is None:
            # Use all metrics from first model
            all_metrics = set()
            for model in models:
                all_metrics.update(model["metrics"].keys())
            metrics = list(all_metrics)
        
        # Build comparison results
        results = {
            "models": [m["version"] for m in models],
            "metrics": {}
        }
        
        # For each metric, get values for all models
        for metric in metrics:
            results["metrics"][metric] = {
                m["version"]: m["metrics"].get(metric, None) for m in models
            }
        
        return results


# Example usage
if __name__ == "__main__":
    registry = ModelRegistry()
    
    # Register a model
    registry.register_model(
        model_path="models/versions/v1",
        version="v1",
        metrics={"accuracy": 0.82, "f1": 0.84},
        hyperparameters={"batch_size": 16, "epochs": 3},
        description="First model version"
    )
    
    # List all models
    print("Registered models:")
    for model in registry.list_models():
        print(f"- {model['version']}: {model['metrics']}")
    
    # Get active model
    active_model = registry.get_active_model()
    print(f"Active model: {active_model['version'] if active_model else None}")
    
    # Get best model by accuracy
    best_model = registry.get_best_model(metric="accuracy")
    print(f"Best model: {best_model['version'] if best_model else None}") 