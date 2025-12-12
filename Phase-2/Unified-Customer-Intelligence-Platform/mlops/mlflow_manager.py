import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import joblib
from pathlib import Path
import os

class MLflowManager:
    def __init__(self, tracking_uri="sqlite:///mlflow.db"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def log_model(self, model, model_name, metrics, params, artifacts=None):
        """Log model with MLflow"""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            if hasattr(model, 'predict'):
                mlflow.sklearn.log_model(model, model_name)
            
            # Log artifacts
            if artifacts:
                for artifact_path, local_path in artifacts.items():
                    mlflow.log_artifact(local_path, artifact_path)
            
            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
            mlflow.register_model(model_uri, model_name)
            
            return mlflow.active_run().info.run_id
    
    def load_model(self, model_name, version="latest"):
        """Load model from MLflow registry"""
        try:
            if version == "latest":
                model_version = self.client.get_latest_versions(
                    model_name, stages=["Production", "Staging"]
                )[0]
            else:
                model_version = self.client.get_model_version(model_name, version)
            
            return mlflow.sklearn.load_model(model_version.source)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return None
    
    def promote_model(self, model_name, version, stage="Production"):
        """Promote model to production"""
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
    
    def compare_models(self, model_name, versions):
        """Compare model versions"""
        results = []
        for version in versions:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)
            results.append({
                "version": version,
                "metrics": run.data.metrics,
                "params": run.data.params
            })
        return results

mlflow_manager = MLflowManager()