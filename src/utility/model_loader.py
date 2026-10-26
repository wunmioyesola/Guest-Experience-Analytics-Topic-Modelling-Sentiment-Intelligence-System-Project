import mlflow
from mlflow.tracking import MlflowClient
import dagshub
import os
from src.logger import configure_logger
from src.exception import MyException
from src.utility.mlflow_setup import setup_mlflow_connection
from config.constant import experiment_name

logging = configure_logger()
setup_mlflow_connection()

def get_existing_model(experiment_name = experiment_name):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None

    runs = client.search_runs([experiment.experiment_id])
    if not runs:
        return None

    existing_models = sorted(
        runs,
        key=lambda x: x.data.metrics.get('eval_f1', 0) if hasattr(x, "data") else x.metrics.get("eval_f1", 0),
        reverse= True
    )

    return existing_models[0]

def get_existing_model_f1_score(experiment_name = experiment_name):
    best_run = get_existing_model(experiment_name)
    if best_run is None:
        return None
    return best_run.data.metrics.get('eval_f1', 0) if hasattr(best_run, 'data') else best_run.metrics.get('eval_f1, 0')

def load_registered_model(model_name="montra-sentiment-model"):
    setup_mlflow_connection()
    model_uri = f"models:/{model_name}/latest"

    sentiment_pipeline = mlflow.transformers.load_model(model_uri)

    return sentiment_pipeline