import os
import dagshub
import mlflow
from config.constant import experiment_name, DAGSHUB_REPO, DAGSHUB_USERNAME

def setup_mlflow_connection():
    dagshub.init(
        repo_owner=DAGSHUB_USERNAME,
        repo_name=DAGSHUB_REPO,
        mlflow=True
    )

    mlflow.set_experiment(experiment_name)
