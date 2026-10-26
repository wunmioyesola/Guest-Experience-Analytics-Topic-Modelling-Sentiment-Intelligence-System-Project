import mlflow
import mlflow.transformers
import os
import sys
from transformers import pipeline
from src.utility.model_loader import get_existing_model_f1_score
from config.constant import model_name, experiment_name, training_args
from src.utility.mlflow_setup import setup_mlflow_connection
from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()

from dotenv import load_dotenv
load_dotenv(override=True)

class ModelPusher:
    def __init__(self, experiment_name = experiment_name):
        try:
            setup_mlflow_connection()
            self.experiment_name = experiment_name
        except Exception as e:
            logging.error(f"error occurred while initalizing mlflow {e}")

    def update_model_pusher(self, trainer, metrics):
        try:
            new_model_f1_score = metrics["eval_f1"]
            old_model_f1_score = get_existing_model_f1_score(self.experiment_name)

            logging.info(f"existing model f1 score is {old_model_f1_score}")
            logging.info(f"new model f1 score is {new_model_f1_score}")

            if old_model_f1_score is None or new_model_f1_score > old_model_f1_score:

                with mlflow.start_run(run_name="sentiment-classification"):

                    # Log evaluation metrics
                    mlflow.log_metric("eval_accuracy", metrics["eval_accuracy"])
                    mlflow.log_metric("eval_f1", metrics["eval_f1"])
                    mlflow.log_metric("eval_loss", metrics["eval_loss"])
                    
                    # Log training parameters
                    mlflow.log_param("num_epochs", training_args.num_train_epochs)
                    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
                    mlflow.log_param("learning_rate", training_args.learning_rate)

                    # CREATE SENTIMENT PIPELINE
                    sentiment_pipeline = pipeline(
                        task="text-classification",
                        model=trainer.model,
                        tokenizer=model_name,
                        return_all_scores=True
                    )

                    # Log and register model
                    mlflow.transformers.log_model(
                        transformers_model=sentiment_pipeline,
                        name="sentiment_model",
                        registered_model_name="montra-sentiment-model"
                    )


                logging.info("\nSentiment model successfully pushed to MLflow!")

            else:
                logging.info("new model was not pushed due to performance not better than the existing model")

        except Exception as e:
            logging.error(f"error occured while pushing the model {e}")
            raise MyException(e, sys)


 