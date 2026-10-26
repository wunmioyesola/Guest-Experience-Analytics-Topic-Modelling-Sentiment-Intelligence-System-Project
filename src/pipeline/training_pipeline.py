from src.data.data_processing import DataProcessor
from src.modelling.model_training import ModelTraining
from src.modelling.model_tracking import ModelPusher
from src.logger import configure_logger
from src.exception import MyException
import sys


logging = configure_logger()


def Train_model():
    try:
        train_dataset, test_dataset, label2id, id2label = DataProcessor()
        training_engine = ModelTraining(train_dataset, test_dataset, label2id, id2label)
        trained_model = training_engine.train()
        results = training_engine.model_evaluation(trained_model)
        logging.info(f"{results}")

        # push the model to mlflow
        pusher = ModelPusher()
        pusher.update_model_pusher(trained_model, results)
        return trained_model, results

    except Exception as e:
        logging.error(f"error occured while retraining the model {e}")
        raise MyException(e, sys)


Train_model()