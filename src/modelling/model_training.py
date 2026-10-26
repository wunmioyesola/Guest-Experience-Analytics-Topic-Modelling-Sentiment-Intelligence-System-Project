from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from src.data.data_processing import DataProcessor
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from config.constant import model_name, label2id, id2label, training_args

class ModelTraining:
    def __init__(self, train_dataset, test_dataset, label2id, id2label):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.label2id = label2id
        self.id2label = id2label
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=id2label,
            label2id=label2id
        )

    def compute_metrics(self, p):
        preds = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average='weighted')
        }

    def train(self):
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            compute_metrics=self.compute_metrics
        )

        self.trainer.train()
        return self.trainer

    def model_evaluation(self, trainer):
        results = trainer.evaluate()
        return results

    def model_evaluation(self, trainer):
        results = trainer.evaluate()
        return results
