import os
import sys
import torch
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from src.data.data_cleaning import DataCleaner
from src.logger import configure_logger
from src.exception import MyException

logging = configure_logger()

class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels


    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
         item['labels'] = torch.tensor(self.labels[idx])
         return item

class SentimentAnalysisDataProcessor:
    def __init__(self, model_name = "distilbert-base-multilingual-cased", max_length=128):
        self.model_name = model_name 
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def split_data(self, dataframe):
        try:
            x = dataframe['review_text'].astype(str)          
            y = dataframe['sentiment_label']

            # Buid a stable label mapping ONCE, use it everywhere
            labelZid = {"negative": 0, "neutral": 1, "positive": 2}
            idZlabel = {v: k for k, v in labelZid.items()}
            y_encoded = y.map(labelZid)

            # x = cleaned_merged['review_text'].astype(str)
            # y_encoded = cleaned_merged['sentiment_label'].map(labelZid)
            # y_encoded = y.map(labelZid)

            # cleaned_mask = y_encoded.notna()
            # x = x[cleaned_mask]
            # y_encoded = y_encoded[cleaned_mask].astype(int)

            X_train, X_test, y_train, y_test = train_test_split(
                x,
                y_encoded,
                test_size=0.2,
                random_state=42,
                stratify=y_encoded                      
            )

            X_train = X_train.tolist()
            X_test = X_test.tolist()
            y_train = y_train.tolist()
            y_test = y_test.tolist()

            logging.info(f"Training sample is {len(X_train)}")
            logging.info(f"Testing sample is {len(X_test)}")

            return X_train, X_test, y_train, y_test, labelZid, idZlabel
        except Exception as e:
            logging.error("error occured while splitting the dataset {e}")
            raise MyException(e, sys)

    def tokenise_data(self, X_train, X_test):
        try:
            train_encodings = self.tokenizer(X_train, truncation=True, padding=True, max_length=self.max_length)
            test_encodings = self.tokenizer(X_test, truncation=True, padding=True, max_length=self.max_length)
            logging.info(f"successfully tokenized the train and testing data sample")

            return train_encodings, test_encodings

        except Exception as e:
            logging.error(f"error occured while tokenizing the dataset {e}")
            raise MyException(e, sys)

    def create_dataset(self, train_encodings, test_encodings, y_train, y_test):
        try:
            train_dataset = SentimentDataset(train_encodings, y_train)
            test_dataset = SentimentDataset(test_encodings, y_test)

            logging.info(f"successfully created sentiment dataset...")
            logging.info(f"training samples : {len(train_dataset)}")
            logging.info(f"testing sample: {len(test_dataset)}")

            return train_dataset, test_dataset
        except Exception as e:
            logging.error(f"error occurred while creating datset {e}")
            raise MyException(e, sys)

def DataProcessor():
    raw_cleaned_data = DataCleaner()
    cleaned_customer_review_data = pd.DataFrame(raw_cleaned_data)
    SentimentAnalysisProcessorEngine = SentimentAnalysisDataProcessor()
    X_train, X_test, y_train, y_test, labelZid, idZlabel = SentimentAnalysisProcessorEngine.split_data(cleaned_customer_review_data)
    train_encodings, test_encodings = SentimentAnalysisProcessorEngine.tokenise_data(X_train=X_train, X_test=X_test)
    train_dataset, test_dataset = SentimentAnalysisProcessorEngine.create_dataset(train_encodings, test_encodings, y_train, y_test)

    return train_dataset, test_dataset, labelZid, idZlabel




    