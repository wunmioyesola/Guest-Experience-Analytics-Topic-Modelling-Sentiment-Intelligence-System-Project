from src.logger import configure_logger
from src.exception import MyException
from src.data.data_ingestion import data_loader
import sys
import re
logging = configure_logger()


class DataCleaning:

    def clean_review(self, text):
        # Convert to string
        text = str(text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', ' ', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        # Remove emojis
        text = text.encode('ascii', 'ignore').decode('ascii')
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        # Keep letters, numbers and spaces
        text =re.sub(r'[^a-z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+',' ', text).strip()

        return text

    def clean_review_columns(self, dataframe):
        try:
            dataframe = dataframe.copy()

            # Apply preprocessing
            dataframe["cleaned_review_text"] = (dataframe["review_text"].map(self.clean_review))
            logging.info(dataframe.head())

            return dataframe
        except Exception as e:
            logging.error(f"error occured while cleaning the datset (e)")
            raise MyException(e, sys)

def DataCleaner():
        customer_review_data = data_loader()
        DataCleaningEngine = DataCleaning()
        cleaned_customer_review = DataCleaningEngine.clean_review_columns(customer_review_data)

        return cleaned_customer_review

DataCleaner()
