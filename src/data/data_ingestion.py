import pandas as pd
from sqlalchemy import create_engine, text

from config.database_config import DATABASE_URL

from src.logger import configure_logger
from src.exception import MyException

import os
import sys

logging = configure_logger()

class DataIngestion:
    def __init__(self):
        self.engine = create_engine(
            DATABASE_URL,
            poolclass=None,
            connect_args={"sslmode": "require","connect_timeout": 10}
        )

    def test_connection(self):
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT 1"))
                print("Database response:", result.fetchone())
                print("Successfully connected to Neon")

        except Exception as e:
            print("Connection failed:", repr(e))

    def load_table(self, table_name: str) -> pd.DataFrame:
        query = text(f"SELECT * FROM {table_name}")
        try:
            dataframe = pd.read_sql(query, self.engine)

            logging.info(f"loaded {table_name} with the length of {len(dataframe)} and columns of {len(dataframe.columns)} columns")
            return dataframe

        except Exception as e:
            logging.error(f"failed to load the data set")
            raise MyException(e, sys)

    def load_all_data(self):
        hotels_detail = self.load_table("hotels")
        guest_data = self.load_table("guests")
        customer_review = self.load_table("reviews")
        bookings_data = self.load_table("bookings")
     
        return {
            "hotels": hotels_detail,
            "guests": guest_data,
            "reviews": customer_review,
            "bookings": bookings_data
        }
    
    def data_merger(self):
        try:
            data = self.load_all_data()
            customer_review = data["reviews"]
            booking_data = data["bookings"]
            hotels_detail = data["hotels"]
            guest_data = data["guests"]

            customer_sentiment_data = customer_review.merge(booking_data, on=["booking_id", "guest_id", "hotel_id"], how="inner")
            customer_sentiment_data = customer_sentiment_data.merge(guest_data, on="guest_id", how="inner")
            customer_sentiment_data = customer_sentiment_data.merge(hotels_detail, on="hotel_id", how="inner")

            return customer_sentiment_data
        except Exception as e:
            logging.error(f"error occured while merging the dataset {e}")
            raise MyException(e, sys)

def data_loader():

    # Database connection
    # ingestion = DataIngestion()
    # customer_review_data = ingestion.data_merger()

    # 100 Sample data for testing model training ppeline
    customer_review_data = pd.read_csv(r"C:/Users/USER/Downloads/Guest-Experience-Analytics-Topic-Modelling-Sentiment-Intelligence-System-Project/Testing_data_sample.csv")

    logging.info(customer_review_data.head())        
    return customer_review_data

 


   