import os
from dotenv import load_dotenv
load_dotenv(override = True)

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("Database url environment has not been set")    
