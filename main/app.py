from src.logger import configure_logger
from src.exception import MyException
import io
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import numpy as np
from src.pipeline.prediction_pipeline import predict_sentiment
from src.pipeline.training_pipeline import Train_model


logging = configure_logger()

app = FastAPI(title = "Montra Guest experience analytics API", version="1.0.0")

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return{
        "message": "Montra guest experience analytics API id running",
        "status": "active",
        "docs": "/docs"
    }

predictor = None

@app.on_event("startup")
async def loa_model_on_startup():
    global predictor

    predictor = predict_sentiment

    logging.info("sentiment model has been loaded successfully")

## single prediction route
@app.post("/predict")
def predict_text(request: TextRequest):
    result = predictor.predict(request.text)
    top_label = max(result, key=lambda x:x["score"])

    return{
        "label": top_label["label"],
        "confidence": top_label["score"]

    }


# retraining mechanism route
@app.get("/train")
def train_model():
    try:
        train_model()
        return {"status": "success", "message": "Model training completed successfully"}
    except Exception as e:
        logging.error(f"error occurred while retraining the model {e}")
        return {"status": "error", "message": f"training failed {e}"}

@app.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8"))).head(100)

        if "review_text" not in df.columns:
            return {"error": "csv must have the review_text columns"}
    #except Exception as e:
        #return {"error": f"file read error: {str(e)}"}

        results_list =[]

        for idx, row in df.iterrows():
            try:
                review = str(row["review_text"])
                if not review.strip() or review.lower() == 'nan':
                    raise ValueError("Empty review test")
                result = predictor.predict(review)
        #except Exception as e:
            #logging.error(f"error for review {idx}: {str(e)}")
            

                if result is None or len(result) == 0:
                    raise ValueError("empty result from the model")

                top_label = max(result, key= lambda x:x["score"])

                result_row = row.to_dict()
                result_row["sentiment_label"] = top_label["label"]
                result_row["sentiment_confidence"] = float(top_label["score"])

                result_row = {key: None if pd.isna(value) else value for key, value in result_row.items()}
                results_list.append(result_row)

           # return result_list
            except Exception as e:
                logging.error(f"error for review {idx}: str(e)")
                result_row = row.to_dict()
                result_row["Sentiment_label"] = f"error {str(e)[:100]}"
                result_row["Sentiment_confidence"] = 0.0
                result_row = {key: None if pd.isna(value) else value for key, value in result_row.items()}
                results_list.append(result_row)

        return results_list

    except Exception as e:
        logging.error(f"batch prediction error {str(e)}")
        return {"error": f"Batch prediction failed {str(e)}"}