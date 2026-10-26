from transformers import TrainingArguments 
model_name = "distilbert-base-multilingual-cased"
experiment_name = "Montra-Guest-Experience-Analytics"
DAGSHUB_USERNAME = "wunmi.abosede"
DAGSHUB_REPO = "Guest-Experience-Analytics-Topic-Modelling-Sentiment-Intelligence-System-Project"

label2id = {
    "negative": 0,
    "neutral": 1,
    "positive": 2
}

id2label = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none"                              # prevent W&B prompt hijacking the run

)