import logging
import os
from datetime import datetime


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def configure_logger():
    logger = logging.getLogger()

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
    )


    timestamp =datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(
        LOG_DIR, f"run_{timestamp}.log"
    )

    # file handler
    file_handler = logging.FileHandler(
        log_file, encoding="utf-8"
    )

    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

