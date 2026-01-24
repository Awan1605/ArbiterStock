import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_NAME = "Stock Prediction AI"
    VERSION = "1.0.0"
    DEBUG = os.getenv("DEBUG", "True") == "True"
    
    # Server
    HOST = "0.0.0.0"
    PORT = 8001
    
    # Model
    LSTM_EPOCHS = 10
    LSTM_BATCH_SIZE = 32
    TIME_STEP = 60
    
    # Stock Data
    DEFAULT_PERIOD = "5y"
    MIN_DATA_POINTS = 60

settings = Settings()