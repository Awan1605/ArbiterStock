import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

# ======================================================
# PATH & ENV
# ======================================================
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
    print(f"🔧 Loaded .env from: {ENV_PATH}")
else:
    print(f"⚠️ .env file not found at {ENV_PATH}")

# ======================================================
# SERVER
# ======================================================
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", 8000))

# ======================================================
# DATABASE
# ======================================================
DATABASE_URL: str | None = os.getenv("DATABASE_URL")
DB_HOST: str | None = os.getenv("DB_HOST")
DB_USER: str | None = os.getenv("DB_USER")
DB_PASSWORD: str | None = os.getenv("DB_PASSWORD")
DB_PORT: int = int(os.getenv("DB_PORT", 3306))

# ======================================================
# CORS
# ======================================================
CORS_ORIGINS: List[str] = os.getenv(
    "CORS_ORIGINS",
    "http://localhost,http://127.0.0.1,http://localhost:3000,http://localhost:5173"
).split(",")

# ======================================================
# LLM / AI
# ======================================================
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")

# ======================================================
# NEWS & SENTIMENT
# ======================================================
NEWSAPI_KEY: str | None = os.getenv("NEWSAPI_KEY")
NEWSENTIMENT_API_KEY: str | None = os.getenv("NEWSENTIMENT_API_KEY")
FINBERT_MODEL: str = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")

# ======================================================
# MARKET
# ======================================================
DEFAULT_TICKERS = [
    'AAPL',
    'GOOGL',
    'MSFT',
    'AMZN',
    'TSLA',
    'META',
    'NVDA',
    'NFLX',
    'BBCA.JK',
    'BBRI.JK',
    'BMRI.JK',
    'TLKM.JK',
    'ASII.JK',
    'UNVR.JK',
    'INDF.JK',
    'ICBP.JK',
    'SMGR.JK',
    'GOTO.JK',
    'BBNI.JK',
    'ANTM.JK',
    'PTBA.JK',
]

# ======================================================
# TECHNICAL ANALYSIS
# ======================================================
TECHNICAL_CONFIG = {
    'sma_periods': [20, 50, 200],
    'ema_periods': [12, 26],
    'rsi_period': 14,
    'stoch_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'adx_period': 14,
    'cci_period': 20,
    'bollinger_period': 20,
    'bollinger_std': 2,
    'atr_period': 14
}

# ======================================================
# MODEL CONFIG - ENHANCED VERSION (FIXED)
# ======================================================
MODEL_CONFIG: Dict = {
    "model_name": "LightGBM",
    "model_version": "2.0.0-enhanced",
    "cv_folds": 10,
    "prediction_horizons": ["1w", "2w", "3w", "1m"],
    "features": {
        "technical_indicators": True,
        "news_sentiment": True,
        "fundamental_data": True
    },
    "sentiment_model": FINBERT_MODEL,
    "llm_provider": LLM_PROVIDER,
    "llm_model": LLM_MODEL,
    "confidence_threshold": 0.6,

    "lookback_days": int(os.getenv("MODEL_LOOKBACK_DAYS", 1825)),
    "time_step": int(os.getenv("MODEL_TIME_STEP", 60)),
    "batch_size": int(os.getenv("MODEL_BATCH_SIZE", 32)),
    "epochs": int(os.getenv("MODEL_EPOCHS", 50)),
    "learning_rate": float(os.getenv("MODEL_LR", 0.001)),

    "lightgbm": {
        "n_estimators": int(os.getenv("LGB_N_ESTIMATORS", 1000)),
        "learning_rate": float(os.getenv("LGB_LR", 0.01)),
        "max_depth": int(os.getenv("LGB_MAX_DEPTH", 8)),
        "num_leaves": int(os.getenv("LGB_NUM_LEAVES", 63)),
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.3,
        "reg_lambda": 0.3,
        "min_split_gain": 0.02,
        "random_state": 42,
        "force_col_wise": True,
        "verbose": -1,
    }
}

# ======================================================
# SCHEDULER
# ======================================================
SCHEDULER_INTERVAL_MARKET: int = int(os.getenv("SCHEDULER_INTERVAL_MARKET", 60))
SCHEDULER_INTERVAL_STOCK: int = int(os.getenv("SCHEDULER_INTERVAL_STOCK", 3600))
SCHEDULER_INTERVAL_NEWS: int = int(os.getenv("SCHEDULER_INTERVAL_NEWS", 900))

# ======================================================
# DATA VALIDATION - ENHANCED
# ======================================================
MIN_DATA_POINTS: int = int(os.getenv("MIN_DATA_POINTS", 500))

# ======================================================
# MODEL FEATURES - ENHANCED (100+ features)
# ======================================================
FEATURES: List[str] = [
    "Close",
    "Volume",
    "Returns",
    "Log_Returns",
    "Price_Range",
    "Gap",
    "Volume_Change",
    "Volume_MA_Ratio",
    "Volatility_5d",
    "Volatility_10d",
    "Volatility_20d",
    "Volatility_30d",
    "RSI_14",
    "RSI_21",
    "MACD",
    "MACD_Signal",
    "MACD_Hist",
    "SMA_5", "SMA_10", "SMA_20", "SMA_50", "SMA_100", "SMA_200",
    "EMA_5", "EMA_10", "EMA_12", "EMA_20", "EMA_26", "EMA_50", "EMA_100", "EMA_200",
    "Price_to_SMA20",
    "Price_to_SMA50",
    "Price_to_SMA200",
    "SMA_20_50_Cross",
    "SMA_50_200_Cross",
    "ATR",
    "ATR_Percent",
    "BB_20_Upper", "BB_20_Lower", "BB_20_Width", "BB_20_Position",
    "BB_50_Upper", "BB_50_Lower", "BB_50_Width", "BB_50_Position",
    "Stoch_K",
    "Stoch_D",
    "ADX",
    "DI_Diff",
    "OBV",
    "OBV_MA",
    "OBV_Trend",
    "Williams_R_14",
    "Williams_R_28",
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "ROC_5",
    "ROC_10",
    "ROC_20",
    "CCI",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5",
    "Close_Lag_10", "Close_Lag_20", "Close_Lag_30",
    "Volume_Lag_1", "Volume_Lag_2", "Volume_Lag_3", "Volume_Lag_5",
    "Volume_Lag_10", "Volume_Lag_20", "Volume_Lag_30",
    "Returns_Lag_1", "Returns_Lag_2", "Returns_Lag_3", "Returns_Lag_5",
    "Returns_Lag_10", "Returns_Lag_20", "Returns_Lag_30",
    "Day_of_Week", "Day_of_Month", "Month", "Quarter",
    "Day_Sin", "Day_Cos", "Month_Sin", "Month_Cos",
]

# ======================================================
# PREDICTION HORIZONS - ENHANCED (30-day focus)
# ======================================================
HORIZONS: Dict[str, int] = {
    "1w": 7,
    "2w": 14,
    "3w": 21,
    "1m": 30,
}

# ======================================================
# TRAINING PARAMS - ENHANCED
# ======================================================
TRAINING_PARAMS: Dict = {
    "test_size": 0.15,
    "random_state": 42,
    "shuffle": False
}

# ======================================================
# NEWS CONFIG - NEW
# ======================================================
NEWS_CONFIG = {
    'update_interval': 1800,
    'articles_per_ticker': 20,
    'sentiment_threshold': 0.6,
    'lookback_days': 30,
    'use_global_fallback': True,
    'use_regional_fallback': True,
    'min_specific_news': 5,
}

# ======================================================
# SCORE WEIGHTS - NEW
# ======================================================
SCORE_WEIGHTS = {
    'technical': 0.35,
    'financial': 0.35,
    'news': 0.30,
}

# ======================================================
# PERFORMANCE THRESHOLDS - NEW
# ======================================================
PERFORMANCE_THRESHOLDS = {
    'min_r2_score': 0.30,
    'max_mape': 20.0,
    'min_direction_accuracy': 55.0,
    'min_confidence': 35.0,
    'target_confidence': 70.0,
}

# ======================================================
# VALIDATION
# ======================================================
def validate_config() -> List[str]:
    warnings: List[str] = []

    if not DATABASE_URL:
        warnings.append("DATABASE_URL is not configured")

    if LLM_PROVIDER == "openai" and not OPENAI_API_KEY:
        warnings.append("OPENAI_API_KEY is missing (LLM explanation disabled)")

    if not NEWSAPI_KEY:
        warnings.append("NEWSAPI_KEY is missing (news sentiment disabled)")

    if not GROQ_API_KEY:
        warnings.append("GROQ_API_KEY is missing (Groq LLM disabled)")

    return warnings

# ======================================================
# SETTINGS CLASS
# ======================================================
class Settings:
    # SERVER
    HOST: str = HOST
    PORT: int = PORT

    # DATABASE
    DATABASE_URL: str | None = DATABASE_URL
    DB_HOST: str | None = DB_HOST
    DB_USER: str | None = DB_USER
    DB_PASSWORD: str | None = DB_PASSWORD
    DB_PORT: int = DB_PORT

    # CORS
    CORS_ORIGINS: List[str] = CORS_ORIGINS

    # LLM / AI
    LLM_PROVIDER: str = LLM_PROVIDER
    OPENAI_API_KEY: str | None = OPENAI_API_KEY
    GROQ_API_KEY: str | None = GROQ_API_KEY
    LLM_MODEL: str = LLM_MODEL

    # NEWS & SENTIMENT
    NEWSAPI_KEY: str | None = NEWSAPI_KEY
    NEWSENTIMENT_API_KEY: str | None = NEWSENTIMENT_API_KEY
    FINBERT_MODEL: str = FINBERT_MODEL

    # MARKET
    DEFAULT_TICKERS: List[str] = DEFAULT_TICKERS

    # TECHNICAL
    TECHNICAL_CONFIG: Dict = TECHNICAL_CONFIG

    # MODEL (ENHANCED)
    MODEL_CONFIG: Dict = MODEL_CONFIG
    FEATURES: List[str] = FEATURES
    HORIZONS: Dict[str, int] = HORIZONS
    MIN_DATA_POINTS: int = MIN_DATA_POINTS
    TRAINING_PARAMS: Dict = TRAINING_PARAMS
    
    # NEW CONFIGS
    NEWS_CONFIG: Dict = NEWS_CONFIG
    SCORE_WEIGHTS: Dict = SCORE_WEIGHTS
    PERFORMANCE_THRESHOLDS: Dict = PERFORMANCE_THRESHOLDS

    # SCHEDULER
    SCHEDULER_INTERVAL_MARKET: int = SCHEDULER_INTERVAL_MARKET
    SCHEDULER_INTERVAL_STOCK: int = SCHEDULER_INTERVAL_STOCK
    SCHEDULER_INTERVAL_NEWS: int = SCHEDULER_INTERVAL_NEWS

# ======================================================
# INSTANCE GLOBAL
# ======================================================
config = Settings()

# ======================================================
# SCHEDULER CONFIG
# ======================================================
SCHEDULER_CONFIG = {
    "market_update_minutes": int(os.getenv("SCHEDULER_INTERVAL_MARKET_MINUTES", 5)),
    "stock_detail_update_hours": int(os.getenv("SCHEDULER_INTERVAL_STOCK_HOURS", 1)),
    "news_update_minutes": int(os.getenv("SCHEDULER_INTERVAL_NEWS_MINUTES", 30)),
    "accuracy_check_hours": int(os.getenv("SCHEDULER_ACCURACY_HOURS", 24)),
}