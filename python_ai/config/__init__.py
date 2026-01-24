"""
Configuration package
Import everything from settings
"""

from .settings import (
    # Server
    config,
    Settings,
    
    # Database
    DATABASE_URL,
    DB_HOST,
    DB_USER,
    DB_PASSWORD,
    DB_PORT,
    
    # API Keys
    OPENAI_API_KEY,
    GROQ_API_KEY,
    NEWSAPI_KEY,
    NEWSENTIMENT_API_KEY,
    
    # LLM
    LLM_PROVIDER,
    LLM_MODEL,
    FINBERT_MODEL,
    
    # Market
    DEFAULT_TICKERS,
    
    # Model Config
    MODEL_CONFIG,
    FEATURES,
    HORIZONS,
    MIN_DATA_POINTS,
    TRAINING_PARAMS,
    TECHNICAL_CONFIG,
    
    # Scheduler
    SCHEDULER_CONFIG,
    SCHEDULER_INTERVAL_MARKET,
    SCHEDULER_INTERVAL_STOCK,
    SCHEDULER_INTERVAL_NEWS,
    
    # CORS
    CORS_ORIGINS,
    
    # Validation
    validate_config,
)

__all__ = [
    'config',
    'Settings',
    'DATABASE_URL',
    'DB_HOST',
    'DB_USER',
    'DB_PASSWORD',
    'DB_PORT',
    'OPENAI_API_KEY',
    'GROQ_API_KEY',
    'NEWSAPI_KEY',
    'NEWSENTIMENT_API_KEY',
    'LLM_PROVIDER',
    'LLM_MODEL',
    'FINBERT_MODEL',
    'DEFAULT_TICKERS',
    'MODEL_CONFIG',
    'FEATURES',
    'HORIZONS',
    'MIN_DATA_POINTS',
    'TRAINING_PARAMS',
    'TECHNICAL_CONFIG',
    'SCHEDULER_CONFIG',
    'SCHEDULER_INTERVAL_MARKET',
    'SCHEDULER_INTERVAL_STOCK',
    'SCHEDULER_INTERVAL_NEWS',
    'CORS_ORIGINS',
    'validate_config',
]