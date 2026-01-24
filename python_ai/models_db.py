"""
Database Models - FIXED VERSION for 30-day predictions
File: python_ai/models_db.py
"""

from sqlalchemy import (
    Column, Integer, String, DECIMAL, DATETIME,
    Text, Index
)
from python_ai.database import Base
from datetime import datetime


# ======================================================
# ================= MARKET ==============================
# ======================================================
class Market(Base):
    """Real-time market data"""
    __tablename__ = "market"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(15), index=True, nullable=False)
    name = Column(String(100))
    price = Column(DECIMAL(15, 4))
    volume = Column(Integer)
    change_percent = Column(DECIMAL(10, 4))
    market_cap = Column(DECIMAL(20, 2))
    updated_at = Column(DATETIME, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_market_ticker_updated", "ticker", "updated_at"),
    )


# ======================================================
# ================= STOCK DETAIL ========================
# ======================================================
class StockDetail(Base):
    """Stock predictions, indicators, and AI analysis"""
    __tablename__ = "stock_detail"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(15), index=True, nullable=False)
    date = Column(DATETIME, default=datetime.utcnow)

    # ================= OHLCV =================
    close = Column(DECIMAL(15, 4))
    open = Column(DECIMAL(15, 4))
    high = Column(DECIMAL(15, 4))
    low = Column(DECIMAL(15, 4))
    volume = Column(Integer)

    # ================= Technical Indicators =================
    rsi = Column(DECIMAL(10, 4))
    macd = Column(DECIMAL(10, 4))
    macd_signal = Column(DECIMAL(10, 4))
    macd_diff = Column(DECIMAL(10, 4))

    sma_20 = Column(DECIMAL(15, 4))
    sma_50 = Column(DECIMAL(15, 4))
    sma_200 = Column(DECIMAL(15, 4))
    ema_12 = Column(DECIMAL(15, 4))
    ema_26 = Column(DECIMAL(15, 4))

    bollinger_high = Column(DECIMAL(15, 4))
    bollinger_mid = Column(DECIMAL(15, 4))
    bollinger_low = Column(DECIMAL(15, 4))

    atr = Column(DECIMAL(10, 4))
    adx = Column(DECIMAL(10, 4))
    cci = Column(DECIMAL(10, 4))
    stoch_k = Column(DECIMAL(10, 4))
    stoch_d = Column(DECIMAL(10, 4))
    williams_r = Column(DECIMAL(10, 4))
    mfi = Column(DECIMAL(10, 4))
    obv = Column(DECIMAL(20, 2))
    volatility_10d = Column(DECIMAL(10, 4))
    volatility_30d = Column(DECIMAL(10, 4))

    # ================= Scores =================
    technical_score = Column(DECIMAL(5, 4))
    fundamental_score = Column(DECIMAL(5, 4))
    news_score = Column(DECIMAL(5, 4))
    combined_score = Column(DECIMAL(10, 4))

    # ================= AI Analysis =================
    llm_explanation = Column(Text)
    model_confidence = Column(DECIMAL(5, 4))

    # ================= Status =================
    is_active = Column(Integer, default=1)

    updated_at = Column(DATETIME, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_stock_ticker_date", "ticker", "date"),
    )


# ======================================================
# ================= NEWS ================================
# ======================================================
class News(Base):
    """Market & stock news with sentiment"""
    __tablename__ = "news"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(15), index=True, nullable=False)
    title = Column(String(500))
    description = Column(Text)
    url = Column(String(500), unique=True)
    source = Column(String(100))
    published_at = Column(DATETIME)

    sentiment_score = Column(DECIMAL(5, 4))
    sentiment_label = Column(String(20))
    sentiment_confidence = Column(DECIMAL(5, 4))

    created_at = Column(DATETIME, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_news_ticker_published", "ticker", "published_at"),
    )


# ======================================================
# ================= PREDICTION HISTORY =================
# ======================================================
class PredictionHistory(Base):
    """Historical stock predictions - 30 days ahead"""
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), index=True, nullable=False)

    prediction_date = Column(DATETIME, nullable=False)
    target_date = Column(String(20), nullable=False)  # Format: 'YYYY-MM-DD'

    horizon = Column(String(10), nullable=False)  # 'day_1', 'day_2', ..., 'day_30'
    predicted_price = Column(DECIMAL(15, 4))
    actual_price = Column(DECIMAL(15, 4))
    accuracy = Column(DECIMAL(10, 4))

    model_version = Column(String(50))
    confidence = Column(DECIMAL(5, 4))

    created_at = Column(DATETIME, default=datetime.utcnow)

    __table_args__ = (
        Index(
            'idx_ticker_pred_target',
            'ticker',
            'prediction_date',
            'target_date',
            'horizon'
        ),
    )


# ======================================================
# ================= PREDICTION ACCURACY =================
# ======================================================
class PredictionAccuracy(Base):
    """Track real prediction accuracy"""
    __tablename__ = "prediction_accuracy"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(15), index=True, nullable=False)

    prediction_date = Column(DATETIME, nullable=False)
    horizon = Column(String(10), nullable=False)

    predicted_price = Column(DECIMAL(15, 4))
    actual_price = Column(DECIMAL(15, 4))
    base_price = Column(DECIMAL(15, 4))

    mae = Column(DECIMAL(15, 4))
    mape = Column(DECIMAL(10, 4))
    rmse = Column(DECIMAL(15, 4))
    r2_score = Column(DECIMAL(10, 4))
    direction_accuracy = Column(DECIMAL(5, 4))
    model_confidence = Column(DECIMAL(5, 4))

    feature_importance = Column(Text)

    created_at = Column(DATETIME, default=datetime.utcnow)
    evaluated_at = Column(DATETIME)

    __table_args__ = (
        Index("idx_accuracy_ticker_horizon_date", "ticker", "horizon", "prediction_date"),
    )


# ======================================================
# ================= MODEL PERFORMANCE ==================
# ======================================================
class ModelPerformance(Base):
    """Aggregated model performance metrics"""
    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(15), index=True, nullable=False)
    model_version = Column(String(50))

    avg_mae = Column(DECIMAL(15, 4))
    avg_mape = Column(DECIMAL(10, 4))
    avg_rmse = Column(DECIMAL(15, 4))
    avg_r2 = Column(DECIMAL(10, 4))
    avg_direction_accuracy = Column(DECIMAL(5, 4))

    training_samples = Column(Integer)
    test_samples = Column(Integer)
    features_count = Column(Integer)

    features_used = Column(Text)
    hyperparameters = Column(Text)

    created_at = Column(DATETIME, default=datetime.utcnow)
    updated_at = Column(DATETIME, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_model_perf_ticker_version", "ticker", "model_version"),
    )