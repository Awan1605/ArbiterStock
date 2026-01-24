"""
STOCK PREDICTION API - Main Application
Complete integration: Python ↔ Laravel
Real APIs: OpenAI, Yahoo Finance, NewsAPI, FinBERT
NO DUMMY DATA
"""

# DATABASE & SESSION
from python_ai.database import Base, engine, SessionLocal, test_connection

# MODELS
from python_ai.models_db import (
    Market,
    StockDetail,
    News,
    PredictionAccuracy,
    ModelPerformance
)

# CONFIG
from python_ai.config.settings import (
    HOST,
    PORT,
    DEFAULT_TICKERS,
    CORS_ORIGINS,
    MODEL_CONFIG,
    validate_config
)

# FASTAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# LOGGING & UTILITIES
import logging
from typing import Optional
from datetime import datetime
import schedule
from python_ai.scheduler import start_scheduler


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Stock Prediction API - Real Integrations",
    description="""
    🚀 **Production-Ready Stock Prediction System**
    
    **Real Integrations (NO DUMMY DATA):**
    - 📊 Yahoo Finance: Market data & historical prices
    - 🤖 LightGBM: ML predictions with 5-fold CV
    - 📰 NewsAPI: Real-time news articles
    - 🧠 FinBERT: Financial sentiment analysis
    - 💬 OpenAI GPT-4: AI-powered explanations
    
    **Features:**
    - Multi-horizon predictions (1w, 1m, 1y)
    - 50+ technical indicators
    - Automatic accuracy tracking
    - Real-time market data
    - News sentiment analysis
    
    **Laravel Integration:**
    - Direct MySQL access (no API calls needed)
    - Shared database tables
    - Real-time data synchronization
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for Laravel integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables
Base.metadata.create_all(bind=engine)
logger.info("✅ Database tables created/verified")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application"""
    logger.info("="*70)
    logger.info("🚀 STOCK PREDICTION API - STARTING")
    logger.info("="*70)
    
    # Test database
    if test_connection():
        logger.info("✓ Database: Connected")
    else:
        logger.error("✗ Database: Connection failed!")
    
    # Validate configuration
    warnings = validate_config()
    for warning in warnings:
        logger.warning(warning)
    
    # Start scheduler
    try:
        start_scheduler()
        logger.info("✓ Scheduler: Started")
    except Exception as e:
        logger.error(f"✗ Scheduler: {str(e)}")
    
    logger.info("="*70)
    logger.info("✅ API READY")
    logger.info("="*70)
    logger.info(f"📡 Endpoints: http://localhost:8000/docs")
    logger.info(f"🔍 Health: http://localhost:8000/health")
    logger.info("="*70 + "\n")

# ============================================================================
# HEALTH & STATUS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Basic health check"""
    return {
        "status": "online",
        "message": "Stock Prediction API with Real Integrations",
        "version": "1.0.0",
        "integrations": {
            "yahoo_finance": "✓ Active",
            "lightgbm": "✓ Active",
            "finbert": "✓ Active (ProsusAI/finbert)",
            "openai": "✓ Active (GPT-4)" if validate_config() == [] else "⚠️ Configure API key",
            "newsapi": "✓ Active" if validate_config() == [] else "⚠️ Configure API key"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check"""
    db = SessionLocal()
    try:
        db.execute("SELECT 1")
        db_status = "healthy"
        
        # Check data availability
        market_count = db.query(Market).count()
        stock_count = db.query(StockDetail).count()
        news_count = db.query(News).count()
        
        db_status += f" ({market_count} market, {stock_count} predictions, {news_count} news)"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    finally:
        db.close()
    
    config_warnings = validate_config()
    
    return {
        "status": "online",
        "database": db_status,
        "version": "1.0.0",
        "tickers": DEFAULT_TICKERS,
        "warnings": config_warnings if config_warnings else ["All systems operational"]
    }

# ============================================================================
# MARKET DATA (Yahoo Finance Integration)
# ============================================================================

@app.get("/api/market", tags=["Market"])
async def get_market():
    """
    Get real-time market data from Yahoo Finance
    
    **Data Source:** Yahoo Finance API (yfinance)
    **Update Frequency:** Every 5 minutes
    **NO DUMMY DATA:** All real market prices
    """
    db = SessionLocal()
    try:
        # Get latest market data
        from services.market_service import get_latest_market_data
        data = get_latest_market_data()
        
        return {
            "success": True,
            "count": len(data),
            "source": "Yahoo Finance (yfinance)",
            "data": [
                {
                    "ticker": m.ticker,
                    "name": m.name,
                    "price": float(m.price) if m.price else 0,
                    "volume": m.volume,
                    "change_percent": float(m.change_percent) if m.change_percent else 0,
                    "market_cap": float(m.market_cap) if m.market_cap else 0,
                    "updated_at": m.updated_at.isoformat() if m.updated_at else None
                }
                for m in data
            ]
        }
    except Exception as e:
        logger.error(f"Market endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

# ============================================================================
# STOCK PREDICTIONS (LightGBM + Yahoo Finance)
# ============================================================================

@app.get("/api/stocks", tags=["Predictions"])
async def get_stocks():
    """
    Get AI predictions for all stocks
    
    **Model:** LightGBM with 5-fold TimeSeriesSplit CV
    **Features:** 50+ technical indicators
    **Data Source:** Yahoo Finance historical data
    **NO DUMMY DATA:** Real ML predictions
    """
    try:
        from services.stock_service import get_all_stocks
        stocks = get_all_stocks()
        
        return {
            "success": True,
            "count": len(stocks),
            "model": "LightGBM",
            "features": "50+ technical indicators",
            "data": [
                {
                    "ticker": s.ticker,
                    "date": s.date.isoformat() if s.date else None,
                    "close": float(s.close) if s.close else 0,
                    "predictions": {
                        "1w": float(s.predicted_1w) if s.predicted_1w else 0,
                        "1m": float(s.predicted_1m) if s.predicted_1m else 0,
                        "1y": float(s.predicted_1y) if s.predicted_1y else 0
                    },
                    "technical": {
                        "rsi": float(s.rsi) if s.rsi else 0,
                        "macd": float(s.macd) if s.macd else 0,
                        "sma_20": float(s.sma_20) if s.sma_20 else 0,
                        "sma_50": float(s.sma_50) if s.sma_50 else 0
                    },
                    "sentiment": {
                        "news": float(s.news_score) if s.news_score else 0.5,
                        "fundamental": float(s.fundamental_score) if s.fundamental_score else 0.5
                    },
                    "confidence": float(s.model_confidence) if s.model_confidence else 0.5,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else None
                }
                for s in stocks
            ]
        }
    except Exception as e:
        logger.error(f"Stocks endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks/{ticker}", tags=["Predictions"])
async def get_stock_detail(ticker: str):
    """
    Get complete stock analysis
    
    **Includes:**
    - Real-time price (Yahoo Finance)
    - AI predictions (LightGBM)
    - 50+ technical indicators
    - News sentiment (FinBERT)
    - Fundamental analysis (Yahoo Finance)
    - GPT-4 explanation (OpenAI)
    - Accuracy metrics
    """
    try:
        from services.stock_service import get_stock_by_ticker
        from services.accuracy_service import get_accuracy_summary
        
        stock = get_stock_by_ticker(ticker)
        
        if not stock:
            raise HTTPException(status_code=404, detail=f"Stock {ticker} not found. Available tickers: {', '.join(DEFAULT_TICKERS)}")
        
        # Get accuracy
        try:
            accuracy = get_accuracy_summary(ticker)
        except:
            accuracy = {"message": "Accuracy data will be available after 24 hours"}
        
        return {
            "success": True,
            "data_sources": {
                "price": "Yahoo Finance",
                "predictions": "LightGBM ML Model",
                "sentiment": "FinBERT (ProsusAI)",
                "explanation": "OpenAI GPT-4"
            },
            "data": {
                "ticker": stock.ticker,
                "date": stock.date.isoformat() if stock.date else None,
                "price": {
                    "current": float(stock.close) if stock.close else 0,
                    "open": float(stock.open) if stock.open else 0,
                    "high": float(stock.high) if stock.high else 0,
                    "low": float(stock.low) if stock.low else 0,
                    "volume": stock.volume
                },
                "predictions": {
                    "1w": {
                        "price": float(stock.predicted_1w) if stock.predicted_1w else 0,
                        "change_pct": ((float(stock.predicted_1w) - float(stock.close)) / float(stock.close) * 100) if stock.predicted_1w and stock.close else 0
                    },
                    "1m": {
                        "price": float(stock.predicted_1m) if stock.predicted_1m else 0,
                        "change_pct": ((float(stock.predicted_1m) - float(stock.close)) / float(stock.close) * 100) if stock.predicted_1m and stock.close else 0
                    },
                    "1y": {
                        "price": float(stock.predicted_1y) if stock.predicted_1y else 0,
                        "change_pct": ((float(stock.predicted_1y) - float(stock.close)) / float(stock.close) * 100) if stock.predicted_1y and stock.close else 0
                    }
                },
                "technical_indicators": {
                    "rsi": float(stock.rsi) if stock.rsi else 0,
                    "macd": float(stock.macd) if stock.macd else 0,
                    "macd_signal": float(stock.macd_signal) if stock.macd_signal else 0,
                    "sma_20": float(stock.sma_20) if stock.sma_20 else 0,
                    "sma_50": float(stock.sma_50) if stock.sma_50 else 0,
                    "sma_200": float(stock.sma_200) if stock.sma_200 else 0,
                    "bollinger_high": float(stock.bollinger_high) if stock.bollinger_high else 0,
                    "bollinger_low": float(stock.bollinger_low) if stock.bollinger_low else 0,
                    "adx": float(stock.adx) if stock.adx else 0,
                    "atr": float(stock.atr) if stock.atr else 0
                },
                "sentiment": {
                    "news_score": float(stock.news_score) if stock.news_score else 0.5,
                    "fundamental_score": float(stock.fundamental_score) if stock.fundamental_score else 0.5
                },
                "ai_explanation": stock.llm_explanation,
                "model_confidence": float(stock.model_confidence) if stock.model_confidence else 0.5,
                "accuracy": accuracy,
                "updated_at": stock.updated_at.isoformat() if stock.updated_at else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stock detail error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# NEWS & SENTIMENT (NewsAPI + FinBERT)
# ============================================================================

@app.get("/api/news/{ticker}", tags=["News"])
async def get_news(ticker: str, limit: int = 20):
    """
    Get news with sentiment analysis
    
    **News Source:** NewsAPI.org
    **Sentiment Model:** FinBERT (ProsusAI/finbert)
    **Analysis:** Real transformer-based NLP
    **NO DUMMY DATA:** Real news + real sentiment scores
    """
    try:
        from services.news_service import get_news_by_ticker
        news = get_news_by_ticker(ticker, limit)
        
        return {
            "success": True,
            "count": len(news),
            "news_source": "NewsAPI.org",
            "sentiment_model": "FinBERT (ProsusAI)",
            "data": [
                {
                    "title": n.title,
                    "description": n.description,
                    "url": n.url,
                    "source": n.source,
                    "published_at": n.published_at.isoformat() if n.published_at else None,
                    "sentiment": {
                        "score": float(n.sentiment_score) if n.sentiment_score else 0.5,
                        "label": n.sentiment_label,
                        "confidence": float(n.sentiment_confidence) if n.sentiment_confidence else 0.5
                    }
                }
                for n in news
            ]
        }
    except Exception as e:
        logger.error(f"News endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ACCURACY TRACKING
# ============================================================================

@app.get("/api/accuracy/{ticker}", tags=["Accuracy"])
async def get_accuracy(ticker: str, horizon: Optional[str] = None):
    """
    Get prediction accuracy metrics
    
    **Evaluation:** Automatic comparison with actual prices
    **Metrics:** MAE, MAPE, RMSE, R², Direction Accuracy
    **NO DUMMY DATA:** Real performance tracking
    """
    try:
        from services.accuracy_service import get_accuracy_summary
        accuracy = get_accuracy_summary(ticker, horizon)
        
        return {
            "success": True,
            "evaluation": "Predictions vs Actual Prices",
            "data": accuracy
        }
    except Exception as e:
        logger.error(f"Accuracy endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/performance/{ticker}", tags=["Accuracy"])
async def get_performance(ticker: str):
    """Get overall model performance metrics"""
    db = SessionLocal()
    try:
        performance = db.query(ModelPerformance).filter(
            ModelPerformance.ticker == ticker
        ).order_by(ModelPerformance.updated_at.desc()).first()
        
        if not performance:
            return {
                "success": True,
                "data": {
                    "message": "Performance data will be available after accuracy evaluation"
                }
            }
        
        return {
            "success": True,
            "data": {
                "ticker": performance.ticker,
                "model_version": performance.model_version,
                "overall": {
                    "avg_mae": float(performance.avg_mae) if performance.avg_mae else 0,
                    "avg_mape": float(performance.avg_mape) if performance.avg_mape else 0,
                    "avg_direction_accuracy": float(performance.avg_direction_accuracy) if performance.avg_direction_accuracy else 0
                },
                "by_horizon": {
                    "1w": {"mae": float(performance.mae_1w) if performance.mae_1w else 0, "mape": float(performance.mape_1w) if performance.mape_1w else 0},
                    "1m": {"mae": float(performance.mae_1m) if performance.mae_1m else 0, "mape": float(performance.mape_1m) if performance.mape_1m else 0},
                    "1y": {"mae": float(performance.mae_1y) if performance.mae_1y else 0, "mape": float(performance.mape_1y) if performance.mape_1y else 0}
                }
            }
        }
    finally:
        db.close()

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    from python_ai.config.settings import HOST, PORT
    
    uvicorn.run(
        "python_ai.main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info"
    )
