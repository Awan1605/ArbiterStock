# python_ai/services/market_service.py
"""
Market Service - Real-time market data from Yahoo Finance
"""

import yfinance as yf
from datetime import datetime
import logging
from python_ai.database import SessionLocal
from python_ai.models_db import Market
from python_ai.config.settings import config

logger = logging.getLogger(__name__)


def update_market_prices():
    """Update market prices for all tickers"""
    db = SessionLocal()
    
    try:
        logger.info("📊 Updating market prices...")
        
        for ticker in config.DEFAULT_TICKERS:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="1d")
                
                if hist.empty:
                    logger.warning(f"⚠️ No data for {ticker}")
                    continue
                
                current_price = float(hist['Close'].iloc[-1])
                prev_close = float(info.get('previousClose', current_price))
                change = current_price - prev_close
                change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                
                # Update or create
                market = db.query(Market).filter(Market.ticker == ticker).first()
                
                if market:
                    market.price = current_price
                    market.change = change
                    market.change_percent = change_pct
                    market.volume = int(hist['Volume'].iloc[-1])
                    market.market_cap = info.get('marketCap', 0)
                    market.updated_at = datetime.utcnow()
                else:
                    market = Market(
                        ticker=ticker,
                        name=info.get('longName', ticker),
                        price=current_price,
                        change=change,
                        change_percent=change_pct,
                        volume=int(hist['Volume'].iloc[-1]),
                        market_cap=info.get('marketCap', 0)
                    )
                    db.add(market)
                
                db.commit()
                logger.info(f"✓ {ticker}: ${current_price:.2f} ({change_pct:+.2f}%)")
                
            except Exception as e:
                logger.error(f"❌ {ticker} error: {e}")
                db.rollback()
        
        logger.info("✅ Market prices updated")
        
    finally:
        db.close()


def get_market_summary():
    """Get market summary"""
    db = SessionLocal()
    
    try:
        markets = db.query(Market).order_by(Market.updated_at.desc()).all()
        
        total = len(markets)
        gainers = sum(1 for m in markets if m.change_percent and m.change_percent > 0)
        losers = total - gainers
        
        return {
            'total': total,
            'gainers': gainers,
            'losers': losers,
            'markets': markets
        }
        
    finally:
        db.close()