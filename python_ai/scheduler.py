# python_ai/scheduler.py
"""
Background Scheduler - Auto-update predictions, news, market data
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()


def update_market_data():
    """Update market data every 5 minutes"""
    try:
        from python_ai.services.market_service import update_market_prices
        logger.info("🔄 Updating market data...")
        update_market_prices()
        logger.info("✅ Market data updated")
    except Exception as e:
        logger.error(f"❌ Market update error: {e}")


def update_stock_predictions():
    """Update 30-day predictions every 1 hour"""
    try:
        from python_ai.services.stock_service import update_stock_detail
        logger.info("🔄 Updating stock predictions...")
        update_stock_detail()
        logger.info("✅ Stock predictions updated")
    except Exception as e:
        logger.error(f"❌ Prediction update error: {e}")


def update_news():
    """Update news every 30 minutes"""
    try:
        from python_ai.services.news_service import update_news_data
        logger.info("🔄 Updating news...")
        update_news_data(limit=20)
        logger.info("✅ News updated")
    except Exception as e:
        logger.error(f"❌ News update error: {e}")


def cleanup_old_data():
    """Cleanup old data daily"""
    try:
        from python_ai.services.news_service import cleanup_old_news
        logger.info("🔄 Cleaning up old data...")
        cleanup_old_news(days=90)
        logger.info("✅ Cleanup complete")
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")


def start_scheduler():
    """Start background scheduler"""
    try:
        # Market data - every 5 minutes during market hours
        scheduler.add_job(
            update_market_data,
            trigger='interval',
            minutes=5,
            id='market_update',
            replace_existing=True
        )
        
        # Stock predictions - every 1 hour
        scheduler.add_job(
            update_stock_predictions,
            trigger='interval',
            hours=1,
            id='stock_update',
            replace_existing=True
        )
        
        # News - every 30 minutes
        scheduler.add_job(
            update_news,
            trigger='interval',
            minutes=30,
            id='news_update',
            replace_existing=True
        )
        
        # Cleanup - daily at 2 AM
        scheduler.add_job(
            cleanup_old_data,
            trigger=CronTrigger(hour=2, minute=0),
            id='cleanup',
            replace_existing=True
        )
        
        # Start scheduler
        if not scheduler.running:
            scheduler.start()
            logger.info("✅ Scheduler started")
        
    except Exception as e:
        logger.error(f"❌ Scheduler start error: {e}")


def stop_scheduler():
    """Stop scheduler"""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("🛑 Scheduler stopped")