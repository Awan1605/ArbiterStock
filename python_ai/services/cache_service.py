# python_ai/services/cache_service.py
"""
Caching Service - In-memory cache for predictions
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PredictionCacheService:
    """Simple in-memory cache for predictions"""
    
    _cache: Dict = {}
    _cache_duration_hours: int = 1
    
    @classmethod
    def cache_prediction(cls, ticker: str, data: dict) -> None:
        """Cache prediction data"""
        try:
            cls._cache[ticker] = {
                'data': data,
                'cached_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=cls._cache_duration_hours)
            }
            logger.info(f"✓ Cached prediction for {ticker}")
        except Exception as e:
            logger.error(f"Cache error for {ticker}: {e}")
    
    @classmethod
    def get_cached_prediction(cls, ticker: str) -> Optional[dict]:
        """Get cached prediction if valid"""
        try:
            if ticker not in cls._cache:
                return None
            
            cache_entry = cls._cache[ticker]
            
            # Check expiration
            if datetime.now() > cache_entry['expires_at']:
                del cls._cache[ticker]
                logger.info(f"Cache expired for {ticker}")
                return None
            
            logger.info(f"✓ Cache hit for {ticker}")
            return cache_entry['data']
            
        except Exception as e:
            logger.error(f"Cache retrieval error for {ticker}: {e}")
            return None
    
    @classmethod
    def clear_cache(cls, ticker: Optional[str] = None) -> None:
        """Clear cache for specific ticker or all"""
        try:
            if ticker:
                if ticker in cls._cache:
                    del cls._cache[ticker]
                    logger.info(f"✓ Cleared cache for {ticker}")
            else:
                cls._cache.clear()
                logger.info("✓ Cleared all cache")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    @classmethod
    def get_cache_stats(cls) -> Dict:
        """Get cache statistics"""
        total = len(cls._cache)
        expired = sum(1 for entry in cls._cache.values() 
                     if datetime.now() > entry['expires_at'])
        valid = total - expired
        
        return {
            'total': total,
            'valid': valid,
            'expired': expired,
            'tickers': list(cls._cache.keys())
        }