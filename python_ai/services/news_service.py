# services/news_service.py
"""
Enhanced News Service - dengan fallback ke berita global/nasional
"""

from python_ai.models_db import News
from python_ai.database import SessionLocal
from datetime import datetime, timedelta
from python_ai.config.settings import DEFAULT_TICKERS
from python_ai.services.sentiment_service import sentiment_analyzer
import yfinance as yf
import logging
import requests
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_ticker(ticker: str) -> str:
    """Remove .JK suffix for search queries"""
    return ticker.replace('.JK', '').replace('.', '')

def get_market_region(ticker: str) -> str:
    """Determine market region from ticker"""
    if ticker.endswith('.JK'):
        return 'indonesia'
    elif any(ticker.endswith(suffix) for suffix in ['.HK', '.SS', '.SZ']):
        return 'asia'
    else:
        return 'global'

def fetch_global_financial_news(limit: int = 10) -> List[Dict]:
    """
    Fetch global financial news sebagai fallback
    Sumber: Yahoo Finance top financial news
    """
    try:
        # Use Yahoo Finance main page news
        import feedparser
        
        # RSS feeds untuk financial news
        feeds = [
            'https://finance.yahoo.com/news/rssindex',
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
        ]
        
        all_news = []
        
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:limit]:
                    all_news.append({
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'published': entry.get('published', ''),
                        'source': 'Yahoo Finance Global'
                    })
            except Exception as e:
                logger.warning(f"Error fetching from {feed_url}: {e}")
                continue
        
        return all_news[:limit]
        
    except Exception as e:
        logger.error(f"Error fetching global news: {e}")
        return []

def fetch_indonesia_financial_news(limit: int = 10) -> List[Dict]:
    """
    Fetch berita finansial Indonesia sebagai fallback
    """
    try:
        # Search untuk berita pasar saham Indonesia
        search_queries = [
            'IHSG Indonesia stock market',
            'Bursa Efek Indonesia',
            'Indonesia market news',
            'IDX composite'
        ]
        
        all_news = []
        
        for query in search_queries:
            try:
                # Use Yahoo Finance search
                import yfinance as yf
                # Gunakan ticker IHSG untuk news
                idx = yf.Ticker('^JKSE')  # Jakarta Stock Exchange Composite Index
                news_items = idx.news
                
                if news_items:
                    for item in news_items[:limit]:
                        all_news.append({
                            'title': item.get('title', ''),
                            'summary': item.get('summary', ''),
                            'link': item.get('link', ''),
                            'published': item.get('providerPublishTime', 0),
                            'source': item.get('publisher', 'Indonesia Market News')
                        })
                    break  # Stop jika sudah dapat news
                    
            except Exception as e:
                logger.warning(f"Error searching '{query}': {e}")
                continue
        
        return all_news[:limit]
        
    except Exception as e:
        logger.error(f"Error fetching Indonesia news: {e}")
        return []

def update_news_data(limit: int = 20):
    """
    Enhanced news fetching dengan fallback strategy:
    1. Coba ambil news spesifik ticker
    2. Jika tidak ada, ambil news regional/global
    """
    db = SessionLocal()
    try:
        logger.info("📰 Updating news data with fallback strategy...")

        for ticker in DEFAULT_TICKERS:
            clean_tick = clean_ticker(ticker)
            region = get_market_region(ticker)
            
            try:
                # === STEP 1: Coba news spesifik ticker ===
                tk = yf.Ticker(clean_tick)
                raw_news = tk.news
                
                news_found = False
                new_count = 0
                
                if raw_news and len(raw_news) > 0:
                    logger.info(f"  ✓ Found {len(raw_news)} specific news for {ticker}")
                    
                    for article in raw_news[:limit]:
                        url_article = article.get('link')
                        if not url_article:
                            continue

                        existing = db.query(News).filter(News.url == url_article).first()
                        if existing:
                            continue

                        title = article.get('title', '')
                        description = article.get('summary', '')
                        content_text = f"{title}. {description}"

                        sentiment = sentiment_analyzer.analyze_sentiment(content_text)

                        published_at = None
                        try:
                            ts = article.get('providerPublishTime')
                            if ts:
                                published_at = datetime.fromtimestamp(ts)
                            else:
                                published_at = datetime.utcnow()
                        except:
                            published_at = datetime.utcnow()

                        news_item = News(
                            ticker=clean_tick,
                            title=title[:500],
                            description=description[:1000],
                            url=url_article[:500],
                            source=article.get('publisher', 'Yahoo Finance')[:100],
                            published_at=published_at,
                            sentiment_score=sentiment['score'],
                            sentiment_label=sentiment['label'],
                            sentiment_confidence=sentiment['confidence'],
                            created_at=datetime.utcnow()
                        )

                        db.add(news_item)
                        new_count += 1
                        news_found = True
                    
                    if new_count > 0:
                        db.commit()
                        logger.info(f"  ✓ {ticker} - Added {new_count} specific articles")
                
                # === STEP 2: Fallback ke regional/global news ===
                if not news_found or new_count < 5:
                    logger.info(f"  ⚠️  {ticker} - Limited specific news, fetching {region} market news...")
                    
                    fallback_news = []
                    if region == 'indonesia':
                        fallback_news = fetch_indonesia_financial_news(limit=10)
                        logger.info(f"  → Fetched {len(fallback_news)} Indonesia market news")
                    else:
                        fallback_news = fetch_global_financial_news(limit=10)
                        logger.info(f"  → Fetched {len(fallback_news)} global market news")
                    
                    fallback_count = 0
                    for article in fallback_news:
                        url_article = article.get('link')
                        if not url_article:
                            continue
                        
                        existing = db.query(News).filter(News.url == url_article).first()
                        if existing:
                            continue
                        
                        title = article.get('title', '')
                        description = article.get('summary', '')
                        content_text = f"{title}. {description}"
                        
                        sentiment = sentiment_analyzer.analyze_sentiment(content_text)
                        
                        # Parse published date
                        published_at = datetime.utcnow()
                        try:
                            pub = article.get('published')
                            if isinstance(pub, int):
                                published_at = datetime.fromtimestamp(pub)
                            elif isinstance(pub, str):
                                from dateutil import parser
                                published_at = parser.parse(pub)
                        except:
                            pass
                        
                        news_item = News(
                            ticker=clean_tick,
                            title=f"[{region.upper()} MARKET] {title[:450]}",
                            description=description[:1000],
                            url=url_article[:500],
                            source=article.get('source', f'{region.title()} Market News')[:100],
                            published_at=published_at,
                            sentiment_score=sentiment['score'],
                            sentiment_label=sentiment['label'],
                            sentiment_confidence=sentiment['confidence'],
                            created_at=datetime.utcnow()
                        )
                        
                        db.add(news_item)
                        fallback_count += 1
                    
                    if fallback_count > 0:
                        db.commit()
                        logger.info(f"  ✓ {ticker} - Added {fallback_count} {region} market articles")
                
                total_added = new_count + (fallback_count if 'fallback_count' in locals() else 0)
                if total_added == 0:
                    logger.info(f"  ℹ️  {ticker} - No new articles (all existing)")

            except Exception as e:
                logger.error(f"  ✗ {ticker} - Error: {str(e)}")
                db.rollback()
                continue

        logger.info("✅ News update complete with fallback strategy\n")
        
    except Exception as e:
        logger.error(f"❌ News update error: {str(e)}")
        db.rollback()
    finally:
        db.close()


def get_recent_sentiment(ticker: str, limit: int = 15) -> float:
    """
    Enhanced sentiment calculation dengan bobot lebih tinggi untuk news terbaru
    """
    db = SessionLocal()
    try:
        clean_tick = clean_ticker(ticker)
        
        # Get news dari 30 hari terakhir
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        recent_news = db.query(News)\
                        .filter(
                            News.ticker == clean_tick,
                            News.published_at >= cutoff_date
                        )\
                        .order_by(News.published_at.desc())\
                        .limit(limit).all()
        
        if not recent_news:
            logger.warning(f"No news found for {ticker} in last 30 days")
            return 0.5  # Neutral
        
        logger.info(f"Analyzing {len(recent_news)} news articles for {ticker}")
        
        total_weight = 0
        weighted_sum = 0
        
        now = datetime.utcnow()
        
        for i, news in enumerate(recent_news):
            # Time decay weight (news lebih baru = bobot lebih tinggi)
            days_old = (now - news.published_at).days
            time_weight = 1.0 / (1 + days_old * 0.1)  # Decay 10% per hari
            
            # Position weight (news pertama = paling penting)
            position_weight = 1.0 / (1 + i * 0.05)  # Decay 5% per posisi
            
            # Confidence weight
            confidence = float(news.sentiment_confidence) if news.sentiment_confidence else 0.5
            
            # Combined weight
            final_weight = time_weight * position_weight * confidence
            
            weighted_sum += float(news.sentiment_score) * final_weight
            total_weight += final_weight
            
            logger.debug(
                f"  News #{i+1}: score={news.sentiment_score:.3f}, "
                f"age={days_old}d, weight={final_weight:.3f}"
            )
        
        avg_sentiment = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        logger.info(
            f"Sentiment for {ticker}: {avg_sentiment:.3f} "
            f"(from {len(recent_news)} articles)"
        )
        
        return float(avg_sentiment)
        
    except Exception as e:
        logger.error(f"Error calculating sentiment for {ticker}: {str(e)}")
        return 0.5
    finally:
        db.close()


def get_news_by_ticker(ticker: str, limit: int = 20):
    """Get news articles for specific ticker"""
    db = SessionLocal()
    try:
        clean_tick = clean_ticker(ticker)
        news = db.query(News)\
                 .filter(News.ticker == clean_tick)\
                 .order_by(News.published_at.desc())\
                 .limit(limit).all()
        return news
    finally:
        db.close()


def get_sentiment_distribution(ticker: str, days: int = 30) -> dict:
    """
    Get sentiment distribution dengan kategori lebih detail
    """
    db = SessionLocal()
    try:
        clean_tick = clean_ticker(ticker)
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        news = db.query(News)\
                 .filter(
                     News.ticker == clean_tick,
                     News.published_at >= cutoff_date
                 )\
                 .all()
        
        if not news:
            return {
                'positive': 0,
                'neutral': 0,
                'negative': 0,
                'very_positive': 0,
                'very_negative': 0,
                'total': 0,
                'avg_score': 0.5,
                'trend': 'neutral'
            }
        
        # Count by label
        positive = sum(1 for n in news if n.sentiment_label == 'positive')
        neutral = sum(1 for n in news if n.sentiment_label == 'neutral')
        negative = sum(1 for n in news if n.sentiment_label == 'negative')
        
        # Very positive/negative (confidence > 0.8)
        very_positive = sum(1 for n in news if n.sentiment_label == 'positive' and n.sentiment_confidence > 0.8)
        very_negative = sum(1 for n in news if n.sentiment_label == 'negative' and n.sentiment_confidence > 0.8)
        
        # Average score
        avg_score = sum(float(n.sentiment_score) for n in news) / len(news)
        
        # Trend (recent vs older)
        mid_point = len(news) // 2
        recent_avg = sum(float(n.sentiment_score) for n in news[:mid_point]) / max(mid_point, 1)
        older_avg = sum(float(n.sentiment_score) for n in news[mid_point:]) / max(len(news) - mid_point, 1)
        
        if recent_avg > older_avg + 0.1:
            trend = 'improving'
        elif recent_avg < older_avg - 0.1:
            trend = 'declining'
        else:
            trend = 'stable'
        
        return {
            'positive': positive,
            'neutral': neutral,
            'negative': negative,
            'very_positive': very_positive,
            'very_negative': very_negative,
            'total': len(news),
            'avg_score': avg_score,
            'trend': trend,
            'recent_avg': recent_avg,
            'older_avg': older_avg
        }
    finally:
        db.close()