from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "ProsusAI/finbert"
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                return_all_scores=False
            )
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {e}")
            raise

    def analyze_sentiment(self, text: str) -> dict:
        """Analyze single text"""
        try:
            result = self.pipeline(text)[0]
            return {
                'label': result['label'].lower(),
                'score': float(result['score']),
                'confidence': float(result['score'])
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'label': 'neutral', 'score': 0.5, 'confidence': 0.5}

    def analyze_batch(self, texts: list) -> dict:
        """Analyze batch of texts"""
        results = []
        for text in texts:
            sentiment = self.analyze_sentiment(text)
            results.append(sentiment['score'])
        average = sum(results) / len(results) if results else 0.5
        return {'average_sentiment': average}

# Global instance
sentiment_analyzer = SentimentAnalyzer()
