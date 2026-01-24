# File: python_ai/services/technical_service.py

import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """
    Technical Analysis Service - Fixed untuk handling multi-dimensional data
    """
    
    def _safe_series(self, data, column='Close'):
        """Safely extract 1D series from DataFrame"""
        try:
            series = data[column]
            # Jika multi-dimensional, flatten
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            # Squeeze untuk remove extra dimensions
            series = series.squeeze()
            return series
        except Exception as e:
            logger.error(f"Error extracting series: {e}")
            return pd.Series([0])
    
    def _safe_last_value(self, series) -> float:
        """Safely extract last value from Series"""
        try:
            if isinstance(series, pd.Series):
                val = series.iloc[-1]
                if isinstance(val, (pd.Series, np.ndarray)):
                    return float(val.flatten()[0])
                return float(val)
            elif isinstance(series, np.ndarray):
                return float(series.flatten()[-1])
            else:
                return float(series)
        except Exception as e:
            logger.warning(f"Error extracting last value: {e}")
            return 0.0
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            close = self._safe_series(data, 'Close')
            
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return self._safe_last_value(rsi)
        except Exception as e:
            logger.error(f"RSI calculation error: {e}")
            return 50.0
    
    def calculate_macd(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            close = self._safe_series(data, 'Close')
            
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            diff = macd - signal
            
            return {
                'macd': self._safe_last_value(macd),
                'macd_signal': self._safe_last_value(signal),
                'macd_diff': self._safe_last_value(diff)
            }
        except Exception as e:
            logger.error(f"MACD calculation error: {e}")
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_diff': 0.0}
    
    def calculate_sma(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Simple Moving Average"""
        try:
            close = self._safe_series(data, 'Close')
            sma = close.rolling(window=period).mean()
            return self._safe_last_value(sma)
        except Exception as e:
            logger.error(f"SMA-{period} calculation error: {e}")
            return 0.0
    
    def calculate_ema(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Exponential Moving Average"""
        try:
            close = self._safe_series(data, 'Close')
            ema = close.ewm(span=period, adjust=False).mean()
            return self._safe_last_value(ema)
        except Exception as e:
            logger.error(f"EMA-{period} calculation error: {e}")
            return 0.0
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        try:
            close = self._safe_series(data, 'Close')
            
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            
            upper = sma + (std * 2)
            lower = sma - (std * 2)
            
            return {
                'bollinger_high': self._safe_last_value(upper),
                'bollinger_mid': self._safe_last_value(sma),
                'bollinger_low': self._safe_last_value(lower)
            }
        except Exception as e:
            logger.error(f"Bollinger Bands calculation error: {e}")
            return {'bollinger_high': 0.0, 'bollinger_mid': 0.0, 'bollinger_low': 0.0}
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            high = self._safe_series(data, 'High')
            low = self._safe_series(data, 'Low')
            close = self._safe_series(data, 'Close')
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            return self._safe_last_value(atr)
        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.0
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        try:
            high = self._safe_series(data, 'High')
            low = self._safe_series(data, 'Low')
            close = self._safe_series(data, 'Close')
            
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            return self._safe_last_value(adx)
        except Exception as e:
            logger.error(f"ADX calculation error: {e}")
            return 0.0
    
    def calculate_cci(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate Commodity Channel Index"""
        try:
            high = self._safe_series(data, 'High')
            low = self._safe_series(data, 'Low')
            close = self._safe_series(data, 'Close')
            
            tp = (high + low + close) / 3
            sma = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
            
            cci = (tp - sma) / (0.015 * mad)
            
            return self._safe_last_value(cci)
        except Exception as e:
            logger.error(f"CCI calculation error: {e}")
            return 0.0
    
    def calculate_stochastic(self, data: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Calculate Stochastic Oscillator"""
        try:
            high = self._safe_series(data, 'High')
            low = self._safe_series(data, 'Low')
            close = self._safe_series(data, 'Close')
            
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            
            stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
            stoch_d = stoch_k.rolling(window=3).mean()
            
            return {
                'stoch_k': self._safe_last_value(stoch_k),
                'stoch_d': self._safe_last_value(stoch_d)
            }
        except Exception as e:
            logger.error(f"Stochastic calculation error: {e}")
            return {'stoch_k': 0.0, 'stoch_d': 0.0}
    
    def calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Williams %R"""
        try:
            high = self._safe_series(data, 'High')
            low = self._safe_series(data, 'Low')
            close = self._safe_series(data, 'Close')
            
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
            
            return self._safe_last_value(williams_r)
        except Exception as e:
            logger.error(f"Williams %R calculation error: {e}")
            return 0.0
    
    def calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Money Flow Index"""
        try:
            high = self._safe_series(data, 'High')
            low = self._safe_series(data, 'Low')
            close = self._safe_series(data, 'Close')
            volume = self._safe_series(data, 'Volume')
            
            tp = (high + low + close) / 3
            mf = tp * volume
            
            mf_pos = mf.where(tp.diff() > 0, 0).rolling(window=period).sum()
            mf_neg = mf.where(tp.diff() < 0, 0).rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + mf_pos / mf_neg))
            
            return self._safe_last_value(mfi)
        except Exception as e:
            logger.error(f"MFI calculation error: {e}")
            return 50.0
    
    def calculate_obv(self, data: pd.DataFrame) -> float:
        """Calculate On Balance Volume"""
        try:
            close = self._safe_series(data, 'Close')
            volume = self._safe_series(data, 'Volume')
            
            obv = (volume * (~close.diff().le(0) * 2 - 1)).cumsum()
            
            return self._safe_last_value(obv)
        except Exception as e:
            logger.error(f"OBV calculation error: {e}")
            return 0.0
    
    def calculate_volatility(self, data: pd.DataFrame, period: int) -> float:
        """Calculate Volatility (Standard Deviation of Returns)"""
        try:
            close = self._safe_series(data, 'Close')
            returns = close.pct_change()
            volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
            
            return self._safe_last_value(volatility)
        except Exception as e:
            logger.error(f"Volatility-{period} calculation error: {e}")
            return 0.0
    
    def calculate_technical_score(self, data: pd.DataFrame) -> float:
        """Calculate overall technical score (0-100)"""
        try:
            close = self._safe_series(data, 'Close')
            current_price = self._safe_last_value(close)
            
            rsi = self.calculate_rsi(data)
            sma_20 = self.calculate_sma(data, 20)
            sma_50 = self.calculate_sma(data, 50)
            macd_data = self.calculate_macd(data)
            
            score = 50.0  # Base score
            
            # RSI scoring (30-70 range is neutral, outside is extreme)
            if 40 <= rsi <= 60:
                score += 20
            elif 30 <= rsi <= 70:
                score += 10
            elif rsi < 30:
                score += 5  # Oversold - potential bounce
            else:
                score -= 5  # Overbought - potential reversal
            
            # Price vs SMA scoring
            if current_price > sma_20:
                score += 15
            if current_price > sma_50:
                score += 15
            
            # MACD scoring
            if macd_data['macd'] > macd_data['macd_signal']:
                score += 10
            
            return min(100.0, max(0.0, score))
            
        except Exception as e:
            logger.error(f"Technical score calculation error: {e}")
            return 50.0
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators"""
        try:
            # Moving Averages
            sma_20 = self.calculate_sma(data, 20)
            sma_50 = self.calculate_sma(data, 50)
            sma_200 = self.calculate_sma(data, 200)
            ema_12 = self.calculate_ema(data, 12)
            ema_26 = self.calculate_ema(data, 26)
            
            # Oscillators
            rsi = self.calculate_rsi(data)
            macd_data = self.calculate_macd(data)
            
            # Bollinger Bands
            bb = self.calculate_bollinger_bands(data)
            
            # Other Indicators
            atr = self.calculate_atr(data)
            adx = self.calculate_adx(data)
            cci = self.calculate_cci(data)
            stoch = self.calculate_stochastic(data)
            williams_r = self.calculate_williams_r(data)
            mfi = self.calculate_mfi(data)
            obv = self.calculate_obv(data)
            
            # Volatility
            vol_10d = self.calculate_volatility(data, 10)
            vol_30d = self.calculate_volatility(data, 30)
            
            # Technical Score
            tech_score = self.calculate_technical_score(data)
            
            indicators = {
                'rsi': float(rsi),
                'macd': float(macd_data['macd']),
                'macd_signal': float(macd_data['macd_signal']),
                'macd_diff': float(macd_data['macd_diff']),
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'sma_200': float(sma_200),
                'ema_12': float(ema_12),
                'ema_26': float(ema_26),
                'bollinger_high': float(bb['bollinger_high']),
                'bollinger_mid': float(bb['bollinger_mid']),
                'bollinger_low': float(bb['bollinger_low']),
                'atr': float(atr),
                'adx': float(adx),
                'cci': float(cci),
                'stoch_k': float(stoch['stoch_k']),
                'stoch_d': float(stoch['stoch_d']),
                'williams_r': float(williams_r),
                'mfi': float(mfi),
                'obv': float(obv),
                'volatility_10d': float(vol_10d),
                'volatility_30d': float(vol_30d),
                'technical_score': float(tech_score)
            }
            
            # Validasi semua nilai adalah float scalar
            for key in indicators:
                if not isinstance(indicators[key], (int, float)):
                    logger.warning(f"Non-scalar value for {key}: {type(indicators[key])}")
                    indicators[key] = 0.0
                elif np.isnan(indicators[key]) or np.isinf(indicators[key]):
                    logger.warning(f"Invalid value for {key}: {indicators[key]}")
                    indicators[key] = 0.0
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating all indicators: {e}", exc_info=True)
            # Return default values
            return {
                'rsi': 50.0,
                'macd': 0.0,
                'macd_signal': 0.0,
                'macd_diff': 0.0,
                'sma_20': 0.0,
                'sma_50': 0.0,
                'sma_200': 0.0,
                'ema_12': 0.0,
                'ema_26': 0.0,
                'bollinger_high': 0.0,
                'bollinger_mid': 0.0,
                'bollinger_low': 0.0,
                'atr': 0.0,
                'adx': 0.0,
                'cci': 0.0,
                'stoch_k': 0.0,
                'stoch_d': 0.0,
                'williams_r': 0.0,
                'mfi': 50.0,
                'obv': 0.0,
                'volatility_10d': 0.0,
                'volatility_30d': 0.0,
                'technical_score': 50.0
            }