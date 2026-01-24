# python_ai/services/prediction_service.py
"""
Stock Predictor - LightGBM with 5 Years Training, 30-Day Predictions
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Tuple, List
import logging
from datetime import datetime, timedelta

from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from python_ai.config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictor:
    """Enhanced Stock Predictor with 30-day forecasting"""
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = []
        
    def fetch_data(self, period: str = "5y") -> pd.DataFrame:
        """
        Fetch historical stock data from Yahoo Finance
        Period: '5y' for 5 years of data
        """
        try:
            logger.info(f"📊 Fetching {period} data for {self.ticker}...")
            
            stock = yf.Ticker(self.ticker)
            df = stock.history(period=period)
            
            if df.empty:
                raise ValueError(f"No data found for {self.ticker}")
            
            logger.info(f"✓ Fetched {len(df)} days of data")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {self.ticker}: {e}")
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create 73 technical indicators
        """
        try:
            logger.info(f"🔧 Engineering features for {self.ticker}...")
            
            df = df.copy()
            
            # Basic returns
            df['Returns'] = df['Close'].pct_change()
            df['Volatility'] = df['Returns'].rolling(window=20).std()
            
            # Moving Averages
            for period in config.SMA_PERIODS:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
            
            for period in config.EMA_PERIODS:
                df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
            
            # RSI
            for period in config.RSI_PERIODS:
                delta = df['Close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=period).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
                rs = gain / loss
                df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI_14']  # Default RSI
            
            # MACD
            exp1 = df['Close'].ewm(span=config.MACD_FAST, adjust=False).mean()
            exp2 = df['Close'].ewm(span=config.MACD_SLOW, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=config.MACD_SIGNAL, adjust=False).mean()
            df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            sma = df['Close'].rolling(window=config.BB_PERIOD).mean()
            std = df['Close'].rolling(window=config.BB_PERIOD).std()
            df['BB_Upper'] = sma + (std * config.BB_STD)
            df['BB_Middle'] = sma
            df['BB_Lower'] = sma - (std * config.BB_STD)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=config.ATR_PERIOD).mean()
            df['ATR_Ratio'] = df['ATR'] / df['Close']
            
            # ADX
            df['ADX'] = self._calculate_adx(df)
            df['DI_Plus'] = self._calculate_di_plus(df)
            df['DI_Minus'] = self._calculate_di_minus(df)
            
            # CCI
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: abs(x - x.mean()).mean())
            df['CCI'] = (tp - sma_tp) / (0.015 * mad)
            
            # Stochastic
            lowest_low = df['Low'].rolling(window=14).min()
            highest_high = df['High'].rolling(window=14).max()
            df['Stoch_K'] = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
            df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
            
            # Williams %R
            df['Williams_R'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
            
            # ROC
            df['ROC'] = df['Close'].pct_change(periods=12) * 100
            
            # MFI
            df['MFI'] = self._calculate_mfi(df)
            
            # OBV
            df['OBV'] = (df['Volume'] * (~df['Close'].diff().le(0) * 2 - 1)).cumsum()
            df['OBV_Change'] = df['OBV'].pct_change()
            
            # Volume indicators
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
            
            # Price patterns
            df['HL_Ratio'] = df['High'] / df['Low']
            df['OC_Ratio'] = df['Open'] / df['Close']
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
            
            for lag in [1, 2, 3]:
                df[f'Return_Lag_{lag}'] = df['Returns'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                df[f'Return_Mean_{window}'] = df['Returns'].rolling(window=window).mean()
                df[f'Return_Std_{window}'] = df['Returns'].rolling(window=window).std()
                df[f'High_Max_{window}'] = df['High'].rolling(window=window).max()
                df[f'Low_Min_{window}'] = df['Low'].rolling(window=window).min()
            
            # Volatility measures
            for window in [10, 20, 30]:
                df[f'Volatility_{window}d'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
            
            # Time features
            df['DayOfWeek'] = pd.to_datetime(df.index).dayofweek
            df['Month'] = pd.to_datetime(df.index).month
            df['Quarter'] = pd.to_datetime(df.index).quarter
            
            # Drop NaN
            df = df.dropna()
            
            logger.info(f"✓ Created {len(df.columns)} features")
            return df
            
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            raise
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
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
            
            return adx
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_di_plus(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate +DI"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            plus_dm = high.diff()
            plus_dm[plus_dm < 0] = 0
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
            
            return plus_di
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_di_minus(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate -DI"""
        try:
            high = df['High']
            low = df['Low']
            close = df['Close']
            
            minus_dm = -low.diff()
            minus_dm[minus_dm < 0] = 0
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            atr = tr.rolling(window=period).mean()
            minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
            
            return minus_di
        except:
            return pd.Series(0, index=df.index)
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index"""
        try:
            tp = (df['High'] + df['Low'] + df['Close']) / 3
            mf = tp * df['Volume']
            
            mf_pos = mf.where(tp.diff() > 0, 0).rolling(window=period).sum()
            mf_neg = mf.where(tp.diff() < 0, 0).rolling(window=period).sum()
            
            mfi = 100 - (100 / (1 + mf_pos / mf_neg))
            return mfi
        except:
            return pd.Series(50, index=df.index)
    
    def prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare sequences for training
        Uses TIME_STEP days lookback
        """
        try:
            feature_cols = [col for col in config.FEATURES if col in df.columns]
            self.feature_cols = feature_cols
            
            X = df[feature_cols].values
            y = df['Close'].values
            
            # Create sequences
            time_step = config.TIME_STEP
            X_seq = []
            y_seq = []
            
            for i in range(time_step, len(X)):
                X_seq.append(X[i-time_step:i].flatten())
                y_seq.append(y[i])
            
            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)
            
            logger.info(f"✓ Prepared {len(X_seq)} sequences with {len(feature_cols)} features")
            return X_seq, y_seq, feature_cols
            
        except Exception as e:
            logger.error(f"Sequence preparation error: {e}")
            raise
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Train LightGBM model with cross-validation
        """
        try:
            logger.info(f"🎯 Training LightGBM for {self.ticker}...")
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=config.MODEL_CONFIG['cv_folds'])
            
            scores = {'mae': [], 'rmse': [], 'r2': []}
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Scale
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                
                # Train
                model = LGBMRegressor(**config.LGBM_PARAMS)
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_val_scaled)
                
                # Metrics
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
                scores['mae'].append(mae)
                scores['rmse'].append(rmse)
                scores['r2'].append(r2)
                
                logger.info(f"  Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
            
            # Final model on all data
            X_scaled = self.scaler.fit_transform(X)
            self.model = LGBMRegressor(**config.LGBM_PARAMS)
            self.model.fit(X_scaled, y)
            
            avg_scores = {
                'mae': np.mean(scores['mae']),
                'rmse': np.mean(scores['rmse']),
                'r2': np.mean(scores['r2']),
                'confidence': max(0, min(1, np.mean(scores['r2'])))  # Clamp 0-1
            }
            
            logger.info(f"✓ Training complete: R²={avg_scores['r2']:.4f}, Confidence={avg_scores['confidence']:.4f}")
            
            return avg_scores
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
    
    def predict_30_days(self, df: pd.DataFrame, last_sequence: np.ndarray) -> Dict[str, float]:
        """
        Predict next 30 days
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained")
            
            logger.info(f"📅 Predicting 30 days for {self.ticker}...")
            
            predictions = {}
            current_sequence = last_sequence.copy()
            
            for day in range(1, 31):
                # Predict next day
                current_scaled = self.scaler.transform(current_sequence.reshape(1, -1))
                next_price = self.model.predict(current_scaled)[0]
                
                predictions[f'day_{day}'] = float(next_price)
                
                # Update sequence for next prediction
                # This is a simplified approach - in production you might want more sophisticated rolling
                current_sequence = np.roll(current_sequence, -len(self.feature_cols))
                current_sequence[-len(self.feature_cols):] = current_sequence[-2*len(self.feature_cols):-len(self.feature_cols)]
            
            logger.info(f"✓ 30-day predictions complete")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise