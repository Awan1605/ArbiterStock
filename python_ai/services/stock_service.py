# File: python_ai/services/stock_service.py
# HYBRID MODEL: LSTM + LightGBM - FIXED VERSION (No Infinity/NaN errors)

import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l1_l2

from python_ai.models_db import StockDetail, PredictionHistory
from python_ai.database import SessionLocal
from python_ai.services.technical_service import TechnicalAnalyzer
from python_ai.services.financial_service import FinancialAnalyzer
from python_ai.services.news_service import get_recent_sentiment
from python_ai.services.llm_service import llm_service
from python_ai.config.settings import config

logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

class HybridStockPredictor:
    """
    Hybrid model: LSTM for temporal patterns + LightGBM for feature importance
    """
    def __init__(self):
        self.time_step = 60
        self.scaler_prices = MinMaxScaler(feature_range=(0, 1))
        self.scaler_features = RobustScaler()  # More robust to outliers
        
        # LSTM model
        self.lstm_model = None
        
        # LightGBM model
        lgbm_config = {
            'n_estimators': 1500,
            'learning_rate': 0.008,
            'max_depth': 10,
            'num_leaves': 127,
            'min_child_samples': 40,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.4,
            'reg_lambda': 0.4,
            'min_split_gain': 0.015,
            'random_state': 42,
            'force_col_wise': True,
            'verbose': -1,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
        }
        self.lgbm_model = LGBMRegressor(**lgbm_config)
        
        self.model_version = "v3.0-hybrid-lstm-lgbm"
        self.skip_tickers = set()
        
    def build_lstm_model(self, input_shape_seq: tuple, input_shape_features: int) -> Model:
        """Build advanced LSTM architecture"""
        # Sequential input
        seq_input = Input(shape=input_shape_seq, name='sequence_input')
        
        x = LSTM(128, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(seq_input)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = LSTM(32, return_sequences=False, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Additional features
        features_input = Input(shape=(input_shape_features,), name='features_input')
        y = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(features_input)
        y = BatchNormalization()(y)
        y = Dropout(0.2)(y)
        y = Dense(32, activation='relu')(y)
        y = Dropout(0.2)(y)
        
        # Combine
        combined = Concatenate()([x, y])
        
        z = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(combined)
        z = BatchNormalization()(z)
        z = Dropout(0.2)(z)
        z = Dense(32, activation='relu')(z)
        z = Dropout(0.1)(z)
        output = Dense(1, activation='linear')(z)
        
        model = Model(inputs=[seq_input, features_input], outputs=output)
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])
        
        return model

    def safe_divide(self, numerator, denominator, fill_value=0.0):
        """Safely divide with protection against division by zero"""
        result = np.where(
            np.abs(denominator) < 1e-10,
            fill_value,
            numerator / denominator
        )
        return result

    def prepare_data(self, ticker: str) -> Tuple[Dict, pd.DataFrame]:
        """Prepare data with robust error handling"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1825)
            
            logger.info(f"Fetching {ticker} data from {start_date.date()} to {end_date.date()}")
            
            data = yf.download(
                ticker,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False
            )
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            if len(data) < 500:
                raise ValueError(f"Insufficient data for {ticker}: only {len(data)} days")

            logger.info(f"Processing {len(data)} days of data for {ticker}")

            # === SAFE FEATURE ENGINEERING ===
            epsilon = 1e-10
            
            # Returns
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log((data['Close'] + epsilon) / (data['Close'].shift(1) + epsilon))
            data['Price_Range'] = self.safe_divide(data['High'] - data['Low'], data['Close'], 0)
            data['Gap'] = self.safe_divide(data['Open'] - data['Close'].shift(1), data['Close'].shift(1), 0)
            data['Volume_Change'] = data['Volume'].pct_change()
            data['Volume_MA_Ratio'] = self.safe_divide(data['Volume'], data['Volume'].rolling(20).mean(), 1)
            
            # Volatility
            for window in [5, 10, 20, 30]:
                data[f'Volatility_{window}d'] = data['Returns'].rolling(window=window).std()
            
            # RSI
            for period in [14, 21]:
                delta = data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = self.safe_divide(gain, loss, 0)
                data[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
            ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD'] = ema_12 - ema_26
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
            
            # Moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                data[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
                data[f'EMA_{period}'] = data['Close'].ewm(span=period, adjust=False).mean()
            
            # Price ratios
            data['Price_to_SMA20'] = self.safe_divide(data['Close'], data['SMA_20'], 1)
            data['Price_to_SMA50'] = self.safe_divide(data['Close'], data['SMA_50'], 1)
            data['Price_to_SMA200'] = self.safe_divide(data['Close'], data['SMA_200'], 1)
            data['SMA_20_50_Cross'] = self.safe_divide(data['SMA_20'] - data['SMA_50'], data['Close'], 0)
            data['SMA_50_200_Cross'] = self.safe_divide(data['SMA_50'] - data['SMA_200'], data['Close'], 0)
            
            # ATR
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift())
            low_close = np.abs(data['Low'] - data['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['ATR'] = tr.rolling(window=14).mean()
            data['ATR_Percent'] = self.safe_divide(data['ATR'], data['Close'], 0)
            
            # Bollinger Bands
            for period in [20, 50]:
                bb_mid = data['Close'].rolling(window=period).mean()
                bb_std = data['Close'].rolling(window=period).std()
                data[f'BB_{period}_Upper'] = bb_mid + (bb_std * 2)
                data[f'BB_{period}_Lower'] = bb_mid - (bb_std * 2)
                data[f'BB_{period}_Width'] = self.safe_divide(
                    data[f'BB_{period}_Upper'] - data[f'BB_{period}_Lower'], 
                    bb_mid, 0
                )
                data[f'BB_{period}_Position'] = self.safe_divide(
                    data['Close'] - data[f'BB_{period}_Lower'],
                    data[f'BB_{period}_Upper'] - data[f'BB_{period}_Lower'], 
                    0.5
                )
            
            # Stochastic
            low_14 = data['Low'].rolling(window=14).min()
            high_14 = data['High'].rolling(window=14).max()
            data['Stoch_K'] = 100 * self.safe_divide(data['Close'] - low_14, high_14 - low_14, 0.5)
            data['Stoch_D'] = data['Stoch_K'].rolling(window=3).mean()
            
            # ADX
            plus_dm = data['High'].diff()
            minus_dm = -data['Low'].diff()
            plus_dm = plus_dm.where(plus_dm > 0, 0)
            minus_dm = minus_dm.where(minus_dm > 0, 0)
            tr_14 = tr.rolling(window=14).sum()
            plus_di = 100 * self.safe_divide(plus_dm.rolling(window=14).sum(), tr_14, 0)
            minus_di = 100 * self.safe_divide(minus_dm.rolling(window=14).sum(), tr_14, 0)
            # Convert to Series first to avoid numpy array issues
            plus_di_series = pd.Series(plus_di, index=data.index)
            minus_di_series = pd.Series(minus_di, index=data.index)
            dx = 100 * ((plus_di_series - minus_di_series).abs() / (plus_di_series + minus_di_series + 1e-10))
            dx = dx.fillna(0).replace([np.inf, -np.inf], 0)
            data['ADX'] = dx.rolling(window=14).mean()
            data['DI_Diff'] = plus_di_series - minus_di_series
            
            # OBV
            obv = [0]
            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv.append(obv[-1] + data['Volume'].iloc[i])
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv.append(obv[-1] - data['Volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            data['OBV'] = obv
            data['OBV_MA'] = data['OBV'].rolling(window=20).mean()
            data['OBV_Trend'] = self.safe_divide(data['OBV'], data['OBV_MA'], 1)
            
            # Williams %R
            for period in [14, 28]:
                high_n = data['High'].rolling(window=period).max()
                low_n = data['Low'].rolling(window=period).min()
                data[f'Williams_R_{period}'] = -100 * self.safe_divide(high_n - data['Close'], high_n - low_n, 0.5)
            
            # Momentum & ROC
            for period in [5, 10, 20]:
                data[f'Momentum_{period}'] = data['Close'].pct_change(periods=period)
                data[f'ROC_{period}'] = 100 * self.safe_divide(
                    data['Close'] - data['Close'].shift(period),
                    data['Close'].shift(period), 0
                )
            
            # CCI
            tp = (data['High'] + data['Low'] + data['Close']) / 3
            tp_mean = tp.rolling(window=20).mean()
            tp_std = tp.rolling(window=20).std()
            data['CCI'] = self.safe_divide(tp - tp_mean, 0.015 * tp_std, 0)
            
            # Lagged features
            for lag in [1, 2, 3, 5, 10, 20, 30]:
                data[f'Close_Lag_{lag}'] = data['Close'].shift(lag)
                data[f'Volume_Lag_{lag}'] = data['Volume'].shift(lag)
                data[f'Returns_Lag_{lag}'] = data['Returns'].shift(lag)
            
            # Time features
            data['Day_of_Week'] = pd.to_datetime(data.index).dayofweek
            data['Day_of_Month'] = pd.to_datetime(data.index).day
            data['Month'] = pd.to_datetime(data.index).month
            data['Quarter'] = pd.to_datetime(data.index).quarter
            data['Day_Sin'] = np.sin(2 * np.pi * data['Day_of_Week'] / 7)
            data['Day_Cos'] = np.cos(2 * np.pi * data['Day_of_Week'] / 7)
            data['Month_Sin'] = np.sin(2 * np.pi * data['Month'] / 12)
            data['Month_Cos'] = np.cos(2 * np.pi * data['Month'] / 12)
            
            # === CRITICAL DATA CLEANING ===
            # Fill NaN values
            data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Replace infinity with 0
            data = data.replace([np.inf, -np.inf], 0)
            
            # Clip extreme outliers (99.9th percentile)
            for col in data.select_dtypes(include=[np.number]).columns:
                q99 = data[col].quantile(0.999)
                q01 = data[col].quantile(0.001)
                if not np.isnan(q99) and not np.isnan(q01):
                    data[col] = data[col].clip(lower=q01, upper=q99)
            
            # Final safety check
            data = data.replace([np.inf, -np.inf, np.nan], 0)
            
            # Prepare price data
            price_data = data['Close'].values.reshape(-1, 1)
            price_data = np.nan_to_num(price_data, nan=0.0, posinf=0.0, neginf=0.0)
            scaled_prices = self.scaler_prices.fit_transform(price_data)
            
            # Prepare features
            feature_columns = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Volume']]
            features = data[feature_columns].values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            scaled_features = self.scaler_features.fit_transform(features)
            
            # Create sequences
            X_seq, X_feat, X_lgbm, y = [], [], [], []
            
            for i in range(self.time_step, len(scaled_prices)):
                X_seq.append(scaled_prices[i-self.time_step:i])
                X_feat.append(scaled_features[i])
                lgbm_features = np.concatenate([
                    scaled_prices[i-self.time_step:i].flatten(),
                    scaled_features[i]
                ])
                X_lgbm.append(lgbm_features)
                y.append(scaled_prices[i, 0])
            
            X_seq = np.array(X_seq)
            X_feat = np.array(X_feat)
            X_lgbm = np.array(X_lgbm)
            y = np.array(y)
            
            logger.info(f"Created {len(X_seq)} sequences - LSTM: {X_seq.shape}, Features: {X_feat.shape}, LightGBM: {X_lgbm.shape}")
            
            return {
                'X_seq': X_seq,
                'X_feat': X_feat,
                'X_lgbm': X_lgbm,
                'y': y,
                'scaled_prices': scaled_prices,
                'scaled_features': scaled_features
            }, data

        except Exception as e:
            logger.error(f"Error preparing data for {ticker}: {e}")
            raise

    def train_hybrid_model(self, data_dict: Dict, last_price: float) -> Dict[str, float]:
        """Train LSTM + LightGBM ensemble"""
        try:
            X_seq = data_dict['X_seq']
            X_feat = data_dict['X_feat']
            X_lgbm = data_dict['X_lgbm']
            y = data_dict['y']
            
            if len(X_seq) < 100:
                return {
                    'model_confidence': 35.0, 'r2': 0.0, 'mape': 80.0,
                    'mae': last_price * 0.08, 'rmse': last_price * 0.12,
                    'samples': len(X_seq)
                }
            
            # Split data
            train_size = int(0.85 * len(X_seq))
            X_seq_train, X_seq_val = X_seq[:train_size], X_seq[train_size:]
            X_feat_train, X_feat_val = X_feat[:train_size], X_feat[train_size:]
            X_lgbm_train, X_lgbm_val = X_lgbm[:train_size], X_lgbm[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # === TRAIN LSTM ===
            logger.info("Training LSTM model...")
            
            self.lstm_model = self.build_lstm_model(
                input_shape_seq=(X_seq.shape[1], X_seq.shape[2]),
                input_shape_features=X_feat.shape[1]
            )
            
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=0
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.00001,
                verbose=0
            )
            
            history = self.lstm_model.fit(
                [X_seq_train, X_feat_train],
                y_train,
                validation_data=([X_seq_val, X_feat_val], y_val),
                epochs=100,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0  # Silent training
            )
            
            lstm_pred_val = self.lstm_model.predict([X_seq_val, X_feat_val], verbose=0).flatten()
            
            # === TRAIN LIGHTGBM ===
            logger.info("Training LightGBM model...")
            
            self.lgbm_model.fit(
                X_lgbm_train,
                y_train,
                eval_set=[(X_lgbm_val, y_val)],
                callbacks=[
                    early_stopping(stopping_rounds=50, verbose=False),
                    log_evaluation(period=0)
                ]
            )
            
            lgbm_pred_val = self.lgbm_model.predict(X_lgbm_val)
            
            # === ENSEMBLE ===
            ensemble_pred_val = 0.6 * lstm_pred_val + 0.4 * lgbm_pred_val
            
            # Calculate metrics
            r2_lstm = r2_score(y_val, lstm_pred_val)
            r2_lgbm = r2_score(y_val, lgbm_pred_val)
            r2_ensemble = r2_score(y_val, ensemble_pred_val)
            
            mae_scaled = mean_absolute_error(y_val, ensemble_pred_val)
            mae_price = mae_scaled * last_price
            
            rmse_scaled = np.sqrt(mean_squared_error(y_val, ensemble_pred_val))
            rmse_price = rmse_scaled * last_price
            
            epsilon = 1e-10
            y_val_safe = np.where(np.abs(y_val) < epsilon, epsilon, y_val)
            mape = np.mean(np.abs((y_val - ensemble_pred_val) / y_val_safe)) * 100
            
            # Direction accuracy
            actual_direction = np.sign(np.diff(y_val))
            pred_direction = np.sign(np.diff(ensemble_pred_val))
            direction_accuracy = np.mean(actual_direction == pred_direction) * 100
            
            # === CONFIDENCE CALCULATION ===
            r2 = r2_ensemble
            
            if r2 < 0:
                base_confidence = 30.0
            elif r2 < 0.4:
                base_confidence = 35.0 + (r2 * 50.0)
            elif r2 < 0.6:
                base_confidence = 55.0 + ((r2 - 0.4) * 50.0)
            elif r2 < 0.75:
                base_confidence = 65.0 + ((r2 - 0.6) * 40.0)
            elif r2 < 0.85:
                base_confidence = 71.0 + ((r2 - 0.75) * 40.0)
            elif r2 < 0.90:
                base_confidence = 75.0 + ((r2 - 0.85) * 40.0)
            else:
                base_confidence = 77.0 + ((r2 - 0.90) * 80.0)
            
            # Adjustments
            mape_adj = 6 if mape < 1 else (4 if mape < 2 else (2 if mape < 3 else (0 if mape < 5 else (-3 if mape < 8 else -6))))
            data_bonus = 8 if len(X_seq) > 1000 else (6 if len(X_seq) > 800 else (4 if len(X_seq) > 600 else 2))
            direction_bonus = 6 if direction_accuracy > 70 else (4 if direction_accuracy > 60 else (2 if direction_accuracy > 55 else -2))
            mae_pct = (mae_price / last_price) * 100
            mae_bonus = 5 if mae_pct < 1 else (3 if mae_pct < 2 else (1 if mae_pct < 3 else (-2 if mae_pct < 5 else -4)))
            hybrid_bonus = 5 if r2_ensemble > max(r2_lstm, r2_lgbm) else 3
            
            raw_confidence = base_confidence + mape_adj + data_bonus + direction_bonus + mae_bonus + hybrid_bonus
            model_confidence = max(40.0, min(88.0, raw_confidence))
            
            logger.info(
                f"\n{'='*60}\n"
                f"HYBRID MODEL RESULTS\n"
                f"{'='*60}\n"
                f"LSTM R²: {r2_lstm:.4f}\n"
                f"LightGBM R²: {r2_lgbm:.4f}\n"
                f"Ensemble R²: {r2_ensemble:.4f}\n"
                f"Confidence: {model_confidence:.1f}%\n"
                f"MAPE: {mape:.2f}%, MAE: ${mae_price:.2f}\n"
                f"Direction: {direction_accuracy:.1f}%\n"
                f"{'='*60}"
            )
            
            return {
                'model_confidence': float(model_confidence),
                'r2': float(r2_ensemble),
                'r2_lstm': float(r2_lstm),
                'r2_lgbm': float(r2_lgbm),
                'mape': float(mape),
                'mae': float(mae_price),
                'rmse': float(rmse_price),
                'direction_accuracy': float(direction_accuracy),
                'samples': len(X_seq)
            }
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {
                'model_confidence': 35.0, 'r2': 0.0, 'r2_lstm': 0.0, 'r2_lgbm': 0.0,
                'mape': 100.0, 'mae': last_price * 0.1, 'rmse': last_price * 0.15,
                'samples': 0
            }

    def predict_30_days(self, data_dict: Dict, original_data: pd.DataFrame) -> List[Dict[str, any]]:
        """Predict 30 days ahead using ensemble"""
        predictions = []
        try:
            X_seq = data_dict['X_seq']
            X_feat = data_dict['X_feat']
            X_lgbm = data_dict['X_lgbm']
            
            current_seq = X_seq[-1:].copy()
            current_feat = X_feat[-1:].copy()
            current_lgbm = X_lgbm[-1:].copy()
            
            last_date = pd.to_datetime(original_data.index[-1])
            
            for day in range(1, 31):
                # LSTM prediction
                lstm_pred = self.lstm_model.predict([current_seq, current_feat], verbose=0)[0, 0]
                
                # LightGBM prediction
                lgbm_pred = self.lgbm_model.predict(current_lgbm)[0]
                
                # Ensemble (60% LSTM, 40% LightGBM)
                ensemble_pred = 0.6 * lstm_pred + 0.4 * lgbm_pred
                
                # Scale back
                pred_price = float(self.scaler_prices.inverse_transform([[ensemble_pred]])[0, 0])
                
                pred_date = last_date + timedelta(days=day)
                predictions.append({
                    'day': day,
                    'date': pred_date.strftime('%Y-%m-%d'),
                    'price': pred_price
                })
                
                # Update sequences
                new_seq = np.roll(current_seq[0], -1, axis=0)
                new_seq[-1, 0] = ensemble_pred
                current_seq = new_seq.reshape(1, current_seq.shape[1], current_seq.shape[2])
                
                current_lgbm = np.concatenate([
                    current_seq[0].flatten(),
                    current_feat[0]
                ]).reshape(1, -1)
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            last_price = float(original_data['Close'].iloc[-1])
            last_date = pd.to_datetime(original_data.index[-1])
            predictions = [
                {
                    'day': day,
                    'date': (last_date + timedelta(days=day)).strftime('%Y-%m-%d'),
                    'price': last_price
                }
                for day in range(1, 31)
            ]
        
        return predictions

    def calculate_combined_score(self, technical_score: float, fundamental_score: float, 
                                news_score: float, model_confidence: float) -> Tuple[float, float]:
        tech = technical_score / 100.0 if technical_score > 1.0 else technical_score
        fund = fundamental_score / 100.0 if fundamental_score > 1.0 else fundamental_score
        news = news_score / 100.0 if news_score > 1.0 else news_score
        weights = {'technical': 0.35, 'financial': 0.35, 'news': 0.30}
        combined = (tech * weights['technical'] * 100 + fund * weights['financial'] * 100 + news * weights['news'] * 100)
        return float(combined), float(model_confidence)

    def _safe_extract_value(self, data: any) -> float:
        try:
            if isinstance(data, (int, float)):
                return float(data)
            elif isinstance(data, np.ndarray):
                return float(data.flatten()[0])
            elif isinstance(data, pd.Series):
                return float(data.iloc[0])
            else:
                return 0.0
        except:
            return 0.0

    def update_stock_detail(self, ticker: str):
        if ticker in self.skip_tickers:
            return None
        
        db = SessionLocal()
        try:
            logger.info(f"\n{'='*80}\nProcessing {ticker} with HYBRID MODEL\n{'='*80}")
            
            try:
                data_dict, historical_data = self.prepare_data(ticker)
            except ValueError as e:
                logger.error(f"Data error: {e}")
                self.skip_tickers.add(ticker)
                raise
            
            current_price = self._safe_extract_value(historical_data['Close'].iloc[-1])
            
            # Train hybrid model
            training_results = self.train_hybrid_model(data_dict, current_price)
            model_confidence = training_results['model_confidence']
            
            # Get 30-day predictions
            daily_predictions = self.predict_30_days(data_dict, historical_data)

            # Technical analysis
            technical_data = {}
            try:
                technical_service = TechnicalAnalyzer()
                technical_data = technical_service.calculate_all_indicators(historical_data)
                for key in technical_data:
                    technical_data[key] = self._safe_extract_value(technical_data[key])
            except Exception as e:
                logger.error(f"Technical error: {e}")
                technical_data = {'technical_score': 50.0}

            # Fundamental analysis
            try:
                financial_service = FinancialAnalyzer()
                financial_score = financial_service.fetch_score(ticker)
                financial_data = {'fundamental_score': financial_score * 100}
            except Exception as e:
                logger.error(f"Fundamental error: {e}")
                financial_data = {'fundamental_score': 50.0}

            # News sentiment
            try:
                news_score_raw = get_recent_sentiment(ticker)
                news_score = float(news_score_raw) * 100 if news_score_raw else 50.0
            except Exception as e:
                logger.error(f"News error: {e}")
                news_score = 50.0

            # Calculate combined score
            combined_score, overall_confidence = self.calculate_combined_score(
                technical_data.get('technical_score', 50.0),
                financial_data.get('fundamental_score', 50.0),
                news_score,
                model_confidence
            )

            # LLM analysis
            llm_data = {
                'current_price': current_price,
                'daily_predictions': daily_predictions,
                'confidence': overall_confidence,
                'technical_score': technical_data.get('technical_score', 50.0),
                'fundamental_score': financial_data.get('fundamental_score', 50.0),
                'news_score': news_score,
                'combined_score': combined_score,
                'rsi': technical_data.get('rsi', 50),
                'macd': technical_data.get('macd', 0),
                'sma_20': technical_data.get('sma_20', 0),
                'sma_50': technical_data.get('sma_50', 0),
                'sma_200': technical_data.get('sma_200', 0),
                'model_type': 'LSTM + LightGBM Hybrid',
                'r2_lstm': training_results.get('r2_lstm', 0),
                'r2_lgbm': training_results.get('r2_lgbm', 0),
                'r2_ensemble': training_results.get('r2', 0),
            }

            try:
                llm_explanation = llm_service.generate_analysis(ticker, llm_data)
            except:
                llm_explanation = f"Hybrid LSTM+LightGBM analysis for {ticker}"

            # Extract OHLCV data
            open_price = self._safe_extract_value(historical_data['Open'].iloc[-1])
            high_price = self._safe_extract_value(historical_data['High'].iloc[-1])
            low_price = self._safe_extract_value(historical_data['Low'].iloc[-1])
            volume = self._safe_extract_value(historical_data['Volume'].iloc[-1])

            # Store stock detail
            stock_item = StockDetail(
                ticker=ticker,
                date=datetime.now(),
                close=current_price,
                open=open_price,
                high=high_price,
                low=low_price,
                volume=int(volume),
                rsi=technical_data.get('rsi', 0),
                macd=technical_data.get('macd', 0),
                macd_signal=technical_data.get('macd_signal', 0),
                macd_diff=technical_data.get('macd_diff', 0),
                sma_20=technical_data.get('sma_20', 0),
                sma_50=technical_data.get('sma_50', 0),
                sma_200=technical_data.get('sma_200', 0),
                ema_12=technical_data.get('ema_12', 0),
                ema_26=technical_data.get('ema_26', 0),
                bollinger_high=technical_data.get('bollinger_high', 0),
                bollinger_low=technical_data.get('bollinger_low', 0),
                bollinger_mid=technical_data.get('bollinger_mid', 0),
                atr=technical_data.get('atr', 0),
                adx=technical_data.get('adx', 0),
                cci=technical_data.get('cci', 0),
                stoch_k=technical_data.get('stoch_k', 0),
                stoch_d=technical_data.get('stoch_d', 0),
                williams_r=technical_data.get('williams_r', 0),
                mfi=technical_data.get('mfi', 0),
                obv=technical_data.get('obv', 0),
                volatility_10d=technical_data.get('volatility_10d', 0),
                volatility_30d=technical_data.get('volatility_30d', 0),
                technical_score=technical_data.get('technical_score', 50.0),
                fundamental_score=financial_data.get('fundamental_score', 50.0),
                news_score=news_score,
                combined_score=combined_score,
                model_confidence=model_confidence,
                llm_explanation=llm_explanation,
                is_active=1,
                updated_at=datetime.now()
            )
            db.add(stock_item)

            # Store each daily prediction
            for pred in daily_predictions:
                history = PredictionHistory(
                    ticker=ticker,
                    prediction_date=datetime.now(),
                    target_date=pred['date'],
                    horizon=f"day_{pred['day']}",
                    predicted_price=float(pred['price']),
                    model_version=self.model_version,
                    confidence=float(model_confidence)
                )
                db.add(history)

            db.commit()
            logger.info(
                f"✓ Successfully updated {ticker} with HYBRID model\n"
                f"  LSTM R²: {training_results.get('r2_lstm', 0):.4f}\n"
                f"  LightGBM R²: {training_results.get('r2_lgbm', 0):.4f}\n"
                f"  Ensemble R²: {training_results.get('r2', 0):.4f}\n"
                f"  Confidence: {model_confidence:.1f}%\n"
                f"{'='*80}\n"
            )

            return {
                "ticker": ticker,
                "current_price": current_price,
                "daily_predictions": daily_predictions,
                "metrics": training_results,
                "scores": {
                    "technical": technical_data.get('technical_score', 50.0),
                    "fundamental": financial_data.get('fundamental_score', 50.0),
                    "news": news_score,
                    "combined": combined_score,
                },
                "confidence": overall_confidence,
                "model_type": "LSTM + LightGBM Hybrid"
            }

        except Exception as e:
            db.rollback()
            logger.error(f"Error for {ticker}: {e}", exc_info=True)
            raise
        finally:
            db.close()


# Initialize hybrid predictor
stock_service = HybridStockPredictor()

def update_stock_detail(tickers: list = None):
    """
    Update stock details using the hybrid LSTM + LightGBM model
    """
    tickers = tickers or config.DEFAULT_TICKERS
    results = []
    failed = []
    
    logger.info(
        f"\n{'='*80}\n"
        f"STARTING HYBRID MODEL BATCH UPDATE (LSTM + LightGBM)\n"
        f"{'='*80}\n"
    )
    
    for i, ticker in enumerate(tickers, 1):
        try:
            logger.info(f"[{i}/{len(tickers)}] Processing {ticker} with hybrid model...")
            result = stock_service.update_stock_detail(ticker)
            if result:
                results.append(result)
                logger.info(
                    f"  ✓ {ticker}: Confidence {result['confidence']:.1f}%, "
                    f"R² {result['metrics']['r2']:.4f}"
                )
            else:
                failed.append(ticker)
        except Exception as e:
            logger.error(f"[{i}/{len(tickers)}] {ticker} failed: {e}")
            failed.append(ticker)
    
    logger.info(
        f"\n{'='*80}\n"
        f"BATCH COMPLETE\n"
        f"Success: {len(results)}\n"
        f"Failed: {len(failed)}\n"
        f"{'='*80}\n"
    )
    
    if results:
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_r2 = np.mean([r['metrics']['r2'] for r in results])
        logger.info(
            f"Average Confidence: {avg_confidence:.1f}%\n"
            f"Average R²: {avg_r2:.4f}\n"
        )
    
    return results