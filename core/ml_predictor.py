"""
Machine Learning Model for Options Profit Prediction
Predicts the most profitable option chains using various features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st
from datetime import datetime, date
import joblib
import os
from pathlib import Path

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Import our analytics functions
from .analytics import bs_greeks_one, compute_profit_metrics, calculate_breakeven_price


class OptionsMLPredictor:
    """
    Machine Learning model to predict option profitability and rank options.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.model_path = Path("models/options_predictor.pkl")
        self.scaler_path = Path("models/scaler.pkl")
        self.encoders_path = Path("models/encoders.pkl")
        
        # Create models directory
        self.model_path.parent.mkdir(exist_ok=True)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive features for ML model.
        
        Args:
            df: Options DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        if df.empty:
            return df
        
        df_features = df.copy()
        
        # Basic features
        df_features['moneyness'] = df_features['spot'] / df_features['strike']
        df_features['time_to_expiry'] = df_features['days_to_expiry'] / 365.0
        df_features['bid_ask_spread'] = df_features['ask'] - df_features['bid']
        df_features['bid_ask_spread_pct'] = df_features['bid_ask_spread'] / df_features['premium_mid']
        df_features['volume_oi_ratio'] = df_features['volume'] / (df_features['openInterest'] + 1)
        
        # Greeks features
        greeks_data = []
        for _, row in df_features.iterrows():
            try:
                greeks = bs_greeks_one(
                    spot=row['spot'],
                    strike=row['strike'],
                    T=row['time_to_expiry'],
                    sigma=row.get('impliedVolatility', 0.2),
                    r=0.04,
                    option_type=row['type']
                )
                greeks_data.append(greeks)
            except:
                greeks_data.append({
                    'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0
                })
        
        greeks_df = pd.DataFrame(greeks_data)
        df_features = pd.concat([df_features, greeks_df], axis=1)
        
        # Profit metrics - calculate both up and down scenarios
        try:
            # Calculate profit ratio for +10% move (calls benefit, puts lose)
            df_features['target_price_up'] = df_features['spot'] * 1.10
            df_features['target_price_down'] = df_features['spot'] * 0.90
            
            # For calls: profit = max(0, target_price - strike - premium)
            call_mask = df_features['type'] == 'call'
            df_features.loc[call_mask, 'profit_up'] = (
                (df_features.loc[call_mask, 'target_price_up'] - 
                 df_features.loc[call_mask, 'strike'] - 
                 df_features.loc[call_mask, 'premium_mid']).clip(lower=0)
            )
            df_features.loc[call_mask, 'profit_down'] = (
                (df_features.loc[call_mask, 'target_price_down'] - 
                 df_features.loc[call_mask, 'strike'] - 
                 df_features.loc[call_mask, 'premium_mid']).clip(lower=0)
            )
            
            # For puts: profit = max(0, strike - target_price - premium)
            put_mask = df_features['type'] == 'put'
            df_features.loc[put_mask, 'profit_up'] = (
                (df_features.loc[put_mask, 'strike'] - 
                 df_features.loc[put_mask, 'target_price_up'] - 
                 df_features.loc[put_mask, 'premium_mid']).clip(lower=0)
            )
            df_features.loc[put_mask, 'profit_down'] = (
                (df_features.loc[put_mask, 'strike'] - 
                 df_features.loc[put_mask, 'target_price_down'] - 
                 df_features.loc[put_mask, 'premium_mid']).clip(lower=0)
            )
            
            # Calculate profit ratios
            df_features['profit_ratio_up'] = np.where(
                df_features['premium_mid'] > 0,
                df_features['profit_up'] / df_features['premium_mid'],
                0
            )
            df_features['profit_ratio_down'] = np.where(
                df_features['premium_mid'] > 0,
                df_features['profit_down'] / df_features['premium_mid'],
                0
            )
            
            # Calculate breakeven prices
            df_features['breakeven_price'] = np.where(
                df_features['type'] == 'call',
                df_features['strike'] + df_features['premium_mid'],
                df_features['strike'] - df_features['premium_mid']
            )
            
            # Calculate max profit and loss
            df_features['max_profit'] = df_features[['profit_ratio_up', 'profit_ratio_down']].max(axis=1) * df_features['premium_mid']
            df_features['max_loss'] = -df_features['premium_mid']
            
        except Exception as e:
            # Fallback if profit calculation fails
            df_features['profit_ratio_up'] = 0
            df_features['profit_ratio_down'] = 0
            df_features['breakeven_price'] = 0
            df_features['max_profit'] = 0
            df_features['max_loss'] = 0
        
        # Advanced features
        df_features['iv_rank'] = df_features.groupby('ticker')['impliedVolatility'].rank(pct=True)
        df_features['volume_rank'] = df_features.groupby('ticker')['volume'].rank(pct=True)
        df_features['oi_rank'] = df_features.groupby('ticker')['openInterest'].rank(pct=True)
        
        # Time-based features
        df_features['is_weekly'] = df_features['days_to_expiry'] <= 7
        df_features['is_monthly'] = (df_features['days_to_expiry'] > 7) & (df_features['days_to_expiry'] <= 30)
        df_features['is_quarterly'] = df_features['days_to_expiry'] > 30
        
        # Volatility features
        df_features['iv_vs_historical'] = df_features['impliedVolatility'] / 0.2  # Assume 20% historical
        df_features['high_iv'] = df_features['impliedVolatility'] > df_features['impliedVolatility'].quantile(0.8)
        
        # Liquidity features
        df_features['high_liquidity'] = (df_features['volume'] > df_features['volume'].quantile(0.7)) & \
                                       (df_features['openInterest'] > df_features['openInterest'].quantile(0.7))
        
        # Risk features
        df_features['risk_reward_ratio'] = df_features['max_profit'] / (abs(df_features['max_loss']) + 1)
        df_features['premium_per_day'] = df_features['premium_mid'] / (df_features['days_to_expiry'] + 1)
        
        return df_features
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data with features and target variable.
        
        Args:
            df: Options DataFrame with features
            
        Returns:
            Tuple of (features, target)
        """
        # Create target variable - combined profit score
        df['target_score'] = (
            df['profit_ratio_up'] * 0.4 +  # 40% weight on upside profit
            df['profit_ratio_down'] * 0.3 +  # 30% weight on downside profit
            df['risk_reward_ratio'] * 0.2 +  # 20% weight on risk/reward
            (df['volume_rank'] + df['oi_rank']) * 0.1  # 10% weight on liquidity
        )
        
        # Select features for training
        feature_columns = [
            'moneyness', 'time_to_expiry', 'bid_ask_spread_pct', 'volume_oi_ratio',
            'delta', 'gamma', 'theta', 'vega', 'rho',
            'iv_rank', 'volume_rank', 'oi_rank',
            'is_weekly', 'is_monthly', 'is_quarterly',
            'iv_vs_historical', 'high_iv', 'high_liquidity',
            'risk_reward_ratio', 'premium_per_day',
            'impliedVolatility', 'volume', 'openInterest'
        ]
        
        # Add option type as encoded feature
        le_type = LabelEncoder()
        df['type_encoded'] = le_type.fit_transform(df['type'])
        feature_columns.append('type_encoded')
        
        # Store encoders
        self.label_encoders['type'] = le_type
        
        # Prepare features and target
        X = df[feature_columns].fillna(0)
        y = df['target_score']
        
        self.feature_columns = feature_columns
        
        return X, y
    
    def train_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Train the ML model on options data.
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dictionary with training metrics
        """
        try:
            # Create features
            df_features = self.create_features(df)
            
            # Prepare training data
            X, y = self.prepare_training_data(df_features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train multiple models and select best
            models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression()
            }
            
            best_model = None
            best_score = -np.inf
            best_name = ""
            
            for name, model in models.items():
                # Train model
                if name == 'XGBoost':
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                
                # Evaluate
                score = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = name
            
            self.model = best_model
            
            # Save model and components
            self.save_model()
            
            # Calculate final metrics
            final_mse = mean_squared_error(y_test, y_pred)
            final_r2 = r2_score(y_test, y_pred)
            
            return {
                'best_model': best_name,
                'r2_score': final_r2,
                'mse': final_mse,
                'n_features': len(self.feature_columns),
                'n_samples': len(X_train)
            }
            
        except Exception as e:
            st.error(f"Error training model: {e}")
            return {}
    
    def predict_profitability(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict profitability scores for options.
        
        Args:
            df: Options DataFrame
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            # Load model if not trained
            if not self.load_model():
                st.warning("No trained model available. Using simple scoring.")
                return self._simple_scoring(df)
        
        try:
            # Create features
            df_features = self.create_features(df)
            
            # Prepare features
            if 'type_encoded' not in df_features.columns:
                le_type = LabelEncoder()
                df_features['type_encoded'] = le_type.fit_transform(df_features['type'])
            
            X = df_features[self.feature_columns].fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(X_scaled)
            else:
                predictions = self.model.predict(X)
            
            # Add predictions to dataframe
            df_features['ml_profit_score'] = predictions
            df_features['ml_rank'] = df_features['ml_profit_score'].rank(ascending=False, method='dense')
            
            return df_features
            
        except Exception as e:
            st.warning(f"ML prediction failed: {e}. Using simple scoring.")
            return self._simple_scoring(df)
    
    def _simple_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simple scoring method as fallback.
        
        Args:
            df: Options DataFrame
            
        Returns:
            DataFrame with simple scores
        """
        df_scored = df.copy()
        
        # Create simple profit metrics if they don't exist
        if 'profit_ratio_up' not in df_scored.columns:
            # Use existing profit_ratio_at_target as base
            base_profit = df_scored.get('profit_ratio_at_target', 0)
            df_scored['profit_ratio_up'] = base_profit * 0.8  # Slightly lower for up scenario
            df_scored['profit_ratio_down'] = base_profit * 0.6  # Lower for down scenario
        
        # Simple profit score using available data
        df_scored['ml_profit_score'] = (
            df_scored.get('profit_ratio_up', 0) * 0.4 +
            df_scored.get('profit_ratio_down', 0) * 0.3 +
            df_scored.get('profit_ratio_at_target', 0) * 0.2 +
            (df_scored['volume'] / (df_scored['volume'].max() + 1)) * 0.1
        )
        
        df_scored['ml_rank'] = df_scored['ml_profit_score'].rank(ascending=False, method='dense')
        
        return df_scored
    
    def get_top_predictions(self, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Get top N predicted profitable options.
        
        Args:
            df: Options DataFrame
            top_n: Number of top predictions to return
            
        Returns:
            DataFrame with top predictions
        """
        # Get predictions
        df_predicted = self.predict_profitability(df)
        
        # Sort by ML score and get top N
        top_options = df_predicted.nlargest(top_n, 'ml_profit_score')
        
        # Add rank
        top_options = top_options.reset_index(drop=True)
        top_options['rank'] = range(1, len(top_options) + 1)
        
        return top_options
    
    def save_model(self):
        """Save trained model and components."""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.label_encoders, self.encoders_path)
        except Exception as e:
            st.warning(f"Could not save model: {e}")
    
    def load_model(self) -> bool:
        """Load trained model and components."""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.label_encoders = joblib.load(self.encoders_path)
                
                # Load feature columns if available
                feature_file = self.model_path.parent / "feature_columns.txt"
                if feature_file.exists():
                    with open(feature_file, "r") as f:
                        self.feature_columns = f.read().strip().split("\n")
                
                return True
        except Exception as e:
            st.warning(f"Could not load model: {e}")
        return False


def create_ml_predictor() -> OptionsMLPredictor:
    """Create and return ML predictor instance."""
    return OptionsMLPredictor()
