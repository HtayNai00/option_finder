"""
Train Machine Learning Model for Options Profit Prediction
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ml_predictor import OptionsMLPredictor
from core.data_loader import load_latest_parquet
from core.analytics import bs_greeks_one, compute_profit_metrics


def train_ml_model():
    """Train the ML model on current options data."""
    print("Training Options ML Model...")
    print("=" * 50)
    
    try:
        # Load data
        print("Loading options data...")
        df = load_latest_parquet()
        
        if df.empty:
            print("No data available for training")
            return False
        
        print(f"Loaded {len(df)} options contracts")
        
        # Create ML predictor
        predictor = OptionsMLPredictor()
        
        # Train model
        print("Training ML model...")
        metrics = predictor.train_model(df)
        
        if metrics:
            print("Model training completed!")
            print(f"Best Model: {metrics.get('best_model', 'Unknown')}")
            print(f"R² Score: {metrics.get('r2_score', 0):.4f}")
            print(f"MSE: {metrics.get('mse', 0):.4f}")
            print(f"Features: {metrics.get('n_features', 0)}")
            print(f"Samples: {metrics.get('n_samples', 0)}")
            
            # Test predictions
            print("\nTesting predictions...")
            top_predictions = predictor.get_top_predictions(df, top_n=5)
            
            if not top_predictions.empty:
                print("Top 5 Predicted Options:")
                for _, row in top_predictions.iterrows():
                    print(f"  {row['rank']}. {row['ticker']} {row['type'].upper()} ${row['strike']:.2f} "
                          f"(Score: {row['ml_profit_score']:.4f})")
            
            return True
        else:
            print("Model training failed")
            return False
            
    except Exception as e:
        print(f"Error training model: {e}")
        return False


def validate_ml_model():
    """Validate the trained ML model."""
    print("Validating ML Model...")
    print("=" * 50)
    
    try:
        # Load data
        df = load_latest_parquet()
        if df.empty:
            print("No data available for validation")
            return False
        
        # Create predictor and load model
        predictor = OptionsMLPredictor()
        if not predictor.load_model():
            print("No trained model found")
            return False
        
        # Get predictions
        top_predictions = predictor.get_top_predictions(df, top_n=10)
        
        if not top_predictions.empty:
            print("Model validation successful!")
            print(f"Generated predictions for {len(top_predictions)} top options")
            
            # Show top 3 predictions
            print("\nTop 3 ML Predictions:")
            for _, row in top_predictions.head(3).iterrows():
                print(f"  {row['rank']}. {row['ticker']} {row['type'].upper()} ${row['strike']:.2f}")
                print(f"     Expiry: {row['expiry_date']} | Score: {row['ml_profit_score']:.4f}")
                print(f"     Profit Ratio Up: {row.get('profit_ratio_up', 0):.2f}")
                print(f"     Volume: {row['volume']:,} | OI: {row['openInterest']:,}")
                print()
            
            return True
        else:
            print("No predictions generated")
            return False
            
    except Exception as e:
        print(f"Error validating model: {e}")
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        success = validate_ml_model()
    else:
        success = train_ml_model()
    
    if success:
        print("\nML Model operations completed successfully!")
        print("You can now use ML predictions in the web interface")
    else:
        print("\nML Model operations failed. Check the errors above.")


if __name__ == "__main__":
    main()
