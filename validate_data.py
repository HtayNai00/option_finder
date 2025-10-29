#!/usr/bin/env python3
"""
Data validation script for Options Finder.
Checks data integrity and provides summary statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from core.data_loader import load_latest_parquet, validate_data_columns, get_data_summary
from core.analytics import compute_profit_metrics, add_implied_itm_probability
from core.filters import basic_filters
from config import LATEST_PARQUET


def validate_data_file(file_path: str = LATEST_PARQUET):
    """Validate the main data file."""
    print(f"Validating data file: {file_path}")
    print("=" * 60)
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"File not found: {file_path}")
        return False
    
    # Load data
    try:
        df = load_latest_parquet(file_path)
        print(f"File loaded successfully")
        print(f"Records: {len(df):,}")
    except Exception as e:
        print(f"Error loading file: {e}")
        return False
    
    if df.empty:
        print("File is empty")
        return False
    
    # Validate columns
    print("\nColumn validation:")
    if validate_data_columns(df):
        print("All required columns present")
    else:
        print("Some required columns missing")
    
    # Data summary
    print("\nData Summary:")
    summary = get_data_summary(df)
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    # Check for missing values
    print("\nMissing values check:")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() == 0:
        print("No missing values")
    else:
        print("Missing values found:")
        for col, count in missing_counts.items():
            if count > 0:
                print(f"   {col}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Check data types
    print("\nData types:")
    for col, dtype in df.dtypes.items():
        print(f"   {col}: {dtype}")
    
    # Sample data
    print("\nSample data (first 3 rows):")
    print(df.head(3).to_string())
    
    return True


def test_analytics_pipeline(df: pd.DataFrame):
    """Test the analytics pipeline."""
    print("\nTesting analytics pipeline:")
    print("=" * 60)
    
    if df.empty:
        print("No data to test")
        return False
    
    try:
        # Test profit metrics
        print("Testing profit metrics...")
        df_calls = df[df['type'] == 'call'].copy()
        df_puts = df[df['type'] == 'put'].copy()
        
        if not df_calls.empty:
            calls_with_metrics = compute_profit_metrics(df_calls, 'call')
            print(f"Call profit metrics: {len(calls_with_metrics)} records")
        
        if not df_puts.empty:
            puts_with_metrics = compute_profit_metrics(df_puts, 'put')
            print(f"Put profit metrics: {len(puts_with_metrics)} records")
        
        # Test ITM probability
        print("Testing ITM probability...")
        if not df_calls.empty:
            calls_with_itm = add_implied_itm_probability(calls_with_metrics)
            print(f"Call ITM probability: {len(calls_with_itm)} records")
        
        if not df_puts.empty:
            puts_with_itm = add_implied_itm_probability(puts_with_metrics)
            print(f"Put ITM probability: {len(puts_with_itm)} records")
        
        # Test filtering
        print("Testing filtering...")
        tickers = df['ticker'].unique()[:2]  # Test with first 2 tickers
        filtered_df = basic_filters(df, tickers.tolist(), 'put')
        print(f"Filtering: {len(filtered_df)} records after filtering")
        
        print("Analytics pipeline test completed successfully")
        return True
        
    except Exception as e:
        print(f"Analytics pipeline test failed: {e}")
        return False


def check_data_quality(df: pd.DataFrame):
    """Check data quality issues."""
    print("\nData quality check:")
    print("=" * 60)
    
    issues = []
    
    # Check for negative values where they shouldn't be
    if 'premium_mid' in df.columns:
        negative_premiums = (df['premium_mid'] < 0).sum()
        if negative_premiums > 0:
            issues.append(f"Negative premiums: {negative_premiums}")
    
    if 'strike' in df.columns:
        negative_strikes = (df['strike'] <= 0).sum()
        if negative_strikes > 0:
            issues.append(f"Non-positive strikes: {negative_strikes}")
    
    if 'spot' in df.columns:
        negative_spots = (df['spot'] <= 0).sum()
        if negative_spots > 0:
            issues.append(f"Non-positive spot prices: {negative_spots}")
    
    if 'days_to_expiry' in df.columns:
        negative_dte = (df['days_to_expiry'] < 0).sum()
        if negative_dte > 0:
            issues.append(f"Negative days to expiry: {negative_dte}")
    
    # Check for extreme values
    if 'impliedVolatility' in df.columns:
        extreme_iv = ((df['impliedVolatility'] > 5) | (df['impliedVolatility'] < 0)).sum()
        if extreme_iv > 0:
            issues.append(f"Extreme implied volatility values: {extreme_iv}")
    
    if issues:
        print("Data quality issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("No data quality issues found")
    
    return len(issues) == 0


def main():
    """Main validation function."""
    print("Options Finder Data Validation")
    print("=" * 60)
    
    # Validate main data file
    if not validate_data_file():
        print("\nData validation failed")
        return
    
    # Load data for additional tests
    df = load_latest_parquet()
    
    # Test analytics pipeline
    test_analytics_pipeline(df)
    
    # Check data quality
    check_data_quality(df)
    
    print("\nData validation completed")
    print("\nYou can now run the application:")
    print("   python run_app.py streamlit")
    print("   python run_app.py api")


if __name__ == "__main__":
    main()
