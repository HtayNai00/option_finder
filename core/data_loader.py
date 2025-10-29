"""
Data loading module for Options Finder.
Handles loading options data from parquet files and API sources.
"""

import os
import time
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import streamlit as st
from config import LATEST_PARQUET, CACHE_TTL


@st.cache_data(ttl=0)  # No caching - always refresh
def load_latest_parquet(path: str = LATEST_PARQUET) -> pd.DataFrame:
    """
    Load the latest options data from parquet file.
    Always refreshes to get the most current data and filters out expired options.
    
    Args:
        path: Path to the parquet file
        
    Returns:
        DataFrame with options data, empty if file not found
    """
    if not os.path.exists(path):
        st.warning(f"Data file not found: {path}")
        return pd.DataFrame()
    
    try:
        # Check file modification time
        file_time = os.path.getmtime(path)
        current_time = time.time()
        age_hours = (current_time - file_time) / 3600
        
        # Load the data
        df = pd.read_parquet(path)
        # Replace infinite values with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Filter out expired options
        df = filter_expired_options(df)
        
        # Show data freshness info
        if age_hours > 24:
            st.warning(f"⚠️ Data is {age_hours:.1f} hours old. Consider updating.")
        elif age_hours > 1:
            st.info(f"ℹ️ Data is {age_hours:.1f} hours old.")
        else:
            st.success(f"✅ Data is fresh ({age_hours*60:.0f} minutes old)")
        
        return df
    except Exception as e:
        st.error(f"Error loading parquet file: {e}")
        return pd.DataFrame()


def filter_expired_options(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out expired options based on current date.
    
    Args:
        df: Options DataFrame
        
    Returns:
        DataFrame with only active (non-expired) options
    """
    if df.empty:
        return df
    
    try:
        from datetime import datetime, date
        
        # Get current date
        current_date = date.today()
        
        # Convert expiry_date to date objects for comparison
        df_copy = df.copy()
        df_copy['expiry_date_parsed'] = pd.to_datetime(df_copy['expiry_date']).dt.date
        
        # Filter out expired options (expiry_date < current_date)
        active_df = df_copy[df_copy['expiry_date_parsed'] >= current_date].copy()
        
        # Drop the temporary column
        active_df = active_df.drop('expiry_date_parsed', axis=1)
        
        # Show filtering results
        total_contracts = len(df)
        active_contracts = len(active_df)
        expired_contracts = total_contracts - active_contracts
        
        if expired_contracts > 0:
            st.info(f"🗑️ Filtered out {expired_contracts} expired contracts. Showing {active_contracts} active contracts.")
        
        return active_df
        
    except Exception as e:
        st.warning(f"Could not filter expired options: {e}")
        return df


def load_yahoo_finance_data(ticker: str, expiry_dates: list = None) -> pd.DataFrame:
    """
    Load options data from Yahoo Finance API.
    
    Args:
        ticker: Stock ticker symbol
        expiry_dates: List of expiry dates to fetch
        
    Returns:
        DataFrame with options data
    """
    # Placeholder for Yahoo Finance integration
    # This would use yfinance or similar library
    return pd.DataFrame()


def load_polygon_data(ticker: str, expiry_dates: list = None) -> pd.DataFrame:
    """
    Load options data from Polygon.io API.
    
    Args:
        ticker: Stock ticker symbol
        expiry_dates: List of expiry dates to fetch
        
    Returns:
        DataFrame with options data
    """
    # Placeholder for Polygon.io integration
    return pd.DataFrame()


def get_available_tickers(df: pd.DataFrame) -> list:
    """
    Get list of available tickers from the data.
    
    Args:
        df: Options DataFrame
        
    Returns:
        List of unique ticker symbols
    """
    if df.empty:
        return []
    return sorted(df["ticker"].dropna().unique().tolist())


def get_available_expiries(df: pd.DataFrame, tickers: list = None, option_type: str = None) -> list:
    """
    Get list of available expiry dates from the data.
    
    Args:
        df: Options DataFrame
        tickers: Filter by specific tickers
        option_type: Filter by option type (call/put)
        
    Returns:
        List of unique expiry dates
    """
    if df.empty:
        return []
    
    filtered_df = df.copy()
    if tickers:
        filtered_df = filtered_df[filtered_df["ticker"].isin(tickers)]
    if option_type:
        filtered_df = filtered_df[filtered_df["type"] == option_type]
    
    return sorted(filtered_df["expiry_date"].dropna().unique().tolist())


def validate_data_columns(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame has required columns.
    
    Args:
        df: Options DataFrame
        
    Returns:
        True if all required columns are present
    """
    required_columns = [
        "ticker", "type", "expiry_date", "days_to_expiry", "strike", 
        "spot", "bid", "ask", "premium_mid", "impliedVolatility", 
        "openInterest", "volume", "asof_ny", "asof_utc"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"Missing required columns: {missing_columns}")
        return False
    
    return True


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics about the loaded data.
    
    Args:
        df: Options DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {"total_contracts": 0, "tickers": 0, "expiries": 0}
    
    return {
        "total_contracts": len(df),
        "tickers": df["ticker"].nunique(),
        "expiries": df["expiry_date"].nunique(),
        "date_range": {
            "earliest": df["expiry_date"].min(),
            "latest": df["expiry_date"].max()
        },
        "option_types": df["type"].value_counts().to_dict()
    }


def refresh_data_from_api():
    """
    Refresh options data from API sources.
    This would integrate with your data ingestion pipeline.
    """
    try:
        # This is where you would call your data ingestion script
        # For now, we'll just show a message
        st.info("🔄 Refreshing data from API...")
        
        # You can integrate with your existing data pipeline here
        # For example, call the ingest_options.py script
        import subprocess
        import sys
        
        # Run the data ingestion script
        result = subprocess.run([
            sys.executable, "note/ingest_options.py"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # After refreshing, also clean up expired data
            cleanup_expired_data()
            st.success("✅ Data refreshed and expired contracts removed!")
            st.rerun()  # Reload the page with new data
        else:
            st.error(f"❌ Error refreshing data: {result.stderr}")
            
    except Exception as e:
        st.error(f"❌ Error refreshing data: {e}")


def cleanup_expired_data(path: str = LATEST_PARQUET):
    """
    Remove expired options from the data file permanently.
    
    Args:
        path: Path to the parquet file
    """
    try:
        if not os.path.exists(path):
            return
        
        # Load the data
        df = pd.read_parquet(path)
        
        # Filter out expired options
        cleaned_df = filter_expired_options_silent(df)
        
        # Save the cleaned data back to the file
        if len(cleaned_df) < len(df):
            cleaned_df.to_parquet(path, index=False)
            print(f"🗑️ Cleaned up expired data. Removed {len(df) - len(cleaned_df)} expired contracts.")
        
    except Exception as e:
        print(f"❌ Error cleaning up expired data: {e}")


def filter_expired_options_silent(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out expired options without showing messages (for cleanup operations).
    
    Args:
        df: Options DataFrame
        
    Returns:
        DataFrame with only active (non-expired) options
    """
    if df.empty:
        return df
    
    try:
        from datetime import datetime, date
        
        # Get current date
        current_date = date.today()
        
        # Convert expiry_date to date objects for comparison
        df_copy = df.copy()
        df_copy['expiry_date_parsed'] = pd.to_datetime(df_copy['expiry_date']).dt.date
        
        # Filter out expired options (expiry_date < current_date)
        active_df = df_copy[df_copy['expiry_date_parsed'] >= current_date].copy()
        
        # Drop the temporary column
        active_df = active_df.drop('expiry_date_parsed', axis=1)
        
        return active_df
        
    except Exception as e:
        print(f"Could not filter expired options: {e}")
        return df


def check_data_freshness(path: str = LATEST_PARQUET) -> Dict[str, Any]:
    """
    Check how fresh the data is and provide refresh options.
    
    Args:
        path: Path to the data file
        
    Returns:
        Dictionary with freshness info
    """
    if not os.path.exists(path):
        return {"exists": False, "age_hours": None, "needs_refresh": True}
    
    try:
        file_time = os.path.getmtime(path)
        current_time = time.time()
        age_hours = (current_time - file_time) / 3600
        
        return {
            "exists": True,
            "age_hours": age_hours,
            "needs_refresh": age_hours > 1,  # Refresh if older than 1 hour
            "file_time": file_time,
            "current_time": current_time
        }
    except Exception as e:
        return {"exists": False, "age_hours": None, "needs_refresh": True, "error": str(e)}
