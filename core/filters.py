"""
Filtering module for Options Finder.
Contains functions to filter options data by various criteria.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from config import (
    LOWER_MONEYNESS, UPPER_MONEYNESS, MIN_PREMIUM, 
    MIN_OI, MIN_VOL, DEFAULT_MAX_DTE
)


def basic_filters(df: pd.DataFrame, tickers: List[str], opt_type: str) -> pd.DataFrame:
    """
    Apply basic filters to options data.
    
    Args:
        df: Options DataFrame
        tickers: List of ticker symbols to include
        opt_type: Option type ('call' or 'put')
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Filter by option type and tickers
    df = df[df["type"] == opt_type]
    df = df[df["ticker"].isin(tickers)]
    
    # Remove rows with missing essential data
    essential_cols = ["spot", "strike", "premium_mid", "openInterest", "volume", "days_to_expiry"]
    df = df.dropna(subset=essential_cols)
    
    # Moneyness filter (strike within range of spot)
    lb = df["spot"] * LOWER_MONEYNESS
    ub = df["spot"] * UPPER_MONEYNESS
    df = df[(df["strike"] >= lb) & (df["strike"] <= ub)]
    
    # Liquidity filters
    df = df[
        (df["premium_mid"] >= MIN_PREMIUM) & 
        (df["openInterest"] >= MIN_OI) & 
        (df["volume"] >= MIN_VOL)
    ]
    
    return df


def apply_expiry_filter(df: pd.DataFrame, mode: str, max_dte: Optional[int] = None, 
                       exact_expiry: Optional[str] = None) -> pd.DataFrame:
    """
    Apply expiry date filters to options data.
    
    Args:
        df: Options DataFrame
        mode: Filter mode ('Range' or 'Exact')
        max_dte: Maximum days to expiry for range mode
        exact_expiry: Exact expiry date for exact mode
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    if mode == "Range":
        if max_dte is None:
            max_dte = DEFAULT_MAX_DTE
        df = df[df["days_to_expiry"].between(1, int(max_dte))]
    elif mode == "Exact" and exact_expiry:
        df = df[df["expiry_date"] == exact_expiry]
    
    return df


def filter_by_strike_range(df: pd.DataFrame, min_strike: Optional[float] = None, 
                          max_strike: Optional[float] = None) -> pd.DataFrame:
    """
    Filter options by strike price range.
    
    Args:
        df: Options DataFrame
        min_strike: Minimum strike price
        max_strike: Maximum strike price
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    if min_strike is not None:
        df = df[df["strike"] >= min_strike]
    if max_strike is not None:
        df = df[df["strike"] <= max_strike]
    
    return df


def filter_by_moneyness(df: pd.DataFrame, min_moneyness: float = 0.8, 
                       max_moneyness: float = 1.2) -> pd.DataFrame:
    """
    Filter options by moneyness (strike/spot ratio).
    
    Args:
        df: Options DataFrame
        min_moneyness: Minimum moneyness ratio
        max_moneyness: Maximum moneyness ratio
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    moneyness = df["strike"] / df["spot"]
    df = df[(moneyness >= min_moneyness) & (moneyness <= max_moneyness)]
    
    return df


def filter_by_liquidity(df: pd.DataFrame, min_premium: Optional[float] = None,
                       min_oi: Optional[int] = None, min_volume: Optional[int] = None) -> pd.DataFrame:
    """
    Filter options by liquidity criteria.
    
    Args:
        df: Options DataFrame
        min_premium: Minimum premium
        min_oi: Minimum open interest
        min_volume: Minimum volume
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    if min_premium is not None:
        df = df[df["premium_mid"] >= min_premium]
    if min_oi is not None:
        df = df[df["openInterest"] >= min_oi]
    if min_volume is not None:
        df = df[df["volume"] >= min_volume]
    
    return df


def filter_by_iv_range(df: pd.DataFrame, min_iv: Optional[float] = None,
                      max_iv: Optional[float] = None) -> pd.DataFrame:
    """
    Filter options by implied volatility range.
    
    Args:
        df: Options DataFrame
        min_iv: Minimum implied volatility
        max_iv: Maximum implied volatility
        
    Returns:
        Filtered DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    if "impliedVolatility" in df.columns:
        if min_iv is not None:
            df = df[df["impliedVolatility"] >= min_iv]
        if max_iv is not None:
            df = df[df["impliedVolatility"] <= max_iv]
    
    return df


def filter_by_profit_ratio(df: pd.DataFrame, min_profit_ratio: float = 0.0) -> pd.DataFrame:
    """
    Filter options by minimum profit ratio.
    
    Args:
        df: Options DataFrame
        min_profit_ratio: Minimum profit ratio at target
        
    Returns:
        Filtered DataFrame
    """
    if df.empty or "profit_ratio_at_target" not in df.columns:
        return df
    
    df = df.copy()
    df = df[df["profit_ratio_at_target"] >= min_profit_ratio]
    
    return df


def filter_by_itm_probability(df: pd.DataFrame, min_itm_prob: Optional[float] = None,
                             max_itm_prob: Optional[float] = None) -> pd.DataFrame:
    """
    Filter options by implied ITM probability range.
    
    Args:
        df: Options DataFrame
        min_itm_prob: Minimum ITM probability
        max_itm_prob: Maximum ITM probability
        
    Returns:
        Filtered DataFrame
    """
    if df.empty or "implied_itm_prob" not in df.columns:
        return df
    
    df = df.copy()
    
    if min_itm_prob is not None:
        df = df[df["implied_itm_prob"] >= min_itm_prob]
    if max_itm_prob is not None:
        df = df[df["implied_itm_prob"] <= max_itm_prob]
    
    return df


def get_filter_summary(df: pd.DataFrame) -> dict:
    """
    Get summary of current filters applied to the data.
    
    Args:
        df: Options DataFrame
        
    Returns:
        Dictionary with filter summary
    """
    if df.empty:
        return {"total_contracts": 0}
    
    return {
        "total_contracts": len(df),
        "tickers": df["ticker"].nunique(),
        "expiries": df["expiry_date"].nunique(),
        "option_types": df["type"].value_counts().to_dict(),
        "strike_range": {
            "min": df["strike"].min(),
            "max": df["strike"].max()
        },
        "premium_range": {
            "min": df["premium_mid"].min(),
            "max": df["premium_mid"].max()
        },
        "dte_range": {
            "min": df["days_to_expiry"].min(),
            "max": df["days_to_expiry"].max()
        }
    }

