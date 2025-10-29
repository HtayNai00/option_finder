"""
Analytics module for Options Finder.
Provides profit metrics, Greeks, ITM probability calculations, and other analytics.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Dict, Optional
from config import TARGET_DROP_PUT, TARGET_UP_CALL, RISK_FREE


def bs_greeks_one(S: float, K: float, T: float, sigma: float, r: float, opt_type: str = "put") -> Dict[str, float]:
    """
    Calculate Black-Scholes Greeks for a single option.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (in years)
        sigma: Implied volatility (annualized, as decimal, e.g., 0.20 for 20%)
        r: Risk-free rate (annualized, as decimal, e.g., 0.04 for 4%)
        opt_type: Option type ("call" or "put")
        
    Returns:
        Dictionary with Greeks: delta, gamma, theta, vega, rho
    """
    # Validate inputs
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan}
    
    try:
        # Ensure minimum values to avoid numerical issues
        T = max(T, 1/365.0)  # At least 1 day
        sigma = max(sigma, 0.01)  # At least 1% IV
        
        # Calculate d1 and d2
        sqrt_T = np.sqrt(T)
        log_SK = np.log(S / K)
        d1 = (log_SK + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        # Calculate PDF and CDF values
        pdf_d1 = norm.pdf(d1)
        if opt_type.lower() == "call":
            cdf_d1 = norm.cdf(d1)
            cdf_d2 = norm.cdf(d2)
            delta = cdf_d1
            theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) - 
                     r * K * np.exp(-r * T) * cdf_d2) / 365.0
            rho = K * T * np.exp(-r * T) * cdf_d2 / 100.0
        else:  # put
            cdf_neg_d1 = norm.cdf(-d1)
            cdf_neg_d2 = norm.cdf(-d2)
            delta = -cdf_neg_d1  # This is norm.cdf(-d1) but negated
            theta = (-S * pdf_d1 * sigma / (2 * sqrt_T) + 
                     r * K * np.exp(-r * T) * cdf_neg_d2) / 365.0
            rho = -K * T * np.exp(-r * T) * cdf_neg_d2 / 100.0
        
        # Gamma and Vega are the same for calls and puts
        gamma = pdf_d1 / (S * sigma * sqrt_T)
        vega = S * pdf_d1 * sqrt_T / 100.0  # Per 1% change in IV
        
        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta),
            "vega": float(vega),
            "rho": float(rho)
        }
    except Exception as e:
        # Return NaN on any error
        return {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan}


def compute_profit_metrics(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
    """
    Compute profit metrics for options.
    
    Args:
        df: Options DataFrame
        opt_type: Option type ("call" or "put")
        
    Returns:
        DataFrame with added profit metric columns
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Ensure required columns exist
    required_cols = ["spot", "strike", "premium_mid"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Set target move percentage based on option type
    if opt_type.lower() == "put":
        target_pct = TARGET_DROP_PUT
    else:  # call
        target_pct = TARGET_UP_CALL
    
    df["target_move_pct"] = target_pct
    
    # Calculate target price
    if opt_type.lower() == "put":
        df["target_price"] = df["spot"] * (1 - target_pct)
    else:  # call
        df["target_price"] = df["spot"] * (1 + target_pct)
    
    # Calculate profit at target price
    if opt_type.lower() == "put":
        # For puts: profit = max(0, strike - target_price - premium)
        profit_per_share = (df["strike"] - df["target_price"] - df["premium_mid"]).clip(lower=0.0)
    else:  # call
        # For calls: profit = max(0, target_price - strike - premium)
        profit_per_share = (df["target_price"] - df["strike"] - df["premium_mid"]).clip(lower=0.0)
    
    # Calculate profit ratio (profit / premium)
    df["profit_at_target"] = profit_per_share * 100.0  # Per contract (100 shares)
    df["profit_ratio_at_target"] = np.where(
        df["premium_mid"] > 0,
        profit_per_share / df["premium_mid"],
        0.0
    )
    
    # Replace infinities and NaNs
    df["profit_ratio_at_target"] = np.nan_to_num(
        df["profit_ratio_at_target"], 
        nan=0.0, posinf=0.0, neginf=0.0
    )
    
    return df


def add_implied_itm_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add implied ITM (in-the-money) probability using Black-Scholes.
    
    Args:
        df: Options DataFrame
        
    Returns:
        DataFrame with added 'implied_itm_prob' column
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Check if required columns exist
    required_cols = ["spot", "strike", "days_to_expiry", "impliedVolatility"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        # If missing, set to NaN
        df["implied_itm_prob"] = np.nan
        return df
    
    # Calculate time to expiration in years
    T = df["days_to_expiry"] / 365.0
    
    # Get implied volatility
    sigma = df["impliedVolatility"].fillna(0)
    
    # Calculate d2 (Black-Scholes)
    # d2 = (ln(S/K) + (r - 0.5*sigma^2)*T) / (sigma * sqrt(T))
    r = RISK_FREE
    
    # Handle edge cases
    valid_mask = (T > 0) & (sigma > 0) & (df["spot"] > 0) & (df["strike"] > 0)
    
    df["implied_itm_prob"] = np.nan
    
    if valid_mask.any():
        S = df.loc[valid_mask, "spot"]
        K = df.loc[valid_mask, "strike"]
        T_valid = T[valid_mask]
        sigma_valid = sigma[valid_mask]
        
        # Calculate d2
        d2 = (np.log(S / K) + (r - 0.5 * sigma_valid ** 2) * T_valid) / (sigma_valid * np.sqrt(T_valid))
        
        # ITM probability depends on option type
        opt_type = df.loc[valid_mask, "type"]
        
        # For calls: P(S > K) = N(d2)
        # For puts: P(S < K) = N(-d2)
        call_mask = valid_mask & (df["type"] == "call")
        put_mask = valid_mask & (df["type"] == "put")
        
        if call_mask.any():
            d2_calls = (np.log(df.loc[call_mask, "spot"] / df.loc[call_mask, "strike"]) + 
                       (r - 0.5 * df.loc[call_mask, "impliedVolatility"] ** 2) * 
                       (df.loc[call_mask, "days_to_expiry"] / 365.0)) / \
                      (df.loc[call_mask, "impliedVolatility"] * 
                       np.sqrt(df.loc[call_mask, "days_to_expiry"] / 365.0))
            df.loc[call_mask, "implied_itm_prob"] = norm.cdf(d2_calls)
        
        if put_mask.any():
            d2_puts = (np.log(df.loc[put_mask, "spot"] / df.loc[put_mask, "strike"]) + 
                      (r - 0.5 * df.loc[put_mask, "impliedVolatility"] ** 2) * 
                      (df.loc[put_mask, "days_to_expiry"] / 365.0)) / \
                     (df.loc[put_mask, "impliedVolatility"] * 
                      np.sqrt(df.loc[put_mask, "days_to_expiry"] / 365.0))
            df.loc[put_mask, "implied_itm_prob"] = norm.cdf(-d2_puts)
    
    # Fill NaN values with 0
    df["implied_itm_prob"] = df["implied_itm_prob"].fillna(0.0).clip(0.0, 1.0)
    
    return df


def add_greeks_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Black-Scholes Greeks to the dataframe.
    
    Args:
        df: Options DataFrame
        
    Returns:
        DataFrame with added Greek columns: delta, gamma, theta, vega, rho
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Initialize Greek columns
    for greek in ["delta", "gamma", "theta", "vega", "rho"]:
        if greek not in df.columns:
            df[greek] = np.nan
    
    # Required columns
    required_cols = ["spot", "strike", "days_to_expiry", "impliedVolatility", "type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return df
    
    r = RISK_FREE
    
    # Calculate Greeks for each row
    for idx in df.index:
        S = float(df.loc[idx, "spot"])
        K = float(df.loc[idx, "strike"])
        T = float(df.loc[idx, "days_to_expiry"]) / 365.0
        sigma = float(df.loc[idx, "impliedVolatility"])
        opt_type = str(df.loc[idx, "type"]).lower()
        
        greeks = bs_greeks_one(S, K, T, sigma, r, opt_type)
        
        for greek_name, greek_value in greeks.items():
            df.loc[idx, greek_name] = greek_value
    
    return df


def calculate_breakeven_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate breakeven price for options.
    
    For calls: breakeven = strike + premium
    For puts: breakeven = strike - premium
    
    Args:
        df: Options DataFrame
        
    Returns:
        DataFrame with added 'breakeven_price' and 'pct_to_breakeven' columns
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Check required columns
    if "premium_mid" not in df.columns or "strike" not in df.columns:
        df["breakeven_price"] = np.nan
        df["pct_to_breakeven"] = np.nan
        return df
    
    # Calculate breakeven price based on option type
    call_mask = df["type"] == "call"
    put_mask = df["type"] == "put"
    
    df["breakeven_price"] = np.nan
    
    if call_mask.any():
        df.loc[call_mask, "breakeven_price"] = (
            df.loc[call_mask, "strike"] + df.loc[call_mask, "premium_mid"]
        )
    
    if put_mask.any():
        df.loc[put_mask, "breakeven_price"] = (
            df.loc[put_mask, "strike"] - df.loc[put_mask, "premium_mid"]
        )
    
    # Calculate percentage to breakeven (relative to current spot)
    if "spot" in df.columns:
        df["pct_to_breakeven"] = np.where(
            df["spot"] > 0,
            ((df["breakeven_price"] - df["spot"]) / df["spot"]) * 100.0,
            np.nan
        )
        df["pct_to_breakeven"] = df["pct_to_breakeven"].fillna(0.0)
    else:
        df["pct_to_breakeven"] = np.nan
    
    return df


def get_top_opportunities(df: pd.DataFrame, top_n: int = 10, 
                          sort_by: str = "profit_ratio_at_target") -> pd.DataFrame:
    """
    Get top opportunities based on profit ratio or other metrics.
    
    Args:
        df: Options DataFrame with profit metrics
        top_n: Number of top opportunities to return
        sort_by: Column to sort by
        
    Returns:
        DataFrame with top opportunities
    """
    if df.empty or sort_by not in df.columns:
        return pd.DataFrame()
    
    return df.nlargest(top_n, sort_by).copy()


def calculate_summary_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate summary metrics for the filtered options.
    
    Args:
        df: Options DataFrame
        
    Returns:
        Dictionary with summary metrics
    """
    if df.empty:
        return {
            "total_contracts": 0,
            "avg_profit_ratio": 0.0,
            "avg_implied_itm_prob": 0.0,
            "total_volume": 0,
            "total_open_interest": 0
        }
    
    summary = {
        "total_contracts": len(df),
        "avg_profit_ratio": float(df["profit_ratio_at_target"].mean()) if "profit_ratio_at_target" in df.columns else 0.0,
        "avg_implied_itm_prob": float(df["implied_itm_prob"].mean()) if "implied_itm_prob" in df.columns else 0.0,
        "total_volume": int(df["volume"].sum()) if "volume" in df.columns else 0,
        "total_open_interest": int(df["openInterest"].sum()) if "openInterest" in df.columns else 0
    }
    
    return summary

