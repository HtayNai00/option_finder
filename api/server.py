"""
REST API Server for Options Finder using FastAPI.
Provides API endpoints for accessing options data, analytics, and ML predictions.
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import pandas as pd
import sys
import os

# Add parent directory to path to import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import (
    load_latest_parquet, get_available_tickers, get_available_expiries,
    get_data_summary
)
from core.analytics import (
    compute_profit_metrics, add_implied_itm_probability, add_greeks_to_dataframe,
    calculate_breakeven_price, calculate_summary_metrics
)
from core.filters import basic_filters, apply_expiry_filter
from core.ml_predictor import create_ml_predictor
from config import API_TITLE, API_DESCRIPTION

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "message": "Options Finder API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/options/list")
def get_options_list(
    tickers: Optional[str] = Query(None, description="Comma-separated list of tickers"),
    option_type: Optional[str] = Query(None, description="Option type: call or put"),
    expiry_mode: Optional[str] = Query("Range", description="Range or Exact"),
    max_dte: Optional[int] = Query(None, description="Max days to expiry"),
    exact_expiry: Optional[str] = Query(None, description="Exact expiry date (YYYY-MM-DD)"),
    limit: Optional[int] = Query(100, description="Maximum number of results")
):
    """
    Get filtered list of options.
    
    Returns JSON array of options with all analytics applied.
    """
    try:
        # Load data
        df = load_latest_parquet()
        
        if df.empty:
            return []
        
        # Parse tickers if provided
        ticker_list = tickers.split(",") if tickers else None
        
        # Apply filters
        if ticker_list:
            df = basic_filters(df, ticker_list, option_type)
        
        if expiry_mode == "Range" and max_dte:
            df = apply_expiry_filter(df, "Range", max_dte, None)
        elif expiry_mode == "Exact" and exact_expiry:
            df = apply_expiry_filter(df, "Exact", None, exact_expiry)
        
        # Apply analytics
        opt_type = option_type or "put"
        df = compute_profit_metrics(df, opt_type)
        df = add_implied_itm_probability(df)
        df = calculate_breakeven_price(df)
        
        # Convert to JSON (limit results)
        result = df.head(limit).to_dict(orient="records")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/options/greeks/{contract_id}")
def get_greeks(contract_id: str):
    """
    Get Greeks for a specific option contract.
    
    Args:
        contract_id: Contract identifier (e.g., "AAPL|2025-01-19|call|150.00")
    """
    try:
        # Load data
        df = load_latest_parquet()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")
        
        # Find the contract
        contract = df[df["contract_id"] == contract_id]
        
        if contract.empty:
            raise HTTPException(status_code=404, detail="Contract not found")
        
        row = contract.iloc[0]
        
        # Calculate Greeks
        from core.analytics import bs_greeks_one
        from config import RISK_FREE
        
        T = float(row["days_to_expiry"]) / 365.0
        sigma = float(row.get("impliedVolatility", 0.2))
        
        greeks = bs_greeks_one(
            float(row["spot"]),
            float(row["strike"]),
            T,
            sigma,
            RISK_FREE,
            row["type"]
        )
        
        return greeks
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/options/summary")
def get_summary():
    """Get summary statistics for all options."""
    try:
        df = load_latest_parquet()
        
        if df.empty:
            return {"total_contracts": 0}
        
        summary = get_data_summary(df)
        metrics = calculate_summary_metrics(df)
        
        return {
            "data_summary": summary,
            "metrics": metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/options/expiries")
def get_expiries():
    """Get list of available expiry dates."""
    try:
        df = load_latest_parquet()
        
        if df.empty:
            return []
        
        expiries = get_available_expiries(df)
        
        return expiries
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/options/tickers")
def get_tickers():
    """Get list of available ticker symbols."""
    try:
        df = load_latest_parquet()
        
        if df.empty:
            return []
        
        tickers = get_available_tickers(df)
        
        return tickers
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/refresh")
def refresh_data():
    """Refresh options data from source."""
    try:
        from core.data_loader import refresh_data_from_api
        refresh_data_from_api()
        
        return {"status": "success", "message": "Data refreshed successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ml/top/{top_n}")
def get_ml_top_opportunities(top_n: int = 10):
    """
    Get top ML-predicted opportunities.
    
    Args:
        top_n: Number of top opportunities to return
    """
    try:
        df = load_latest_parquet()
        
        if df.empty:
            return []
        
        # Create ML predictor
        predictor = create_ml_predictor()
        
        # Get top predictions
        top_options = predictor.get_top_predictions(df, top_n=top_n)
        
        # Convert to JSON
        result = top_options.to_dict(orient="records")
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)

