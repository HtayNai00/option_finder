"""
Options Finder - Main Streamlit Application
A TradingView-inspired options analysis platform.
"""

import os
import urllib.parse
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Import our custom modules
from config import (
    STREAMLIT_PAGE_TITLE, STREAMLIT_LAYOUT, DEFAULT_OPTION_TYPE, 
    DEFAULT_MAX_DTE, LATEST_PARQUET
)
from core.data_loader import (
    load_latest_parquet, get_available_tickers, get_available_expiries,
    validate_data_columns, get_data_summary
)
from core.analytics import (
    compute_profit_metrics, add_implied_itm_probability, add_greeks_to_dataframe,
    calculate_breakeven_price, get_top_opportunities, calculate_summary_metrics
)
from core.filters import (
    basic_filters, apply_expiry_filter, get_filter_summary
)
from core.ml_predictor import create_ml_predictor
from core.utils import (
    format_currency, format_percentage, format_ratio, create_detail_link,
    create_list_link, get_expiry_label, get_color_for_value
)

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_PAGE_TITLE,
    layout=STREAMLIT_LAYOUT,
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    """Load custom CSS styling."""
    css_path = "assets/style.css"
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def tradingview_expiry_slider(dates, selected=None, key="expiry_slider"):
    """Create a working TradingView-style expiry selector using Streamlit native components."""
    if not dates:
        st.warning("No expiries available.")
        return None

    # Filter dates to only show upcoming dates (not expired)
    from datetime import datetime, date
    current_date = date.today()
    
    upcoming_dates = []
    for date_str in dates:
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d').date()
            if dt >= current_date:  # Only include today and future dates
                upcoming_dates.append(date_str)
        except:
            continue
    
    if not upcoming_dates:
        st.warning("No upcoming expiry dates available.")
        return None

    # Group dates by month
    months_data = {}
    for date_str in upcoming_dates:
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            month_key = dt.strftime('%Y-%m')
            if month_key not in months_data:
                months_data[month_key] = []
            months_data[month_key].append(date_str)
        except:
            continue

    # Create a clean, working interface with proper styling
    st.markdown("### Expiry Dates", unsafe_allow_html=True)
    st.markdown('<div class="expiry-selector">', unsafe_allow_html=True)
    
    # Show current date info
    current_date_str = current_date.strftime('%Y-%m-%d')
    st.markdown(f"""
    <div style="color: #ffffff; font-size: 0.8rem; margin-bottom: 0px; padding: 8px; 
                background: #000000; border: 1px solid #333333; border-radius: 6px; text-align: center;">
        Today: {current_date.strftime('%B %d, %Y')} | Showing only upcoming expiry dates
    </div>
    """, unsafe_allow_html=True)
    
    # Sort months chronologically
    sorted_months = sorted(months_data.keys())
    
    # Render months and date buttons
    for month_key in sorted_months:
        month_dates = months_data[month_key]
        month_name = datetime.strptime(month_dates[0], '%Y-%m-%d').strftime('%b')
        
        st.markdown(f'<div class="month-label" style="margin-top: 12px; margin-bottom: 8px;">{month_name}</div>', unsafe_allow_html=True)
        
        # Create columns for date buttons (max 8 per row)
        cols = st.columns(min(len(month_dates), 8))
        
        for i, date_str in enumerate(month_dates):
            with cols[i % len(cols)]:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                day = dt.day
                
                # Determine if this is today or future
                is_today = date_str == current_date_str
                is_selected = date_str == selected
                
                # Button styling based on date type
                if is_today:
                    button_type = "primary" if is_selected else "secondary"
                    help_text = f"TODAY: {dt.strftime('%B %d, %Y')}"
                else:
                    button_type = "primary" if is_selected else "secondary"
                    help_text = f"Select {dt.strftime('%B %d, %Y')}"
                
                if st.button(
                    str(day), 
                    key=f"date_{date_str}_{key}",
                    help=help_text,
                    type=button_type,
                    use_container_width=True
                ):
                    # Update URL parameters and rerun
                    params = st.query_params
                    params["view"] = "list"
                    params["expiry_mode"] = "Exact"
                    params["exact_expiry"] = date_str
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return selected


def render_header():
    """Render the main header with refresh functionality."""
    # Check data freshness
    from core.data_loader import check_data_freshness
    freshness = check_data_freshness()
    
    # Create header with refresh button
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown("""
        <div class="main-header">
            <h1>Options Finder</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Data freshness indicator
        if freshness["exists"]:
            age_hours = freshness["age_hours"]
            if age_hours < 1:
                st.success(f"✅ Fresh ({age_hours*60:.0f}m)")
            elif age_hours < 24:
                st.warning(f"⚠️ {age_hours:.1f}h old")
            else:
                st.error(f"❌ {age_hours:.1f}h old")
        else:
            st.error("❌ No data")
    
    with col3:
        # Refresh and ML training buttons
        col3a, col3b = st.columns(2)
        
        with col3a:
            if st.button("🔄 Refresh", help="Update options data from API", use_container_width=True):
                from core.data_loader import refresh_data_from_api
                refresh_data_from_api()
        
        with col3b:
            if st.button("🤖 Train ML", help="Train ML model for predictions", use_container_width=True):
                with st.spinner("Training ML model..."):
                    try:
                        import subprocess
                        import sys
                        result = subprocess.run([sys.executable, "train_ml_model.py"], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            st.success("✅ ML model trained successfully!")
                            st.rerun()
                        else:
                            st.error(f"❌ Training failed: {result.stderr}")
                    except Exception as e:
                        st.error(f"❌ Training error: {e}")


def render_filters(df_all, params):
    """Render the filter controls."""
    st.markdown('<div class="filter-row">', unsafe_allow_html=True)
    
    # Get available data
    tickers = get_available_tickers(df_all)
    if not tickers:
        st.error("No ticker data available")
        return None, None, None, None, None
    
    # Filter controls
    colA, colB, colC, colD = st.columns([2, 1, 1, 3])
    
    with colA:
        sel_tickers = st.multiselect(
            "Tickers", 
            options=tickers, 
            default=tickers[:5] if len(tickers) > 5 else tickers,
            help="Select ticker symbols to analyze"
        )
    
    with colB:
        opt_type = st.segmented_control(
            "Option type", 
            options=["put", "call"], 
            default=DEFAULT_OPTION_TYPE
        )
    
    with colC:
        expiry_mode_default = "Exact" if params.get("exact_expiry") else "Range"
        expiry_mode = st.segmented_control(
            "Expiry filter", 
            options=["Range", "Exact"], 
            default=expiry_mode_default
        )
    
    # Available expiries for selected tickers & side
    avail_expiries = get_available_expiries(df_all, sel_tickers, opt_type)
    
    with colD:
        if expiry_mode == "Range":
            max_dte = st.number_input(
                "Expiry within (days)", 
                min_value=1, max_value=60, 
                value=DEFAULT_MAX_DTE, step=1
            )
            exact_expiry = None
        else:
            # Get selected expiry from URL params
            url_selected = params.get("exact_expiry")
            if url_selected and url_selected in avail_expiries:
                exact_expiry = url_selected
            else:
                exact_expiry = None
            
            # Render the TradingView-style slider (includes the "Expiry Dates" heading internally)
            selected_from_slider = tradingview_expiry_slider(avail_expiries, selected=exact_expiry)
            
            # Show selected date info
            if exact_expiry:
                from datetime import datetime
                try:
                    selected_date = datetime.strptime(exact_expiry, '%Y-%m-%d')
                    formatted_date = selected_date.strftime('%B %d, %Y')
                    st.success(f"📌 **Selected: {formatted_date}**")
                    
                    # Show contract count for selected date
                    selected_contracts = df_all[
                        (df_all["expiry_date"] == exact_expiry) & 
                        (df_all["ticker"].isin(sel_tickers)) & 
                        (df_all["type"] == opt_type)
                    ]
                    st.caption(f"📊 {len(selected_contracts)} contracts available for this expiry")
                    
                except:
                    st.info(f"📌 Selected: **{exact_expiry}**")
            else:
                # Removed: "Click on a date above to filter by exact expiry" info box
                pass
            
            max_dte = None
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return sel_tickers, opt_type, expiry_mode, max_dte, exact_expiry


def render_detail_view(df_all, contract_id):
    """Render the detail view for a specific contract."""
    # Find the contract
    contract = df_all[df_all["contract_id"] == contract_id]
    if contract.empty:
        st.error("Contract not found.")
        st.link_button("← Back to list", create_list_link())
        return
    
    # Compute metrics first to ensure all data is available
    df_metrics = compute_profit_metrics(contract, contract.iloc[0]["type"])
    df_metrics = add_implied_itm_probability(df_metrics)
    df_metrics = calculate_breakeven_price(df_metrics)
    
    # Get the row with computed metrics
    row = df_metrics.iloc[0]
    typ = row["type"]
    
    # Calculate Greeks
    from core.analytics import bs_greeks_one
    from config import RISK_FREE
    
    # Extract values safely
    S = float(row.get("spot", 0))
    K = float(row.get("strike", 0))
    dte = float(row.get("days_to_expiry", 0))
    T = max(dte / 365.0, 1/365.0) if dte > 0 else 1/365.0  # Minimum 1 day to avoid division by zero
    
    # Get implied volatility - Yahoo Finance typically stores IV as decimal (0-1), e.g., 0.20 for 20%
    # But some sources use percentage format (0-100), e.g., 20 for 20%
    sigma_raw = row.get("impliedVolatility", None)
    
    # Handle different IV formats
    if pd.isna(sigma_raw) or sigma_raw is None or sigma_raw == 0:
        # If IV is missing or zero, use a reasonable estimate based on option premium
        # Try to infer IV from the premium using a rough approximation
        premium = float(row.get("premium_mid", 0))
        if premium > 0 and S > 0 and K > 0:
            # Rough IV estimate: use premium as percentage of spot
            # This is a heuristic - actual IV would require iterative solving
            moneyness = abs(K / S - 1)
            sigma = max(premium / S / np.sqrt(T), 0.15)  # At least 15% IV
            # Cap at reasonable max (200%)
            sigma = min(sigma, 2.0)
        else:
            # Default to 20% IV if we can't estimate
            sigma = 0.20
    else:
        try:
            sigma = float(sigma_raw)
            # Determine format: if > 1, it's likely percentage format (20 = 20%)
            # If <= 1, it's likely decimal format (0.20 = 20%)
            if sigma > 1:
                # Percentage format: convert to decimal
                sigma = sigma / 100.0
            # Ensure sigma is in reasonable range (0.01 to 5.0 = 1% to 500%)
            sigma = max(0.01, min(sigma, 5.0))
        except (ValueError, TypeError):
            sigma = 0.20  # Default fallback
    
    # Calculate Greeks - always try to calculate if we have valid inputs
    if S > 0 and K > 0 and T > 0 and sigma > 0:
        try:
            greeks = bs_greeks_one(S, K, T, sigma, RISK_FREE, typ)
            # Ensure we have valid numeric values (not NaN or None)
            for key in greeks:
                if pd.isna(greeks[key]) or greeks[key] is None:
                    # Try fallback calculation with default IV
                    greeks_fallback = bs_greeks_one(S, K, T, 0.20, RISK_FREE, typ)
                    if not pd.isna(greeks_fallback[key]):
                        greeks[key] = greeks_fallback[key]
        except Exception as e:
            # If calculation fails, try with default sigma
            try:
                greeks = bs_greeks_one(S, K, T, 0.20, RISK_FREE, typ)
            except Exception:
                greeks = {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan}
    else:
        # If critical inputs are missing, try with defaults anyway
        if S > 0 and K > 0:
            try:
                # Use default values for missing parameters
                T_default = max(T, 1/365.0) if T > 0 else 30/365.0  # Default 30 days
                sigma_default = sigma if sigma > 0 else 0.20  # Default 20% IV
                greeks = bs_greeks_one(S, K, T_default, sigma_default, RISK_FREE, typ)
            except Exception:
                greeks = {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan}
        else:
            greeks = {"delta": np.nan, "gamma": np.nan, "theta": np.nan, "vega": np.nan, "rho": np.nan}
    
    # Header - plain text without box
    st.markdown(f"""
    <div style="margin-bottom: 20px;">
        <h1 style="color: #ffffff; margin: 0 0 10px 0; font-size: 2rem;">Option Detail</h1>
        <p style="color: #cccccc; margin: 0; font-size: 1.1rem;">
            {row['ticker']} | Exp {row['expiry_date']} | Strike {format_currency(row['strike'])} | {typ.upper()}
        </p>
        <p style="color: #888888; margin: 5px 0 0 0; font-size: 0.9rem;">
            Contract ID: {contract_id}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Back button with better styling
    col_back, col_spacer = st.columns([1, 4])
    with col_back:
        st.link_button("← Back to List", create_list_link(), type="secondary")
    
    # Key metrics with professional styling
    st.markdown("### Key Metrics")
    
    # First row of metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "Spot Price", 
            format_currency(row['spot']),
            help="Current stock price"
        )
    with c2:
        st.metric(
            "Premium", 
            format_currency(row['premium_mid'], 3),
            help="Option premium (mid price)"
        )
    with c3:
        st.metric(
            "Breakeven", 
            format_currency(row.get('breakeven_price', 0)),
            help="Breakeven price for this option"
        )
    with c4:
        profit_ratio = row.get('profit_ratio_at_target', 0)
        target_pct = int(row.get('target_move_pct', 0)*100)
        st.metric(
            f"Profit Ratio @ {target_pct}%", 
            format_ratio(profit_ratio),
            help=f"Profit ratio if stock moves {target_pct}%"
        )
    
    # Second row of metrics
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.metric(
            "Days to Expiry", 
            f"{int(row['days_to_expiry'])}",
            help="Days remaining until expiration"
        )
    with d2:
        st.metric(
            "Open Interest", 
            f"{int(row['openInterest']):,}",
            help="Number of outstanding contracts"
        )
    with d3:
        st.metric(
            "Volume", 
            f"{int(row['volume']):,}",
            help="Trading volume today"
        )
    with d4:
        itm_prob = row.get('implied_itm_prob', 0)
        st.metric(
            "ITM Probability", 
            format_percentage(itm_prob),
            help="Implied probability of finishing in-the-money"
        )
    
    # Greeks section - no box, just metrics
    st.markdown("### Greeks (Black–Scholes)")
    
    greek_cols = st.columns(5)
    greek_names = ["Delta", "Gamma", "Theta (per day)", "Vega (per +1% IV)", "Rho (per +1% rate)"]
    greek_keys = ["delta", "gamma", "theta", "vega", "rho"]
    greek_help = [
        "Price sensitivity to underlying asset",
        "Rate of change of delta",
        "Time decay per calendar day",
        "Sensitivity to implied volatility",
        "Sensitivity to interest rates"
    ]
    
    for i, (name, key, help_text) in enumerate(zip(greek_names, greek_keys, greek_help)):
        with greek_cols[i]:
            value = greeks.get(key, np.nan)
            if not pd.isna(value):
                st.metric(name, f"{value:.4f}", help=help_text)
            else:
                st.metric(name, "N/A", help=help_text)
    
    # Contract Details - Trader-focused with big fonts and clean boxes
    st.markdown("### Contract Details")
    
    # Create trader-focused boxes with big fonts
    st.markdown("""
    <style>
    .trader-box {
        background: #1e1e1e;
        border: 2px solid #404040;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .trader-label {
        color: #888888;
        font-size: 0.9rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .trader-value {
        color: #ffffff;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 16px;
    }
    .trader-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0;
        border-bottom: 1px solid #333333;
    }
    .trader-row:last-child {
        border-bottom: none;
    }
    .trader-row-label {
        color: #cccccc;
        font-size: 1.1rem;
        font-weight: 500;
    }
    .trader-row-value {
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main contract info box
    st.markdown(f"""
    <div class="trader-box">
        <div class="trader-label">Contract</div>
        <div class="trader-value">{row['ticker']} {row['type'].upper()} ${row['strike']:.2f}</div>
        <div class="trader-row">
            <span class="trader-row-label">Expiry</span>
            <span class="trader-row-value">{row['expiry_date']} ({int(row['days_to_expiry'])} days)</span>
        </div>
        <div class="trader-row">
            <span class="trader-row-label">Spot Price</span>
            <span class="trader-row-value">${row['spot']:.2f}</span>
        </div>
        <div class="trader-row">
            <span class="trader-row-label">Strike Price</span>
            <span class="trader-row-value">${row['strike']:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Pricing box
    st.markdown(f"""
    <div class="trader-box">
        <div class="trader-label">Pricing</div>
        <div class="trader-row">
            <span class="trader-row-label">Bid</span>
            <span class="trader-row-value">${row['bid']:.3f}</span>
        </div>
        <div class="trader-row">
            <span class="trader-row-label">Ask</span>
            <span class="trader-row-value">${row['ask']:.3f}</span>
        </div>
        <div class="trader-row">
            <span class="trader-row-label">Mid Premium</span>
            <span class="trader-row-value">${row['premium_mid']:.3f}</span>
        </div>
        <div class="trader-row">
            <span class="trader-row-label">Breakeven</span>
            <span class="trader-row-value">${row.get('breakeven_price', 0):.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Market activity box
    st.markdown(f"""
    <div class="trader-box">
        <div class="trader-label">Market Activity</div>
        <div class="trader-row">
            <span class="trader-row-label">Open Interest</span>
            <span class="trader-row-value">{int(row['openInterest']):,}</span>
        </div>
        <div class="trader-row">
            <span class="trader-row-label">Volume</span>
            <span class="trader-row-value">{int(row['volume']):,}</span>
        </div>
        <div class="trader-row">
            <span class="trader-row-label">Implied Volatility</span>
            <span class="trader-row-value">{format_percentage(row.get('impliedVolatility', 0))}</span>
        </div>
        <div class="trader-row">
            <span class="trader-row-label">ITM Probability</span>
            <span class="trader-row-value">{format_percentage(row.get('implied_itm_prob', 0))}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_list_view(df_all, sel_tickers, opt_type, expiry_mode, max_dte, exact_expiry):
    """Render the main list view with tables."""
    # Apply filters
    df = basic_filters(df_all, sel_tickers, opt_type)
    df = apply_expiry_filter(df, expiry_mode, max_dte, exact_expiry)
    
    if df.empty:
        st.info("No contracts match the selection.")
        return
    
    # Compute analytics
    df = compute_profit_metrics(df, opt_type)
    df = add_implied_itm_probability(df)
    df = calculate_breakeven_price(df)
    
    # Summary metrics
    summary = calculate_summary_metrics(df)
    
    # Display summary
    st.markdown("### 📈 Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Contracts", f"{summary.get('total_contracts', 0):,}")
    with col2:
        st.metric("Avg Profit Ratio", format_ratio(summary.get('avg_profit_ratio', 0)))
    with col3:
        st.metric("Avg ITM Probability", format_percentage(summary.get('avg_implied_itm_prob', 0)))
    with col4:
        st.metric("Total Volume", f"{summary.get('total_volume', 0):,}")
    
    st.divider()
    
    # By expiration table
    st.subheader(f"By Expiration — {opt_type.upper()}s")
    
    # Group by expiry
    grp = (df.groupby(["expiry_date", "days_to_expiry"], as_index=False)
           .agg(
               contracts=("contract_id", "count"),
               median_premium=("premium_mid", "median"),
               median_itm_prob=("implied_itm_prob", "median"),
               min_pct_to_breakeven=("pct_to_breakeven", "min"),
           ))
    
    # Get best contract per expiry
    best = (df.sort_values(["expiry_date", "profit_ratio_at_target"], ascending=[True, False])
            .groupby("expiry_date", as_index=False)
            .head(1)[["expiry_date", "contract_id", "profit_ratio_at_target", "ticker", "strike"]])
    
    grp = grp.merge(best, on="expiry_date", how="left")
    grp["Details"] = grp["contract_id"].apply(
        lambda cid: create_detail_link(cid) if pd.notna(cid) else ""
    )
    
    # Display by expiration table
    display_cols = [
        "Details", "expiry_date", "days_to_expiry", "contracts", 
        "median_premium", "median_itm_prob", "min_pct_to_breakeven",
        "ticker", "strike", "profit_ratio_at_target"
    ]
    
    st.dataframe(
        grp[display_cols].sort_values("days_to_expiry"),
        width='stretch',
        hide_index=True,
        column_config={
            "Details": st.column_config.LinkColumn("Details", display_text="Open"),
            "median_premium": st.column_config.NumberColumn("Median Premium", format="$%.3f"),
            "median_itm_prob": st.column_config.NumberColumn("Median ITM %", format="%.1f%%"),
            "min_pct_to_breakeven": st.column_config.NumberColumn("Best % to Breakeven", format="%.2f%%"),
            "profit_ratio_at_target": st.column_config.NumberColumn("Best Profit Ratio", format="%.2f×"),
            "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
            "days_to_expiry": st.column_config.NumberColumn("DTE", format="%.0f"),
        }
    )
    
    st.divider()
    
    # Top profit ratio table
    st.subheader(f"Top Profit Ratio — {opt_type.upper()}s")
    
    # Sort and add detail links
    df_sorted = df.sort_values(
        ["profit_ratio_at_target", "openInterest", "volume"], 
        ascending=[False, False, False]
    ).copy()
    
    df_sorted["Details"] = df_sorted["contract_id"].apply(create_detail_link)
    
    # Display columns
    display_cols = [
        "Details", "ticker", "expiry_date", "days_to_expiry", "strike", 
        "premium_mid", "breakeven_price", "pct_to_breakeven", 
        "openInterest", "volume", "profit_ratio_at_target", "implied_itm_prob"
    ]
    
    st.dataframe(
        df_sorted[display_cols].head(200),
        width='stretch',
        hide_index=True,
        column_config={
            "Details": st.column_config.LinkColumn("Details", display_text="Open"),
            "premium_mid": st.column_config.NumberColumn("Premium", format="$%.3f"),
            "breakeven_price": st.column_config.NumberColumn("Breakeven", format="$%.2f"),
            "pct_to_breakeven": st.column_config.NumberColumn("% to Breakeven", format="%.2f%%"),
            "profit_ratio_at_target": st.column_config.NumberColumn("Profit Ratio", format="%.2f×"),
            "implied_itm_prob": st.column_config.NumberColumn("ITM %", format="%.1f%%"),
            "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
            "openInterest": st.column_config.NumberColumn("OI", format="%.0f"),
            "volume": st.column_config.NumberColumn("Vol", format="%.0f"),
            "days_to_expiry": st.column_config.NumberColumn("DTE", format="%.0f"),
        }
    )
    
    st.divider()
    
    # ML Predictions Section
    st.subheader("🤖 ML-Predicted Top 10 Opportunities")
    
    try:
        # Create ML predictor
        ml_predictor = create_ml_predictor()
        
        # Get ML predictions
        with st.spinner("🧠 Running ML predictions..."):
            top_ml_predictions = ml_predictor.get_top_predictions(df, top_n=10)
        
        if not top_ml_predictions.empty:
            # Create display dataframe for ML predictions
            ml_display_cols = [
                "rank", "ticker", "type", "expiry_date", "strike", "spot", 
                "premium_mid", "ml_profit_score"
            ]
            
            # Add profit ratio columns if they exist
            if 'profit_ratio_up' in top_ml_predictions.columns:
                ml_display_cols.append("profit_ratio_up")
            if 'profit_ratio_down' in top_ml_predictions.columns:
                ml_display_cols.append("profit_ratio_down")
            if 'profit_ratio_at_target' in top_ml_predictions.columns:
                ml_display_cols.append("profit_ratio_at_target")
            
            # Add other columns if they exist
            optional_cols = ["implied_itm_prob", "volume", "openInterest", "impliedVolatility"]
            for col in optional_cols:
                if col in top_ml_predictions.columns:
                    ml_display_cols.append(col)
            
            # Add detail links
            top_ml_predictions["Details"] = top_ml_predictions["contract_id"].apply(create_detail_link)
            ml_display_cols.insert(1, "Details")
            
            st.dataframe(
                top_ml_predictions[ml_display_cols],
                width='stretch',
                hide_index=True,
                column_config={
                    "Details": st.column_config.LinkColumn("Details", display_text="Open"),
                    "rank": st.column_config.NumberColumn("Rank", format="%.0f"),
                    "premium_mid": st.column_config.NumberColumn("Premium", format="$%.3f"),
                    "ml_profit_score": st.column_config.NumberColumn("ML Score", format="%.3f"),
                    "profit_ratio_up": st.column_config.NumberColumn("Profit Up", format="%.2f×"),
                    "profit_ratio_down": st.column_config.NumberColumn("Profit Down", format="%.2f×"),
                    "implied_itm_prob": st.column_config.NumberColumn("ITM %", format="%.1f%%"),
                    "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                    "spot": st.column_config.NumberColumn("Spot", format="$%.2f"),
                    "openInterest": st.column_config.NumberColumn("OI", format="%.0f"),
                    "volume": st.column_config.NumberColumn("Vol", format="%.0f"),
                    "impliedVolatility": st.column_config.NumberColumn("IV", format="%.1f%%"),
                }
            )
            
            # Show ML model info
            st.info("🤖 **ML Model**: Analyzes 20+ features including Greeks, liquidity, volatility, and profit metrics to predict the most profitable options.")
        else:
            st.warning("No ML predictions available. Model may need training.")
            
    except Exception as e:
        st.error(f"ML prediction error: {e}")
        st.info("ML predictions temporarily unavailable. Using traditional profit ratio analysis above.")


def main():
    """Main application function."""
    # Load CSS
    load_css()
    
    # Check if we should auto-refresh data
    from core.data_loader import check_data_freshness
    freshness = check_data_freshness()
    
    # Auto-refresh if data is stale (older than 1 hour)
    if freshness.get("needs_refresh", False) and freshness.get("exists", False):
        st.info("🔄 Data is stale. Auto-refreshing...")
        from core.data_loader import refresh_data_from_api
        refresh_data_from_api()
        return
    
    # Render header
    render_header()
    
    # Load data
    df_all = load_latest_parquet()
    if df_all.empty:
        st.error(f"Latest parquet not found or empty. Expected path: {LATEST_PARQUET}")
        st.info("Please ensure the data file exists and contains valid options data.")
        return
    
    # Validate data columns
    if not validate_data_columns(df_all):
        st.warning("Data validation failed. Some features may not work correctly.")
    
    # Get URL parameters
    params = st.query_params
    view = params.get("view", ["list"])
    view = view[0] if isinstance(view, list) else view
    
    # Handle detail view
    if view == "detail":
        contract_id = params.get("id", [None])
        contract_id = contract_id[0] if isinstance(contract_id, list) else contract_id
        
        if not contract_id:
            st.error("Missing contract ID.")
            st.link_button("← Back to list", create_list_link())
            return
        
        render_detail_view(df_all, contract_id)
        return
    
    # Main list view
    # Render filters
    filter_result = render_filters(df_all, params)
    if filter_result[0] is None:  # No tickers available
        return
    
    sel_tickers, opt_type, expiry_mode, max_dte, exact_expiry = filter_result
    
    # Render list view
    render_list_view(df_all, sel_tickers, opt_type, expiry_mode, max_dte, exact_expiry)


if __name__ == "__main__":
    main()
