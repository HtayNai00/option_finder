import os
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

LATEST_PARQUET = "data/latest/options_puts_latest.parquet"

# ---------------------------
# Helpers
# ---------------------------
def load_latest(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Latest parquet not found: {path}")
        return pd.DataFrame()
    df = pd.read_parquet(path)
    # defensive cleanup
    df = df.copy()
    if "type" in df.columns:
      df = df[df["type"] == "put"]
    df = df.replace([np.inf, -np.inf], np.nan)
    # ensure needed cols exist
    needed = ["ticker","spot","strike","premium_mid","adjustive_price","pct_to_breakeven",
              "expiry_date","days_to_expiry","openInterest","volume","asof_ny","asof_utc"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        st.error(f"Missing expected columns in latest parquet: {missing}")
    return df

def compute_profit_metrics(df: pd.DataFrame, target_drop: float) -> pd.DataFrame:
    if df.empty: 
        return df
    df = df.copy()
    df["target_drop_pct"] = target_drop
    df["target_price"] = df["spot"] * (1 - target_drop)
    # per-share profit at expiry vs target
    num = (df["strike"] - df["target_price"] - df["premium_mid"]).clip(lower=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(df["premium_mid"] > 0, num / df["premium_mid"], 0.0)
        pr = np.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0)
    df["profit_ratio_at_target"] = pr
    df["profit_at_target"] = num * 100.0  # per contract
    return df

def apply_filters(
    df: pd.DataFrame,
    lower_moneyness: float, upper_moneyness: float,
    min_premium: float, min_oi: int, min_vol: int,
    max_dte: int
) -> pd.DataFrame:
    if df.empty: 
        return df
    df = df.copy()
    # drop NAs
    df = df.dropna(subset=["spot","strike","premium_mid","openInterest","volume","days_to_expiry"])
    # moneyness window (strike relative to spot)
    lb = df["spot"] * lower_moneyness
    ub = df["spot"] * upper_moneyness
    df = df[(df["strike"] >= lb) & (df["strike"] <= ub)]
    # liquidity / premium / DTE guards
    df = df[(df["premium_mid"] >= min_premium) &
            (df["openInterest"] >= min_oi) &
            (df["volume"] >= min_vol) &
            (df["days_to_expiry"] <= max_dte)]
    return df

def leaderboard_top_profit_ratio(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    cols = ["ticker","expiry_date","days_to_expiry","strike","premium_mid","adjustive_price",
            "pct_to_breakeven","openInterest","volume","profit_ratio_at_target"]
    exist = [c for c in cols if c in df.columns]
    return (df.sort_values(["profit_ratio_at_target","openInterest","volume"],
                           ascending=[False, False, False])
              .loc[:, exist]
              .head(n))

def leaderboard_closest_breakeven(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if "pct_to_breakeven" not in df.columns:
        return pd.DataFrame()
    temp = df.copy()
    temp["abs_breakeven"] = temp["pct_to_breakeven"].abs()
    cols = ["ticker","expiry_date","days_to_expiry","strike","premium_mid","adjustive_price",
            "pct_to_breakeven","openInterest","volume","profit_ratio_at_target"]
    exist = [c for c in cols if c in temp.columns]
    return (temp.sort_values(["abs_breakeven","openInterest","volume"],
                             ascending=[True, False, False])
                .loc[:, exist]
                .head(n))

def format_pct(x):
    if pd.isna(x): return ""
    return f"{x*100:.2f}%"

def insight_sentences(row: pd.Series) -> list[str]:
    lines = []
    # Breakeven
    pct = row.get("pct_to_breakeven", np.nan)
    if pd.notna(pct):
        pct_abs = abs(pct)
        if pct_abs > 0.05:
            lines.append(f"Breakeven is **far** (~{pct_abs*100:.2f}% from spot). Small drops may not move this put enough yet.")
        elif pct_abs > 0.02:
            lines.append(f"Breakeven is **moderate** (~{pct_abs*100:.2f}% from spot). A modest decline could flip to profit.")
        else:
            lines.append(f"Breakeven is **close** (~{pct_abs*100:.2f}% from spot). A small move can tip this profitable.")
    # Premium efficiency
    pr = row.get("profit_ratio_at_target", 0.0)
    if pr >= 2.0:
        lines.append("High payoff efficiency **if** price reaches your target zone.")
    elif pr >= 0.5:
        lines.append("Payoff is reasonable at your target; not extreme.")
    else:
        lines.append("At your target drop, payoff is limited relative to premium.")
    # Liquidity
    oi = row.get("openInterest", 0); vol = row.get("volume", 0)
    if oi < 100 or (pd.notna(vol) and vol < 50):
        lines.append("**Thin liquidity** (low OI/volume) may mean wider fills and harder exits.")
    else:
        lines.append("Liquidity looks **adequate** for entry/exit.")
    # DTE hint
    dte = row.get("days_to_expiry", None)
    if pd.notna(dte) and dte <= 7:
        lines.append("Near expiry — limited time for the move.")
    return lines

# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Put Option Finder (V1)", layout="wide")
st.title("📉 Put Option Finder — V1")

# Load data
df_latest = load_latest(LATEST_PARQUET)
if df_latest.empty:
    st.stop()

# Status / freshness
asof_ny = df_latest.get("asof_ny")
last_ts = None
if isinstance(asof_ny, pd.Series) and not asof_ny.empty:
    # take max string
    last_ts = max(asof_ny)
st.caption(f"Last updated (NY): **{last_ts}**  | Rows: **{len(df_latest)}**")

# Sidebar controls
st.sidebar.header("Filters")
target_drop = st.sidebar.slider("Target drop (%)", 0.0, 0.30, 0.10, 0.01)
lower_moneyness = st.sidebar.slider("Lower moneyness (strike / spot)", 0.40, 1.00, 0.60, 0.01)
upper_moneyness = st.sidebar.slider("Upper moneyness (strike / spot)", 0.80, 1.20, 1.05, 0.01)
min_premium = st.sidebar.number_input("Min premium ($)", value=0.10, min_value=0.00, step=0.05)
min_oi = st.sidebar.number_input("Min open interest", value=200, min_value=0, step=50)
min_vol = st.sidebar.number_input("Min volume (today)", value=10, min_value=0, step=5)
max_dte = st.sidebar.number_input("Max days to expiry", value=30, min_value=1, step=1)

# Ticker subset (optional)
tickers_sorted = sorted(df_latest["ticker"].dropna().unique().tolist())
selected_tickers = st.sidebar.multiselect("Tickers", options=tickers_sorted, default=tickers_sorted)

# Apply subset
df = df_latest[df_latest["ticker"].isin(selected_tickers)].copy()

# Compute profit metrics for chosen target
df = compute_profit_metrics(df, target_drop=target_drop)

# Apply filters
df_filt = apply_filters(df, lower_moneyness, upper_moneyness, min_premium, min_oi, min_vol, max_dte)

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Contracts scanned", f"{len(df):,}")
col2.metric("Contracts after filters", f"{len(df_filt):,}")
median_breakeven = df_filt["pct_to_breakeven"].abs().median() if not df_filt.empty else np.nan
col3.metric("Median |% to breakeven|", format_pct(median_breakeven) if pd.notna(median_breakeven) else "—")

st.divider()

# Leaderboards
left, right = st.columns(2)

with left:
    st.subheader(f"🏆 Top Profit Ratio  @ {int(target_drop*100)}% drop")
    top_pr = leaderboard_top_profit_ratio(df_filt, n=20)
    st.dataframe(top_pr, use_container_width=True)

with right:
    st.subheader("🎯 Closest Breakeven (lowest % to breakeven)")
    close_be = leaderboard_closest_breakeven(df_filt, n=20)
    # display % nicely
    if "pct_to_breakeven" in close_be.columns:
        close_be = close_be.copy()
        close_be["pct_to_breakeven"] = close_be["pct_to_breakeven"].map(format_pct)
    st.dataframe(close_be, use_container_width=True)

st.divider()

# Insights Panel
st.subheader("🧠 Insights (pick a contract)")
# Selection controls (simple & robust)
sel_ticker = st.selectbox("Ticker", options=tickers_sorted)
exp_choices = sorted(df_filt[df_filt["ticker"] == sel_ticker]["expiry_date"].dropna().unique().tolist())
sel_expiry = st.selectbox("Expiry", options=exp_choices)
strike_choices = sorted(df_filt[(df_filt["ticker"] == sel_ticker) & (df_filt["expiry_date"] == sel_expiry)]["strike"].unique().tolist())
sel_strike = st.selectbox("Strike", options=strike_choices)

row = df_filt[(df_filt["ticker"] == sel_ticker) &
              (df_filt["expiry_date"] == sel_expiry) &
              (df_filt["strike"] == sel_strike)].head(1)

if row.empty:
    st.info("No row matches the selection with the current filters.")
else:
    r = row.iloc[0]
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Spot", f"{r['spot']:.2f}")
    colB.metric("Premium", f"{r['premium_mid']:.3f}")
    colC.metric("Breakeven", f"{r['adjustive_price']:.2f}")
    colD.metric("Profit Ratio@", f"{r['profit_ratio_at_target']:.2f}")

    with st.expander("View raw fields", expanded=False):
        st.write(row)

    st.markdown("**Coach notes:**")
    for line in insight_sentences(r):
        st.write(f"- {line}")

# Download buttons
st.divider()
c1, c2 = st.columns(2)
with c1:
    if not df_filt.empty:
        st.download_button(
            "Download filtered table (CSV)",
            df_filt.to_csv(index=False).encode("utf-8"),
            file_name="puts_filtered.csv",
            mime="text/csv",
            use_container_width=True
        )
with c2:
    if not df_filt.empty:
        st.download_button(
            "Download Top Profit Ratio (CSV)",
            leaderboard_top_profit_ratio(df_filt, n=200).to_csv(index=False).encode("utf-8"),
            file_name="top_profit_ratio.csv",
            mime="text/csv",
            use_container_width=True
        )

st.caption("Tip: toggle filters on the left to refine, adjust target drop %, then inspect a specific contract with the selectors above.")
