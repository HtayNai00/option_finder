# app/app.py
import os, urllib.parse, time
import numpy as np
import pandas as pd
import streamlit as st

# Optional nicer charts
try:
    import altair as alt
    HAS_ALTAIR = True
except Exception:
    HAS_ALTAIR = False

# ---------------- config you already use ----------------
LATEST_PARQUET = "data/latest/options_puts_latest.parquet"

# Keep these if you need profit-ratio math on detail page:
TARGET_DROP = 0.10  # fixed for detail view; your main page can override/slider as before

# ---------------- common loaders/helpers ----------------
@st.cache_data(ttl=30)
def load_latest(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def compute_profit_metrics(df: pd.DataFrame, target_drop: float) -> pd.DataFrame:
    """Used by Detail and Bird’s Eye. Your main page can keep its own logic/filters."""
    if df.empty: return df
    df = df.copy()
    if "spot" in df.columns and "strike" in df.columns and "premium_mid" in df.columns:
        df["target_drop_pct"] = target_drop
        df["target_price"] = df["spot"] * (1 - target_drop)
        num = (df["strike"] - df["target_price"] - df["premium_mid"]).clip(lower=0.0)  # per share
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = np.where(df["premium_mid"] > 0, num / df["premium_mid"], 0.0)
            pr = np.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0)
        df["profit_ratio_at_target"] = pr
        df["profit_at_target"] = num * 100.0
    return df

def format_pct(x):
    return "" if pd.isna(x) else f"{x*100:.2f}%"

def link_for_contract(cid: str) -> str:
    return f"?view=detail&id={urllib.parse.quote(cid)}"

def coach_notes(row: pd.Series) -> list[str]:
    """Short explanations on the Detail page."""
    lines = []
    pct = row.get("pct_to_breakeven", np.nan)
    if pd.notna(pct):
        d = abs(pct)
        if d > 0.05:
            lines.append(f"Breakeven is **far** (~{d*100:.2f}% from spot). Small drops may not move this put enough yet.")
        elif d > 0.02:
            lines.append(f"Breakeven is **moderate** (~{d*100:.2f}% from spot). A modest decline could flip to profit.")
        else:
            lines.append(f"Breakeven is **close** (~{d*100:.2f}% from spot). A small move can tip this profitable.")
    pr = row.get("profit_ratio_at_target", 0.0)
    if pr >= 2.0:
        lines.append("High payoff efficiency **if** price reaches your target zone.")
    elif pr >= 0.5:
        lines.append("Payoff is reasonable at the target; not extreme.")
    else:
        lines.append("At the target drop, payoff is limited relative to premium.")
    oi = row.get("openInterest", 0); vol = row.get("volume", 0)
    if oi < 100 or (pd.notna(vol) and vol < 50):
        lines.append("🔴 **Thin liquidity** (low OI/vol) → wider fills possible.")
    else:
        lines.append("🟢 Liquidity looks **adequate** for entry/exit.")
    dte = row.get("days_to_expiry", None)
    if pd.notna(dte) and dte <= 7:
        lines.append("⏳ Near expiry — limited time for the move.")
    return lines

# ------------------- PAGES -------------------
def render_detail(df_latest: pd.DataFrame):
    st.title("📄 Contract Detail")
    params = st.query_params
    contract_id = params.get("id", [None])
    if isinstance(contract_id, list): contract_id = contract_id[0]
    if not contract_id:
        st.warning("Missing contract id.")
        st.link_button("← Back", "?view=list")
        st.stop()

    # Compute metrics (if not in file) with fixed TARGET_DROP for consistency here
    df = compute_profit_metrics(df_latest, target_drop=TARGET_DROP)
    row = df[df["contract_id"] == contract_id].head(1)
    if row.empty:
        st.error("Contract not found. Try going back and refreshing.")
        st.link_button("← Back", "?view=list")
        st.stop()

    r = row.iloc[0]
    st.link_button("← Back", "?view=list")
    st.subheader(f"{r.get('ticker','?')} | Exp {r.get('expiry_date','?')} | Strike ${r.get('strike',0):.2f}")
    st.caption(f"As of (NY): {r.get('asof_ny','?')}  |  Contract ID: {r.get('contract_id','?')}")

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Spot", f"${r.get('spot',0):.2f}")
    k2.metric("Premium", f"${r.get('premium_mid',0):.3f}")
    k3.metric("Breakeven", f"${r.get('adjustive_price',0):.2f}")
    k4.metric(f"Profit Ratio @ {int(TARGET_DROP*100)}%", f"{r.get('profit_ratio_at_target',0):.2f}×")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("DTE", f"{int(r.get('days_to_expiry',0))}")
    k6.metric("Open Interest", f"{int(r.get('openInterest',0))}")
    k7.metric("Volume (today)", f"{int(r.get('volume',0) or 0)}")
    k8.metric("% to Breakeven", f"{format_pct(r.get('pct_to_breakeven', np.nan))}")

    st.markdown("### Raw fields")
    cols = ["ticker","contract_id","expiry_date","days_to_expiry","strike","spot",
            "bid","ask","lastPrice","premium_mid","adjustive_price","pct_to_breakeven",
            "impliedVolatility","openInterest","volume","asof_ny","asof_utc"]
    show = [c for c in cols if c in row.columns]
    st.dataframe(row[show], use_container_width=True, hide_index=True)

    st.markdown("### Coach notes")
    for line in coach_notes(r):
        st.write(f"- {line}")

def render_birdseye(df_latest: pd.DataFrame):
    st.title("👁️ Bird’s Eye — Market Overview")
    st.caption("High-level snapshot across tickers and expiries")

    df = compute_profit_metrics(df_latest, TARGET_DROP)
    if df.empty:
        st.warning("No data available.")
        st.link_button("← Back to Table", "?view=list")
        st.stop()

    last_ts = pd.to_datetime(df.get("asof_ny")).max()
    st.caption(f"Last updated (NY): **{last_ts}**  | Rows: **{len(df):,}**")

    # 1) Open Interest by expiry (stacked by ticker)
    st.subheader("📊 Open Interest by Expiry")
    if "openInterest" in df.columns and "expiry_date" in df.columns and "ticker" in df.columns:
        oi_by = df.groupby(["expiry_date","ticker"])["openInterest"].sum().reset_index()
        if HAS_ALTAIR:
            chart = (alt.Chart(oi_by)
                     .mark_bar()
                     .encode(
                         x=alt.X("expiry_date:N", title="Expiry"),
                         y=alt.Y("openInterest:Q", title="Open Interest"),
                         color=alt.Color("ticker:N", legend=alt.Legend(title="Ticker")),
                         tooltip=["expiry_date","ticker","openInterest"]
                     ))
            st.altair_chart(chart, use_container_width=True)
        else:
            st.bar_chart(oi_by.pivot(index="expiry_date", columns="ticker", values="openInterest").fillna(0))

    # 2) Volume vs OI scatter per ticker
    st.subheader("⚡ Volume vs Open Interest (Sum per Ticker)")
    if "openInterest" in df.columns and "volume" in df.columns and "ticker" in df.columns:
        agg = df.groupby("ticker")[["openInterest","volume"]].sum().reset_index()
        if HAS_ALTAIR:
            c = (alt.Chart(agg)
                 .mark_circle(size=120)
                 .encode(
                     x=alt.X("openInterest:Q", title="Open Interest (sum)"),
                     y=alt.Y("volume:Q", title="Volume (sum)"),
                     color="ticker:N",
                     tooltip=["ticker","openInterest","volume"]
                 ))
            st.altair_chart(c, use_container_width=True)
        else:
            st.dataframe(agg, use_container_width=True)

    # 3) % to breakeven distribution
    st.subheader("🎯 % to Breakeven Distribution")
    if "pct_to_breakeven" in df.columns:
        pct = df["pct_to_breakeven"].dropna().clip(-1, 1)
        if HAS_ALTAIR:
            hist_df = pd.DataFrame({"pct": pct})
            h = (alt.Chart(hist_df)
                 .mark_bar()
                 .encode(
                     x=alt.X("pct:Q", bin=alt.Bin(maxbins=40), title="% to breakeven"),
                     y=alt.Y("count()", title="# contracts"),
                     tooltip=[alt.Tooltip("count()", title="# contracts")]
                 ))
            st.altair_chart(h, use_container_width=True)
        else:
            st.bar_chart(pct)

    st.link_button("← Back to Table", "?view=list")

def render_main(df_latest: pd.DataFrame):
    """YOUR existing main page lives here: put/call toggle, exact/range date, both tables, etc."""
    st.title("Options Finder")

    # Top-right button to Bird’s Eye, keep your current top controls intact
    cols = st.columns([6,1])
    with cols[1]:
        if st.button("👁️ Bird’s Eye", use_container_width=True):
            st.query_params.update({"view": "analytics"})
            st.rerun()

    # Freshness line (optional; keep yours if you already show it)
    if "asof_ny" in df_latest.columns:
        last_ts = pd.to_datetime(df_latest["asof_ny"]).max()
        st.caption(f"Last updated (NY): **{last_ts}**  | Rows: **{len(df_latest):,}**")

    # >>> YOUR EXISTING MAIN UI START
    import os, urllib.parse
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm  # pip install scipy

LATEST_PARQUET = "data/latest/options_latest.parquet"

# ---- constants (tweakable) ----
TARGET_DROP_PUT  = 0.10    # 10% down target for puts
TARGET_UP_CALL   = 0.10    # 10% up target for calls
LOWER_MONEYNESS  = 0.60
UPPER_MONEYNESS  = 1.05
MIN_PREMIUM      = 0.10
MIN_OI           = 200
MIN_VOL          = 10
RISK_FREE        = 0.04    # annual risk-free used in Greeks / ITM prob

st.set_page_config(page_title="Options Finder — By Expiration", layout="wide")

# ---------------- helpers ----------------
@st.cache_data(ttl=30)
def load_latest(path: str) -> pd.DataFrame:
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.read_parquet(path).replace([np.inf,-np.inf], np.nan)
    return df

def compute_profit_metrics(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    if opt_type == "put":
        target = TARGET_DROP_PUT
        df["target_price"] = df["spot"] * (1 - target)
        num = (df["strike"] - df["target_price"] - df["premium_mid"]).clip(lower=0.0)
    else:
        target = TARGET_UP_CALL
        df["target_price"] = df["spot"] * (1 + target)
        num = (df["target_price"] - df["strike"] - df["premium_mid"]).clip(lower=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(df["premium_mid"]>0, num/df["premium_mid"], 0.0)
        pr = np.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0)
    df["profit_ratio_at_target"] = pr
    df["profit_at_target"] = num * 100.0
    df["target_move_pct"] = target
    return df

def add_implied_itm_probability(df: pd.DataFrame, r=RISK_FREE):
    if df.empty or "impliedVolatility" not in df.columns:
        df["implied_itm_prob"]=np.nan; return df
    out = df.copy()
    ok = (out["spot"].gt(0) & out["strike"].gt(0) &
          out["impliedVolatility"].gt(0) & out["days_to_expiry"].gt(0))
    T = out.loc[ok, "days_to_expiry"].astype(float)/365.0
    S = out.loc[ok, "spot"].astype(float)
    K = out.loc[ok, "strike"].astype(float)
    sigma = out.loc[ok, "impliedVolatility"].astype(float)
    d2 = (np.log(S/K) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    # P(ITM at expiry): put ⇒ P(S_T<=K)=Phi(-d2); call ⇒ P(S_T>=K)=Phi(-d2) as well
    prob = norm.cdf(-d2)
    out["implied_itm_prob"] = np.nan
    out.loc[ok, "implied_itm_prob"] = prob
    return out

def bs_greeks_one(spot, strike, T, sigma, r, opt_type):
    """
    Black–Scholes greeks.
    Theta is per calendar day; Vega per +1% IV; Rho per +1% rate.
    """
    if spot<=0 or strike<=0 or T<=0 or sigma<=0:
        return dict(delta=np.nan, gamma=np.nan, theta=np.nan, vega=np.nan, rho=np.nan)

    sqrtT = np.sqrt(T)
    d1 = (np.log(spot/strike) + (r + 0.5*sigma**2)*T) / (sigma*sqrtT)
    d2 = d1 - sigma*sqrtT
    pdf_d1 = norm.pdf(d1)

    gamma = pdf_d1 / (spot * sigma * sqrtT)
    vega  = (spot * pdf_d1 * sqrtT) / 100.0  # per +1% IV

    if opt_type == "call":
        delta = norm.cdf(d1)
        theta = (-(spot*pdf_d1*sigma)/(2*sqrtT) - r*strike*np.exp(-r*T)*norm.cdf(d2)) / 365.0
        rho   = (strike*T*np.exp(-r*T)*norm.cdf(d2)) / 100.0   # per +1% rate
    else:
        delta = -norm.cdf(-d1)
        theta = (-(spot*pdf_d1*sigma)/(2*sqrtT) + r*strike*np.exp(-r*T)*norm.cdf(-d2)) / 365.0
        rho   = (-strike*T*np.exp(-r*T)*norm.cdf(-d2)) / 100.0

    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)

def basic_filters(df: pd.DataFrame, tickers: list[str], opt_type: str) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    df = df[df["type"] == opt_type]
    df = df[df["ticker"].isin(tickers)]
    df = df.dropna(subset=["spot","strike","premium_mid","openInterest","volume","days_to_expiry"])
    lb = df["spot"] * LOWER_MONEYNESS
    ub = df["spot"] * UPPER_MONEYNESS
    df = df[(df["strike"]>=lb) & (df["strike"]<=ub)]
    df = df[(df["premium_mid"]>=MIN_PREMIUM) & (df["openInterest"]>=MIN_OI) & (df["volume"]>=MIN_VOL)]
    return df

def link_for(cid: str) -> str:
    q = urllib.parse.urlencode({"view":"detail","id":cid})
    return f"?{q}"

def pct_fmt(x): return "" if pd.isna(x) else f"{x*100:.2f}%"

# ---------------- routing + data ----------------
df_all = load_latest(LATEST_PARQUET)
if df_all.empty:
    st.error("Latest parquet not found or empty. Run `note/ingest_options.py` first."); st.stop()

params = st.query_params
view = params.get("view", ["list"]); view = view[0] if isinstance(view, list) else view

# ---------------- DETAIL PAGE ----------------
if view == "detail":
    cid = params.get("id", [None]); cid = cid[0] if isinstance(cid, list) else cid
    if not cid:
        st.error("Missing contract id."); st.stop()
    row = df_all[df_all["contract_id"]==cid].head(1)
    if row.empty:
        st.error("Contract not found."); st.link_button("← Back", "?view=list"); st.stop()

    typ = row.iloc[0]["type"]
    df = compute_profit_metrics(row, typ)
    df = add_implied_itm_probability(df)

    # ---- Greeks (computed from row) ----
    r0 = df.iloc[0]
    T = float(r0["days_to_expiry"])/365.0
    sigma = float(r0.get("impliedVolatility", np.nan))
    greeks = bs_greeks_one(float(r0["spot"]), float(r0["strike"]), T, sigma, RISK_FREE, typ)

    st.title("📄 Option Detail")
    st.link_button("← Back to list", "?view=list")
    st.subheader(f"{r0['ticker']} | Exp {r0['expiry_date']} | Strike ${r0['strike']:.2f} | {typ.upper()}")
    st.caption(f"As of (NY): {r0['asof_ny']}  |  Contract ID: {r0['contract_id']}")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Spot", f"${r0['spot']:.2f}")
    c2.metric("Premium", f"${r0['premium_mid']:.3f}")
    c3.metric("Breakeven", f"${r0['adjustive_price']:.2f}")
    c4.metric(f"Profit Ratio @ {int(r0['target_move_pct']*100)}%", f"{r0['profit_ratio_at_target']:.2f}×")

    d1,d2,d3,d4 = st.columns(4)
    d1.metric("Days to Expiry", f"{int(r0['days_to_expiry'])}")
    d2.metric("Open Interest", f"{int(r0['openInterest'])}")
    d3.metric("Volume (today)", f"{int(r0['volume'])}")
    d4.metric("Implied ITM %", f"{pct_fmt(r0.get('implied_itm_prob'))}")

    st.markdown("### Greeks (Black–Scholes)")
    st.write(f"**Delta:** {greeks['delta']:.4f}")
    st.write(f"**Gamma:** {greeks['gamma']:.6f}")
    st.write(f"**Theta (per day):** {greeks['theta']:.4f}")
    st.write(f"**Vega (per +1% IV):** {greeks['vega']:.4f}")
    st.write(f"**Rho (per +1% rate):** {greeks['rho']:.4f}")


    st.markdown("### Raw fields")
    show = ["ticker","type","contract_id","expiry_date","days_to_expiry","strike","spot","bid","ask","lastPrice",
            "premium_mid","adjustive_price","pct_to_breakeven","impliedVolatility","openInterest","volume",
            "profit_ratio_at_target","target_price","asof_ny","asof_utc"]
    st.dataframe(df[[c for c in show if c in df.columns]], hide_index=True, use_container_width=True)
    st.stop()

# ---------------- LIST PAGE ----------------
st.title("Options Finder")

# Controls row (top)
tickers = sorted(df_all["ticker"].dropna().unique().tolist())
colA,colB,colC,colD = st.columns([2,1,1,2])

with colA:
    sel_tickers = st.multiselect("Tickers", options=tickers, default=tickers)

with colB:
    opt_type = st.segmented_control("Option type", options=["put","call"], default="put")

with colC:
    expiry_mode = st.segmented_control("Expiry filter", options=["Range","Exact"], default="Range")

# Build list of available expiries for selected tickers & side
avail_expiries = sorted(df_all[(df_all["type"]==opt_type) & (df_all["ticker"].isin(sel_tickers))]
                        ["expiry_date"].dropna().unique().tolist())

with colD:
    if expiry_mode == "Range":
        max_dte = st.number_input("Expiry within (days)", min_value=1, max_value=60, value=30, step=1)
        exact_expiry = None
    else:
        exact_expiry = st.selectbox("Exact expiry date", options=avail_expiries, index=0 if avail_expiries else None)
        max_dte = None

# Filter base set
df = basic_filters(df_all, sel_tickers, opt_type)

# Apply expiry filter
if expiry_mode == "Range":
    df = df[df["days_to_expiry"].between(1, int(max_dte))]
else:
    if exact_expiry:
        df = df[df["expiry_date"] == exact_expiry]

# Enrich
df = compute_profit_metrics(df, opt_type)
df = add_implied_itm_probability(df)

if df.empty:
    st.info("No contracts match the selection."); st.stop()

# ---- By expiration table (like TradingView) ----
grp = (df.groupby(["expiry_date","days_to_expiry"], as_index=False)
         .agg(
            contracts=("contract_id","count"),
            median_premium=("premium_mid","median"),
            median_itm_prob=("implied_itm_prob","median"),
            min_pct_to_breakeven=("pct_to_breakeven","min"),
         ))
best = (df.sort_values(["expiry_date","profit_ratio_at_target"], ascending=[True,False])
          .groupby("expiry_date", as_index=False)
          .head(1)[["expiry_date","contract_id","profit_ratio_at_target","ticker","strike"]])
grp = grp.merge(best, on="expiry_date", how="left")
grp["Open"] = grp["contract_id"].apply(lambda cid: f"?{urllib.parse.urlencode({'view':'detail','id':cid})}" if pd.notna(cid) else "")

st.subheader(f"By expiration — {opt_type.upper()}s")
st.dataframe(
    grp.loc[:, ["Open","expiry_date","days_to_expiry","contracts","median_premium","median_itm_prob",
                "min_pct_to_breakeven","ticker","strike","profit_ratio_at_target"]]
       .sort_values("days_to_expiry")
       .rename(columns={
           "days_to_expiry":"DTE",
           "contracts":"#contracts",
           "median_premium":"median premium",
           "median_itm_prob":"median implied ITM %",
           "min_pct_to_breakeven":"best (min) % to breakeven",
           "profit_ratio_at_target":"best profit ratio",
       }),
    use_container_width=True, hide_index=True,
    column_config={
        "Open": st.column_config.LinkColumn("Details", display_text="Open"),
        "median premium": st.column_config.NumberColumn(format="$%.3f"),
        "median implied ITM %": st.column_config.NumberColumn(format="%.1f%%"),
        "best (min) % to breakeven": st.column_config.NumberColumn(format="%.2f%%"),
        "best profit ratio": st.column_config.NumberColumn(format="%.2f×"),
        "strike": st.column_config.NumberColumn(format="$%.2f"),
        "DTE": st.column_config.NumberColumn(format="%.0f"),
    }
)

st.divider()

# ---- Top Profit Ratio table (multi-ticker) ----
st.subheader(f"Top Profit Ratio — {opt_type.upper()}s")
df = df.sort_values(["profit_ratio_at_target","openInterest","volume"], ascending=[False,False,False]).copy()
df["Details"] = df["contract_id"].apply(lambda cid: f"?{urllib.parse.urlencode({'view':'detail','id':cid})}")
cols = ["Details","ticker","expiry_date","days_to_expiry","strike","premium_mid","adjustive_price",
        "pct_to_breakeven","openInterest","volume","profit_ratio_at_target","implied_itm_prob"]
st.dataframe(
    df[cols].head(200),
    use_container_width=True, hide_index=True,
    column_config={
        "Details": st.column_config.LinkColumn("Details", display_text="Open"),
        "premium_mid": st.column_config.NumberColumn("premium", format="$%.3f"),
        "adjustive_price": st.column_config.NumberColumn("breakeven", format="$%.2f"),
        "pct_to_breakeven": st.column_config.NumberColumn("% to breakeven", format="%.2f%%"),
        "profit_ratio_at_target": st.column_config.NumberColumn("profit ratio @ target", format="%.2f×"),
        "implied_itm_prob": st.column_config.NumberColumn("implied ITM %", format="%.1f%%"),
        "strike": st.column_config.NumberColumn("strike", format="$%.2f"),
        "openInterest": st.column_config.NumberColumn("OI", format="%.0f"),
        "volume": st.column_config.NumberColumn("Vol", format="%.0f"),
        "days_to_expiry": st.column_config.NumberColumn("DTE", format="%.0f"),
    }
)

    # Paste all your existing controls & tables here:
    # - put/call toggle
    # - exact/range date filters
    # - Top Profit Ratio table
    # - Additional table (e.g., by expiry time)
    #
    # IMPORTANT: to enable click-through to Detail page from your table,
    # add a 'Details' column with URLs like '?view=detail&id=<contract_id>'
    # See snippet below after this function.
    # >>> YOUR EXISTING MAIN UI END

# ----------------- ROUTER -----------------
st.set_page_config(page_title="Options Finder", layout="wide")
df_latest = load_latest(LATEST_PARQUET)
if df_latest.empty:
    st.error("Latest parquet not found or empty.")
    st.stop()

params = st.query_params
view = params.get("view", ["list"])
if isinstance(view, list): view = view[0]

if view == "detail":
    render_detail(df_latest)
elif view == "analytics":
    render_birdseye(df_latest)
else:
    render_main(df_latest)





