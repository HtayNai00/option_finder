#!/usr/bin/env python3
"""
Puts-only ingester (≤30 DTE) with daily snapshots + live latest view.

- Reads tickers from config/tickers.csv (one per line).
- Filters expiries with 1 ≤ days_to_expiry ≤ expiries_max_days (default 30).
- Computes premium_mid (mid of bid/ask; fallback last; clamp min).
- Derives adjustive_price, pct_to_breakeven.
- Writes:
    data/snapshots/YYYY-MM-DD/options_puts.parquet   (append-only daily)
    data/snapshots/YYYY-MM-DD/_ingest_log.csv        (small log)
    data/latest/options_puts_latest.parquet          (overwritten "latest")

You can run this multiple times per day; snapshots append, "latest" upserts to most-recent.
"""

import os
import sys
import csv
import math
import warnings
from datetime import datetime, date
from typing import List, Optional

import pytz
import pandas as pd
import yfinance as yf
from dateutil import tz

# --------- CONFIG DEFAULTS (overridden if files exist) ----------
TICKERS_FILE = "config/tickers.csv"
EXPIRIES_MAX_DAYS = 30           # max DTE to include (inclusive)
EXPIRIES_TO_PULL_MAX = 4         # safety cap if many weekly expiries within DTE
OPTION_TYPE = "put"              # puts-only
PREMIUM_METHOD = "mid"           # "mid" with fallback to last
MIN_PREMIUM = 0.05               # clamp tiny premiums for stable ratios
DEFAULT_TARGET_DROP_PCT = 0.10   # used later in UI (not computed here)
LIQUIDITY_WARN_OI = 100          # for insights later (stored but not used here)
LIQUIDITY_WARN_VOL = 50          # stored but not used here
TZ_DISPLAY = "America/New_York"
SNAPSHOT_ROOT = "data/snapshots"
LATEST_PATH = "data/latest/options_puts_latest.parquet"

# ----------------- UTILITIES -----------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_tickers(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tickers file not found: {path}")
    ticks: List[str] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for line in f:
            sym = line.strip()
            if sym and not sym.startswith("#"):
                ticks.append(sym.upper())
    if not ticks:
        raise ValueError("No tickers found in config/tickers.csv")
    return ticks

def now_times():
    utc = pytz.UTC
    ny_tz = pytz.timezone(TZ_DISPLAY)
    dt_utc = datetime.now(utc)
    dt_ny = dt_utc.astimezone(ny_tz)
    asof_utc = dt_utc.isoformat().replace("+00:00", "Z")
    asof_ny = dt_ny.isoformat()
    asof_date = dt_ny.date()  # snapshot folder keyed by NY date
    asof_bucket = dt_utc.replace(second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    return asof_utc, asof_ny, asof_date, asof_bucket

def dte_from_str(expiry_str: str, ref_date: date) -> int:
    # expiry_str like "2025-09-19"
    try:
        yyyy, mm, dd = expiry_str.split("-")
        exp_date = date(int(yyyy), int(mm), int(dd))
        return (exp_date - ref_date).days
    except Exception:
        return -1

def safe_mid(bid, ask, last) -> Optional[float]:
    """
    Choose premium per contract (per share):
      - If bid & ask present and > 0 => mid
      - elif last present and > 0    => last
      - else None
    """
    try:
        if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask > 0:
            return float((bid + ask) / 2.0)
        if pd.notna(last) and last > 0:
            return float(last)
        return None
    except Exception:
        return None

def clamp_premium(x: float) -> float:
    return max(float(x), MIN_PREMIUM)

def build_contract_id(ticker: str, expiry_date: str, strike: float) -> str:
    return f"{ticker}|{expiry_date}|put|{strike:.2f}"

def to_float(x):
    try:
        if pd.isna(x): return None
        return float(x)
    except Exception:
        return None

# ----------------- CORE INGEST -----------------
def ingest_puts_for_ticker(ticker: str, ref_date: date) -> pd.DataFrame:
    """
    Returns a DataFrame of puts within DTE limit for the ticker.
    Columns standardized per data dictionary.
    """
    tk = yf.Ticker(ticker)

    # Spot (use last close or live ‘currentPrice’ proxy via history)
    try:
        spot_hist = tk.history(period="1d")
        if spot_hist is not None and not spot_hist.empty and "Close" in spot_hist.columns:
            spot = float(spot_hist["Close"].iloc[-1])
        else:
            spot = math.nan
    except Exception:
        spot = math.nan

    # All expiries
    try:
        all_expiries = tk.options or []
    except Exception:
        all_expiries = []

    # Filter expiries by DTE
    rows: List[pd.DataFrame] = []
    valid_expiries = []
    for exp in all_expiries:
        dte = dte_from_str(exp, ref_date)
        if 1 <= dte <= EXPIRIES_MAX_DAYS:
            valid_expiries.append((exp, dte))
    valid_expiries.sort(key=lambda x: x[1])  # nearest first
    if EXPIRIES_TO_PULL_MAX > 0 and len(valid_expiries) > EXPIRIES_TO_PULL_MAX:
        valid_expiries = valid_expiries[:EXPIRIES_TO_PULL_MAX]

    # Pull puts for each valid expiry
    for exp, dte in valid_expiries:
        try:
            oc = tk.option_chain(exp)
            puts = oc.puts.copy()
            if puts is None or puts.empty:
                continue
        except Exception:
            continue

        # Standardize columns
        # Yahoo columns: contractSymbol, lastTradeDate, strike, lastPrice, bid, ask, change,
        # percentChange, volume, openInterest, impliedVolatility, inTheMoney, contractSize, currency
        puts["ticker"] = ticker
        puts["spot"] = spot
        puts["expiry_date"] = exp
        puts["days_to_expiry"] = int(dte)
        puts["type"] = "put"

        # Clean numeric
        for col in ["strike", "lastPrice", "bid", "ask", "impliedVolatility", "volume", "openInterest"]:
            if col in puts.columns:
                puts[col] = puts[col].apply(to_float)

        # Premium selection + derived fields
        premium_list = []
        quote_source = []
        adjustive_price_list = []
        pct_to_breakeven_list = []
        contract_id_list = []
        data_quality_flag = []

        for _, r in puts.iterrows():
            bid = r.get("bid")
            ask = r.get("ask")
            last = r.get("lastPrice")

            premium = safe_mid(bid, ask, last)
            src = "mid" if (premium is not None and pd.notna(bid) and pd.notna(ask) and bid and ask) else \
                  ("last" if premium is not None else "missing")

            if premium is None:
                premium_list.append(None)
                quote_source.append("missing")
                adjustive_price_list.append(None)
                pct_to_breakeven_list.append(None)
                contract_id_list.append(build_contract_id(ticker, exp, float(r["strike"])))
                data_quality_flag.append("missing_quotes")
                continue

            # Clamp
            if premium < MIN_PREMIUM:
                premium = clamp_premium(premium)
                src = "clamped"

            # Derived
            strike = float(r["strike"])
            adj_price = strike - premium
            adj_pct = None
            if pd.notna(spot) and spot > 0:
                adj_pct = (spot - adj_price) / spot

            premium_list.append(premium)
            quote_source.append(src)
            adjustive_price_list.append(adj_price)
            pct_to_breakeven_list.append(adj_pct)
            contract_id_list.append(build_contract_id(ticker, exp, strike))
            data_quality_flag.append("ok")

        puts = puts.assign(
            premium_mid=pd.Series(premium_list, index=puts.index),
            quote_source=pd.Series(quote_source, index=puts.index),
            adjustive_price=pd.Series(adjustive_price_list, index=puts.index),
            pct_to_breakeven=pd.Series(pct_to_breakeven_list, index=puts.index),
            contract_id=pd.Series(contract_id_list, index=puts.index),
            data_quality_flag=pd.Series(data_quality_flag, index=puts.index)
        )

        # Keep only useful columns
        keep = [
            "contract_id", "ticker", "spot", "expiry_date", "days_to_expiry", "type",
            "strike", "bid", "ask", "lastPrice", "impliedVolatility", "volume", "openInterest",
            "premium_mid", "quote_source", "adjustive_price", "pct_to_breakeven"
        ]
        existing = [c for c in keep if c in puts.columns]
        rows.append(puts[existing])

    if not rows:
        return pd.DataFrame(columns=[
            "contract_id","ticker","spot","expiry_date","days_to_expiry","type",
            "strike","bid","ask","lastPrice","impliedVolatility","volume","openInterest",
            "premium_mid","quote_source","adjustive_price","pct_to_breakeven"
        ])

    df = pd.concat(rows, ignore_index=True)
    return df


def upsert_latest(latest_path: str, new_df: pd.DataFrame, asof_utc: str, asof_ny: str, asof_bucket: str) -> pd.DataFrame:
    """
    Upsert logic: keep the row with max(asof_utc) per contract_id.
    Since we don't store asof in new_df yet, attach it here.
    """
    if new_df.empty:
        return new_df

    # Attach timestamps
    new_df = new_df.copy()
    new_df["asof_utc"] = asof_utc
    new_df["asof_ny"] = asof_ny
    new_df["asof_bucket"] = asof_bucket

    # Load existing latest (if any)
    if os.path.exists(latest_path):
        try:
            old = pd.read_parquet(latest_path)
        except Exception:
            warnings.warn("Failed to read existing latest; overwriting.")
            old = pd.DataFrame()
    else:
        old = pd.DataFrame()

    if old is None or old.empty:
        # First-time write
        return new_df

    # Concatenate & keep newest by contract_id
    combined = pd.concat([old, new_df], ignore_index=True)
    # If asof_utc equal, last-in wins; otherwise sort then drop_duplicates
    combined.sort_values(["contract_id", "asof_utc"], inplace=True)
    latest = combined.drop_duplicates(subset=["contract_id"], keep="last")

    return latest


def append_snapshot(snapshot_root: str, asof_date: date, df: pd.DataFrame, asof_utc: str, asof_ny: str, tickers_total:int, tickers_success:int, tickers_skipped:int):
    """
    Writes the daily snapshot parquet (append) and a small ingest log csv.
    """
    day_dir = os.path.join(snapshot_root, asof_date.isoformat())
    ensure_dir(day_dir)

    snap_path = os.path.join(day_dir, "options_puts.parquet")
    log_path = os.path.join(day_dir, "_ingest_log.csv")

    # Append or write parquet
    if df is not None and not df.empty:
        # Attach timestamps to snapshot rows
        snap_df = df.copy()
        if "asof_utc" not in snap_df.columns:
            # If called before upsert_latest, we ensure timestamps here as well
            snap_df["asof_utc"] = asof_utc
            snap_df["asof_ny"] = asof_ny
        if os.path.exists(snap_path):
            prev = pd.read_parquet(snap_path)
            combined = pd.concat([prev, snap_df], ignore_index=True)
            combined.to_parquet(snap_path, index=False)
        else:
            snap_df.to_parquet(snap_path, index=False)

    # Update log
    log_fields = ["asof_utc","asof_ny","tickers_total","tickers_success","tickers_skipped","rows_written"]
    rows_written = 0 if df is None else len(df)
    log_row = {
        "asof_utc": asof_utc,
        "asof_ny": asof_ny,
        "tickers_total": tickers_total,
        "tickers_success": tickers_success,
        "tickers_skipped": tickers_skipped,
        "rows_written": rows_written
    }
    # Append CSV log
    if os.path.exists(log_path):
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writerow(log_row)
    else:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=log_fields)
            writer.writeheader()
            writer.writerow(log_row)


def main():
    # Timestamps
    asof_utc, asof_ny, asof_date, asof_bucket = now_times()
    ref_date = asof_date  # DTE computed relative to NY calendar date

    # Ensure folders
    ensure_dir("data/latest")
    ensure_dir(SNAPSHOT_ROOT)

    # Read tickers
    tickers = read_tickers(TICKERS_FILE)

    all_rows = []
    skipped = 0
    success = 0

    for t in tickers:
        try:
            df_t = ingest_puts_for_ticker(t, ref_date)
            if df_t is None or df_t.empty:
                skipped += 1
                continue
            all_rows.append(df_t)
            success += 1
        except Exception as e:
            warnings.warn(f"Failed ingest for {t}: {e}")
            skipped += 1

    if all_rows:
        df_all = pd.concat(all_rows, ignore_index=True)
    else:
        df_all = pd.DataFrame(columns=[
            "contract_id","ticker","spot","expiry_date","days_to_expiry","type",
            "strike","bid","ask","lastPrice","impliedVolatility","volume","openInterest",
            "premium_mid","quote_source","adjustive_price","pct_to_breakeven"
        ])

    # UPSERT into latest
    latest_df = upsert_latest(LATEST_PATH, df_all, asof_utc, asof_ny, asof_bucket)

    # Write Latest atomically (temp -> move)
    tmp_latest = LATEST_PATH + ".tmp"
    latest_df.to_parquet(tmp_latest, index=False)
    os.replace(tmp_latest, LATEST_PATH)

    # Append daily snapshot + log
    append_snapshot(SNAPSHOT_ROOT, asof_date, latest_df, asof_utc, asof_ny,
                    tickers_total=len(tickers), tickers_success=success, tickers_skipped=skipped)

    # Console summary
    print(f"[OK] asof_ny={asof_ny} tickers_total={len(tickers)} success={success} skipped={skipped} rows_latest={len(latest_df)}")
    print(f"Latest:   {LATEST_PATH}")
    print(f"Snapshot: {os.path.join(SNAPSHOT_ROOT, asof_date.isoformat(), 'options_puts.parquet')}")


if __name__ == "__main__":
    # Optional: allow EXPIRIES_MAX_DAYS override via CLI arg (e.g., python ingest_puts.py 30)
    if len(sys.argv) > 1:
        try:
            EXPIRIES_MAX_DAYS = int(sys.argv[1])
        except Exception:
            pass
    main()


