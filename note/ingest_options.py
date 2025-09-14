#!/usr/bin/env python3
"""
Options ingester (calls + puts, ≤30 DTE) with daily snapshots + live latest.

Writes:
  data/latest/options_latest.parquet           (overwritten; most recent per contract_id)
  data/snapshots/YYYY-MM-DD/options.parquet    (append-only events)
  data/snapshots/YYYY-MM-DD/_ingest_log.csv    (small run log)

Requires: config/tickers.csv  (one symbol per line)
"""

import os, sys, csv, math, warnings
from datetime import datetime, date
from typing import List, Optional
import pytz
import pandas as pd
import yfinance as yf

# ---------------- config ----------------
TICKERS_FILE = "config/tickers.csv"
EXPIRIES_MAX_DAYS = 30
EXPIRIES_TO_PULL_MAX = 4
MIN_PREMIUM = 0.05
TZ_DISPLAY = "America/New_York"
SNAPSHOT_ROOT = "data/snapshots"
LATEST_PATH = "data/latest/options_latest.parquet"

# -------------- utils -------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def read_tickers(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tickers file not found: {path}")
    xs = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if s and not s.startswith("#"):
                xs.append(s.upper())
    if not xs: raise ValueError("No tickers in config/tickers.csv")
    return xs

def now_times():
    utc = pytz.UTC
    ny = pytz.timezone(TZ_DISPLAY)
    dt_utc = datetime.now(utc)
    dt_ny  = dt_utc.astimezone(ny)
    return (
        dt_utc.isoformat().replace("+00:00", "Z"),
        dt_ny.isoformat(),
        dt_ny.date(),
        dt_utc.replace(second=0, microsecond=0).isoformat().replace("+00:00", "Z"),
    )

def dte(exp_str: str, ref: date) -> int:
    y,m,d = [int(x) for x in exp_str.split("-")]
    return (date(y,m,d) - ref).days

def safe_mid(bid, ask, last) -> Optional[float]:
    try:
        if pd.notna(bid) and pd.notna(ask) and bid>0 and ask>0:
            return float((bid+ask)/2)
        if pd.notna(last) and last>0:
            return float(last)
        return None
    except Exception:
        return None

def clamp(x: float) -> float:
    return max(float(x), MIN_PREMIUM)

def to_float(x):
    try:
        if pd.isna(x): return None
        return float(x)
    except Exception:
        return None

def contract_id(ticker: str, expiry: str, opt_type: str, strike: float) -> str:
    return f"{ticker}|{expiry}|{opt_type}|{strike:.2f}"

# -------- core: per ticker ----------
def _pull_side(df: pd.DataFrame, ticker: str, spot: float, expiry: str, dte_i: int, opt_type: str) -> pd.DataFrame:
    if df is None or df.empty: 
        return pd.DataFrame()
    df = df.copy()

    # normalize numeric columns
    for c in ["strike","lastPrice","bid","ask","impliedVolatility","volume","openInterest"]:
        if c in df.columns:
            df[c] = df[c].apply(to_float)

    # compute premium + breakeven (adjustive_price) + distance
    prem, src, adj, pct, cid, flag = [],[],[],[],[],[]
    for _, r in df.iterrows():
        p = safe_mid(r.get("bid"), r.get("ask"), r.get("lastPrice"))
        s = "missing"
        if p is not None:
            s = "mid" if (pd.notna(r.get("bid")) and pd.notna(r.get("ask")) and r["bid"] and r["ask"]) else "last"
            if p < MIN_PREMIUM:
                p = clamp(p); s = "clamped"
        if p is None:
            prem.append(None); src.append("missing"); adj.append(None); pct.append(None)
            cid.append(contract_id(ticker, expiry, opt_type, float(r["strike"])))
            flag.append("missing_quotes")
            continue

        k = float(r["strike"])
        if opt_type == "put":
            breakeven = k - p
            pct_dist = (spot - breakeven)/spot if (pd.notna(spot) and spot>0) else None
        else: # call
            breakeven = k + p
            pct_dist = (breakeven - spot)/spot if (pd.notna(spot) and spot>0) else None

        prem.append(p); src.append(s); adj.append(breakeven); pct.append(pct_dist)
        cid.append(contract_id(ticker, expiry, opt_type, k)); flag.append("ok")

    keep = [
      "contract_id","ticker","spot","expiry_date","days_to_expiry","type",
      "strike","bid","ask","lastPrice","impliedVolatility","volume","openInterest",
      "premium_mid","quote_source","adjustive_price","pct_to_breakeven"
    ]
    out = df.assign(
        ticker=ticker, spot=spot, expiry_date=expiry, days_to_expiry=int(dte_i), type=opt_type,
        premium_mid=pd.Series(prem, index=df.index),
        quote_source=pd.Series(src, index=df.index),
        adjustive_price=pd.Series(adj, index=df.index),
        pct_to_breakeven=pd.Series(pct, index=df.index),
        contract_id=pd.Series(cid, index=df.index),
        data_quality_flag=pd.Series(flag, index=df.index)
    )
    return out[[c for c in keep if c in out.columns]]

def ingest_ticker(ticker: str, ref_date: date) -> pd.DataFrame:
    tk = yf.Ticker(ticker)
    # spot
    try:
        h = tk.history(period="1d")
        spot = float(h["Close"].iloc[-1]) if (h is not None and not h.empty) else math.nan
    except Exception:
        spot = math.nan

    # expiries
    try:
        exps = tk.options or []
    except Exception:
        exps = []
    valid = []
    for e in exps:
        d = dte(e, ref_date)
        if 1 <= d <= EXPIRIES_MAX_DAYS:
            valid.append((e,d))
    valid.sort(key=lambda x: x[1])
    if EXPIRIES_TO_PULL_MAX>0 and len(valid)>EXPIRIES_TO_PULL_MAX:
        valid = valid[:EXPIRIES_TO_PULL_MAX]

    rows = []
    for e, d in valid:
        try:
            oc = tk.option_chain(e)
        except Exception:
            continue
        puts = getattr(oc, "puts", None)
        calls = getattr(oc, "calls", None)
        if puts is not None and not puts.empty:
            rows.append(_pull_side(puts, ticker, spot, e, d, "put"))
        if calls is not None and not calls.empty:
            rows.append(_pull_side(calls, ticker, spot, e, d, "call"))

    if not rows:
        return pd.DataFrame(columns=[
            "contract_id","ticker","spot","expiry_date","days_to_expiry","type",
            "strike","bid","ask","lastPrice","impliedVolatility","volume","openInterest",
            "premium_mid","quote_source","adjustive_price","pct_to_breakeven"
        ])
    return pd.concat(rows, ignore_index=True)

def upsert_latest(latest_path: str, new_df: pd.DataFrame, asof_utc: str, asof_ny: str, asof_bucket: str) -> pd.DataFrame:
    if new_df.empty: return new_df.copy()
    new_df = new_df.copy()
    new_df["asof_utc"] = asof_utc
    new_df["asof_ny"]  = asof_ny
    new_df["asof_bucket"] = asof_bucket

    if os.path.exists(latest_path):
        try:
            old = pd.read_parquet(latest_path)
        except Exception:
            old = pd.DataFrame()
    else:
        old = pd.DataFrame()

    if old is None or old.empty:
        return new_df

    comb = pd.concat([old, new_df], ignore_index=True)
    comb.sort_values(["contract_id","asof_utc"], inplace=True)
    latest = comb.drop_duplicates(subset=["contract_id"], keep="last")
    return latest

def append_snapshot(snapshot_root: str, asof_date: date, df: pd.DataFrame,
                    asof_utc: str, asof_ny: str,
                    tickers_total:int, tickers_success:int, tickers_skipped:int):
    day_dir = os.path.join(snapshot_root, asof_date.isoformat())
    ensure_dir(day_dir)
    snap_path = os.path.join(day_dir, "options.parquet")
    log_path  = os.path.join(day_dir, "_ingest_log.csv")

    if df is not None and not df.empty:
        snap_df = df.copy()
        if "asof_utc" not in snap_df.columns:
            snap_df["asof_utc"] = asof_utc
            snap_df["asof_ny"]  = asof_ny
        if os.path.exists(snap_path):
            prev = pd.read_parquet(snap_path)
            pd.concat([prev, snap_df], ignore_index=True).to_parquet(snap_path, index=False)
        else:
            snap_df.to_parquet(snap_path, index=False)

    fields = ["asof_utc","asof_ny","tickers_total","tickers_success","tickers_skipped","rows_written"]
    rows_written = 0 if df is None else len(df)
    row = dict(asof_utc=asof_utc, asof_ny=asof_ny,
               tickers_total=tickers_total, tickers_success=tickers_success,
               tickers_skipped=tickers_skipped, rows_written=rows_written)
    if os.path.exists(log_path):
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
    else:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerow(row)

def main():
    asof_utc, asof_ny, asof_date, asof_bucket = now_times()
    ensure_dir("data/latest"); ensure_dir(SNAPSHOT_ROOT)
    tickers = read_tickers(TICKERS_FILE)

    all_rows = []; skipped=0; success=0
    for t in tickers:
        try:
            df_t = ingest_ticker(t, asof_date)
            if df_t is None or df_t.empty:
                skipped += 1; continue
            all_rows.append(df_t); success += 1
        except Exception as e:
            warnings.warn(f"Failed ingest for {t}: {e}")
            skipped += 1

    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    latest_df = upsert_latest(LATEST_PATH, df_all, asof_utc, asof_ny, asof_bucket)
    tmp = LATEST_PATH + ".tmp"
    latest_df.to_parquet(tmp, index=False); os.replace(tmp, LATEST_PATH)

    append_snapshot(SNAPSHOT_ROOT, asof_date, latest_df, asof_utc, asof_ny,
                    tickers_total=len(tickers), tickers_success=success, tickers_skipped=skipped)

    print(f"[OK] asof_ny={asof_ny} tickers_total={len(tickers)} success={success} skipped={skipped} rows_latest={len(latest_df)}")
    print(f"Latest:   {LATEST_PATH}")
    print(f"Snapshot: {os.path.join(SNAPSHOT_ROOT, asof_date.isoformat(), 'options.parquet')}")

if __name__ == "__main__":
    if len(sys.argv)>1:
        try: EXPIRIES_MAX_DAYS=int(sys.argv[1])
        except: pass
    main()
