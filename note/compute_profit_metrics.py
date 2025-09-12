#!/usr/bin/env python3
"""
Compute Profit Ratio leaderboards (with practical filters).

Usage:
  python note/compute_profit_metrics.py                 # defaults
  python note/compute_profit_metrics.py 0.15 0.60 1.05  # 15% drop, strikes in [60%, 105%] of spot
"""

import os, sys, numpy as np, pandas as pd

LATEST_IN  = "data/latest/options_puts_latest.parquet"
LATEST_OUT = "data/latest/options_puts_enriched.parquet"

def main():
    # --- Params (tweak here or via CLI for moneyness) ---
    target_drop = 0.10
    lower_moneyness = 0.60
    upper_moneyness = 1.05

    # Practical guards (helps remove junk)
    min_premium = 0.10         # drop contracts with premium < $0.10
    min_oi = 200               # require openInterest ≥ 200
    min_vol = 10               # require volume ≥ 10
    max_days_to_expiry = 30    # defensive; should already be true from ingest

    if len(sys.argv) >= 2:
        try: target_drop = float(sys.argv[1])
        except: pass
    if len(sys.argv) >= 4:
        try:
            lower_moneyness = float(sys.argv[2])
            upper_moneyness = float(sys.argv[3])
        except: pass

    print(f"[START] drop={target_drop:.0%} moneyness=[{lower_moneyness:.2f},{upper_moneyness:.2f}] guards: minPrem={min_premium}, minOI={min_oi}, minVol={min_vol}")

    if not os.path.exists(LATEST_IN):
        print(f"[ERR] Missing: {LATEST_IN}"); sys.exit(1)

    df = pd.read_parquet(LATEST_IN)
    print(f"[INFO] Loaded {len(df)} rows")

    # Keep puts, sanity fields
    df = df[df.get("type","put") == "put"].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["spot","strike","premium_mid","openInterest","volume","days_to_expiry"])
    print(f"[INFO] After NA cleanup: {len(df)}")

    # Moneyness window
    lb = df["spot"] * lower_moneyness
    ub = df["spot"] * upper_moneyness
    df = df[(df["strike"] >= lb) & (df["strike"] <= ub)]
    print(f"[INFO] After moneyness filter: {len(df)}")

    # Practical guards
    df = df[(df["premium_mid"] >= min_premium) &
            (df["openInterest"] >= min_oi) &
            (df["volume"] >= min_vol) &
            (df["days_to_expiry"] <= max_days_to_expiry)]
    print(f"[INFO] After liquidity/premium guards: {len(df)}")

    if df.empty:
        print("[WARN] No rows after filters; loosen guards or widen moneyness.")
        sys.exit(0)

    # Compute profit ratio @ target
    df["target_drop_pct"] = target_drop
    df["target_price"] = df["spot"] * (1 - target_drop)
    num = (df["strike"] - df["target_price"] - df["premium_mid"]).clip(lower=0.0)  # per-share
    pr = np.where(df["premium_mid"] > 0, num / df["premium_mid"], 0.0)
    df["profit_ratio_at_target"] = np.nan_to_num(pr, nan=0.0, posinf=0.0, neginf=0.0)
    df["profit_at_target"] = num * 100.0

    # Save enriched
    os.makedirs(os.path.dirname(LATEST_OUT), exist_ok=True)
    df.to_parquet(LATEST_OUT, index=False)
    print(f"[OK] Wrote enriched: {LATEST_OUT} (rows={len(df)})")

    # Leaderboards
    show = ["ticker","expiry_date","days_to_expiry","strike","premium_mid","adjustive_price",
            "pct_to_breakeven","openInterest","volume","profit_ratio_at_target"]

    # Top Profit Ratio
    top = df.sort_values(["profit_ratio_at_target","openInterest","volume"],
                         ascending=[False, False, False])[show].head(20)
    print("\n=== Top Profit Ratio (practical) ===")
    print(top.to_string(index=False, justify="left", max_colwidth=14))

    # Closest Breakeven
    df["abs_breakeven"] = df["pct_to_breakeven"].abs()
    close = df.sort_values(["abs_breakeven","openInterest","volume"],
                           ascending=[True, False, False])[show].head(20)
    print("\n=== Closest Breakeven (practical) ===")
    print(close.to_string(index=False, justify="left", max_colwidth=14))

    print("\n[DONE]")

if __name__ == "__main__":
    main()
