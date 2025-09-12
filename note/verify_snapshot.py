import pandas as pd

LATEST_PATH = "data/latest/options_puts_latest.parquet"

df = pd.read_parquet(LATEST_PATH)

print("\n--- Sample rows ---")
print(df.head(10))

print("\n--- Column names ---")
print(df.columns.tolist())

# Simple consistency check: adjustive_price = strike - premium_mid
df["calc_adjustive"] = df["strike"] - df["premium_mid"]
df["diff"] = df["adjustive_price"] - df["calc_adjustive"]

print("\n--- Adjustive price check (first 5 rows) ---")
print(df[["ticker", "strike", "premium_mid", "adjustive_price", "calc_adjustive", "diff"]].head())

print("\n--- Unique tickers captured ---")
print(df["ticker"].unique())

print("\n--- Expiry dates (≤30D) ---")
print(df["expiry_date"].unique())
