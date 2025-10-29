"""
Configuration constants for Options Finder project.
All configurable parameters are centralized here.
"""

# Target price movements for profit calculations
TARGET_DROP_PUT = 0.10    # 10% down target for puts
TARGET_UP_CALL = 0.10     # 10% up target for calls

# Moneyness filters (strike price relative to spot)
LOWER_MONEYNESS = 0.60    # Minimum strike as % of spot
UPPER_MONEYNESS = 1.05    # Maximum strike as % of spot

# Liquidity filters
MIN_PREMIUM = 0.10        # Minimum premium required
MIN_OI = 200             # Minimum open interest
MIN_VOL = 10             # Minimum volume

# Financial parameters
RISK_FREE = 0.04         # Annual risk-free rate for Greeks/ITM probability

# Data paths
LATEST_PARQUET = "data/latest/options_latest.parquet"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Options Finder API"
API_DESCRIPTION = "REST API for options chain analysis and metrics"

# Streamlit configuration
STREAMLIT_PAGE_TITLE = "Options Finder — By Expiration"
STREAMLIT_LAYOUT = "wide"

# Cache settings
CACHE_TTL = 30  # seconds for data cache

# Default filters
DEFAULT_MAX_DTE = 30     # Default days to expiry filter
DEFAULT_OPTION_TYPE = "put"  # Default option type

