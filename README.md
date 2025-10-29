# Options Finder 📊

A comprehensive full-stack Python application for analyzing and visualizing stock option chains with a TradingView-inspired interface.

## 🎯 Features

- **🤖 Machine Learning Predictions**: ML-powered ranking of top 10 most profitable options
- **📊 Advanced Options Analysis**: Compute Greeks, ITM probability, profit ratios, and breakeven prices
- **🎨 TradingView-Style UI**: Dark theme with horizontal expiry slider and interactive tables
- **🔌 REST API**: FastAPI backend with CORS-enabled endpoints
- **📈 Real-time Data**: Support for local parquet files and live API integration
- **🏗️ Modular Design**: Clean separation of concerns with core, API, and UI layers
- **🔍 Comprehensive Filtering**: Filter by ticker, expiry, moneyness, liquidity, and more
- **🗑️ Smart Date Filtering**: Automatically removes expired options and shows only upcoming dates

## 🏗️ Project Structure

```
options_finder/
├── app.py                  # Main Streamlit application
├── config.py              # Configuration constants
├── core/                  # Core business logic
│   ├── data_loader.py     # Data loading and validation
│   ├── analytics.py       # Greeks, ITM prob, profit calculations
│   ├── filters.py         # Data filtering functions
│   ├── ml_predictor.py    # Machine learning predictions
│   └── utils.py           # Utility functions
├── api/                   # FastAPI backend
│   ├── server.py          # Main API server
│   └── routes/            # API route modules
├── assets/                # Static assets
│   ├── style.css          # TradingView-inspired dark theme
│   └── favicon.png        # App favicon
├── data/                  # Data storage
│   └── latest/            # Latest options data
├── models/                # ML model storage
├── note/                  # Data ingestion scripts
│   ├── ingest_options.py  # Options data ingestion
│   └── compute_profit_metrics.py # Profit calculations
├── tests/                 # Test suite
│   ├── test_analytics.py  # Analytics tests
│   ├── test_api.py        # API tests
│   └── test_integration.py # Integration tests
├── train_ml_model.py      # ML model training script
├── refresh_data.py        # Data refresh utility
├── validate_data.py       # Data validation script
├── run_app.py             # Application launcher
├── ML_FEATURES.md         # ML features documentation
└── requirements.txt       # Python dependencies
```

## 🤖 Machine Learning Features

The application includes advanced ML capabilities for predicting the most profitable options:

- **20+ Features**: Analyzes Greeks, liquidity, volatility, profit metrics, and market conditions
- **Multiple Algorithms**: Tests Random Forest, XGBoost, Gradient Boosting, and Linear Regression
- **Auto-Selection**: Automatically selects the best performing model
- **Ranked Predictions**: Shows top 10 most profitable options with ML confidence scores
- **Real-time Updates**: Predictions update with latest market data

### ML Commands

```bash
# Train the ML model
python run_app.py train

# Validate the trained model
python train_ml_model.py validate
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd options_finder
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**
   - Place your options data in `data/latest/options_latest.parquet`
   - Or use the existing data files in the `data/` directory

4. **Run the application**

   **Streamlit UI:**
   ```bash
   streamlit run app.py
   ```

   **FastAPI Server:**
   ```bash
   python api/server.py
   ```

## 📊 Data Format

The application expects options data in the following format:

| Column | Type | Description |
|--------|------|-------------|
| ticker | string | Stock ticker symbol |
| type | string | Option type (call/put) |
| expiry_date | string | Expiry date (YYYY-MM-DD) |
| days_to_expiry | int | Days to expiry |
| strike | float | Strike price |
| spot | float | Current stock price |
| bid | float | Bid price |
| ask | float | Ask price |
| premium_mid | float | Mid premium |
| impliedVolatility | float | Implied volatility |
| openInterest | int | Open interest |
| volume | int | Volume |
| asof_ny | string | Timestamp (NY time) |
| asof_utc | string | Timestamp (UTC) |

## 🔧 Configuration

Edit `config.py` to customize:

- Target price movements (default: ±10%)
- Moneyness filters (default: 60%-105% of spot)
- Liquidity thresholds (premium, OI, volume)
- Risk-free rate for Greeks calculations
- API and UI settings

## 🎨 UI Features

### TradingView-Style Interface
- **Dark Theme**: Professional dark color scheme
- **Expiry Slider**: Horizontal scrollable date picker
- **Interactive Tables**: Sortable columns with custom formatting
- **Responsive Design**: Works on desktop and mobile

### Key Views
1. **List View**: Filtered options with summary tables
2. **Detail View**: Individual contract analysis with Greeks
3. **By Expiration**: Grouped view showing best opportunities per expiry

## 🔌 API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `GET /options/list` - Filtered options list
- `GET /options/greeks/{contract_id}` - Greeks for specific contract
- `GET /options/summary` - Summary statistics
- `GET /options/expiries` - Available expiry dates
- `GET /options/tickers` - Available ticker symbols
- `POST /refresh` - Refresh data cache

### Example API Usage
```python
import requests

# Get options list
response = requests.get("http://localhost:8000/options/list?tickers=AAPL&option_type=call")
data = response.json()

# Get Greeks for specific contract
response = requests.get("http://localhost:8000/options/greeks/AAPL_20240119_150.00_call")
greeks = response.json()
```

## 🧮 Analytics Features

### Greeks Calculation
- **Delta**: Price sensitivity to underlying
- **Gamma**: Delta sensitivity to underlying
- **Theta**: Time decay per day
- **Vega**: Volatility sensitivity
- **Rho**: Interest rate sensitivity

### Profit Analysis
- **Profit Ratio**: Target profit / Premium paid
- **Breakeven Price**: Strike ± Premium
- **ITM Probability**: Implied probability of finishing in-the-money

### Filtering Options
- **Moneyness**: Strike within 60%-105% of spot
- **Liquidity**: Minimum premium, OI, and volume
- **Expiry**: Range (1-60 days) or exact date
- **Profit Potential**: Minimum profit ratio threshold

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_analytics.py
pytest tests/test_api.py
pytest tests/test_integration.py

# Run with coverage
pytest --cov=core --cov=api
```

## 🔄 Data Sources

### Supported Sources
- **Local Parquet**: Primary data source
- **Yahoo Finance**: Live data integration (optional)
- **Polygon.io**: Professional data feed (optional)

### Adding New Data Sources
1. Extend `core/data_loader.py`
2. Add new loader functions
3. Update configuration as needed

## 🚀 Deployment

### Local Development
```bash
# Start Streamlit UI
streamlit run app.py --server.port 8501

# Start FastAPI server
python api/server.py
```

### Production Deployment
- Use a production WSGI server (e.g., Gunicorn)
- Configure proper CORS settings
- Set up data refresh mechanisms
- Consider caching strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request




## 🆘 Troubleshooting

### Common Issues

1. **No data available**
   - Check that `data/latest/options_latest.parquet` exists
   - Verify data format matches expected schema

2. **API connection issues**
   - Ensure FastAPI server is running
   - Check CORS settings for cross-origin requests

3. **Missing dependencies**
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility

### Getting Help
- Check the test suite for usage examples
- Review the configuration options in `config.py`
- Examine the core modules for implementation details

## 🔮 Future Enhancements

- [ ] Real-time data streaming
- [ ] Advanced charting capabilities
- [ ] Portfolio analysis features
- [ ] Risk management tools
- [ ] Mobile app development
- [ ] Database integration
- [ ] User authentication
- [ ] Alert system

---

**Built with ❤️ for options traders and analysts**
