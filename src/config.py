# src/config.py

import os
from dotenv import load_dotenv
load_dotenv()


os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
# --- Data Paths ---
# Create a 'data' folder in your project root and place test_df_features.csv inside it.
DATA_FILE_PATH = 'data/test_df_features.csv'

# --- Binance Historical Data Configuration ---
USE_BINANCE_DATA = True  # Set to False to use test data (DATA_FILE_PATH)
BINANCE_DATA_DIR = 'data/historical'  # Directory for Binance CSV files
BINANCE_SYMBOL = 'BTCUSDT'  # Symbol to fetch from Binance
BINANCE_INTERVAL = '1h'  # Candlestick interval (1m, 5m, 15m, 1h, 4h, 1d)

# -- Model Configuration ---
MODEL_TYPE = 'xgboost' # Type of model to use
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_model.json') # Path to save/load model

# -- Postgres DATABASE Configuration --
# IMPORTANT: When running with Docker Compose, 'localhost' in the bot container refers to the container itself.
# Use the service name 'db' to refer to the PostgreSQL container.
# For local development without Docker, we use SQLite

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./trading_bot.db")
 # Default for local development - SQLite
# For Docker Compose, set in .env: DATABASE_URL=postgresql://testuser:testpass@db:5432/tradingbot_test

# --- Binance API Configuration ---
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
BINANCE_TESTNET = True # Set to False for real trading, but START WITH TRUE
BINANCE_API_URL = "https://api.binance.com"
BINANCE_TESTNET_API_URL = "https://testnet.binance.vision/api"

# --- Live Trading Parameters ---
TRADE_SYMBOL = 'BTCUSDT'
TRADE_INTERVAL = '1h'
TRADE_QUANTITY = 0.001
INITIAL_CANDLES_HISTORY = 100

# --- Technical Indicator Parameters (existing) ---
RSI_WINDOW = 14
BB_WINDOW = 20
BB_WINDOW_DEV = 2
SMA_SHORT_WINDOW = 20
SMA_LONG_WINDOW = 50
ATR_WINDOW = 14

# --- RSI Signal Quantile Thresholds (for dynamic thresholds) ---
RSI_LOWER_QUANTILE = 0.2
RSI_UPPER_QUANTILE = 0.8

# --- Time Series Modeling Configuration ---
TIMESERIES_ROLLING_WINDOW = 50  # Rolling window for GARCH, ARIMA, HMM, Kalman
TIMESERIES_MIN_OBSERVATIONS = 30  # Minimum observations required for model fitting
HMM_N_REGIMES = 3  # Number of HMM regimes (2 or 3)
TSFRESH_TOP_K = 10  # Number of top tsfresh features to keep
TSFRESH_ROLLING_WINDOW = 30  # Window size for tsfresh rolling extraction

# Enable/disable specific time series models (for performance tuning)
# ALL ADVANCED TIME SERIES MODELS ARE DISABLED BY DEFAULT FOR PERFORMANCE
# They are very computationally expensive and cause the bot to hang for 30+ minutes
# GARCH and ARIMA fit models for EVERY SINGLE ROW in the dataset (rolling window)
# For 1000 rows: GARCH = 30+ min, ARIMA = 30+ min, HMM = 5+ min, Kalman = 2+ min
# 
# The bot works perfectly fine with just the basic technical indicators:
# - RSI, Bollinger Bands, SMA, ATR (always available, fast)
# - Lag features, rolling stats, momentum (always available, fast)
# 
# Enable advanced features ONLY if you:
# 1. Have a small dataset (<100 rows), OR
# 2. Have a very fast CPU and can wait 30+ minutes, OR
# 3. Are willing to modify the code to use larger steps instead of every row
ENABLE_GARCH = False    # DISABLED - Extremely slow (30+ min for 1000 rows)
ENABLE_ARIMA = False    # DISABLED - Extremely slow (30+ min for 1000 rows)  
ENABLE_HMM = False      # DISABLED - Slow (5+ min for 1000 rows)
ENABLE_KALMAN = False   # DISABLED - Moderate (2+ min for 1000 rows)
ENABLE_TSFRESH = False  # DISABLED - Extremely slow (30+ min for 1000 rows)

# --- Model Parameters (existing) ---
TARGET_COLUMN = 'signal'
TEST_SIZE = 0.2
RANDOM_STATE = 42
CONFIDENCE_THRESHOLD = 0.30

# --- Backtesting Parameters (existing) ---
INITIAL_BALANCE = 10000
TRANSACTION_FEE_PCT = 0.001

# --- Feature Columns (for UI - original features only) ---
# Streamlit UI uses these for manual input
FEATURE_COLUMNS = [
    'rsi', 'bb_upper', 'bb_lower', 'bb_mid', 'bb_pct_b',
    'sma_20', 'sma_50', 'ma_cross', 'price_momentum',
    'atr', 'atr_pct'
]

# --- All Features for Model Training (includes time series) ---
# Used internally for model training and predictions
ALL_FEATURE_COLUMNS = [
    # Technical indicators (user-visible)
    'rsi', 'bb_upper', 'bb_lower', 'bb_mid', 'bb_pct_b',
    'sma_20', 'sma_50', 'ma_cross', 'price_momentum',
    'atr', 'atr_pct',
    # Time series features (auto-calculated, not user input)
    'close_lag_1', 'close_lag_2', 'close_lag_3',
    'close_rolling_mean_7', 'close_rolling_std_7',
    'close_rolling_min_7', 'close_rolling_max_7',
    'close_position_in_range',
    'volume_lag_1', 'volume_rolling_mean_7',
    'price_velocity_3', 'price_acceleration',
    # Advanced time series modeling features
    'garch_volatility', 'garch_std_residuals',
    'arima_forecast', 'arima_forecast_error', 'arima_conf_interval_width',
    'hmm_regime', 'hmm_regime_0_prob', 'hmm_regime_1_prob', 'hmm_regime_2_prob',
    'kalman_level', 'kalman_slope', 'kalman_prediction_error',
    # Note: tsfresh features are added dynamically and will be included at runtime
]

# --- Email Notification Configuration ---
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", 587))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

# --- Telegram Notification Configuration (existing) ---
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'dummy_token_for_testing')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'dummy_chat_id_for_testing')

# --- JWT Configuration (for future web UI, not currently used by bot core) ---
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY')