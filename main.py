# main.py

import logging
import warnings
import os
import time
import pandas as pd
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
import requests
from datetime import datetime
from sklearn.exceptions import UndefinedMetricWarning
from binance.enums import SIDE_BUY, SIDE_SELL

# Import only the config constants actually used in this file
from src.config import (
    FEATURE_COLUMNS, ALL_FEATURE_COLUMNS, MODEL_SAVE_PATH, CONFIDENCE_THRESHOLD, TARGET_COLUMN,
    DATA_FILE_PATH, TRADE_SYMBOL, TRADE_INTERVAL, TRADE_QUANTITY,
    INITIAL_CANDLES_HISTORY, TRANSACTION_FEE_PCT,
    USE_BINANCE_DATA  # New config for toggling data source
)

# Import only the functions/classes used directly in main.py
from src.data_loader import load_and_preprocess_data, load_historical_data  # Added load_historical_data
from src.feature_engineer import (
    calculate_technical_indicators, 
    add_time_series_features, 
    add_advanced_timeseries_features,
    add_derivatives_features,  # New function for derivatives features
    get_rsi_quantile_thresholds, 
    apply_rsi_labels, 
    normalize_features
)
from src.model_manager import prepare_model_data, train_xgboost_model, make_predictions, load_trained_model
from src.backtester import backtest_strategy
from src.visualizer import visualize_trading_results
from src.binance_manager import BinanceManager
from src.db import init_db, SessionLocal, Trade
from src.notifier import TelegramNotifier

# Load environment variables from .env file
load_dotenv()

# Configure logging and warnings globally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
warnings.filterwarnings('ignore', category=UserWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

init_db()

# Initialize Telegram notifier
telegram_notifier = TelegramNotifier()

# Check if advanced time series libraries are available
try:
    from src import timeseries_models
    available_models = []
    if hasattr(timeseries_models, 'ARCH_AVAILABLE') and timeseries_models.ARCH_AVAILABLE:
        available_models.append("GARCH")
    if hasattr(timeseries_models, 'STATSMODELS_AVAILABLE') and timeseries_models.STATSMODELS_AVAILABLE:
        available_models.append("ARIMA")
    if hasattr(timeseries_models, 'HMMLEARN_AVAILABLE') and timeseries_models.HMMLEARN_AVAILABLE:
        available_models.append("HMM")
    if hasattr(timeseries_models, 'PYKALMAN_AVAILABLE') and timeseries_models.PYKALMAN_AVAILABLE:
        available_models.append("Kalman")
    if hasattr(timeseries_models, 'TSFRESH_AVAILABLE') and timeseries_models.TSFRESH_AVAILABLE:
        available_models.append("tsfresh")
    
    if available_models:
        print(f"✅ Advanced time series models available: {', '.join(available_models)}")
    else:
        print("⚠️  WARNING: No advanced time series libraries installed!")
        print("   Bot will work with basic features only.")
        print("   To install: run install_timeseries_deps_v2.bat")
        print("   Or see TROUBLESHOOTING_IMPORTS.md for help")
except Exception as e:
    logging.debug(f"Could not check time series models: {e}")

print("✅ Setup Complete: All required libraries imported and configuration set.")

# Global variables for live trading state
live_candles_history = pd.DataFrame()
current_position_live = 0  # 0: no position, 1: long position

def run_backtesting_pipeline(train_new_model=True):
    """
    Executes the complete backtesting pipeline using historical data.
    Supports both Binance historical data and test data based on USE_BINANCE_DATA config.
    """
    logging.info("Starting the backtesting pipeline...")
    try:
        # Load data based on configuration
        if USE_BINANCE_DATA:
            logging.info("Loading Binance historical data...")
            data_dict = load_historical_data()
            
            # Use OHLCV data as primary dataset
            raw_price_data = data_dict['ohlcv']
            
            if raw_price_data.empty:
                logging.warning("Binance data is empty, falling back to test data...")
                raw_price_data = load_and_preprocess_data(DATA_FILE_PATH)
            else:
                logging.info(f"✅ Loaded {len(raw_price_data)} rows from Binance historical data")
                
                # Store derivatives data for later use
                funding_data = data_dict.get('funding')
                open_interest_data = data_dict.get('open_interest')
                liquidations_data = data_dict.get('liquidations')
        else:
            logging.info("Loading test data from CSV...")
            raw_price_data = load_and_preprocess_data(DATA_FILE_PATH)
            funding_data = None
            open_interest_data = None
            liquidations_data = None
        
        if raw_price_data.empty:
            raise ValueError("Initial data loading resulted in an empty DataFrame. Cannot proceed.")

        df_for_features_and_signals = raw_price_data.copy()
        df_for_features_and_signals = calculate_technical_indicators(df_for_features_and_signals)
        df_for_features_and_signals = add_time_series_features(df_for_features_and_signals)
        logging.info('✅ Technical indicators and time series features calculated')
        
        # Add derivatives features if Binance data is available
        if USE_BINANCE_DATA and funding_data is not None:
            try:
                df_for_features_and_signals = add_derivatives_features(
                    df_for_features_and_signals,
                    funding_data,
                    open_interest_data,
                    liquidations_data
                )
                logging.info('✅ Derivatives features (funding rate, OI, liquidations) calculated')
            except Exception as e:
                logging.warning(f"Could not add derivatives features: {e}. Continuing without them.")
        
        # Add advanced time series modeling features (GARCH, ARIMA, HMM, Kalman, tsfresh)
        df_for_features_and_signals = add_advanced_timeseries_features(df_for_features_and_signals)
        logging.info('✅ Advanced time series modeling features calculated')
        
        # Get all numeric features excluding OHLCV and target columns
        # This dynamically adapts to which time series libraries are available
        exclude_cols = ['signal', 'open', 'high', 'low', 'close', 'volume', 'volume_pct_change']
        current_features = [col for col in df_for_features_and_signals.columns 
                           if col not in exclude_cols
                           and df_for_features_and_signals[col].dtype in ['float64', 'float32', 'int64', 'int32']]
        
        # Update ALL_FEATURE_COLUMNS dynamically based on actual generated features
        from src import config
        config.ALL_FEATURE_COLUMNS = current_features
        ALL_FEATURE_COLUMNS = current_features
        
        logging.info(f'Total features detected after feature engineering: {len(ALL_FEATURE_COLUMNS)}')
        logging.info(f'Features: {sorted(ALL_FEATURE_COLUMNS)}')
        
        # Validate required base features are present (RSI and Bollinger Bands are critical)
        required_base_features = ['rsi', 'bb_upper', 'bb_lower', 'bb_mid', 'sma_20', 'sma_50']
        missing_base_features = [col for col in required_base_features if col not in current_features]
        if missing_base_features:
            logging.error(f"Required base features missing: {missing_base_features}")
            raise ValueError(f"Required base features missing: {missing_base_features}")
        
        lower_thresh, upper_thresh = get_rsi_quantile_thresholds(df_for_features_and_signals['rsi'])
        df_for_features_and_signals = apply_rsi_labels(df_for_features_and_signals, lower_threshold=lower_thresh, upper_threshold=upper_thresh)
        logging.info('✅ Trading signals generated')
        
        df_normalized = normalize_features(df_for_features_and_signals.copy())
        logging.info('✅ Features normalized')
        
        # Final feature validation: ensure all features exist after normalization
        # This handles edge cases where normalization might drop NaN columns
        available_features = [col for col in ALL_FEATURE_COLUMNS if col in df_normalized.columns]
        if len(available_features) < len(ALL_FEATURE_COLUMNS):
            dropped = set(ALL_FEATURE_COLUMNS) - set(available_features)
            logging.warning(f"Some features were dropped during normalization: {dropped}")
            logging.info(f"Using {len(available_features)} features for model training.")
        
        # Use available features for model training
        FINAL_FEATURE_COLUMNS = available_features
        logging.info(f"Final feature count: {len(FINAL_FEATURE_COLUMNS)}")
        logging.info(f"Final features: {sorted(FINAL_FEATURE_COLUMNS)}")
        
        # FIX: Ensure the variable returned matches the one defined
        df_original_aligned = raw_price_data.loc[df_normalized.index]
        
        model = None
        if train_new_model:
            X_train, X_test, y_train, y_test = prepare_model_data(
                df_normalized, feature_cols=FINAL_FEATURE_COLUMNS, target_col=TARGET_COLUMN
            )
            model, _ = train_xgboost_model(X_train, y_train, X_test, y_test)
            logging.info('✅ Model trained and saved.')
            # Update FINAL_FEATURE_COLUMNS to match what was used in training
            FINAL_FEATURE_COLUMNS = X_train.columns.tolist()
        else:
            try:
                model_result = load_trained_model()
                if isinstance(model_result, tuple):
                    model, saved_features = model_result
                    if saved_features:
                        # Use saved feature columns from model training
                        FINAL_FEATURE_COLUMNS = saved_features
                        logging.info(f'✅ Existing model loaded with {len(saved_features)} saved features.')
                    else:
                        logging.info('✅ Existing model loaded (no saved features, using current features).')
                else:
                    model = model_result
                    logging.info('✅ Existing model loaded.')
            except Exception as e:
                logging.warning(f"Could not load existing model: {e}. Training a new model instead.")
                X_train, X_test, y_train, y_test = prepare_model_data(
                    df_normalized, feature_cols=FINAL_FEATURE_COLUMNS, target_col=TARGET_COLUMN
                )
                model, _ = train_xgboost_model(X_train, y_train, X_test, y_test)
                logging.info('✅ New model trained and saved (fallback).')
                # Update FINAL_FEATURE_COLUMNS to match what was used in training
                FINAL_FEATURE_COLUMNS = X_train.columns.tolist()

        if model is None:
            raise RuntimeError("Failed to train or load a model.")
        
        # Final safety check: ensure FINAL_FEATURE_COLUMNS is set and matches data
        if not FINAL_FEATURE_COLUMNS:
            raise RuntimeError("FINAL_FEATURE_COLUMNS is empty. Feature engineering failed.")
        
        missing_in_data = [col for col in FINAL_FEATURE_COLUMNS if col not in df_normalized.columns]
        if missing_in_data:
            raise RuntimeError(f"Features expected by model are missing from data: {missing_in_data}")
        
        logging.info(f"Using {len(FINAL_FEATURE_COLUMNS)} features for prediction")

        X_full_for_predictions = df_normalized[FINAL_FEATURE_COLUMNS]
        predictions, confidences = make_predictions(model, X_full_for_predictions, confidence_threshold=CONFIDENCE_THRESHOLD)
        logging.info('✅ Predictions generated for backtesting.')

        trades_df, daily_portfolio_df = backtest_strategy(df_original_aligned, predictions)
        logging.info('✅ Backtesting completed.')
        
        df_viz_data = df_for_features_and_signals.loc[predictions.index]

        if not daily_portfolio_df.empty:
            visualize_trading_results(df_viz_data, trades_df, daily_portfolio_df, lower_thresh, upper_thresh)
            logging.info('✅ Visualization displayed.')
        else:
            logging.warning('No daily portfolio data generated for visualization (possibly no trades).')

        logging.info('🚀 Backtesting pipeline completed successfully!')
        return df_original_aligned, model, trades_df, daily_portfolio_df, FINAL_FEATURE_COLUMNS 
        
    except Exception as e:
        logging.critical(f'❌ Error executing backtesting pipeline: {str(e)}', exc_info=True)
        raise


def run_live_trade_loop(model, feature_columns):
    """
    Executes the live trading loop.
    Fetches live data, calculates indicators, makes predictions, and places trades.
    
    Args:
        model: Trained ML model
        feature_columns: List of feature column names used during training
    """
    global live_candles_history, current_position_live

    
    logging.info("Starting live trading loop...")
    binance_manager = BinanceManager()

    # Initial data fetch for indicators
    logging.info(f"Fetching initial {INITIAL_CANDLES_HISTORY} candles for {TRADE_SYMBOL} ({TRADE_INTERVAL})...")
    live_candles_history = binance_manager.get_latest_ohlcv_candles(
        symbol=TRADE_SYMBOL, interval=TRADE_INTERVAL, limit=INITIAL_CANDLES_HISTORY
    )
    if live_candles_history.empty:
        logging.critical("Failed to fetch initial historical data for live trading. Exiting.")
        return

    # Check initial balance
    quote_asset = TRADE_SYMBOL[len(TRADE_SYMBOL) // 2:] # e.g., USDT from BTCUSDT
    base_asset = TRADE_SYMBOL[:len(TRADE_SYMBOL) // 2] # e.g., BTC from BTCUSDT
    
    usdt_balance = binance_manager.get_account_balance(asset=quote_asset)
    crypto_balance = binance_manager.get_account_balance(asset=base_asset)
    
    logging.info(f"Initial live balances: {quote_asset}: {usdt_balance}, {base_asset}: {crypto_balance}")

    # Determine initial position (simple: if we have crypto, assume long)
    if crypto_balance > TRADE_QUANTITY: # If we have more than a trade quantity of crypto
        current_position_live = 1 # Assume we are currently in a long position
        logging.info(f"Detected existing crypto balance ({crypto_balance} {base_asset}). Assuming initial long position.")
    else:
        current_position_live = 0 # No position

    # Determine sleep duration based on interval (e.g., 1h interval -> check every hour)
    interval_minutes = binance_manager._interval_to_minutes(TRADE_INTERVAL)
    sleep_seconds = interval_minutes * 60 # Check every interval period
    
    sleep_seconds_with_buffer = sleep_seconds + 5 # 5 second buffer

    logging.info(f"Live trading for {TRADE_SYMBOL} at {TRADE_INTERVAL} interval. Checking every {sleep_seconds_with_buffer} seconds.")
    logging.info(f"Starting with position: {current_position_live} (0=None, 1=Long)")

    while True:
        try:
            latest_candle_df = binance_manager.get_latest_ohlcv_candles(
                symbol=TRADE_SYMBOL, interval=TRADE_INTERVAL, limit=1
            )
            if latest_candle_df.empty:
                logging.warning("No new candle data fetched. Skipping this iteration.")
                time.sleep(sleep_seconds_with_buffer)
                continue

            if not live_candles_history.empty and latest_candle_df.index[-1] <= live_candles_history.index[-1]:
                logging.debug("Latest candle is not new or already processed. Waiting for next.")
                time.sleep(sleep_seconds_with_buffer)
                continue
            
            live_candles_history = pd.concat([live_candles_history, latest_candle_df])
            live_candles_history = live_candles_history.iloc[-INITIAL_CANDLES_HISTORY*2:] 
            
            df_for_features_live = live_candles_history.copy() 
            df_for_features_live = calculate_technical_indicators(df_for_features_live)
            df_for_features_live = add_time_series_features(df_for_features_live)
            
            # Add advanced time series modeling features
            df_for_features_live = add_advanced_timeseries_features(df_for_features_live)
            
            if df_for_features_live.empty:
                logging.warning("Not enough history to calculate indicators for live trading. Waiting for more data.")
                time.sleep(sleep_seconds_with_buffer)
                continue
            
            latest_data_point = df_for_features_live.iloc[[-1]] 

            normalized_latest_data = normalize_features(latest_data_point)
            
            # Use only the features that were used during training
            # This ensures consistency between training and prediction
            missing_features = [col for col in feature_columns if col not in normalized_latest_data.columns]
            if missing_features:
                logging.error(f"Critical: Features missing in live data: {missing_features}")
                logging.error(f"Available features: {normalized_latest_data.columns.tolist()}")
                logging.error("Skipping prediction for this iteration. Check feature engineering pipeline.")
                time.sleep(sleep_seconds_with_buffer)
                continue
            
            # Extract only the features used during training, in the same order
            X_live = normalized_latest_data[feature_columns]
            live_prediction, live_confidence = make_predictions(model, X_live, CONFIDENCE_THRESHOLD)
            live_signal = live_prediction.iloc[0]
            
            logging.info(f"Live Signal for {TRADE_SYMBOL} on {latest_data_point.index[-1].strftime('%Y-%m-%d %H:%M')}: {live_signal} (Confidence: {live_confidence.iloc[0]:.2f})")

            current_usdt_balance = binance_manager.get_account_balance(asset=quote_asset)
            current_crypto_balance = binance_manager.get_account_balance(asset=base_asset)
            
            current_price = latest_data_point['close'].iloc[0]

            if live_signal == 1 and current_position_live == 0:
                cost_with_fee = TRADE_QUANTITY * current_price * (1 + TRANSACTION_FEE_PCT)
                if current_usdt_balance >= cost_with_fee:
                    order = binance_manager.place_market_order(TRADE_SYMBOL, TRADE_QUANTITY, SIDE_BUY)
                    if order:
                        current_position_live = 1
                        logging.info(f"✅ LIVE BUY ORDER placed: {TRADE_QUANTITY} {base_asset} @ {current_price}")
                        record_trade(TRADE_SYMBOL, "BUY", TRADE_QUANTITY, current_price)
                else:
                    logging.warning(f"BUY signal but insufficient {quote_asset} balance ({current_usdt_balance:.2f} needed {cost_with_fee:.2f}).")
            
            elif live_signal == -1 and current_position_live == 1:
                if current_crypto_balance >= TRADE_QUANTITY:
                    order = binance_manager.place_market_order(TRADE_SYMBOL, TRADE_QUANTITY, SIDE_SELL)
                    if order:
                        current_position_live = 0
                        logging.info(f"✅ LIVE SELL ORDER placed: {TRADE_QUANTITY} {base_asset} @ {current_price}")
                        record_trade(TRADE_SYMBOL, "SELL", TRADE_QUANTITY, current_price)
                else:
                    logging.warning(f"SELL signal but insufficient {base_asset} balance ({current_crypto_balance:.4f} needed {TRADE_QUANTITY}).")
            
            elif live_signal == 0 and current_position_live == 1:
                if current_crypto_balance >= TRADE_QUANTITY and binance_manager.get_account_balance(asset=quote_asset) > 0:
                    logging.info("Optional: HOLD signal while in position. Deciding whether to exit...")
                    order = binance_manager.place_market_order(TRADE_SYMBOL, TRADE_QUANTITY, SIDE_SELL)
                    if order:
                        current_position_live = 0
                        logging.info(f"✅ LIVE SELL (on HOLD) ORDER placed: {TRADE_QUANTITY} {base_asset} @ {current_price}")
                        record_trade(TRADE_SYMBOL, "SELL", TRADE_QUANTITY, current_price)
                else:
                    logging.debug("HOLD signal while in position, but not exiting or insufficient funds for sell.")

            time.sleep(sleep_seconds_with_buffer)

        except Exception as e:
            error_msg = f"An unexpected error occurred in live trading loop: {e}"
            logging.critical(error_msg, exc_info=True)
            send_email_notification("Trading Bot Live Loop Error", error_msg)
            send_telegram_notification(error_msg)
            time.sleep(60)


# Save records of Trades places
def record_trade(symbol, side, quantity, price):
    db = SessionLocal()
    trade = Trade(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=datetime.now()
    )
    db.add(trade)
    db.commit()
    db.close()


def send_email_notification(subject, message):
    try:
        host = os.getenv("EMAIL_HOST")
        port = int(os.getenv("EMAIL_PORT", 587))
        user = os.getenv("EMAIL_USER")
        password = os.getenv("EMAIL_PASS")
        to_addr = os.getenv("EMAIL_TO")
        
        # Skip if email not configured
        if not all([host, user, password, to_addr]):
            logging.debug("Email not configured. Skipping notification.")
            return
            
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_addr
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())
    except Exception as e:
        logging.error(f"Failed to send email notification: {e}")

def send_telegram_notification(message):
    try:
        if telegram_notifier and telegram_notifier.enabled:
            telegram_notifier.send_message(message)
    except Exception as e:
        logging.error(f"Failed to send Telegram notification: {e}")


# --- Main execution block ---
if __name__ == "__main__":
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    MODE = os.getenv("MODE", "backtest")  # Read MODE from .env, default to 'backtest'
    
    try:
        # First, ensure model is trained and saved, regardless of mode
        df_backtest_data, trained_model, _, _, trained_feature_columns = run_backtesting_pipeline(train_new_model=True) 
        
        if MODE == 'live':
            if trained_model:
                logging.info(f"Switching to LIVE trading mode for {TRADE_SYMBOL}...")
                run_live_trade_loop(trained_model, trained_feature_columns)
            else:
                logging.critical("Cannot start live trading: no trained model available.")
        elif MODE == 'backtest':
            logging.info("Backtesting pipeline executed. No live trading initiated.")
        else:
            logging.error(f"Invalid MODE specified in main.py: {MODE}. Must be 'backtest' or 'live'.")

    except Exception as e:
        error_msg = f"Pipeline execution failed: {e}"
        logging.critical(error_msg, exc_info=True)
        send_email_notification("Trading Bot Critical Failure", error_msg)
        send_telegram_notification(error_msg)
        print("Pipeline execution failed. Check logs for details.")
    finally:
        logging.info("Application finished.")

