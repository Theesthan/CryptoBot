# src/data_loader.py

import pandas as pd
import logging
import numpy as np
from pathlib import Path


def load_and_preprocess_data(file_path): # file_path is now passed as argument
    """
    Loads historical cryptocurrency price data, converts timestamp,
    sets index, sorts, and handles missing values.
    
    This function can load data from:
    1. The default test CSV (data/test_df_features.csv)
    2. The new historical Binance data (data/historical/btc_ohlcv.csv)
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with timestamp index
    """
    try:
        df = pd.read_csv(file_path)

        # Convert timestamp to datetime and set as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        else:
            logging.warning("No 'timestamp' column found. Ensure data is time-indexed.")
        
        # Sort by index to ensure chronological order
        df = df.sort_index()
        
        # Handle missing values
        missing_count = df.isnull().sum()
        if missing_count.any():
            logging.warning(f'Found missing values:\n{missing_count[missing_count > 0]}')
            df = df.ffill()  # Forward fill
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            
        logging.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Data file not found at {file_path}. Please check the path.")
        raise
    except Exception as e:
        logging.error(f'Error loading or preprocessing data: {str(e)}')
        raise


def load_historical_data(use_binance_data=True):
    """
    Load historical data from Binance CSV files or fallback to test data.
    
    This function is the new entry point for loading data with Binance historical data.
    It will:
    1. Try to load from data/historical/btc_ohlcv.csv (if use_binance_data=True)
    2. Fallback to data/test_df_features.csv if Binance data not available
    
    Args:
        use_binance_data (bool): Whether to use Binance historical data
        
    Returns:
        dict: Dictionary containing all DataFrames:
            - 'ohlcv': Main OHLCV DataFrame
            - 'funding': Funding rate DataFrame (if available)
            - 'open_interest': Open interest DataFrame (if available)
            - 'liquidations': Liquidations DataFrame (if available)
    """
    result = {
        'ohlcv': None,
        'funding': None,
        'open_interest': None,
        'liquidations': None
    }
    
    # Define file paths
    binance_data_dir = Path("data/historical")
    ohlcv_file = binance_data_dir / "btc_ohlcv.csv"
    funding_file = binance_data_dir / "btc_funding.csv"
    oi_file = binance_data_dir / "btc_open_interest.csv"
    liq_file = binance_data_dir / "btc_liquidations.csv"
    fallback_file = Path("data/test_df_features.csv")
    
    if use_binance_data and ohlcv_file.exists():
        logging.info("Loading Binance historical data...")
        
        # Load main OHLCV data
        try:
            result['ohlcv'] = load_and_preprocess_data(str(ohlcv_file))
            logging.info(f"✅ Loaded OHLCV data: {len(result['ohlcv'])} records")
        except Exception as e:
            logging.error(f"Failed to load OHLCV data: {e}")
            logging.info("Falling back to test data...")
            result['ohlcv'] = load_and_preprocess_data(str(fallback_file))
        
        # Load funding rate data
        if funding_file.exists():
            try:
                funding_df = pd.read_csv(funding_file)
                if 'fundingTime' in funding_df.columns:
                    funding_df['fundingTime'] = pd.to_datetime(funding_df['fundingTime'])
                result['funding'] = funding_df
                logging.info(f"✅ Loaded funding rate data: {len(funding_df)} records")
            except Exception as e:
                logging.warning(f"Failed to load funding rate data: {e}")
        else:
            logging.info("ℹ️  No funding rate data available")
        
        # Load open interest data
        if oi_file.exists():
            try:
                oi_df = pd.read_csv(oi_file)
                if 'timestamp' in oi_df.columns:
                    oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'])
                result['open_interest'] = oi_df
                logging.info(f"✅ Loaded open interest data: {len(oi_df)} records")
            except Exception as e:
                logging.warning(f"Failed to load open interest data: {e}")
        else:
            logging.info("ℹ️  No open interest data available")
        
        # Load liquidations data
        if liq_file.exists():
            try:
                liq_df = pd.read_csv(liq_file)
                if 'time' in liq_df.columns:
                    liq_df['time'] = pd.to_datetime(liq_df['time'])
                result['liquidations'] = liq_df
                logging.info(f"✅ Loaded liquidations data: {len(liq_df)} records")
            except Exception as e:
                logging.warning(f"Failed to load liquidations data: {e}")
        else:
            logging.info("ℹ️  No liquidations data available")
    
    else:
        # Use fallback test data
        logging.info("Using fallback test data...")
        result['ohlcv'] = load_and_preprocess_data(str(fallback_file))
    
    return result
