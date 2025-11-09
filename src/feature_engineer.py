# src/feature_engineer.py

import pandas as pd
import numpy as np
import logging
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import SMAIndicator
from sklearn.preprocessing import StandardScaler

from src.config import (
    RSI_WINDOW, BB_WINDOW, BB_WINDOW_DEV, SMA_SHORT_WINDOW, SMA_LONG_WINDOW,
    ATR_WINDOW, RSI_LOWER_QUANTILE, RSI_UPPER_QUANTILE,
    TIMESERIES_ROLLING_WINDOW, TIMESERIES_MIN_OBSERVATIONS, HMM_N_REGIMES,
    TSFRESH_TOP_K, TSFRESH_ROLLING_WINDOW,
    ENABLE_GARCH, ENABLE_ARIMA, ENABLE_HMM, ENABLE_KALMAN, ENABLE_TSFRESH
)

# Import the new time series modeling functions
from src.timeseries_models import add_all_timeseries_features


def add_time_series_features(df):
    """
    Add time series features including lag features and rolling statistics.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data
    
    Returns:
        pd.DataFrame: DataFrame with additional time series features
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')
    
    try:
        result = df.copy()
        
        # Lag features for close price
        result['close_lag_1'] = result['close'].shift(1)
        result['close_lag_2'] = result['close'].shift(2)
        result['close_lag_3'] = result['close'].shift(3)
        
        # Rolling statistics (7-period window)
        result['close_rolling_mean_7'] = result['close'].rolling(window=7).mean()
        result['close_rolling_std_7'] = result['close'].rolling(window=7).std()
        result['close_rolling_min_7'] = result['close'].rolling(window=7).min()
        result['close_rolling_max_7'] = result['close'].rolling(window=7).max()
        
        # Price position within rolling range
        rolling_range = result['close_rolling_max_7'] - result['close_rolling_min_7']
        result['close_position_in_range'] = (result['close'] - result['close_rolling_min_7']) / rolling_range
        result['close_position_in_range'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Volume lag features
        result['volume_lag_1'] = result['volume'].shift(1)
        result['volume_rolling_mean_7'] = result['volume'].rolling(window=7).mean()
        
        # Price velocity (rate of change)
        result['price_velocity_3'] = result['close'].pct_change(3)
        result['price_acceleration'] = result['price_velocity_3'] - result['price_velocity_3'].shift(1)
        
        logging.info(f"Time series features added. New shape: {result.shape}")
        return result
        
    except Exception as e:
        logging.error(f'Error adding time series features: {str(e)}')
        raise


def calculate_technical_indicators(df):
    """
    Calculates essential technical indicators and adds them to the DataFrame.
    Drops original 'BB_hband', 'BB_lband', 'BB_mavg', 'RSI', 'SMA_long', 'SMA_short' columns
    if they exist from previous processing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input for technical indicators must be a pandas DataFrame')
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logging.error(f"Missing required columns for indicators: {missing}")
        raise ValueError(f"DataFrame must contain {required_cols} columns")

    try:
        # RSI
        rsi = RSIIndicator(close=df['close'], window=RSI_WINDOW)
        df['rsi'] = rsi.rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=BB_WINDOW, window_dev=BB_WINDOW_DEV)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        # Handle division by zero if bb_upper == bb_lower (e.g., flat line)
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_pct_b'].replace([np.inf, -np.inf], np.nan, inplace=True) # Replace inf with NaN

        # Moving Averages
        df['sma_20'] = SMAIndicator(close=df['close'], window=SMA_SHORT_WINDOW).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=SMA_LONG_WINDOW).sma_indicator()
        df['ma_cross'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # Price momentum
        df['price_momentum'] = df['close'].pct_change(5) # Example, could be configured
        
        # Average True Range
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=ATR_WINDOW)
        df['atr'] = atr.average_true_range()
        df['atr_pct'] = df['atr'] / df['close']  # ATR as percentage of price
            
        # Volume metrics
        df['volume_pct_change'] = df['volume'].pct_change()
        
        # Remove old duplicate columns from previous processing if they exist
        columns_to_drop_from_input = ['BB_hband', 'BB_lband', 'BB_mavg', 'RSI', 'SMA_long', 'SMA_short']
        df = df.drop(columns=[col for col in columns_to_drop_from_input if col in df.columns])
        
        # Drop rows with NaN values resulting from indicator calculations (e.g., initial rows)
        original_rows = len(df)
        df.dropna(inplace=True)
        logging.info(f"Calculated technical indicators. Dropped {original_rows - len(df)} rows due to NaN values.")
        
        return df
    except Exception as e:
        logging.error(f'Error calculating technical indicators: {str(e)}')
        raise

def get_rsi_quantile_thresholds(rsi_series, lower_quantile=RSI_LOWER_QUANTILE, upper_quantile=RSI_UPPER_QUANTILE):
    """
    Calculate RSI thresholds based on historical distribution.
    
    Args:
        rsi_series (pd.Series): Series containing RSI values
        lower_quantile (float): Quantile for oversold threshold
        upper_quantile (float): Quantile for overbought threshold
    
    Returns:
        tuple: (lower_threshold, upper_threshold)
    """
    if not isinstance(rsi_series, pd.Series):
        raise TypeError('rsi_series must be a pandas Series')
    
    rsi_clean = rsi_series.dropna()
    if rsi_clean.empty:
        logging.warning('RSI series is empty or contains only NaN values. Using default thresholds.')
        return 30, 70 # Fallback to default
    
    if not (0 < lower_quantile < upper_quantile < 1):
        raise ValueError('Quantiles must be between 0 and 1, and lower must be less than upper')
    
    try:
        lower_threshold = rsi_clean.quantile(lower_quantile)
        upper_threshold = rsi_clean.quantile(upper_quantile)
        
        # Ensure thresholds are within valid RSI range (0-100)
        lower_threshold = max(0, min(100, lower_threshold))
        upper_threshold = max(0, min(100, upper_threshold))
        
        return lower_threshold, upper_threshold
    except Exception as e:
        logging.error(f'Error calculating dynamic RSI thresholds: {str(e)}')
        raise

def apply_rsi_labels(df, rsi_col='rsi', lower_threshold=30, upper_threshold=70):
    """
    Apply trading labels based on RSI values.
    
    Args:
        df (pd.DataFrame): DataFrame containing RSI values
        rsi_col (str): Name of the RSI column
        lower_threshold (float): RSI oversold threshold
        upper_threshold (float): RSI overbought threshold
    
    Returns:
        pd.DataFrame: DataFrame with new 'signal' column (1:buy, 0:hold, -1:sell)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input for applying RSI labels must be a pandas DataFrame')
    if rsi_col not in df.columns:
        raise ValueError(f'Column {rsi_col} not found in DataFrame for RSI labeling')
    
    result = df.copy() # Work on a copy
    
    try:
        result['signal'] = 0 # Initialize signals column with 0 (hold)
        result.loc[result[rsi_col] <= lower_threshold, 'signal'] = 1  # Buy signal
        result.loc[result[rsi_col] >= upper_threshold, 'signal'] = -1 # Sell signal
        result['signal'] = result['signal'].astype(int)
        
        # Verify signal values
        unique_signals = set(result['signal'].unique())
        expected_signals = {-1, 0, 1}
        if not unique_signals.issubset(expected_signals):
            logging.warning(f'Unexpected signal values generated: {unique_signals}. Expected: {expected_signals}')
        
        logging.info(f"Trading signals generated based on RSI thresholds ({lower_threshold:.2f}, {upper_threshold:.2f}).")
        return result
    except Exception as e:
        logging.error(f'Error applying RSI labels: {str(e)}')
        raise

def normalize_features(df):
    """
    Normalize features while preserving binary columns.
    Uses StandardScaler.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input for feature normalization must be a pandas DataFrame')
        
    try:
        # Identify binary columns that should NOT be scaled
        binary_cols = ['ma_cross', 'signal'] # 'signal' is the target, should not be scaled
        binary_cols = [col for col in binary_cols if col in df.columns]
        
        # Store binary values if they exist
        binary_data = df[binary_cols] if binary_cols else None
        
        # Get numeric columns to be normalized, excluding binary and object types
        numeric_cols_to_normalize = df.select_dtypes(include=np.number).columns.difference(binary_cols)
        
        if numeric_cols_to_normalize.empty:
            logging.warning("No numeric columns to normalize found after excluding binary/target columns.")
            return df.copy() # Return copy if nothing to normalize

        scaler = StandardScaler()
        df_normalized = pd.DataFrame(
            scaler.fit_transform(df[numeric_cols_to_normalize]),
            columns=numeric_cols_to_normalize,
            index=df.index
        )
        
        # Restore binary columns
        if binary_data is not None:
            df_normalized = pd.concat([df_normalized, binary_data], axis=1)
            
        logging.info("Features normalized successfully.")
        return df_normalized
        
    except Exception as e:
        logging.error(f'Error normalizing features: {str(e)}')
        raise


def add_advanced_timeseries_features(df):
    """
    Add advanced time series modeling features to the dataframe.
    
    This function integrates GARCH, ARIMA, HMM, Kalman Filter, and tsfresh features.
    
    Args:
        df (pd.DataFrame): DataFrame with OHLCV data and basic features
    
    Returns:
        pd.DataFrame: DataFrame with advanced time series features added
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')
    
    try:
        result = add_all_timeseries_features(
            df,
            enable_garch=ENABLE_GARCH,
            enable_arima=ENABLE_ARIMA,
            enable_hmm=ENABLE_HMM,
            enable_kalman=ENABLE_KALMAN,
            enable_tsfresh=ENABLE_TSFRESH,
            rolling_window=TIMESERIES_ROLLING_WINDOW,
            hmm_regimes=HMM_N_REGIMES,
            tsfresh_top_k=TSFRESH_TOP_K
        )
        
        logging.info(f"Advanced time series features added. New shape: {result.shape}")
        return result
        
    except Exception as e:
        logging.error(f'Error adding advanced time series features: {str(e)}')
        raise


def add_derivatives_features(df, funding_df=None, open_interest_df=None, liquidations_df=None):
    """
    Add derivatives market features from Binance Futures data.
    
    Features include:
    - Funding Rate: current rate, rolling changes, z-score
    - Open Interest: current value, rolling delta/change
    - Liquidations: counts and volumes in time windows
    
    Args:
        df (pd.DataFrame): Main OHLCV DataFrame with 'timestamp' column
        funding_df (pd.DataFrame): Funding rate data with 'fundingTime' and 'fundingRate'
        open_interest_df (pd.DataFrame): Open interest data with 'timestamp' and 'openInterest'
        liquidations_df (pd.DataFrame): Liquidation data with 'time', 'origQty', 'side'
    
    Returns:
        pd.DataFrame: DataFrame with derivatives features added
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input must be a pandas DataFrame')
    
    result = df.copy()
    
    # Ensure timestamp exists either as column or index
    if 'timestamp' not in result.columns:
        if result.index.name == 'open_time' or pd.api.types.is_datetime64_any_dtype(result.index):
            # Reset index to convert datetime index to timestamp column
            result = result.reset_index()
            if 'open_time' in result.columns:
                result = result.rename(columns={'open_time': 'timestamp'})
            elif 'index' in result.columns and pd.api.types.is_datetime64_any_dtype(result['index']):
                result = result.rename(columns={'index': 'timestamp'})
            logging.info("✅ Converted datetime index to 'timestamp' column for derivatives features")
        else:
            logging.warning("No 'timestamp' column or datetime index found, skipping derivatives features")
            return result
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(result['timestamp']):
        result['timestamp'] = pd.to_datetime(result['timestamp'])
    
    # ===== FUNDING RATE FEATURES =====
    if funding_df is not None and not funding_df.empty:
        try:
            logging.info(f"Adding funding rate features from {len(funding_df)} records")
            
            # Prepare funding data
            funding_data = funding_df.copy()
            if 'fundingTime' in funding_data.columns:
                funding_data['fundingTime'] = pd.to_datetime(funding_data['fundingTime'])
                funding_data = funding_data.sort_values('fundingTime')
                
                # Convert funding rate to numeric
                funding_data['fundingRate'] = pd.to_numeric(funding_data['fundingRate'], errors='coerce')
                
                # Merge funding rate with main dataframe using nearest time
                result = pd.merge_asof(
                    result.sort_values('timestamp'),
                    funding_data[['fundingTime', 'fundingRate']].rename(columns={'fundingTime': 'timestamp'}),
                    on='timestamp',
                    direction='backward',
                    tolerance=pd.Timedelta('8h')  # Funding rate updates every 8 hours
                )
                
                # Current funding rate
                result['funding_rate'] = result['fundingRate']
                
                # Rolling changes in funding rate
                result['funding_rate_change_1'] = result['funding_rate'].diff(1)
                result['funding_rate_change_3'] = result['funding_rate'].diff(3)
                result['funding_rate_rolling_mean_24'] = result['funding_rate'].rolling(window=24).mean()
                result['funding_rate_rolling_std_24'] = result['funding_rate'].rolling(window=24).std()
                
                # Z-score of funding rate
                mean = result['funding_rate_rolling_mean_24']
                std = result['funding_rate_rolling_std_24']
                result['funding_rate_zscore'] = (result['funding_rate'] - mean) / (std + 1e-8)
                
                # Clean up temporary columns
                if 'fundingRate' in result.columns:
                    result = result.drop('fundingRate', axis=1)
                
                logging.info("✅ Funding rate features added successfully")
        except Exception as e:
            logging.error(f"Error adding funding rate features: {e}")
    else:
        # Add placeholder columns with NaN
        result['funding_rate'] = np.nan
        result['funding_rate_change_1'] = np.nan
        result['funding_rate_change_3'] = np.nan
        result['funding_rate_rolling_mean_24'] = np.nan
        result['funding_rate_rolling_std_24'] = np.nan
        result['funding_rate_zscore'] = np.nan
        logging.info("⚠️  No funding rate data provided, using NaN placeholders")
    
    # ===== OPEN INTEREST FEATURES =====
    if open_interest_df is not None and not open_interest_df.empty:
        try:
            logging.info(f"Adding open interest features from {len(open_interest_df)} records")
            
            # Prepare open interest data
            oi_data = open_interest_df.copy()
            if 'timestamp' in oi_data.columns:
                oi_data['timestamp'] = pd.to_datetime(oi_data['timestamp'])
                oi_data = oi_data.sort_values('timestamp')
                
                # Convert open interest to numeric
                oi_data['openInterest'] = pd.to_numeric(oi_data['openInterest'], errors='coerce')
                
                # Merge with main dataframe
                result = pd.merge_asof(
                    result.sort_values('timestamp'),
                    oi_data[['timestamp', 'openInterest']],
                    on='timestamp',
                    direction='backward',
                    tolerance=pd.Timedelta('1h')
                )
                
                # Current open interest
                result['open_interest'] = result['openInterest']
                
                # Rolling delta/change in open interest
                result['oi_change_1'] = result['open_interest'].diff(1)
                result['oi_change_24'] = result['open_interest'].diff(24)
                result['oi_pct_change_1'] = result['open_interest'].pct_change(1)
                result['oi_pct_change_24'] = result['open_interest'].pct_change(24)
                
                # Rolling statistics
                result['oi_rolling_mean_24'] = result['open_interest'].rolling(window=24).mean()
                result['oi_rolling_std_24'] = result['open_interest'].rolling(window=24).std()
                
                # Z-score of open interest
                mean = result['oi_rolling_mean_24']
                std = result['oi_rolling_std_24']
                result['oi_zscore'] = (result['open_interest'] - mean) / (std + 1e-8)
                
                # Clean up temporary columns
                if 'openInterest' in result.columns:
                    result = result.drop('openInterest', axis=1)
                
                logging.info("✅ Open interest features added successfully")
        except Exception as e:
            logging.error(f"Error adding open interest features: {e}")
    else:
        # Add placeholder columns with NaN
        result['open_interest'] = np.nan
        result['oi_change_1'] = np.nan
        result['oi_change_24'] = np.nan
        result['oi_pct_change_1'] = np.nan
        result['oi_pct_change_24'] = np.nan
        result['oi_rolling_mean_24'] = np.nan
        result['oi_rolling_std_24'] = np.nan
        result['oi_zscore'] = np.nan
        logging.info("⚠️  No open interest data provided, using NaN placeholders")
    
    # ===== LIQUIDATION FEATURES =====
    if liquidations_df is not None and not liquidations_df.empty:
        try:
            logging.info(f"Adding liquidation features from {len(liquidations_df)} records")
            
            # Prepare liquidation data
            liq_data = liquidations_df.copy()
            if 'time' in liq_data.columns:
                liq_data['time'] = pd.to_datetime(liq_data['time'])
                liq_data = liq_data.sort_values('time')
                
                # Convert quantity to numeric
                if 'origQty' in liq_data.columns:
                    liq_data['origQty'] = pd.to_numeric(liq_data['origQty'], errors='coerce')
                
                # Aggregate liquidations by hour
                liq_data['hour'] = liq_data['time'].dt.floor('H')
                
                # Count liquidations per hour
                liq_counts = liq_data.groupby('hour').size().reset_index(name='liq_count')
                
                # Sum liquidation volume per hour
                if 'origQty' in liq_data.columns:
                    liq_volume = liq_data.groupby('hour')['origQty'].sum().reset_index(name='liq_volume')
                    liq_counts = liq_counts.merge(liq_volume, on='hour', how='left')
                
                # Count by side (buy vs sell liquidations)
                if 'side' in liq_data.columns:
                    liq_side = liq_data.groupby(['hour', 'side']).size().unstack(fill_value=0)
                    if 'BUY' in liq_side.columns:
                        liq_side = liq_side.rename(columns={'BUY': 'liq_buy_count'})
                    if 'SELL' in liq_side.columns:
                        liq_side = liq_side.rename(columns={'SELL': 'liq_sell_count'})
                    liq_side = liq_side.reset_index()
                    liq_counts = liq_counts.merge(liq_side, on='hour', how='left')
                
                # Merge with main dataframe
                result['hour'] = result['timestamp'].dt.floor('H')
                result = result.merge(liq_counts, on='hour', how='left')
                result = result.drop('hour', axis=1)
                
                # Fill NaN with 0 (no liquidations in that hour)
                liq_columns = ['liq_count', 'liq_volume', 'liq_buy_count', 'liq_sell_count']
                for col in liq_columns:
                    if col in result.columns:
                        result[col] = result[col].fillna(0)
                
                # Rolling aggregations (24-hour windows)
                if 'liq_count' in result.columns:
                    result['liq_count_24h'] = result['liq_count'].rolling(window=24).sum()
                if 'liq_volume' in result.columns:
                    result['liq_volume_24h'] = result['liq_volume'].rolling(window=24).sum()
                
                # Liquidation imbalance (buy vs sell)
                if 'liq_buy_count' in result.columns and 'liq_sell_count' in result.columns:
                    total_liq = result['liq_buy_count'] + result['liq_sell_count']
                    result['liq_imbalance'] = (result['liq_buy_count'] - result['liq_sell_count']) / (total_liq + 1)
                
                logging.info("✅ Liquidation features added successfully")
        except Exception as e:
            logging.error(f"Error adding liquidation features: {e}")
    else:
        # Add placeholder columns with 0
        result['liq_count'] = 0
        result['liq_volume'] = 0
        result['liq_buy_count'] = 0
        result['liq_sell_count'] = 0
        result['liq_count_24h'] = 0
        result['liq_volume_24h'] = 0
        result['liq_imbalance'] = 0
        logging.info("⚠️  No liquidation data provided, using zero placeholders")
    
    # Restore timestamp as index if original DataFrame had datetime index
    if pd.api.types.is_datetime64_any_dtype(df.index):
        result = result.set_index('timestamp')
        logging.info("✅ Restored timestamp as index")
    
    logging.info(f"Derivatives features added. Final shape: {result.shape}")
    return result
