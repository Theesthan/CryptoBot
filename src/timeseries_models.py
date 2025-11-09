# src/timeseries_models.py

# CRITICAL: Disable CUDA for stumpy BEFORE any imports to avoid hanging
import os
os.environ['NUMBA_DISABLE_CUDA'] = '1'

import pandas as pd
import numpy as np
import logging
from warnings import catch_warnings, filterwarnings
from typing import Tuple, Optional

# Time series modeling libraries - import with error handling
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("arch library not installed. GARCH features will be disabled. Run: pip install arch>=6.2.0")

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not installed. ARIMA features will be disabled. Run: pip install statsmodels>=0.14.0")

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logging.warning("hmmlearn not installed. HMM features will be disabled. Run: pip install hmmlearn>=0.3.0")

try:
    from pykalman import KalmanFilter
    PYKALMAN_AVAILABLE = True
except ImportError:
    PYKALMAN_AVAILABLE = False
    logging.warning("pykalman not installed. Kalman Filter features will be disabled. Run: pip install pykalman>=0.9.5")

try:
    from tsfresh import extract_features, select_features
    from tsfresh.utilities.dataframe_functions import impute
    TSFRESH_AVAILABLE = True
except ImportError:
    TSFRESH_AVAILABLE = False
    logging.warning("tsfresh not installed. Automated feature extraction will be disabled. Run: pip install tsfresh>=0.20.0")
except Exception as e:
    # Handle any other tsfresh initialization errors (e.g., CUDA issues)
    TSFRESH_AVAILABLE = False
    logging.warning(f"tsfresh available but failed to initialize: {e}. Automated feature extraction will be disabled.")


class TimeSeriesModeler:
    """
    Advanced time series modeling for cryptocurrency trading features.
    Implements GARCH, ARIMA, HMM, Kalman Filter, and tsfresh automated feature extraction.
    """
    
    def __init__(self, rolling_window: int = 50, min_observations: int = 30):
        """
        Initialize the time series modeler.
        
        Args:
            rolling_window (int): Window size for rolling models (default 50)
            min_observations (int): Minimum observations required for model fitting (default 30)
        """
        self.rolling_window = rolling_window
        self.min_observations = min_observations
        self.progress_interval = 100  # Log progress every N iterations
        
    def calculate_log_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns from price series."""
        log_returns = np.log(prices / prices.shift(1))
        return log_returns
    
    def add_garch_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add GARCH(1,1) volatility features using rolling window.
        
        Features added:
        - garch_volatility: Conditional volatility forecast
        - garch_std_residuals: Standardized residuals
        
        Args:
            df (pd.DataFrame): DataFrame with 'close' price column
            
        Returns:
            pd.DataFrame: DataFrame with GARCH features added
        """
        if not ARCH_AVAILABLE:
            logging.warning("GARCH features skipped: arch library not installed")
            return df
            
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column for GARCH modeling")
        
        result = df.copy()
        log_returns = self.calculate_log_returns(result['close'])
        
        # Initialize feature columns
        result['garch_volatility'] = np.nan
        result['garch_std_residuals'] = np.nan
        
        logging.info(f"Calculating GARCH features with rolling window of {self.rolling_window}...")
        
        total_iterations = len(result) - self.rolling_window
        successful_fits = 0
        
        for i in range(self.rolling_window, len(result)):
            try:
                # Progress logging
                if (i - self.rolling_window) % self.progress_interval == 0:
                    progress_pct = ((i - self.rolling_window) / total_iterations) * 100
                    logging.info(f"GARCH progress: {progress_pct:.1f}% ({i - self.rolling_window}/{total_iterations})")
                
                # Get rolling window of returns
                window_returns = log_returns.iloc[i-self.rolling_window:i].dropna()
                
                if len(window_returns) < self.min_observations:
                    continue
                
                # Scale returns to percentage for numerical stability
                returns_scaled = window_returns * 100
                
                with catch_warnings():
                    filterwarnings('ignore')
                    
                    # Fit GARCH(1,1) model with faster settings
                    model = arch_model(
                        returns_scaled,
                        vol='Garch',
                        p=1,
                        q=1,
                        rescale=False
                    )
                    
                    # Use faster optimization with limits
                    fitted = model.fit(
                        disp='off',
                        show_warning=False,
                        options={'maxiter': 100}  # Limit iterations to avoid hanging
                    )
                    
                    # One-step ahead forecast
                    forecast = fitted.forecast(horizon=1, reindex=False)
                    
                    # Get conditional volatility (scale back to decimal)
                    conditional_vol = np.sqrt(forecast.variance.values[-1, 0]) / 100
                    result.loc[result.index[i], 'garch_volatility'] = conditional_vol
                    
                    # Calculate standardized residuals
                    if fitted.conditional_volatility.iloc[-1] > 0:
                        std_resid = fitted.resid.iloc[-1] / fitted.conditional_volatility.iloc[-1]
                        result.loc[result.index[i], 'garch_std_residuals'] = std_resid
                    
                    successful_fits += 1
                        
            except Exception as e:
                logging.debug(f"GARCH model failed at index {i}: {str(e)}")
                continue
        
        logging.info(f"GARCH features calculated. Successful fits: {successful_fits}/{total_iterations} ({(successful_fits/total_iterations)*100:.1f}%)")
        return result
    
    def add_arima_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ARIMA modeling features using rolling window.
        
        Features added:
        - arima_forecast: One-step ahead forecast
        - arima_forecast_error: Forecast error (actual - forecast)
        - arima_conf_interval_width: Width of 95% confidence interval
        
        Args:
            df (pd.DataFrame): DataFrame with 'close' price column
            
        Returns:
            pd.DataFrame: DataFrame with ARIMA features added
        """
        if not STATSMODELS_AVAILABLE:
            logging.warning("ARIMA features skipped: statsmodels library not installed")
            return df
            
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column for ARIMA modeling")
        
        result = df.copy()
        log_returns = self.calculate_log_returns(result['close'])
        
        # Initialize feature columns
        result['arima_forecast'] = np.nan
        result['arima_forecast_error'] = np.nan
        result['arima_conf_interval_width'] = np.nan
        
        logging.info(f"Calculating ARIMA features with rolling window of {self.rolling_window}...")
        
        for i in range(self.rolling_window, len(result)):
            try:
                # Get rolling window of returns
                window_returns = log_returns.iloc[i-self.rolling_window:i].dropna()
                
                if len(window_returns) < self.min_observations:
                    continue
                
                with catch_warnings():
                    filterwarnings('ignore', category=ConvergenceWarning)
                    filterwarnings('ignore', category=UserWarning)
                    
                    # Fit ARIMA(1,0,1) model - simple configuration
                    model = ARIMA(window_returns, order=(1, 0, 1))
                    fitted = model.fit()
                    
                    # One-step ahead forecast with confidence interval
                    forecast_result = fitted.get_forecast(steps=1)
                    forecast_mean = forecast_result.predicted_mean.iloc[0]
                    conf_int = forecast_result.conf_int(alpha=0.05)
                    
                    # Calculate forecast price from log return forecast
                    last_price = result['close'].iloc[i-1]
                    forecast_price = last_price * np.exp(forecast_mean)
                    
                    result.loc[result.index[i], 'arima_forecast'] = forecast_price
                    
                    # Calculate forecast error (using actual price)
                    actual_price = result['close'].iloc[i]
                    result.loc[result.index[i], 'arima_forecast_error'] = actual_price - forecast_price
                    
                    # Width of confidence interval
                    conf_width = conf_int.iloc[0, 1] - conf_int.iloc[0, 0]
                    result.loc[result.index[i], 'arima_conf_interval_width'] = conf_width
                    
            except Exception as e:
                logging.debug(f"ARIMA model failed at index {i}: {str(e)}")
                continue
        
        logging.info(f"ARIMA features calculated. Valid values: {result['arima_forecast'].notna().sum()}/{len(result)}")
        return result
    
    def add_hmm_features(self, df: pd.DataFrame, n_regimes: int = 3) -> pd.DataFrame:
        """
        Add Hidden Markov Model regime features using rolling window.
        
        Features added:
        - hmm_regime: Most likely regime (0, 1, or 2)
        - hmm_regime_0_prob: Probability of regime 0
        - hmm_regime_1_prob: Probability of regime 1
        - hmm_regime_2_prob: Probability of regime 2 (if n_regimes=3)
        
        Args:
            df (pd.DataFrame): DataFrame with 'close' price column
            n_regimes (int): Number of hidden regimes (2 or 3)
            
        Returns:
            pd.DataFrame: DataFrame with HMM features added
        """
        if not HMMLEARN_AVAILABLE:
            logging.warning("HMM features skipped: hmmlearn library not installed")
            return df
            
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column for HMM modeling")
        
        result = df.copy()
        log_returns = self.calculate_log_returns(result['close'])
        
        # Initialize feature columns
        result['hmm_regime'] = np.nan
        for regime_idx in range(n_regimes):
            result[f'hmm_regime_{regime_idx}_prob'] = np.nan
        
        logging.info(f"Calculating HMM features with {n_regimes} regimes and rolling window of {self.rolling_window}...")
        
        for i in range(self.rolling_window, len(result)):
            try:
                # Get rolling window of returns
                window_returns = log_returns.iloc[i-self.rolling_window:i].dropna()
                
                if len(window_returns) < self.min_observations:
                    continue
                
                # Prepare data for HMM (needs to be 2D array)
                returns_array = window_returns.values.reshape(-1, 1)
                
                with catch_warnings():
                    filterwarnings('ignore')
                    
                    # Fit Gaussian HMM
                    model = GaussianHMM(
                        n_components=n_regimes,
                        covariance_type='full',
                        n_iter=100,
                        random_state=42
                    )
                    
                    model.fit(returns_array)
                    
                    # Predict regime probabilities for the last observation
                    last_return = returns_array[-1:, :]
                    regime_probs = model.predict_proba(last_return)[0]
                    most_likely_regime = np.argmax(regime_probs)
                    
                    # Store results
                    result.loc[result.index[i], 'hmm_regime'] = most_likely_regime
                    for regime_idx in range(n_regimes):
                        result.loc[result.index[i], f'hmm_regime_{regime_idx}_prob'] = regime_probs[regime_idx]
                        
            except Exception as e:
                logging.debug(f"HMM model failed at index {i}: {str(e)}")
                continue
        
        logging.info(f"HMM features calculated. Valid values: {result['hmm_regime'].notna().sum()}/{len(result)}")
        return result
    
    def add_kalman_filter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Kalman Filter features for price level and slope estimation.
        
        Features added:
        - kalman_level: Filtered price level
        - kalman_slope: Filtered price slope (trend)
        - kalman_prediction_error: Prediction error
        
        Args:
            df (pd.DataFrame): DataFrame with 'close' price column
            
        Returns:
            pd.DataFrame: DataFrame with Kalman Filter features added
        """
        if not PYKALMAN_AVAILABLE:
            logging.warning("Kalman Filter features skipped: pykalman library not installed")
            return df
            
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column for Kalman filtering")
        
        result = df.copy()
        prices = result['close'].values
        
        # Initialize feature columns
        result['kalman_level'] = np.nan
        result['kalman_slope'] = np.nan
        result['kalman_prediction_error'] = np.nan
        
        logging.info(f"Calculating Kalman Filter features with rolling window of {self.rolling_window}...")
        
        for i in range(self.rolling_window, len(result)):
            try:
                # Get rolling window of prices
                window_prices = prices[i-self.rolling_window:i]
                
                if len(window_prices) < self.min_observations:
                    continue
                
                # Local level + slope Kalman filter
                # State: [level, slope]
                # Observation: [level]
                
                # Transition matrix: level(t) = level(t-1) + slope(t-1), slope(t) = slope(t-1)
                transition_matrices = [[1, 1], [0, 1]]
                
                # Observation matrix: we observe the level
                observation_matrices = [[1, 0]]
                
                # Initialize Kalman filter
                kf = KalmanFilter(
                    transition_matrices=transition_matrices,
                    observation_matrices=observation_matrices,
                    initial_state_mean=[window_prices[0], 0],
                    initial_state_covariance=np.eye(2) * 1000,
                    observation_covariance=1.0,
                    transition_covariance=np.eye(2) * 0.01
                )
                
                # Filter the data
                state_means, state_covariances = kf.filter(window_prices.reshape(-1, 1))
                
                # Extract filtered level and slope for the last observation
                filtered_level = state_means[-1, 0]
                filtered_slope = state_means[-1, 1]
                
                # Prediction error
                predicted_level = state_means[-2, 0] + state_means[-2, 1]  # level + slope from previous step
                actual_price = window_prices[-1]
                prediction_error = actual_price - predicted_level
                
                # Store results
                result.loc[result.index[i], 'kalman_level'] = filtered_level
                result.loc[result.index[i], 'kalman_slope'] = filtered_slope
                result.loc[result.index[i], 'kalman_prediction_error'] = prediction_error
                
            except Exception as e:
                logging.debug(f"Kalman filter failed at index {i}: {str(e)}")
                continue
        
        logging.info(f"Kalman Filter features calculated. Valid values: {result['kalman_level'].notna().sum()}/{len(result)}")
        return result
    
    def add_tsfresh_features(
        self, 
        df: pd.DataFrame, 
        top_k: int = 10,
        rolling_window_size: int = 30
    ) -> pd.DataFrame:
        """
        Add automated feature extraction using tsfresh on rolling windows.
        
        Features extracted from rolling windows of:
        - close price
        - log returns
        - volume
        
        Only the top-k most relevant features are retained.
        
        Args:
            df (pd.DataFrame): DataFrame with 'close' and 'volume' columns
            top_k (int): Number of top features to keep
            rolling_window_size (int): Window size for tsfresh rolling extraction
            
        Returns:
            pd.DataFrame: DataFrame with tsfresh features added
        """
        if not TSFRESH_AVAILABLE:
            logging.warning("tsfresh features skipped: tsfresh library not installed")
            return df
            
        if 'close' not in df.columns or 'volume' not in df.columns:
            raise ValueError("DataFrame must contain 'close' and 'volume' columns for tsfresh")
        
        if 'signal' not in df.columns:
            logging.warning("No 'signal' column found for tsfresh feature selection. Creating dummy target.")
            df['signal'] = 0  # Dummy target for feature selection
        
        result = df.copy()
        log_returns = self.calculate_log_returns(result['close'])
        
        logging.info(f"Calculating tsfresh features with rolling window of {rolling_window_size} and top-{top_k} selection...")
        
        try:
            # Prepare data in tsfresh format (needs 'id', 'time', and 'value' columns)
            ts_data_list = []
            
            for i in range(rolling_window_size, len(result)):
                # Create a rolling window ID
                window_id = i
                
                # Extract rolling window data
                window_close = result['close'].iloc[i-rolling_window_size:i].reset_index(drop=True)
                window_volume = result['volume'].iloc[i-rolling_window_size:i].reset_index(drop=True)
                window_returns = log_returns.iloc[i-rolling_window_size:i].reset_index(drop=True).dropna()
                
                # Add close price time series
                for t, val in enumerate(window_close):
                    ts_data_list.append({
                        'id': window_id,
                        'time': t,
                        'close': val,
                        'volume': window_volume.iloc[t],
                        'returns': window_returns.iloc[t] if t < len(window_returns) else np.nan
                    })
            
            if not ts_data_list:
                logging.warning("Not enough data for tsfresh feature extraction")
                return result
            
            # Convert to DataFrame
            ts_df = pd.DataFrame(ts_data_list)
            
            # Extract features for each column separately
            with catch_warnings():
                filterwarnings('ignore')
                
                # Extract features
                extracted_features = extract_features(
                    ts_df,
                    column_id='id',
                    column_sort='time',
                    default_fc_parameters={
                        'length': None,
                        'mean': None,
                        'standard_deviation': None,
                        'variance': None,
                        'maximum': None,
                        'minimum': None,
                        'median': None,
                        'sum_values': None,
                        'abs_energy': None,
                        'mean_abs_change': None,
                    },
                    impute_function=impute,
                    disable_progressbar=True
                )
                
                # Handle NaN and inf values
                extracted_features = extracted_features.replace([np.inf, -np.inf], np.nan)
                extracted_features = extracted_features.fillna(0)
                
                # Prepare target for feature selection (align with extracted features)
                # The extracted features start from index rolling_window_size
                y_target = result['signal'].iloc[rolling_window_size:rolling_window_size+len(extracted_features)]
                y_target = y_target.reset_index(drop=True)
                
                # Feature selection - keep only top-k most relevant
                if len(extracted_features) > 0 and len(y_target) > 0:
                    selected_features = select_features(
                        extracted_features,
                        y_target,
                        fdr_level=0.05,
                        n_jobs=1
                    )
                    
                    # If too many features, take top-k by absolute correlation with target
                    if len(selected_features.columns) > top_k:
                        correlations = selected_features.corrwith(y_target).abs().sort_values(ascending=False)
                        top_features = correlations.head(top_k).index.tolist()
                        selected_features = selected_features[top_features]
                    
                    # Rename columns to add tsfresh prefix
                    selected_features.columns = [f'tsfresh_{col}' for col in selected_features.columns]
                    
                    # Align indices and merge
                    selected_features.index = result.index[rolling_window_size:rolling_window_size+len(selected_features)]
                    
                    # Add selected features to result
                    for col in selected_features.columns:
                        result[col] = np.nan
                        result.loc[selected_features.index, col] = selected_features[col]
                    
                    logging.info(f"tsfresh features extracted. Added {len(selected_features.columns)} features.")
                else:
                    logging.warning("No features selected by tsfresh")
                    
        except Exception as e:
            logging.error(f"tsfresh feature extraction failed: {str(e)}", exc_info=True)
            # Return original dataframe if tsfresh fails
            pass
        
        return result


def add_all_timeseries_features(
    df: pd.DataFrame,
    enable_garch: bool = True,
    enable_arima: bool = True,
    enable_hmm: bool = True,
    enable_kalman: bool = True,
    enable_tsfresh: bool = True,
    rolling_window: int = 50,
    hmm_regimes: int = 3,
    tsfresh_top_k: int = 10
) -> pd.DataFrame:
    """
    Add all time series modeling features to the dataframe.
    
    This is the main function to call from the feature engineering pipeline.
    
    Args:
        df (pd.DataFrame): Input dataframe with OHLCV data
        enable_garch (bool): Enable GARCH features
        enable_arima (bool): Enable ARIMA features
        enable_hmm (bool): Enable HMM features
        enable_kalman (bool): Enable Kalman Filter features
        enable_tsfresh (bool): Enable tsfresh automated feature extraction
        rolling_window (int): Rolling window size for models
        hmm_regimes (int): Number of HMM regimes
        tsfresh_top_k (int): Number of top tsfresh features to keep
        
    Returns:
        pd.DataFrame: DataFrame with all selected time series features added
    """
    result = df.copy()
    modeler = TimeSeriesModeler(rolling_window=rolling_window)
    
    logging.info("Starting advanced time series feature engineering...")
    
    if enable_garch:
        try:
            result = modeler.add_garch_features(result)
            logging.info("✅ GARCH features added successfully")
        except Exception as e:
            logging.error(f"Failed to add GARCH features: {str(e)}")
    
    if enable_arima:
        try:
            result = modeler.add_arima_features(result)
            logging.info("✅ ARIMA features added successfully")
        except Exception as e:
            logging.error(f"Failed to add ARIMA features: {str(e)}")
    
    if enable_hmm:
        try:
            result = modeler.add_hmm_features(result, n_regimes=hmm_regimes)
            logging.info("✅ HMM features added successfully")
        except Exception as e:
            logging.error(f"Failed to add HMM features: {str(e)}")
    
    if enable_kalman:
        try:
            result = modeler.add_kalman_filter_features(result)
            logging.info("✅ Kalman Filter features added successfully")
        except Exception as e:
            logging.error(f"Failed to add Kalman Filter features: {str(e)}")
    
    if enable_tsfresh:
        try:
            result = modeler.add_tsfresh_features(result, top_k=tsfresh_top_k)
            logging.info("✅ tsfresh features added successfully")
        except Exception as e:
            logging.error(f"Failed to add tsfresh features: {str(e)}")
    
    logging.info(f"Advanced time series features complete. Final shape: {result.shape}")
    
    return result
