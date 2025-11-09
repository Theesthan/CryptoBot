# tests/test_timeseries_models.py

import pytest
import pandas as pd
import numpy as np
from src.timeseries_models import (
    TimeSeriesModeler,
    add_all_timeseries_features
)


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
    
    # Generate realistic price data with trend and volatility
    returns = np.random.normal(0.0001, 0.002, 200)
    prices = 50000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, 200)),
        'high': prices * (1 + np.random.uniform(0.001, 0.003, 200)),
        'low': prices * (1 + np.random.uniform(-0.003, -0.001, 200)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, 200)
    }, index=dates)
    
    return df


class TestTimeSeriesModeler:
    """Test the TimeSeriesModeler class."""
    
    def test_initialization(self):
        """Test TimeSeriesModeler initialization."""
        modeler = TimeSeriesModeler(rolling_window=50, min_observations=30)
        assert modeler.rolling_window == 50
        assert modeler.min_observations == 30
    
    def test_calculate_log_returns(self, sample_ohlcv_data):
        """Test log returns calculation."""
        modeler = TimeSeriesModeler()
        log_returns = modeler.calculate_log_returns(sample_ohlcv_data['close'])
        
        assert isinstance(log_returns, pd.Series)
        assert len(log_returns) == len(sample_ohlcv_data)
        assert log_returns.iloc[0] != log_returns.iloc[0]  # First value is NaN
        assert not log_returns.iloc[1:].isnull().all()
    
    def test_add_garch_features(self, sample_ohlcv_data):
        """Test GARCH feature generation."""
        modeler = TimeSeriesModeler(rolling_window=50, min_observations=30)
        result = modeler.add_garch_features(sample_ohlcv_data)
        
        # Check that new columns were added
        assert 'garch_volatility' in result.columns
        assert 'garch_std_residuals' in result.columns
        
        # Check that some values are not NaN (after rolling window)
        assert result['garch_volatility'].notna().sum() > 0
        assert result['garch_std_residuals'].notna().sum() > 0
        
        # Check that volatility is positive
        valid_vol = result['garch_volatility'].dropna()
        if len(valid_vol) > 0:
            assert (valid_vol >= 0).all()
    
    def test_add_arima_features(self, sample_ohlcv_data):
        """Test ARIMA feature generation."""
        modeler = TimeSeriesModeler(rolling_window=50, min_observations=30)
        result = modeler.add_arima_features(sample_ohlcv_data)
        
        # Check that new columns were added
        assert 'arima_forecast' in result.columns
        assert 'arima_forecast_error' in result.columns
        assert 'arima_conf_interval_width' in result.columns
        
        # Check that some values are not NaN
        assert result['arima_forecast'].notna().sum() > 0
        
        # Check that forecast values are reasonable (close to actual prices)
        valid_forecasts = result['arima_forecast'].dropna()
        actual_prices = result.loc[valid_forecasts.index, 'close']
        if len(valid_forecasts) > 0:
            # Forecasts should be within 20% of actual prices
            relative_error = np.abs((valid_forecasts - actual_prices) / actual_prices)
            assert (relative_error < 0.2).mean() > 0.5  # At least 50% within 20%
    
    def test_add_hmm_features(self, sample_ohlcv_data):
        """Test HMM feature generation."""
        modeler = TimeSeriesModeler(rolling_window=50, min_observations=30)
        result = modeler.add_hmm_features(sample_ohlcv_data, n_regimes=3)
        
        # Check that new columns were added
        assert 'hmm_regime' in result.columns
        assert 'hmm_regime_0_prob' in result.columns
        assert 'hmm_regime_1_prob' in result.columns
        assert 'hmm_regime_2_prob' in result.columns
        
        # Check that some values are not NaN
        assert result['hmm_regime'].notna().sum() > 0
        
        # Check that regime values are valid (0, 1, or 2)
        valid_regimes = result['hmm_regime'].dropna()
        if len(valid_regimes) > 0:
            assert valid_regimes.isin([0, 1, 2]).all()
        
        # Check that probabilities sum to approximately 1
        prob_cols = ['hmm_regime_0_prob', 'hmm_regime_1_prob', 'hmm_regime_2_prob']
        prob_sum = result[prob_cols].sum(axis=1).dropna()
        if len(prob_sum) > 0:
            assert np.allclose(prob_sum, 1.0, atol=0.01)
    
    def test_add_kalman_filter_features(self, sample_ohlcv_data):
        """Test Kalman Filter feature generation."""
        modeler = TimeSeriesModeler(rolling_window=50, min_observations=30)
        result = modeler.add_kalman_filter_features(sample_ohlcv_data)
        
        # Check that new columns were added
        assert 'kalman_level' in result.columns
        assert 'kalman_slope' in result.columns
        assert 'kalman_prediction_error' in result.columns
        
        # Check that some values are not NaN
        assert result['kalman_level'].notna().sum() > 0
        
        # Check that filtered level is close to actual price
        valid_level = result['kalman_level'].dropna()
        actual_prices = result.loc[valid_level.index, 'close']
        if len(valid_level) > 0:
            # Kalman filtered level should be within 10% of actual prices
            relative_error = np.abs((valid_level - actual_prices) / actual_prices)
            assert (relative_error < 0.1).mean() > 0.8  # At least 80% within 10%
    
    def test_add_tsfresh_features_basic(self, sample_ohlcv_data):
        """Test tsfresh feature generation (basic test)."""
        # Add a dummy signal column for tsfresh
        sample_ohlcv_data['signal'] = np.random.choice([0, 1, -1], size=len(sample_ohlcv_data))
        
        modeler = TimeSeriesModeler(rolling_window=30, min_observations=20)
        result = modeler.add_tsfresh_features(
            sample_ohlcv_data, 
            top_k=5, 
            rolling_window_size=30
        )
        
        # Check that result is returned (even if tsfresh fails, it should not crash)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        
        # Check if any tsfresh features were added
        tsfresh_cols = [col for col in result.columns if col.startswith('tsfresh_')]
        # Note: tsfresh might not add features if they're not statistically significant
        # So we just check that the function doesn't crash


class TestAddAllTimeseriesFeatures:
    """Test the integrated add_all_timeseries_features function."""
    
    def test_add_all_features_default(self, sample_ohlcv_data):
        """Test adding all time series features with default settings."""
        result = add_all_timeseries_features(
            sample_ohlcv_data,
            enable_garch=True,
            enable_arima=True,
            enable_hmm=True,
            enable_kalman=True,
            enable_tsfresh=False,  # Disable for speed
            rolling_window=50
        )
        
        # Check that original columns are preserved
        assert 'close' in result.columns
        assert 'volume' in result.columns
        
        # Check that new feature columns were added
        expected_features = [
            'garch_volatility', 'garch_std_residuals',
            'arima_forecast', 'arima_forecast_error', 'arima_conf_interval_width',
            'hmm_regime', 'hmm_regime_0_prob', 'hmm_regime_1_prob', 'hmm_regime_2_prob',
            'kalman_level', 'kalman_slope', 'kalman_prediction_error'
        ]
        
        for feature in expected_features:
            assert feature in result.columns
    
    def test_add_all_features_selective(self, sample_ohlcv_data):
        """Test selective feature addition."""
        result = add_all_timeseries_features(
            sample_ohlcv_data,
            enable_garch=True,
            enable_arima=False,
            enable_hmm=False,
            enable_kalman=True,
            enable_tsfresh=False,
            rolling_window=50
        )
        
        # GARCH and Kalman features should exist
        assert 'garch_volatility' in result.columns
        assert 'kalman_level' in result.columns
        
        # ARIMA and HMM features should not exist
        assert 'arima_forecast' not in result.columns
        assert 'hmm_regime' not in result.columns
    
    def test_add_all_features_handles_errors(self, sample_ohlcv_data):
        """Test that errors in individual models don't crash the entire process."""
        # Use very small data that might cause some models to fail
        small_data = sample_ohlcv_data.iloc[:30].copy()
        
        result = add_all_timeseries_features(
            small_data,
            enable_garch=True,
            enable_arima=True,
            enable_hmm=True,
            enable_kalman=True,
            enable_tsfresh=False,
            rolling_window=20
        )
        
        # Should return a DataFrame even if some models fail
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(small_data)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_insufficient_data(self):
        """Test with insufficient data."""
        small_df = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        modeler = TimeSeriesModeler(rolling_window=50, min_observations=30)
        result = modeler.add_garch_features(small_df)
        
        # Should not crash, but features will be all NaN
        assert 'garch_volatility' in result.columns
        assert result['garch_volatility'].isna().all()
    
    def test_missing_required_columns(self):
        """Test with missing required columns."""
        df = pd.DataFrame({'price': [100, 101, 102]})
        
        modeler = TimeSeriesModeler()
        
        with pytest.raises(ValueError, match="must contain 'close' column"):
            modeler.add_garch_features(df)
    
    def test_non_dataframe_input(self):
        """Test with non-DataFrame input."""
        modeler = TimeSeriesModeler()
        
        with pytest.raises(TypeError):
            add_all_timeseries_features([1, 2, 3])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
