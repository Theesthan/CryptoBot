"""
Binance Data Fetcher - Historical Data Acquisition Module

This module handles fetching historical data from Binance Spot and Futures APIs:
- OHLCV (Open, High, Low, Close, Volume) data
- Funding Rate data (Futures)
- Open Interest data (Futures)
- Liquidation data (Futures)

Data is saved to CSV files with automatic deduplication and incremental updates.

API Endpoints Used:
- Spot OHLCV: GET /api/v3/klines
- Futures Funding Rate: GET /fapi/v1/fundingRate
- Futures Open Interest: GET /fapi/v1/openInterest
- Futures Liquidations: GET /fapi/v1/allForceOrders

Author: Crypto Trading Bot
Date: 2025-11-06
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Optional, Tuple, List
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """
    Fetches and manages historical data from Binance Spot and Futures APIs.
    Implements incremental data updates with CSV-based caching.
    """
    
    # API Base URLs
    SPOT_BASE_URL = "https://api.binance.com"
    FUTURES_BASE_URL = "https://fapi.binance.com"
    
    # API Endpoints
    SPOT_KLINES = "/api/v3/klines"
    FUTURES_FUNDING_RATE = "/fapi/v1/fundingRate"
    FUTURES_OPEN_INTEREST = "/fapi/v1/openInterest"
    FUTURES_LIQUIDATIONS = "/fapi/v1/allForceOrders"
    
    # Rate limiting
    REQUEST_DELAY = 0.2  # seconds between requests
    MAX_RETRIES = 3
    
    def __init__(self, data_dir: str = "data/historical"):
        """
        Initialize the Binance data fetcher.
        
        Args:
            data_dir: Directory to store CSV files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file paths
        self.ohlcv_file = self.data_dir / "btc_ohlcv.csv"
        self.funding_file = self.data_dir / "btc_funding.csv"
        self.open_interest_file = self.data_dir / "btc_open_interest.csv"
        self.liquidations_file = self.data_dir / "btc_liquidations.csv"
        
        logger.info(f"Initialized BinanceDataFetcher with data directory: {self.data_dir}")
    
    def _make_request(self, url: str, params: dict, retries: int = 0) -> Optional[dict]:
        """
        Make HTTP request to Binance API with retry logic.
        
        Args:
            url: Full API URL
            params: Query parameters
            retries: Current retry attempt
            
        Returns:
            JSON response or None if failed
        """
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            time.sleep(self.REQUEST_DELAY)  # Rate limiting
            return response.json()
        except requests.exceptions.RequestException as e:
            if retries < self.MAX_RETRIES:
                logger.warning(f"Request failed, retry {retries + 1}/{self.MAX_RETRIES}: {e}")
                time.sleep(2 ** retries)  # Exponential backoff
                return self._make_request(url, params, retries + 1)
            else:
                logger.error(f"Request failed after {self.MAX_RETRIES} retries: {e}")
                return None
    
    def _load_existing_csv(self, file_path: Path, date_column: str = 'timestamp') -> pd.DataFrame:
        """
        Load existing CSV file and return existing dates.
        
        Args:
            file_path: Path to CSV file
            date_column: Name of the date/timestamp column
            
        Returns:
            DataFrame with existing data or empty DataFrame
        """
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                if date_column in df.columns:
                    df[date_column] = pd.to_datetime(df[date_column])
                logger.info(f"Loaded {len(df)} existing records from {file_path.name}")
                return df
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
                return pd.DataFrame()
        else:
            logger.info(f"No existing file found at {file_path.name}, will create new")
            return pd.DataFrame()
    
    def _save_to_csv(self, df: pd.DataFrame, file_path: Path, date_column: str = 'timestamp'):
        """
        Save DataFrame to CSV, removing duplicates.
        
        Args:
            df: DataFrame to save
            file_path: Path to save CSV
            date_column: Column to use for deduplication
        """
        try:
            # Remove duplicates based on date column
            df = df.drop_duplicates(subset=[date_column], keep='last')
            df = df.sort_values(date_column)
            
            # Save to CSV
            df.to_csv(file_path, index=False)
            logger.info(f"Saved {len(df)} records to {file_path.name}")
        except Exception as e:
            logger.error(f"Error saving to {file_path.name}: {e}")
    
    def fetch_ohlcv(self, symbol: str = "BTCUSDT", interval: str = "1h", 
                    days_back: int = 365) -> pd.DataFrame:
        """
        Fetch OHLCV data from Binance Spot API up to yesterday (to avoid incomplete candles).
        Implements smart incremental updates - only fetches new data.
        
        API: GET /api/v3/klines
        Docs: https://binance-docs.github.io/apidocs/spot/en/#kline-candlestick-data
        
        Args:
            symbol: Trading pair symbol
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            days_back: Number of days of historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Fetching OHLCV data for {symbol} ({interval} interval, {days_back} days)")
        
        # Load existing data
        existing_df = self._load_existing_csv(self.ohlcv_file, 'timestamp')
        
        # Calculate time range - up to NOW (current hour) to get latest data
        now = datetime.now()
        end_time = now.replace(minute=0, second=0, microsecond=0)  # Current hour, rounded down
        logger.info(f"Fetching data up to current hour: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check if data is already up to date
        if not existing_df.empty and 'timestamp' in existing_df.columns:
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            last_timestamp = existing_df['timestamp'].max()
            
            if last_timestamp >= pd.Timestamp(end_time):
                logger.info(f"✅ Data already up to date! Last timestamp: {last_timestamp}")
                return existing_df
            
            # Start from 1 hour after last timestamp
            start_time = last_timestamp + timedelta(hours=1)
            logger.info(f"Incremental update: fetching from {start_time} to {end_time}")
        else:
            start_time = end_time - timedelta(days=days_back)
            logger.info(f"Full fetch: {days_back} days from {start_time} to {end_time}")
        
        # Binance API limit: 1000 candles per request
        all_data = []
        current_start = int(start_time.timestamp() * 1000)
        current_end = int(end_time.timestamp() * 1000)
        
        url = f"{self.SPOT_BASE_URL}{self.SPOT_KLINES}"
        
        while current_start < current_end:
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': current_end,
                'limit': 1000
            }
            
            data = self._make_request(url, params)
            if not data:
                break
            
            if len(data) == 0:
                break
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            all_data.append(df)
            logger.info(f"Fetched {len(df)} candles. Total so far: {len(all_data) * len(df)}")
            
            # Move to next batch
            current_start = int(data[-1][0]) + 1
            
            # Check if we've reached the end
            if len(data) < 1000:
                break
            
            time.sleep(self.REQUEST_DELAY)  # Rate limiting
        
        # Combine with existing data
        if all_data:
            new_df = pd.concat(all_data, ignore_index=True)
            
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                             'quote_asset_volume', 'taker_buy_base_asset_volume', 
                             'taker_buy_quote_asset_volume']
            for col in numeric_columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            
            self._save_to_csv(combined_df, self.ohlcv_file, 'timestamp')
            logger.info(f"✅ Total OHLCV records: {len(combined_df)}")
            logger.info(f"   Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
            return combined_df
        else:
            logger.info("No new OHLCV data to fetch")
            return existing_df
    
    def fetch_funding_rate(self, symbol: str = "BTCUSDT", days_back: int = 365) -> pd.DataFrame:
        """
        Fetch funding rate data from Binance Futures API.
        
        API: GET /fapi/v1/fundingRate
        Docs: https://binance-docs.github.io/apidocs/futures/en/#get-funding-rate-history
        
        Funding rate is updated every 8 hours.
        
        Args:
            symbol: Trading pair symbol
            days_back: Number of days of historical data
            
        Returns:
            DataFrame with funding rate data
        """
        logger.info(f"Fetching funding rate data for {symbol} ({days_back} days)")
        
        # Load existing data
        existing_df = self._load_existing_csv(self.funding_file, 'fundingTime')
        existing_timestamps = set()
        if not existing_df.empty and 'fundingTime' in existing_df.columns:
            existing_timestamps = set(pd.to_datetime(existing_df['fundingTime']))
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        url = f"{self.FUTURES_BASE_URL}{self.FUTURES_FUNDING_RATE}"
        
        params = {
            'symbol': symbol,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': 1000
        }
        
        all_data = []
        
        while True:
            data = self._make_request(url, params)
            if not data or len(data) == 0:
                break
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
            
            # Filter out existing timestamps
            if existing_timestamps:
                df = df[~df['fundingTime'].isin(existing_timestamps)]
            
            if not df.empty:
                all_data.append(df)
                logger.info(f"Fetched {len(df)} new funding rate records")
            
            # Move to next batch
            if len(data) < 1000:
                break
            params['startTime'] = int(data[-1]['fundingTime']) + 1
        
        # Combine with existing data
        if all_data:
            new_df = pd.concat(all_data, ignore_index=True)
            
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Convert numeric columns
            combined_df['fundingRate'] = pd.to_numeric(combined_df['fundingRate'], errors='coerce')
            
            self._save_to_csv(combined_df, self.funding_file, 'fundingTime')
            logger.info(f"Total funding rate records: {len(combined_df)}")
            return combined_df
        else:
            logger.info("No new funding rate data to fetch")
            return existing_df
    
    def fetch_open_interest(self, symbol: str = "BTCUSDT", interval: str = "1h",
                           days_back: int = 365) -> pd.DataFrame:
        """
        Fetch open interest data from Binance Futures API.
        
        API: GET /fapi/v1/openInterest
        Docs: https://binance-docs.github.io/apidocs/futures/en/#open-interest
        
        Note: This endpoint returns only the current open interest.
        For historical data, we need to call it periodically or use another method.
        
        Args:
            symbol: Trading pair symbol
            interval: Time interval
            days_back: Number of days (limited by API)
            
        Returns:
            DataFrame with open interest data
        """
        logger.info(f"Fetching open interest data for {symbol}")
        
        # Load existing data
        existing_df = self._load_existing_csv(self.open_interest_file, 'timestamp')
        
        # Get current open interest
        url = f"{self.FUTURES_BASE_URL}{self.FUTURES_OPEN_INTEREST}"
        params = {'symbol': symbol}
        
        data = self._make_request(url, params)
        if data:
            # Create new record
            new_record = {
                'timestamp': pd.Timestamp.now(),
                'symbol': data['symbol'],
                'openInterest': float(data['openInterest']),
                'time': pd.to_datetime(data['time'], unit='ms')
            }
            
            new_df = pd.DataFrame([new_record])
            
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            self._save_to_csv(combined_df, self.open_interest_file, 'timestamp')
            logger.info(f"Total open interest records: {len(combined_df)}")
            return combined_df
        else:
            logger.warning("Failed to fetch current open interest")
            return existing_df
    
    def fetch_liquidations(self, symbol: str = "BTCUSDT", days_back: int = 7) -> pd.DataFrame:
        """
        Fetch liquidation data from Binance Futures API.
        
        API: GET /fapi/v1/allForceOrders
        Docs: https://binance-docs.github.io/apidocs/futures/en/#all-orders-user_data
        
        Note: Binance only provides liquidation data for the last 7 days.
        
        Args:
            symbol: Trading pair symbol
            days_back: Number of days (max 7)
            
        Returns:
            DataFrame with liquidation data
        """
        logger.info(f"Fetching liquidation data for {symbol} ({min(days_back, 7)} days)")
        
        # Load existing data
        existing_df = self._load_existing_csv(self.liquidations_file, 'time')
        existing_times = set()
        if not existing_df.empty and 'time' in existing_df.columns:
            existing_times = set(pd.to_datetime(existing_df['time']))
        
        # Binance only keeps 7 days of liquidation data
        days_back = min(days_back, 7)
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        url = f"{self.FUTURES_BASE_URL}{self.FUTURES_LIQUIDATIONS}"
        
        params = {
            'symbol': symbol,
            'startTime': int(start_time.timestamp() * 1000),
            'endTime': int(end_time.timestamp() * 1000),
            'limit': 1000
        }
        
        all_data = []
        
        while True:
            data = self._make_request(url, params)
            if not data or len(data) == 0:
                break
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'], unit='ms')
                
                # Filter out existing times
                if existing_times:
                    df = df[~df['time'].isin(existing_times)]
                
                if not df.empty:
                    all_data.append(df)
                    logger.info(f"Fetched {len(df)} new liquidation records")
            
            # Move to next batch
            if len(data) < 1000:
                break
            if 'time' in data[-1]:
                params['startTime'] = int(data[-1]['time']) + 1
        
        # Combine with existing data
        if all_data:
            new_df = pd.concat(all_data, ignore_index=True)
            
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            # Convert numeric columns
            if 'price' in combined_df.columns:
                combined_df['price'] = pd.to_numeric(combined_df['price'], errors='coerce')
            if 'origQty' in combined_df.columns:
                combined_df['origQty'] = pd.to_numeric(combined_df['origQty'], errors='coerce')
            if 'executedQty' in combined_df.columns:
                combined_df['executedQty'] = pd.to_numeric(combined_df['executedQty'], errors='coerce')
            
            self._save_to_csv(combined_df, self.liquidations_file, 'time')
            logger.info(f"Total liquidation records: {len(combined_df)}")
            return combined_df
        else:
            logger.info("No new liquidation data to fetch")
            return existing_df
    
    def fetch_all_data(self, symbol: str = "BTCUSDT", days_back: int = 365) -> dict:
        """
        Fetch all available data types for the given symbol.
        
        Args:
            symbol: Trading pair symbol
            days_back: Number of days of historical data
            
        Returns:
            Dictionary with all DataFrames
        """
        logger.info(f"=" * 80)
        logger.info(f"Starting comprehensive data fetch for {symbol}")
        logger.info(f"Target: {days_back} days of historical data")
        logger.info(f"=" * 80)
        
        results = {}
        
        # Fetch OHLCV data
        try:
            results['ohlcv'] = self.fetch_ohlcv(symbol, interval="1h", days_back=days_back)
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            results['ohlcv'] = pd.DataFrame()
        
        # Fetch funding rate
        try:
            results['funding'] = self.fetch_funding_rate(symbol, days_back=days_back)
        except Exception as e:
            logger.error(f"Error fetching funding rate: {e}")
            results['funding'] = pd.DataFrame()
        
        # Fetch open interest (current only due to API limitations)
        try:
            results['open_interest'] = self.fetch_open_interest(symbol)
        except Exception as e:
            logger.error(f"Error fetching open interest: {e}")
            results['open_interest'] = pd.DataFrame()
        
        # Fetch liquidations (last 7 days only due to API limitations)
        try:
            results['liquidations'] = self.fetch_liquidations(symbol, days_back=min(days_back, 7))
        except Exception as e:
            logger.error(f"Error fetching liquidations: {e}")
            results['liquidations'] = pd.DataFrame()
        
        logger.info(f"=" * 80)
        logger.info(f"Data fetch complete!")
        logger.info(f"OHLCV records: {len(results['ohlcv'])}")
        logger.info(f"Funding rate records: {len(results['funding'])}")
        logger.info(f"Open interest records: {len(results['open_interest'])}")
        logger.info(f"Liquidation records: {len(results['liquidations'])}")
        logger.info(f"=" * 80)
        
        return results


def main():
    """
    Main function for standalone execution and testing.
    """
    print("=" * 80)
    print("Binance Historical Data Fetcher")
    print("=" * 80)
    print()
    
    # Initialize fetcher
    fetcher = BinanceDataFetcher()
    
    # Fetch all data for BTC
    print("Fetching comprehensive historical data for BTCUSDT...")
    print("This may take several minutes depending on how much data needs to be fetched.")
    print()
    
    results = fetcher.fetch_all_data(symbol="BTCUSDT", days_back=365)
    
    print()
    print("=" * 80)
    print("Summary:")
    print("=" * 80)
    for data_type, df in results.items():
        if not df.empty:
            print(f"✅ {data_type.upper()}: {len(df)} records")
            if 'timestamp' in df.columns:
                print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            elif 'fundingTime' in df.columns:
                print(f"   Date range: {df['fundingTime'].min()} to {df['fundingTime'].max()}")
            elif 'time' in df.columns:
                print(f"   Date range: {df['time'].min()} to {df['time'].max()}")
        else:
            print(f"⚠️  {data_type.upper()}: No data")
    print("=" * 80)


if __name__ == "__main__":
    main()
