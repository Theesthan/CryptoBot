import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model_manager import load_trained_model, make_predictions
from src.config import FEATURE_COLUMNS, MODEL_SAVE_PATH, DATA_FILE_PATH, TRADE_SYMBOL
from src.data_loader import load_and_preprocess_data
from src.feature_engineer import calculate_technical_indicators
from src.db import SessionLocal, Trade
from src.binance_manager import BinanceManager

# Page configuration
st.set_page_config(
    page_title="Crypto Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🤖 Crypto Trading Bot Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Mode selection
    mode = st.radio("Select Mode:", ["📊 View Dashboard", "🔮 Make Prediction", "💼 Trade History", "📈 Live Market"])
    
    st.markdown("---")
    st.markdown("### 📋 Model Info")
    if os.path.exists(MODEL_SAVE_PATH):
        st.success("✅ Model Loaded")
        model_size = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
        st.info(f"Model Size: {model_size:.2f} MB")
        
        # Check feature file
        feature_path = MODEL_SAVE_PATH.replace('.json', '_features.pkl')
        if os.path.exists(feature_path):
            st.info(f"✅ Features Saved")
        
        # Add cache clear button
        if st.button("🔄 Reload Model", help="Clear cache and reload model"):
            st.cache_resource.clear()
            st.rerun()
    else:
        st.error("❌ Model Not Found")
    
    st.markdown("---")
    st.markdown("### 🎯 Trading Symbol")
    st.info(f"**{TRADE_SYMBOL}**")

# Load model (cached)
@st.cache_resource
def get_model():
    try:
        result = load_trained_model()
        # Handle both old format (just model) and new format (model, features)
        if isinstance(result, tuple):
            model, features = result
            return model, features
        else:
            return result, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model_result = get_model()
if model_result[0] is not None:
    model, saved_features = model_result
else:
    model, saved_features = None, None

# Mode: Dashboard
if mode == "📊 View Dashboard":
    st.header("📊 Trading Performance Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        db = SessionLocal()
        trades = db.query(Trade).all()
        db.close()
        
        if trades:
            df_trades = pd.DataFrame([{
                'timestamp': t.timestamp,
                'symbol': t.symbol,
                'side': t.side,
                'quantity': t.quantity,
                'price': t.price
            } for t in trades])
            
            total_trades = len(df_trades)
            buy_trades = len(df_trades[df_trades['side'] == 'BUY'])
            sell_trades = len(df_trades[df_trades['side'] == 'SELL'])
            
            # Calculate P&L
            buy_value = df_trades[df_trades['side'] == 'BUY']['price'].sum() * df_trades[df_trades['side'] == 'BUY']['quantity'].sum()
            sell_value = df_trades[df_trades['side'] == 'SELL']['price'].sum() * df_trades[df_trades['side'] == 'SELL']['quantity'].sum()
            pnl = sell_value - buy_value
            
            with col1:
                st.metric("Total Trades", total_trades)
            with col2:
                st.metric("Buy Trades", buy_trades, delta=None)
            with col3:
                st.metric("Sell Trades", sell_trades, delta=None)
            with col4:
                st.metric("P&L (Approx)", f"${pnl:.2f}", delta=f"{pnl:.2f}")
            
            st.markdown("---")
            
            # Trade history chart
            st.subheader("📈 Trade History")
            df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp'])
            df_trades = df_trades.sort_values('timestamp')
            
            fig = go.Figure()
            
            # Buy trades
            buy_df = df_trades[df_trades['side'] == 'BUY']
            fig.add_trace(go.Scatter(
                x=buy_df['timestamp'],
                y=buy_df['price'],
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ))
            
            # Sell trades
            sell_df = df_trades[df_trades['side'] == 'SELL']
            fig.add_trace(go.Scatter(
                x=sell_df['timestamp'],
                y=sell_df['price'],
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ))
            
            fig.update_layout(
                title=f"{TRADE_SYMBOL} Trade History",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent trades table
            st.subheader("📋 Recent Trades")
            st.dataframe(df_trades.tail(10).sort_values('timestamp', ascending=False), use_container_width=True)
            
        else:
            st.info("No trades recorded yet. Start trading to see data here!")
            
    except Exception as e:
        st.error(f"Error loading trade data: {e}")

# Mode: Make Prediction
elif mode == "🔮 Make Prediction":
    st.header("🔮 Make Trading Prediction")
    
    if model is None:
        st.error("Model not loaded. Please train the model first.")
    else:
        st.info("Enter the technical indicator values below to get a trading signal prediction.")
        
        # Initialize session state for input values
        if 'prediction_inputs' not in st.session_state:
            st.session_state.prediction_inputs = {col: 0.0 for col in FEATURE_COLUMNS}
        
        # Sample value buttons with NORMALIZED values
        col_sample1, col_sample2, col_sample3 = st.columns(3)
        with col_sample1:
            if st.button("📝 Fill with Sample Values (Bullish)", type="secondary", use_container_width=True):
                # BULLISH = OVERSOLD = LOW RSI < 43.61 (BUY zone)
                # These are actual normalized values from training data
                st.session_state.prediction_inputs = {
                    'rsi': -1.682516,        # Low RSI (normalized ~30) - OVERSOLD = BUY signal
                    'bb_upper': -0.234567,   # Normalized Bollinger upper
                    'bb_lower': -0.456789,   # Normalized Bollinger lower
                    'bb_mid': -0.345678,     # Normalized Bollinger middle
                    'bb_pct_b': -0.863198,   # Below Bollinger bands (<0) - oversold
                    'sma_20': -0.345678,     # Normalized SMA 20
                    'sma_50': -0.567890,     # Normalized SMA 50
                    'ma_cross': -1.0,        # Below moving average - potential reversal
                    'price_momentum': -1.124788,  # Negative momentum - oversold
                    'atr': 0.5,              # Moderate volatility
                    'atr_pct': 0.234567      # Normalized ATR %
                }
                st.success("✅ Bullish values loaded! (Low RSI = Oversold = BUY signal)")
        
        with col_sample2:
            if st.button("📝 Fill with Sample Values (Bearish)", type="secondary", use_container_width=True):
                # BEARISH = OVERBOUGHT = HIGH RSI > 66.10 (SELL zone)
                # These are actual normalized values from training data
                st.session_state.prediction_inputs = {
                    'rsi': 1.565890,         # High RSI (normalized ~75) - OVERBOUGHT = SELL signal
                    'bb_upper': 1.901895,    # Normalized Bollinger upper
                    'bb_lower': 1.870674,    # Normalized Bollinger lower
                    'bb_mid': 1.894323,      # Normalized Bollinger middle
                    'bb_pct_b': 1.115224,    # Above Bollinger bands (>1) - overbought
                    'sma_20': 1.894323,      # Normalized SMA 20
                    'sma_50': 1.634630,      # Normalized SMA 50
                    'ma_cross': 1.0,         # Above moving average - potential reversal
                    'price_momentum': 0.845147,  # Positive momentum - overbought
                    'atr': 1.261728,         # High volatility
                    'atr_pct': -1.010409     # Normalized ATR %
                }
                st.success("✅ Bearish values loaded! (High RSI = Overbought = SELL signal)")
        
        with col_sample3:
            if st.button("🔄 Reset to Zero", type="secondary", use_container_width=True):
                st.session_state.prediction_inputs = {col: 0.0 for col in FEATURE_COLUMNS}
                st.info("� All values reset to zero")
        
        st.markdown("---")
        
        # Field descriptions/tooltips (full form + short meaning)
        field_descriptions = {
            'rsi': "Relative Strength Index - Measures momentum to identify overbought/oversold conditions",
            'bb_upper': "Bollinger Band Upper - Upper price volatility boundary based on standard deviation",
            'bb_lower': "Bollinger Band Lower - Lower price volatility boundary based on standard deviation",
            'bb_mid': "Bollinger Band Middle - Moving average baseline between upper and lower bands",
            'bb_pct_b': "Bollinger Percent B - Price position within bands (0-1 inside, >1 above, <0 below)",
            'sma_20': "Simple Moving Average (20) - Average price over last 20 periods for trend direction",
            'sma_50': "Simple Moving Average (50) - Average price over last 50 periods for long-term trend",
            'ma_cross': "Moving Average Crossover - Signal when short-term MA crosses long-term MA",
            'price_momentum': "Price Momentum - Rate of price change indicating trend strength and direction",
            'atr': "Average True Range - Measures market volatility using price range",
            'atr_pct': "ATR Percentage - Volatility relative to price expressed as percentage"
        }
        
        # Create input form with descriptive tooltips
        col1, col2 = st.columns(2)
        
        user_input = {}
        for idx, col in enumerate(FEATURE_COLUMNS):
            with col1 if idx % 2 == 0 else col2:
                user_input[col] = st.number_input(
                    f"{col.upper().replace('_', ' ')}",
                    value=st.session_state.prediction_inputs[col],
                    format="%.6f",
                    help=field_descriptions.get(col, f"Enter value for {col}"),
                    key=f"input_{col}"
                )
                # Update session state
                st.session_state.prediction_inputs[col] = user_input[col]
        
        st.markdown("---")
        
        # Warning about realistic values
        st.warning("⚠️ **Note:** For realistic predictions, use actual market data values. All zeros will produce unreliable results!")
        
        with st.expander("💡 Example Realistic Values (Normalized)"):
            st.markdown("""
            **Typical normalized values (all features are standardized):**
            - **RSI**: -1.5 to +1.5 (negative = oversold, positive = overbought)
            - **BB_UPPER**: -2.0 to +2.0 (standardized upper band)
            - **BB_LOWER**: -2.0 to +2.0 (standardized lower band)
            - **BB_MID**: -2.0 to +2.0 (standardized middle band)
            - **BB_PCT_B**: -1.0 to +2.0 (position within bands, 0-1 = inside)
            - **SMA_20**: -2.0 to +2.0 (standardized short-term MA)
            - **SMA_50**: -2.0 to +2.0 (standardized long-term MA)
            - **MA_CROSS**: -1, 0, or 1 (crossover signal)
            - **PRICE_MOMENTUM**: -2.0 to +2.0 (standardized momentum)
            - **ATR**: -1.5 to +1.5 (standardized volatility)
            - **ATR_PCT**: -1.5 to +1.5 (standardized volatility %)
            
            *Note: All values are normalized using z-score standardization (mean=0, std=1)*
            """)
        
        if st.button("🎯 Generate Prediction", type="primary"):
            # Check if model is loaded
            if model is None:
                st.error("❌ Model not loaded! Please train the model first by running CryptoStart.bat")
                st.stop()
            
            try:
                # Check if all values are zero
                all_zeros = all(v == 0.0 for v in user_input.values())
                if all_zeros:
                    st.error("⚠️ All input values are zero! This will produce unrealistic predictions. Please enter actual market indicator values.")
                
                # Create dataframe with user input (basic features only)
                X_basic = pd.DataFrame([user_input])[FEATURE_COLUMNS]
                
                # Add placeholder time series features (since we don't have historical data in manual input)
                # These will be set to neutral/average values
                time_series_features = {
                    'close_lag_1': 0.0,  # Neutral normalized value
                    'close_lag_2': 0.0,
                    'close_lag_3': 0.0,
                    'close_rolling_mean_7': 0.0,
                    'close_rolling_std_7': 1.0,  # Average volatility
                    'close_rolling_min_7': -1.0,
                    'close_rolling_max_7': 1.0,
                    'close_position_in_range': 0.5,  # Midpoint
                    'volume_lag_1': 0.0,
                    'volume_rolling_mean_7': 0.0,
                    'price_velocity_3': 0.0,  # No momentum assumed
                    'price_acceleration': 0.0
                }
                
                # Combine basic features with time series features
                X_full = pd.concat([X_basic, pd.DataFrame([time_series_features])], axis=1)
                
                # Use the saved features from the loaded model
                if saved_features:
                    # Add any missing features as zeros (neutral values)
                    for feat in saved_features:
                        if feat not in X_full.columns:
                            X_full[feat] = 0.0
                    
                    # Reorder columns to match model's expected features
                    X_full = X_full[saved_features]
                else:
                    # Fallback: use ALL_FEATURE_COLUMNS if features weren't saved
                    from src.config import ALL_FEATURE_COLUMNS
                    # Add any missing features as zeros
                    for feat in ALL_FEATURE_COLUMNS:
                        if feat not in X_full.columns:
                            X_full[feat] = 0.0
                    X_full = X_full[ALL_FEATURE_COLUMNS]
                
                preds, probs = make_predictions(model, X_full)
                
                prediction = int(preds.iloc[0])
                confidence = float(probs[0])
                
                # Get probabilities for all classes
                model_probs = model.predict_proba(X_full)[0]
                
                # Display result
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    signal_map = {-1: "SELL 🔴", 0: "HOLD 🟡", 1: "BUY 🟢"}
                    signal_color = {-1: "red", 0: "orange", 1: "green"}
                    st.markdown(f"### Signal: {signal_map[prediction]}")
                    st.markdown(f"<h1 style='color:{signal_color[prediction]}'>{signal_map[prediction]}</h1>", unsafe_allow_html=True)
                
                with col2:
                    st.metric("Confidence Score", f"{confidence:.2%}")
                    st.progress(min(confidence, 1.0))
                    
                    # Show warning if confidence is unrealistically high
                    if confidence > 0.95:
                        st.warning("⚠️ Very high confidence may indicate unrealistic input values!")
                
                with col3:
                    st.metric("Raw Prediction", prediction)
                    interpretation = {
                        -1: "Strong sell signal",
                        0: "Hold or no clear direction",
                        1: "Strong buy signal"
                    }
                    st.info(interpretation[prediction])
                
                # Show detailed probabilities
                st.markdown("---")
                st.subheader("📊 Detailed Prediction Probabilities")
                prob_col1, prob_col2, prob_col3 = st.columns(3)
                
                with prob_col1:
                    st.metric("SELL Probability", f"{model_probs[0]:.2%}", delta=None)
                with prob_col2:
                    st.metric("HOLD Probability", f"{model_probs[1]:.2%}", delta=None)
                with prob_col3:
                    st.metric("BUY Probability", f"{model_probs[2]:.2%}", delta=None)
                
                st.success("✅ Prediction generated successfully!")
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")

# Mode: Trade History
elif mode == "💼 Trade History":
    st.header("💼 Complete Trade History")
    
    try:
        db = SessionLocal()
        trades = db.query(Trade).all()
        db.close()
        
        if trades:
            df_trades = pd.DataFrame([{
                'ID': t.id,
                'Timestamp': t.timestamp,
                'Symbol': t.symbol,
                'Side': t.side,
                'Quantity': t.quantity,
                'Price': t.price,
                'Order ID': t.order_id if t.order_id else 'N/A',
            } for t in trades])
            
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                filter_side = st.selectbox("Filter by Side:", ["All", "BUY", "SELL"])
            with col2:
                filter_symbol = st.selectbox("Filter by Symbol:", ["All"] + list(df_trades['Symbol'].unique()))
            with col3:
                date_filter = st.date_input("From Date:", value=datetime.now() - timedelta(days=30))
            
            # Apply filters
            filtered_df = df_trades.copy()
            if filter_side != "All":
                filtered_df = filtered_df[filtered_df['Side'] == filter_side]
            if filter_symbol != "All":
                filtered_df = filtered_df[filtered_df['Symbol'] == filter_symbol]
            
            # Display
            st.dataframe(filtered_df.sort_values('Timestamp', ascending=False), use_container_width=True)
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"trade_history_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        else:
            st.info("No trades recorded yet.")
            
    except Exception as e:
        st.error(f"Error loading trade history: {e}")

# Mode: Live Market
elif mode == "📈 Live Market":
    st.header("📈 Live Market Data & Backtesting")
    
    try:
        from src.data_loader import load_historical_data
        from src.feature_engineer import (
            calculate_technical_indicators, add_time_series_features,
            add_derivatives_features, get_rsi_quantile_thresholds, apply_rsi_labels, normalize_features
        )
        from src.model_manager import prepare_model_data
        from src.backtester import backtest_strategy
        from src.config import USE_BINANCE_DATA, TARGET_COLUMN
        from plotly.subplots import make_subplots
        
        st.info("📥 Loading latest data from Binance...")
        
        # Load data
        if USE_BINANCE_DATA:
            data_dict = load_historical_data()
            df = data_dict['ohlcv']
            
            if df.empty:
                st.error("No data loaded. Please fetch data first using fetch_binance_data.bat")
                st.stop()
            
            st.success(f"✅ Loaded {len(df)} records from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
            
            # Display latest data
            st.subheader("📊 Latest Market Indicators")
            latest = df.iloc[-1]
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Close Price", f"${latest['close']:,.2f}")
            with col2:
                st.metric("High (24h)", f"${df['high'].tail(24).max():,.2f}")
            with col3:
                st.metric("Low (24h)", f"${df['low'].tail(24).min():,.2f}")
            with col4:
                st.metric("Volume (24h)", f"{df['volume'].tail(24).sum():,.0f}")
            with col5:
                price_change = ((latest['close'] - df['close'].iloc[-25]) / df['close'].iloc[-25]) * 100
                st.metric("Change (24h)", f"{price_change:+.2f}%", delta=f"{price_change:.2f}%")
            
            st.markdown("---")
            
            # Calculate features
            with st.spinner("Calculating technical indicators..."):
                df = calculate_technical_indicators(df)
                df = add_time_series_features(df)
                
                # Add derivatives features if available
                if data_dict['funding'] is not None:
                    df = add_derivatives_features(
                        df,
                        data_dict['funding'],
                        data_dict['open_interest'],
                        data_dict['liquidations']
                    )
                    st.info("✅ Derivatives features included (funding rate, open interest, liquidations)")
            
            # Generate signals
            lower_thresh, upper_thresh = get_rsi_quantile_thresholds(df['rsi'])
            df = apply_rsi_labels(df, lower_threshold=lower_thresh, upper_threshold=upper_thresh)
            
            # Normalize
            df_normalized = normalize_features(df.copy())
            
            # Get feature columns - Use saved features from model if available, otherwise calculate
            if saved_features is not None:
                feature_cols = saved_features
                # Ensure all required features exist in the data
                missing_features = [col for col in feature_cols if col not in df_normalized.columns]
                if missing_features:
                    st.error(f"❌ Missing features in data: {missing_features}")
                    st.stop()
            else:
                # Fallback: Calculate features dynamically (MUST match training exclusions)
                exclude_cols = ['signal', 'open', 'high', 'low', 'close', 'volume', 'volume_pct_change']
                feature_cols = [col for col in df_normalized.columns 
                               if col not in exclude_cols
                               and df_normalized[col].dtype in ['float64', 'float32', 'int64', 'int32']]
                st.warning("⚠️ Using dynamically calculated features (saved features not found)")
            
            # Ensure features are in the same order as training
            X_full = df_normalized[feature_cols]
            
            # Make predictions with model
            if model is not None:
                with st.spinner("Running backtest with model predictions..."):
                    from src.model_manager import make_predictions
                    from src.config import CONFIDENCE_THRESHOLD
                    
                    predictions, confidences = make_predictions(model, X_full, confidence_threshold=CONFIDENCE_THRESHOLD)
                    
                    # Run backtest
                    df_original = data_dict['ohlcv'].loc[predictions.index]
                    trades_df, daily_portfolio_df = backtest_strategy(df_original, predictions)
                    
                    # Show metrics
                    st.subheader("💰 Backtesting Performance")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    if not daily_portfolio_df.empty:
                        initial_balance = daily_portfolio_df.iloc[0]['total_value']
                        final_balance = daily_portfolio_df.iloc[-1]['total_value']
                        total_return = ((final_balance - initial_balance) / initial_balance) * 100
                        num_trades = len(trades_df) // 2 if len(trades_df) >= 2 else 0
                        
                        # Calculate win rate
                        profitable_trades = 0
                        if num_trades > 0:
                            for i in range(0, len(trades_df) - 1, 2):
                                if trades_df.iloc[i]['type'] == 'buy' and trades_df.iloc[i+1]['type'] == 'sell':
                                    buy_cost = trades_df.iloc[i]['shares'] * trades_df.iloc[i]['price'] + trades_df.iloc[i]['fee']
                                    sell_proceeds = trades_df.iloc[i+1]['shares'] * trades_df.iloc[i+1]['price'] - trades_df.iloc[i+1]['fee']
                                    if sell_proceeds > buy_cost:
                                        profitable_trades += 1
                            win_rate = (profitable_trades / num_trades) * 100
                        else:
                            win_rate = 0
                        
                        col1.metric("Initial Balance", f"${initial_balance:,.2f}")
                        col2.metric("Final Balance", f"${final_balance:,.2f}", delta=f"{total_return:.2f}%")
                        col3.metric("Total Return", f"{total_return:.2f}%")
                        col4.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{num_trades} trades")
                    
                    st.markdown("---")
                    
                    # Create comprehensive visualization
                    st.subheader("📊 Complete Trading Performance Chart")
                    
                    # Interactive controls for chart features
                    with st.expander("🎨 Chart Display Options", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**Price Indicators:**")
                            show_bb = st.checkbox("Bollinger Bands", value=True, key="show_bb")
                            show_sma20 = st.checkbox("SMA 20", value=True, key="show_sma20")
                            show_sma50 = st.checkbox("SMA 50", value=True, key="show_sma50")
                        
                        with col2:
                            st.markdown("**Trade Signals:**")
                            show_buy_signals = st.checkbox("Buy Signals", value=True, key="show_buy")
                            show_sell_signals = st.checkbox("Sell Signals", value=True, key="show_sell")
                            st.markdown("**Chart Panels:**")
                            show_volume = st.checkbox("Volume Panel", value=True, key="show_volume")
                        
                        with col3:
                            st.markdown("**Technical Indicators:**")
                            show_rsi = st.checkbox("RSI Panel", value=True, key="show_rsi")
                            show_atr = st.checkbox("ATR Panel", value=True, key="show_atr")
                            show_portfolio = st.checkbox("Portfolio Value", value=True, key="show_portfolio")
                    
                    # Get data aligned with predictions
                    df_viz = df.loc[predictions.index]
                    
                    # Calculate dynamic rows based on selected panels
                    panels = []
                    row_heights = []
                    subplot_titles = []
                    
                    # Price panel (always shown)
                    panels.append('price')
                    row_heights.append(0.4)
                    subplot_titles.append('Price Action & Trades with Indicators')
                    
                    if show_volume:
                        panels.append('volume')
                        row_heights.append(0.15)
                        subplot_titles.append('Volume')
                    
                    if show_rsi:
                        panels.append('rsi')
                        row_heights.append(0.15)
                        subplot_titles.append('Relative Strength Index (RSI)')
                    
                    if show_atr:
                        panels.append('atr')
                        row_heights.append(0.15)
                        subplot_titles.append('Average True Range (ATR)')
                    
                    if show_portfolio:
                        panels.append('portfolio')
                        row_heights.append(0.15)
                        subplot_titles.append('Account Value')
                    
                    # Normalize row heights to sum to 1
                    total_height = sum(row_heights)
                    row_heights = [h/total_height for h in row_heights]
                    
                    # Create chart with dynamic rows
                    fig = make_subplots(rows=len(panels), cols=1,
                                       shared_xaxes=True,
                                       vertical_spacing=0.03,
                                       subplot_titles=subplot_titles,
                                       row_heights=row_heights)
                    
                    current_row = 1
                    
                    # Row 1: Candlestick + indicators (always shown)
                    fig.add_trace(
                        go.Candlestick(x=df_viz.index,
                                      open=df_viz['open'],
                                      high=df_viz['high'],
                                      low=df_viz['low'],
                                      close=df_viz['close'],
                                      name='OHLC',
                                      increasing_line_color='green',
                                      decreasing_line_color='red'),
                        row=current_row, col=1
                    )
                    
                    # Bollinger Bands (optional)
                    if show_bb and 'bb_upper' in df_viz.columns:
                        fig.add_trace(go.Scatter(x=df_viz.index, y=df_viz['bb_upper'], 
                                                line=dict(color='rgba(255,0,0,0.5)', width=1), name='BB Upper'), row=current_row, col=1)
                        fig.add_trace(go.Scatter(x=df_viz.index, y=df_viz['bb_lower'], 
                                                line=dict(color='rgba(0,255,0,0.5)', width=1), name='BB Lower'), row=current_row, col=1)
                        fig.add_trace(go.Scatter(x=df_viz.index, y=df_viz['bb_mid'], 
                                                line=dict(color='rgba(0,0,255,0.5)', width=1), name='BB Mid'), row=current_row, col=1)
                    
                    # SMAs (optional)
                    if show_sma20 and 'sma_20' in df_viz.columns:
                        fig.add_trace(go.Scatter(x=df_viz.index, y=df_viz['sma_20'], 
                                                line=dict(color='orange', width=2), name='SMA 20'), row=current_row, col=1)
                    if show_sma50 and 'sma_50' in df_viz.columns:
                        fig.add_trace(go.Scatter(x=df_viz.index, y=df_viz['sma_50'], 
                                                line=dict(color='purple', width=2), name='SMA 50'), row=current_row, col=1)
                    
                    # Buy/Sell signals (optional)
                    if not trades_df.empty:
                        if show_buy_signals:
                            buys = trades_df[trades_df['type'] == 'buy']
                            if not buys.empty:
                                fig.add_trace(
                                    go.Scatter(x=buys.index, y=buys['price'], mode='markers', name='Buy Signals',
                                              marker=dict(color='green', size=12, symbol='triangle-up', line=dict(color='darkgreen', width=1))),
                                    row=current_row, col=1
                                )
                        
                        if show_sell_signals:
                            sells = trades_df[trades_df['type'] == 'sell']
                            if not sells.empty:
                                fig.add_trace(
                                    go.Scatter(x=sells.index, y=sells['price'], mode='markers', name='Sell Signals',
                                              marker=dict(color='red', size=12, symbol='triangle-down', line=dict(color='darkred', width=1))),
                                    row=current_row, col=1
                                )
                    
                    current_row += 1
                    
                    # Volume panel (optional)
                    if show_volume:
                        fig.add_trace(
                            go.Bar(x=df_viz.index, y=df_viz['volume'], name='Volume', 
                                  marker_color='blue', opacity=0.3, showlegend=False),
                            row=current_row, col=1
                        )
                        current_row += 1
                    
                    # RSI panel (optional)
                    if show_rsi and 'rsi' in df_viz.columns:
                        fig.add_trace(go.Scatter(x=df_viz.index, y=df_viz['rsi'],
                                                line=dict(color='darkblue', width=1.5), name='RSI', showlegend=False),
                                     row=current_row, col=1)
                        fig.add_hline(y=upper_thresh, line_dash="dot", line_color="red", row=current_row, col=1)
                        fig.add_hline(y=lower_thresh, line_dash="dot", line_color="green", row=current_row, col=1)
                        fig.update_yaxes(range=[0, 100], row=current_row, col=1)
                        current_row += 1
                    
                    # ATR panel (optional)
                    if show_atr and 'atr' in df_viz.columns:
                        fig.add_trace(go.Scatter(x=df_viz.index, y=df_viz['atr'],
                                                line=dict(color='darkgreen', width=1.5), name='ATR', showlegend=False),
                                     row=current_row, col=1)
                        current_row += 1
                    
                    # Portfolio Value panel (optional)
                    if show_portfolio and not daily_portfolio_df.empty:
                        fig.add_trace(
                            go.Scatter(x=daily_portfolio_df.index, y=daily_portfolio_df['total_value'],
                                      name='Account Value', line=dict(color='purple', width=2)),
                            row=current_row, col=1
                        )
                    
                    # Dynamic layout based on number of panels
                    layout_updates = {
                        'title': f'{TRADE_SYMBOL} Trading Strategy Performance - Latest Data',
                        f'xaxis{len(panels)}_title': 'Date',
                        'yaxis_title': 'Price (USD)',
                        'height': 300 + (len(panels) * 200),  # Dynamic height
                        'showlegend': True,
                        'hovermode': "x unified",
                    }
                    
                    # Add y-axis titles for additional panels
                    panel_idx = 2
                    if show_volume:
                        layout_updates[f'yaxis{panel_idx}_title'] = 'Volume'
                        panel_idx += 1
                    if show_rsi:
                        layout_updates[f'yaxis{panel_idx}_title'] = 'RSI'
                        panel_idx += 1
                    if show_atr:
                        layout_updates[f'yaxis{panel_idx}_title'] = 'ATR'
                        panel_idx += 1
                    if show_portfolio:
                        layout_updates[f'yaxis{panel_idx}_title'] = 'Portfolio Value ($)'
                    
                    # Disable rangeslider for all except last panel
                    for i in range(1, len(panels) + 1):
                        layout_updates[f'xaxis{i}_rangeslider_visible'] = False
                    
                    fig.update_layout(**layout_updates)
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show trade details
                    if not trades_df.empty:
                        st.subheader("📝 Recent Trades")
                        st.dataframe(trades_df.tail(20), use_container_width=True)
                        
                        # Download button
                        csv = trades_df.to_csv(index=True)
                        st.download_button(
                            label="📥 Download All Trades (CSV)",
                            data=csv,
                            file_name=f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
            else:
                st.error("❌ Model not loaded. Please train the model first using CryptoStart.bat")
        else:
            st.warning("⚠️ Binance data is disabled. Enable USE_BINANCE_DATA=True in src/config.py and run fetch_binance_data.bat")
            
    except Exception as e:
        st.error(f"❌ Error in loading market data: {e}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🤖 Crypto Trading Bot Dashboard v1.0</p>
        <p>⚠️ For educational purposes only. Not financial advice.</p>
    </div>
""", unsafe_allow_html=True)