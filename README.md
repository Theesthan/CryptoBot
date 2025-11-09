# Cryptocurrency Trading Bot

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A modular, production-ready cryptocurrency trading system using machine learning (XGBoost), technical indicators, and robust infrastructure for both backtesting and live trading. The project features a full pipeline from data collection and feature engineering to model training, backtesting, live trading, and interactive dashboards. It is designed for secure, scalable deployment using Docker and best practices.

## ✨ Features

*   **Data Collection & Preprocessing:** Handles historical and live OHLCV data loading and cleaning.
*   **Technical Analysis:** Calculates indicators such as:
    *   Relative Strength Index (RSI)
    *   Bollinger Bands (BB_upper, BB_lower, BB_mid, BB_pct_b)
    *   Simple Moving Averages (SMA_20, SMA_50)
    *   Moving Average Crossover Signal (MA_cross)
    *   Price Momentum
    *   Average True Range (ATR)
*   **Advanced Time Series Features:**
    *   **GARCH**: Volatility modeling with conditional variance forecasts
    *   **ARIMA**: Autoregressive forecasts with confidence intervals
    *   **HMM**: Hidden Markov Model regime detection (bull/bear/sideways)
    *   **Kalman Filter**: Smooth price level and trend estimation
    *   **tsfresh**: Automated extraction of statistically significant features
*   **Feature Engineering:** Generates trading signals and prepares features for the ML model.
*   **Advanced Time Series Modeling:** Includes GARCH, ARIMA, HMM, Kalman Filter, and tsfresh automated feature extraction for enhanced predictive power (see `TIMESERIES_FEATURES_GUIDE.md` for details).
*   **Model Training & Optimization:**
    *   Uses **XGBoost Classifier** for buy/sell/hold predictions
    *   Handles class imbalance with SMOTE
    *   Hyperparameter optimization via `RandomizedSearchCV`
*   **Backtesting Framework:** Simulates trades on historical data, calculates key metrics (Total Return, Win Rate, etc.)
*   **Interactive Visualization:**
    *   **Plotly**: Generates detailed, interactive charts (candlesticks, signals, indicators, portfolio value)
    *   **Streamlit**: Serves dashboards and analytics web apps, embedding Plotly charts for user interaction
*   **API Service:**
    *   **FastAPI**: REST API for model inference, health checks, and management
    *   Input validation, robust error handling, and secure authentication (API key & OAuth2/JWT)
*   **Experiment Tracking:**
    *   **MLflow**: Track model training, parameters, and results
*   **Database Integration:**
    *   **Postgres** with SQLAlchemy ORM for trade and user data
*   **Notifications:**
    *   Email and Telegram alerts for critical events
*   **Live Trading (Binance Testnet):**
    *   Real-time trading simulation with Binance Testnet API
*   **Dockerized & Orchestrated:**
    *   Dockerfile and docker-compose for seamless deployment of all services (bot, API, Streamlit, MLflow, Postgres)
*   **Security & Production Readiness:**
    *   Environment variables for secrets, JWT authentication, error logging, and best practices for deployment

## 🚀 Getting Started

### Prerequisites

*   **Python:** 3.12.x
*   **uv:** Fast Python package installer (`pip install uv`)
*   **Docker Desktop:** [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)

### Installation & Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Theesthan/CrytoBot.git
    cd CrytoBot
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Using Python venv
    python -m venv .venv
    
    # On Windows
    .venv\Scripts\activate
    
    # On Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    *   Copy `.env.example` to `.env`:
        ```bash
        cp .env.example .env
        ```
    *   Edit `.env` and fill in your credentials:
        - Binance Testnet API keys (get from [Binance Testnet](https://testnet.binance.vision/))
        - JWT secret key (generate with: `python -c "import secrets; print(secrets.token_hex(32))"`)
        - Database credentials
        - Email/Telegram notification settings (optional)
    *   **⚠️ NEVER commit `.env` to version control!**

5.  **Initialize Database (if running locally):**
    ```bash
    python scripts/init_database.py
    ```

## ⚙️ Configuration

All major parameters are in `src/config.py` and `.env`:
*   `TRADE_SYMBOL`, `TRADE_INTERVAL`, `TRADE_QUANTITY`
*   `CONFIDENCE_THRESHOLD` (model predictions)
*   Technical indicator windows (`RSI_WINDOW`, `BB_WINDOW`, `SMAs` etc.)
*   `MODE` (`live` or `backtest`)
*   `BINANCE_TESTNET` (True/False)
*   Database, email, Telegram, and JWT settings

## 💻 Usage

### Backtesting Mode (Local)

1.  Set `MODE=backtest` in `.env`
2.  Ensure your virtual environment is activated
3.  Run:
    ```bash
    python main.py
    ```
4.  A Plotly chart will open in your browser showing backtest results

### Live Trading Mode (Docker Compose)

**⚠️ WARNING: Live trading can place real orders! Always test with Binance Testnet first.**

1.  Set `MODE=live` in `.env`
2.  Configure your Binance Testnet API keys in `.env`
3.  Build and start all services:
    ```bash
    docker-compose up --build
    ```
4.  Access the services:
    *   **Trading Bot**: Running in background with logs in terminal
    *   **FastAPI**: http://localhost:8000/docs
    *   **Streamlit Dashboard**: http://localhost:8501
    *   **MLflow UI**: http://localhost:5000
    *   **PostgreSQL**: localhost:5432

5.  Stop all services:
    ```bash
    docker-compose down
    ```

### API & Dashboard

*   **FastAPI:** http://localhost:8000/docs (interactive API docs)
*   **Streamlit Dashboard:** http://localhost:8501
*   **MLflow Tracking UI:** http://localhost:5000

## 📁 Project Structure

Key files and folders:
*   `src/` — All Python modules
*   `src/api.py` — FastAPI app
*   `src/streamlit_app.py` — Streamlit dashboard
*   `src/model_manager.py` — ML model loading/inference
*   `src/db.py` — Database models and ORM
*   `src/feature_engineer.py` — Feature engineering
*   `src/binance_manager.py` — Binance API integration
*   `src/visualizer.py` — Plotly chart generation
*   `src/config.py` — Configuration
*   `main.py` — Entry point, trading logic
*   `data/` — Data files (e.g., test_df_features.csv)
*   `requirements.txt`, `pyproject.toml`, `Dockerfile`, `docker-compose.yml`, `.env`, `.dockerignore`

## 🔒 Security & Best Practices

*   All secrets and credentials are managed via `.env`.
*   API endpoints are protected with API key and OAuth2/JWT authentication.
*   Error handling, logging, and notifications are integrated.
*   Docker Compose orchestrates all services for production deployment.
*   Use strong, unique secrets for JWT and database credentials.
*   Regularly audit and update dependencies in `pyproject.toml` and `requirements.txt`.

## 📈 Visualization: Plotly & Streamlit

*   **Plotly** is used for generating interactive charts and analytics.
*   **Streamlit** serves these charts and dashboards to users via a web interface.

## 🛠️ Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational purposes only. Cryptocurrency trading carries significant risk. Never trade with money you cannot afford to lose. Always test strategies thoroughly on testnet before considering live deployment.

## 📞 Support

For questions or issues, please open an issue on GitHub.

---

**Happy Trading! 🚀**

