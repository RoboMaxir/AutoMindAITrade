"""
Configuration file for Professional Crypto Trading Agent

This contains all configurable parameters for the trading system.
"""

# Trading Configuration
TRADING_CONFIG = {
    # Symbols to trade (format: BASEQUOTE like BTCUSDT, ETHUSDT)
    'symbols': ['BTCUSDT', 'ETHUSDT'],
    
    # Trading intervals (in minutes)
    'check_interval_minutes': 60,  # How often to check for trades
    
    # Risk management
    'max_position_size': 0.1,      # Maximum 10% of capital per position
    'max_drawdown': 0.15,          # Maximum 15% drawdown allowed
    'risk_per_trade': 0.02,        # Risk 2% of capital per trade maximum
    'stop_loss_pct': 0.02,         # 2% stop loss
    'take_profit_pct': 0.05,       # 5% take profit
    
    # Starting capital
    'initial_capital': 10000.0,    # Starting with $10,000
    
    # Technical analysis settings
    'rsi_overbought': 70,          # RSI level considered overbought
    'rsi_oversold': 30,            # RSI level considered oversold
    'atr_period': 14,              # ATR calculation period
    'ema_fast': 9,                 # Fast EMA period
    'ema_slow': 21,                # Slow EMA period
    
    # Sentiment thresholds
    'sentiment_positive_threshold': 0.6,
    'sentiment_negative_threshold': -0.6,
}

# Exchange Configuration
EXCHANGE_CONFIG = {
    # For simulation mode
    'use_simulation': True,
    
    # Real exchange configuration would go here
    'exchange_name': 'binance',  # Options: 'binance', 'coinbase', 'kucoin', etc.
    'api_key': '',               # Your API key (don't commit this!)
    'api_secret': '',            # Your API secret (don't commit this!)
    'testnet': True,             # Use testnet for development
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'file': 'trading_agent.log',
    'console_output': True,
}

# Data Configuration
DATA_CONFIG = {
    'kline_interval': '1h',        # K-line interval ('1m', '5m', '15m', '1h', '4h', '1d')
    'kline_limit': 100,           # Number of k-lines to fetch for analysis
    'data_source': 'exchange',    # Source of market data
}

# News/Sentiment Configuration
NEWS_CONFIG = {
    'enable_news_analysis': True,
    'news_sources': [
        'cryptopanic',
        'rss_feeds',
        'social_media'
    ],
    'sentiment_api_key': '',      # API key for sentiment analysis service
}

# Advanced Features Configuration
ADVANCED_CONFIG = {
    # Machine learning features
    'enable_ml_patterns': False,   # Enable ML-based pattern recognition
    'ml_model_path': './models/',  # Path to ML models
    
    # Backtesting
    'enable_backtesting': False,   # Enable backtesting mode
    'backtest_start_date': '2023-01-01',
    'backtest_end_date': '2023-12-31',
    
    # Performance tracking
    'track_performance': True,
    'metrics_save_path': './performance/',
}