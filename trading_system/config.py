"""
Trading System Configuration
"""

# Exchange Configuration
EXCHANGE_CONFIG = {
    'binance': {
        'api_key': '',
        'secret': '',
        'sandbox': True,  # Use testnet initially
    },
    'okx': {
        'api_key': '',
        'secret': '',
        'password': '',
        'sandbox': True,
    },
    'coinbase': {
        'api_key': '',
        'secret': '',
        'sandbox': True,
    }
}

# Trading Parameters
TRADING_PARAMS = {
    'default_symbol': 'BTC/USDT',
    'timeframe': '1h',  # Technical analysis timeframe
    'risk_per_trade': 0.02,  # 2% risk per trade
    'max_position_size': 0.10,  # 10% max position size
    'max_drawdown': 0.15,  # 15% max drawdown before pause
    'min_balance_usd': 100,  # Minimum balance to continue trading
}

# Agent Configuration
AGENT_CONFIG = {
    'technical_analysis': {
        'lookback_period': 200,  # Number of candles to analyze
        'indicators': ['ema', 'rsi', 'atr', 'macd'],
        'signal_threshold': 0.6,  # Confidence threshold for signals
    },
    'fundamental_analysis': {
        'news_sources': [
            'https://api.coindesk.com/v1/bpi/currentprice.json',
            'https://min-api.cryptocompare.com/data/v2/news/',
        ],
        'sentiment_threshold': 0.3,  # Below this is negative sentiment
        'risk_tolerance': 0.7,  # How much risk we tolerate
    },
    'risk_management': {
        'position_sizing_method': 'kelly_criterion',  # Options: kelly_criterion, fixed_fraction, volatility_target
        'max_correlation': 0.7,  # Max correlation between positions
        'stop_loss_type': 'atr',  # atr, percentage, volatility
        'take_profit_ratio': 2.0,  # Take profit at 2x stop loss distance
    },
    'decision_orchestrator': {
        'confidence_threshold': 0.7,  # Minimum confidence to execute trade
        'conflict_resolution': 'conservative',  # conservative, aggressive, weighted_average
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': ['console', 'file'],
    'log_file': '/workspace/trading_system/logs/trading_system.log'
}

# Market Conditions
MARKET_CONDITIONS = {
    'bull_trend': {'rsi_threshold': 50, 'volume_multiplier': 1.2},
    'bear_trend': {'rsi_threshold': 50, 'volume_multiplier': 1.2},
    'sideways': {'rsi_threshold': 40, 'upper_threshold': 60},
    'high_volatility': {'atr_multiplier': 1.5},
}