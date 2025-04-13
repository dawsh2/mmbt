# config/defaults.py

"""
Default configuration values.
"""

DEFAULT_CONFIG = {
    'backtester': {
        'market_simulation': {
            'slippage_model': 'fixed',
            'slippage_bps': 5,
            'price_impact': 0.1,
            'fee_model': 'fixed',
            'fee_bps': 10
        },
        'initial_capital': 100000,
        'close_positions_eod': True
    },
    'position_management': {
        'position_sizing': {
            'method': 'percent_equity',
            'percent': 0.02,
            'fixed_size': 100,
            'risk_pct': 0.01,
            'kelly_fraction': 0.5
        },
        'risk_management': {
            'max_position_pct': 0.25,
            'max_drawdown_pct': 0.10,
            'use_stop_loss': False,
            'stop_loss_pct': 0.05,
            'use_take_profit': False,
            'take_profit_pct': 0.10
        },
        'allocation': {
            'method': 'equal',
            'max_instruments': 10
        }
    },
    'signals': {
        'processing': {
            'use_filtering': False,
            'filter_type': 'moving_average',
            'window_size': 5,
            'use_transformations': False,
            'wavelet_type': 'db1',
            'wavelet_level': 3
        },
        'confidence': {
            'use_confidence_score': False,
            'min_confidence': 0.5,
            'prior_accuracy': 0.5
        }
    },
    'regime_detection': {
        'detector_type': 'trend',
        'trend': {
            'adx_period': 14,
            'adx_threshold': 25
        },
        'volatility': {
            'lookback_period': 20,
            'volatility_threshold': 0.015
        },
        'composite': {
            'combination_method': 'majority',
            'weights': {}
        }
    }
}
