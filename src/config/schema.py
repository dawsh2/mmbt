# config/schema.py

"""
Configuration schema for validation.
"""

CONFIG_SCHEMA = {
    'type': 'dict',
    'properties': {
        'backtester': {
            'type': 'dict',
            'properties': {
                'market_simulation': {
                    'type': 'dict',
                    'properties': {
                        'slippage_model': {
                            'type': 'str',
                            'enum': ['none', 'fixed', 'volume'],
                            'default': 'fixed'
                        },
                        'slippage_bps': {
                            'type': 'int',
                            'min': 0,
                            'max': 100,
                            'default': 5
                        },
                        'price_impact': {
                            'type': 'float',
                            'min': 0,
                            'max': 1,
                            'default': 0.1
                        },
                        'fee_model': {
                            'type': 'str',
                            'enum': ['none', 'fixed', 'tiered'],
                            'default': 'fixed'
                        },
                        'fee_bps': {
                            'type': 'int',
                            'min': 0,
                            'max': 100,
                            'default': 10
                        }
                    }
                },
                'initial_capital': {
                    'type': 'float',
                    'min': 1000,
                    'default': 100000
                },
                'close_positions_eod': {
                    'type': 'bool',
                    'default': True
                }
            }
        },
        'position_management': {
            'type': 'dict',
            'properties': {
                'position_sizing': {
                    'type': 'dict',
                    'properties': {
                        'method': {
                            'type': 'str',
                            'enum': ['fixed', 'percent_equity', 'volatility', 'kelly'],
                            'default': 'percent_equity'
                        },
                        'percent': {
                            'type': 'float',
                            'min': 0.001,
                            'max': 1.0,
                            'default': 0.02
                        },
                        'fixed_size': {
                            'type': 'int',
                            'min': 1,
                            'default': 100
                        },
                        'risk_pct': {
                            'type': 'float',
                            'min': 0.001,
                            'max': 0.1,
                            'default': 0.01
                        },
                        'kelly_fraction': {
                            'type': 'float',
                            'min': 0.1,
                            'max': 1.0,
                            'default': 0.5
                        }
                    }
                },
                'risk_management': {
                    'type': 'dict',
                    'properties': {
                        'max_position_pct': {
                            'type': 'float',
                            'min': 0.01,
                            'max': 1.0,
                            'default': 0.25
                        },
                        'max_drawdown_pct': {
                            'type': 'float',
                            'min': 0.01,
                            'max': 0.5,
                            'default': 0.10
                        },
                        'use_stop_loss': {
                            'type': 'bool',
                            'default': False
                        },
                        'stop_loss_pct': {
                            'type': 'float',
                            'min': 0.01,
                            'max': 0.5,
                            'default': 0.05
                        },
                        'use_take_profit': {
                            'type': 'bool',
                            'default': False
                        },
                        'take_profit_pct': {
                            'type': 'float',
                            'min': 0.01,
                            'max': 0.5,
                            'default': 0.10
                        }
                    }
                },
                'allocation': {
                    'type': 'dict',
                    'properties': {
                        'method': {
                            'type': 'str',
                            'enum': ['equal', 'volatility_parity', 'optimize'],
                            'default': 'equal'
                        },
                        'max_instruments': {
                            'type': 'int',
                            'min': 1,
                            'default': 10
                        }
                    }
                }
            }
        },
        'signals': {
            'type': 'dict',
            'properties': {
                'processing': {
                    'type': 'dict',
                    'properties': {
                        'use_filtering': {
                            'type': 'bool',
                            'default': False
                        },
                        'filter_type': {
                            'type': 'str',
                            'enum': ['moving_average', 'kalman', 'exponential'],
                            'default': 'moving_average'
                        },
                        'window_size': {
                            'type': 'int',
                            'min': 2,
                            'max': 100,
                            'default': 5
                        },
                        'use_transformations': {
                            'type': 'bool',
                            'default': False
                        },
                        'wavelet_type': {
                            'type': 'str',
                            'enum': ['db1', 'db4', 'sym4', 'haar'],
                            'default': 'db1'
                        },
                        'wavelet_level': {
                            'type': 'int',
                            'min': 1,
                            'max': 6,
                            'default': 3
                        }
                    }
                },
                'confidence': {
                    'type': 'dict',
                    'properties': {
                        'use_confidence_score': {
                            'type': 'bool',
                            'default': False
                        },
                        'min_confidence': {
                            'type': 'float',
                            'min': 0.0,
                            'max': 1.0,
                            'default': 0.5
                        },
                        'prior_accuracy': {
                            'type': 'float',
                            'min': 0.1,
                            'max': 0.9,
                            'default': 0.5
                        }
                    }
                }
            }
        },
        'regime_detection': {
            'type': 'dict',
            'properties': {
                'detector_type': {
                    'type': 'str',
                    'enum': ['trend', 'volatility', 'composite'],
                    'default': 'trend'
                },
                'trend': {
                    'type': 'dict',
                    'properties': {
                        'adx_period': {
                            'type': 'int',
                            'min': 5,
                            'max': 50,
                            'default': 14
                        },
                        'adx_threshold': {
                            'type': 'float',
                            'min': 10,
                            'max': 50,
                            'default': 25
                        }
                    }
                },
                'volatility': {
                    'type': 'dict',
                    'properties': {
                        'lookback_period': {
                            'type': 'int',
                            'min': 5,
                            'max': 50,
                            'default': 20
                        },
                        'volatility_threshold': {
                            'type': 'float',
                            'min': 0.001,
                            'max': 0.1,
                            'default': 0.015
                        }
                    }
                },
                'composite': {
                    'type': 'dict',
                    'properties': {
                        'combination_method': {
                            'type': 'str',
                            'enum': ['majority', 'consensus', 'weighted'],
                            'default': 'majority'
                        },
                        'weights': {
                            'type': 'dict',
                            'default': {}
                        }
                    }
                }
            }
        }
    }
}
