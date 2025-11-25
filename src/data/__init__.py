"""
Módulo de coleta e preparação de dados
"""

from .data_fetcher import (
    fetch_binance_klines,
    fetch_binance_klines_multiple,
    prepare_binance_data_for_training,
    BINANCE_SYMBOLS,
    CRYPTO_LIST
)

from .indicators import (
    compute_indicators,
    select_orthogonal_features,
    build_indicator_dataset
)

__all__ = [
    'fetch_binance_klines',
    'fetch_binance_klines_multiple',
    'prepare_binance_data_for_training',
    'BINANCE_SYMBOLS',
    'CRYPTO_LIST',
    'compute_indicators',
    'select_orthogonal_features',
    'build_indicator_dataset'
]

