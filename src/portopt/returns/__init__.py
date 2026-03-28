"""
Returns models package.
"""

from .historical import HistoricalReturns
from .capm import CAPMReturns
from .black_litterman import BlackLittermanReturns
from .ml_forecast import MLForecastReturns
from .consolidator import ReturnsConsolidator
from .data_downloader import (
    AlpacaDataDownloader,
    MassiveDataDownloader,
    RiskFreeRateDownloader,
)

__all__ = [
    "HistoricalReturns",
    "CAPMReturns",
    "BlackLittermanReturns",
    "MLForecastReturns",
    "ReturnsConsolidator",
    "AlpacaDataDownloader",
    "MassiveDataDownloader",
    "RiskFreeRateDownloader",
]
