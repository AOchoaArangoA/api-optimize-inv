"""
HistoricalReturns — Expected returns from historical average.

Receives already-computed returns (no internal pct_change).
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..core.returns import ExpectedReturnsModel


class HistoricalReturns(ExpectedReturnsModel):
    """
    Retornos esperados estimados como promedio histórico.

    Parameters
    ----------
    method : {'mean', 'median', 'exponential'}
        Método de agregación temporal.
    window : int, optional
        Número de períodos más recientes a usar.  None = todos.
    """

    def __init__(self, method: str = "mean", window: Optional[int] = None):
        self.method = method
        self.window = window
        self.expected_returns_: Optional[np.ndarray] = None
        self.tickers_: Optional[list] = None

    def fit(self, returns: pd.DataFrame) -> "HistoricalReturns":
        """
        Ajusta el modelo a los retornos históricos.

        Parameters
        ----------
        returns : pd.DataFrame
            Retornos ya computados. Shape (n_periods, n_assets).
        """
        self.tickers_ = list(returns.columns)
        data = returns.tail(self.window) if self.window else returns

        if self.method == "mean":
            self.expected_returns_ = data.mean().values
        elif self.method == "median":
            self.expected_returns_ = data.median().values
        elif self.method == "exponential":
            span = self.window // 2 if self.window else 30
            self.expected_returns_ = data.ewm(span=span).mean().iloc[-1].values
        else:
            raise ValueError(
                f"method='{self.method}' no reconocido. "
                "Opciones: 'mean', 'median', 'exponential'."
            )

        return self

    def predict(self) -> np.ndarray:
        """Devuelve el vector de retornos esperados estimado en fit()."""
        if self.expected_returns_ is None:
            raise RuntimeError("Llama a fit() antes de predict().")
        return self.expected_returns_

    def __repr__(self) -> str:
        return f"HistoricalReturns(method={self.method!r}, window={self.window})"
