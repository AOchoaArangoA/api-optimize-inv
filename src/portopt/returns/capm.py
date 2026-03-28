"""
CAPMReturns — Capital Asset Pricing Model expected returns.

Receives already-computed returns (no internal pct_change).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..core.returns import ExpectedReturnsModel


class CAPMReturns(ExpectedReturnsModel):
    """
    Retornos esperados bajo el modelo CAPM.

        E[R_i] = R_f + β_i * (E[R_m] - R_f)

    Parameters
    ----------
    risk_free_rate : float
        Tasa libre de riesgo (misma frecuencia que los retornos).
    market_index : str, optional
        Columna del índice de mercado en el DataFrame.
        Si es None o no existe, se usa el promedio equiponderado.
    """

    def __init__(
        self,
        risk_free_rate: float = 0.0,
        market_index: Optional[str] = None,
    ):
        self.risk_free_rate = risk_free_rate
        self.market_index = market_index
        self.betas_: Optional[Dict[str, float]] = None
        self.market_return_: Optional[float] = None
        self.tickers_: Optional[list] = None

    def fit(self, returns: pd.DataFrame) -> "CAPMReturns":
        """
        Estima betas y retorno de mercado a partir de los retornos.

        Parameters
        ----------
        returns : pd.DataFrame
            Retornos ya computados. Shape (n_periods, n_assets).
        """
        self.tickers_ = list(returns.columns)

        if self.market_index and self.market_index in returns.columns:
            market_returns = returns[self.market_index]
        else:
            market_returns = returns.mean(axis=1)

        self.market_return_ = float(market_returns.mean())
        market_var = float(market_returns.var())

        self.betas_ = {}
        for asset in returns.columns:
            if asset == self.market_index:
                self.betas_[asset] = 1.0
            else:
                cov = float(np.cov(returns[asset], market_returns)[0, 1])
                self.betas_[asset] = cov / market_var if market_var > 0 else 1.0

        return self

    def predict(self) -> np.ndarray:
        """Devuelve el vector de retornos esperados CAPM estimado en fit()."""
        if self.betas_ is None or self.market_return_ is None:
            raise RuntimeError("Llama a fit() antes de predict().")

        market_premium = self.market_return_ - self.risk_free_rate
        return np.array([
            self.risk_free_rate + self.betas_[t] * market_premium
            for t in self.tickers_
        ])

    def __repr__(self) -> str:
        return (
            f"CAPMReturns(risk_free_rate={self.risk_free_rate}, "
            f"market_index={self.market_index!r})"
        )
