"""
SampleCovariance — Direct historical covariance estimator.

Receives already-computed returns (no internal pct_change).
"""

from typing import Optional

import numpy as np
import pandas as pd

from ..core.risk import RiskModel


class SampleCovariance(RiskModel):
    """
    Covarianza muestral directa a partir de retornos históricos.

    Parameters
    ----------
    window : int, optional
        Número de períodos más recientes a usar.  None = todos.
    annualize : bool
        Si True, multiplica por 252 (asume datos diarios).
    """

    def __init__(self, window: Optional[int] = None, annualize: bool = True):
        self.window = window
        self.annualize = annualize
        self.covariance_matrix_: Optional[np.ndarray] = None

    def fit(self, returns: pd.DataFrame) -> "SampleCovariance":
        """
        Estima la covarianza muestral.

        Parameters
        ----------
        returns : pd.DataFrame
            Retornos ya computados. Shape (n_periods, n_assets).
        """
        data = returns.tail(self.window) if self.window else returns
        cov = data.cov().values
        self.covariance_matrix_ = cov * 252 if self.annualize else cov
        return self

    def covariance(self) -> np.ndarray:
        """Devuelve la matriz de covarianza estimada en fit()."""
        if self.covariance_matrix_ is None:
            raise RuntimeError("Llama a fit() antes de covariance().")
        return self.covariance_matrix_

    def __repr__(self) -> str:
        return f"SampleCovariance(window={self.window}, annualize={self.annualize})"
