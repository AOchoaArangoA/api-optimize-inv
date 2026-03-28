"""
ShrinkageCovariance — Ledoit-Wolf shrinkage covariance estimator.

Receives already-computed returns (no internal pct_change).
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from ..core.risk import RiskModel


class ShrinkageCovariance(RiskModel):
    """
    Covarianza regularizada con estimador Ledoit-Wolf.

    Combina la covarianza muestral con una matriz target para mejorar
    la estabilidad numérica cuando n_assets es grande relativo a n_periods.

    Parameters
    ----------
    annualize : bool
        Si True, multiplica por 252 (asume datos diarios).
    """

    def __init__(self, annualize: bool = True):
        self.annualize = annualize
        self.covariance_matrix_: Optional[np.ndarray] = None
        self.shrinkage_: Optional[float] = None

    def fit(self, returns: pd.DataFrame) -> "ShrinkageCovariance":
        """
        Ajusta el estimador Ledoit-Wolf.

        Parameters
        ----------
        returns : pd.DataFrame
            Retornos ya computados. Shape (n_periods, n_assets).
        """
        lw = LedoitWolf()
        lw.fit(returns.values)
        self.shrinkage_ = float(lw.shrinkage_)
        cov = lw.covariance_
        self.covariance_matrix_ = cov * 252 if self.annualize else cov
        return self

    def covariance(self) -> np.ndarray:
        """Devuelve la matriz de covarianza shrunken estimada en fit()."""
        if self.covariance_matrix_ is None:
            raise RuntimeError("Llama a fit() antes de covariance().")
        return self.covariance_matrix_

    def __repr__(self) -> str:
        shrinkage = f"{self.shrinkage_:.4f}" if self.shrinkage_ is not None else "N/A"
        return f"ShrinkageCovariance(annualize={self.annualize}, shrinkage={shrinkage})"
