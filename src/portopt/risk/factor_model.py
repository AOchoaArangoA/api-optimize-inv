"""
FactorModel — PCA-based factor covariance estimator.

Receives already-computed returns (no internal pct_change).
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ..core.risk import RiskModel


class FactorModel(RiskModel):
    """
    Covarianza basada en un modelo de factores (PCA).

        Σ = B * F * B' + D

    donde:
        - B  — factor loadings  (n_assets × n_factors)
        - F  — covarianza de factores  (n_factors × n_factors)
        - D  — varianza idiosincrática diagonal

    Parameters
    ----------
    n_factors : int
        Número de factores a extraer.
    annualize : bool
        Si True, multiplica por 252 (asume datos diarios).
    """

    def __init__(self, n_factors: int = 5, annualize: bool = True):
        self.n_factors = n_factors
        self.annualize = annualize
        self.factor_loadings_: Optional[np.ndarray] = None
        self.factor_covariance_: Optional[np.ndarray] = None
        self.idiosyncratic_variance_: Optional[np.ndarray] = None

    def fit(self, returns: pd.DataFrame) -> "FactorModel":
        """
        Ajusta el modelo PCA de factores.

        Parameters
        ----------
        returns : pd.DataFrame
            Retornos ya computados. Shape (n_periods, n_assets).
        """
        n_factors = min(self.n_factors, returns.shape[1], returns.shape[0])
        pca = PCA(n_components=n_factors)
        factor_returns = pca.fit_transform(returns.values)

        self.factor_loadings_ = pca.components_.T         # (n_assets, n_factors)
        self.factor_covariance_ = np.cov(factor_returns.T)  # (n_factors, n_factors)

        reconstructed = factor_returns @ pca.components_
        residuals = returns.values - reconstructed
        self.idiosyncratic_variance_ = np.var(residuals, axis=0)

        return self

    def covariance(self) -> np.ndarray:
        """Devuelve la matriz de covarianza del modelo de factores estimada en fit()."""
        if self.factor_loadings_ is None:
            raise RuntimeError("Llama a fit() antes de covariance().")

        cov = (
            self.factor_loadings_ @ self.factor_covariance_ @ self.factor_loadings_.T
            + np.diag(self.idiosyncratic_variance_)
        )
        return cov * 252 if self.annualize else cov

    def __repr__(self) -> str:
        return f"FactorModel(n_factors={self.n_factors}, annualize={self.annualize})"
