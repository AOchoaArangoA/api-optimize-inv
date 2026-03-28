"""
GARCHModel — GARCH-based covariance estimator.

Receives already-computed returns (no internal pct_change).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from arch import arch_model

from ..core.risk import RiskModel


class GARCHModel(RiskModel):
    """
    Covarianza dinámica usando modelos GARCH univariados por activo.

    Estima la volatilidad condicional forward-looking para cada activo
    y construye la matriz de covarianza como Σ = D * R * D donde:
        - D = diag(σ_GARCH) — volatilidades forecasted
        - R = correlación histórica muestral

    Parameters
    ----------
    p, q : int
        Órdenes del modelo GARCH(p, q).
    mean : str
        Especificación de la media ('Constant', 'Zero', 'AR').
    vol : str
        Especificación de volatilidad ('GARCH', 'EGARCH', 'GJR-GARCH').
    annualize : bool
        Si True, multiplica por 252 (asume datos diarios).
    """

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        mean: str = "Constant",
        vol: str = "GARCH",
        annualize: bool = True,
    ):
        self.p = p
        self.q = q
        self.mean = mean
        self.vol = vol
        self.annualize = annualize
        self.forecasted_vol_: Dict[str, float] = {}
        self.corr_matrix_: Optional[np.ndarray] = None
        self.tickers_: Optional[list] = None

    def fit(self, returns: pd.DataFrame) -> "GARCHModel":
        """
        Ajusta modelos GARCH univariados y calcula correlaciones históricas.

        Parameters
        ----------
        returns : pd.DataFrame
            Retornos ya computados. Shape (n_periods, n_assets).
        """
        self.tickers_ = list(returns.columns)
        self.corr_matrix_ = returns.corr().values

        for asset in returns.columns:
            series = returns[asset] * 100  # escalar para estabilidad numérica
            model = arch_model(series, mean=self.mean, vol=self.vol, p=self.p, q=self.q)
            try:
                res = model.fit(disp="off")
                forecast = res.forecast(horizon=1)
                self.forecasted_vol_[asset] = float(
                    np.sqrt(forecast.variance.values[-1, 0])
                )
            except Exception:
                self.forecasted_vol_[asset] = float(returns[asset].std() * 100)

        return self

    def covariance(self) -> np.ndarray:
        """Devuelve la matriz de covarianza dinámica estimada en fit()."""
        if self.tickers_ is None:
            raise RuntimeError("Llama a fit() antes de covariance().")

        # Desescalar (/ 100) y construir D
        vols = np.array([self.forecasted_vol_[t] / 100 for t in self.tickers_])
        D = np.diag(vols)
        cov = D @ self.corr_matrix_ @ D

        return cov * 252 if self.annualize else cov

    def __repr__(self) -> str:
        return f"GARCHModel(p={self.p}, q={self.q}, vol={self.vol!r})"
