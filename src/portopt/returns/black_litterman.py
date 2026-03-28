"""
BlackLittermanReturns — Black-Litterman expected returns.

Receives already-computed returns (no internal pct_change).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..core.returns import ExpectedReturnsModel


class BlackLittermanReturns(ExpectedReturnsModel):
    """
    Retornos esperados bajo el modelo Black-Litterman.

    Combina los retornos implícitos de equilibrio con las views del gestor.

        Π = δ * Σ * w_mkt           (retornos de equilibrio)
        μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹Π + P'Ω⁻¹Q]

    Parameters
    ----------
    risk_aversion : float
        Coeficiente de aversión al riesgo δ.
    tau : float
        Escalar de incertidumbre sobre el prior.
    views : dict, optional
        ``{ticker: expected_return}`` — views absolutas del gestor.
    view_confidences : np.ndarray, optional
        Diagonal de Ω (varianzas de los errores de las views).
        Si es None, se usa τ * varianza histórica de cada activo.
    """

    def __init__(
        self,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        views: Optional[Dict[str, float]] = None,
        view_confidences: Optional[np.ndarray] = None,
    ):
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.views = views or {}
        self.view_confidences = view_confidences
        self.market_implied_returns_: Optional[np.ndarray] = None
        self.cov_matrix_: Optional[np.ndarray] = None
        self.tickers_: Optional[list] = None

    def fit(self, returns: pd.DataFrame) -> "BlackLittermanReturns":
        """
        Calcula los retornos de equilibrio implícitos a partir de los retornos.

        Parameters
        ----------
        returns : pd.DataFrame
            Retornos ya computados. Shape (n_periods, n_assets).
        """
        self.tickers_ = list(returns.columns)
        self.cov_matrix_ = returns.cov().values
        n = len(self.tickers_)
        market_weights = np.ones(n) / n

        self.market_implied_returns_ = (
            self.risk_aversion * self.cov_matrix_ @ market_weights
        )

        return self

    def predict(self) -> np.ndarray:
        """Devuelve los retornos BL combinando equilibrio y views."""
        if self.market_implied_returns_ is None:
            raise RuntimeError("Llama a fit() antes de predict().")

        if not self.views:
            return self.market_implied_returns_.copy()

        # Construir P (pick matrix) y Q (view returns) para views absolutas
        view_assets = [t for t in self.tickers_ if t in self.views]
        if not view_assets:
            return self.market_implied_returns_.copy()

        k = len(view_assets)
        n = len(self.tickers_)
        P = np.zeros((k, n))
        Q = np.zeros(k)
        for i, asset in enumerate(view_assets):
            j = self.tickers_.index(asset)
            P[i, j] = 1.0
            Q[i] = self.views[asset]

        # Omega: incertidumbre en las views
        if self.view_confidences is not None:
            omega = np.diag(self.view_confidences[:k])
        else:
            omega = np.diag(np.diag(self.tau * P @ self.cov_matrix_ @ P.T))

        tau_sigma = self.tau * self.cov_matrix_
        try:
            tau_sigma_inv = np.linalg.inv(tau_sigma)
            omega_inv = np.linalg.inv(omega)
            left = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
            right = tau_sigma_inv @ self.market_implied_returns_ + P.T @ omega_inv @ Q
            return left @ right
        except np.linalg.LinAlgError:
            # Fallback: combinar linealmente con igual peso
            bl = self.market_implied_returns_.copy()
            for i, asset in enumerate(view_assets):
                j = self.tickers_.index(asset)
                bl[j] = 0.5 * bl[j] + 0.5 * Q[i]
            return bl

    def __repr__(self) -> str:
        return (
            f"BlackLittermanReturns(risk_aversion={self.risk_aversion}, "
            f"tau={self.tau}, n_views={len(self.views)})"
        )
