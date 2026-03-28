"""
MeanVarianceOptimization — Markowitz mean-variance portfolio optimization.
"""

from typing import List, Optional

import numpy as np

from ...core.formulation import OptimizationProblem
from ...core.constraint import Constraint


class MeanVarianceOptimization(OptimizationProblem):
    """
    Optimización Media-Varianza de Markowitz.

    Objetivo (minimización):   -μ'w + (λ/2) * w'Σw
    Equivalente a maximizar:    μ'w - (λ/2) * w'Σw

    Parameters
    ----------
    n_assets : int
    expected_returns : np.ndarray  shape (n_assets,)
    covariance_matrix : np.ndarray  shape (n_assets, n_assets)
    risk_aversion : float
        Parámetro λ ≥ 0.  Mayor λ → menor riesgo.
    constraints : list of Constraint, optional
    """

    def __init__(
        self,
        n_assets: int,
        expected_returns: Optional[np.ndarray] = None,
        covariance_matrix: Optional[np.ndarray] = None,
        risk_aversion: float = 1.0,
        constraints: Optional[List[Constraint]] = None,
    ):
        super().__init__(n_assets, constraints)
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_aversion = risk_aversion

    # ------------------------------------------------------------------
    # SciPy interface
    # ------------------------------------------------------------------

    def objective(self, weights: np.ndarray) -> float:
        """Negativo de la utility (para minimización con SciPy)."""
        ret = float(self.expected_returns @ weights)
        var = float(weights @ self.covariance_matrix @ weights)
        return -ret + (self.risk_aversion / 2) * var

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return -self.expected_returns + self.risk_aversion * (
            self.covariance_matrix @ weights
        )

    # ------------------------------------------------------------------
    # CVXPY interface
    # ------------------------------------------------------------------

    def build_cvxpy_objective(self, w):
        import cvxpy as cp
        return cp.Minimize(
            -self.expected_returns @ w
            + (self.risk_aversion / 2) * cp.quad_form(w, self.covariance_matrix)
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def portfolio_return(self, weights: np.ndarray) -> float:
        return float(self.expected_returns @ weights)

    def portfolio_variance(self, weights: np.ndarray) -> float:
        return float(weights @ self.covariance_matrix @ weights)

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        return float(np.sqrt(self.portfolio_variance(weights)))
