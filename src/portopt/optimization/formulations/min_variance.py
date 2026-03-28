"""
MinVarianceOptimization — Global minimum variance portfolio.
"""

from typing import List, Optional

import numpy as np

from ...core.formulation import OptimizationProblem
from ...core.constraint import Constraint


class MinVarianceOptimization(OptimizationProblem):
    """
    Mínima Varianza Global.

    Objetivo:   min  w'Σw

    Parameters
    ----------
    n_assets : int
    covariance_matrix : np.ndarray  shape (n_assets, n_assets)
    constraints : list of Constraint, optional
    """

    def __init__(
        self,
        n_assets: int,
        covariance_matrix: Optional[np.ndarray] = None,
        constraints: Optional[List[Constraint]] = None,
    ):
        super().__init__(n_assets, constraints)
        self.covariance_matrix = covariance_matrix

    # ------------------------------------------------------------------
    # SciPy interface
    # ------------------------------------------------------------------

    def objective(self, weights: np.ndarray) -> float:
        return float(weights @ self.covariance_matrix @ weights)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        return 2.0 * (self.covariance_matrix @ weights)

    # ------------------------------------------------------------------
    # CVXPY interface
    # ------------------------------------------------------------------

    def build_cvxpy_objective(self, w):
        import cvxpy as cp
        return cp.Minimize(cp.quad_form(w, self.covariance_matrix))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        return float(np.sqrt(self.objective(weights)))
