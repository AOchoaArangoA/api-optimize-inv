"""
TrackingErrorOptimization — Minimize tracking error against a benchmark.
"""

from typing import List, Optional

import numpy as np

from ...core.formulation import OptimizationProblem
from ...core.constraint import Constraint


class TrackingErrorOptimization(OptimizationProblem):
    """
    Minimización de Tracking Error respecto a un benchmark.

    Objetivo:   min  (w - w_b)'Σ(w - w_b)

    Útil para portafolios indexados o enhanced-index.

    Parameters
    ----------
    n_assets : int
    covariance_matrix : np.ndarray  shape (n_assets, n_assets)
    benchmark_weights : np.ndarray  shape (n_assets,)
    constraints : list of Constraint, optional
    """

    def __init__(
        self,
        n_assets: int,
        covariance_matrix: Optional[np.ndarray] = None,
        benchmark_weights: Optional[np.ndarray] = None,
        constraints: Optional[List[Constraint]] = None,
    ):
        super().__init__(n_assets, constraints)
        self.covariance_matrix = covariance_matrix
        self.benchmark_weights = benchmark_weights

    # ------------------------------------------------------------------
    # SciPy interface
    # ------------------------------------------------------------------

    def objective(self, weights: np.ndarray) -> float:
        active = weights - self.benchmark_weights
        return float(active @ self.covariance_matrix @ active)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        active = weights - self.benchmark_weights
        return 2.0 * (self.covariance_matrix @ active)

    # ------------------------------------------------------------------
    # CVXPY interface
    # ------------------------------------------------------------------

    def build_cvxpy_objective(self, w):
        import cvxpy as cp
        active = w - self.benchmark_weights
        return cp.Minimize(cp.quad_form(active, self.covariance_matrix))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def tracking_error(self, weights: np.ndarray) -> float:
        """Tracking error anualizado (raíz cuadrada del objetivo)."""
        return float(np.sqrt(self.objective(weights)))
