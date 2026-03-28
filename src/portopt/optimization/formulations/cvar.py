"""
CVaROptimization — Conditional Value at Risk portfolio optimization.
"""

from typing import List, Optional

import numpy as np

from ...core.formulation import OptimizationProblem
from ...core.constraint import Constraint


class CVaROptimization(OptimizationProblem):
    """
    Optimización basada en CVaR (Conditional Value at Risk).

    El CVaR (también llamado Expected Shortfall) es la pérdida esperada
    en el α% de peores escenarios.  Es una medida de riesgo coherente
    y más robusta que la varianza para capturar riesgo de cola.

    Formulación lineal (Rockafellar & Uryasev, 2000):

        min  ζ + (1 / (α·T)) · Σ_t u_t
        s.t. u_t ≥ -r_t'w - ζ,   ∀t
             u_t ≥ 0

    Parameters
    ----------
    n_assets : int
    returns_scenarios : np.ndarray  shape (T, n_assets)
        Matriz de escenarios de retornos históricos o simulados.
    alpha : float
        Nivel de pérdida (0.05 = CVaR al 95%).
    constraints : list of Constraint, optional
    """

    def __init__(
        self,
        n_assets: int,
        returns_scenarios: Optional[np.ndarray] = None,
        alpha: float = 0.05,
        constraints: Optional[List[Constraint]] = None,
    ):
        super().__init__(n_assets, constraints)
        self.returns_scenarios = returns_scenarios
        self.alpha = alpha
        self.n_scenarios: int = (
            returns_scenarios.shape[0] if returns_scenarios is not None else 0
        )

    # ------------------------------------------------------------------
    # SciPy interface (evaluación directa)
    # ------------------------------------------------------------------

    def objective(self, weights: np.ndarray) -> float:
        """CVaR negativo (pérdida, para minimización)."""
        port_returns = self.returns_scenarios @ weights
        var = np.percentile(port_returns, self.alpha * 100)
        tail = port_returns[port_returns <= var]
        return float(-tail.mean()) if len(tail) > 0 else 0.0

    # gradient heredado usa NotImplementedError → SciPy calculará numéricamente

    # ------------------------------------------------------------------
    # CVXPY interface (formulación lineal exacta)
    # ------------------------------------------------------------------

    def build_cvxpy_objective(self, w):
        """
        Formulación lineal del CVaR usando variables auxiliares.
        Permite que CVXPY resuelva el problema exactamente como un QP/LP.
        """
        import cvxpy as cp

        T = self.n_scenarios
        zeta = cp.Variable()                 # VaR escalar
        u = cp.Variable(T, nonneg=True)      # excesos sobre VaR

        portfolio_returns = self.returns_scenarios @ w
        # u_t >= -r_t'w - ζ  ↔  r_t'w + ζ + u_t >= 0
        aux_constraints = [u >= -portfolio_returns - zeta]

        cvar_expr = zeta + (1.0 / (self.alpha * T)) * cp.sum(u)
        objective = cp.Minimize(cvar_expr)

        # Adjuntamos las restricciones auxiliares al objetivo como atributo
        # para que CVXPYSolver las pueda combinar con las del problema.
        objective._cvar_aux_constraints = aux_constraints

        return objective

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def var(self, weights: np.ndarray) -> float:
        """Value at Risk al nivel alpha."""
        port_returns = self.returns_scenarios @ weights
        return float(-np.percentile(port_returns, self.alpha * 100))

    def cvar(self, weights: np.ndarray) -> float:
        """Conditional Value at Risk (Expected Shortfall)."""
        return float(self.objective(weights))
