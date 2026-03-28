"""
ScipySolver — Gradient-based optimization via SciPy.
"""

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from ...core.optimizer import Optimizer
from ...core.formulation import OptimizationProblem
from ...core.result import OptimizationResult


class ScipySolver(Optimizer):
    """
    Solver basado en ``scipy.optimize.minimize``.

    Usa el método SLSQP por defecto, que admite restricciones de igualdad
    e inequaldad y bounds.  El gradiente se toma de ``problem.gradient()``
    si está implementado; de lo contrario SciPy usa diferencias finitas.

    Parameters
    ----------
    method : str
        Método numérico ('SLSQP', 'L-BFGS-B', 'trust-constr').
    max_iter : int
        Máximo de iteraciones.
    """

    def __init__(self, method: str = "SLSQP", max_iter: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.max_iter = max_iter

    def optimize(
        self,
        problem: OptimizationProblem,
        initial_weights: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """
        Resuelve el problema con SciPy.

        Las restricciones se recogen llamando a ``constraint.to_scipy(n_assets)``
        en cada restricción registrada, eliminando los isinstance en el solver.
        """
        x0 = (
            np.ones(problem.n_assets) / problem.n_assets
            if initial_weights is None
            else np.clip(initial_weights, 0, 1)
        )
        if x0.sum() > 0:
            x0 = x0 / x0.sum()

        # Recolectar restricciones SciPy desde cada objeto Constraint
        scipy_constraints = []
        for c in problem.get_constraints():
            scipy_constraints.extend(c.to_scipy(problem.n_assets))

        bounds = [(0, 1)] * problem.n_assets

        # Gradiente analítico si está disponible
        try:
            problem.gradient(x0)
            jac = problem.gradient
        except NotImplementedError:
            jac = None

        try:
            result = minimize(
                fun=problem.objective,
                x0=x0,
                method=self.method,
                jac=jac,
                bounds=bounds,
                constraints=scipy_constraints,
                options={"maxiter": self.max_iter, **self.params},
            )
        except Exception as e:
            return OptimizationResult(
                weights=None,
                objective_value=None,
                status="error",
                metadata={"error": str(e), "solver": "scipy", "method": self.method},
            )

        w = np.clip(result.x, 0, 1)
        if w.sum() > 0:
            w /= w.sum()

        return OptimizationResult(
            weights=w,
            objective_value=float(result.fun),
            status="success" if result.success else "failed",
            metadata={
                "n_iterations": result.nit,
                "message": result.message,
                "solver": "scipy",
                "method": self.method,
            },
        )

    def __repr__(self) -> str:
        return f"ScipySolver(method={self.method!r})"
