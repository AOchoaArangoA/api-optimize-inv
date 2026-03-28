"""
CVXPYSolver — Convex optimization via CVXPY.
"""

from typing import Optional

import numpy as np
import cvxpy as cp

from ...core.optimizer import Optimizer
from ...core.formulation import OptimizationProblem
from ...core.result import OptimizationResult


class CVXPYSolver(Optimizer):
    """
    Solver basado en CVXPY para problemas convexos.

    Delega la construcción del objetivo a ``problem.build_cvxpy_objective(w)``,
    eliminando el acoplamiento con tipos concretos de formulación.

    Parameters
    ----------
    solver : str
        Backend de CVXPY ('ECOS', 'SCS', 'OSQP', 'CLARABEL', …).
    verbose : bool
        Si True, imprime información del solver.
    """

    def __init__(self, solver: str = "CLARABEL", verbose: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.solver = solver
        self.verbose = verbose

    def optimize(
        self,
        problem: OptimizationProblem,
        initial_weights: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """
        Resuelve el problema de optimización con CVXPY.

        Parameters
        ----------
        problem : OptimizationProblem
            Debe implementar ``build_cvxpy_objective(w)``.
        initial_weights : ignored
            CVXPY no requiere punto de partida para problemas convexos.
        """
        w = cp.Variable(problem.n_assets)

        # --- Objetivo (cada formulación lo construye internamente) ---
        objective = problem.build_cvxpy_objective(w)

        # --- Restricciones ---
        constraints: list = []

        # Restricciones auxiliares del objetivo (e.g. variables CVaR)
        aux = getattr(objective, "_cvar_aux_constraints", [])
        constraints.extend(aux)

        # Restricciones del problema (presupuesto, long-only, etc.)
        for c in problem.get_constraints():
            constraints.extend(c.to_cvxpy(w))

        # --- Resolver ---
        cp_problem = cp.Problem(objective, constraints)
        try:
            cp_problem.solve(
                solver=self.solver,
                verbose=self.verbose,
                **self.params,
            )
        except Exception as e:
            return OptimizationResult(
                weights=None,
                objective_value=None,
                status="error",
                metadata={"error": str(e), "solver": self.solver},
            )

        if cp_problem.status not in ("optimal", "optimal_inaccurate"):
            return OptimizationResult(
                weights=None,
                objective_value=None,
                status="failed",
                metadata={"cvxpy_status": cp_problem.status, "solver": self.solver},
            )

        solve_time = getattr(
            cp_problem.solver_stats, "solve_time", None
        )
        return OptimizationResult(
            weights=np.array(w.value),
            objective_value=float(cp_problem.value),
            status="success",
            metadata={
                "cvxpy_status": cp_problem.status,
                "solver": self.solver,
                "solve_time": solve_time,
            },
        )

    def __repr__(self) -> str:
        return f"CVXPYSolver(solver={self.solver!r})"
