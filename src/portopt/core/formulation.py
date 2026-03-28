"""
OptimizationProblem — Abstract base class for portfolio optimization formulations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from .constraint import Constraint


class OptimizationProblem(ABC):
    """
    Clase abstracta para formulaciones de optimización de portafolio.

    Subclases implementan:
        ``objective(weights)``          — función objetivo escalar (para SciPy).
        ``build_cvxpy_objective(w)``    — expresión CVXPY (para CVXPYSolver).

    El método ``gradient()`` es opcional: por defecto lanza ``NotImplementedError``,
    lo que indica que el solver no debe usarlo a menos que la subclase lo implemente.
    """

    def __init__(
        self,
        n_assets: int,
        constraints: Optional[List[Constraint]] = None,
    ):
        self.n_assets = n_assets
        self.constraints: List[Constraint] = constraints or []

    # ------------------------------------------------------------------
    # Interfaz SciPy (evaluación numérica)
    # ------------------------------------------------------------------

    @abstractmethod
    def objective(self, weights: np.ndarray) -> float:
        """Valor escalar de la función objetivo (para ScipySolver)."""

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """
        Gradiente analítico de la función objetivo.

        Implementar en la subclase si se desea pasar ``jac`` a SciPy.
        Por defecto, SciPy usa diferencias finitas si se devuelve None
        o si este método no se sobreescribe.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} no implementa gradient(). "
            "SciPy calculará el gradiente numéricamente."
        )

    # ------------------------------------------------------------------
    # Interfaz CVXPY (expresión simbólica)
    # ------------------------------------------------------------------

    @abstractmethod
    def build_cvxpy_objective(self, w) -> object:
        """
        Construye y devuelve la expresión CVXPY de la función objetivo.

        Parameters
        ----------
        w : cp.Variable
            Variable de pesos del portafolio (shape n_assets,).

        Returns
        -------
        cp.Expression
            Expresión CVXPY que representa el objetivo (minimización).
        """

    # ------------------------------------------------------------------
    # Restricciones
    # ------------------------------------------------------------------

    def add_constraint(self, constraint: Constraint) -> None:
        """Agrega una restricción al problema."""
        self.constraints.append(constraint)

    def get_constraints(self) -> List[Constraint]:
        """Devuelve la lista de restricciones registradas."""
        return self.constraints

    def is_feasible(self, weights: np.ndarray) -> bool:
        """True si los pesos satisfacen todas las restricciones."""
        return all(c.is_satisfied(weights) for c in self.constraints)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_assets={self.n_assets}, "
            f"n_constraints={len(self.constraints)})"
        )
