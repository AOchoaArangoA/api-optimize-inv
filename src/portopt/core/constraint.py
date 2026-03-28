"""
Constraint — Abstract base class for portfolio constraints.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np


class Constraint(ABC):
    """
    Clase abstracta para restricciones de portafolio.

    Métodos obligatorios:
        ``is_satisfied(weights)`` — verificación booleana.

    Métodos opcionales (con comportamiento por defecto):
        ``project(weights)``      — proyección al espacio factible.
                                    Por defecto devuelve los pesos sin cambios.
        ``to_cvxpy(w)``           — lista de restricciones CVXPY.
                                    Por defecto devuelve lista vacía.
        ``to_scipy(n_assets)``    — lista de dicts para scipy.optimize.minimize.
                                    Por defecto devuelve lista vacía.
    """

    def __init__(self, **kwargs):
        self.params: Dict[str, Any] = kwargs

    # ------------------------------------------------------------------
    # Obligatorio
    # ------------------------------------------------------------------

    @abstractmethod
    def is_satisfied(self, weights: np.ndarray) -> bool:
        """True si los pesos satisfacen esta restricción."""

    # ------------------------------------------------------------------
    # Opcional — comportamiento por defecto
    # ------------------------------------------------------------------

    def project(self, weights: np.ndarray) -> np.ndarray:
        """
        Proyecta los pesos al espacio factible.

        La implementación por defecto es una proyección identidad
        (no hace nada).  Subclases pueden sobreescribir si la proyección
        tiene forma cerrada.
        """
        return weights

    def to_cvxpy(self, weights_var) -> List:
        """
        Convierte la restricción a expresiones CVXPY.

        Returns
        -------
        list
            Lista de restricciones CVXPY.  Vacía por defecto.
        """
        return []

    def to_scipy(self, n_assets: int) -> List[Dict[str, Any]]:
        """
        Convierte la restricción a formato SciPy (eq / ineq).

        Returns
        -------
        list of dict
            Restricciones para ``scipy.optimize.minimize``.  Vacía por defecto.
        """
        return []

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Devuelve los parámetros de la restricción."""
        return self.params

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
