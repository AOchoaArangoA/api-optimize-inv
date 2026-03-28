"""
Optimizer — Abstract base class for portfolio optimization algorithms.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from .formulation import OptimizationProblem
from .result import OptimizationResult


class Optimizer(ABC):
    """
    Clase abstracta para algoritmos de optimización.

    Todos los solvers devuelven un ``OptimizationResult`` tipado en lugar
    de un diccionario genérico, lo que permite acceso directo a sus campos
    y la propiedad ``result.success``.
    """

    def __init__(self, **kwargs):
        self.params = kwargs

    @abstractmethod
    def optimize(
        self,
        problem: OptimizationProblem,
        initial_weights: Optional[np.ndarray] = None,
    ) -> OptimizationResult:
        """
        Resuelve el problema de optimización.

        Parameters
        ----------
        problem : OptimizationProblem
            Formulación del problema (objetivo + restricciones).
        initial_weights : np.ndarray, optional
            Punto de partida para solvers iterativos.

        Returns
        -------
        OptimizationResult
            Pesos óptimos, valor objetivo, estado y metadatos.
        """

    def set_params(self, **kwargs) -> None:
        """Actualiza los parámetros del optimizador."""
        self.params.update(kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
