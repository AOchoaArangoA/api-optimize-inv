"""
OptimizationResult — typed result returned by all solvers.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class OptimizationResult:
    """
    Resultado tipado de un proceso de optimización.

    Attributes
    ----------
    weights : np.ndarray or None
        Pesos óptimos del portafolio.  None si la optimización falló.
    objective_value : float or None
        Valor de la función objetivo en la solución.
    status : str
        ``"success"`` | ``"failed"`` | ``"error"``
    metadata : dict
        Información adicional del solver (iteraciones, tiempo, mensajes).
    """

    weights: Optional[np.ndarray]
    objective_value: Optional[float]
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True si la optimización terminó con solución válida."""
        return self.status == "success"

    def __repr__(self) -> str:
        w_str = (
            f"weights={self.weights.round(4)}"
            if self.weights is not None
            else "weights=None"
        )
        return f"OptimizationResult({w_str}, status={self.status!r})"
