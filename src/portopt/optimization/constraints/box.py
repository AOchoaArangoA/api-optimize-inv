"""
Box Constraint - Individual asset weight bounds.
"""

import numpy as np
import cvxpy as cp
from typing import Optional

from ...core.constraint import Constraint


class BoxConstraint(Constraint):
    """
    Restricción de caja (box constraints).
    
    Impone límites individuales para cada activo.
    lb_i <= w_i <= ub_i para todo i
    """
    
    def __init__(
        self,
        lower_bounds: Optional[np.ndarray] = None,
        upper_bounds: Optional[np.ndarray] = None
    ):
        """
        Inicializa la restricción de caja.
        
        Args:
            lower_bounds: Límites inferiores por activo (None = 0)
            upper_bounds: Límites superiores por activo (None = 1)
        """
        super().__init__(lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
    
    def is_satisfied(self, weights: np.ndarray) -> bool:
        """
        Verifica si los pesos satisfacen los límites.
        
        Args:
            weights: Pesos del portafolio
            
        Returns:
            True si satisface la restricción
        """
        if self.lower_bounds is not None:
            if np.any(weights < self.lower_bounds - 1e-9):
                return False
        
        if self.upper_bounds is not None:
            if np.any(weights > self.upper_bounds + 1e-9):
                return False
        
        return True
    
    def project(self, weights: np.ndarray) -> np.ndarray:
        """
        Proyecta los pesos al espacio factible.
        
        Args:
            weights: Pesos a proyectar
            
        Returns:
            Pesos proyectados
        """
        projected = weights.copy()
        
        if self.lower_bounds is not None:
            projected = np.maximum(projected, self.lower_bounds)
        
        if self.upper_bounds is not None:
            projected = np.minimum(projected, self.upper_bounds)
        
        # Renormalizar
        if projected.sum() > 0:
            projected = projected / projected.sum()
        
        return projected
    
    def to_cvxpy(self, weights_var) -> list:
        """
        Convierte la restricción a formato CVXPY.
        
        Args:
            weights_var: Variable CVXPY para los pesos
            
        Returns:
            Lista de restricciones CVXPY
        """
        constraints = []
        
        if self.lower_bounds is not None:
            constraints.append(weights_var >= self.lower_bounds)
        
        if self.upper_bounds is not None:
            constraints.append(weights_var <= self.upper_bounds)
        
        return constraints
