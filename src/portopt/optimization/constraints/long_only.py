"""
Long Only Constraint - Ensures all weights are non-negative.
"""

import numpy as np
import cvxpy as cp

from ...core.constraint import Constraint


class LongOnlyConstraint(Constraint):
    """
    Restricción long-only.
    
    Asegura que todos los pesos sean no negativos (no short selling).
    w_i >= 0 para todo i
    """
    
    def __init__(self):
        """Inicializa la restricción long-only."""
        super().__init__()
    
    def is_satisfied(self, weights: np.ndarray) -> bool:
        """
        Verifica si todos los pesos son no negativos.
        
        Args:
            weights: Pesos del portafolio
            
        Returns:
            True si satisface la restricción
        """
        return np.all(weights >= -1e-9)  # Pequeña tolerancia numérica
    
    def project(self, weights: np.ndarray) -> np.ndarray:
        """
        Proyecta los pesos al espacio no negativo.
        
        Args:
            weights: Pesos a proyectar
            
        Returns:
            Pesos proyectados (negativos -> 0)
        """
        projected = np.maximum(weights, 0)
        
        # Renormalizar si es necesario
        if projected.sum() > 0:
            projected = projected / projected.sum()
        else:
            # Si todos eran negativos, equal-weighted
            projected = np.ones(len(weights)) / len(weights)
        
        return projected
    
    def to_cvxpy(self, weights_var) -> list:
        """
        Convierte la restricción a formato CVXPY.
        
        Args:
            weights_var: Variable CVXPY para los pesos
            
        Returns:
            Lista de restricciones CVXPY
        """
        return [weights_var >= 0]
