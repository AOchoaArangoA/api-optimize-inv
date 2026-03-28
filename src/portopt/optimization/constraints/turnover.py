"""
Turnover Constraint - Limits portfolio turnover.
"""

import numpy as np
import cvxpy as cp
from typing import Optional

from ...core.constraint import Constraint


class TurnoverConstraint(Constraint):
    """
    Restricción de turnover.
    
    Limita el turnover del portafolio (cambio desde posiciones actuales).
    ∑ |w_i - w_i^current| <= max_turnover
    """
    
    def __init__(
        self,
        current_weights: np.ndarray,
        max_turnover: float = 0.2
    ):
        """
        Inicializa la restricción de turnover.
        
        Args:
            current_weights: Pesos actuales del portafolio
            max_turnover: Máximo turnover permitido (e.g., 0.2 = 20%)
        """
        super().__init__(current_weights=current_weights, max_turnover=max_turnover)
        self.current_weights = current_weights
        self.max_turnover = max_turnover
    
    def is_satisfied(self, weights: np.ndarray) -> bool:
        """
        Verifica si el turnover está dentro del límite.
        
        Args:
            weights: Pesos del portafolio
            
        Returns:
            True si satisface la restricción
        """
        turnover = np.abs(weights - self.current_weights).sum()
        return turnover <= self.max_turnover + 1e-6
    
    def project(self, weights: np.ndarray) -> np.ndarray:
        """
        Proyecta los pesos para satisfacer el turnover máximo.
        
        Args:
            weights: Pesos a proyectar
            
        Returns:
            Pesos proyectados
        """
        # Implementación simplificada - interpolar hacia pesos actuales
        turnover = np.abs(weights - self.current_weights).sum()
        
        if turnover <= self.max_turnover:
            return weights
        
        # Escalar el cambio
        scale = self.max_turnover / turnover
        adjustment = weights - self.current_weights
        
        return self.current_weights + scale * adjustment
    
    def to_cvxpy(self, weights_var) -> list:
        """
        Convierte la restricción a formato CVXPY.
        
        Args:
            weights_var: Variable CVXPY para los pesos
            
        Returns:
            Lista de restricciones CVXPY
        """
        # Usar norma L1 para el turnover
        turnover = cp.norm(weights_var - self.current_weights, 1)
        return [turnover <= self.max_turnover]
