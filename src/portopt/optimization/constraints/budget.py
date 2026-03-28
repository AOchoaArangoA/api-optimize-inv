"""
Budget Constraint - Ensures weights sum to 1.
"""

import numpy as np
import cvxpy as cp

from ...core.constraint import Constraint


class BudgetConstraint(Constraint):
    """
    Restricción presupuestal.
    
    Asegura que los pesos sumen 1 (o algún valor target).
    ∑ w_i = budget_target
    """
    
    def __init__(self, budget_target: float = 1.0, tolerance: float = 1e-6):
        """
        Inicializa la restricción presupuestal.
        
        Args:
            budget_target: Suma objetivo de los pesos (default=1.0)
            tolerance: Tolerancia para considerar la restricción satisfecha
        """
        super().__init__(budget_target=budget_target, tolerance=tolerance)
        self.budget_target = budget_target
        self.tolerance = tolerance
    
    def is_satisfied(self, weights: np.ndarray) -> bool:
        """
        Verifica si los pesos suman al target.
        
        Args:
            weights: Pesos del portafolio
            
        Returns:
            True si satisface la restricción
        """
        return abs(weights.sum() - self.budget_target) <= self.tolerance
    
    def project(self, weights: np.ndarray) -> np.ndarray:
        """
        Proyecta los pesos para satisfacer la restricción.
        
        Args:
            weights: Pesos a proyectar
            
        Returns:
            Pesos proyectados
        """
        # Normalizar para que sumen al target
        current_sum = weights.sum()
        if current_sum == 0:
            return np.ones(len(weights)) * (self.budget_target / len(weights))
        
        return weights * (self.budget_target / current_sum)
    
    def to_cvxpy(self, weights_var) -> list:
        """
        Convierte la restricción a formato CVXPY.

        Args:
            weights_var: Variable CVXPY para los pesos

        Returns:
            Lista de restricciones CVXPY
        """
        return [cp.sum(weights_var) == self.budget_target]

    def to_scipy(self, n_assets: int) -> list:
        target = self.budget_target

        def budget_eq(w, t=target):
            return np.sum(w) - t

        return [{"type": "eq", "fun": budget_eq}]
