"""
Heuristic Solver - Uses heuristic algorithms.
"""

import numpy as np
from typing import Dict, Any, Optional

from ...core.optimizer import Optimizer
from ...core.formulation import OptimizationProblem


class HeuristicSolver(Optimizer):
    """
    Solver heurístico.
    
    Usa algoritmos heurísticos como equal-weighted, risk-parity, etc.
    """
    
    def __init__(self, strategy: str = "equal_weighted", **kwargs):
        """
        Inicializa el solver heurístico.
        
        Args:
            strategy: Estrategia heurística ('equal_weighted', 'risk_parity')
            **kwargs: Parámetros adicionales
        """
        super().__init__(strategy=strategy, **kwargs)
        self.strategy = strategy
    
    def optimize(
        self,
        problem: OptimizationProblem,
        initial_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Aplica estrategia heurística.
        
        Args:
            problem: Problema de optimización
            initial_weights: Ignorado
        
        Returns:
            Diccionario con resultados
        """
        if self.strategy == "equal_weighted":
            weights = self._equal_weighted(problem.n_assets)
        
        elif self.strategy == "risk_parity":
            weights = self._risk_parity(problem)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Aplicar restricciones si es necesario
        for constraint in problem.get_constraints():
            if not constraint.is_satisfied(weights):
                weights = constraint.project(weights)
        
        return {
            "weights": weights,
            "objective_value": problem.objective(weights),
            "status": "success",
            "metadata": {"strategy": self.strategy}
        }
    
    def _equal_weighted(self, n_assets: int) -> np.ndarray:
        """Retorna pesos equal-weighted."""
        return np.ones(n_assets) / n_assets
    
    def _risk_parity(self, problem: OptimizationProblem) -> np.ndarray:
        """
        Calcula pesos risk-parity (contribución igual al riesgo).
        
        Implementación simplificada usando inversión de volatilidades.
        """
        from ...optimization.formulations.min_variance import MinVarianceOptimization
        
        if isinstance(problem, MinVarianceOptimization):
            # Usar inversión de volatilidades
            volatilities = np.sqrt(np.diag(problem.covariance_matrix))
            inv_vol = 1 / volatilities
            weights = inv_vol / inv_vol.sum()
            return weights
        
        # Fallback: equal-weighted
        return self._equal_weighted(problem.n_assets)
