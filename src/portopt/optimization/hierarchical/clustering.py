"""
Clustering-based Optimization.
"""

import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Any, Optional

from ...core.optimizer import Optimizer
from ...core.formulation import OptimizationProblem


class ClusteringOptimizer(Optimizer):
    """
    Optimizador basado en clustering.
    
    Agrupa activos en clusters y optimiza dentro de cada cluster.
    """
    
    def __init__(
        self,
        n_clusters: int = 5,
        cluster_method: str = "kmeans",
        **kwargs
    ):
        """
        Inicializa el optimizador de clustering.
        
        Args:
            n_clusters: Número de clusters
            cluster_method: Método de clustering ('kmeans')
            **kwargs: Parámetros adicionales
        """
        super().__init__(n_clusters=n_clusters, cluster_method=cluster_method, **kwargs)
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method
    
    def optimize(
        self,
        problem: OptimizationProblem,
        initial_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Optimiza usando clustering.
        
        Args:
            problem: Problema de optimización
            initial_weights: Ignorado
        
        Returns:
            Diccionario con resultados
        """
        # Obtener datos de retornos o correlación
        cov_matrix = self._get_covariance_matrix(problem)
        corr_matrix = self._cov_to_corr(cov_matrix)
        
        # Clustering
        if self.cluster_method == "kmeans":
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            labels = kmeans.fit_predict(corr_matrix)
        else:
            raise ValueError(f"Unknown cluster_method: {self.cluster_method}")
        
        # Asignar pesos equal-weighted dentro de cada cluster
        # Y equal-weighted entre clusters
        weights = np.zeros(problem.n_assets)
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = labels == cluster_id
            n_in_cluster = cluster_mask.sum()
            
            if n_in_cluster > 0:
                weights[cluster_mask] = 1 / (self.n_clusters * n_in_cluster)
        
        # Aplicar restricciones
        for constraint in problem.get_constraints():
            if not constraint.is_satisfied(weights):
                weights = constraint.project(weights)
        
        return {
            "weights": weights,
            "objective_value": problem.objective(weights),
            "status": "success",
            "metadata": {
                "cluster_labels": labels.tolist(),
                "n_clusters": self.n_clusters
            }
        }
    
    def _get_covariance_matrix(self, problem: OptimizationProblem) -> np.ndarray:
        """Extrae la matriz de covarianza del problema."""
        if hasattr(problem, "covariance_matrix"):
            return problem.covariance_matrix
        else:
            raise ValueError("Problem must have covariance_matrix attribute")
    
    def _cov_to_corr(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Convierte covarianza a correlación."""
        std = np.sqrt(np.diag(cov_matrix))
        corr = cov_matrix / np.outer(std, std)
        corr[corr < -1], corr[corr > 1] = -1, 1
        return corr
