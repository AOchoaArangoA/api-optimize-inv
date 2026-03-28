"""
Hierarchical Risk Parity (HRP) - Non-convex optimization.
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
from typing import Dict, Any, Optional

from ...core.optimizer import Optimizer
from ...core.formulation import OptimizationProblem


class HierarchicalRiskParity(Optimizer):
    """
    Hierarchical Risk Parity (HRP).
    
    Algoritmo que usa clustering jerárquico para construir portafolios
    diversificados sin necesidad de inversión de matrices.
    """
    
    def __init__(self, linkage_method: str = "single", **kwargs):
        """
        Inicializa HRP.
        
        Args:
            linkage_method: Método de linkage ('single', 'complete', 'average')
            **kwargs: Parámetros adicionales
        """
        super().__init__(linkage_method=linkage_method, **kwargs)
        self.linkage_method = linkage_method
    
    def optimize(
        self,
        problem: OptimizationProblem,
        initial_weights: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calcula pesos HRP.
        
        Args:
            problem: Problema de optimización (debe tener covariance_matrix)
            initial_weights: Ignorado
        
        Returns:
            Diccionario con resultados
        """
        # Obtener matriz de covarianza del problema
        cov_matrix = self._get_covariance_matrix(problem)
        
        # 1. Calcular matriz de correlación
        corr_matrix = self._cov_to_corr(cov_matrix)
        
        # 2. Calcular distancias
        dist_matrix = np.sqrt((1 - corr_matrix) / 2)
        
        # 3. Clustering jerárquico
        dist_condensed = squareform(dist_matrix, checks=False)
        link = linkage(dist_condensed, method=self.linkage_method)
        
        # 4. Quasi-diagonalización
        sort_ix = self._get_quasi_diag(link)
        
        # 5. Recursive bisection para obtener pesos
        weights = self._get_recursive_bisection(cov_matrix, sort_ix)
        
        return {
            "weights": weights,
            "objective_value": None,
            "status": "success",
            "metadata": {
                "linkage_method": self.linkage_method,
                "sort_order": sort_ix.tolist()
            }
        }
    
    def _get_covariance_matrix(self, problem: OptimizationProblem) -> np.ndarray:
        """Extrae la matriz de covarianza del problema."""
        if hasattr(problem, "covariance_matrix"):
            return problem.covariance_matrix
        else:
            raise ValueError("Problem must have covariance_matrix attribute for HRP")
    
    def _cov_to_corr(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Convierte covarianza a correlación."""
        std = np.sqrt(np.diag(cov_matrix))
        corr = cov_matrix / np.outer(std, std)
        corr[corr < -1], corr[corr > 1] = -1, 1  # Numerical stability
        return corr
    
    def _get_quasi_diag(self, link: np.ndarray) -> np.ndarray:
        """Obtiene el orden quasi-diagonal del dendrograma."""
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3]
        
        while sort_ix.max() >= num_items:
            sort_ix.index = list(range(0, sort_ix.shape[0] * 2, 2))
            df0 = sort_ix[sort_ix >= num_items]
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0]
            df0 = pd.Series(link[j, 1], index=i + 1)
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = list(range(sort_ix.shape[0]))
        
        return sort_ix.tolist()
    
    def _get_recursive_bisection(
        self,
        cov_matrix: np.ndarray,
        sort_ix: list
    ) -> np.ndarray:
        """Aplica bisección recursiva para calcular pesos."""
        w = pd.Series(1, index=sort_ix)
        c_items = [sort_ix]
        
        while len(c_items) > 0:
            c_items = [
                i[j:k]
                for i in c_items
                for j, k in ((0, len(i) // 2), (len(i) // 2, len(i)))
                if len(i) > 1
            ]
            
            for i in range(0, len(c_items), 2):
                c_items0 = c_items[i]
                c_items1 = c_items[i + 1]
                
                # Calcular varianza de cada cluster
                cov0 = cov_matrix[np.ix_(c_items0, c_items0)]
                cov1 = cov_matrix[np.ix_(c_items1, c_items1)]
                
                inv_vol0 = 1 / np.sqrt(np.diag(cov0).sum())
                inv_vol1 = 1 / np.sqrt(np.diag(cov1).sum())
                
                # Asignar pesos
                alpha = inv_vol0 / (inv_vol0 + inv_vol1)
                w[c_items0] *= alpha
                w[c_items1] *= (1 - alpha)
        
        return w.values
