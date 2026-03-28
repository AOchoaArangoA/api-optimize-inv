"""
Sector Constraint - Sector exposure limits.
"""

import numpy as np
import cvxpy as cp
from typing import Dict, Optional

from ...core.constraint import Constraint


class SectorConstraint(Constraint):
    """
    Restricción de sector.
    
    Impone límites a la exposición total por sector.
    lb_s <= ∑_{i∈s} w_i <= ub_s para cada sector s
    """
    
    def __init__(
        self,
        sector_mapping: Dict[int, str],
        sector_lower: Optional[Dict[str, float]] = None,
        sector_upper: Optional[Dict[str, float]] = None
    ):
        """
        Inicializa la restricción de sector.
        
        Args:
            sector_mapping: Mapeo {asset_index: sector_name}
            sector_lower: Límites inferiores por sector {sector: lower_bound}
            sector_upper: Límites superiores por sector {sector: upper_bound}
        """
        super().__init__(
            sector_mapping=sector_mapping,
            sector_lower=sector_lower,
            sector_upper=sector_upper
        )
        self.sector_mapping = sector_mapping
        self.sector_lower = sector_lower or {}
        self.sector_upper = sector_upper or {}
        
        # Crear matriz de pertenencia sector
        self.sectors = list(set(sector_mapping.values()))
        self.n_assets = max(sector_mapping.keys()) + 1
        self.sector_matrix = self._build_sector_matrix()
    
    def _build_sector_matrix(self) -> np.ndarray:
        """Construye matriz de pertenencia a sectores."""
        matrix = np.zeros((len(self.sectors), self.n_assets))
        
        for asset_idx, sector in self.sector_mapping.items():
            sector_idx = self.sectors.index(sector)
            matrix[sector_idx, asset_idx] = 1
        
        return matrix
    
    def is_satisfied(self, weights: np.ndarray) -> bool:
        """
        Verifica si los pesos satisfacen los límites sectoriales.
        
        Args:
            weights: Pesos del portafolio
            
        Returns:
            True si satisface la restricción
        """
        sector_weights = self.sector_matrix @ weights
        
        for i, sector in enumerate(self.sectors):
            if sector in self.sector_lower:
                if sector_weights[i] < self.sector_lower[sector] - 1e-6:
                    return False
            
            if sector in self.sector_upper:
                if sector_weights[i] > self.sector_upper[sector] + 1e-6:
                    return False
        
        return True
    
    def project(self, weights: np.ndarray) -> np.ndarray:
        """
        Proyecta los pesos (simplificado - puede no ser óptimo).
        
        Args:
            weights: Pesos a proyectar
            
        Returns:
            Pesos proyectados
        """
        # Implementación simplificada - escalar pesos por sector
        projected = weights.copy()
        
        for i, sector in enumerate(self.sectors):
            sector_mask = self.sector_matrix[i, :] > 0
            sector_weight = projected[sector_mask].sum()
            
            # Aplicar límites
            if sector in self.sector_upper and sector_weight > self.sector_upper[sector]:
                scale = self.sector_upper[sector] / sector_weight
                projected[sector_mask] *= scale
            
            if sector in self.sector_lower and sector_weight < self.sector_lower[sector]:
                # Aumentar proporcionalmente
                deficit = self.sector_lower[sector] - sector_weight
                projected[sector_mask] += deficit / sector_mask.sum()
        
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
        
        for i, sector in enumerate(self.sectors):
            sector_weight = self.sector_matrix[i, :] @ weights_var
            
            if sector in self.sector_lower:
                constraints.append(sector_weight >= self.sector_lower[sector])
            
            if sector in self.sector_upper:
                constraints.append(sector_weight <= self.sector_upper[sector])
        
        return constraints
