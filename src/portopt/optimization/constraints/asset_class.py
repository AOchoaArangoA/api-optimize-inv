"""
Asset Class Constraint - Restricciones robustas a nivel de clase de activo.
"""

import numpy as np
from typing import Optional, Dict

from ...core.constraint import Constraint
from ...core.asset_class import AssetClass
from ...core.asset_class_registry import AssetClassRegistry


class AssetClassConstraint(Constraint):
    """
    Restricción de límites por clase de activo.
    
    Permite especificar límites mínimos y máximos para la asignación
    total a cada clase de activo (e.g., "Equities entre 30-60%").
    
    Útil para políticas de inversión robustas y diversificación obligatoria.
    """
    
    def __init__(
        self,
        registry: AssetClassRegistry,
        class_limits: Dict[AssetClass, tuple],
        asset_list: list
    ):
        """
        Inicializa la restricción por clase de activo.
        
        Args:
            registry: Registro de clasificación de activos
            class_limits: Diccionario {AssetClass: (min, max)}
                         Ejemplo: {AssetClass.EQUITIES: (0.3, 0.6)}
            asset_list: Lista ordenada de nombres de activos
        """
        super().__init__()
        self.registry = registry
        self.class_limits = class_limits
        self.asset_list = asset_list
        
        # Validar configuración
        for asset_class, (min_weight, max_weight) in class_limits.items():
            if not 0 <= min_weight <= max_weight <= 1:
                raise ValueError(
                    f"Invalid limits for {asset_class}: ({min_weight}, {max_weight})"
                )
    
    def is_satisfied(self, weights: np.ndarray, tol: float = 1e-6) -> bool:
        """
        Verifica si los pesos satisfacen los límites por clase.
        
        Args:
            weights: Pesos del portafolio
            tol: Tolerancia para comparaciones
            
        Returns:
            True si satisface la restricción
        """
        weight_dict = {asset: weights[i] for i, asset in enumerate(self.asset_list)}
        class_weights = self.registry.get_class_weights(weight_dict)
        
        for asset_class, (min_w, max_w) in self.class_limits.items():
            actual_weight = class_weights.get(asset_class, 0.0)
            if actual_weight < min_w - tol or actual_weight > max_w + tol:
                return False
        
        return True
    
    def project(self, weights: np.ndarray) -> np.ndarray:
        """
        Proyecta pesos al espacio factible (implementación simplificada).
        
        Args:
            weights: Pesos actuales
            
        Returns:
            Pesos proyectados
        """
        # Proyección compleja - por ahora retornamos los originales
        # Una implementación real requeriría optimización cuadrática
        return weights
    
    def to_cvxpy(self, weights_var):
        """
        Convierte la restricción a formato CVXPY.
        
        Args:
            weights_var: Variable de CVXPY para los pesos
            
        Returns:
            Lista de restricciones CVXPY
        """
        constraints = []
        
        # Crear matriz de mapeo: asset class → assets
        for asset_class, (min_w, max_w) in self.class_limits.items():
            # Obtener índices de activos en esta clase
            class_assets = self.registry.get_assets_by_class(asset_class)
            asset_indices = [
                i for i, asset in enumerate(self.asset_list)
                if asset in class_assets
            ]
            
            if asset_indices:
                # Vector selector: 1 para activos de esta clase, 0 para otros
                selector = np.zeros(len(self.asset_list))
                selector[asset_indices] = 1
                
                # Restricciones: min_w <= selector' * w <= max_w
                class_weight = selector @ weights_var
                constraints.append(class_weight >= min_w)
                constraints.append(class_weight <= max_w)
        
        return constraints
    
    def to_scipy(self, n_assets: int) -> list:
        """Convierte los límites por clase a restricciones SciPy (ineq)."""
        constraints = []
        for asset_class, (min_w, max_w) in self.class_limits.items():
            class_assets = self.registry.get_assets_by_class(asset_class)
            indices = [
                i for i, a in enumerate(self.asset_list) if a in class_assets
            ]
            if not indices:
                continue
            selector = np.zeros(n_assets)
            selector[indices] = 1.0

            def class_min(w, sel=selector.copy(), lo=min_w):
                return (sel @ w) - lo

            def class_max(w, sel=selector.copy(), hi=max_w):
                return hi - (sel @ w)

            constraints.append({"type": "ineq", "fun": class_min})
            constraints.append({"type": "ineq", "fun": class_max})

        return constraints

    def __repr__(self) -> str:
        """Representación de la restricción."""
        limits_str = ", ".join(
            f"{ac.value}: [{mn:.1%}, {mx:.1%}]"
            for ac, (mn, mx) in self.class_limits.items()
        )
        return f"AssetClassConstraint({limits_str})"
