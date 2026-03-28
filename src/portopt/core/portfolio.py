"""
Portfolio class - Represents a portfolio with assets and weights.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class Portfolio:
    """
    Representa un portafolio de inversión con activos y pesos.
    
    Attributes:
        assets: Lista de identificadores de activos
        weights: Pesos de cada activo (deben sumar 1.0)
        data: DataFrame opcional con datos históricos de precios/retornos
    """
    
    def __init__(
        self,
        assets: List[str],
        weights: Optional[np.ndarray] = None,
        data: Optional[pd.DataFrame] = None
    ):
        """
        Inicializa un portafolio.
        
        Args:
            assets: Lista de identificadores de activos
            weights: Array de pesos (opcional, default: equal-weighted)
            data: DataFrame con datos de precios/retornos (opcional)
        """
        self.assets = assets
        self.n_assets = len(assets)
        
        if weights is None:
            # Equal-weighted por default
            self.weights = np.ones(self.n_assets) / self.n_assets
        else:
            self._validate_weights(weights)
            self.weights = weights
            
        self.data = data
    
    def _validate_weights(self, weights: np.ndarray) -> None:
        """Valida que los pesos sean correctos."""
        if len(weights) != self.n_assets:
            raise ValueError(
                f"Number of weights ({len(weights)}) must match "
                f"number of assets ({self.n_assets})"
            )
        
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {weights.sum()}")
    
    def update_weights(self, new_weights: np.ndarray) -> None:
        """Actualiza los pesos del portafolio."""
        self._validate_weights(new_weights)
        self.weights = new_weights
    
    def get_asset_weights(self) -> Dict[str, float]:
        """Retorna un diccionario asset -> weight."""
        return dict(zip(self.assets, self.weights))
    
    def get_returns(self) -> Optional[pd.DataFrame]:
        """Retorna los retornos si data está disponible."""
        if self.data is None:
            return None
        return self.data.pct_change().dropna()
    
    def __repr__(self) -> str:
        return f"Portfolio(n_assets={self.n_assets}, weights={self.weights})"
