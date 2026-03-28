"""
Asset Class Registry - Manages asset classification.
"""

from typing import Dict, List, Optional, Set
from .asset_class import AssetClass


class AssetClassRegistry:
    """
    Registro de clasificación de activos por tipo.
    
    Permite asignar activos a clases específicas y obtener
    información sobre la composición del portafolio.
    
    Attributes:
        asset_mapping: Diccionario que mapea ticker → asset class
    """
    
    def __init__(self):
        """Inicializa el registro vacío."""
        self.asset_mapping: Dict[str, AssetClass] = {}
    
    def register_asset(self, ticker: str, asset_class: AssetClass) -> None:
        """
        Registra un activo con su clase.
        
        Args:
            ticker: Símbolo del activo
            asset_class: Clase de activo
        """
        if not isinstance(asset_class, AssetClass):
            raise TypeError(f"asset_class must be AssetClass, got {type(asset_class)}")
        
        self.asset_mapping[ticker] = asset_class
    
    def register_batch(self, mappings: Dict[str, AssetClass]) -> None:
        """
        Registra múltiples activos.
        
        Args:
            mappings: Diccionario {ticker: asset_class}
        """
        for ticker, asset_class in mappings.items():
            self.register_asset(ticker, asset_class)
    
    def get_asset_class(self, ticker: str) -> Optional[AssetClass]:
        """
        Obtiene la clase de un activo.
        
        Args:
            ticker: Símbolo del activo
            
        Returns:
            AssetClass o None si no está registrado
        """
        return self.asset_mapping.get(ticker)
    
    def get_assets_by_class(self, asset_class: AssetClass) -> List[str]:
        """
        Obtiene todos los activos de una clase específica.
        
        Args:
            asset_class: Clase de activo
            
        Returns:
            Lista de tickers
        """
        return [
            ticker for ticker, cls in self.asset_mapping.items()
            if cls == asset_class
        ]
    
    def get_all_classes(self) -> Set[AssetClass]:
        """
        Obtiene todas las asset classes registradas.
        
        Returns:
            Set de asset classes
        """
        return set(self.asset_mapping.values())
    
    def get_class_weights(self, weights_dict: Dict[str, float]) -> Dict[AssetClass, float]:
        """
        Calcula los pesos por asset class.
        
        Args:
            weights_dict: Diccionario {ticker: peso}
            
        Returns:
            Diccionario {asset_class: peso_total}
        """
        class_weights = {}
        
        for ticker, weight in weights_dict.items():
            asset_class = self.get_asset_class(ticker)
            if asset_class:
                class_weights[asset_class] = class_weights.get(asset_class, 0) + weight
        
        return class_weights
    
    def is_registered(self, ticker: str) -> bool:
        """
        Verifica si un activo está registrado.
        
        Args:
            ticker: Símbolo del activo
            
        Returns:
            True si está registrado
        """
        return ticker in self.asset_mapping
    
    def classify_portfolio(self, tickers: List[str]) -> Dict[AssetClass, List[str]]:
        """
        Clasifica una lista de activos por clase.
        
        Args:
            tickers: Lista de símbolos
            
        Returns:
            Diccionario {asset_class: [tickers]}
        """
        classification = {}
        
        for ticker in tickers:
            asset_class = self.get_asset_class(ticker)
            if asset_class:
                if asset_class not in classification:
                    classification[asset_class] = []
                classification[asset_class].append(ticker)
        
        return classification
    
    def get_summary(self) -> Dict[str, int]:
        """
        Obtiene un resumen de la clasificación.
        
        Returns:
            Diccionario {asset_class_name: count}
        """
        summary = {}
        for asset_class in self.get_all_classes():
            assets = self.get_assets_by_class(asset_class)
            summary[asset_class.value] = len(assets)
        
        return summary
    
    def __len__(self) -> int:
        """Retorna el número de activos registrados."""
        return len(self.asset_mapping)
    
    def __repr__(self) -> str:
        """Representación del registro."""
        n_assets = len(self.asset_mapping)
        n_classes = len(self.get_all_classes())
        return f"AssetClassRegistry(assets={n_assets}, classes={n_classes})"
