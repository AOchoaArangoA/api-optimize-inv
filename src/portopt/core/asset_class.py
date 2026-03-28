"""
Asset Class Enumeration.

Defines the supported asset classes for portfolio optimization.
"""

from enum import Enum


class AssetClass(Enum):
    """
    Clases de activos soportadas en el framework.
    
    Cada asset class puede tener modelos de retornos y riesgo específicos.
    """
    
    EQUITIES = "equities"              # Renta variable (acciones)
    FIXED_INCOME = "fixed_income"      # Renta fija (bonos)
    COMMODITIES = "commodities"        # Materias primas
    DERIVATIVES = "derivatives"        # Derivados (opciones, futuros)
    CASH = "cash"                      # Efectivo y equivalentes
    REAL_ESTATE = "real_estate"        # Bienes raíces (REITs)
    CRYPTO = "crypto"                  # Criptomonedas
    ALTERNATIVES = "alternatives"      # Inversiones alternativas
    
    def __str__(self) -> str:
        """Representación en string."""
        return self.value
    
    def __repr__(self) -> str:
        """Representación para debugging."""
        return f"AssetClass.{self.name}"
    
    @classmethod
    def from_string(cls, value: str) -> 'AssetClass':
        """
        Crea AssetClass desde string.
        
        Args:
            value: String con el nombre de la clase
            
        Returns:
            AssetClass correspondiente
            
        Raises:
            ValueError: Si el valor no es válido
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid_values = [e.value for e in cls]
            raise ValueError(
                f"Invalid asset class: {value}. "
                f"Valid values: {valid_values}"
            )
