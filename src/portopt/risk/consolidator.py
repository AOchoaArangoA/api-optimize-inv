"""
RiskConsolidator — Aggregates covariance estimates across multiple asset classes.

Data contract: receives already-computed returns (no internal pct_change).

Strategy
--------
1. Fit each per-class RiskModel to get class-specific volatilities.
2. Compute a global correlation matrix from all returns.
3. Combine:  Σ = D · ρ · D   (D = diag of per-class volatilities)
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.asset_class import AssetClass
from ..core.asset_class_registry import AssetClassRegistry
from ..core.risk import RiskModel


class RiskConsolidator:
    """
    Consolida matrices de covarianza de modelos específicos por clase de activo.

    Permite asignar un ``RiskModel`` distinto a cada ``AssetClass`` y producir
    una única matriz de covarianza para todo el portafolio.

    Parameters
    ----------
    registry : AssetClassRegistry, optional

    Example
    -------
    >>> consolidator = RiskConsolidator(registry)
    >>> consolidator.add_model(AssetClass.EQUITIES, GARCHModel())
    >>> consolidator.add_model(AssetClass.FIXED_INCOME, SampleCovariance())
    >>> returns = prices.pct_change().dropna()
    >>> cov = consolidator.consolidate(returns)
    """

    def __init__(self, registry: Optional[AssetClassRegistry] = None):
        self.registry = registry or AssetClassRegistry()
        self._models: Dict[AssetClass, RiskModel] = {}
        self._covariance_matrix: Optional[np.ndarray] = None
        self._correlation_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_model(self, asset_class: AssetClass, model: RiskModel) -> None:
        """Registra un modelo de riesgo para una clase de activo."""
        if not isinstance(asset_class, AssetClass):
            raise TypeError(f"asset_class debe ser AssetClass, recibido {type(asset_class)}")
        if not isinstance(model, RiskModel):
            raise TypeError(f"model debe ser RiskModel, recibido {type(model)}")
        self._models[asset_class] = model

    # Alias para compatibilidad con código anterior
    def add_asset_class_model(self, asset_class: AssetClass, model: RiskModel) -> None:
        self.add_model(asset_class, model)

    def has_model(self, asset_class: AssetClass) -> bool:
        return asset_class in self._models

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estima la matriz de covarianza consolidada.

        Parameters
        ----------
        returns : pd.DataFrame
            Retornos ya computados. Columnas = tickers, filas = fechas.

        Returns
        -------
        np.ndarray
            Matriz de covarianza (n_assets × n_assets).
        """
        assets: List[str] = list(returns.columns)
        n = len(assets)
        self._validate(assets)

        volatilities = np.zeros(n)
        classification = self.registry.classify_portfolio(assets)

        # 1. Volatilidades específicas por clase
        for asset_class, class_assets in classification.items():
            model = self._models[asset_class]
            class_returns = returns[class_assets]
            model.fit(class_returns)
            vols = model.volatility()          # shape (len(class_assets),)

            for i, asset in enumerate(class_assets):
                volatilities[assets.index(asset)] = vols[i]

        # 2. Correlación global
        self._correlation_matrix = returns.corr().values

        # 3. Σ = D · ρ · D
        D = np.diag(volatilities)
        self._covariance_matrix = D @ self._correlation_matrix @ D

        return self._covariance_matrix

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def covariance_matrix(self) -> Optional[np.ndarray]:
        """Matriz de covarianza del último llamado a consolidate()."""
        return self._covariance_matrix

    def correlation_matrix(self) -> Optional[np.ndarray]:
        """Matriz de correlación del último llamado a consolidate()."""
        return self._correlation_matrix

    def get_covariance_matrix(self) -> Optional[np.ndarray]:
        return self._covariance_matrix

    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        return self._correlation_matrix

    def get_summary(self) -> dict:
        return {
            "n_asset_classes": len(self._models),
            "asset_classes": [ac.value for ac in self._models],
            "n_assets_registered": len(self.registry),
            "has_estimates": self._covariance_matrix is not None,
        }

    # ------------------------------------------------------------------
    # Validation (private)
    # ------------------------------------------------------------------

    def _validate(self, assets: List[str]) -> None:
        unclassified = [a for a in assets if not self.registry.is_registered(a)]
        if unclassified:
            raise ValueError(f"Activos sin clasificar: {unclassified}")

        classification = self.registry.classify_portfolio(assets)
        missing = [ac for ac in classification if ac not in self._models]
        if missing:
            raise ValueError(f"Sin modelo de riesgo para las clases: {missing}")

    def __repr__(self) -> str:
        return (
            f"RiskConsolidator("
            f"asset_classes={len(self._models)}, "
            f"assets={len(self.registry)})"
        )
