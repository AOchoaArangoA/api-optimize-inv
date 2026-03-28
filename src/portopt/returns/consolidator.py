"""
ReturnsConsolidator — Aggregates expected returns across multiple asset classes.

Data contract: receives already-computed returns (no internal pct_change).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.asset_class import AssetClass
from ..core.asset_class_registry import AssetClassRegistry
from ..core.returns import ExpectedReturnsModel


class ReturnsConsolidator:
    """
    Consolida retornos esperados de modelos específicos por clase de activo.

    Permite asignar un ``ExpectedReturnsModel`` distinto a cada ``AssetClass``
    y produce un único vector de retornos esperados para todo el portafolio.

    Parameters
    ----------
    registry : AssetClassRegistry, optional
        Registro de clasificación de activos.  Si es None, se crea uno vacío.

    Example
    -------
    >>> consolidator = ReturnsConsolidator(registry)
    >>> consolidator.add_model(AssetClass.EQUITIES, HistoricalReturns())
    >>> consolidator.add_model(AssetClass.FIXED_INCOME, CAPMReturns())
    >>> returns = prices.pct_change().dropna()
    >>> mu = consolidator.consolidate(returns)
    """

    def __init__(self, registry: Optional[AssetClassRegistry] = None):
        self.registry = registry or AssetClassRegistry()
        self._models: Dict[AssetClass, ExpectedReturnsModel] = {}
        self._last_returns: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_model(self, asset_class: AssetClass, model: ExpectedReturnsModel) -> None:
        """Registra un modelo de retornos para una clase de activo."""
        if not isinstance(asset_class, AssetClass):
            raise TypeError(f"asset_class debe ser AssetClass, recibido {type(asset_class)}")
        if not isinstance(model, ExpectedReturnsModel):
            raise TypeError(f"model debe ser ExpectedReturnsModel, recibido {type(model)}")
        self._models[asset_class] = model

    # Alias para compatibilidad con código anterior
    def add_asset_class_model(
        self, asset_class: AssetClass, model: ExpectedReturnsModel
    ) -> None:
        self.add_model(asset_class, model)

    def has_model(self, asset_class: AssetClass) -> bool:
        return asset_class in self._models

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def consolidate(self, returns: pd.DataFrame) -> np.ndarray:
        """
        Estima retornos esperados para todos los activos del portafolio.

        Parameters
        ----------
        returns : pd.DataFrame
            Retornos ya computados. Columnas = tickers, filas = fechas.

        Returns
        -------
        np.ndarray
            Vector de retornos esperados ordenado igual que ``returns.columns``.
        """
        assets: List[str] = list(returns.columns)
        self._validate(assets)

        expected = np.zeros(len(assets))
        self._last_returns = {}
        classification = self.registry.classify_portfolio(assets)

        for asset_class, class_assets in classification.items():
            model = self._models[asset_class]
            class_returns = returns[class_assets]
            class_mu = model.fit(class_returns).predict()

            for i, asset in enumerate(class_assets):
                idx = assets.index(asset)
                expected[idx] = class_mu[i]
                self._last_returns[asset] = float(class_mu[i])

        return expected

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_all_returns(self) -> Optional[Dict[str, float]]:
        """Retorna el último vector de retornos estimados por ticker."""
        return self._last_returns

    def get_summary(self) -> dict:
        return {
            "n_asset_classes": len(self._models),
            "asset_classes": [ac.value for ac in self._models],
            "n_assets_registered": len(self.registry),
            "has_estimates": self._last_returns is not None,
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
            raise ValueError(f"Sin modelo para las clases: {missing}")

    def __repr__(self) -> str:
        return (
            f"ReturnsConsolidator("
            f"asset_classes={len(self._models)}, "
            f"assets={len(self.registry)})"
        )
