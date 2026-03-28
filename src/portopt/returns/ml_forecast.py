"""
MLForecastReturns — Machine-learning based expected returns.

Receives already-computed returns (no internal pct_change).
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from ..core.returns import ExpectedReturnsModel


class MLForecastReturns(ExpectedReturnsModel):
    """
    Retornos esperados estimados con un modelo de Machine Learning.

    Construye features técnicas (retornos rezagados, medias móviles,
    volatilidad) y entrena un regresor por activo.

    Parameters
    ----------
    model_type : str
        Tipo de modelo ML.  Actualmente sólo ``'random_forest'``.
    lookback_window : int
        Número de retornos rezagados usados como features.
    **model_params
        Parámetros adicionales para el estimador sklearn.
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        lookback_window: int = 20,
        **model_params,
    ):
        self.model_type = model_type
        self.lookback_window = lookback_window
        self.model_params = model_params
        self.models_: Dict[str, RandomForestRegressor] = {}
        self.scaler_ = StandardScaler()
        self.returns_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _create_features(self, series: pd.Series) -> np.ndarray:
        features = []
        for lag in range(1, self.lookback_window + 1):
            features.append(series.shift(lag).iloc[-1])
        features.append(series.rolling(5).mean().iloc[-1])
        features.append(series.rolling(20).mean().iloc[-1])
        features.append(series.rolling(20).std().iloc[-1])
        return np.array(features)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def fit(self, returns: pd.DataFrame) -> "MLForecastReturns":
        """
        Entrena un regresor por activo usando los retornos históricos.

        Parameters
        ----------
        returns : pd.DataFrame
            Retornos ya computados. Shape (n_periods, n_assets).
        """
        self.returns_ = returns

        for asset in returns.columns:
            X, y = [], []
            series = returns[asset]
            for i in range(self.lookback_window + 1, len(series)):
                features = self._create_features(series.iloc[:i])
                if not np.isnan(features).any():
                    X.append(features)
                    y.append(series.iloc[i])

            if len(X) == 0:
                continue

            X_arr = self.scaler_.fit_transform(np.array(X))
            model = RandomForestRegressor(**self.model_params)
            model.fit(X_arr, np.array(y))
            self.models_[asset] = model

        return self

    def predict(self) -> np.ndarray:
        """Predice retornos esperados para el siguiente período."""
        if self.returns_ is None:
            raise RuntimeError("Llama a fit() antes de predict().")

        expected = []
        for asset in self.returns_.columns:
            series = self.returns_[asset]
            if asset in self.models_:
                features = self._create_features(series)
                if not np.isnan(features).any():
                    f = self.scaler_.transform(features.reshape(1, -1))
                    expected.append(float(self.models_[asset].predict(f)[0]))
                else:
                    expected.append(float(series.mean()))
            else:
                expected.append(float(series.mean()))

        return np.array(expected)

    def __repr__(self) -> str:
        return (
            f"MLForecastReturns(model_type={self.model_type!r}, "
            f"lookback_window={self.lookback_window})"
        )
