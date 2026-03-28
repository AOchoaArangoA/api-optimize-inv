"""
RiskModel — Abstract base class for risk (covariance) estimation models.

Data contract
-------------
All models receive a DataFrame of **already-computed returns**.
The conversion from raw prices to returns must happen before calling ``fit()``.

Typical usage::

    returns = prices.pct_change().dropna()   # done once, outside the model
    cov = SampleCovariance().fit(returns).covariance()
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class RiskModel(ABC):
    """
    Clase abstracta para estimación de riesgo (matriz de covarianza).

    Convención:
        ``fit(returns)``   — ajusta el modelo recibiendo un DataFrame de retornos.
        ``covariance()``   — devuelve la matriz de covarianza estimada.
        ``volatility()``   — conveniencia: raíz de la diagonal de la covarianza.
    """

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> "RiskModel":
        """
        Ajusta el modelo a los retornos históricos.

        Parameters
        ----------
        returns : pd.DataFrame
            Retornos históricos ya computados.
            Shape: (n_periods, n_assets).  Columnas = tickers.

        Returns
        -------
        self
            Permite encadenamiento (``model.fit(r).covariance()``).
        """

    @abstractmethod
    def covariance(self) -> np.ndarray:
        """
        Devuelve la matriz de covarianza estimada tras ``fit()``.

        Returns
        -------
        np.ndarray
            Shape (n_assets, n_assets), simétrica semidefinida positiva.
        """

    def volatility(self) -> np.ndarray:
        """
        Devuelve la volatilidad (desviación estándar) de cada activo.

        Returns
        -------
        np.ndarray
            Shape (n_assets,).
        """
        return np.sqrt(np.diag(self.covariance()))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
