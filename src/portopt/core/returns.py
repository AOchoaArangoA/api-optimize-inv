"""
ExpectedReturnsModel — Abstract base class for return estimation models.

Data contract
-------------
All models receive a DataFrame of **already-computed returns** (e.g. log-returns
or simple returns).  The conversion from raw prices to returns is the caller's
responsibility and should happen once, before calling ``fit()``.

Typical usage::

    returns = prices.pct_change().dropna()   # done once, outside the model
    mu = HistoricalReturns().fit(returns).predict()
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ExpectedReturnsModel(ABC):
    """
    Clase abstracta para estimación de retornos esperados.

    Convención:
        ``fit(returns)``  — ajusta el modelo recibiendo un DataFrame de retornos.
        ``predict()``     — devuelve el vector de retornos esperados (np.ndarray).
    """

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> "ExpectedReturnsModel":
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
            Permite encadenamiento (``model.fit(r).predict()``).
        """

    @abstractmethod
    def predict(self) -> np.ndarray:
        """
        Devuelve el vector de retornos esperados estimado tras ``fit()``.

        Returns
        -------
        np.ndarray
            Shape (n_assets,).
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
