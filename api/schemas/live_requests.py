"""
Schema del request para POST /optimize-live.

Diferencia con OptimizeRequest: en lugar de recibir precios ya calculados,
recibe tickers + credenciales Alpaca y el API descarga los precios internamente.

Por qué separar en un archivo aparte y no extender OptimizeRequest:
  - Principio de responsabilidad única: las credenciales son específicas
    del endpoint live y no deben contaminar el schema base.
  - Permite versionar ambos contratos independientemente.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from api.schemas.requests import ConstraintsConfig


class OptimizeLiveRequest(BaseModel):
    """
    Request para POST /optimize-live.
    El API descarga precios históricos desde Alpaca antes de optimizar.
    """

    tickers: List[str] = Field(..., min_length=2, description="Lista de tickers del universo")
    years: int = Field(1, ge=1, le=10, description="Años de historia a descargar")

    # Credenciales Alpaca — se pueden pasar por request o leer de variables de entorno
    alpaca_api_key: Optional[str] = Field(
        None, description="Alpaca API Key. Si es None usa la variable ALPACA_API_KEY"
    )
    alpaca_secret_key: Optional[str] = Field(
        None, description="Alpaca Secret Key. Si es None usa ALPACA_SECRET_KEY"
    )

    formulation: Literal["mean_variance", "min_variance", "cvar", "tracking_error"] = (
        "mean_variance"
    )
    risk_aversion: float = Field(2.0, gt=0.0)
    solver: Literal["cvxpy", "scipy"] = "cvxpy"
    returns_model: Literal["historical", "capm"] = "historical"
    risk_model: Literal["sample", "shrinkage", "garch"] = "sample"
    constraints: ConstraintsConfig = ConstraintsConfig()


class FrontierLiveRequest(BaseModel):
    """
    Request para POST /frontier-live.
    Descarga precios desde Alpaca y calcula la frontera eficiente completa.
    Combina la descarga automática de /optimize-live con el barrido de /frontier.
    """

    tickers: List[str] = Field(..., min_length=2, description="Lista de tickers del universo")
    years: int = Field(1, ge=1, le=10, description="Años de historia a descargar")

    alpaca_api_key: Optional[str] = Field(
        None, description="Alpaca API Key. Si es None usa ALPACA_API_KEY del entorno"
    )
    alpaca_secret_key: Optional[str] = Field(
        None, description="Alpaca Secret Key. Si es None usa ALPACA_SECRET_KEY del entorno"
    )

    n_points: int = Field(20, ge=5, le=100, description="Número de puntos en la frontera")
    risk_aversion_min: float = Field(0.5, gt=0.0)
    risk_aversion_max: float = Field(10.0, gt=0.0)
    solver: Literal["cvxpy", "scipy"] = "cvxpy"
    returns_model: Literal["historical", "capm"] = "historical"
    risk_model: Literal["sample", "shrinkage", "garch"] = "sample"
    constraints: ConstraintsConfig = ConstraintsConfig()
