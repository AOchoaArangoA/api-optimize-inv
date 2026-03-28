"""
Schemas de request para el API de optimización de portafolios.

Por qué Pydantic: valida y documenta el contrato de datos automáticamente.
FastAPI serializa errores de validación como HTTP 422 sin código extra.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ConstraintsConfig(BaseModel):
    """
    Especifica qué restricciones aplicar al problema de optimización.
    BudgetConstraint: suma de pesos = 1.
    LongOnly: todos los pesos >= 0.
    Box: rango [box_min, box_max] por activo (escalar, aplica igual a todos).
    """

    long_only: bool = True
    budget: bool = True
    box_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    box_max: Optional[float] = Field(None, ge=0.0, le=1.0)


class OptimizeRequest(BaseModel):
    """
    Request para POST /optimize.
    Produce un único portafolio óptimo dado un objetivo y restricciones.
    """

    tickers: List[str] = Field(..., min_length=2, description="Lista de tickers del universo")
    prices: Dict[str, List[float]] = Field(
        ...,
        description="Precios históricos: {ticker: [precio_t0, precio_t1, ...]}",
    )
    formulation: Literal["mean_variance", "min_variance", "cvar", "tracking_error"] = (
        "mean_variance"
    )
    risk_aversion: float = Field(2.0, gt=0.0, description="Solo aplica para mean_variance")
    solver: Literal["cvxpy", "scipy"] = "cvxpy"
    returns_model: Literal["historical", "capm"] = "historical"
    risk_model: Literal["sample", "shrinkage", "garch"] = "sample"
    constraints: ConstraintsConfig = ConstraintsConfig()


class FrontierRequest(BaseModel):
    """
    Request para POST /frontier.
    Produce N portafolios barriendo el parámetro risk_aversion
    para trazar la frontera eficiente completa.
    """

    tickers: List[str] = Field(..., min_length=2)
    prices: Dict[str, List[float]]
    n_points: int = Field(20, ge=5, le=100, description="Número de puntos en la frontera")
    risk_aversion_min: float = Field(0.5, gt=0.0)
    risk_aversion_max: float = Field(10.0, gt=0.0)
    solver: Literal["cvxpy", "scipy"] = "cvxpy"
    returns_model: Literal["historical", "capm"] = "historical"
    risk_model: Literal["sample", "shrinkage", "garch"] = "sample"
    constraints: ConstraintsConfig = ConstraintsConfig()
