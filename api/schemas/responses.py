"""
Schemas de respuesta del API.

Todas las métricas monetarias/retornos están anualizadas.
El campo 'status' usa la terminología de solvers convexos estándar.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel


class PortfolioResult(BaseModel):
    """
    Resultado de un único portafolio optimizado.
    Todas las métricas son anualizadas.
    """

    weights: Dict[str, float]          # {ticker: peso}, suma = 1.0
    expected_return: float             # retorno esperado anual
    expected_volatility: float         # desviación estándar anual
    sharpe_ratio: Optional[float]      # None si volatilidad ≈ 0
    status: str                        # "optimal" | "infeasible" | "unbounded" | "error"
    solver_used: str
    computation_ms: float


class FrontierResult(BaseModel):
    """
    Resultado de la frontera eficiente completa.
    'frontier' está ordenado de menor a mayor volatilidad.
    """

    frontier: List[PortfolioResult]
    min_variance_idx: int              # índice en frontier del portafolio de mínima varianza
    max_sharpe_idx: int                # índice del portafolio con máximo Sharpe
    computation_ms: float
