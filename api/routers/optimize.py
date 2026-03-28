"""
Router POST /optimize

Flujo:
  1. Pydantic valida el request automáticamente (FastAPI lo hace antes de entrar aquí).
  2. prices dict → pd.DataFrame con tickers como columnas.
  3. builder.py instancia formulación, solver, modelos de retorno/riesgo y restricciones.
  4. PortfolioOrchestrator.run() ejecuta el pipeline completo:
       precios → retornos → μ → Σ → problema → solver → OptimizationResult
  5. OptimizationResult + matrices → PortfolioResult (Pydantic).
  6. FastAPI serializa a JSON automáticamente.
"""

import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas.requests import OptimizeRequest
from api.schemas.responses import PortfolioResult
from api.factories.builder import (
    build_formulation,
    build_solver,
    build_returns_model,
    build_risk_model,
    build_constraints,
)

# Importar Orchestrator después de asegurar el path (resuelto en dependencies.py / builder.py)
from portopt.pipeline.orchestrator import PortfolioOrchestrator

router = APIRouter(prefix="/optimize", tags=["optimization"])

# Tasa libre de riesgo anual para calcular Sharpe (configurable en futuras versiones)
RISK_FREE_RATE = 0.0


@router.post("/", response_model=PortfolioResult)
async def optimize_portfolio(request: OptimizeRequest) -> PortfolioResult:
    """
    Optimiza un portafolio dado un universo de activos y sus precios históricos.

    Por qué async: FastAPI es ASGI. Aunque el cómputo es CPU-bound (scipy/cvxpy),
    declararlo async evita bloquear el event loop en llamadas cortas y permite
    que el servidor siga respondiendo health checks mientras optimiza.
    """
    t0 = time.perf_counter()

    try:
        # --- Paso 2: reconstruir DataFrame de precios ---
        prices_df = pd.DataFrame(request.prices, columns=request.tickers)
        n_assets = len(request.tickers)

        # --- Paso 3: instanciar objetos via factories ---
        problem = build_formulation(request.formulation, n_assets, request.risk_aversion)
        solver = build_solver(request.solver)
        returns_model = build_returns_model(request.returns_model)
        risk_model = build_risk_model(request.risk_model)

        for constraint in build_constraints(request.constraints, n_assets):
            problem.add_constraint(constraint)

        # --- Paso 4: ejecutar Orchestrator ---
        orch = PortfolioOrchestrator()
        result = orch.run(
            data=prices_df,
            returns_model=returns_model,
            risk_model=risk_model,
            problem=problem,
            solver=solver,
        )

        # --- Paso 5: mapear resultado → PortfolioResult ---
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Normalizar status al vocabulario del schema
        status_map = {"success": "optimal", "failed": "infeasible", "error": "error"}
        api_status = status_map.get(result.status, result.status)

        if not result.success or result.weights is None:
            return PortfolioResult(
                weights={t: 0.0 for t in request.tickers},
                expected_return=0.0,
                expected_volatility=0.0,
                sharpe_ratio=None,
                status=api_status,
                solver_used=request.solver,
                computation_ms=elapsed_ms,
            )

        weights_dict = dict(zip(request.tickers, result.weights.tolist()))

        # Métricas: usar matrices ya calculadas por el Orchestrator
        mu = orch.expected_returns_
        cov = orch.cov_matrix_
        port_return = float(mu @ result.weights)
        port_vol = float(np.sqrt(result.weights @ cov @ result.weights))
        sharpe = (
            (port_return - RISK_FREE_RATE) / port_vol if port_vol > 1e-8 else None
        )

        return PortfolioResult(
            weights=weights_dict,
            expected_return=port_return,
            expected_volatility=port_vol,
            sharpe_ratio=sharpe,
            status=api_status,
            solver_used=request.solver,
            computation_ms=elapsed_ms,
        )

    except ValueError as e:
        # Errores de configuración (formulación no registrada, etc.)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
