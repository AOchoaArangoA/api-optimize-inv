"""
Router POST /frontier

Estrategia: barrer risk_aversion en np.linspace(min, max, n_points),
resolver mean_variance N veces, recolectar resultados.

Paralelización: las N optimizaciones son independientes entre sí
(mismos datos, distinto λ), por lo que se ejecutan con ThreadPoolExecutor.
Por qué ThreadPoolExecutor y no asyncio.gather puro: cvxpy y scipy son
CPU-bound y liberan el GIL parcialmente. ThreadPoolExecutor permite
aprovechar núcleos disponibles sin bloquear el event loop de FastAPI.

Nota pedagógica: la frontera eficiente es el conjunto de portafolios
que maximizan el retorno esperado para cada nivel de riesgo. Se genera
paramétricamente variando λ (risk_aversion): λ alto → menor riesgo,
λ bajo → mayor retorno.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas.requests import FrontierRequest
from api.schemas.responses import FrontierResult, PortfolioResult
from api.factories.builder import (
    build_solver,
    build_returns_model,
    build_risk_model,
    build_constraints,
)
from portopt.optimization.formulations import MeanVarianceOptimization
from portopt.pipeline.orchestrator import PortfolioOrchestrator

router = APIRouter(prefix="/frontier", tags=["frontier"])

RISK_FREE_RATE = 0.0


def _optimize_single(
    prices_df: pd.DataFrame,
    tickers: list,
    risk_aversion: float,
    request: FrontierRequest,
) -> PortfolioResult:
    """
    Resuelve un único punto de la frontera para el lambda dado.
    Función pura: no comparte estado con otras llamadas — segura para threading.
    """
    t0 = time.perf_counter()
    n_assets = len(tickers)

    problem = MeanVarianceOptimization(n_assets=n_assets, risk_aversion=risk_aversion)
    solver = build_solver(request.solver)
    returns_model = build_returns_model(request.returns_model)
    risk_model = build_risk_model(request.risk_model)

    for constraint in build_constraints(request.constraints, n_assets):
        problem.add_constraint(constraint)

    orch = PortfolioOrchestrator()
    result = orch.run(
        data=prices_df,
        returns_model=returns_model,
        risk_model=risk_model,
        problem=problem,
        solver=solver,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    status_map = {"success": "optimal", "failed": "infeasible", "error": "error"}
    api_status = status_map.get(result.status, result.status)

    if not result.success or result.weights is None:
        return PortfolioResult(
            weights={t: 0.0 for t in tickers},
            expected_return=0.0,
            expected_volatility=0.0,
            sharpe_ratio=None,
            status=api_status,
            solver_used=request.solver,
            computation_ms=elapsed_ms,
        )

    weights_dict = dict(zip(tickers, result.weights.tolist()))
    mu = orch.expected_returns_
    cov = orch.cov_matrix_
    port_return = float(mu @ result.weights)
    port_vol = float(np.sqrt(result.weights @ cov @ result.weights))
    sharpe = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 1e-8 else None

    return PortfolioResult(
        weights=weights_dict,
        expected_return=port_return,
        expected_volatility=port_vol,
        sharpe_ratio=sharpe,
        status=api_status,
        solver_used=request.solver,
        computation_ms=elapsed_ms,
    )


@router.post("/", response_model=FrontierResult)
async def compute_frontier(request: FrontierRequest) -> FrontierResult:
    """
    Calcula la frontera eficiente barriendo N valores de risk_aversion.

    Retorna los portafolios ordenados de menor a mayor volatilidad,
    con índices del portafolio de mínima varianza y máximo Sharpe.
    """
    t0 = time.perf_counter()

    try:
        prices_df = pd.DataFrame(request.prices, columns=request.tickers)
        lambdas = np.linspace(
            request.risk_aversion_min, request.risk_aversion_max, request.n_points
        )

        # Paralelizar las N optimizaciones con ThreadPoolExecutor
        # n_points suele ser <= 100, workers=4 es un balance razonable
        results: list[PortfolioResult] = [None] * request.n_points
        with ThreadPoolExecutor(max_workers=min(4, request.n_points)) as executor:
            future_to_idx = {
                executor.submit(
                    _optimize_single,
                    prices_df,
                    request.tickers,
                    float(lam),
                    request,
                ): idx
                for idx, lam in enumerate(lambdas)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        # Ordenar por volatilidad ascendente
        results.sort(key=lambda p: p.expected_volatility)

        # Índice mínima varianza: menor volatilidad → índice 0 tras ordenar
        min_var_idx = 0

        # Índice máximo Sharpe: ignorar puntos fallidos (sharpe_ratio None)
        sharpes = [
            (i, p.sharpe_ratio)
            for i, p in enumerate(results)
            if p.sharpe_ratio is not None
        ]
        max_sharpe_idx = max(sharpes, key=lambda x: x[1])[0] if sharpes else 0

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FrontierResult(
            frontier=results,
            min_variance_idx=min_var_idx,
            max_sharpe_idx=max_sharpe_idx,
            computation_ms=elapsed_ms,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
