"""
Router POST /frontier-live

Combina descarga de precios desde Alpaca con el cálculo de frontera eficiente.

Flujo:
  1. Resolver credenciales Alpaca (request > entorno).
  2. AlpacaDataDownloader.fetch_stock_data() → DataFrame multi-ticker.
  3. Pivotar DataFrame → formato wide (tickers × fechas).
  4. Barrer risk_aversion en linspace(min, max, n_points) con ThreadPoolExecutor.
     Cada punto reutiliza el mismo prices_df ya descargado — una sola llamada a Alpaca.
  5. Ordenar por volatilidad y calcular índices min_variance / max_sharpe.
  6. Retornar FrontierResult.

Diferencia clave vs /frontier: el cliente no envía prices, solo tickers + años.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas.live_requests import FrontierLiveRequest
from api.schemas.responses import FrontierResult, PortfolioResult
from api.factories.builder import (
    build_solver,
    build_returns_model,
    build_risk_model,
    build_constraints,
)
from api.routers.optimize_live import _resolve_credentials, _alpaca_df_to_prices
from portopt.optimization.formulations import MeanVarianceOptimization
from portopt.pipeline.orchestrator import PortfolioOrchestrator
from portopt.returns.data_downloader import AlpacaDataDownloader

router = APIRouter(prefix="/frontier-live", tags=["frontier-live"])

RISK_FREE_RATE = 0.0


def _optimize_point(
    prices_df: pd.DataFrame,
    tickers: list,
    risk_aversion: float,
    request: FrontierLiveRequest,
) -> PortfolioResult:
    """
    Resuelve un único punto de la frontera para el lambda dado.
    prices_df ya está descargado — no hace llamadas a Alpaca.
    Función pura: segura para threading.
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
async def compute_frontier_live(request: FrontierLiveRequest) -> FrontierResult:
    """
    Descarga precios desde Alpaca y calcula la frontera eficiente completa.
    La descarga ocurre una sola vez; los N puntos de la frontera se computan en paralelo.
    """
    t0 = time.perf_counter()

    try:
        # --- Paso 1: credenciales ---
        api_key, secret_key = _resolve_credentials(request)

        # --- Paso 2: descargar precios (una sola vez para toda la frontera) ---
        downloader = AlpacaDataDownloader(api_key, secret_key)
        raw_df = downloader.fetch_stock_data(request.tickers, years=request.years)

        # --- Paso 3: pivotar a formato wide ---
        prices_df = _alpaca_df_to_prices(raw_df, request.tickers)
        tickers = list(prices_df.columns)

        # --- Paso 4: barrer lambdas en paralelo ---
        lambdas = np.linspace(
            request.risk_aversion_min, request.risk_aversion_max, request.n_points
        )

        results: list[PortfolioResult] = [None] * request.n_points
        with ThreadPoolExecutor(max_workers=min(4, request.n_points)) as executor:
            future_to_idx = {
                executor.submit(
                    _optimize_point,
                    prices_df,
                    tickers,
                    float(lam),
                    request,
                ): idx
                for idx, lam in enumerate(lambdas)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        # --- Paso 5: ordenar y calcular índices ---
        results.sort(key=lambda p: p.expected_volatility)

        sharpes = [
            (i, p.sharpe_ratio)
            for i, p in enumerate(results)
            if p.sharpe_ratio is not None
        ]
        max_sharpe_idx = max(sharpes, key=lambda x: x[1])[0] if sharpes else 0

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return FrontierResult(
            frontier=results,
            min_variance_idx=0,
            max_sharpe_idx=max_sharpe_idx,
            computation_ms=elapsed_ms,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
