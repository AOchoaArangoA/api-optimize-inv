"""
Router POST /optimize-live

Extiende /optimize añadiendo la descarga de precios desde Alpaca.

Flujo:
  1. Pydantic valida el request (automático).
  2. Leer credenciales Alpaca: request > variables de entorno > error claro.
  3. AlpacaDataDownloader.fetch_stock_data() → DataFrame multi-ticker.
  4. Pivotar DataFrame: {ticker: [close_t0, close_t1, ...]} para alimentar
     el mismo pipeline que /optimize.
  5. Delegar al pipeline estándar (builder + Orchestrator).
  6. Retornar PortfolioResult idéntico al de /optimize.

Por qué reutilizar el mismo pipeline: DRY. La lógica de optimización
no cambia; solo cambia el origen de los datos.
"""

import os
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas.live_requests import OptimizeLiveRequest
from api.schemas.responses import PortfolioResult
from api.factories.builder import (
    build_formulation,
    build_solver,
    build_returns_model,
    build_risk_model,
    build_constraints,
)
from portopt.pipeline.orchestrator import PortfolioOrchestrator
from portopt.returns.data_downloader import AlpacaDataDownloader

router = APIRouter(prefix="/optimize-live", tags=["optimization-live"])

RISK_FREE_RATE = 0.0


def _resolve_credentials(request: OptimizeLiveRequest) -> tuple[str, str]:
    """
    Resuelve las credenciales Alpaca con precedencia:
      1. Campos del request (útil para pruebas desde el agente).
      2. Variables de entorno ALPACA_API_KEY / ALPACA_SECRET_KEY.

    Lanza ValueError con mensaje legible si no se encuentran.
    """
    key = request.alpaca_api_key or os.getenv("ALPACA_API_KEY")
    secret = request.alpaca_secret_key or os.getenv("ALPACA_SECRET_KEY")

    if not key or not secret:
        raise ValueError(
            "Credenciales Alpaca no encontradas. "
            "Proporciónalas en el request (alpaca_api_key / alpaca_secret_key) "
            "o en las variables de entorno ALPACA_API_KEY / ALPACA_SECRET_KEY."
        )
    return key, secret


def _alpaca_df_to_prices(raw_df: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """
    Transforma el DataFrame devuelto por AlpacaDataDownloader al formato
    que espera el Orchestrator: columnas = tickers, filas = fechas, valores = close.

    AlpacaDataDownloader retorna un DataFrame con columna 'ticker' y 'close'
    (multi-ticker concatenado). Se pivota a formato wide.
    """
    # El df tiene columna 'ticker' si fetch_stock_data concatenó múltiples tickers
    if "ticker" in raw_df.columns:
        prices = raw_df.pivot_table(index=raw_df.index, columns="ticker", values="close")
    else:
        # Fallback: asumir que el índice es fecha y las columnas son tickers
        prices = raw_df[["close"]].copy()

    # Mantener solo los tickers solicitados (en el orden del request)
    available = [t for t in tickers if t in prices.columns]
    if len(available) < 2:
        raise ValueError(
            f"Alpaca devolvió datos insuficientes. "
            f"Tickers solicitados: {tickers}. Disponibles: {list(prices.columns)}"
        )

    prices = prices[available].dropna()
    return prices


@router.post("/", response_model=PortfolioResult)
async def optimize_live(request: OptimizeLiveRequest) -> PortfolioResult:
    """
    Descarga precios desde Alpaca y optimiza el portafolio en un solo paso.
    Útil cuando el agente no tiene los precios precargados.
    """
    t0 = time.perf_counter()

    try:
        # --- Paso 2: credenciales ---
        api_key, secret_key = _resolve_credentials(request)

        # --- Paso 3: descargar precios desde Alpaca ---
        downloader = AlpacaDataDownloader(api_key, secret_key)
        raw_df = downloader.fetch_stock_data(request.tickers, years=request.years)

        # --- Paso 4: pivotar a formato wide ---
        prices_df = _alpaca_df_to_prices(raw_df, request.tickers)
        tickers = list(prices_df.columns)
        n_assets = len(tickers)

        # --- Paso 5: pipeline estándar (idéntico a /optimize) ---
        problem = build_formulation(request.formulation, n_assets, request.risk_aversion)
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

        # --- Paso 6: mapear resultado ---
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

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
