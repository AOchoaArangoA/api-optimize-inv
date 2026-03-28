"""
MCP que expone el optimizador al agente CIO en financial-agent.

Patrón: cliente HTTP asíncrono (httpx) que llama al API local.
Retorna dicts con schema fijo — contrato estándar de los MCPs del agente.

Por qué httpx y no requests:
  - httpx soporta async/await nativamente.
  - LangGraph corre en contexto async; requests bloquearía el event loop.
  - httpx.AsyncClient reutiliza conexiones TCP (connection pooling).

Configuración: los timeouts y la URL base se leen de configs/api.yaml
si está disponible; de lo contrario se usan los defaults del módulo.
"""

import os
from typing import Dict, List, Optional

import httpx
import yaml

# --- Cargar configuración ---
_config_path = os.path.join(os.path.dirname(__file__), "configs", "api.yaml")

def _load_config() -> dict:
    """Lee configs/api.yaml si existe; retorna defaults en caso contrario."""
    if os.path.exists(_config_path):
        with open(_config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

_cfg = _load_config()
_server = _cfg.get("server", {})
_client_cfg = _cfg.get("client", {})

API_HOST = _server.get("host", "localhost")
# "0.0.0.0" como host de servidor no funciona como URL de cliente → usar localhost
API_BASE = f"http://localhost:{_server.get('port', 8000)}"
OPTIMIZE_TIMEOUT = _client_cfg.get("optimize_timeout", 30)
FRONTIER_TIMEOUT = _client_cfg.get("frontier_timeout", 60)


async def optimize(
    tickers: List[str],
    prices: Dict[str, List[float]],
    formulation: str = "mean_variance",
    risk_aversion: float = 2.0,
    solver: str = "cvxpy",
    returns_model: str = "historical",
    risk_model: str = "sample",
    constraints: Optional[dict] = None,
) -> dict:
    """
    Llama a POST /optimize y retorna dict con schema MCP estándar.

    Retorna:
    {
        "status": "ok" | "error",
        "source": "portfolio_optimizer",
        "weights": {ticker: peso},
        "expected_return": float,
        "expected_volatility": float,
        "sharpe_ratio": float | None,
        "solver_status": str,
        "computation_ms": float
    }
    """
    payload = {
        "tickers": tickers,
        "prices": prices,
        "formulation": formulation,
        "risk_aversion": risk_aversion,
        "solver": solver,
        "returns_model": returns_model,
        "risk_model": risk_model,
    }
    if constraints is not None:
        payload["constraints"] = constraints

    async with httpx.AsyncClient(timeout=OPTIMIZE_TIMEOUT) as client:
        try:
            r = await client.post(f"{API_BASE}/optimize/", json=payload)
            r.raise_for_status()
            data = r.json()
            return {
                "status": "ok",
                "source": "portfolio_optimizer",
                **data,
                # Alias para claridad en el agente
                "solver_status": data.get("status"),
            }
        except httpx.HTTPStatusError as e:
            return {
                "status": "error",
                "source": "portfolio_optimizer",
                "detail": e.response.text,
                "http_status": e.response.status_code,
            }
        except Exception as e:
            return {"status": "error", "source": "portfolio_optimizer", "detail": str(e)}


async def frontier(
    tickers: List[str],
    prices: Dict[str, List[float]],
    n_points: int = 20,
    risk_aversion_min: float = 0.5,
    risk_aversion_max: float = 10.0,
    solver: str = "cvxpy",
    returns_model: str = "historical",
    risk_model: str = "sample",
    constraints: Optional[dict] = None,
) -> dict:
    """
    Llama a POST /frontier y retorna dict con schema MCP estándar.

    Retorna:
    {
        "status": "ok" | "error",
        "source": "portfolio_optimizer",
        "frontier": [...],
        "min_variance_idx": int,
        "max_sharpe_idx": int,
        "computation_ms": float
    }
    """
    payload = {
        "tickers": tickers,
        "prices": prices,
        "n_points": n_points,
        "risk_aversion_min": risk_aversion_min,
        "risk_aversion_max": risk_aversion_max,
        "solver": solver,
        "returns_model": returns_model,
        "risk_model": risk_model,
    }
    if constraints is not None:
        payload["constraints"] = constraints

    async with httpx.AsyncClient(timeout=FRONTIER_TIMEOUT) as client:
        try:
            r = await client.post(f"{API_BASE}/frontier/", json=payload)
            r.raise_for_status()
            data = r.json()
            return {"status": "ok", "source": "portfolio_optimizer", **data}
        except httpx.HTTPStatusError as e:
            return {
                "status": "error",
                "source": "portfolio_optimizer",
                "detail": e.response.text,
                "http_status": e.response.status_code,
            }
        except Exception as e:
            return {"status": "error", "source": "portfolio_optimizer", "detail": str(e)}
