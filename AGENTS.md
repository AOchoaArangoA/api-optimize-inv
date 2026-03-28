# AGENTS.md — api-optimize-inv / portopt API Layer

Este archivo es la fuente de verdad para Claude Code.
Lee esto **completo** antes de escribir cualquier línea de código.

---

## Propósito de este sprint

Añadir una capa API (FastAPI) sobre el paquete `portopt` ya existente,
y un MCP cliente (`mcp_portfolio.py`) para que el agente `financial-agent`
pueda consumir el optimizador vía HTTP.

**No modificar nada dentro de `src/portopt/`.**

---

## Principios de diseño (no negociables)

1. **Delgadez** — El API layer no tiene lógica de negocio. Solo traduce JSON ↔ objetos portopt.
2. **Config-driven** — Ningún valor hardcodeado. Todo en `configs/api.yaml`.
3. **Contratos explícitos** — Schemas Pydantic completos en `api/schemas/`. Sin `dict` libres.
4. **No-break** — Cada endpoint captura excepciones y retorna `HTTPException` con mensaje legible.
5. **Parsimonia** — Funciones cortas. Sin capas innecesarias. Si cabe en 20 líneas, no crear clase.
6. **Comentarios pedagógicos** — Cada función/clase debe tener docstring explicando el *por qué*, no solo el *qué*.

---

## Estructura a crear (solo estos archivos — no tocar nada más)

```
api-optimize-inv/
├── api/
│   ├── __init__.py
│   ├── app.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── optimize.py
│   │   └── frontier.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   └── responses.py
│   ├── factories/
│   │   ├── __init__.py
│   │   └── builder.py
│   └── dependencies.py
├── mcp_portfolio.py
├── configs/api.yaml
└── tests/
    └── test_api.py
```

---

## Schemas Pydantic — contrato de datos

### `api/schemas/requests.py`

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
import pandas as pd

class ConstraintsConfig(BaseModel):
    """
    Especifica qué restricciones aplicar al problema de optimización.
    BudgetConstraint: suma de pesos = 1.
    LongOnly: todos los pesos >= 0.
    Box: rango [min_weight, max_weight] por activo.
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
        description="Precios históricos: {ticker: [precio_t0, precio_t1, ...]}"
    )
    formulation: Literal["mean_variance", "min_variance", "cvar", "tracking_error"] = "mean_variance"
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
```

### `api/schemas/responses.py`

```python
from pydantic import BaseModel
from typing import Dict, List, Optional

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
```

---

## Factories — `api/factories/builder.py`

Este módulo es el traductor central: convierte los strings del JSON en objetos portopt.

```python
# Pedagogía: usar registries (dicts) en lugar de if/elif encadenados.
# Añadir una nueva formulación = agregar una línea al dict, no modificar lógica.

from portopt.optimization.formulations import (
    MeanVarianceOptimization, MinVarianceOptimization,
    CVaROptimization, TrackingErrorOptimization
)
from portopt.optimization.solvers import CVXPYSolver, ScipySolver
from portopt.returns import HistoricalReturns, CAPMReturns
from portopt.risk import SampleCovariance, ShrinkageCovariance, GARCHModel
from portopt.optimization.constraints import (
    BudgetConstraint, LongOnlyConstraint, BoxConstraint
)

FORMULATION_REGISTRY = {
    "mean_variance":  MeanVarianceOptimization,
    "min_variance":   MinVarianceOptimization,
    "cvar":           CVaROptimization,
    "tracking_error": TrackingErrorOptimization,
}

SOLVER_REGISTRY = {
    "cvxpy": CVXPYSolver,
    "scipy": ScipySolver,
}

RETURNS_REGISTRY = {
    "historical": HistoricalReturns,
    "capm":       CAPMReturns,
}

RISK_REGISTRY = {
    "sample":    SampleCovariance,
    "shrinkage": ShrinkageCovariance,
    "garch":     GARCHModel,
}

def build_constraints(config: ConstraintsConfig) -> list:
    """Construye la lista de restricciones a partir del config del request."""
    constraints = []
    if config.budget:
        constraints.append(BudgetConstraint())
    if config.long_only:
        constraints.append(LongOnlyConstraint())
    if config.box_min is not None and config.box_max is not None:
        constraints.append(BoxConstraint(config.box_min, config.box_max))
    return constraints
```

---

## Routers

### `api/routers/optimize.py`

```python
"""
POST /optimize

Flujo:
  1. Pydantic valida el request (automático)
  2. prices dict → pd.DataFrame
  3. builder.py instancia los objetos portopt
  4. PortfolioOrchestrator.run() resuelve el problema
  5. OptimizationResult → PortfolioResult (Pydantic)
  6. FastAPI serializa a JSON (automático)
"""
from fastapi import APIRouter, HTTPException
from api.schemas.requests import OptimizeRequest
from api.schemas.responses import PortfolioResult
# ... imports builder, orchestrator

router = APIRouter(prefix="/optimize", tags=["optimization"])

@router.post("/", response_model=PortfolioResult)
async def optimize_portfolio(request: OptimizeRequest) -> PortfolioResult:
    try:
        # paso 2: reconstruir DataFrame de precios
        # paso 3: instanciar objetos via factories
        # paso 4: correr orchestrator
        # paso 5: mapear resultado
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### `api/routers/frontier.py`

```python
"""
POST /frontier

Estrategia: barrer risk_aversion en np.linspace(min, max, n_points),
resolver el problema mean_variance N veces, recolectar resultados.

Nota pedagógica: la frontera eficiente es el conjunto de portafolios
que no pueden mejorar retorno sin aumentar riesgo. Se genera paramétricamente
variando el trade-off entre retorno y riesgo (risk_aversion).
"""
```

---

## `api/app.py`

```python
"""
Punto de entrada de FastAPI.

Pedagogía: FastAPI usa un patrón 'application factory' —
la instancia `app` se crea aquí y los routers se montan sobre ella.
Esto permite testear cada router independientemente.
"""
from fastapi import FastAPI
from api.routers import optimize, frontier

app = FastAPI(
    title="Portfolio Optimizer API",
    description="API para optimización de portafolios multi-activo. Basado en portopt.",
    version="1.0.0",
)

app.include_router(optimize.router)
app.include_router(frontier.router)

@app.get("/health")
async def health():
    """Endpoint de health check. El agente lo llama antes de enviar requests."""
    return {"status": "ok", "version": "1.0.0"}
```

---

## `mcp_portfolio.py` — MCP para financial-agent

```python
"""
MCP que expone el optimizador al agente CIO en financial-agent.

Patrón: cliente HTTP asíncrono (httpx) que llama al API local.
Retorna dicts con schema fijo — contrato estándar de los MCPs del agente.

Por qué httpx y no requests:
  - httpx soporta async/await nativamente
  - LangGraph corre en contexto async
  - requests bloquea el event loop
"""
import httpx
from typing import Dict, List

API_BASE = "http://localhost:8000"  # sobreescribible por configs/api.yaml

async def optimize(
    tickers: List[str],
    prices: Dict[str, List[float]],
    formulation: str = "mean_variance",
    risk_aversion: float = 2.0,
    **kwargs
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
        "sharpe_ratio": float,
        "solver_status": str
    }
    """
    payload = {
        "tickers": tickers,
        "prices": prices,
        "formulation": formulation,
        "risk_aversion": risk_aversion,
        **kwargs
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            r = await client.post(f"{API_BASE}/optimize", json=payload)
            r.raise_for_status()
            data = r.json()
            return {"status": "ok", "source": "portfolio_optimizer", **data}
        except Exception as e:
            return {"status": "error", "source": "portfolio_optimizer", "detail": str(e)}


async def frontier(
    tickers: List[str],
    prices: Dict[str, List[float]],
    n_points: int = 20,
    **kwargs
) -> dict:
    """
    Llama a POST /frontier y retorna dict con schema MCP estándar.

    Retorna:
    {
        "status": "ok" | "error",
        "source": "portfolio_optimizer",
        "frontier": [...],
        "min_variance_idx": int,
        "max_sharpe_idx": int
    }
    """
    payload = {"tickers": tickers, "prices": prices, "n_points": n_points, **kwargs}
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.post(f"{API_BASE}/frontier", json=payload)
            r.raise_for_status()
            data = r.json()
            return {"status": "ok", "source": "portfolio_optimizer", **data}
        except Exception as e:
            return {"status": "error", "source": "portfolio_optimizer", "detail": str(e)}
```

---

## `configs/api.yaml`

```yaml
# Configuración del servidor API
server:
  host: "0.0.0.0"
  port: 8000
  reload: true           # solo en desarrollo

# Timeouts del cliente MCP (segundos)
client:
  optimize_timeout: 30
  frontier_timeout: 60

# Parámetros por defecto de la frontera
frontier_defaults:
  n_points: 20
  risk_aversion_min: 0.5
  risk_aversion_max: 10.0
```

---

## Tests — `tests/test_api.py`

```python
"""
Tests del API usando TestClient de FastAPI.

Pedagogía: TestClient crea un servidor en memoria — no necesitas
levantar uvicorn para testear. Cada test es independiente.
"""
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

# Datos de prueba mínimos (3 activos, 252 días)
SAMPLE_PRICES = {
    "AAPL": [150.0 + i * 0.1 for i in range(252)],
    "TLT":  [100.0 - i * 0.05 for i in range(252)],
    "GLD":  [170.0 + i * 0.02 for i in range(252)],
}

def test_health():
    r = client.get("/health")
    assert r.status_code == 200

def test_optimize_mean_variance():
    payload = {
        "tickers": ["AAPL", "TLT", "GLD"],
        "prices": SAMPLE_PRICES,
        "formulation": "mean_variance",
        "risk_aversion": 2.0,
    }
    r = client.post("/optimize/", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "weights" in data
    assert abs(sum(data["weights"].values()) - 1.0) < 1e-6  # pesos suman 1
    assert data["status"] == "optimal"

def test_frontier_returns_n_points():
    payload = {
        "tickers": ["AAPL", "TLT", "GLD"],
        "prices": SAMPLE_PRICES,
        "n_points": 10,
    }
    r = client.post("/frontier/", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert len(data["frontier"]) == 10

def test_optimize_invalid_single_ticker():
    """Pydantic debe rechazar un universo de 1 solo ticker."""
    payload = {"tickers": ["AAPL"], "prices": {"AAPL": [100.0] * 252}}
    r = client.post("/optimize/", json=payload)
    assert r.status_code == 422   # Unprocessable Entity — validación Pydantic
```

---

## Reglas para Claude Code

1. **No modificar** nada dentro de `src/portopt/`. Solo lectura.
2. **No hardcodear** valores. Usar `configs/api.yaml` para todo parámetro configurable.
3. **No agregar** dependencias sin confirmar. Las nuevas son: `fastapi`, `uvicorn`, `httpx`.
4. **Siempre** correr `python -m pytest tests/test_api.py -v` después de implementar un endpoint.
5. **Comentarios pedagógicos** en cada función: docstring con flujo paso a paso y el *por qué* de decisiones de diseño.
6. Si `PortfolioOrchestrator` requiere argumentos que no están en los schemas, **reportar a Claude Chat** en lugar de asumir.
7. El archivo `builder.py` debe fallar con `ValueError` claro si llega un string no registrado (ej. `"formulation": "xyz"`).
8. El endpoint `/frontier` debe paralelizar las N optimizaciones con `asyncio.gather` o `ThreadPoolExecutor`.

---

## Flujo de implementación recomendado (en orden)

```
1. configs/api.yaml                          ← sin dependencias
2. api/schemas/requests.py + responses.py   ← solo Pydantic, testeable solo
3. api/factories/builder.py                 ← traduce JSON → objetos portopt
4. api/routers/optimize.py                  ← primer endpoint funcional
5. api/app.py                               ← montar router, /health
6. tests/test_api.py (test_health + test_optimize)
7. api/routers/frontier.py                  ← segundo endpoint
8. tests/test_api.py (test_frontier)
9. mcp_portfolio.py                         ← cliente del agente
```

---

## Estado del sprint

- [ ] `configs/api.yaml`
- [ ] `api/schemas/requests.py`
- [ ] `api/schemas/responses.py`
- [ ] `api/factories/builder.py`
- [ ] `api/routers/optimize.py`
- [ ] `api/app.py`
- [ ] Tests pasando: `test_health`, `test_optimize_mean_variance`
- [ ] `api/routers/frontier.py`
- [ ] Tests pasando: `test_frontier_returns_n_points`, `test_optimize_invalid_single_ticker`
- [ ] `mcp_portfolio.py`
- [ ] Integración confirmada con `financial-agent`