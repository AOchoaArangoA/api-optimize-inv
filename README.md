# api-optimize-inv — Portfolio Optimizer API

API REST sobre el paquete `portopt` para optimización de portafolios multi-activo.
Construida con **FastAPI**. Consumible desde cualquier cliente HTTP o desde el MCP `mcp_portfolio.py`.

---

## Instalación y arranque

```bash
pip install -r requirements.txt
pip install fastapi uvicorn httpx pyyaml

# Levantar el servidor
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Documentación interactiva disponible en `http://localhost:8000/docs` una vez levantado.

---

## Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/optimize/` | Optimiza con precios pre-cargados |
| POST | `/frontier/` | Calcula la frontera eficiente completa |
| POST | `/optimize-live/` | Descarga precios desde Alpaca y optimiza |
| POST | `/frontier-live/` | Descarga precios desde Alpaca y calcula la frontera |

---

## Cómo debe recibir la información la API

### Formato general de precios

Todos los endpoints que reciben precios (excepto `/optimize-live`) esperan el siguiente formato:

```json
"prices": {
  "AAPL": [150.0, 150.5, 151.2, ...],
  "TLT":  [100.0,  99.8,  99.5, ...],
  "GLD":  [170.0, 170.3, 170.1, ...]
}
```

- Cada clave es un ticker (debe coincidir con la lista `tickers`).
- Cada valor es una lista de precios de cierre en orden cronológico.
- Se recomienda mínimo **60 observaciones** (aprox. 3 meses diarios). Para modelos GARCH se requieren al menos 100.
- Los precios deben estar en la misma frecuencia (diaria, semanal, etc.).

---

## POST `/optimize/`

Produce un único portafolio óptimo.

### Request

```json
{
  "tickers": ["AAPL", "TLT", "GLD"],
  "prices": {
    "AAPL": [150.0, 150.5, 151.2],
    "TLT":  [100.0,  99.8,  99.5],
    "GLD":  [170.0, 170.3, 170.1]
  },
  "formulation": "mean_variance",
  "risk_aversion": 2.0,
  "solver": "cvxpy",
  "returns_model": "historical",
  "risk_model": "sample",
  "constraints": {
    "long_only": true,
    "budget": true,
    "box_min": null,
    "box_max": null
  }
}
```

### Parámetros

| Campo | Tipo | Default | Opciones / Rango | Descripción |
|-------|------|---------|-----------------|-------------|
| `tickers` | `List[str]` | requerido | mín. 2 elementos | Tickers del universo (deben estar en `prices`) |
| `prices` | `Dict[str, List[float]]` | requerido | — | Precios históricos por ticker |
| `formulation` | `str` | `"mean_variance"` | `mean_variance` `min_variance` `cvar` `tracking_error` | Objetivo de optimización |
| `risk_aversion` | `float` | `2.0` | `> 0` | λ del problema Markowitz. Mayor → menos riesgo. Solo aplica a `mean_variance` |
| `solver` | `str` | `"cvxpy"` | `cvxpy` `scipy` | Backend de optimización |
| `returns_model` | `str` | `"historical"` | `historical` `capm` | Modelo de retornos esperados |
| `risk_model` | `str` | `"sample"` | `sample` `shrinkage` `garch` | Modelo de covarianza |
| `constraints.long_only` | `bool` | `true` | — | Prohibir posiciones cortas (w ≥ 0) |
| `constraints.budget` | `bool` | `true` | — | Pesos suman 1 |
| `constraints.box_min` | `float\|null` | `null` | `[0.0, 1.0]` | Peso mínimo por activo (igual para todos) |
| `constraints.box_max` | `float\|null` | `null` | `[0.0, 1.0]` | Peso máximo por activo (igual para todos) |

### Response

```json
{
  "weights": {
    "AAPL": 0.45,
    "TLT":  0.40,
    "GLD":  0.15
  },
  "expected_return": 0.124,
  "expected_volatility": 0.087,
  "sharpe_ratio": 1.43,
  "status": "optimal",
  "solver_used": "cvxpy",
  "computation_ms": 45.2
}
```

| Campo | Descripción |
|-------|-------------|
| `weights` | Pesos por ticker. Suma = 1.0 |
| `expected_return` | Retorno esperado anualizado |
| `expected_volatility` | Volatilidad anualizada (desviación estándar) |
| `sharpe_ratio` | Sharpe ratio (r_f = 0). `null` si volatilidad ≈ 0 |
| `status` | `"optimal"` `"infeasible"` `"error"` |
| `solver_used` | Solver efectivamente usado |
| `computation_ms` | Tiempo de cómputo en milisegundos |

---

## POST `/frontier/`

Calcula N portafolios barriendo `risk_aversion` de min a max, trazando la frontera eficiente.

### Request

```json
{
  "tickers": ["AAPL", "TLT", "GLD"],
  "prices": {
    "AAPL": [150.0, 150.5, 151.2],
    "TLT":  [100.0,  99.8,  99.5],
    "GLD":  [170.0, 170.3, 170.1]
  },
  "n_points": 20,
  "risk_aversion_min": 0.5,
  "risk_aversion_max": 10.0,
  "solver": "cvxpy",
  "returns_model": "historical",
  "risk_model": "sample",
  "constraints": {
    "long_only": true,
    "budget": true
  }
}
```

### Parámetros adicionales vs `/optimize/`

| Campo | Tipo | Default | Rango | Descripción |
|-------|------|---------|-------|-------------|
| `n_points` | `int` | `20` | `[5, 100]` | Número de portafolios en la frontera |
| `risk_aversion_min` | `float` | `0.5` | `> 0` | λ mínimo (portafolio más agresivo) |
| `risk_aversion_max` | `float` | `10.0` | `> 0` | λ máximo (portafolio más conservador) |

### Response

```json
{
  "frontier": [
    {
      "weights": {"AAPL": 0.60, "TLT": 0.30, "GLD": 0.10},
      "expected_return": 0.15,
      "expected_volatility": 0.12,
      "sharpe_ratio": 1.25,
      "status": "optimal",
      "solver_used": "cvxpy",
      "computation_ms": 38.1
    }
  ],
  "min_variance_idx": 0,
  "max_sharpe_idx": 7,
  "computation_ms": 820.4
}
```

| Campo | Descripción |
|-------|-------------|
| `frontier` | Lista de `n_points` portafolios ordenados de menor a mayor volatilidad |
| `min_variance_idx` | Índice del portafolio de mínima varianza en `frontier` |
| `max_sharpe_idx` | Índice del portafolio con mayor Sharpe en `frontier` |
| `computation_ms` | Tiempo total (incluye los N cómputos en paralelo) |

---

## POST `/optimize-live/` *(rama `feature/api-conn-alpa`)*

Igual que `/optimize/` pero **sin enviar `prices`**: el API descarga los precios históricos desde Alpaca automáticamente.

### Request

```json
{
  "tickers": ["AAPL", "MSFT", "TLT", "GLD"],
  "years": 1,
  "alpaca_api_key": "TU_KEY_AQUI",
  "alpaca_secret_key": "TU_SECRET_AQUI",
  "formulation": "mean_variance",
  "risk_aversion": 2.0,
  "solver": "cvxpy",
  "returns_model": "historical",
  "risk_model": "sample",
  "constraints": {
    "long_only": true,
    "budget": true
  }
}
```

### Parámetros exclusivos

| Campo | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `years` | `int` | `1` | Años de historia a descargar desde Alpaca (1–10) |
| `alpaca_api_key` | `str\|null` | `null` | API Key de Alpaca. Si es `null` usa `ALPACA_API_KEY` del entorno |
| `alpaca_secret_key` | `str\|null` | `null` | Secret Key de Alpaca. Si es `null` usa `ALPACA_SECRET_KEY` del entorno |

### Credenciales — precedencia

```
1. Campos del request   →  alpaca_api_key / alpaca_secret_key
2. Variables de entorno →  ALPACA_API_KEY / ALPACA_SECRET_KEY
```

Si no se encuentran por ninguna vía, el endpoint retorna **HTTP 400** con mensaje de error.

### Response

Idéntica a `/optimize/`.

---

## Opciones de parámetros (referencia rápida)

### `formulation`

| Valor | Objetivo | Cuándo usarlo |
|-------|----------|---------------|
| `mean_variance` | Maximizar retorno ajustado por riesgo (Markowitz) | Caso general |
| `min_variance` | Minimizar varianza sin importar el retorno | Portafolios defensivos |
| `cvar` | Minimizar pérdida esperada en el 5% de peores escenarios | Gestión de riesgo de cola |
| `tracking_error` | Minimizar desviación respecto a un benchmark igual-ponderado | Portafolios indexados |

### `returns_model`

| Valor | Método | Cuándo usarlo |
|-------|--------|---------------|
| `historical` | Media histórica de retornos | Datos suficientes, sin vista de mercado |
| `capm` | E[R] = Rf + β(Rm - Rf) | Cuando se quiere ajustar por riesgo sistemático |

### `risk_model`

| Valor | Método | Cuándo usarlo |
|-------|--------|---------------|
| `sample` | Covarianza histórica directa | Muestras largas (> 250 obs.) |
| `shrinkage` | Shrinkage (Ledoit-Wolf) | Pocas observaciones o muchos activos |
| `garch` | GARCH(1,1) por activo + correlación global | Volatilidad dinámica / activos volátiles |

### `solver`

| Valor | Backend | Cuándo usarlo |
|-------|---------|---------------|
| `cvxpy` | CLARABEL (por defecto) | Problemas convexos: `mean_variance`, `min_variance`, `cvar` |
| `scipy` | SLSQP | Problemas no convexos o cuando cvxpy falla |

---

## Ejemplos con `curl`

### Health check
```bash
curl http://localhost:8000/health
```

### Optimización básica
```bash
curl -X POST http://localhost:8000/optimize/ \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "TLT", "GLD"],
    "prices": {
      "AAPL": [150.0, 150.5, 151.2, 151.8, 152.0],
      "TLT":  [100.0,  99.8,  99.5, 100.1, 100.3],
      "GLD":  [170.0, 170.3, 170.1, 170.8, 171.0]
    },
    "formulation": "mean_variance",
    "risk_aversion": 2.0
  }'
```

### Frontera eficiente (10 puntos)
```bash
curl -X POST http://localhost:8000/frontier/ \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "TLT", "GLD"],
    "prices": {
      "AAPL": [150.0, 150.5, 151.2, 151.8, 152.0],
      "TLT":  [100.0,  99.8,  99.5, 100.1, 100.3],
      "GLD":  [170.0, 170.3, 170.1, 170.8, 171.0]
    },
    "n_points": 10
  }'
```

### Optimize-live con Alpaca (credenciales en el request)
```bash
curl -X POST http://localhost:8000/optimize-live/ \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "TLT"],
    "years": 1,
    "alpaca_api_key": "TU_KEY",
    "alpaca_secret_key": "TU_SECRET",
    "formulation": "mean_variance",
    "risk_aversion": 2.0
  }'
```

### Optimize-live con credenciales de entorno
```bash
export ALPACA_API_KEY="TU_KEY"
export ALPACA_SECRET_KEY="TU_SECRET"

curl -X POST http://localhost:8000/optimize-live/ \
  -H "Content-Type: application/json" \
  -d '{
    "tickers": ["AAPL", "MSFT", "TLT"],
    "years": 1
  }'
```

---

## Uso desde Python

```python
import httpx, asyncio

async def ejemplo():
    precios = {
        "AAPL": [150.0 + i * 0.1 for i in range(252)],
        "TLT":  [100.0 - i * 0.05 for i in range(252)],
        "GLD":  [170.0 + i * 0.02 for i in range(252)],
    }

    async with httpx.AsyncClient() as client:
        # Optimización
        r = await client.post("http://localhost:8000/optimize/", json={
            "tickers": ["AAPL", "TLT", "GLD"],
            "prices": precios,
            "risk_aversion": 3.0,
        })
        print(r.json()["weights"])

        # Frontera
        r = await client.post("http://localhost:8000/frontier/", json={
            "tickers": ["AAPL", "TLT", "GLD"],
            "prices": precios,
            "n_points": 15,
        })
        print(f"Puntos frontera: {len(r.json()['frontier'])}")

asyncio.run(ejemplo())
```

---

## Uso desde el MCP (`mcp_portfolio.py`)

```python
import asyncio
from mcp_portfolio import optimize, frontier

precios = {
    "AAPL": [150.0 + i * 0.1 for i in range(252)],
    "TLT":  [100.0 - i * 0.05 for i in range(252)],
    "GLD":  [170.0 + i * 0.02 for i in range(252)],
}

async def main():
    resultado = await optimize(
        tickers=["AAPL", "TLT", "GLD"],
        prices=precios,
        formulation="mean_variance",
        risk_aversion=2.0,
    )
    print(resultado["weights"])

asyncio.run(main())
```

---

## POST `/frontier-live/` *(rama `feature/api-conn-alpa`)*

Igual que `/frontier/` pero sin enviar `prices`: el API descarga desde Alpaca y barre los N puntos de la frontera. La descarga ocurre **una sola vez**; las N optimizaciones corren en paralelo.

### Request

```json
{
  "tickers": ["LRCX", "GOOGL", "JPM", "XOM", "KO", "MCD"],
  "years": 1,
  "alpaca_api_key": "TU_KEY",
  "alpaca_secret_key": "TU_SECRET",
  "n_points": 20,
  "risk_aversion_min": 0.5,
  "risk_aversion_max": 10.0
}
```

### Response

Idéntica a `/frontier/`.

---

### Script demo — `frontier_alpaca.py`

Corre la frontera eficiente para los 17 tickers del notebook directamente desde la línea de comandos, sin necesidad de levantar el servidor:

```bash
# Con variables de entorno
export ALPACA_API_KEY="tu_key"
export ALPACA_SECRET_KEY="tu_secret"
python frontier_alpaca.py

# O pasando las claves directo
python frontier_alpaca.py --key TU_KEY --secret TU_SECRET --years 1 --points 20
```

Universo: `LRCX GOOGL GOOG ASML GLW JPM XOM KO MCD IBM CAT BA DIS PG PYPL F NU`

---

## Códigos de error

| HTTP | Causa |
|------|-------|
| `422` | Validación Pydantic fallida (ej. menos de 2 tickers, tipo incorrecto) |
| `400` | Parámetro inválido (ej. formulación no registrada, credenciales Alpaca ausentes) |
| `500` | Error interno del solver u Orchestrator |
