"""
Tests del API usando TestClient de FastAPI.

Por qué TestClient: crea un servidor en memoria (ASGI transport directo),
no necesitas levantar uvicorn para testear. Cada test es independiente
y puede correr en CI sin puertos ocupados.
"""

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

# Datos de prueba mínimos (3 activos, 252 días)
# Precios lineales con pendientes distintas → retornos constantes no nulos
# suficientes para que los modelos de covarianza sean no degenerados.
SAMPLE_PRICES = {
    "AAPL": [150.0 + i * 0.1 for i in range(252)],
    "TLT":  [100.0 - i * 0.05 for i in range(252)],
    "GLD":  [170.0 + i * 0.02 for i in range(252)],
}


def test_health():
    """El endpoint /health debe responder 200 con status ok."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_optimize_mean_variance():
    """
    Optimización mean_variance básica.
    Verifica: status 200, pesos presentes y suma = 1.
    """
    payload = {
        "tickers": ["AAPL", "TLT", "GLD"],
        "prices": SAMPLE_PRICES,
        "formulation": "mean_variance",
        "risk_aversion": 2.0,
    }
    r = client.post("/optimize/", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "weights" in data
    assert abs(sum(data["weights"].values()) - 1.0) < 1e-4, (
        f"Pesos no suman 1: {sum(data['weights'].values())}"
    )
    assert data["status"] == "optimal", f"Status inesperado: {data['status']}"


def test_frontier_returns_n_points():
    """
    /frontier debe retornar exactamente n_points portafolios.
    """
    payload = {
        "tickers": ["AAPL", "TLT", "GLD"],
        "prices": SAMPLE_PRICES,
        "n_points": 10,
    }
    r = client.post("/frontier/", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["frontier"]) == 10, (
        f"Se esperaban 10 puntos, se obtuvieron {len(data['frontier'])}"
    )


def test_optimize_invalid_single_ticker():
    """Pydantic debe rechazar un universo de 1 solo ticker (min_length=2)."""
    payload = {"tickers": ["AAPL"], "prices": {"AAPL": [100.0] * 252}}
    r = client.post("/optimize/", json=payload)
    assert r.status_code == 422, f"Se esperaba 422, se obtuvo {r.status_code}"
