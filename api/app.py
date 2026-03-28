"""
Punto de entrada de FastAPI.

Patrón 'application factory': la instancia `app` se crea aquí y los routers
se montan sobre ella. Esto permite importar `app` en tests sin levantar uvicorn,
y en producción lanzarla con: uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import sys
import os

# Asegurar path a src/ antes de que los routers importen portopt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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
    """
    Endpoint de health check.
    El agente (MCP) lo llama antes de enviar requests para verificar
    que el servidor está disponible.
    """
    return {"status": "ok", "version": "1.0.0"}
