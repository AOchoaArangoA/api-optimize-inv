"""
Traductor central: convierte los strings del JSON en objetos portopt.

Por qué registries (dicts) en lugar de if/elif encadenados:
  - Añadir una nueva formulación = agregar una línea al dict, no modificar lógica.
  - Más legible y testeable.
  - Falla con KeyError claro si llega un string no registrado.
"""

import sys
import os

# Asegurar que src/ esté en el path para importar portopt sin instalación
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np

from portopt.optimization.formulations import (
    MeanVarianceOptimization,
    MinVarianceOptimization,
    CVaROptimization,
    TrackingErrorOptimization,
)
from portopt.optimization.solvers import CVXPYSolver, ScipySolver
from portopt.returns import HistoricalReturns, CAPMReturns
from portopt.risk import SampleCovariance, ShrinkageCovariance, GARCHModel
from portopt.optimization.constraints import (
    BudgetConstraint,
    LongOnlyConstraint,
    BoxConstraint,
)

from api.schemas.requests import ConstraintsConfig

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


def build_formulation(formulation: str, n_assets: int, risk_aversion: float):
    """
    Instancia la formulación del problema de optimización.

    TrackingErrorOptimization usa benchmark igual-ponderado por defecto
    (se puede sobreescribir después vía problem.benchmark_weights).
    CVaROptimization recibe returns_scenarios inyectados por el Orchestrator.
    """
    if formulation not in FORMULATION_REGISTRY:
        raise ValueError(
            f"Formulación '{formulation}' no registrada. "
            f"Opciones: {list(FORMULATION_REGISTRY)}"
        )

    cls = FORMULATION_REGISTRY[formulation]

    if formulation == "mean_variance":
        return cls(n_assets=n_assets, risk_aversion=risk_aversion)
    elif formulation == "tracking_error":
        # Benchmark igual-ponderado: cada activo pesa 1/n
        benchmark = np.full(n_assets, 1.0 / n_assets)
        return cls(n_assets=n_assets, benchmark_weights=benchmark)
    else:
        # min_variance y cvar solo necesitan n_assets; la covarianza/escenarios
        # los inyecta el Orchestrator en tiempo de run().
        return cls(n_assets=n_assets)


def build_solver(solver: str):
    """
    Instancia el solver. CLARABEL es el backend por defecto de CVXPY
    (más estable numéricamente que OSQP para problemas mal condicionados).
    """
    if solver not in SOLVER_REGISTRY:
        raise ValueError(
            f"Solver '{solver}' no registrado. Opciones: {list(SOLVER_REGISTRY)}"
        )
    return SOLVER_REGISTRY[solver]()


def build_returns_model(returns_model: str):
    """Instancia el modelo de retornos esperados."""
    if returns_model not in RETURNS_REGISTRY:
        raise ValueError(
            f"Modelo de retornos '{returns_model}' no registrado. "
            f"Opciones: {list(RETURNS_REGISTRY)}"
        )
    return RETURNS_REGISTRY[returns_model]()


def build_risk_model(risk_model: str):
    """
    Instancia el modelo de riesgo.
    SampleCovariance anualiza por defecto (factor 252 días hábiles).
    """
    if risk_model not in RISK_REGISTRY:
        raise ValueError(
            f"Modelo de riesgo '{risk_model}' no registrado. "
            f"Opciones: {list(RISK_REGISTRY)}"
        )
    cls = RISK_REGISTRY[risk_model]
    if risk_model == "sample":
        return cls(annualize=True)
    return cls()


def build_constraints(config: ConstraintsConfig, n_assets: int) -> list:
    """
    Construye la lista de restricciones a partir del config del request.

    BoxConstraint recibe arrays numpy para que pueda operar por activo.
    Se construyen con el mismo escalar replicado n_assets veces.
    """
    constraints = []
    if config.budget:
        constraints.append(BudgetConstraint())
    if config.long_only:
        constraints.append(LongOnlyConstraint())
    if config.box_min is not None and config.box_max is not None:
        lower = np.full(n_assets, config.box_min)
        upper = np.full(n_assets, config.box_max)
        constraints.append(BoxConstraint(lower_bounds=lower, upper_bounds=upper))
    return constraints
