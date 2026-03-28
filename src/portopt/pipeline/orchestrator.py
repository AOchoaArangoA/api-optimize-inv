"""
PortfolioOrchestrator — End-to-end portfolio optimization workflow.

Data flow
---------
prices (raw)
    │  pct_change().dropna()    ← single conversion, done here
    ▼
returns (pd.DataFrame)
    ├──► returns_model.fit(returns).predict()   → expected_returns (np.ndarray)
    └──► risk_model.fit(returns).covariance()   → cov_matrix (np.ndarray)
                    │
                    ▼
            problem.expected_returns = ...
            problem.covariance_matrix = ...
                    │
                    ▼
            solver.optimize(problem)  →  OptimizationResult
                    │
                    ▼
            Portfolio + OptimizationResult
"""

import logging
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..core.portfolio import Portfolio
from ..core.returns import ExpectedReturnsModel
from ..core.risk import RiskModel
from ..core.optimizer import Optimizer
from ..core.formulation import OptimizationProblem
from ..core.result import OptimizationResult

logger = logging.getLogger(__name__)


class PortfolioOrchestrator:
    """
    Orquestador del flujo completo de optimización.

    Acepta tanto modelos simples (``ExpectedReturnsModel``, ``RiskModel``)
    como consolidadores multi-clase (``ReturnsConsolidator``, ``RiskConsolidator``).

    La conversión precios → retornos ocurre **una sola vez** al inicio.
    Todos los modelos reciben retornos ya computados.

    Parameters
    ----------
    prices_to_returns : callable, optional
        Función para transformar precios en retornos.
        Por defecto: ``lambda df: df.pct_change().dropna()``.
    """

    def __init__(self, prices_to_returns=None):
        self._to_returns = prices_to_returns or (
            lambda df: df.pct_change().dropna()
        )
        self.result_: Optional[OptimizationResult] = None
        self.portfolio_: Optional[Portfolio] = None
        self.expected_returns_: Optional[np.ndarray] = None
        self.cov_matrix_: Optional[np.ndarray] = None

    def run(
        self,
        data: pd.DataFrame,
        returns_model: Any,
        risk_model: Any,
        problem: OptimizationProblem,
        solver: Optimizer,
    ) -> OptimizationResult:
        """
        Ejecuta el flujo completo de optimización.

        Parameters
        ----------
        data : pd.DataFrame
            Precios históricos. Columnas = tickers, filas = fechas.
        returns_model : ExpectedReturnsModel | ReturnsConsolidator
            Modelo para estimar retornos esperados.
        risk_model : RiskModel | RiskConsolidator
            Modelo para estimar la covarianza.
        problem : OptimizationProblem
            Formulación (debe tener atributos ``expected_returns`` y/o
            ``covariance_matrix`` que se inyectan aquí).
        solver : Optimizer
            Solver a usar.

        Returns
        -------
        OptimizationResult
        """
        assets = list(data.columns)
        logger.info("Iniciando optimización de portafolio | %d activos", len(assets))

        # ------------------------------------------------------------------
        # Paso 1: Retornos (conversión única)
        # ------------------------------------------------------------------
        returns = self._to_returns(data)
        logger.info("Retornos computados | shape %s", returns.shape)

        # ------------------------------------------------------------------
        # Paso 2: Retornos esperados
        # ------------------------------------------------------------------
        from ..returns.consolidator import ReturnsConsolidator

        if isinstance(returns_model, ReturnsConsolidator):
            self.expected_returns_ = returns_model.consolidate(returns)
            logger.info(
                "ReturnsConsolidator | clases: %s",
                returns_model.get_summary()["asset_classes"],
            )
        else:
            self.expected_returns_ = returns_model.fit(returns).predict()

        logger.info(
            "Retornos esperados | media=%.4f  std=%.4f",
            float(self.expected_returns_.mean()),
            float(self.expected_returns_.std()),
        )

        # ------------------------------------------------------------------
        # Paso 3: Covarianza
        # ------------------------------------------------------------------
        from ..risk.consolidator import RiskConsolidator

        if isinstance(risk_model, RiskConsolidator):
            self.cov_matrix_ = risk_model.consolidate(returns)
            logger.info(
                "RiskConsolidator | clases: %s",
                risk_model.get_summary()["asset_classes"],
            )
        else:
            self.cov_matrix_ = risk_model.fit(returns).covariance()

        avg_vol = float(np.sqrt(np.diag(self.cov_matrix_)).mean())
        logger.info("Covarianza estimada | volatilidad promedio=%.4f", avg_vol)

        # ------------------------------------------------------------------
        # Paso 4: Inyectar estimaciones en el problema
        # ------------------------------------------------------------------
        if hasattr(problem, "expected_returns"):
            problem.expected_returns = self.expected_returns_
        if hasattr(problem, "covariance_matrix"):
            problem.covariance_matrix = self.cov_matrix_
        if hasattr(problem, "returns_scenarios"):
            problem.returns_scenarios = returns.values
            if hasattr(problem, "n_scenarios"):
                problem.n_scenarios = returns.shape[0]

        logger.info(
            "Problema configurado | %s | %d restricciones",
            type(problem).__name__,
            len(problem.get_constraints()),
        )

        # ------------------------------------------------------------------
        # Paso 5: Resolver
        # ------------------------------------------------------------------
        self.result_ = solver.optimize(problem)

        if self.result_.success:
            logger.info(
                "Optimización exitosa | objetivo=%.6f",
                self.result_.objective_value,
            )
        else:
            logger.warning(
                "Optimización no convergió | estado=%s | %s",
                self.result_.status,
                self.result_.metadata,
            )

        # ------------------------------------------------------------------
        # Paso 6: Construir Portfolio
        # ------------------------------------------------------------------
        weights = (
            self.result_.weights
            if self.result_.weights is not None
            else np.zeros(len(assets))
        )
        self.portfolio_ = Portfolio(assets=assets, weights=weights, data=data)

        return self.result_

    # ------------------------------------------------------------------
    # Backwards-compatible alias
    # ------------------------------------------------------------------

    def run_optimization(self, data, returns_model, risk_model, problem, solver, **_):
        """Alias de run() para compatibilidad con código anterior."""
        result = self.run(data, returns_model, risk_model, problem, solver)
        return {
            "portfolio": self.portfolio_,
            "optimization_result": result,
            "expected_returns": self.expected_returns_,
            "covariance_matrix": self.cov_matrix_,
        }

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_portfolio(self) -> Optional[Portfolio]:
        return self.portfolio_

    def get_results(self) -> Dict[str, Any]:
        return {
            "portfolio": self.portfolio_,
            "optimization_result": self.result_,
            "expected_returns": self.expected_returns_,
            "covariance_matrix": self.cov_matrix_,
        }
