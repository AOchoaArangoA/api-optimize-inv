"""
example_flow.py — Flujo end-to-end del paquete portopt
=======================================================

Cubre los tres escenarios de uso principales:

  ESCENARIO 1 — Flujo directo (un modelo por bloque)
  ESCENARIO 2 — Flujo multi-clase (consolidadores)
  ESCENARIO 3 — Flujo vía Orchestrator

Cada escenario puede ejecutarse de forma independiente.
Los datos son sintéticos para no depender de APIs externas.

Flujo de datos en todos los escenarios:
                                          ┌─ returns_model.fit(returns).predict() ─► μ ─┐
  prices ─► pct_change().dropna() ─► returns                                             ├─► problem ─► solver ─► OptimizationResult
                                          └─ risk_model.fit(returns).covariance() ─► Σ ──┘
"""

import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, "src")

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

# ===========================================================================
# DATOS SINTÉTICOS (compartidos por todos los escenarios)
# ===========================================================================

np.random.seed(42)
N_PERIODS = 504    # ~2 años de datos diarios
TICKERS_EQ  = ["AAPL", "MSFT", "GOOG", "AMZN"]   # renta variable
TICKERS_FI  = ["TLT", "IEF"]                        # renta fija
ALL_TICKERS = TICKERS_EQ + TICKERS_FI

# Simulación de precios con distintas características por clase
returns_eq = np.random.randn(N_PERIODS, len(TICKERS_EQ)) * 0.012 + 0.0003
returns_fi = np.random.randn(N_PERIODS, len(TICKERS_FI)) * 0.004 + 0.0001
raw_returns = np.concatenate([returns_eq, returns_fi], axis=1)
prices = pd.DataFrame(
    np.cumprod(1 + raw_returns, axis=0) * 100,
    columns=ALL_TICKERS,
)

print("=" * 70)
print(f"  Datos: {prices.shape[0]} períodos | {prices.shape[1]} activos")
print(f"  Activos: {ALL_TICKERS}")
print("=" * 70)


# ===========================================================================
# ESCENARIO 1 — FLUJO DIRECTO
#
# Un único modelo de retornos + un único modelo de riesgo
# Problema: Media-Varianza    Solver: CVXPY
# ===========================================================================

def escenario_1_flujo_directo():
    print("\n" + "=" * 70)
    print("  ESCENARIO 1: Flujo directo — un modelo por bloque")
    print("=" * 70)

    from portopt.returns.historical import HistoricalReturns
    from portopt.risk.sample_cov import SampleCovariance
    from portopt.optimization.formulations.mean_variance import MeanVarianceOptimization
    from portopt.optimization.constraints.budget import BudgetConstraint
    from portopt.optimization.constraints.long_only import LongOnlyConstraint
    from portopt.optimization.constraints.box import BoxConstraint
    from portopt.optimization.solvers.cvxpy import CVXPYSolver

    n = len(ALL_TICKERS)

    # ------------------------------------------------------------------
    # Paso 1: retornos (única conversión)
    # ------------------------------------------------------------------
    returns = prices.pct_change().dropna()
    print(f"\n[1] Retornos computados: shape={returns.shape}")

    # ------------------------------------------------------------------
    # Paso 2: modelo de retornos
    # ------------------------------------------------------------------
    returns_model = HistoricalReturns(method="mean")
    mu = returns_model.fit(returns).predict()
    print(f"[2] Retornos esperados (mu): {mu.round(5)}")

    # ------------------------------------------------------------------
    # Paso 3: modelo de riesgo
    # ------------------------------------------------------------------
    risk_model = SampleCovariance(annualize=True)
    cov = risk_model.fit(returns).covariance()
    vols = risk_model.volatility()
    print(f"[3] Covarianza estimada: shape={cov.shape}")
    print(f"    Volatilidades anualizadas: {dict(zip(ALL_TICKERS, vols.round(4)))}")

    # ------------------------------------------------------------------
    # Paso 4: formulación + restricciones
    # ------------------------------------------------------------------
    problem = MeanVarianceOptimization(
        n_assets=n,
        expected_returns=mu,
        covariance_matrix=cov,
        risk_aversion=2.0,
    )
    problem.add_constraint(BudgetConstraint())
    problem.add_constraint(LongOnlyConstraint())
    problem.add_constraint(BoxConstraint(
        lower_bounds=np.full(n, 0.02),    # mínimo 2% por activo
        upper_bounds=np.full(n, 0.40),    # máximo 40% por activo
    ))
    print(f"[4] Problema: {problem}")

    # ------------------------------------------------------------------
    # Paso 5: resolver
    # ------------------------------------------------------------------
    result = CVXPYSolver(solver="CLARABEL").optimize(problem)

    print(f"[5] Estado: {result.status} | objetivo={result.objective_value:.6f}")
    if result.success:
        w_dict = dict(zip(ALL_TICKERS, result.weights.round(4)))
        print(f"    Pesos: {w_dict}")
        print(f"    Suma pesos: {result.weights.sum():.6f}")
        print(f"    Retorno esperado portafolio: {(mu @ result.weights):.5f}")
        print(f"    Volatilidad portafolio:      {np.sqrt(result.weights @ cov @ result.weights):.5f}")

    return result


# ===========================================================================
# ESCENARIO 2 — FLUJO MULTI-CLASE CON CONSOLIDADORES
#
# Modelo de retornos distinto por clase de activo:
#   Renta variable → CAPM
#   Renta fija     → HistoricalReturns
#
# Modelo de riesgo distinto por clase:
#   Renta variable → ShrinkageCovariance
#   Renta fija     → SampleCovariance
#
# Problema: Mínima Varianza    Solver: SciPy
# ===========================================================================

def escenario_2_flujo_multi_clase():
    print("\n" + "=" * 70)
    print("  ESCENARIO 2: Flujo multi-clase — consolidadores")
    print("=" * 70)

    from portopt.core.asset_class import AssetClass
    from portopt.core.asset_class_registry import AssetClassRegistry
    from portopt.returns.historical import HistoricalReturns
    from portopt.returns.capm import CAPMReturns
    from portopt.returns.consolidator import ReturnsConsolidator
    from portopt.risk.sample_cov import SampleCovariance
    from portopt.risk.shrinkage import ShrinkageCovariance
    from portopt.risk.consolidator import RiskConsolidator
    from portopt.optimization.formulations.min_variance import MinVarianceOptimization
    from portopt.optimization.constraints.budget import BudgetConstraint
    from portopt.optimization.constraints.long_only import LongOnlyConstraint
    from portopt.optimization.solvers.scipy import ScipySolver

    n = len(ALL_TICKERS)

    # ------------------------------------------------------------------
    # Registro de clasificación
    # ------------------------------------------------------------------
    registry = AssetClassRegistry()
    registry.register_batch({t: AssetClass.EQUITIES    for t in TICKERS_EQ})
    registry.register_batch({t: AssetClass.FIXED_INCOME for t in TICKERS_FI})
    print(f"\n[1] Registry: {registry}")
    print(f"    Clasificación: {registry.get_summary()}")

    # ------------------------------------------------------------------
    # Paso 1: retornos
    # ------------------------------------------------------------------
    returns = prices.pct_change().dropna()

    # ------------------------------------------------------------------
    # Paso 2: consolidador de retornos
    # ------------------------------------------------------------------
    ret_consolidator = ReturnsConsolidator(registry=registry)
    ret_consolidator.add_model(AssetClass.EQUITIES,    CAPMReturns(risk_free_rate=0.0001))
    ret_consolidator.add_model(AssetClass.FIXED_INCOME, HistoricalReturns(method="mean"))

    mu = ret_consolidator.consolidate(returns)
    print(f"[2] Retornos esperados por modelo:")
    for t, m in zip(ALL_TICKERS, mu):
        cls = registry.get_asset_class(t).value
        print(f"    {t:6s} ({cls:12s}): {m:.5f}")

    # ------------------------------------------------------------------
    # Paso 3: consolidador de riesgo
    # ------------------------------------------------------------------
    risk_consolidator = RiskConsolidator(registry=registry)
    risk_consolidator.add_model(AssetClass.EQUITIES,    ShrinkageCovariance(annualize=True))
    risk_consolidator.add_model(AssetClass.FIXED_INCOME, SampleCovariance(annualize=True))

    cov = risk_consolidator.consolidate(returns)
    print(f"[3] Covarianza consolidada: shape={cov.shape}")
    vols = np.sqrt(np.diag(cov))
    print(f"    Volatilidades: {dict(zip(ALL_TICKERS, vols.round(4)))}")

    # ------------------------------------------------------------------
    # Paso 4: formulación + restricciones
    # ------------------------------------------------------------------
    problem = MinVarianceOptimization(
        n_assets=n,
        covariance_matrix=cov,
        constraints=[BudgetConstraint(), LongOnlyConstraint()],
    )
    print(f"[4] Problema: {problem}")

    # ------------------------------------------------------------------
    # Paso 5: resolver con SciPy
    # ------------------------------------------------------------------
    result = ScipySolver(method="SLSQP").optimize(problem)

    print(f"[5] Estado: {result.status}")
    if result.success:
        w_dict = dict(zip(ALL_TICKERS, result.weights.round(4)))
        print(f"    Pesos: {w_dict}")
        print(f"    Suma pesos: {result.weights.sum():.6f}")
        print(f"    Volatilidad portafolio: {np.sqrt(result.weights @ cov @ result.weights):.5f}")
        print(f"    Iteraciones SciPy: {result.metadata['n_iterations']}")

    return result


# ===========================================================================
# ESCENARIO 3 — FLUJO VÍA ORCHESTRATOR
#
# El PortfolioOrchestrator encapsula todo el flujo:
#   1. Conversión precios → retornos (una sola vez)
#   2. Inyección de μ y Σ en el problema
#   3. Invocación del solver
#
# Problema: CVaR (Expected Shortfall)    Solver: CVXPY
# ===========================================================================

def escenario_3_orchestrator():
    print("\n" + "=" * 70)
    print("  ESCENARIO 3: Orchestrator — flujo encapsulado")
    print("=" * 70)

    from portopt.returns.historical import HistoricalReturns
    from portopt.risk.shrinkage import ShrinkageCovariance
    from portopt.optimization.formulations.cvar import CVaROptimization
    from portopt.optimization.constraints.budget import BudgetConstraint
    from portopt.optimization.constraints.long_only import LongOnlyConstraint
    from portopt.optimization.solvers.cvxpy import CVXPYSolver
    from portopt.pipeline.orchestrator import PortfolioOrchestrator

    n = len(ALL_TICKERS)

    # El problema CVaR necesita el array de escenarios,
    # el Orchestrator lo inyecta automáticamente si el atributo existe.
    problem = CVaROptimization(
        n_assets=n,
        alpha=0.05,   # CVaR al 95%
        constraints=[BudgetConstraint(), LongOnlyConstraint()],
    )

    orch = PortfolioOrchestrator()
    result = orch.run(
        data=prices,
        returns_model=HistoricalReturns(method="mean"),
        risk_model=ShrinkageCovariance(annualize=True),
        problem=problem,
        solver=CVXPYSolver(solver="CLARABEL"),
    )

    print(f"\n[resultado] Estado: {result.status}")
    if result.success:
        w_dict = dict(zip(ALL_TICKERS, result.weights.round(4)))
        print(f"  Pesos: {w_dict}")
        print(f"  Suma pesos: {result.weights.sum():.6f}")
        print(f"  Valor CVaR objetivo: {result.objective_value:.6f}")
        print(f"  Metadatos solver: {result.metadata}")

    port = orch.get_portfolio()
    print(f"  Portafolio: {port}")

    return result


# ===========================================================================
# EXTRA — Comparativa de formulaciones
#
# Muestra cómo el mismo par (μ, Σ) genera pesos distintos según la
# formulación elegida.
# ===========================================================================

def comparativa_formulaciones():
    print("\n" + "=" * 70)
    print("  EXTRA: Comparativa de formulaciones")
    print("=" * 70)

    from portopt.returns.historical import HistoricalReturns
    from portopt.risk.shrinkage import ShrinkageCovariance
    from portopt.optimization.formulations.mean_variance import MeanVarianceOptimization
    from portopt.optimization.formulations.min_variance import MinVarianceOptimization
    from portopt.optimization.formulations.tracking_error import TrackingErrorOptimization
    from portopt.optimization.formulations.cvar import CVaROptimization
    from portopt.optimization.constraints.budget import BudgetConstraint
    from portopt.optimization.constraints.long_only import LongOnlyConstraint
    from portopt.optimization.solvers.cvxpy import CVXPYSolver

    n = len(ALL_TICKERS)
    returns = prices.pct_change().dropna()

    mu  = HistoricalReturns().fit(returns).predict()
    cov = ShrinkageCovariance(annualize=True).fit(returns).covariance()

    solver = CVXPYSolver(solver="CLARABEL")
    base_constraints = [BudgetConstraint(), LongOnlyConstraint()]

    problems = {
        "MeanVariance (ra=1)": MeanVarianceOptimization(n, mu, cov, risk_aversion=1.0, constraints=base_constraints),
        "MeanVariance (ra=5)": MeanVarianceOptimization(n, mu, cov, risk_aversion=5.0, constraints=base_constraints),
        "MinVariance":         MinVarianceOptimization(n, cov, constraints=base_constraints),
        "TrackingError":       TrackingErrorOptimization(n, cov,
                                   benchmark_weights=np.ones(n)/n,
                                   constraints=base_constraints),
        "CVaR (alpha=5%)":     CVaROptimization(n, returns.values, alpha=0.05, constraints=base_constraints),
    }

    header = f"{'Formulacion':28s}" + "".join(f"{t:8s}" for t in ALL_TICKERS) + "  Vol (anual)"
    print(f"\n{header}")
    print("-" * len(header))

    for name, prob in problems.items():
        res = solver.optimize(prob)
        if res.success:
            w = res.weights
            vol = np.sqrt(w @ cov @ w)
            row = f"{name:28s}" + "".join(f"{wi:8.3f}" for wi in w) + f"  {vol:.4f}"
            print(row)
        else:
            print(f"{name:28s}  FAILED ({res.status})")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    r1 = escenario_1_flujo_directo()
    r2 = escenario_2_flujo_multi_clase()
    r3 = escenario_3_orchestrator()
    comparativa_formulaciones()

    print("\n" + "=" * 70)
    print("  Todos los escenarios completados exitosamente.")
    print("=" * 70)
