from src.portopt.returns.data_downloader import AlpacaDataDownloader, MassiveDataDownloader
import os
import numpy as np 

from src.portopt import AssetClass, AssetClassRegistry
from src.portopt.returns import HistoricalReturns, CAPMReturns, ReturnsConsolidator
from src.portopt.risk import SampleCovariance, GARCHModel, RiskConsolidator
from src.portopt.optimization.formulations import MeanVarianceOptimization
from src.portopt.optimization.constraints import (
    BudgetConstraint,
    LongOnlyConstraint,
    AssetClassConstraint
)
from src.portopt.optimization.solvers import CVXPYSolver, ScipySolver
from src.portopt.optimization import (
    RebalancingStrategy,
    AdaptiveRebalancingStrategy,
    RebalancingTrigger
)

from src.portopt.pipeline.preprocessing import pivot_data
from src.portopt.utils import (
    plot_asset_weights,
    plot_price_evolution,
    plot_efficient_frontier,
    plot_correlation_heatmap,
    plot_risk_return_scatter,
)

ALPACA_KEY = os.getenv("ALPACA_API_KEY", "PKS5UVEPDZSMXL4UI5Y76OFRNX")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "5wbyXagWumPCydrKFSwo1ocdxCRpjG3Z1CG4EeY9tH8k")


print("\n--- Probando Alpaca Downloader ---")
alpaca = AlpacaDataDownloader(ALPACA_KEY, ALPACA_SECRET)

tickets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 'NU','CRM']

try:
    print(f"\nDescargando Stocks {tickets}...")
    stock_data = alpaca.fetch_stock_data(tickets, years=1)
    print("\nSe realiza la descarga de manera efectiva")
    stock_data = pivot_data(stock_data)
except Exception as e:
    print(f"Error en Alpaca: {e}")


# Registro de activos

registry = AssetClassRegistry()
registry.register_batch({
    'AAPL': AssetClass.EQUITIES, 'MSFT': AssetClass.EQUITIES, 'GOOGL': AssetClass.EQUITIES,
    'AMZN': AssetClass.EQUITIES, 'TSLA': AssetClass.EQUITIES, 'NU': AssetClass.EQUITIES,
    'CRM': AssetClass.EQUITIES
})

# Registro de modelos de retornos
returns_consolidator = ReturnsConsolidator(registry=registry)
returns_consolidator.add_asset_class_model(AssetClass.EQUITIES, HistoricalReturns(method='mean'))

# Registro de modelos de riesgo
risk_consolidator = RiskConsolidator(registry=registry)
risk_consolidator.add_asset_class_model(AssetClass.EQUITIES, SampleCovariance(annualize=True))

# Consolidación de retornos y riesgo
returns = returns_consolidator.consolidate(stock_data)
risk = risk_consolidator.consolidate(stock_data)


# Límites por clase: con solo EQUITIES, el total en equities debe ser 100%.
# Para que sea factible, solo restringir EQUITIES (ej. mínimo 5%, máximo 100%).
# Las clases sin activos (FIXED_INCOME, COMMODITIES) deben permitir 0%: (0, 1).
class_limits = {
    AssetClass.EQUITIES: (0.05, 1.00),
    AssetClass.FIXED_INCOME: (0.00, 1.00),
    AssetClass.COMMODITIES: (0.00, 1.00),
}

problem = MeanVarianceOptimization(n_assets=len(tickets), risk_aversion=1)
problem.expected_returns = returns
problem.covariance_matrix = risk

problem.add_constraint(BudgetConstraint(budget_target=1.0))
problem.add_constraint(LongOnlyConstraint())
problem.add_constraint(AssetClassConstraint(registry, class_limits, tickets))

# Intentar primero CVXPY; si falla (ej. infeasible), usar SciPy como respaldo
solver = CVXPYSolver(solver="OSQP")
result = solver.optimize(problem)

if result["status"] != "success":
    print(f"CVXPY: {result['status']} ({result.get('metadata', {})}). Probando SciPy...")
    solver = ScipySolver(method="SLSQP", max_iter=2000)
    result = solver.optimize(problem)
    if result["status"] == "success":
        print("SciPy optimizó correctamente.")
    else:
        print(f"SciPy: {result['status']} - {result.get('metadata', {}).get('message', '')}")

print(result)
print(result.keys())
print(result["weights"])

# --- Proceso de graficación para la estrategia de optimización ---
if result["status"] == "success":
    weights = result["weights"]
    weights_dict = {tickets[i]: weights[i] for i in range(len(tickets))}
    asset_classes_dict = {t: registry.get_asset_class(t).value for t in tickets}

    # 1. Pesos óptimos del portafolio (barras)
    fig_weights = plot_asset_weights(
        weights_dict,
        asset_classes=asset_classes_dict,
        title="Pesos Óptimos - Estrategia Media-Varianza (Equities)",
    )
    fig_weights.show()

    # 2. Evolución de precios normalizados
    fig_prices = plot_price_evolution(
        stock_data,
        title="Evolución de Precios Normalizados (Base 100)",
    )
    fig_prices.show()

    # 3. Matriz de correlación
    fig_corr = plot_correlation_heatmap(
        risk, tickets,
        title="Matriz de Correlación (desde covarianza)",
    )
    fig_corr.show()

    # 4. Riesgo vs retorno por activo y portafolio óptimo
    ann_ret = returns * 252
    ann_vol = np.sqrt(np.diag(risk)) * np.sqrt(252)
    port_ret = np.dot(returns, weights) * 252
    port_vol = np.sqrt(weights @ risk @ weights) * np.sqrt(252)
    fig_scatter = plot_risk_return_scatter(
        ann_ret, ann_vol, tickets,
        port_return=port_ret, port_vol=port_vol,
        title="Riesgo vs Retorno - Activos y Portafolio Óptimo",
    )
    fig_scatter.show()

    # 5. Frontera eficiente (varios niveles de aversión al riesgo)
    lambdas = np.logspace(-1, 2, 20)
    frontier_returns, frontier_vols = [], []
    for lam in lambdas:
        problem.risk_aversion = lam
        res = solver.optimize(problem)
        if res["status"] == "success":
            w = res["weights"]
            r_p = np.dot(returns, w) * 252
            v_p = np.sqrt(w @ risk @ w) * np.sqrt(252)
            frontier_returns.append(r_p)
            frontier_vols.append(v_p)
    fig_frontier = plot_efficient_frontier(
        frontier_returns, frontier_vols,
        ann_ret.tolist(), ann_vol.tolist(), tickets,
        current_port_return=port_ret, current_port_vol=port_vol,
        title="Frontera Eficiente - Media-Varianza",
    )
    fig_frontier.show()
else:
    print("No se generan gráficos: la optimización no fue exitosa.")