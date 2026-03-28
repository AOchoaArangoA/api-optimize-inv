"""
Portfolio Optimizer - Main Entry Point

Demonstración completa del framework con:
- Risk by Asset Class (RiskConsolidator)
- Asset Class Constraints (Robust Constraints)
- Rebalancing Strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.portopt import AssetClass, AssetClassRegistry
from src.portopt.returns import HistoricalReturns, CAPMReturns, ReturnsConsolidator
from src.portopt.risk import SampleCovariance, GARCHModel, RiskConsolidator
from src.portopt.optimization.formulations import MeanVarianceOptimization
from src.portopt.optimization.constraints import (
    BudgetConstraint,
    LongOnlyConstraint,
    AssetClassConstraint
)
from src.portopt.optimization.solvers import CVXPYSolver
from src.portopt.optimization import (
    RebalancingStrategy,
    AdaptiveRebalancingStrategy,
    RebalancingTrigger
)


def simulate_gbm(tickers, base_prices, annual_returns, annual_vols, n_days, corr_matrix):
    """Simula precios usando Movimiento Browniano Geométrico Correlacionado."""
    dt = 1/252
    n_assets = len(tickers)
    mu = np.array(annual_returns) * dt
    sigma = np.array(annual_vols) * np.sqrt(dt)
    L = np.linalg.cholesky(corr_matrix)
    random_shocks = np.random.randn(n_days, n_assets)
    correlated_shocks = random_shocks @ L.T
    log_returns = (mu - 0.5 * sigma**2) + sigma * correlated_shocks
    price_paths = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        price_paths[:, i] = base_prices[i] * np.exp(np.cumsum(log_returns[:, i]))
    return pd.DataFrame(price_paths, columns=tickers)


def simulate_bonds(tickers, maturities, initial_yields, yield_vol, n_days):
    """Simula precios de bonos a partir de una evolución de curva de rendimientos."""
    dt = 1/252
    n_assets = len(tickers)
    prices = np.zeros((n_days, n_assets))
    current_yields = np.array(initial_yields)
    for t in range(n_days):
        drift = 0.01 * (initial_yields - current_yields) * dt
        shock = yield_vol * np.sqrt(dt) * np.random.randn(n_assets)
        current_yields += drift + shock
        for i in range(n_assets):
            prices[t, i] = 100 / (1 + current_yields[i])**maturities[i]
    return pd.DataFrame(prices, columns=tickers)


def main():
    """
    Función principal - Demostración completa de características avanzadas.
    """
    print("=" * 80)
    print("  PORTFOLIO OPTIMIZER - ADVANCED MULTI-ASSET CLASS DEMONSTRATION")
    print("=" * 80)

    # 1. SIMULACIÓN DE DATOS REALISTA
    print("\n[Step 1/5] Generating realistic market data...")
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range('2021-01-01', periods=n_days, freq='D')

    # Equities & Commodities
    eq_comm_tickers = ['AAPL', 'MSFT', 'GOOGL', 'GOLD', 'OIL']
    df_eq_comm = simulate_gbm(
        eq_comm_tickers, [150, 250, 2000, 1800, 75],
        [0.12, 0.10, 0.08, 0.05, 0.07],
        [0.25, 0.22, 0.20, 0.15, 0.40],
        n_days,
        np.array([[1.0, 0.7, 0.6, 0.1, 0.2],
                  [0.7, 1.0, 0.65, 0.1, 0.2],
                  [0.6, 0.65, 1.0, 0.15, 0.1],
                  [0.1, 0.1, 0.15, 1.0, 0.3],
                  [0.2, 0.2, 0.1, 0.3, 1.0]])
    )

    # Fixed Income
    df_bonds = simulate_bonds(['BOND_10Y', 'BOND_30Y'], [10, 30], [0.035, 0.042], 0.005, n_days)

    data = pd.concat([df_eq_comm, df_bonds], axis=1)
    data.index = dates
    assets = list(data.columns)
    print(f"  [OK] Simulated {len(assets)} assets over {n_days} days")

    # 2. CLASIFICACIÓN DE ACTIVOS
    print("\n[Step 2/5] Classifying assets by class...")
    registry = AssetClassRegistry()
    registry.register_batch({
        'AAPL': AssetClass.EQUITIES, 'MSFT': AssetClass.EQUITIES, 'GOOGL': AssetClass.EQUITIES,
        'BOND_10Y': AssetClass.FIXED_INCOME, 'BOND_30Y': AssetClass.FIXED_INCOME,
        'GOLD': AssetClass.COMMODITIES, 'OIL': AssetClass.COMMODITIES
    })

    for ac_name, count in registry.get_summary().items():
        print(f"  - {ac_name}: {count} assets")

    # 3. RISK BY ASSET CLASS
    print("\n[Step 3/5] Consolidating risk by asset class...")
    risk_consolidator = RiskConsolidator(registry=registry)

    # Modelos específicos por clase
    risk_consolidator.add_asset_class_model(AssetClass.EQUITIES, SampleCovariance(annualize=True))
    risk_consolidator.add_asset_class_model(AssetClass.FIXED_INCOME, SampleCovariance(annualize=True))
    risk_consolidator.add_asset_class_model(AssetClass.COMMODITIES, GARCHModel())

    print("  - Equities: Sample Covariance")
    print("  - Fixed Income: Sample Covariance")
    print("  - Commodities: GARCH Model")

    cov_matrix = risk_consolidator.consolidate(data)
    avg_vol = np.sqrt(np.diag(cov_matrix)).mean()
    print(f"  [OK] Average volatility: {avg_vol:.4f}")

    # Retornos consolidados
    returns_consolidator = ReturnsConsolidator(registry=registry)
    returns_consolidator.add_asset_class_model(AssetClass.EQUITIES, HistoricalReturns(method='mean'))
    returns_consolidator.add_asset_class_model(AssetClass.FIXED_INCOME, HistoricalReturns(method='median'))
    returns_consolidator.add_asset_class_model(AssetClass.COMMODITIES, HistoricalReturns(method='mean'))

    expected_returns = returns_consolidator.consolidate(data)

    # 4. OPTIMIZACIÓN CON ASSET CLASS CONSTRAINTS
    print("\n[Step 4/5] Optimizing with robust asset class constraints...")

    class_limits = {
        AssetClass.EQUITIES: (0.20, 0.50),
        AssetClass.FIXED_INCOME: (0.30, 0.60),
        AssetClass.COMMODITIES: (0.05, 0.25)
    }

    print("  Asset Class Limits:")
    for ac, (mn, mx) in class_limits.items():
        print(f"    {ac.value:15s}: {mn:.0%} - {mx:.0%}")

    problem = MeanVarianceOptimization(n_assets=len(assets), risk_aversion=2.0)
    problem.expected_returns = expected_returns
    problem.covariance_matrix = cov_matrix
    problem.add_constraint(BudgetConstraint(budget_target=1.0))
    problem.add_constraint(LongOnlyConstraint())
    problem.add_constraint(AssetClassConstraint(registry, class_limits, assets))

    solver = CVXPYSolver(solver='OSQP')
    result = solver.optimize(problem)

    if result['status'] == 'success':
        print("\n  [OK] Optimization successful!")

        weights_dict = {assets[i]: result['weights'][i] for i in range(len(assets))}

        # Mostrar asignación por activo
        print("\n  Portfolio Weights by Asset:")
        for asset, weight in weights_dict.items():
            ac = registry.get_asset_class(asset)
            print(f"    {asset:10s} ({ac.value:12s}): {weight*100:6.2f}%")

        # Verificar límites por clase
        print("\n  Asset Class Allocation (with limits check):")
        class_weights = registry.get_class_weights(weights_dict)
        for ac, w in class_weights.items():
            mn, mx = class_limits.get(ac, (0, 1))
            status = "[OK]" if mn <= w <= mx else "[X]"
            print(f"    {status} {ac.value:15s}: {w*100:6.2f}% [Limit: {mn:.0%}-{mx:.0%}]")

        # Métricas del portafolio
        port_return = np.dot(expected_returns, result['weights']) * 252
        port_vol = np.sqrt(result['weights'].T @ cov_matrix @ result['weights']) * np.sqrt(252)
        sharpe = (port_return - 0.03) / port_vol

        print("\n  Portfolio Performance Metrics:")
        print(f"    Expected Return (annual): {port_return:.2%}")
        print(f"    Volatility (annual):      {port_vol:.2%}")
        print(f"    Sharpe Ratio:             {sharpe:.2f}")

    # 5. REBALANCING STRATEGIES
    print("\n[Step 5/5] Testing rebalancing strategies...")

    # Simular drift en pesos
    target_weights = result['weights'].copy()
    current_weights = target_weights + np.random.randn(len(assets)) * 0.04
    current_weights = np.clip(current_weights, 0, 1)
    current_weights /= current_weights.sum()

    max_deviation = np.max(np.abs(current_weights - target_weights))
    print(f"\n  Simulated drift: max deviation = {max_deviation:.2%}")

    # Strategy 1: Threshold
    threshold_strategy = RebalancingStrategy(
        trigger=RebalancingTrigger.THRESHOLD,
        threshold=0.05,
        transaction_cost=0.001
    )

    # Strategy 2: Calendar
    calendar_strategy = RebalancingStrategy(
        trigger=RebalancingTrigger.CALENDAR,
        frequency_days=30,
        transaction_cost=0.001
    )
    calendar_strategy.last_rebalance_date = datetime.now() - timedelta(days=45)

    # Strategy 3: Adaptive
    adaptive_strategy = AdaptiveRebalancingStrategy(
        base_threshold=0.05,
        trigger=RebalancingTrigger.HYBRID,
        frequency_days=90,
        transaction_cost=0.001
    )

    print("\n  Rebalancing Decisions:")

    rebal_threshold = threshold_strategy.should_rebalance(current_weights, target_weights, datetime.now())
    print(f"    Threshold (5%):        {'REBALANCE [OK]' if rebal_threshold else 'HOLD'}")

    rebal_calendar = calendar_strategy.should_rebalance(current_weights, target_weights, datetime.now())
    print(f"    Calendar (30 days):    {'REBALANCE [OK]' if rebal_calendar else 'HOLD'}")

    rebal_adaptive = adaptive_strategy.should_rebalance(current_weights, target_weights, datetime.now())
    print(f"    Adaptive (hybrid):     {'REBALANCE [OK]' if rebal_adaptive else 'HOLD'}")

    # Calcular trades si se rebalancea
    if rebal_threshold:
        trade_info = threshold_strategy.calculate_trades(
            current_weights, target_weights, portfolio_value=1_000_000
        )

        print("\n  Trade Analysis (if rebalancing):")
        print(f"    Portfolio Turnover:     {trade_info['turnover']:.2%}")
        print(f"    Transaction Costs:      ${trade_info['transaction_costs']:,.2f}")
        print(f"    Net Impact on Return:   {trade_info['net_impact']:.4%}")

    print("\n" + "=" * 80)
    print("  DEMONSTRATION COMPLETE - All Advanced Features Working!")
    print("=" * 80)


if __name__ == "__main__":
    main()
