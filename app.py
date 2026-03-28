"""
Portfolio Optimizer - Streamlit Web App

Aplicación interactiva para optimización multi-asset con:
- Risk by Asset Class
- Asset Class Constraints
- Rebalancing Strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.portopt import (
    AssetClass,
    AssetClassRegistry,
    plot_price_evolution,
    plot_asset_weights,
    plot_class_allocation,
    plot_efficient_frontier,
    create_rebalancing_table
)
from src.portopt.returns import HistoricalReturns, CAPMReturns, ReturnsConsolidator
from src.portopt.risk import SampleCovariance, RiskConsolidator
from src.portopt.optimization.formulations import MeanVarianceOptimization
from src.portopt.optimization.constraints import (
    BudgetConstraint,
    LongOnlyConstraint,
    AssetClassConstraint
)
from src.portopt.optimization.solvers import CVXPYSolver
from src.portopt.optimization import (
    RebalancingStrategy,
    RebalancingTrigger
)


def simulate_gbm(tickers, base_prices, annual_returns, annual_vols, n_days, corr_matrix):
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


# Configuración
st.set_page_config(page_title="Portfolio Optimizer", layout="wide", page_icon="📈")

st.title("📈 Portfolio Optimizer: Advanced Multi-Asset Framework")
st.markdown("""
Esta aplicación demuestra las **capacidades avanzadas** del optimizador:
- 🎯 **Risk by Asset Class**: Modelos de riesgo específicos por clase
- 🛡️ **Robust Constraints**: Límites a nivel de clase de activo
- 🔄 **Rebalancing Strategies**: Estrategias inteligentes de rebalanceo
""")

# Sidebar
st.sidebar.header("⚙️ Configuración")

n_days = st.sidebar.slider("Días de Simulación", 100, 1000, 500)
risk_aversion = st.sidebar.slider("Aversión al Riesgo (λ)", 0.1, 5.0, 2.0)

st.sidebar.subheader("Límites por Clase de Activo")
eq_min = st.sidebar.slider("Equities Min", 0.0, 0.5, 0.2, 0.05)
eq_max = st.sidebar.slider("Equities Max", eq_min, 1.0, 0.5, 0.05)
fi_min = st.sidebar.slider("Fixed Income Min", 0.0, 0.5, 0.3, 0.05)
fi_max = st.sidebar.slider("Fixed Income Max", fi_min, 1.0, 0.6, 0.05)
comm_min = st.sidebar.slider("Commodities Min", 0.0, 0.3, 0.05, 0.05)
comm_max = st.sidebar.slider("Commodities Max", comm_min, 0.5, 0.25, 0.05)

st.sidebar.subheader("Rebalancing Strategy")
rebal_threshold = st.sidebar.slider("Threshold (%)", 1.0, 10.0, 5.0) / 100
transaction_cost = st.sidebar.slider("Transaction Cost (bps)", 5, 50, 10) / 10000

with st.expander("📖 ¿Cómo funciona?"):
    st.info("""
    **Proceso de Optimización Avanzada:**

    1. **Simulación Realista**:
       - Equities/Commodities: Geometric Brownian Motion
       - Fixed Income: Yield Dynamics con reversión a la media

    2. **Risk by Asset Class**:
       - Equities: Sample Covariance
       - Fixed Income: Sample Covariance
       - Commodities: Sample Covariance

    3. **Asset Class Constraints**:
       - Límites robustos a nivel de clase
       - Garantiza diversificación estructural

    4. **Rebalancing Strategies**:
       - Threshold-based: Rebalancea cuando drift > umbral
       - Calendar-based: Rebalancea en fechas fijas
       - Hybrid: Combinación inteligente
    """)

if st.button("▶️ Ejecutar Optimización Completa", type="primary"):

    with st.spinner("Ejecutando simulación y optimización..."):

        np.random.seed(42)
        dates = pd.date_range('2021-01-01', periods=n_days, freq='D')

        eq_comm_tickers = ['AAPL', 'MSFT', 'GOOGL', 'GOLD', 'OIL']
        df_eq_comm = simulate_gbm(
            eq_comm_tickers, [150, 250, 2000, 1800, 75],
            [0.12, 0.10, 0.08, 0.05, 0.07],
            [0.25, 0.22, 0.20, 0.15, 0.40],
            n_days,
            np.array([[1.0, 0.7, 0.6, 0.1, 0.2], [0.7, 1.0, 0.65, 0.1, 0.2],
                      [0.6, 0.65, 1.0, 0.15, 0.1], [0.1, 0.1, 0.15, 1.0, 0.3],
                      [0.2, 0.2, 0.1, 0.3, 1.0]])
        )

        df_bonds = simulate_bonds(['BOND_10Y', 'BOND_30Y'], [10, 30], [0.035, 0.042], 0.005, n_days)

        data = pd.concat([df_eq_comm, df_bonds], axis=1)
        data.index = dates
        assets = list(data.columns)

        registry = AssetClassRegistry()
        registry.register_batch({
            'AAPL': AssetClass.EQUITIES, 'MSFT': AssetClass.EQUITIES, 'GOOGL': AssetClass.EQUITIES,
            'BOND_10Y': AssetClass.FIXED_INCOME, 'BOND_30Y': AssetClass.FIXED_INCOME,
            'GOLD': AssetClass.COMMODITIES, 'OIL': AssetClass.COMMODITIES
        })

        risk_consolidator = RiskConsolidator(registry=registry)
        risk_consolidator.add_asset_class_model(AssetClass.EQUITIES, SampleCovariance(annualize=True))
        risk_consolidator.add_asset_class_model(AssetClass.FIXED_INCOME, SampleCovariance(annualize=True))
        risk_consolidator.add_asset_class_model(AssetClass.COMMODITIES, SampleCovariance(annualize=True))

        cov_matrix = risk_consolidator.consolidate(data)

        returns_consolidator = ReturnsConsolidator(registry=registry)
        returns_consolidator.add_asset_class_model(AssetClass.EQUITIES, CAPMReturns(risk_free_rate=0.03))
        returns_consolidator.add_asset_class_model(AssetClass.FIXED_INCOME, HistoricalReturns(method='median'))
        returns_consolidator.add_asset_class_model(AssetClass.COMMODITIES, HistoricalReturns(method='mean'))

        expected_returns = returns_consolidator.consolidate(data)

        class_limits = {
            AssetClass.EQUITIES: (eq_min, eq_max),
            AssetClass.FIXED_INCOME: (fi_min, fi_max),
            AssetClass.COMMODITIES: (comm_min, comm_max)
        }

        problem = MeanVarianceOptimization(n_assets=len(assets), risk_aversion=risk_aversion)
        problem.expected_returns = expected_returns
        problem.covariance_matrix = cov_matrix
        problem.add_constraint(BudgetConstraint(budget_target=1.0))
        problem.add_constraint(LongOnlyConstraint())
        problem.add_constraint(AssetClassConstraint(registry, class_limits, assets))

        solver = CVXPYSolver(solver='OSQP')
        result = solver.optimize(problem)

    if result['status'] == 'success':
        st.success("✅ Optimización completada exitosamente!")

        weights_dict = {assets[i]: result['weights'][i] for i in range(len(assets))}

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Allocation", "📈 Prices", "🏁 Frontier", "🔄 Rebalancing"])

        class_weights = registry.get_class_weights(weights_dict)
        asset_classes_dict = {
            asset: registry.get_asset_class(asset).value
            for asset in assets
        }

        with tab1:
            st.subheader("Asignación de Portafolio")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Por Activo Individual**")
                fig_assets = plot_asset_weights(weights_dict, asset_classes_dict)
                st.plotly_chart(fig_assets, use_container_width=True)

            with col2:
                st.write("**Por Clase de Activo**")
                fig_class = plot_class_allocation(class_weights)
                st.plotly_chart(fig_class, use_container_width=True)

            st.write("**Verificación de Límites por Clase**")
            verification_data = []
            for ac, w in class_weights.items():
                mn, mx = class_limits.get(ac, (0, 1))
                status = "[OK]" if mn <= w <= mx else "[X]"
                verification_data.append({
                    'Status': status,
                    'Asset Class': ac.value,
                    'Weight': f"{w:.2%}",
                    'Limit': f"{mn:.0%} - {mx:.0%}",
                    'Check': 'OK' if mn <= w <= mx else 'VIOLATED'
                })
            st.dataframe(pd.DataFrame(verification_data), use_container_width=True, hide_index=True)

            port_return = np.dot(expected_returns, result['weights']) * 252
            port_vol = np.sqrt(result['weights'].T @ cov_matrix @ result['weights']) * np.sqrt(252)
            sharpe = (port_return - 0.03) / port_vol

            m1, m2, m3 = st.columns(3)
            m1.metric("Expected Return", f"{port_return:.2%}")
            m2.metric("Volatility", f"{port_vol:.2%}")
            m3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        with tab2:
            st.subheader("Evolución de Precios Simulados")
            fig_prices = plot_price_evolution(data)
            st.plotly_chart(fig_prices, use_container_width=True)

            st.subheader("Estadísticas por Activo (Anualizadas)")
            asset_stats = []
            asset_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
            asset_rets = expected_returns * 252

            for i, asset in enumerate(assets):
                asset_stats.append({
                    "Activo": asset,
                    "Retorno Esperado": f"{asset_rets[i]:.2%}",
                    "Riesgo (Volatilidad)": f"{asset_vols[i]:.2%}",
                    "Sharpe Ratio": f"{(asset_rets[i]-0.03)/asset_vols[i]:.2f}"
                })

            st.table(pd.DataFrame(asset_stats))

        with tab3:
            st.subheader("Frontera Eficiente")

            with st.spinner("Calculando frontera..."):
                lambdas = np.logspace(-1, 2, 15)
                frontier_returns, frontier_vols = [], []

                for l in lambdas:
                    problem.risk_aversion = l
                    res = solver.optimize(problem)
                    if res['status'] == 'success':
                        w = res['weights']
                        r_p = np.dot(expected_returns, w) * 252
                        v_p = np.sqrt(w.T @ cov_matrix @ w) * np.sqrt(252)
                        frontier_returns.append(r_p)
                        frontier_vols.append(v_p)

                asset_vols = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
                asset_rets = expected_returns * 252

                fig = plot_efficient_frontier(
                    frontier_returns, frontier_vols,
                    asset_rets, asset_vols, assets,
                    current_port_return=port_return,
                    current_port_vol=port_vol
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Análisis de Rebalanceo")

            target_weights = result['weights'].copy()
            current_weights = target_weights + np.random.randn(len(assets)) * 0.04
            current_weights = np.clip(current_weights, 0, 1)
            current_weights /= current_weights.sum()

            max_dev = np.max(np.abs(current_weights - target_weights))

            st.write(f"**Desviación Simulada**: {max_dev:.2%}")

            threshold_strategy = RebalancingStrategy(
                trigger=RebalancingTrigger.THRESHOLD,
                threshold=rebal_threshold,
                transaction_cost=transaction_cost
            )

            should_rebal = threshold_strategy.should_rebalance(
                current_weights, target_weights, datetime.now()
            )

            if should_rebal:
                st.warning("⚠️ **REBALANCE REQUERIDO**")

                trade_info = threshold_strategy.calculate_trades(
                    current_weights, target_weights, portfolio_value=1_000_000
                )

                col1, col2, col3 = st.columns(3)
                col1.metric("Turnover", f"{trade_info['turnover']:.2%}")
                col2.metric("Transaction Costs", f"${trade_info['transaction_costs']:,.0f}")
                col3.metric("Net Impact", f"{trade_info['net_impact']:.4%}")

                trade_details = []
                for i, asset in enumerate(assets):
                    delta = current_weights[i] - target_weights[i]
                    if abs(delta) > 0.001:
                        trade_details.append({
                            'Asset': asset,
                            'Current': f"{current_weights[i]:.2%}",
                            'Target': f"{target_weights[i]:.2%}",
                            'Action': 'BUY' if delta < 0 else 'SELL',
                            'Amount': f"{abs(delta):.2%}"
                        })

                df_trades = create_rebalancing_table(trade_details)
                st.dataframe(df_trades, use_container_width=True, hide_index=True)
            else:
                st.success("✅ **NO REBALANCE NECESARIO** - Portafolio dentro de límites")

    else:
        st.error(f"❌ Optimización falló: {result['status']}")

else:
    st.info("👈 Configure los parámetros y presione el botón para comenzar")

st.divider()
st.caption("Portfolio Optimizer - Advanced Multi-Asset Class Framework")
