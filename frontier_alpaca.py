"""
Demo: Frontera eficiente con datos reales de Alpaca.

Tickers: universo del notebook 01-prototipo-descarga-api.ipynb
Flujo:
  1. Descarga precios históricos desde Alpaca (1 año).
  2. Construye la frontera eficiente (20 puntos).
  3. Imprime tabla: volatilidad | retorno | sharpe | pesos top-3.
  4. Marca el portafolio de mínima varianza y el de máximo Sharpe.

Uso:
  export ALPACA_API_KEY="tu_key"
  export ALPACA_SECRET_KEY="tu_secret"
  python frontier_alpaca.py

  # O pasando las claves directo:
  python frontier_alpaca.py --key TU_KEY --secret TU_SECRET
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np

from portopt.returns.data_downloader import AlpacaDataDownloader
from portopt.optimization.formulations import MeanVarianceOptimization
from portopt.optimization.constraints import BudgetConstraint, LongOnlyConstraint
from portopt.optimization.solvers import CVXPYSolver
from portopt.returns import HistoricalReturns
from portopt.risk import SampleCovariance
from portopt.pipeline.orchestrator import PortfolioOrchestrator

# ── Universo del notebook ──────────────────────────────────────────────────────
TICKERS = [
    "LRCX", "GOOGL", "GOOG", "ASML", "GLW",
    "JPM",  "XOM",   "KO",   "MCD",  "IBM",
    "CAT",  "BA",    "DIS",  "PG",   "PYPL",
    "F",    "NU",
]

YEARS       = 1
N_POINTS    = 20
LAMBDA_MIN  = 0.5
LAMBDA_MAX  = 10.0
RISK_FREE   = 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Frontera eficiente vía Alpaca")
    p.add_argument("--key",    default=os.getenv("ALPACA_API_KEY"),    help="Alpaca API Key")
    p.add_argument("--secret", default=os.getenv("ALPACA_SECRET_KEY"), help="Alpaca Secret Key")
    p.add_argument("--years",  type=int, default=YEARS)
    p.add_argument("--points", type=int, default=N_POINTS)
    return p.parse_args()


def download_prices(api_key: str, secret_key: str, tickers: list, years: int):
    """Descarga y pivota precios desde Alpaca → DataFrame wide."""
    print(f"  Descargando {len(tickers)} tickers ({years} año(s)) desde Alpaca...")
    dl = AlpacaDataDownloader(api_key, secret_key)
    raw = dl.fetch_stock_data(tickers, years=years)

    # Pivotar a wide: columnas = tickers, filas = fechas
    prices = raw.pivot_table(index=raw.index, columns="ticker", values="close")

    # Conservar solo tickers con datos completos
    available = [t for t in tickers if t in prices.columns]
    prices = prices[available].dropna()

    dropped = set(tickers) - set(available)
    if dropped:
        print(f"  Tickers sin datos descartados: {sorted(dropped)}")

    print(f"  Datos: {prices.shape[0]} observaciones × {prices.shape[1]} activos")
    return prices


def compute_frontier(prices_df, n_points: int, lambda_min: float, lambda_max: float):
    """
    Barre risk_aversion de lambda_min a lambda_max con n_points pasos.
    Retorna lista de dicts con métricas por punto.
    """
    tickers  = list(prices_df.columns)
    n_assets = len(tickers)
    lambdas  = np.linspace(lambda_min, lambda_max, n_points)
    results  = []

    for i, lam in enumerate(lambdas, 1):
        problem = MeanVarianceOptimization(n_assets=n_assets, risk_aversion=float(lam))
        problem.add_constraint(BudgetConstraint())
        problem.add_constraint(LongOnlyConstraint())

        orch = PortfolioOrchestrator()
        result = orch.run(
            data=prices_df,
            returns_model=HistoricalReturns(),
            risk_model=SampleCovariance(annualize=True),
            problem=problem,
            solver=CVXPYSolver(),
        )

        if result.success and result.weights is not None:
            w   = result.weights
            mu  = orch.expected_returns_
            cov = orch.cov_matrix_
            ret = float(mu @ w)
            vol = float(np.sqrt(w @ cov @ w))
            sharpe = (ret - RISK_FREE) / vol if vol > 1e-8 else None
            results.append({
                "lambda":  float(lam),
                "return":  ret,
                "vol":     vol,
                "sharpe":  sharpe,
                "weights": dict(zip(tickers, w.tolist())),
                "status":  "optimal",
            })
        else:
            results.append({
                "lambda": float(lam),
                "status": result.status,
            })

        print(f"  [{i:02d}/{n_points}] λ={lam:.2f}  "
              f"ret={results[-1].get('return', 0):.2%}  "
              f"vol={results[-1].get('vol', 0):.2%}  "
              f"sharpe={results[-1].get('sharpe') or 0:.2f}")

    return results


def print_summary(frontier: list):
    """Imprime tabla resumen y destaca min-varianza y max-Sharpe."""
    optimal = [p for p in frontier if p["status"] == "optimal"]
    if not optimal:
        print("No se encontraron portafolios óptimos.")
        return

    # Ordenar por volatilidad
    optimal.sort(key=lambda p: p["vol"])

    min_var   = optimal[0]
    max_sharpe = max(optimal, key=lambda p: p["sharpe"] or -999)

    print("\n" + "═" * 80)
    print(f"{'VOL':>8}  {'RETORNO':>8}  {'SHARPE':>7}  {'TOP-3 PESOS'}")
    print("─" * 80)

    for p in optimal:
        top3 = sorted(p["weights"].items(), key=lambda x: -x[1])[:3]
        top3_str = "  ".join(f"{t}:{w:.1%}" for t, w in top3)
        tag = ""
        if p is min_var:
            tag = " ◄ MIN VAR"
        elif p is max_sharpe:
            tag = " ◄ MAX SHARPE"
        print(f"{p['vol']:>8.2%}  {p['return']:>8.2%}  {p['sharpe']:>7.2f}  {top3_str}{tag}")

    print("═" * 80)
    print(f"\nPortafolio mínima varianza:")
    print(f"  Retorno: {min_var['return']:.2%}  |  Vol: {min_var['vol']:.2%}  |  Sharpe: {min_var['sharpe']:.2f}")
    print(f"  Pesos: { {k: f'{v:.1%}' for k, v in sorted(min_var['weights'].items(), key=lambda x: -x[1])[:5]} }")

    print(f"\nPortafolio máximo Sharpe:")
    print(f"  Retorno: {max_sharpe['return']:.2%}  |  Vol: {max_sharpe['vol']:.2%}  |  Sharpe: {max_sharpe['sharpe']:.2f}")
    print(f"  Pesos: { {k: f'{v:.1%}' for k, v in sorted(max_sharpe['weights'].items(), key=lambda x: -x[1])[:5]} }")


def main():
    args = parse_args()

    if not args.key or not args.secret:
        print("Error: credenciales Alpaca no encontradas.")
        print("  Usa --key / --secret o las variables ALPACA_API_KEY / ALPACA_SECRET_KEY")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  Frontera Eficiente — {len(TICKERS)} activos vía Alpaca")
    print(f"  {args.points} puntos  |  λ ∈ [{LAMBDA_MIN}, {LAMBDA_MAX}]")
    print(f"{'─'*60}")

    # 1. Descargar precios
    prices_df = download_prices(args.key, args.secret, TICKERS, args.years)

    # 2. Calcular frontera
    print(f"\nCalculando {args.points} puntos de la frontera...")
    frontier = compute_frontier(prices_df, args.points, LAMBDA_MIN, LAMBDA_MAX)

    # 3. Resumen
    print_summary(frontier)


if __name__ == "__main__":
    main()
