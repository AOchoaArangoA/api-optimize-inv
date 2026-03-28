import os
import numpy as np
from src.portopt.returns.data_downloader import AlpacaDataDownloader, MassiveDataDownloader
from src.portopt.returns.historical import HistoricalReturns
from src.portopt.risk.sample_cov import SampleCovariance
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Configuración de claves (Valores por defecto tomados del notebook para el prototipo)
ALPACA_KEY = os.getenv("ALPACA_API_KEY", "PKCWXT3QJF56JT6EXGKTFRR5GZ")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "Dgv2L3wTP8GakAY1AFkF16KTJBcg7rs6yoExKycD1cAr")
MASSIVE_KEY = os.getenv("MASSIVE_API_KEY", "*")

# 1. Prueba Alpaca
print("\n--- Probando Alpaca Downloader ---")
alpaca = AlpacaDataDownloader(ALPACA_KEY, ALPACA_SECRET)

tickets = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", 'NU','CRM']

try:
    print(f"\nDescargando Stocks {tickets}...")
    stock_data = alpaca.fetch_stock_data(tickets, years=1)
    print("\nSe realiza la descarga de manera efectiva")
except Exception as e:
    print(f"Error en Alpaca: {e}")

# 2. Prueba Proceso de retornos históricos

mean_returns, returns = HistoricalReturns(method="mean").estimate(stock_data[['ticker', 'close']], window=252, anual=True)
print(mean_returns)
print(returns)

riesgo = SampleCovariance(returns)

# Preparar variables para optimización
assets = riesgo.columns.tolist()
mu = mean_returns.values # vector de retornos esperados anuales
Sigma = riesgo.values # matriz de covarianza anual
n = len(assets)

# Rango de retornos objetivo (entre mínimo y máximo observado)
target_returns = np.linspace(mu.min(), mu.max(), 50)
frontier_weights = []
frontier_risks = []
frontier_returns = []

# Función objetivo: varianza
def portfolio_variance(w, Sigma):
    return float(w.T @ Sigma @ w)

# Restricción suma pesos = 1
cons_sum = {'type':'eq', 'fun': lambda w: np.sum(w)- 1}
bounds = [(0,1)] * n # no short selling

for r_t in target_returns:
    cons_return = {'type':'eq', 'fun': lambda w, r=r_t: w @ mu- r}
    cons = (cons_sum, cons_return)
    w0 = np.ones(n) / n
    res = minimize(lambda w: portfolio_variance(w, Sigma), w0, method='SLSQP',
    bounds=bounds, constraints=cons)
    
    if res.success:
        w_opt = res.x
        frontier_weights.append(w_opt)
        r = mu @ w_opt
        sigma = np.sqrt(w_opt.T @ Sigma @ w_opt)
        frontier_returns.append(r)
        frontier_risks.append(sigma)

plt.figure(figsize=(8,6))
plt.plot(frontier_risks, frontier_returns, 'bo-', label='Frontera eficiente')
plt.xlabel('Riesgo (Desviación estándar)')
plt.ylabel('Retorno esperado')
plt.title('Frontera eficiente')
plt.grid(True)
plt.legend()
plt.show()
print("\nSe realiza el proceso de retornos históricos de manera efectiva")