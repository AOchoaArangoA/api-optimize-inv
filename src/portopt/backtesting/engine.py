"""
Backtesting Engine - Test portfolio strategies on historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional
from datetime import datetime

from ..core.portfolio import Portfolio


class BacktestEngine:
    """
    Motor de backtesting para estrategias de portafolio.
    
    Simula la evolución del portafolio en datos históricos,
    rebalanceando según una estrategia definida.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        rebalance_frequency: str = "monthly",
        transaction_cost: float = 0.001
    ):
        """
        Inicializa el motor de backtesting.
        
        Args:
            data: DataFrame con precios históricos (columnas=activos, index=fechas)
            rebalance_frequency: Frecuencia de rebalanceo ('daily', 'weekly', 'monthly')
            transaction_cost: Costo de transacción como fracción
        """
        self.data = data
        self.rebalance_frequency = rebalance_frequency
        self.transaction_cost = transaction_cost
        self.results_ = None
    
    def run(
        self,
        strategy: Callable,
        initial_capital: float = 100000
    ) -> Dict:
        """
        Ejecuta el backtest.
        
        Args:
            strategy: Función que recibe datos históricos y retorna pesos
            initial_capital: Capital inicial
        
        Returns:
            Diccionario con resultados del backtest
        """
        # Calcular retornos
        returns = self.data.pct_change().dropna()
        
        # Determinar fechas de rebalanceo
        rebalance_dates = self._get_rebalance_dates(returns.index)
        
        # Inicializar tracking
        portfolio_values = []
        portfolio_weights_history = []
        dates = []
        
        current_weights = None
        current_value = initial_capital
        
        for i, date in enumerate(returns.index):
            # Rebalancear si corresponde
            if date in rebalance_dates:
                # Obtener nuevos pesos de la estrategia
                lookback_data = self.data.loc[:date]
                new_weights = strategy(lookback_data)
                
                # Calcular costos de transacción
                if current_weights is not None:
                    turnover = np.abs(new_weights - current_weights).sum()
                    transaction_costs = current_value * turnover * self.transaction_cost
                    current_value -= transaction_costs
                
                current_weights = new_weights
                portfolio_weights_history.append(new_weights)
            
            # Calcular retorno del portafolio
            if current_weights is not None:
                portfolio_return = np.dot(current_weights, returns.loc[date].values)
                current_value *= (1 + portfolio_return)
            
            # Guardar valores
            portfolio_values.append(current_value)
            dates.append(date)
        
        # Calcular métricas
        portfolio_values = np.array(portfolio_values)
        portfolio_returns = pd.Series(
            portfolio_values,
            index=dates
        ).pct_change().dropna()
        
        self.results_ = {
            "portfolio_values": portfolio_values,
            "portfolio_returns": portfolio_returns,
            "dates": dates,
            "weights_history": portfolio_weights_history,
            "rebalance_dates": rebalance_dates,
            "final_value": current_value,
            "total_return": (current_value - initial_capital) / initial_capital
        }
        
        return self.results_
    
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> List:
        """Determina las fechas de rebalanceo."""
        if self.rebalance_frequency == "daily":
            return dates.tolist()
        
        elif self.rebalance_frequency == "weekly":
            return dates[dates.weekday == 0].tolist()
        
        elif self.rebalance_frequency == "monthly":
            return dates[dates.is_month_end].tolist()
        
        else:
            raise ValueError(f"Unknown frequency: {self.rebalance_frequency}")
    
    def get_results(self) -> Optional[Dict]:
        """Retorna los resultados del backtest."""
        return self.results_
