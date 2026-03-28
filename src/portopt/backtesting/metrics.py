"""
Performance Metrics - Calculate portfolio performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Optional


class PerformanceMetrics:
    """
    Clase para calcular métricas de rendimiento de portafolios.
    
    Incluye Sharpe, Sortino, Max Drawdown, Calmar, etc.
    """
    
    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calcula el ratio de Sharpe.
        
        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo anualizada
            periods_per_year: Períodos por año (252 para diario)
        
        Returns:
            Ratio de Sharpe
        """
        excess_returns = returns - risk_free_rate / periods_per_year
        
        if excess_returns.std() == 0:
            return 0.0
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> float:
        """
        Calcula el ratio de Sortino.
        
        Similar a Sharpe pero solo considera downside volatility.
        
        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo
            periods_per_year: Períodos por año
        
        Returns:
            Ratio de Sortino
        """
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
    
    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """
        Calcula el máximo drawdown.
        
        Args:
            returns: Serie de retornos
        
        Returns:
            Máximo drawdown (negativo)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calcula el ratio de Calmar.
        
        Retorno anualizado / Max Drawdown absoluto
        
        Args:
            returns: Serie de retornos
            periods_per_year: Períodos por año
        
        Returns:
            Ratio de Calmar
        """
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1
        max_dd = abs(PerformanceMetrics.max_drawdown(returns))
        
        if max_dd == 0:
            return 0.0
        
        return annualized_return / max_dd
    
    @staticmethod
    def annualized_return(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calcula el retorno anualizado.
        
        Args:
            returns: Serie de retornos
            periods_per_year: Períodos por año
        
        Returns:
            Retorno anualizado
        """
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        
        if n_periods == 0:
            return 0.0
        
        return (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    @staticmethod
    def annualized_volatility(
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calcula la volatilidad anualizada.
        
        Args:
            returns: Serie de retornos
            periods_per_year: Períodos por año
        
        Returns:
            Volatilidad anualizada
        """
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def get_all_metrics(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252
    ) -> dict:
        """
        Calcula todas las métricas.
        
        Args:
            returns: Serie de retornos
            risk_free_rate: Tasa libre de riesgo
            periods_per_year: Períodos por año
        
        Returns:
            Diccionario con todas las métricas
        """
        return {
            "annualized_return": PerformanceMetrics.annualized_return(
                returns, periods_per_year
            ),
            "annualized_volatility": PerformanceMetrics.annualized_volatility(
                returns, periods_per_year
            ),
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(
                returns, risk_free_rate, periods_per_year
            ),
            "sortino_ratio": PerformanceMetrics.sortino_ratio(
                returns, risk_free_rate, periods_per_year
            ),
            "max_drawdown": PerformanceMetrics.max_drawdown(returns),
            "calmar_ratio": PerformanceMetrics.calmar_ratio(
                returns, periods_per_year
            ),
        }
