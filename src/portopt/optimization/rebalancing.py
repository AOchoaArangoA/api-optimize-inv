"""
Rebalancing Strategies - Estrategias para rebalanceo de portafolios.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum


class RebalancingTrigger(Enum):
    """Tipos de trigger para rebalanceo."""
    THRESHOLD = "threshold"      # Basado en desviación de pesos
    CALENDAR = "calendar"         # Basado en calendario (mensual, trimestral, etc.)
    HYBRID = "hybrid"            # Combinación de ambos
    VOLATILITY = "volatility"    # Basado en cambios de volatilidad


class RebalancingStrategy:
    """
    Estrategia de rebalanceo de portafolios.
    
    Determina cuándo y cómo rebalancear un portafolio para
    mantener la asignación objetivo.
    """
    
    def __init__(
        self,
        trigger: RebalancingTrigger = RebalancingTrigger.THRESHOLD,
        threshold: float = 0.05,
        frequency_days: Optional[int] = None,
        transaction_cost: float = 0.001
    ):
        """
        Inicializa la estrategia de rebalanceo.
        
        Args:
            trigger: Tipo de trigger a usar
            threshold: Desviación máxima permitida (para THRESHOLD/HYBRID)
            frequency_days: Días entre rebalanceos (para CALENDAR/HYBRID)
            transaction_cost: Costo de transacción (bps)
        """
        self.trigger = trigger
        self.threshold = threshold
        self.frequency_days = frequency_days
        self.transaction_cost = transaction_cost
        self.last_rebalance_date: Optional[datetime] = None
    
    def should_rebalance(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        current_date: datetime,
        volatility: Optional[float] = None
    ) -> bool:
        """
        Determina si se debe rebalancear el portafolio.
        
        Args:
            current_weights: Pesos actuales del portafolio
            target_weights: Pesos objetivo
            current_date: Fecha actual
            volatility: Volatilidad actual del portafolio (opcional)
            
        Returns:
            True si se debe rebalancear
        """
        if self.trigger == RebalancingTrigger.THRESHOLD:
            return self._check_threshold(current_weights, target_weights)
        
        elif self.trigger == RebalancingTrigger.CALENDAR:
            return self._check_calendar(current_date)
        
        elif self.trigger == RebalancingTrigger.HYBRID:
            threshold_check = self._check_threshold(current_weights, target_weights)
            calendar_check = self._check_calendar(current_date)
            return threshold_check or calendar_check
        
        elif self.trigger == RebalancingTrigger.VOLATILITY:
            # Implementación simplificada - rebalancear si vol > umbral
            if volatility is not None and volatility > 0.25:  # 25% anual
                return True
            return False
        
        return False
    
    def _check_threshold(self, current: np.ndarray, target: np.ndarray) -> bool:
        """Verifica si la desviación supera el umbral."""
        max_deviation = np.max(np.abs(current - target))
        return max_deviation > self.threshold
    
    def _check_calendar(self, current_date: datetime) -> bool:
        """Verifica si ha pasado el tiempo necesario desde el último rebalanceo."""
        if self.frequency_days is None:
            return False
        
        if self.last_rebalance_date is None:
            return True
        
        days_since = (current_date - self.last_rebalance_date).days
        return days_since >= self.frequency_days
    
    def calculate_trades(
        self,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        portfolio_value: float
    ) -> Dict[str, float]:
        """
        Calcula las operaciones necesarias para rebalancear.
        
        Args:
            current_weights: Pesos actuales
            target_weights: Pesos objetivo
            portfolio_value: Valor total del portafolio
            
        Returns:
            Diccionario con información de trading
        """
        delta_weights = target_weights - current_weights
        
        # Valor de las operaciones
        trade_values = delta_weights * portfolio_value
        
        # Costos de transacción
        total_turnover = np.sum(np.abs(delta_weights))
        costs = total_turnover * portfolio_value * self.transaction_cost
        
        return {
            "trade_values": trade_values,
            "turnover": total_turnover,
            "transaction_costs": costs,
            "net_impact": -costs / portfolio_value  # Impacto en retorno
        }
    
    def execute_rebalance(self, current_date: datetime) -> None:
        """Marca que se ejecutó un rebalanceo."""
        self.last_rebalance_date = current_date
    
    def get_summary(self) -> Dict[str, any]:
        """Obtiene resumen de la estrategia."""
        return {
            "trigger": self.trigger.value,
            "threshold": self.threshold if self.trigger in [
                RebalancingTrigger.THRESHOLD, 
                RebalancingTrigger.HYBRID
            ] else None,
            "frequency_days": self.frequency_days,
            "transaction_cost": self.transaction_cost,
            "last_rebalance": self.last_rebalance_date
        }
    
    def __repr__(self) -> str:
        """Representación de la estrategia."""
        return (
            f"RebalancingStrategy(trigger={self.trigger.value}, "
            f"threshold={self.threshold:.2%})"
        )


class AdaptiveRebalancingStrategy(RebalancingStrategy):
    """
    Estrategia adaptativa que ajusta umbrales basado en condiciones de mercado.
    """
    
    def __init__(self, base_threshold: float = 0.05, **kwargs):
        """
        Inicializa estrategia adaptativa.
        
        Args:
            base_threshold: Umbral base
            **kwargs: Argumentos adicionales para RebalancingStrategy
        """
        super().__init__(threshold=base_threshold, **kwargs)
        self.base_threshold = base_threshold
    
    def adjust_threshold(self, market_volatility: float) -> None:
        """
        Ajusta el umbral basado en volatilidad del mercado.
        
        En mercados volátiles, aumenta el umbral para evitar
        rebalanceos excesivos.
        
        Args:
            market_volatility: Volatilidad del mercado (anualizada)
        """
        # Fórmula: threshold = base * (1 + vol/0.20)
        # Si vol = 20%, threshold = base
        # Si vol = 40%, threshold = 2 * base
        volatility_multiplier = 1 + max(0, market_volatility - 0.20) / 0.20
        self.threshold = self.base_threshold * volatility_multiplier
