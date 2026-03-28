"""
Portfolio Optimizer Package.

A modular framework for portfolio optimization.
"""

__version__ = "0.1.0"

# Re-export main components for convenience
from .core.portfolio import Portfolio
from .core.returns import ExpectedReturnsModel
from .core.risk import RiskModel
from .core.optimizer import Optimizer
from .core.formulation import OptimizationProblem
from .core.constraint import Constraint
from .core.result import OptimizationResult
from .core.asset_class import AssetClass
from .core.asset_class_registry import AssetClassRegistry
from .utils import (
    plot_price_evolution,
    plot_asset_weights,
    plot_class_allocation,
    plot_efficient_frontier,
    create_rebalancing_table,
    plot_correlation_heatmap,
    plot_risk_return_scatter,
)

__all__ = [
    "__version__",
    "Portfolio",
    "ExpectedReturnsModel",
    "RiskModel",
    "Optimizer",
    "OptimizationProblem",
    "Constraint",
    "OptimizationResult",
    "AssetClass",
    "AssetClassRegistry",
    # Utils
    "plot_price_evolution",
    "plot_asset_weights",
    "plot_class_allocation",
    "plot_efficient_frontier",
    "create_rebalancing_table",
    "plot_correlation_heatmap",
    "plot_risk_return_scatter",
]
