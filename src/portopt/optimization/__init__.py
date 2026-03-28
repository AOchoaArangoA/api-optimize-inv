"""
Optimization package.
"""

# Re-export from submodules for convenience
from .formulations.mean_variance import MeanVarianceOptimization
from .formulations.min_variance import MinVarianceOptimization
from .formulations.tracking_error import TrackingErrorOptimization
from .formulations.cvar import CVaROptimization

from .constraints.budget import BudgetConstraint
from .constraints.long_only import LongOnlyConstraint
from .constraints.box import BoxConstraint
from .constraints.sector import SectorConstraint
from .constraints.turnover import TurnoverConstraint
from .constraints.asset_class import AssetClassConstraint

from .solvers.cvxpy import CVXPYSolver
from .solvers.scipy import ScipySolver
from .solvers.heuristic import HeuristicSolver

from .hierarchical.hrp import HierarchicalRiskParity
from .hierarchical.clustering import ClusteringOptimizer

from .rebalancing import (
    RebalancingStrategy,
    AdaptiveRebalancingStrategy,
    RebalancingTrigger
)

__all__ = [
    # Formulations
    "MeanVarianceOptimization",
    "MinVarianceOptimization",
    "TrackingErrorOptimization",
    "CVaROptimization",
    # Constraints
    "BudgetConstraint",
    "LongOnlyConstraint",
    "BoxConstraint",
    "SectorConstraint",
    "TurnoverConstraint",
    "AssetClassConstraint",
    # Solvers
    "CVXPYSolver",
    "ScipySolver",
    "HeuristicSolver",
    # Hierarchical
    "HierarchicalRiskParity",
    "ClusteringOptimizer",
    # Rebalancing
    "RebalancingStrategy",
    "AdaptiveRebalancingStrategy",
    "RebalancingTrigger",
]
