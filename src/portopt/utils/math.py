"""
Math utilities - Mathematical helper functions.
"""

import numpy as np
from scipy import linalg


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """
    Encuentra la matriz definida positiva más cercana.
    
    Útil para corregir matrices de covarianza numéricamente inestables.
    
    Basado en Higham (1988): "Computing a nearest symmetric positive 
    semidefinite matrix"
    
    Args:
        matrix: Matriz a corregir
    
    Returns:
        Matriz definida positiva más cercana
    """
    # Asegurar simetría
    symmetric = (matrix + matrix.T) / 2
    
    # Eigenvalue decomposition
    eigenvalues, eigenvectors = linalg.eigh(symmetric)
    
    # Reemplazar eigenvalues negativos con epsilon
    epsilon = 1e-8
    eigenvalues[eigenvalues < epsilon] = epsilon
    
    # Reconstruir matriz
    result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    
    return result


def sharpe_optimal_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.0
) -> np.ndarray:
    """
    Calcula pesos óptimos para maximizar ratio de Sharpe.
    
    Solución analítica para portafolio sin restricciones (excepto presupuesto).
    
    Args:
        expected_returns: Vector de retornos esperados
        cov_matrix: Matriz de covarianza
        risk_free_rate: Tasa libre de riesgo
    
    Returns:
        Pesos óptimos (normalizados)
    """
    # Exceso de retornos
    excess_returns = expected_returns - risk_free_rate
    
    # Pesos óptimos: Σ^(-1) * (μ - r_f)
    try:
        weights = np.linalg.solve(cov_matrix, excess_returns)
    except np.linalg.LinAlgError:
        # Si la matriz es singular, usar pseudo-inversa
        weights = np.linalg.lstsq(cov_matrix, excess_returns, rcond=None)[0]
    
    # Normalizar para que sumen 1
    weights = weights / weights.sum()
    
    return weights


def portfolio_metrics(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray
) -> dict:
    """
    Calcula métricas básicas del portafolio.
    
    Args:
        weights: Pesos del portafolio
        expected_returns: Retornos esperados
        cov_matrix: Matriz de covarianza
    
    Returns:
        Diccionario con métricas
    """
    expected_return = np.dot(weights, expected_returns)
    variance = weights.T @ cov_matrix @ weights
    volatility = np.sqrt(variance)
    
    return {
        "expected_return": expected_return,
        "variance": variance,
        "volatility": volatility,
        "sharpe_ratio": expected_return / volatility if volatility > 0 else 0
    }
