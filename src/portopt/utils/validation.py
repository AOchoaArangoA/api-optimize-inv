"""
Validation utilities - Input data validation.
"""

import numpy as np
from typing import Optional


def validate_weights(
    weights: np.ndarray,
    min_value: float = 0.0,
    max_value: float = 1.0,
    sum_target: float = 1.0,
    tolerance: float = 1e-6
) -> bool:
    """
    Valida que los pesos sean correctos.
    
    Args:
        weights: Array de pesos
        min_value: Valor mínimo permitido
        max_value: Valor máximo permitido
        sum_target: Suma objetivo
        tolerance: Tolerancia para comparaciones
    
    Returns:
        True si los pesos son válidos
    
    Raises:
        ValueError: Si los pesos no son válidos
    """
    if not isinstance(weights, np.ndarray):
        raise TypeError("Weights must be a numpy array")
    
    if len(weights.shape) != 1:
        raise ValueError("Weights must be a 1D array")
    
    if np.any(weights < min_value - tolerance):
        raise ValueError(f"Weights must be >= {min_value}")
    
    if np.any(weights > max_value + tolerance):
        raise ValueError(f"Weights must be <= {max_value}")
    
    if not np.isclose(weights.sum(), sum_target, atol=tolerance):
        raise ValueError(
            f"Weights must sum to {sum_target}, got {weights.sum()}"
        )
    
    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
        raise ValueError("Weights contain NaN or Inf values")
    
    return True


def validate_covariance_matrix(
    cov_matrix: np.ndarray,
    tolerance: float = 1e-6
) -> bool:
    """
    Valida que la matriz de covarianza sea válida.
    
    Args:
        cov_matrix: Matriz de covarianza
        tolerance: Tolerancia numérica
    
    Returns:
        True si la matriz es válida
    
    Raises:
        ValueError: Si la matriz no es válida
    """
    if not isinstance(cov_matrix, np.ndarray):
        raise TypeError("Covariance matrix must be a numpy array")
    
    if len(cov_matrix.shape) != 2:
        raise ValueError("Covariance matrix must be 2D")
    
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square")
    
    # Verificar simetría
    if not np.allclose(cov_matrix, cov_matrix.T, atol=tolerance):
        raise ValueError("Covariance matrix must be symmetric")
    
    # Verificar que sea semi-definida positiva
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    if np.any(eigenvalues < -tolerance):
        raise ValueError(
            "Covariance matrix must be positive semi-definite. "
            f"Minimum eigenvalue: {eigenvalues.min()}"
        )
    
    # Verificar que no tenga NaN o Inf
    if np.any(np.isnan(cov_matrix)) or np.any(np.isinf(cov_matrix)):
        raise ValueError("Covariance matrix contains NaN or Inf values")
    
    return True
