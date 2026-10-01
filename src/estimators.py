from typing import Tuple, Dict
import numpy as np
from sklearn.covariance import LedoitWolf

def compute_sample_covariance(returns: np.ndarray) -> np.ndarray:
    return np.cov(returns, rowvar=False)

def compute_ledoit_wolf_covariance(returns: np.ndarray) -> Tuple[np.ndarray, float]:
    lw = LedoitWolf(assume_centered=False)
    lw.fit(returns)
    return lw.covariance_, float(lw.shrinkage_)

def inspect_spectrum(cov_matrix: np.ndarray) -> Dict[str, float]:
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    eigenvalues = np.sort(eigenvalues)[::-1]
    lambda_max = eigenvalues[0]
    lambda_min = max(eigenvalues[-1], 1e-12)
    condition_number = lambda_max / lambda_min
    return {
        "lambda_max": float(lambda_max),
        "lambda_min": float(lambda_min),
        "condition_number": float(condition_number),
        "trace": float(np.sum(eigenvalues))
    }
