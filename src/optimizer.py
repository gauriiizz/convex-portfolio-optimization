from typing import Optional
import numpy as np
import cvxpy as cp

def solve_markowitz_qp(
    mu: np.ndarray,
    Sigma: np.ndarray,
    gamma: float = 2.0,
    w_prev: Optional[np.ndarray] = None,
    max_turnover: Optional[float] = None,
    max_weight: float = 0.15
) -> np.ndarray:
    n = len(mu)
    w = cp.Variable(n)
    Sigma_sym = (Sigma + Sigma.T) / 2.0
    
    expected_return = mu @ w
    risk_penalty = 0.5 * gamma * cp.quad_form(w, Sigma_sym)
    objective = cp.Maximize(expected_return - risk_penalty)
    
    constraints = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w <= max_weight
    ]
    if w_prev is not None and max_turnover is not None:
        constraints.append(cp.norm(w - w_prev, 1) <= max_turnover)
        
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6)
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver failed: {problem.status}")
        
    weights = np.clip(w.value, 0.0, 1.0)
    return weights / np.sum(weights)
