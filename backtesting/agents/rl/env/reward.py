"""
Pluggable reward functions for TradingEnv.

Each function takes scalar inputs (no pandas) and returns a RewardComponents
dataclass for interpretability. TradingEnv calls the selected function and
uses .total as the RL reward signal.
"""

from dataclasses import dataclass


@dataclass
class RewardComponents:
    """Breakdown of a single step's reward for logging and debugging."""
    step_return: float
    drawdown_penalty: float
    transaction_cost_penalty: float
    concentration_penalty: float
    total: float


def risk_adjusted(
    prev_equity: float,
    curr_equity: float,
    drawdown_from_peak: float,
    turnover: float,
    transaction_cost: float,
    weights: list,
    lambda_drawdown: float = 10.0,
    lambda_transaction_cost: float = 1.0,
    lambda_concentration: float = 0.1,
    drawdown_penalty_threshold: float = 0.05,
    # Accept legacy short-form names too
    lambda_dd: float | None = None,
    lambda_tc: float | None = None,
    lambda_conc: float | None = None,
    dd_threshold: float | None = None,
    **_ignored,
) -> RewardComponents:
    """
    Recommended composite reward.

    step_return:          Annualized daily return (encourages positive returns).
    drawdown_penalty:     Quadratic past dd_threshold (discourages large drawdowns).
    transaction_cost:     Proportional to actual turnover cost (discourages overtrading).
    concentration_penalty: L2 deviation from equal-weight (encourages diversification).
    """
    # Prefer long-form names from config; fall back to legacy short-form
    _ldd = lambda_dd if lambda_dd is not None else lambda_drawdown
    _ltc = lambda_tc if lambda_tc is not None else lambda_transaction_cost
    _lconc = lambda_conc if lambda_conc is not None else lambda_concentration
    _ddth = dd_threshold if dd_threshold is not None else drawdown_penalty_threshold

    step_return = (curr_equity / max(prev_equity, 1.0) - 1.0) * 252.0  # annualized

    dd_excess = max(0.0, abs(drawdown_from_peak) - _ddth)
    drawdown_penalty = _ldd * dd_excess ** 2

    tc_penalty = _ltc * transaction_cost / max(prev_equity, 1.0)

    n = len(weights)
    if n > 0:
        equal_w = 1.0 / n
        conc = sum((w - equal_w) ** 2 for w in weights)
        conc_penalty = _lconc * conc
    else:
        conc_penalty = 0.0

    total = step_return - drawdown_penalty - tc_penalty - conc_penalty

    return RewardComponents(
        step_return=step_return,
        drawdown_penalty=drawdown_penalty,
        transaction_cost_penalty=tc_penalty,
        concentration_penalty=conc_penalty,
        total=total,
    )


def sharpe_scaled(
    prev_equity: float,
    curr_equity: float,
    rolling_vol: float,
    **kwargs,
) -> RewardComponents:
    """Step return divided by rolling volatility. Numerically unstable near zero vol."""
    r = curr_equity / max(prev_equity, 1.0) - 1.0
    vol = max(rolling_vol, 1e-6)
    scaled = r / vol
    return RewardComponents(
        step_return=scaled,
        drawdown_penalty=0.0,
        transaction_cost_penalty=0.0,
        concentration_penalty=0.0,
        total=scaled,
    )


def log_return(
    prev_equity: float,
    curr_equity: float,
    **kwargs,
) -> RewardComponents:
    """Raw log return — simplest baseline, no risk adjustment."""
    import math
    r = math.log(max(curr_equity, 1.0) / max(prev_equity, 1.0))
    return RewardComponents(
        step_return=r,
        drawdown_penalty=0.0,
        transaction_cost_penalty=0.0,
        concentration_penalty=0.0,
        total=r,
    )


REWARD_FNS = {
    "risk_adjusted": risk_adjusted,
    "sharpe_scaled": sharpe_scaled,
    "log_return": log_return,
}


def get_reward_fn(name: str):
    if name not in REWARD_FNS:
        raise ValueError(f"Unknown reward function '{name}'. Choose from: {list(REWARD_FNS)}")
    return REWARD_FNS[name]
