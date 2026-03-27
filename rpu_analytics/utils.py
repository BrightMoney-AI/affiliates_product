import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


def ewma(series: pd.Series, decay: float = 0.7, min_periods: int = 1) -> pd.Series:
    """Exponentially weighted moving average on a pandas Series."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    return series.ewm(alpha=1 - decay, min_periods=min_periods).mean()


def rolling_avg(series: pd.Series, window: int = 4, shift: int = 1) -> pd.Series:
    """Shifted rolling average."""
    if series is None or len(series) == 0:
        return pd.Series(dtype=float)
    return series.rolling(window=window, min_periods=1).mean().shift(shift)


def trend_extrapolate(
    series: pd.Series, periods_ahead: int = 1
) -> Tuple[float, float]:
    """Linear regression extrapolation.

    Returns (projected_value, r_squared).
    If the series has fewer than 2 points or zero variance, returns
    (last known value or NaN, 0.0).
    """
    if series is None or len(series) < 2:
        last = float(series.iloc[-1]) if series is not None and len(series) == 1 else np.nan
        return (last, 0.0)

    clean = series.dropna()
    if len(clean) < 2:
        return (float(clean.iloc[-1]) if len(clean) == 1 else np.nan, 0.0)

    x = np.arange(len(clean), dtype=float)
    y = clean.values.astype(float)

    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    ss_yy = ((y - y_mean) ** 2).sum()
    ss_xy = ((x - x_mean) * (y - y_mean)).sum()

    if ss_xx == 0:
        return (float(y_mean), 0.0)

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    r_squared = (ss_xy**2) / (ss_xx * ss_yy) if ss_yy != 0 else 0.0
    projected = intercept + slope * (len(clean) - 1 + periods_ahead)

    return (float(projected), float(r_squared))


def shapley_decomposition(
    prev: Dict[str, float], curr: Dict[str, float], factors: List[str]
) -> Dict[str, float]:
    """Decompose change in product of factors into per-factor contributions.

    Uses the Shapley-value inspired symmetric decomposition.  For two factors
    a, b the product change is decomposed as:
        da contribution = db_held_at_avg * da
    where the "average" of the other factor is (prev + curr) / 2.

    For the general case with n factors, each factor's marginal contribution is
    computed while holding all other factors at their midpoint between prev and
    curr.  Any residual interaction term is reported separately.

    prev/curr are dicts like {"imp": 0.4, "ctr": 0.02, "rpc": 50}.
    Returns dict with per-factor contribution + "interaction" term.
    """
    if not factors:
        return {"interaction": 0.0}

    prev_product = 1.0
    curr_product = 1.0
    for f in factors:
        prev_product *= prev.get(f, 0.0)
        curr_product *= curr.get(f, 0.0)

    total_change = curr_product - prev_product
    contributions: Dict[str, float] = {}

    for target in factors:
        # Hold all other factors at their midpoint, vary target factor
        other_product = 1.0
        for f in factors:
            if f != target:
                mid = (prev.get(f, 0.0) + curr.get(f, 0.0)) / 2.0
                other_product *= mid
        delta = curr.get(target, 0.0) - prev.get(target, 0.0)
        contributions[target] = other_product * delta

    attributed = sum(contributions.values())
    contributions["interaction"] = total_change - attributed

    return contributions


def z_score(value: float, history: pd.Series) -> float:
    """Z-score of value against pandas Series history."""
    if history is None or len(history) < 2:
        return 0.0
    std = history.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float((value - history.mean()) / std)


def percentile_rank(value: float, distribution: pd.Series) -> float:
    """Percentile rank of value within distribution (0-100)."""
    if distribution is None or len(distribution) == 0:
        return 0.0
    count_below = (distribution < value).sum()
    return float(count_below / len(distribution) * 100.0)


def hhi(shares: List[float]) -> float:
    """Herfindahl-Hirschman Index from list of shares (fractions, not pcts)."""
    if not shares:
        return 0.0
    return float(sum(s**2 for s in shares))


def format_pct(value: float, decimals: int = 1) -> str:
    """Format as percentage string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def trend_arrow(current: float, previous: float) -> str:
    """Returns 'up', 'down', or 'flat'."""
    if current is None or previous is None:
        return "flat"
    try:
        if float(current) > float(previous):
            return "up"
        elif float(current) < float(previous):
            return "down"
        return "flat"
    except (TypeError, ValueError):
        return "flat"
