from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_market(rows: int = 1100, seed: int = 42) -> tuple[pd.Series, pd.Series]:
    if isinstance(rows, bool) or not isinstance(rows, int):
        raise TypeError("rows must be an integer")
    if rows < 2:
        raise ValueError("rows must be at least 2")
    rng = np.random.default_rng(seed)
    returns = np.zeros(rows)
    volume = rng.lognormal(14.8, 0.30, rows)
    for i in range(1, rows):
        if i < rows * 0.35:
            phi, sigma = 0.12, 0.010
        elif i < rows * 0.70:
            phi, sigma = -0.08, 0.016
        else:
            phi, sigma = 0.05, 0.008
        seasonal = 0.0015 * np.sin(i / 17.0)
        returns[i] = phi * returns[i - 1] + seasonal + rng.normal(0.0, sigma)
        volume[i] *= 1.0 + min(2.0, abs(returns[i]) * 25)
    close = 100.0 * np.exp(np.cumsum(returns))
    index = pd.date_range("2021-01-04", periods=rows, freq="B")
    return pd.Series(close, index=index, name="close"), pd.Series(
        volume, index=index, name="volume"
    )
