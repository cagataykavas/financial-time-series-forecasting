from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


def make_features(close: pd.Series) -> pd.DataFrame:
    r = close.pct_change()
    x = pd.DataFrame(index=close.index)
    for lag in [1, 2, 3, 5, 10, 20]:
        x[f"return_lag_{lag}"] = r.shift(lag)
    for window in [5, 10, 20, 60]:
        x[f"mean_{window}"] = r.shift(1).rolling(window).mean()
        x[f"vol_{window}"] = r.shift(1).rolling(window).std()
        x[f"momentum_{window}"] = close.shift(1).pct_change(window)
    x["target_next_return"] = r.shift(-1)
    return x.dropna()


def walk_forward(data: pd.DataFrame, min_train: int = 250, test_size: int = 20):
    features = [c for c in data.columns if c != "target_next_return"]
    predictions = []
    for start in range(min_train, len(data), test_size):
        train = data.iloc[:start]
        test = data.iloc[start : start + test_size]
        if test.empty:
            break
        model = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=150, random_state=42
        )
        model.fit(train[features], train["target_next_return"])
        pred = model.predict(test[features])
        fold = pd.DataFrame(
            {"actual": test["target_next_return"], "prediction": pred}, index=test.index
        )
        predictions.append(fold)
    return pd.concat(predictions)


def metrics(pred: pd.DataFrame) -> dict[str, float]:
    direction = np.mean(np.sign(pred["actual"]) == np.sign(pred["prediction"]))
    strategy = np.sign(pred["prediction"]) * pred["actual"]
    sharpe = np.sqrt(252) * strategy.mean() / (strategy.std() + 1e-12)
    return {
        "mae": float(mean_absolute_error(pred["actual"], pred["prediction"])),
        "directional_accuracy": float(direction),
        "toy_strategy_sharpe": float(sharpe),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    n = 900
    noise = rng.normal(0, 0.012, n)
    returns = np.zeros(n)
    for i in range(1, n):
        returns[i] = 0.08 * returns[i - 1] + noise[i]
    close = pd.Series(
        100 * np.exp(np.cumsum(returns)),
        index=pd.date_range("2021-01-01", periods=n, freq="B"),
    )
    dataset = make_features(close)
    prediction_frame = walk_forward(dataset)
    print(metrics(prediction_frame))
