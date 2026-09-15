from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baselines import Regressor
from .features import assert_no_target_leakage
from .metrics import BusinessCost, mae, mase, rmse, smape
from .splits import WalkForwardSplitter


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame
    folds: tuple[dict[str, object], ...]

    def model_names(self) -> tuple[str, ...]:
        return tuple(
            column for column in self.predictions.columns if column not in {"actual", "fold"}
        )


class BacktestRunner:
    def __init__(
        self,
        splitter: WalkForwardSplitter,
        model_factories: dict[str, Callable[[], Regressor]],
        *,
        target_column: str = "target_next_return",
    ) -> None:
        if not model_factories:
            raise ValueError("at least one model factory is required")
        reserved = {"actual", "fold"}.intersection(model_factories)
        if reserved:
            raise ValueError(f"model names are reserved output columns: {sorted(reserved)}")
        if not target_column:
            raise ValueError("target_column cannot be empty")
        self.splitter = splitter
        self.model_factories = model_factories
        self.target_column = target_column

    def run(self, data: pd.DataFrame) -> BacktestResult:
        assert_no_target_leakage(data, self.target_column)
        if not data.index.is_monotonic_increasing or not data.index.is_unique:
            raise ValueError("observations must have a unique, increasing time index")
        feature_columns = [column for column in data.columns if column != self.target_column]
        frames: list[pd.DataFrame] = []
        fold_reports: list[dict[str, object]] = []
        for boundary in self.splitter.split(len(data)):
            train = data.iloc[boundary.train_start : boundary.train_end]
            test = data.iloc[boundary.test_start : boundary.test_end]
            frame = pd.DataFrame(
                {"actual": test[self.target_column], "fold": boundary.fold},
                index=test.index,
            )
            report: dict[str, object] = {"boundary": boundary.as_dict(), "models": {}}
            for name, factory in self.model_factories.items():
                model = factory()
                model.fit(train[feature_columns], train[self.target_column])
                predicted = np.asarray(model.predict(test[feature_columns]), dtype=float)
                if predicted.shape != (len(test),) or not np.isfinite(predicted).all():
                    raise ValueError(f"model {name!r} returned an invalid prediction vector")
                frame[name] = predicted
                report["models"][name] = {  # type: ignore[index]
                    "mae": mae(frame["actual"], predicted),
                    "rmse": rmse(frame["actual"], predicted),
                    "smape": smape(frame["actual"], predicted),
                    "mase": mase(frame["actual"], predicted, train[self.target_column]),
                }
            frames.append(frame)
            fold_reports.append(report)
        predictions = pd.concat(frames).sort_index()
        if predictions.index.has_duplicates:
            raise ValueError(
                "overlapping test windows produced duplicate predictions; increase step_size"
            )
        return BacktestResult(predictions, tuple(fold_reports))

    @staticmethod
    def aggregate(
        result: BacktestResult, business_cost: BusinessCost | None = None
    ) -> dict[str, dict[str, float]]:
        actual = result.predictions["actual"]
        output: dict[str, dict[str, float]] = {}
        for name in result.model_names():
            predicted = result.predictions[name]
            metrics = {
                "mae": mae(actual, predicted),
                "rmse": rmse(actual, predicted),
                "smape": smape(actual, predicted),
                "directional_accuracy": float((np.sign(actual) == np.sign(predicted)).mean()),
            }
            if business_cost is not None:
                metrics.update(business_cost.evaluate(actual, predicted))
            output[name] = metrics
        return output
