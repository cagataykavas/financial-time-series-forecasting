from __future__ import annotations

import pytest

from finforecast.synthetic import synthetic_market


@pytest.mark.parametrize("rows", [0, 1, -10])
def test_synthetic_market_rejects_too_few_rows(rows: int):
    with pytest.raises(ValueError, match="at least 2"):
        synthetic_market(rows)


def test_synthetic_market_rejects_non_integer_rows():
    with pytest.raises(TypeError, match="integer"):
        synthetic_market(10.5)  # type: ignore[arg-type]
