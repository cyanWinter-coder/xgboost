import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import xgboost as xgb


def _make_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_samples = 512
    cat = rng.integers(0, 3, size=n_samples).astype(np.float32)
    missing = rng.choice(n_samples, size=n_samples // 4, replace=False)
    cat[missing] = np.nan
    numeric = rng.normal(size=(n_samples, 1)).astype(np.float32)
    y = np.where(np.isnan(cat), 5.0, -5.0) + 0.1 * numeric[:, 0]
    return np.column_stack([cat, numeric]).astype(np.float32), y.astype(np.float32)


def _train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_cat_to_onehot: int,
    monotone_constraints: tuple[int, int] = (0, 0),
    nthread: int = 4,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(
        X,
        label=y,
        feature_types=["c", "q"],
        enable_categorical=True,
    )
    params: dict[str, Any] = {
        "tree_method": "hist",
        "max_depth": 2,
        "eta": 1.0,
        "base_score": 0.0,
        "min_child_weight": 0.0,
        "lambda": 0.0,
        "max_cat_to_onehot": max_cat_to_onehot,
        "monotone_constraints": monotone_constraints,
        "seed": 3,
        "nthread": nthread,
    }
    return xgb.train(params, dtrain, num_boost_round=8)


@pytest.mark.parametrize("max_cat_to_onehot", [1, 4])
def test_missing_only_categorical_split(max_cat_to_onehot: int, tmp_path: Path) -> None:
    X, y = _make_data()
    booster = _train(X, y, max_cat_to_onehot=max_cat_to_onehot)
    root = json.loads(booster.get_dump(dump_format="json")[0])

    assert root["split"] == "f0"
    assert sorted(root["split_condition"]) == [0, 1, 2]
    left_id = root["children"][0]["nodeid"]
    right_id = root["children"][1]["nodeid"]
    assert root["missing"] == left_id

    dtrain = xgb.DMatrix(
        X,
        feature_types=["c", "q"],
        enable_categorical=True,
    )
    pred = booster.predict(dtrain)
    assert np.mean(pred[np.isnan(X[:, 0])]) > np.mean(pred[~np.isnan(X[:, 0])])
    assert left_id != right_id

    model_path = tmp_path / "categorical-missing.json"
    booster.save_model(model_path)
    loaded = xgb.Booster(model_file=str(model_path))
    np.testing.assert_array_equal(pred, loaded.predict(dtrain))


@pytest.mark.parametrize("max_cat_to_onehot", [1, 4])
def test_missing_only_categorical_auc(max_cat_to_onehot: int) -> None:
    X, _ = _make_data(seed=2)
    y = np.isnan(X[:, 0]).astype(np.float32)
    dtrain = xgb.DMatrix(
        X,
        label=y,
        feature_types=["c", "q"],
        enable_categorical=True,
    )
    evals_result: dict[str, dict[str, list[float]]] = {}
    xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "max_depth": 2,
            "max_cat_to_onehot": max_cat_to_onehot,
            "seed": 3,
        },
        dtrain,
        num_boost_round=4,
        evals=[(dtrain, "train")],
        evals_result=evals_result,
    )
    assert evals_result["train"]["auc"][-1] > 0.99


@pytest.mark.parametrize("max_cat_to_onehot", [1, 4])
def test_categorical_missing_and_monotone(max_cat_to_onehot: int) -> None:
    X, y = _make_data(seed=4)
    y = y + 2.0 * X[:, 1]
    booster = _train(
        X,
        y,
        max_cat_to_onehot=max_cat_to_onehot,
        monotone_constraints=(0, 1),
    )

    grid = np.linspace(-3, 3, 64, dtype=np.float32)
    for cat in [0.0, 1.0, 2.0, np.nan]:
        X_test = np.column_stack([np.full_like(grid, cat), grid]).astype(np.float32)
        dtest = xgb.DMatrix(
            X_test,
            feature_types=["c", "q"],
            enable_categorical=True,
        )
        pred = booster.predict(dtest)
        assert np.all(np.diff(pred) >= -1e-6)


@pytest.mark.parametrize("max_cat_to_onehot", [1, 4])
def test_categorical_missing_reproducible(max_cat_to_onehot: int) -> None:
    X, y = _make_data(seed=5)
    params = {
        "max_cat_to_onehot": max_cat_to_onehot,
    }
    single = _train(X, y, nthread=1, **params)
    parallel = _train(X, y, nthread=4, **params)

    dtest = xgb.DMatrix(
        X,
        feature_types=["c", "q"],
        enable_categorical=True,
    )
    np.testing.assert_array_equal(single.predict(dtest), parallel.predict(dtest))
