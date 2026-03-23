"""モデル訓練モジュール

LinearRegression / RandomForest / XGBoost / LightGBM 対応。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split

logger = logging.getLogger(__name__)


def split_data(
    X: NDArray[np.floating[Any]] | pd.DataFrame,
    y: NDArray[np.floating[Any]] | pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
]:
    """データを訓練/検証/テストに分割する (70/15/15)。"""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("比率の合計は 1.0 でなければなりません")

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    # まず train と temp に分割
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_arr, y_arr, test_size=temp_ratio, random_state=random_state
    )

    # temp を val と test に分割
    val_fraction = val_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1.0 - val_fraction, random_state=random_state
    )

    logger.info(
        "データ分割: train=%d, val=%d, test=%d", len(X_train), len(X_val), len(X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_linear_regression(
    X_train: NDArray[np.floating[Any]],
    y_train: NDArray[np.floating[Any]],
) -> LinearRegression:
    """線形回帰ベースラインモデルを訓練する。"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("LinearRegression 訓練完了: R²=%.4f", model.score(X_train, y_train))
    return model


def train_random_forest(
    X_train: NDArray[np.floating[Any]],
    y_train: NDArray[np.floating[Any]],
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
    **kwargs: Any,
) -> RandomForestRegressor:
    """ランダムフォレストモデルを訓練する。"""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs,
    )
    model.fit(X_train, y_train)
    logger.info("RandomForest 訓練完了: R²=%.4f", model.score(X_train, y_train))
    return model


def train_xgboost(
    X_train: NDArray[np.floating[Any]],
    y_train: NDArray[np.floating[Any]],
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """XGBoost モデルを訓練する。"""
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbosity=0,
        **kwargs,
    )
    model.fit(X_train, y_train)
    logger.info("XGBoost 訓練完了: R²=%.4f", model.score(X_train, y_train))
    return model


def train_lightgbm(
    X_train: NDArray[np.floating[Any]],
    y_train: NDArray[np.floating[Any]],
    n_estimators: int = 100,
    max_depth: int = -1,
    learning_rate: float = 0.1,
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """LightGBM モデルを訓練する。"""
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        verbosity=-1,
        **kwargs,
    )
    model.fit(X_train, y_train)
    logger.info("LightGBM 訓練完了: R²=%.4f", model.score(X_train, y_train))
    return model


def cross_validate_model(
    model: Any,
    X: NDArray[np.floating[Any]] | pd.DataFrame,
    y: NDArray[np.floating[Any]] | pd.Series,
    n_folds: int = 5,
    scoring: str = "neg_mean_squared_error",
    random_state: int = 42,
) -> dict[str, Any]:
    """K 分割交差検証を実行する。"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(
        model, np.asarray(X), np.asarray(y), cv=kf, scoring=scoring
    )

    result = {
        "n_folds": n_folds,
        "scoring": scoring,
        "scores": scores,
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }
    logger.info(
        "交差検証 (%d-fold): mean=%.4f, std=%.4f",
        n_folds,
        result["mean"],
        result["std"],
    )
    return result
