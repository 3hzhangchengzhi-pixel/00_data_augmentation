"""モデル評価モジュール

RMSE / MAE / R² / MAPE 指標計算、モデル比較。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calc_rmse(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
) -> float:
    """RMSE (Root Mean Squared Error) を計算する。"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calc_mae(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
) -> float:
    """MAE (Mean Absolute Error) を計算する。"""
    return float(mean_absolute_error(y_true, y_pred))


def calc_r2(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
) -> float:
    """R² (決定係数) を計算する。"""
    return float(r2_score(y_true, y_pred))


def calc_mape(
    y_true: NDArray[np.floating[Any]],
    y_pred: NDArray[np.floating[Any]],
) -> float:
    """MAPE (Mean Absolute Percentage Error) を計算する。

    y_true にゼロが含まれる場合、該当行を除外して計算する。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true != 0
    if not np.any(mask):
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def generate_comparison_table(
    models: dict[str, Any],
    X_test: NDArray[np.floating[Any]],
    y_test: NDArray[np.floating[Any]],
) -> pd.DataFrame:
    """複数モデルの評価指標を比較するテーブルを生成する。

    Parameters
    ----------
    models : dict[str, model]
        モデル名とモデルオブジェクトの辞書。
    X_test : array-like
        テスト特徴量。
    y_test : array-like
        テスト正解値。

    Returns
    -------
    pd.DataFrame
        RMSE, MAE, R², MAPE の比較表（RMSE 昇順ソート）。
    """
    rows: list[dict[str, Any]] = []
    y_test_arr = np.asarray(y_test)

    for name, model in models.items():
        y_pred = model.predict(np.asarray(X_test))
        rows.append(
            {
                "model": name,
                "RMSE": calc_rmse(y_test_arr, y_pred),
                "MAE": calc_mae(y_test_arr, y_pred),
                "R2": calc_r2(y_test_arr, y_pred),
                "MAPE": calc_mape(y_test_arr, y_pred),
            }
        )

    df = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return df
