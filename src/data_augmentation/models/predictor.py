"""推論インターフェース

モデル読み込み・価格予測・信頼区間出力。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def load_model(path: str | Path) -> Any:
    """保存済みモデルファイルを読み込む。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301


def predict(
    model: Any,
    X: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    """車両パラメータから価格を予測する。"""
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    return model.predict(X_arr)


def predict_with_confidence(
    model: Any,
    X: NDArray[np.floating[Any]],
    confidence: float = 0.95,
) -> dict[str, NDArray[np.floating[Any]] | float]:
    """予測値と信頼区間を出力する。

    RandomForest 系の場合は各木の予測から区間を推定。
    それ以外は残差ベースの近似区間を返す。
    """
    X_arr = np.asarray(X)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)

    point_pred = model.predict(X_arr)

    # RandomForest 系: 各決定木の予測から分位点で区間推定
    if hasattr(model, "estimators_"):
        tree_preds = np.array([tree.predict(X_arr) for tree in model.estimators_])
        alpha = (1 - confidence) / 2
        lower = np.quantile(tree_preds, alpha, axis=0)
        upper = np.quantile(tree_preds, 1 - alpha, axis=0)
    else:
        # 残差ベースの近似: 正規分布の z 値 × 予測値の 10% を標準偏差として使用
        from scipy.stats import norm

        z = norm.ppf(1 - (1 - confidence) / 2)
        std_approx = np.abs(point_pred) * 0.10
        lower = point_pred - z * std_approx
        upper = point_pred + z * std_approx

    return {
        "prediction": point_pred,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
    }
