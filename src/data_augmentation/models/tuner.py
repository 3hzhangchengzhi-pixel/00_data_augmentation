"""ハイパーパラメータチューニングモジュール

Grid Search / Random Search 対応。
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logger = logging.getLogger(__name__)


def grid_search(
    model: Any,
    param_grid: dict[str, list[Any]],
    X_train: NDArray[np.floating[Any]],
    y_train: NDArray[np.floating[Any]],
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Grid Search によるハイパーパラメータ探索。"""
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )
    gs.fit(np.asarray(X_train), np.asarray(y_train))

    result = {
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
        "best_estimator": gs.best_estimator_,
        "cv_results": gs.cv_results_,
    }
    logger.info(
        "Grid Search 完了: best_score=%.4f, params=%s",
        result["best_score"],
        result["best_params"],
    )
    return result


def random_search(
    model: Any,
    param_distributions: dict[str, Any],
    X_train: NDArray[np.floating[Any]],
    y_train: NDArray[np.floating[Any]],
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = "neg_mean_squared_error",
    random_state: int = 42,
    n_jobs: int = -1,
) -> dict[str, Any]:
    """Random Search によるハイパーパラメータ探索。"""
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
    rs.fit(np.asarray(X_train), np.asarray(y_train))

    result = {
        "best_params": rs.best_params_,
        "best_score": float(rs.best_score_),
        "best_estimator": rs.best_estimator_,
        "cv_results": rs.cv_results_,
        "n_iter": n_iter,
    }
    logger.info(
        "Random Search 完了: best_score=%.4f, params=%s",
        result["best_score"],
        result["best_params"],
    )
    return result
