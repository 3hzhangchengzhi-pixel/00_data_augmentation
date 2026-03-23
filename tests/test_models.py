"""機械学習モデルモジュールのテスト"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from data_augmentation.models.evaluator import (
    calc_mae,
    calc_mape,
    calc_r2,
    calc_rmse,
    generate_comparison_table,
)
from data_augmentation.models.predictor import (
    load_model,
    predict,
    predict_with_confidence,
)
from data_augmentation.models.trainer import (
    cross_validate_model,
    split_data,
    train_lightgbm,
    train_linear_regression,
    train_random_forest,
    train_xgboost,
)
from data_augmentation.models.tuner import grid_search, random_search

# ---------------------------------------------------------------------------
# 共通フィクスチャ: 合成回帰データ
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def regression_data():
    """小規模合成データ (200 samples, 5 features)"""
    X, y = make_regression(
        n_samples=200, n_features=5, noise=10.0, random_state=42
    )
    return X, y


@pytest.fixture(scope="module")
def split_regression_data(regression_data):
    """70/15/15 に分割済みデータ"""
    X, y = regression_data
    return split_data(X, y)


@pytest.fixture(scope="module")
def trained_lr(split_regression_data):
    X_train, _, _, y_train, _, _ = split_regression_data
    return train_linear_regression(X_train, y_train)


@pytest.fixture(scope="module")
def trained_rf(split_regression_data):
    X_train, _, _, y_train, _, _ = split_regression_data
    return train_random_forest(X_train, y_train, n_estimators=20, random_state=42)


# ===========================================================================
# trainer テスト
# ===========================================================================


class TestTrainer:
    """モデル訓練テスト"""

    def test_train_linear_regression(self, split_regression_data):
        """#66: 線形回帰ベースラインモデルを訓練できる"""
        X_train, _, _, y_train, _, _ = split_regression_data
        model = train_linear_regression(X_train, y_train)
        assert isinstance(model, LinearRegression)
        assert hasattr(model, "coef_")
        assert model.score(X_train, y_train) > 0

    def test_train_random_forest(self, split_regression_data):
        """#67: ランダムフォレストモデルを訓練できる"""
        X_train, _, _, y_train, _, _ = split_regression_data
        model = train_random_forest(
            X_train, y_train, n_estimators=20, random_state=42
        )
        assert isinstance(model, RandomForestRegressor)
        assert len(model.estimators_) == 20
        assert model.score(X_train, y_train) > 0

    def test_train_xgboost(self, split_regression_data):
        """#68: XGBoost モデルを訓練できる"""
        X_train, _, _, y_train, _, _ = split_regression_data
        model = train_xgboost(
            X_train, y_train, n_estimators=20, random_state=42
        )
        preds = model.predict(X_train)
        assert len(preds) == len(X_train)
        assert model.score(X_train, y_train) > 0

    def test_train_lightgbm(self, split_regression_data):
        """#69: LightGBM モデルを訓練できる"""
        X_train, _, _, y_train, _, _ = split_regression_data
        model = train_lightgbm(
            X_train, y_train, n_estimators=20, random_state=42
        )
        preds = model.predict(X_train)
        assert len(preds) == len(X_train)
        assert model.score(X_train, y_train) > 0

    def test_split_data_ratios(self, regression_data):
        """#70: データを 70/15/15 に正しく分割できる"""
        X, y = regression_data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        total = len(X)
        # 許容誤差 ±5%
        assert abs(len(X_train) / total - 0.70) < 0.05
        assert abs(len(X_val) / total - 0.15) < 0.05
        assert abs(len(X_test) / total - 0.15) < 0.05
        assert len(X_train) + len(X_val) + len(X_test) == total
        assert len(y_train) + len(y_val) + len(y_test) == total

    def test_cross_validate(self, regression_data):
        """#71: 5 折交差検証を実行できる"""
        X, y = regression_data
        model = LinearRegression()
        result = cross_validate_model(model, X, y, n_folds=5)
        assert result["n_folds"] == 5
        assert len(result["scores"]) == 5
        assert "mean" in result
        assert "std" in result
        # neg_mean_squared_error なのでスコアは負
        assert result["mean"] < 0


# ===========================================================================
# evaluator テスト
# ===========================================================================


class TestEvaluator:
    """モデル評価テスト"""

    def test_calc_rmse(self):
        """#72: RMSE を正しく計算できる"""
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        rmse = calc_rmse(y_true, y_pred)
        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert abs(rmse - expected) < 1e-10

    def test_calc_mae(self):
        """#73: MAE を正しく計算できる"""
        y_true = np.array([3.0, -0.5, 2.0, 7.0])
        y_pred = np.array([2.5, 0.0, 2.0, 8.0])
        mae = calc_mae(y_true, y_pred)
        expected = np.mean(np.abs(y_true - y_pred))
        assert abs(mae - expected) < 1e-10

    def test_calc_r2(self):
        """#74: R² を正しく計算できる"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.0, 2.9, 4.1, 5.0])
        r2 = calc_r2(y_true, y_pred)
        assert 0.95 < r2 <= 1.0

    def test_calc_mape(self):
        """#75: MAPE を正しく計算できる"""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 330.0])
        mape = calc_mape(y_true, y_pred)
        # (10/100 + 10/200 + 30/300) / 3 * 100 = (0.1+0.05+0.1)/3*100 ≈ 8.33%
        assert abs(mape - 8.333333) < 0.01

    def test_calc_mape_with_zero(self):
        """MAPE: y_true にゼロがあっても計算できる"""
        y_true = np.array([0.0, 100.0, 200.0])
        y_pred = np.array([10.0, 110.0, 190.0])
        mape = calc_mape(y_true, y_pred)
        # ゼロ行は除外: (10/100 + 10/200) / 2 * 100 = 7.5%
        assert abs(mape - 7.5) < 0.01

    def test_generate_comparison_table(
        self, split_regression_data, trained_lr, trained_rf
    ):
        """#76: 多モデル比較テーブルを生成できる"""
        _, _, X_test, _, _, y_test = split_regression_data
        models = {"LinearRegression": trained_lr, "RandomForest": trained_rf}
        table = generate_comparison_table(models, X_test, y_test)

        assert isinstance(table, pd.DataFrame)
        assert len(table) == 2
        assert set(table.columns) >= {"model", "RMSE", "MAE", "R2", "MAPE"}
        # RMSE 昇順ソート
        assert table["RMSE"].iloc[0] <= table["RMSE"].iloc[1]


# ===========================================================================
# predictor テスト
# ===========================================================================


class TestPredictor:
    """推論テスト"""

    def test_load_model(self, trained_lr):
        """#77: 保存済みモデルを読み込める"""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(trained_lr, f)
            f.flush()
            path = Path(f.name)

        loaded = load_model(path)
        assert hasattr(loaded, "predict")
        path.unlink()

    def test_load_model_file_not_found(self):
        """load_model: 存在しないファイルで FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_model("/tmp/nonexistent_model_abc123.pkl")

    def test_predict(self, trained_lr, split_regression_data):
        """#78: 車両パラメータから価格を予測できる"""
        _, _, X_test, _, _, _ = split_regression_data
        preds = predict(trained_lr, X_test)
        assert len(preds) == len(X_test)
        assert all(np.isfinite(preds))

    def test_predict_single_sample(self, trained_lr):
        """predict: 1D 入力でも予測できる"""
        sample = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        preds = predict(trained_lr, sample)
        assert len(preds) == 1

    def test_predict_with_confidence_rf(self, trained_rf, split_regression_data):
        """#79: RandomForest で信頼区間を出力できる"""
        _, _, X_test, _, _, _ = split_regression_data
        result = predict_with_confidence(trained_rf, X_test, confidence=0.95)
        assert "prediction" in result
        assert "lower" in result
        assert "upper" in result
        assert result["confidence"] == 0.95
        assert len(result["prediction"]) == len(X_test)
        # lower <= prediction <= upper
        assert np.all(result["lower"] <= result["prediction"])
        assert np.all(result["prediction"] <= result["upper"])

    def test_predict_with_confidence_lr(self, trained_lr, split_regression_data):
        """predict_with_confidence: LinearRegression でも区間を返す"""
        _, _, X_test, _, _, _ = split_regression_data
        result = predict_with_confidence(trained_lr, X_test, confidence=0.90)
        assert result["confidence"] == 0.90
        assert len(result["lower"]) == len(X_test)
        assert np.all(result["lower"] <= result["upper"])


# ===========================================================================
# tuner テスト
# ===========================================================================


class TestTuner:
    """ハイパーパラメータチューニングテスト"""

    def test_grid_search(self, regression_data):
        """#80: Grid Search でハイパーパラメータ探索できる"""
        X, y = regression_data
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        }
        result = grid_search(model, param_grid, X, y, cv=3)

        assert "best_params" in result
        assert "best_score" in result
        assert "best_estimator" in result
        assert result["best_params"]["n_estimators"] in [10, 20]
        assert result["best_params"]["max_depth"] in [3, 5]

    def test_random_search(self, regression_data):
        """#81: Random Search でハイパーパラメータ探索できる"""
        X, y = regression_data
        model = RandomForestRegressor(random_state=42)
        param_distributions = {
            "n_estimators": [10, 20, 50],
            "max_depth": [3, 5, 10, None],
        }
        result = random_search(
            model, param_distributions, X, y, n_iter=5, cv=3, random_state=42
        )

        assert "best_params" in result
        assert "best_score" in result
        assert "best_estimator" in result
        assert result["n_iter"] == 5
