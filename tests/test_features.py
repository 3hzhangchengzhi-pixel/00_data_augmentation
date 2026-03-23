"""特徴量エンジニアリングモジュールのテスト"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data_augmentation.features.numerical import (
    CURRENT_YEAR,
    add_car_age,
    add_displacement_bucket,
    add_log_price,
    add_mileage_bucket,
)
from data_augmentation.features.categorical import (
    label_encode,
    one_hot_encode,
)
from data_augmentation.features.derived import (
    add_annual_mileage,
    add_depreciation_rate,
    add_inspection_remaining,
    add_price_per_km,
)
from data_augmentation.features.builder import (
    build_features,
    handle_missing_values,
    remove_high_correlation,
)


# ---------------------------------------------------------------------------
# Numerical Features Tests (#50, #51, #52, #53)
# ---------------------------------------------------------------------------
class TestNumericalFeatures:
    """数値特徴量テスト"""

    def test_car_age(self):
        """#50: 車齢計算"""
        df = pd.DataFrame({"year": [2020, 2015, 2022]})
        result = add_car_age(df)
        assert "car_age" in result.columns
        assert result["car_age"].iloc[0] == CURRENT_YEAR - 2020
        assert result["car_age"].iloc[1] == CURRENT_YEAR - 2015

    def test_mileage_bucket(self):
        """#51: 里程分桶"""
        df = pd.DataFrame({"mileage_km": [10000, 45000, 75000, 120000]})
        result = add_mileage_bucket(df)
        assert "mileage_bucket" in result.columns
        assert result["mileage_bucket"].iloc[0] == "0-30k"
        assert result["mileage_bucket"].iloc[1] == "30k-60k"
        assert result["mileage_bucket"].iloc[2] == "60k-100k"
        assert result["mileage_bucket"].iloc[3] == "100k+"

    def test_displacement_bucket(self):
        """#52: 排気量分桶"""
        df = pd.DataFrame({"displacement_cc": [660, 1200, 1800, 2500, 3500]})
        result = add_displacement_bucket(df)
        assert "displacement_bucket" in result.columns
        assert result["displacement_bucket"].iloc[0] == "~1000cc"
        assert result["displacement_bucket"].iloc[1] == "1000-1500cc"
        assert result["displacement_bucket"].iloc[2] == "1500-2000cc"
        assert result["displacement_bucket"].iloc[3] == "2000-3000cc"
        assert result["displacement_bucket"].iloc[4] == "3000cc+"

    def test_log_price(self):
        """#53: 価格対数変換"""
        df = pd.DataFrame({"total_price_yen": [1890000, 1000000, 500000]})
        result = add_log_price(df)
        assert "log_price" in result.columns
        assert not result["log_price"].isna().any()
        assert np.isclose(result["log_price"].iloc[0], np.log1p(1890000))

    def test_log_price_handles_zero(self):
        """log1p(0) = 0"""
        df = pd.DataFrame({"total_price_yen": [0]})
        result = add_log_price(df)
        assert result["log_price"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# Categorical Features Tests (#54, #55, #56, #57)
# ---------------------------------------------------------------------------
class TestCategoricalFeatures:
    """カテゴリ特徴量テスト"""

    def test_region_one_hot(self):
        """#54: 地域独熱編碼"""
        df = pd.DataFrame({"region": ["東京都", "大阪府", "東京都", "愛知県"]})
        result = one_hot_encode(df, "region")
        assert "region" not in result.columns
        assert "region_東京都" in result.columns
        assert "region_大阪府" in result.columns
        assert "region_愛知県" in result.columns
        # 値は 0 or 1
        for col in result.columns:
            assert set(result[col].unique()).issubset({0, 1})

    def test_transmission_one_hot(self):
        """#55: 変速箱独熱編碼"""
        df = pd.DataFrame({"transmission": ["AT", "MT", "CVT", "AT"]})
        result = one_hot_encode(df, "transmission")
        assert "transmission" not in result.columns
        assert "transmission_AT" in result.columns
        assert "transmission_MT" in result.columns
        assert "transmission_CVT" in result.columns

    def test_color_label_encode(self):
        """#56: 色標籤編碼"""
        df = pd.DataFrame({"color": ["白", "黒", "白", "赤"]})
        result, le = label_encode(df, "color")
        assert "color_encoded" in result.columns
        assert result["color_encoded"].dtype in (np.int32, np.int64, int)
        # 同じ色は同じ値
        whites = result[result["color"] == "白"]["color_encoded"]
        assert whites.nunique() == 1

    def test_maker_label_encode(self):
        """#57: 廠商標籤編碼"""
        df = pd.DataFrame({"maker": ["トヨタ", "ホンダ", "トヨタ", "日産"]})
        result, le = label_encode(df, "maker")
        assert "maker_encoded" in result.columns
        # 映射一致
        toyotas = result[result["maker"] == "トヨタ"]["maker_encoded"]
        assert toyotas.nunique() == 1
        # 3 種類 → 3 種類の整数
        assert result["maker_encoded"].nunique() == 3


# ---------------------------------------------------------------------------
# Derived Features Tests (#58, #59, #60, #61, #62)
# ---------------------------------------------------------------------------
class TestDerivedFeatures:
    """派生特徴量テスト"""

    def test_price_per_km(self):
        """#58: 単位里程価格"""
        df = pd.DataFrame({
            "total_price_yen": [1890000, 1000000],
            "mileage_km": [52000, 30000],
        })
        result = add_price_per_km(df)
        assert "price_per_km" in result.columns
        assert np.isclose(result["price_per_km"].iloc[0], 1890000 / 52000)

    def test_price_per_km_zero_mileage(self):
        """里程 0 の場合 NaN"""
        df = pd.DataFrame({
            "total_price_yen": [1000000],
            "mileage_km": [0],
        })
        result = add_price_per_km(df)
        assert np.isnan(result["price_per_km"].iloc[0])

    def test_annual_mileage(self):
        """#59: 年均里程"""
        df = pd.DataFrame({
            "mileage_km": [60000, 30000],
            "car_age": [6, 3],
        })
        result = add_annual_mileage(df)
        assert "annual_mileage" in result.columns
        assert np.isclose(result["annual_mileage"].iloc[0], 10000.0)
        assert np.isclose(result["annual_mileage"].iloc[1], 10000.0)

    def test_annual_mileage_zero_age(self):
        """車齢 0 の場合 NaN"""
        df = pd.DataFrame({"mileage_km": [10000], "car_age": [0]})
        result = add_annual_mileage(df)
        assert np.isnan(result["annual_mileage"].iloc[0])

    def test_depreciation_rate(self):
        """#60: 折旧率"""
        df = pd.DataFrame({
            "total_price_yen": [1890000, 500000],
            "car_age": [5, 10],
        })
        result = add_depreciation_rate(df)
        assert "depreciation_rate" in result.columns
        assert np.isclose(result["depreciation_rate"].iloc[0], 1890000 / 6)
        # 値は正の範囲内
        assert (result["depreciation_rate"] > 0).all()

    def test_inspection_remaining_months(self):
        """#61: 車検残り月数"""
        ref = datetime(2026, 3, 1)
        df = pd.DataFrame({"inspection": ["2027-06", "2025-01", "2026-03"]})
        result = add_inspection_remaining(df, reference_date=ref)
        assert "inspection_remaining_months" in result.columns
        assert result["inspection_remaining_months"].iloc[0] == 15  # 2027-06 - 2026-03
        assert result["inspection_remaining_months"].iloc[1] == 0   # 過期
        assert result["inspection_remaining_months"].iloc[2] == 0   # 同月=0

    def test_has_inspection(self):
        """#62: 車検有無ブール"""
        ref = datetime(2026, 3, 1)
        df = pd.DataFrame({"inspection": ["2027-06", "2025-01"]})
        result = add_inspection_remaining(df, reference_date=ref)
        assert "has_inspection" in result.columns
        assert result["has_inspection"].iloc[0] == True  # noqa: E712
        assert result["has_inspection"].iloc[1] == False  # noqa: E712


# ---------------------------------------------------------------------------
# Builder Tests (#63, #64, #65)
# ---------------------------------------------------------------------------
class TestBuilder:
    """特徴量構築入口テスト"""

    def _make_sample_df(self) -> pd.DataFrame:
        """テスト用サンプル DataFrame"""
        return pd.DataFrame({
            "total_price_yen": [1890000, 1000000, 500000, 800000, 1200000],
            "mileage_km": [52000, 30000, 80000, 45000, 60000],
            "year": [2020, 2018, 2015, 2019, 2017],
            "transmission": ["CVT", "AT", "MT", "CVT", "AT"],
            "region": ["東京都", "大阪府", "愛知県", "東京都", "大阪府"],
            "inspection": ["2027-06", "2026-01", "2025-06", "2027-03", "2026-09"],
        })

    def test_handle_missing_values(self):
        """#63: 缺失值処理"""
        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0],
            "b": [np.nan, 2.0, np.nan],
            "c": ["x", None, "x"],
        })
        result = handle_missing_values(df)
        assert not result.select_dtypes(include=[np.number]).isna().any().any()
        assert not result["c"].isna().any()
        # 数値は中央値で補完
        assert result["a"].iloc[1] == 2.0  # median of [1, 3]
        # カテゴリは最頻値で補完
        assert result["c"].iloc[1] == "x"

    def test_remove_high_correlation(self):
        """#64: 高相関除去"""
        np.random.seed(42)
        n = 100
        a = np.random.randn(n)
        df = pd.DataFrame({
            "a": a,
            "b": a + np.random.randn(n) * 0.01,  # a と高相関
            "c": np.random.randn(n),  # 独立
        })
        result = remove_high_correlation(df, threshold=0.95)
        # a と b のうち 1 つが除去される
        assert result.shape[1] < 3
        assert "c" in result.columns

    def test_build_features_output(self):
        """#65: X と y の出力"""
        df = self._make_sample_df()
        X, y = build_features(
            df,
            target_col="total_price_yen",
            onehot_cols=["transmission", "region"],
        )
        # X は数値のみ
        for col in X.columns:
            assert pd.api.types.is_numeric_dtype(X[col]), f"{col} is not numeric: {X[col].dtype}"
        # y は一維
        assert isinstance(y, pd.Series)
        assert len(y) == len(X)
        # 行数一致
        assert X.shape[0] == 5
        # 無 NaN
        assert not X.isna().any().any()
