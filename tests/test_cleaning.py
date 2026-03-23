"""データクリーニングモジュールのテスト"""

import pickle
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from data_augmentation.cleaning.normalize import (
    normalize_accident_history,
    normalize_inspection,
    normalize_mileage,
    normalize_price,
    normalize_region,
    normalize_transmission,
    normalize_year,
)
from data_augmentation.cleaning.dedup import (
    cross_source_dedup,
    dedup_by_key,
)
from data_augmentation.cleaning.cleaner import (
    clean_dataframe,
    load_and_merge,
    load_pickle,
    run_cleaning_pipeline,
)


# ---------------------------------------------------------------------------
# normalize_price Tests (#40)
# ---------------------------------------------------------------------------
class TestNormalizePrice:
    """价格标准化テスト"""

    def test_man_yen_to_yen(self):
        """#40: 万円 → 日元整数"""
        assert normalize_price(189.0) == 1890000

    def test_decimal_price(self):
        """小数点付き"""
        assert normalize_price(68.5) == 685000

    def test_none_returns_none(self):
        assert normalize_price(None) is None

    def test_zero(self):
        assert normalize_price(0.0) == 0


# ---------------------------------------------------------------------------
# normalize_mileage Tests (#41)
# ---------------------------------------------------------------------------
class TestNormalizeMileage:
    """里程标准化テスト"""

    def test_float_to_int(self):
        """#41: 浮点数 → 整数公里"""
        assert normalize_mileage(52000.0) == 52000

    def test_small_mileage(self):
        assert normalize_mileage(8000.0) == 8000

    def test_none_returns_none(self):
        assert normalize_mileage(None) is None


# ---------------------------------------------------------------------------
# normalize_year Tests (#42)
# ---------------------------------------------------------------------------
class TestNormalizeYear:
    """年式标准化テスト"""

    def test_reiwa_full(self):
        """#42: 令和2年 → 2020"""
        assert normalize_year("令和2年") == 2020

    def test_reiwa_short(self):
        """R2年 → 2020"""
        assert normalize_year("R2年") == 2020

    def test_heisei(self):
        """平成30年 → 2018"""
        assert normalize_year("平成30年") == 2018

    def test_western_with_wareki(self):
        """2020(R2)年 → 2020"""
        assert normalize_year("2020(R2)年") == 2020

    def test_western_string(self):
        """'2020年' → 2020"""
        assert normalize_year("2020年") == 2020

    def test_integer_passthrough(self):
        """整数はそのまま"""
        assert normalize_year(2020) == 2020

    def test_none_returns_none(self):
        assert normalize_year(None) is None

    def test_plain_number_string(self):
        """'2020' → 2020"""
        assert normalize_year("2020") == 2020


# ---------------------------------------------------------------------------
# normalize_inspection Tests (#43)
# ---------------------------------------------------------------------------
class TestNormalizeInspection:
    """車検日期标准化テスト"""

    def test_yyyymm_format(self):
        """#43: '202611' → '2026-11'"""
        assert normalize_inspection("202611") == "2026-11"

    def test_yyyy_mm_format(self):
        """'2026-11' → '2026-11'"""
        assert normalize_inspection("2026-11") == "2026-11"

    def test_japanese_format(self):
        """'2026年11月' → '2026-11'"""
        assert normalize_inspection("2026年11月") == "2026-11"

    def test_single_digit_month(self):
        """'2025-3' → '2025-03'"""
        assert normalize_inspection("2025-3") == "2025-03"

    def test_none_returns_none(self):
        assert normalize_inspection(None) is None

    def test_unparseable_returns_none(self):
        assert normalize_inspection("なし") is None


# ---------------------------------------------------------------------------
# normalize_accident_history Tests (#44)
# ---------------------------------------------------------------------------
class TestNormalizeAccidentHistory:
    """修復歴标准化テスト"""

    def test_ari_returns_true(self):
        """#44: 'あり' → True"""
        assert normalize_accident_history("あり") is True

    def test_nashi_returns_false(self):
        """'なし' → False"""
        assert normalize_accident_history("なし") is False

    def test_yuu_returns_true(self):
        """'有' → True"""
        assert normalize_accident_history("有") is True

    def test_mu_returns_false(self):
        """'無' → False"""
        assert normalize_accident_history("無") is False

    def test_none_returns_none(self):
        assert normalize_accident_history(None) is None

    def test_unknown_returns_none(self):
        assert normalize_accident_history("不明") is None


# ---------------------------------------------------------------------------
# normalize_transmission Tests (#45)
# ---------------------------------------------------------------------------
class TestNormalizeTransmission:
    """変速箱标准化テスト"""

    def test_cvt(self):
        """#45: 'CVT' → 'CVT'"""
        assert normalize_transmission("CVT") == "CVT"

    def test_automatic(self):
        """'オートマチック' → 'AT'"""
        assert normalize_transmission("オートマチック") == "AT"

    def test_manual(self):
        """'マニュアル' → 'MT'"""
        assert normalize_transmission("マニュアル") == "MT"

    def test_at_string(self):
        """'AT' → 'AT'"""
        assert normalize_transmission("AT") == "AT"

    def test_mt_string(self):
        """'MT' → 'MT'"""
        assert normalize_transmission("MT") == "MT"

    def test_none_returns_none(self):
        assert normalize_transmission(None) is None


# ---------------------------------------------------------------------------
# normalize_region Tests (#49)
# ---------------------------------------------------------------------------
class TestNormalizeRegion:
    """地域标准化テスト"""

    def test_full_with_district(self):
        """#49: '東京都 渋谷区' → '東京都'"""
        assert normalize_region("東京都 渋谷区") == "東京都"

    def test_short_name_tokyo(self):
        """'東京 渋谷' → '東京都'"""
        assert normalize_region("東京 渋谷") == "東京都"

    def test_osaka_fu(self):
        """'大阪 梅田' → '大阪府'"""
        assert normalize_region("大阪 梅田") == "大阪府"

    def test_already_full(self):
        """'北海道' → '北海道'"""
        assert normalize_region("北海道") == "北海道"

    def test_prefecture_suffix_added(self):
        """'愛知' → '愛知県'"""
        assert normalize_region("愛知") == "愛知県"

    def test_none_returns_none(self):
        assert normalize_region(None) is None

    def test_empty_returns_none(self):
        assert normalize_region("") is None


# ---------------------------------------------------------------------------
# dedup_by_key Tests (#46)
# ---------------------------------------------------------------------------
class TestDedupByKey:
    """去重テスト"""

    def test_removes_duplicates_keeps_most_complete(self):
        """#46: 重複排除し、最完整なレコードを保持"""
        df = pd.DataFrame([
            {"title": "シエンタ", "total_price": 189.0, "mileage_km": 52000, "color": "白"},
            {"title": "シエンタ", "total_price": 189.0, "mileage_km": 52000, "color": None},
            {"title": "フィット", "total_price": 100.0, "mileage_km": 30000, "color": "黒"},
        ])
        result = dedup_by_key(df)
        assert len(result) == 2
        # 最完整な（color ありの）レコードが保持される
        sienta = result[result["title"] == "シエンタ"].iloc[0]
        assert sienta["color"] == "白"

    def test_empty_dataframe(self):
        """空 DataFrame"""
        df = pd.DataFrame()
        result = dedup_by_key(df)
        assert result.empty

    def test_no_duplicates(self):
        """重複なし"""
        df = pd.DataFrame([
            {"title": "A", "total_price": 100, "mileage_km": 1000},
            {"title": "B", "total_price": 200, "mileage_km": 2000},
        ])
        result = dedup_by_key(df)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# cross_source_dedup Tests (#47)
# ---------------------------------------------------------------------------
class TestCrossSourceDedup:
    """跨源去重テスト"""

    def test_cross_source_removes_duplicates(self):
        """#47: 跨数据源去重"""
        df = pd.DataFrame([
            {"title": "シエンタ", "total_price": 189.0, "mileage_km": 52000, "source": "carsensor", "region": "東京都"},
            {"title": "シエンタ", "total_price": 189.0, "mileage_km": 52000, "source": "mobilico", "region": None},
            {"title": "フィット", "total_price": 100.0, "mileage_km": 30000, "source": "aucsupport", "region": "大阪府"},
        ])
        result = cross_source_dedup(df)
        assert len(result) == 2
        # 最完整（region あり）のレコードが保持
        sienta = result[result["title"] == "シエンタ"].iloc[0]
        assert sienta["region"] == "東京都"


# ---------------------------------------------------------------------------
# cleaner load_and_merge Tests (#48)
# ---------------------------------------------------------------------------
class TestCleaner:
    """cleaner テスト"""

    def test_load_and_merge_multiple_sources(self):
        """#48: 加载并合并多源 pickle 数据"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 創建 3 個 mock pickle 文件
            sources = {
                "carsensor": [
                    {"title": "シエンタ A", "total_price": 189.0, "mileage_km": 52000},
                    {"title": "シエンタ B", "total_price": 150.0, "mileage_km": 30000},
                ],
                "mobilico": [
                    {"title": "シエンタ C", "total_price": 200.0, "mileage_km": 40000},
                ],
                "aucsupport": [
                    {"title": "シエンタ D", "total_price": 120.0, "mileage_km": 60000},
                    {"title": "シエンタ E", "total_price": 80.0, "mileage_km": 80000},
                ],
            }

            paths = []
            names = []
            for name, records in sources.items():
                path = Path(tmpdir) / f"{name}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(records, f)
                paths.append(path)
                names.append(name)

            result = load_and_merge(paths, names)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 5  # 2 + 1 + 2
            assert "source" in result.columns
            assert set(result["source"].unique()) == {"carsensor", "mobilico", "aucsupport"}

    def test_clean_dataframe_applies_normalization(self):
        """clean_dataframe が標準化を適用"""
        df = pd.DataFrame([{
            "title": "シエンタ",
            "total_price": 189.0,
            "base_price": 170.0,
            "mileage_km": 52000.5,
            "year": "令和2年",
            "inspection": "202611",
            "accident_history": "なし",
            "transmission": "CVT",
            "region": "東京都 渋谷区",
        }])

        result = clean_dataframe(df)
        assert result["total_price_yen"].iloc[0] == 1890000
        assert result["base_price_yen"].iloc[0] == 1700000
        assert result["mileage_km"].iloc[0] == 52000
        assert result["year"].iloc[0] == 2020
        assert result["inspection"].iloc[0] == "2026-11"
        assert result["accident_history"].iloc[0] == False  # noqa: E712
        assert result["transmission"].iloc[0] == "CVT"
        assert result["region"].iloc[0] == "東京都"

    def test_run_cleaning_pipeline(self):
        """完全なパイプライン: 読み込み → 標準化 → 重複排除"""
        with tempfile.TemporaryDirectory() as tmpdir:
            records = [
                {"title": "シエンタ", "total_price": 189.0, "mileage_km": 52000, "year": "2020年"},
                {"title": "シエンタ", "total_price": 189.0, "mileage_km": 52000, "year": "2020年"},
                {"title": "フィット", "total_price": 100.0, "mileage_km": 30000, "year": "2019年"},
            ]
            path = Path(tmpdir) / "test.pkl"
            with open(path, "wb") as f:
                pickle.dump(records, f)

            result = run_cleaning_pipeline([path], ["test"])
            assert len(result) == 2  # 重複が排除される
            assert result["year"].iloc[0] == 2020
