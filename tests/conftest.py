"""pytest 共有フィクスチャ"""

import pytest
import pandas as pd


@pytest.fixture
def sample_car_data() -> list[dict]:
    """各テストで利用可能なサンプル車両データ"""
    return [
        {
            "title": "トヨタ シエンタ G",
            "maker": "トヨタ",
            "year": 2020,
            "mileage_km": 30000,
            "inspection": "202611",
            "accident_history": "なし",
            "warranty": "あり",
            "displacement": "1500cc",
            "transmission": "CVT",
            "total_price": 189.0,
            "base_price": 170.0,
            "region": "東京都",
            "color": "ホワイト",
            "grade": "G",
        },
        {
            "title": "ホンダ フィット 13G",
            "maker": "ホンダ",
            "year": 2018,
            "mileage_km": 55000,
            "inspection": "202503",
            "accident_history": "あり",
            "warranty": "なし",
            "displacement": "1300cc",
            "transmission": "CVT",
            "total_price": 120.0,
            "base_price": 105.0,
            "region": "大阪府",
            "color": "ブラック",
            "grade": "13G",
        },
    ]


@pytest.fixture
def sample_dataframe(sample_car_data: list[dict]) -> pd.DataFrame:
    """サンプルデータの DataFrame"""
    return pd.DataFrame(sample_car_data)
