"""派生特徴量モジュール

単位里程価格・年間平均里程・減価償却率・車検残り月数。
"""

from datetime import datetime

import numpy as np
import pandas as pd


def add_price_per_km(
    df: pd.DataFrame,
    price_col: str = "total_price_yen",
    mileage_col: str = "mileage_km",
) -> pd.DataFrame:
    """単位里程価格を追加: price_per_km = price / mileage"""
    df = df.copy()
    df["price_per_km"] = np.where(
        df[mileage_col] > 0,
        df[price_col] / df[mileage_col],
        np.nan,
    )
    return df


def add_annual_mileage(
    df: pd.DataFrame,
    mileage_col: str = "mileage_km",
    car_age_col: str = "car_age",
) -> pd.DataFrame:
    """年間平均里程を追加: annual_mileage = mileage / car_age"""
    df = df.copy()
    df["annual_mileage"] = np.where(
        df[car_age_col] > 0,
        df[mileage_col] / df[car_age_col],
        np.nan,
    )
    return df


def add_depreciation_rate(
    df: pd.DataFrame,
    price_col: str = "total_price_yen",
    car_age_col: str = "car_age",
) -> pd.DataFrame:
    """折旧率を追加

    簡易的に年あたり価格低下率を計算:
      depreciation_rate = price / (car_age + 1)
    """
    df = df.copy()
    df["depreciation_rate"] = df[price_col] / (df[car_age_col] + 1)
    return df


def add_inspection_remaining(
    df: pd.DataFrame,
    inspection_col: str = "inspection",
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """車検残り月数と車検有無ブール値を追加

    Args:
        df: inspection カラム（'YYYY-MM' 形式）を含む DataFrame
        inspection_col: 車検日期カラム名
        reference_date: 基準日（デフォルト: 今日）
    """
    df = df.copy()
    if reference_date is None:
        reference_date = datetime.now()

    ref_year = reference_date.year
    ref_month = reference_date.month

    def calc_remaining(val: str | None) -> int:
        if val is None or not isinstance(val, str):
            return 0
        try:
            parts = val.split("-")
            y, m = int(parts[0]), int(parts[1])
            months = (y - ref_year) * 12 + (m - ref_month)
            return max(0, months)
        except (ValueError, IndexError):
            return 0

    df["inspection_remaining_months"] = df[inspection_col].apply(calc_remaining)
    df["has_inspection"] = df["inspection_remaining_months"] > 0
    return df
