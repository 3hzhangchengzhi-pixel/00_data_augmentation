"""数値特徴量モジュール

車齢・里程区間・排気量区間・価格対数変換。
"""

from datetime import datetime

import numpy as np
import pandas as pd

CURRENT_YEAR = datetime.now().year

# 里程分桶境界（km）
MILEAGE_BINS = [0, 30000, 60000, 100000, float("inf")]
MILEAGE_LABELS = ["0-30k", "30k-60k", "60k-100k", "100k+"]

# 排気量分桶境界（cc）
DISPLACEMENT_BINS = [0, 1000, 1500, 2000, 3000, float("inf")]
DISPLACEMENT_LABELS = ["~1000cc", "1000-1500cc", "1500-2000cc", "2000-3000cc", "3000cc+"]


def add_car_age(df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
    """車齢カラムを追加: car_age = 現在年 - year"""
    df = df.copy()
    df["car_age"] = CURRENT_YEAR - df[year_col]
    return df


def add_mileage_bucket(
    df: pd.DataFrame,
    mileage_col: str = "mileage_km",
) -> pd.DataFrame:
    """里程を区間にバケット化"""
    df = df.copy()
    df["mileage_bucket"] = pd.cut(
        df[mileage_col],
        bins=MILEAGE_BINS,
        labels=MILEAGE_LABELS,
        right=False,
    )
    return df


def add_displacement_bucket(
    df: pd.DataFrame,
    displacement_col: str = "displacement_cc",
) -> pd.DataFrame:
    """排気量を区間にバケット化"""
    df = df.copy()
    df["displacement_bucket"] = pd.cut(
        df[displacement_col],
        bins=DISPLACEMENT_BINS,
        labels=DISPLACEMENT_LABELS,
        right=False,
    )
    return df


def add_log_price(
    df: pd.DataFrame,
    price_col: str = "total_price_yen",
) -> pd.DataFrame:
    """価格の対数変換（log1p）"""
    df = df.copy()
    df["log_price"] = np.log1p(df[price_col].clip(lower=0))
    return df
