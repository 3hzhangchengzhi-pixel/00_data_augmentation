"""特徴量構築入口

数値・カテゴリ・派生特徴量を組み合わせ、訓練用行列を出力。
"""

import numpy as np
import pandas as pd

from .categorical import label_encode, one_hot_encode
from .derived import (
    add_annual_mileage,
    add_depreciation_rate,
    add_inspection_remaining,
    add_price_per_km,
)
from .numerical import add_car_age, add_log_price, add_mileage_bucket


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """欠損値の処理: 数値は中央値、カテゴリは最頻値で補完"""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object", "category", "string"]).columns:
        if df[col].isna().any():
            mode = df[col].mode()
            fill_val = mode.iloc[0] if len(mode) > 0 else "unknown"
            df[col] = df[col].fillna(fill_val)
    return df


def remove_high_correlation(
    df: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """高相関特徴量の除去

    相関係数が threshold を超える特徴量ペアのうち、
    後方のカラムを削除する。
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return df

    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)


def build_features(
    df: pd.DataFrame,
    target_col: str = "total_price_yen",
    onehot_cols: list[str] | None = None,
    label_cols: list[str] | None = None,
    correlation_threshold: float = 0.95,
) -> tuple[pd.DataFrame, pd.Series]:
    """特徴量を構築し、訓練用の X と y を返す

    Args:
        df: 標準化済みの DataFrame
        target_col: 目標変数カラム名
        onehot_cols: One-Hot Encoding 対象カラム
        label_cols: Label Encoding 対象カラム
        correlation_threshold: 高相関除去の閾値

    Returns:
        (X, y): 特徴量行列と目標変数
    """
    if onehot_cols is None:
        onehot_cols = []
    if label_cols is None:
        label_cols = []

    df = df.copy()

    # 数値特徴量
    if "year" in df.columns:
        df = add_car_age(df)

    if target_col in df.columns:
        df = add_log_price(df, price_col=target_col)

    # 派生特徴量
    if "mileage_km" in df.columns and target_col in df.columns:
        df = add_price_per_km(df, price_col=target_col)

    if "mileage_km" in df.columns and "car_age" in df.columns:
        df = add_annual_mileage(df)

    if target_col in df.columns and "car_age" in df.columns:
        df = add_depreciation_rate(df, price_col=target_col)

    if "inspection" in df.columns:
        df = add_inspection_remaining(df)

    # カテゴリ特徴量
    for col in onehot_cols:
        if col in df.columns:
            df = one_hot_encode(df, col)

    for col in label_cols:
        if col in df.columns:
            df, _ = label_encode(df, col)

    # 欠損値処理
    df = handle_missing_values(df)

    # 目標変数の分離
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    y = df[target_col]

    # 非数値カラムと目標変数を除外
    drop_cols = [target_col]
    for col in df.columns:
        if df[col].dtype == "object" or df[col].dtype.name == "category" or pd.api.types.is_string_dtype(df[col]):
            drop_cols.append(col)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 高相関除去
    X = remove_high_correlation(X, threshold=correlation_threshold)

    return X, y
