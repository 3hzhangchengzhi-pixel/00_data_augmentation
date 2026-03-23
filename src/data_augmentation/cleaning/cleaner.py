"""統一クリーニング入口

複数ソースの原始データを読み込み、標準化・重複排除・統合を行う。
"""

import logging
import pickle
from pathlib import Path

import pandas as pd

from .dedup import dedup_by_key
from .normalize import (
    normalize_accident_history,
    normalize_inspection,
    normalize_mileage,
    normalize_price,
    normalize_region,
    normalize_transmission,
    normalize_year,
)

logger = logging.getLogger(__name__)


def load_pickle(path: str | Path) -> list[dict]:
    """pickle ファイルから車両データを読み込む"""
    path = Path(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    logger.info("Loaded %d records from %s", len(data), path)
    return data


def load_and_merge(
    pickle_paths: list[str | Path],
    source_names: list[str] | None = None,
) -> pd.DataFrame:
    """複数ソースの pickle データを読み込み・統合

    Args:
        pickle_paths: pickle ファイルのパスリスト
        source_names: 各ソースの名前（pickle_paths と同じ長さ）

    Returns:
        統合された DataFrame（source カラム付き）
    """
    if source_names is None:
        source_names = [Path(p).stem for p in pickle_paths]

    all_frames: list[pd.DataFrame] = []
    for path, source in zip(pickle_paths, source_names):
        records = load_pickle(path)
        df = pd.DataFrame(records)
        df["source"] = source
        all_frames.append(df)
        logger.info("Source %s: %d records", source, len(df))

    if not all_frames:
        return pd.DataFrame()

    merged = pd.concat(all_frames, ignore_index=True)
    logger.info("Merged total: %d records", len(merged))
    return merged


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame に標準化処理を適用

    Args:
        df: 生データの DataFrame

    Returns:
        標準化済みの DataFrame
    """
    df = df.copy()

    # 価格標準化
    if "total_price" in df.columns:
        df["total_price_yen"] = df["total_price"].apply(normalize_price)
    if "base_price" in df.columns:
        df["base_price_yen"] = df["base_price"].apply(normalize_price)

    # 里程標準化
    if "mileage_km" in df.columns:
        df["mileage_km"] = df["mileage_km"].apply(normalize_mileage)

    # 年式標準化
    if "year" in df.columns:
        df["year"] = df["year"].apply(normalize_year)

    # 車検標準化
    if "inspection" in df.columns:
        df["inspection"] = df["inspection"].apply(normalize_inspection)

    # 修復歴標準化
    for col in ("accident_history", "bodywork_history"):
        if col in df.columns:
            df[col] = df[col].apply(normalize_accident_history)

    # 変速箱標準化
    if "transmission" in df.columns:
        df["transmission"] = df["transmission"].apply(normalize_transmission)

    # 地域標準化
    if "region" in df.columns:
        df["region"] = df["region"].apply(normalize_region)

    return df


def run_cleaning_pipeline(
    pickle_paths: list[str | Path],
    source_names: list[str] | None = None,
) -> pd.DataFrame:
    """完全なクリーニングパイプライン: 読み込み → 標準化 → 重複排除

    Args:
        pickle_paths: pickle ファイルのパスリスト
        source_names: 各ソースの名前

    Returns:
        クリーニング済みの DataFrame
    """
    df = load_and_merge(pickle_paths, source_names)
    if df.empty:
        return df

    df = clean_dataframe(df)
    df = dedup_by_key(df)

    logger.info("After cleaning: %d records", len(df))
    return df
