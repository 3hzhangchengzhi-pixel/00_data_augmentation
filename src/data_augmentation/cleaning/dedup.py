"""重複排除モジュール

タイトル + 価格 + 里程の組合せによる重複排除。
クロスソース重複排除対応。
"""

import pandas as pd


def dedup_by_key(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """タイトル + 価格 + 里程の組合せで重複排除

    重複がある場合は、非NULL値が最も多い（情報最完整な）レコードを保持する。

    Args:
        df: 入力 DataFrame
        subset: 重複判定に使うカラム名リスト（デフォルト: title, total_price, mileage_km）

    Returns:
        重複排除後の DataFrame
    """
    if df.empty:
        return df

    if subset is None:
        subset = ["title", "total_price", "mileage_km"]

    # subset のカラムが存在しない場合はそのまま返す
    available = [c for c in subset if c in df.columns]
    if not available:
        return df

    # 非NULL値の数でソート（多い方を優先）
    df = df.copy()
    df["_completeness"] = df.notna().sum(axis=1)
    df = df.sort_values("_completeness", ascending=False)

    # 重複排除（最初の=最完整なレコードを保持）
    df = df.drop_duplicates(subset=available, keep="first")
    df = df.drop(columns=["_completeness"])
    return df.reset_index(drop=True)


def cross_source_dedup(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """クロスソース重複排除

    異なるデータソース（source カラム）から来た同一車両を検出・統合する。
    情報最完整なレコードを保持。

    Args:
        df: source カラムを含む DataFrame
        subset: 重複判定に使うカラム名リスト

    Returns:
        重複排除後の DataFrame
    """
    return dedup_by_key(df, subset=subset)
