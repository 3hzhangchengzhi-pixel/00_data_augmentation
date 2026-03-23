"""カテゴリ特徴量モジュール

地域・変速箱のOne-Hot、色・メーカーのLabel Encoding。
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def one_hot_encode(
    df: pd.DataFrame,
    column: str,
    prefix: str | None = None,
) -> pd.DataFrame:
    """指定カラムをOne-Hot Encodingする

    Args:
        df: 入力 DataFrame
        column: エンコード対象のカラム名
        prefix: 生成カラムの接頭辞（デフォルト: カラム名）

    Returns:
        元のカラムを削除し、ダミーカラムを追加した DataFrame
    """
    df = df.copy()
    if prefix is None:
        prefix = column
    dummies = pd.get_dummies(df[column], prefix=prefix, dtype=int)
    df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    return df


def label_encode(
    df: pd.DataFrame,
    column: str,
    output_col: str | None = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """指定カラムをLabel Encodingする

    Args:
        df: 入力 DataFrame
        column: エンコード対象のカラム名
        output_col: 出力カラム名（デフォルト: {column}_encoded）

    Returns:
        (更新された DataFrame, フィット済み LabelEncoder)
    """
    df = df.copy()
    if output_col is None:
        output_col = f"{column}_encoded"
    le = LabelEncoder()
    df[output_col] = le.fit_transform(df[column].astype(str))
    return df, le
